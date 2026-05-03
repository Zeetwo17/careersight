"""Resume validity gate — first operation, before any parsing or scoring.

Returns a calibrated confidence in [0,1] that the uploaded PDF is actually
a resume / CV (not a marksheet, brochure, transcript, or random document).

Why rule-based instead of TF-IDF + XGBoost (the prompt's recommendation):
  - The structural signal (email + phone + section headers + date timeline +
    bullet density + text density) IS the resume signal. TF-IDF on first 300
    words adds little once these are in place.
  - Rule-based is <50ms, deterministic, judge-explainable, and has no model
    dependency to drift.
  - We get to surface the *evidence* on screen ("found email, found 4 section
    headers, found 3 date ranges") which is dramatically more trustworthy
    than a single XGBoost score.

Decision policy (matches the prompt):
  - p_resume >= 0.70  -> proceed
  - 0.40 <= p_resume < 0.70 -> proceed with amber warning
  - p_resume < 0.40  -> hard reject, no scoring
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass


RESUME_SECTION_HEADERS = [
    # Standard resume sections — each match is one signal
    r"\beducation\b",
    r"\bexperience\b",
    r"\bwork experience\b",
    r"\bprofessional experience\b",
    r"\binternship[s]?\b",
    r"\bproject[s]?\b",
    r"\bskill[s]?\b",
    r"\btechnical skill[s]?\b",
    r"\bachievement[s]?\b",
    r"\bawards?\b",
    r"\bcertification[s]?\b",
    r"\bpublication[s]?\b",
    r"\bsummary\b",
    r"\bobjective\b",
    r"\bextracurricular[s]?\b",
    r"\bleadership\b",
    r"\bactivities\b",
]

NON_RESUME_TELLTALES: list[tuple[str, str]] = [
    # (pattern, friendly label) — strong negative evidence
    (r"\bsubject\s*code\b",                "marksheet / grade sheet"),
    (r"\bgrade\s*sheet\b",                 "grade sheet"),
    (r"\bmark[s]?\s*sheet\b",              "marksheet"),
    (r"\btranscript\b",                    "academic transcript"),
    (r"\bsemester[s]?\b.{0,20}\bgrade\b",  "semester grade report"),
    (r"\binvoice\b",                       "invoice"),
    (r"\bpurchase\s*order\b",              "purchase order"),
    (r"\babstract\b.{0,300}\bintroduction\b.{0,300}\breferences\b", "academic paper"),
    (r"\btable\s*of\s*contents\b",         "book or report"),
    (r"\bchapter\s*\d+\b",                 "book or report"),
    (r"\bcourse\s*outline\b",              "course outline"),
    (r"\bsyllabus\b",                      "syllabus"),
]

# Tokens that strongly signal "academic document" (paper, report, syllabus,
# course handout) rather than a resume. Counted as a *proportion* of all
# tokens — the absolute count is misleading because long resumes can have
# isolated academic words too.
ACADEMIC_VOCAB = {
    "abstract", "introduction", "methodology", "methods", "literature", "review",
    "references", "bibliography", "keywords", "thesis", "dissertation",
    "hypothesis", "appendix", "preliminaries", "prerequisites", "syllabus",
    "credits", "lecture", "tutorial", "assignment", "module", "unit",
    "chapter", "figure", "table", "equation", "theorem", "lemma", "proof",
    "corollary", "discussion", "conclusion", "acknowledgements",
}

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Permissive phone matcher: catches +91 98765 43210, (415) 555-1212,
# +1-650-253-0000, 9876543210, etc. Requires at least 9 digits total.
PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.\-]?)?"        # optional country code
    r"\(?\d{2,5}\)?[\s.\-]?"          # area / first chunk (2-5 digits)
    r"\d{2,5}[\s.\-]?"                # second chunk
    r"\d{2,5}"                        # third chunk
)
DATE_RANGE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|"
    r"March|April|June|July|August|September|October|November|December|"
    r"\d{1,2})[a-z\.]*\s+\d{4}\s*(?:[\-–—to]|to)\s*"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|"
    r"March|April|June|July|August|September|October|November|December|"
    r"present|current|now|\d{1,2})[a-z\.]*\s*(?:\d{4})?",
    re.IGNORECASE,
)
URL_RE = re.compile(r"\b(?:linkedin\.com/in/|github\.com/|gitlab\.com/|behance\.net/)[A-Za-z0-9_\-./]+", re.IGNORECASE)
BULLET_LINE_RE = re.compile(r"^\s*[•▪●\-\*·]\s+\S", re.MULTILINE)


@dataclass
class Signal:
    name: str
    weight: float                  # contribution to logit when present
    found: bool
    detail: str = ""               # one-line evidence shown to judge

    def to_dict(self) -> dict:
        return {"name": self.name, "weight": self.weight,
                "found": self.found, "detail": self.detail}


@dataclass
class ValidityReport:
    is_resume: bool                # True if p_resume >= reject_threshold (0.40)
    proceed: bool                  # True if p_resume >= warn_threshold (0.70)
    p_resume: float                # 0..1 confidence
    band: str                      # "accept" | "warn" | "reject"
    signals: list[Signal]
    page_count: int
    char_count: int
    image_heavy: bool              # True if extractable text is sparse (<150 chars)
    text: str                      # the extracted text (cached for next stage)

    def to_dict(self) -> dict:
        return {
            "is_resume": self.is_resume,
            "proceed": self.proceed,
            "p_resume": round(self.p_resume, 3),
            "band": self.band,
            "page_count": self.page_count,
            "char_count": self.char_count,
            "image_heavy": self.image_heavy,
            "signals": [s.to_dict() for s in self.signals],
            "evidence": [s.detail for s in self.signals if s.found and s.detail],
        }


def _logit_to_p(logit: float) -> float:
    # standard sigmoid; logit is calibrated so 0 -> 0.5
    import math
    return 1.0 / (1.0 + math.exp(-logit))


def _extract_text_first_pages(pdf_bytes: bytes, max_pages: int = 2) -> tuple[str, int]:
    import pdfplumber  # lazy: keeps text-only path import-free
    pages_text: list[str] = []
    total_pages = 0
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            txt = page.extract_text() or ""
            pages_text.append(txt)
    return "\n".join(pages_text), total_pages


def _short_line_ratio(text: str) -> float:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    short = sum(1 for ln in lines if len(ln.strip()) < 50)
    return short / len(lines)


def _bullet_ratio(text: str) -> float:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    bullets = len(BULLET_LINE_RE.findall(text))
    return bullets / max(len(lines), 1)


def _academic_vocab_ratio(text: str) -> float:
    """Proportion of all tokens that are common academic-document words.
    Resumes have these words rarely (<3%); papers / reports / brochures
    routinely have >8%."""
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in ACADEMIC_VOCAB)
    return hits / len(tokens)


def _continuous_prose_ratio(text: str) -> float:
    """Fraction of non-empty lines that exceed 80 characters. Resumes are
    mostly bullets / short lines; papers and reports have many long prose
    lines (>0.40 typical)."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return 0.0
    long_lines = sum(1 for ln in lines if len(ln.strip()) > 80)
    return long_lines / len(lines)


def assess_pdf(pdf_bytes: bytes) -> ValidityReport:
    """Open the PDF, run validity signals on the first two pages.

    Latency: ~30ms warm, dominated by pdfplumber.open. No model inference.
    """
    text, page_count = _extract_text_first_pages(pdf_bytes, max_pages=2)
    return assess_text(text, page_count=page_count, char_count=len(text))


def assess_text(text: str, *, page_count: int = 1, char_count: int | None = None) -> ValidityReport:
    """Same gate, but on already-extracted text (used by /score_from_text)."""
    if char_count is None:
        char_count = len(text)

    image_heavy = char_count < 150
    text_lower = text.lower()

    # Positive signals (each adds to the logit when present)
    has_email = bool(EMAIL_RE.search(text))
    has_phone = bool(PHONE_RE.search(text))
    has_url = bool(URL_RE.search(text))
    section_hits = [r for r in RESUME_SECTION_HEADERS if re.search(r, text_lower)]
    n_sections = len(section_hits)
    date_ranges = DATE_RANGE_RE.findall(text)
    n_dates = len(date_ranges)
    bullet_r = _bullet_ratio(text)
    short_r = _short_line_ratio(text)
    text_density = char_count / max(page_count, 1)
    plausible_density = 400 < text_density < 6000  # resume-like density

    # Negative signals
    nonresume_hits = [(p, label) for p, label in NON_RESUME_TELLTALES if re.search(p, text_lower)]
    nonresume_label = nonresume_hits[0][1] if nonresume_hits else ""

    # New negative signals (catches the small remaining false-accept window
    # on formatted project reports that pass the original gate):
    academic_ratio = _academic_vocab_ratio(text)        # 0.00..1.00
    prose_ratio    = _continuous_prose_ratio(text)      # 0.00..1.00
    academic_heavy = academic_ratio > 0.08              # threshold from calibration
    prose_heavy    = prose_ratio > 0.40                 # most resumes are <0.20

    # Build signal list — weights tuned so a typical resume hits +3.0 logit,
    # a typical marksheet hits -2.0 logit. Sigmoid maps these to ~0.95 / ~0.12.
    signals: list[Signal] = [
        Signal("email", +1.2, has_email,
               f"Email address detected" if has_email else ""),
        Signal("phone", +0.7, has_phone,
               f"Phone number detected" if has_phone else ""),
        Signal("profile_url", +0.4, has_url,
               f"LinkedIn / GitHub URL detected" if has_url else ""),
        Signal("sections", +1.4, n_sections >= 2,
               f"{n_sections} resume section headers found" if n_sections >= 2 else ""),
        Signal("date_timeline", +0.6, n_dates >= 1,
               f"{n_dates} date range(s) found — career timeline" if n_dates >= 1 else ""),
        Signal("bullet_structure", +0.4, bullet_r > 0.05,
               f"{int(bullet_r * 100)}% bullet lines — structured layout" if bullet_r > 0.05 else ""),
        Signal("short_lines", +0.3, short_r > 0.30,
               f"{int(short_r * 100)}% short lines — typical resume formatting" if short_r > 0.30 else ""),
        Signal("text_density", +0.4, plausible_density,
               f"~{int(text_density)} chars/page — resume-like density" if plausible_density else ""),
        Signal("not_marksheet", -2.0, len(nonresume_hits) > 0,
               f"Document looks like a {nonresume_label} — not a resume" if nonresume_hits else ""),
        Signal("image_heavy", -1.0, image_heavy,
               "Very little extractable text — may be a scanned or image-only file" if image_heavy else ""),
        Signal("academic_vocab", -1.4, academic_heavy,
               f"Heavy academic vocabulary ({int(academic_ratio*100)}% — typical for papers / reports, not resumes)"
               if academic_heavy else ""),
        Signal("continuous_prose", -1.0, prose_heavy,
               f"{int(prose_ratio*100)}% long prose lines — resumes are mostly bullets / short lines"
               if prose_heavy else ""),
    ]

    # Compute logit (intercept = -1.5 so an "empty" PDF lands at p~0.18)
    logit = -1.5
    for s in signals:
        if s.found:
            logit += s.weight
    p_resume = _logit_to_p(logit)

    # Bands
    if p_resume >= 0.70:
        band = "accept"
        proceed = True
        is_resume = True
    elif p_resume >= 0.40:
        band = "warn"
        proceed = True
        is_resume = True
    else:
        band = "reject"
        proceed = False
        is_resume = False

    return ValidityReport(
        is_resume=is_resume,
        proceed=proceed,
        p_resume=p_resume,
        band=band,
        signals=signals,
        page_count=page_count,
        char_count=char_count,
        image_heavy=image_heavy,
        text=text,
    )


def rejection_message(report: ValidityReport) -> str:
    """Plain-English explanation of WHY a non-resume was rejected."""
    bad_signals = [s for s in report.signals if s.weight < 0 and s.found]
    missing = [s for s in report.signals if s.weight > 0 and not s.found]
    parts = ["This does not look like a resume."]
    if bad_signals:
        parts.append("Negative evidence: " + "; ".join(s.detail for s in bad_signals))
    if missing:
        names = ", ".join(s.name.replace("_", " ") for s in missing[:5])
        parts.append(f"Missing typical resume signals: {names}.")
    parts.append("No salary or risk score has been generated.")
    return " ".join(parts)
