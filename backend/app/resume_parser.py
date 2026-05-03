"""Resume PDF -> StudentProfile.

Two-stage parser, matching the deck's architecture:

  1. Try Gemini Flash structured extraction if GEMINI_API_KEY is set
     (the deck's "Resume PDF -> JSON in <5s" path).
  2. Fall back to a deterministic rule-based parser over the raw PDF text.
     Hackathon-friendly: no API key required to run the demo.
"""
from __future__ import annotations

import io
import json
import os
import re
from typing import Optional

import pdfplumber

from .schema import COURSE_TYPES, StudentProfile

TOP_TIER_COMPANIES = {
    "google", "microsoft", "amazon", "meta", "facebook", "apple", "netflix",
    "goldman sachs", "morgan stanley", "jp morgan", "jpmorgan", "blackrock",
    "deutsche bank", "barclays", "citi", "citibank", "hsbc",
    "mckinsey", "bcg", "bain", "deloitte", "pwc", "ey", "kpmg",
    "samsung", "intel", "nvidia", "qualcomm", "uber", "airbnb",
    "swiggy", "zomato", "flipkart", "ola", "razorpay", "phonepe",
    "stripe", "atlassian", "salesforce", "oracle", "ibm", "adobe",
    "icici", "hdfc", "axis bank", "kotak", "sbi", "yes bank",
}

CERT_KEYWORDS = [
    "aws", "azure", "gcp", "google cloud", "kubernetes", "docker",
    "tensorflow", "pytorch", "cisco", "ccna", "comptia", "pmp", "cfa",
    "frm", "six sigma", "scrum", "tableau", "power bi", "sql", "snowflake",
    "databricks", "hadoop", "spark", "react", "angular", "node",
]

COURSE_KEYWORDS = {
    "BTech-CS": ["computer science", "cse", "b.tech cs", "btech cs", "computer engg", "computer engineering"],
    "MTech-CS": ["m.tech cs", "mtech cs", "ms computer", "ms in computer"],
    "MCA": ["mca", "master of computer applications"],
    "BTech-ECE": ["electronics", "ece", "electronics and communication"],
    "BTech-Mech": ["mechanical", "b.tech mech", "btech mech"],
    "BTech-Civil": ["civil engineering", "b.tech civil"],
    "MBA-Finance": ["mba finance", "mba (finance)", "pgdm finance"],
    "MBA-Marketing": ["mba marketing", "pgdm marketing"],
    "MBA-HR": ["mba hr", "mba human resources", "pgdm hr"],
    "MBA-Operations": ["mba operations", "mba ops", "pgdm operations"],
    "BSc-Nursing": ["b.sc nursing", "bsc nursing", "nursing"],
    "BCom": ["b.com", "bcom", "bachelor of commerce"],
}

TIER1_INSTITUTES = {
    "iit", "iim", "iiit", "iisc", "nit ", "bits", "isb", "xlri", "fms", "spjimr",
    "aiims", "dtu", "nsut", "vit",
}


def _read_pdf_text(pdf_bytes: bytes) -> str:
    """Plain (single-column) extraction. Kept for backwards-compat.

    For all new callers prefer pdf_extract.extract_pdf which is column-aware."""
    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            parts.append(txt)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tail-detection — flags elite outlier profiles whose salary should NOT be
# anchored to the institute median. The rule (validated against the prompt):
# any 2+ of these 5 signals trigger is_elite_outlier=True.
# ---------------------------------------------------------------------------

_FAANG_TIER = {
    "google", "alphabet", "meta", "facebook", "amazon", "apple", "microsoft",
    "netflix", "openai", "anthropic", "deepmind", "nvidia", "stripe",
    "databricks", "snowflake", "rubrik", "uber", "airbnb",
    "goldman sachs", "morgan stanley", "jpmorgan", "jp morgan", "blackrock",
    "two sigma", "jane street", "citadel", "de shaw",
    "mckinsey", "bcg", "bain",
}


def _detect_elite_outlier(text: str, profile: "StudentProfile") -> tuple[bool, list[str]]:
    """Return (is_elite, reasons[]). Used to bypass salary median anchoring."""
    low = text.lower()
    reasons: list[str] = []

    # Signal 1: top-tier internship from explicit FAANG list
    intern_hits = [c for c in _FAANG_TIER if c in low]
    if intern_hits:
        reasons.append(f"Top-tier company experience ({', '.join(intern_hits[:2])})")

    # Signal 2: research output >=1 (paper / publication)
    if profile.paper_publications >= 1:
        reasons.append(f"{profile.paper_publications} research publication(s)")

    # Signal 3: hackathon wins >=2
    if profile.hackathon_wins >= 2:
        reasons.append(f"{profile.hackathon_wins} hackathon wins")

    # Signal 4: very high CGPA from any tier (>=9.0)
    if profile.cgpa >= 9.0:
        reasons.append(f"Top-decile CGPA ({profile.cgpa:.1f})")

    # Signal 5: deep competitive programming (>=500 problems solved)
    if profile.coding_problem_count >= 500:
        reasons.append(f"{profile.coding_problem_count}+ competitive problems solved")

    # Bonus signals (don't count toward 2/5 but get listed if present)
    if "icpc" in low or "putnam" in low or "google code jam" in low:
        reasons.insert(0, "ACM-ICPC / Putnam / Google Code Jam track record")

    is_elite = len(reasons) >= 2
    return is_elite, reasons


def _detect_course(text: str) -> str:
    low = text.lower()
    for course, keywords in COURSE_KEYWORDS.items():
        for k in keywords:
            if k in low:
                return course
    # Default by hint words
    if "engineering" in low:
        return "BTech-CS" if "computer" in low else "BTech-ECE"
    if "mba" in low or "pgdm" in low:
        return "MBA-Marketing"
    return "BTech-CS"


def _detect_tier(text: str) -> int:
    low = text.lower()
    for k in TIER1_INSTITUTES:
        if k in low:
            return 1
    if any(s in low for s in ["state university", "engineering college", "institute of technology"]):
        return 2
    return 2


def _detect_region(text: str) -> str:
    low = text.lower()
    metros = ["mumbai", "delhi", "bangalore", "bengaluru", "chennai", "hyderabad", "kolkata", "pune"]
    if any(m in low for m in metros):
        return "Metro"
    tier2 = ["jaipur", "lucknow", "indore", "bhopal", "nagpur", "kanpur", "vadodara", "patna",
             "ludhiana", "coimbatore", "kochi", "vizag", "visakhapatnam", "thiruvananthapuram",
             "allahabad", "prayagraj", "ranchi", "raipur", "guwahati", "chandigarh"]
    if any(t in low for t in tier2):
        return "Tier2"
    return "Tier3"


def _extract_cgpa(text: str) -> tuple[float | None, float]:
    """Returns (cgpa, confidence). cgpa is None if missing — DO NOT fabricate.

    Confidence rises when the matched value sits next to a CGPA/GPA token
    (vs. just any X.X / 10 number that could be unrelated).
    """
    # Ordered: most specific first; confidence reflects specificity.
    patterns = [
        (r"(?:CGPA|cgpa|C\.G\.P\.A\.)\s*[:=-]?\s*([\d]\.\d{1,2})\s*(?:/\s*10)?", 0.95),
        (r"(?:GPA|gpa|G\.P\.A\.)\s*[:=-]?\s*([\d]\.\d{1,2})\s*(?:/\s*10)?",      0.85),
        (r"([\d]\.\d{1,2})\s*/\s*10\b",                                         0.70),
        (r"\b([\d]{2}\.\d{1,2})\s*%",                                           0.60),
    ]
    for pat, conf in patterns:
        m = re.search(pat, text)
        if m:
            v = float(m.group(1))
            if v > 10:          # percentage → convert to /10 scale
                v = round(v / 10, 2)
            if v < 1.0:         # 0 or near-0 is never a valid CGPA — skip
                continue
            return v, conf
    return None, 0.0


def _extract_institute(text: str) -> tuple[str | None, float]:
    """Try to extract the institute name with a confidence score.

    Strategy: look for the line in the Education section whose text contains
    a known institute keyword (IIT/IIIT/NIT/IIM/BITS/VIT/college/university/
    institute). If multiple match, pick the longest.
    """
    edu_match = re.search(r"\beducation\b(.*?)(?:\n[A-Z][A-Za-z ]{3,30}\n|$)",
                          text, flags=re.IGNORECASE | re.DOTALL)
    body = edu_match.group(1) if edu_match else text
    known = re.compile(
        r"(?:Indian Institute of Technology|Indian Institute of Information Technology|"
        r"Indian Institute of Management|National Institute of Technology|"
        r"BITS|VIT|DTU|NSUT|IIIT|IIM|IIT|NIT|college|university|institute of technology)",
        re.IGNORECASE,
    )
    candidates: list[tuple[str, float]] = []
    for line in body.split("\n"):
        line = line.strip(" ,;-—|*•")
        if 6 < len(line) < 110 and known.search(line):
            # Prefer lines that ALSO mention a city / location-like proper noun
            has_city = bool(re.search(r"\b[A-Z][a-z]+(?:abad|pur|nagar|bay|elhi|pradesh|engaluru|hyderabad|hennai|olkata)\b", line))
            conf = 0.85 if has_city else 0.70
            candidates.append((line, conf))
    if not candidates:
        return None, 0.0
    # Prefer the one with the highest confidence; tiebreak by length
    candidates.sort(key=lambda c: (c[1], len(c[0])), reverse=True)
    return candidates[0]


def _extract_grad_year(text: str) -> int:
    years = re.findall(r"\b(20\d{2})\b", text)
    if years:
        ys = [int(y) for y in years]
        return max(ys)
    return 2024


def _extract_internships(text: str) -> list[dict]:
    low = text.lower()
    sections = re.split(r"\n(?:internships?|work experience|experience|professional experience)\s*[:\n]",
                        low, flags=re.IGNORECASE)
    body = sections[-1] if len(sections) > 1 else low

    # Stop at next section header
    body = re.split(r"\n(?:projects?|skills?|certifications?|education|achievements?)\s*[:\n]",
                    body, flags=re.IGNORECASE)[0]

    # Look for company names from our top-tier list
    internships: list[dict] = []
    for company in TOP_TIER_COMPANIES:
        if company in body:
            internships.append({"company": company.title(), "is_top_tier": True,
                                "duration_months": 3, "relevance": 0.85})
    # Heuristic: count "intern" mentions to guess additional internships
    intern_mentions = len(re.findall(r"\bintern(?:ship)?\b", body))
    additional = max(0, intern_mentions - len(internships))
    for _ in range(min(additional, 3)):
        internships.append({"company": "Other", "is_top_tier": False,
                            "duration_months": 2, "relevance": 0.5})
    return internships[:5]


def _extract_certifications(text: str) -> list[str]:
    low = text.lower()
    found: list[str] = []
    for kw in CERT_KEYWORDS:
        if kw in low:
            found.append(kw.upper() if len(kw) <= 4 else kw.title())
    # dedupe preserve order
    seen = set()
    out = []
    for c in found:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _extract_skills(text: str) -> list[str]:
    low = text.lower()
    skills_section_match = re.search(r"skills?\s*[:\n](.+?)(?:\n[a-z][a-z ]{3,15}\s*[:\n]|\Z)",
                                     low, flags=re.IGNORECASE | re.DOTALL)
    pool = skills_section_match.group(1) if skills_section_match else low
    candidates = re.findall(r"[a-zA-Z+#.]{2,20}", pool)
    return [c for c in candidates[:30] if len(c) > 1]


def _count_pattern(text: str, patterns: list[str]) -> int:
    low = text.lower()
    return sum(low.count(p) for p in patterns)


def _extract_name(text: str) -> str:
    # First non-empty line, capped to 40 chars, that looks like a name
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(c.isdigit() for c in line):
            continue
        if "@" in line:
            continue
        words = line.split()
        if 1 < len(words) <= 4 and all(w[:1].isalpha() for w in words):
            return line[:40]
    return "Anonymous"


def _heuristic_parse(text: str) -> StudentProfile:
    """Deterministic rule-based parser. Populates StudentProfile + per-field
    confidence in [0,1]. When a field cannot be found, the confidence is 0.0
    and the StudentProfile carries the schema default (so downstream code keeps
    working) — but predict.py reads field_confidence to widen ranges and warn
    the user that this field was imputed, not extracted.
    """
    course = _detect_course(text)
    tier = _detect_tier(text)
    region = _detect_region(text)
    name = _extract_name(text)
    cgpa, cgpa_conf = _extract_cgpa(text)
    institute, inst_conf = _extract_institute(text)
    grad_year = _extract_grad_year(text)
    internships = _extract_internships(text)
    certifications = _extract_certifications(text)
    skills = _extract_skills(text)

    github = _count_pattern(text, ["github.com", "github:", "github -"])
    coding = 100 + 50 * _count_pattern(text, ["leetcode", "codeforces", "codechef", "hackerrank"])
    hackathons = _count_pattern(text, ["hackathon", "hack ", "smart india hackathon"])
    leadership = _count_pattern(text, ["president", "lead", "captain", "head", "coordinator"])
    publications = _count_pattern(text, ["published", "paper", "ieee", "springer"])
    extracurriculars = _count_pattern(text, ["club", "society", "volunteer", "ngo", "sports", "music"])

    backlogs = 0
    backlog_conf = 0.0
    m = re.search(r"backlog[s]?\s*[:=-]?\s*(\d+)", text, flags=re.IGNORECASE)
    if m:
        backlogs = int(m.group(1))
        backlog_conf = 0.90

    # Per-field confidence — keyed by StudentProfile field name. Fields not
    # explicitly extractable from a resume (e.g. portal_activity_30d) get 0.0
    # — predict.py treats those as imputed defaults, not real signal.
    field_conf = {
        "name":                   0.85 if name != "Anonymous" else 0.0,
        "course_type":            0.80 if course != "BTech-CS" or "computer" in text.lower() else 0.30,
        "institute_name":         inst_conf,
        "institute_tier":         0.50,    # heuristic — IPR slug match raises this later
        "region":                 0.55 if region != "Tier3" else 0.40,
        "graduation_year":        0.70 if "20" in str(grad_year) else 0.30,
        "cgpa":                   cgpa_conf,
        "backlogs_count":         backlog_conf if backlog_conf else 0.50,  # absence == 0 backlogs is plausible
        "internships":            0.75 if internships else 0.0,
        "certifications":         0.65 if certifications else 0.0,
        "skills":                 0.70 if len(skills) >= 3 else 0.30,
        "github_projects":        0.65 if github > 0 else 0.0,
        "coding_problem_count":   0.50 if coding > 100 else 0.0,
        "hackathon_wins":         0.65 if hackathons > 0 else 0.0,
        "leadership_roles_count": 0.55 if leadership > 0 else 0.0,
        "paper_publications":     0.60 if publications > 0 else 0.0,
        "extracurriculars_count": 0.50 if extracurriculars > 0 else 0.0,
        "languages_known":        0.40,
        "portal_activity_30d":    0.0,    # not in resume — pure imputation
        "interview_invites_count":0.0,    # not in resume — pure imputation
        "salary_expectation_lpa": 0.0,    # not in resume — pure imputation
    }

    profile = StudentProfile(
        name=name,
        course_type=course,
        institute_name=institute or "Unknown",
        institute_tier=tier,
        region=region,
        graduation_year=grad_year,
        cgpa=cgpa if cgpa is not None else 7.0,   # default for the model vector,
        backlogs_count=backlogs,                  # but field_confidence flags it
        internships=internships,
        certifications=certifications,
        skills=skills,
        projects_count=_count_pattern(text, ["project"]),
        github_projects=max(github, _count_pattern(text, ["repo"])),
        coding_problem_count=coding,
        hackathon_wins=hackathons,
        leadership_roles_count=leadership,
        paper_publications=publications,
        extracurriculars_count=extracurriculars,
        languages_known=max(1, _count_pattern(text, ["english", "hindi", "tamil", "telugu", "marathi", "bengali"])),
        portal_activity_30d=15,
        interview_invites_count=0,
        salary_expectation_lpa=8.0 if course in {"BTech-CS", "MBA-Finance", "MTech-CS"} else 5.0,
        field_confidence=field_conf,
    )
    is_elite, reasons = _detect_elite_outlier(text, profile)
    profile.is_elite_outlier = is_elite
    profile.elite_outlier_reasons = reasons
    return profile


# Last-known LLM call status — surfaced via /api/llm_health to drive the
# nav indicator. Updated on every Gemini / OpenRouter call.
LLM_STATUS = {
    "provider": "none",                   # openrouter / gemini / none
    "model": None,                        # populated when a provider is selected
    "key_set": False,
    "sdk_importable": False,
    "last_call_status": "no_calls_yet",   # ok / rate_limited / auth_failed / parse_failed / no_calls_yet
    "last_call_at": None,
    "last_call_latency_ms": None,
    "last_error": None,
    "calls_total": 0,
    "calls_ok": 0,
}


# OpenRouter free model. As of 2026-04 the most reliable free models for
# strict-JSON extraction are openai/gpt-oss-120b:free and gpt-oss-20b:free
# (Llama 3.3 70B is heavily rate-limited upstream). Override via
# OPENROUTER_MODEL env var in .env.
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b:free"
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


def _build_extraction_prompt(text: str) -> str:
    """Shared LLM extraction prompt — strict JSON schema for StudentProfile."""
    return (
        "Extract a JSON profile from this resume. Return ONLY valid JSON, no prose.\n"
        f"Schema: {{\"name\": str, \"course_type\": one of {COURSE_TYPES}, "
        "\"institute_tier\": 1|2|3, \"region\": \"Metro\"|\"Tier2\"|\"Tier3\", "
        "\"institute_name\": str, "
        "\"graduation_year\": int, \"cgpa\": float, \"backlogs_count\": int, "
        "\"internships\": [{\"company\": str, \"is_top_tier\": bool, "
        "\"duration_months\": int, \"relevance\": float}], "
        "\"certifications\": [str], \"skills\": [str], "
        "\"github_projects\": int, \"coding_problem_count\": int, "
        "\"hackathon_wins\": int, \"leadership_roles_count\": int, "
        "\"paper_publications\": int, \"extracurriculars_count\": int, "
        "\"languages_known\": int, \"salary_expectation_lpa\": float}}\n\n"
        f"Resume text:\n{text[:8000]}"
    )


def _classify_error(exc: Exception) -> str:
    """Map an LLM exception to a status code surfaced in the UI."""
    msg = str(exc).lower()
    if "quota" in msg or "rate" in msg or "429" in msg or "resource has been exhausted" in msg:
        return "rate_limited"
    if "api key" in msg or "permission" in msg or "401" in msg or "403" in msg or "unauthorized" in msg:
        return "auth_failed"
    if isinstance(exc, json.JSONDecodeError):
        return "parse_failed"
    return "error"


def _coerce_profile(data: dict, source_text: str = "") -> StudentProfile:
    """Build a StudentProfile from an LLM-returned JSON dict, with safe defaults.

    LLM extractions get 0.85 confidence on every field the LLM produced; fields
    not present in the LLM output get 0.0 (pure imputation). The elite-outlier
    flag is derived from source_text + the structured fields.
    """
    # Track which fields the LLM actually returned non-null (vs. our defaults)
    def present(k):
        v = data.get(k)
        return v is not None and v != "" and v != []

    # Sanitise CGPA: LLMs sometimes return 0 or null when it's absent from
    # the resume. 0 is never a valid CGPA — treat it as missing and use the
    # population-mean default so the model doesn't score the student at rock bottom.
    raw_cgpa = data.get("cgpa")
    cgpa_valid = (raw_cgpa is not None
                  and str(raw_cgpa).strip() not in ("", "null", "N/A", "NA")
                  and float(raw_cgpa) >= 1.0)
    cgpa_value = float(raw_cgpa) if cgpa_valid else 7.0

    profile = StudentProfile(
        name=data.get("name") or "Anonymous",
        course_type=data.get("course_type") or "BTech-CS",
        institute_name=data.get("institute_name") or "Unknown",
        institute_tier=int(data.get("institute_tier", 2)),
        region=data.get("region") or "Tier2",
        graduation_year=int(data.get("graduation_year", 2024)),
        cgpa=cgpa_value,
        backlogs_count=int(data.get("backlogs_count", 0)),
        internships=data.get("internships") or [],
        certifications=data.get("certifications") or [],
        skills=data.get("skills") or [],
        github_projects=int(data.get("github_projects", 0)),
        coding_problem_count=int(data.get("coding_problem_count", 0)),
        hackathon_wins=int(data.get("hackathon_wins", 0)),
        leadership_roles_count=int(data.get("leadership_roles_count", 0)),
        paper_publications=int(data.get("paper_publications", 0)),
        extracurriculars_count=int(data.get("extracurriculars_count", 0)),
        languages_known=int(data.get("languages_known", 1)),
        salary_expectation_lpa=float(data.get("salary_expectation_lpa", 8.0)),
    )
    # Per-field confidence — LLM-extracted fields get 0.85, defaults get 0.0.
    # CGPA confidence is 0 when the LLM returned an invalid/absent value.
    field_conf = {f: (0.85 if present(f) else 0.0) for f in [
        "name", "course_type", "institute_name", "institute_tier", "region",
        "graduation_year", "cgpa", "backlogs_count", "internships",
        "certifications", "skills", "github_projects", "coding_problem_count",
        "hackathon_wins", "leadership_roles_count", "paper_publications",
        "extracurriculars_count", "languages_known", "salary_expectation_lpa",
    ]}
    # Things never present in a resume — explicit imputation marker
    field_conf["portal_activity_30d"] = 0.0
    field_conf["interview_invites_count"] = 0.0
    # Override CGPA confidence to 0 when we rejected the LLM's value
    if not cgpa_valid:
        field_conf["cgpa"] = 0.0
    profile.field_confidence = field_conf

    is_elite, reasons = _detect_elite_outlier(source_text, profile)
    profile.is_elite_outlier = is_elite
    profile.elite_outlier_reasons = reasons
    return profile


def _openrouter_parse(text: str) -> Optional[StudentProfile]:
    """Try OpenRouter (free-tier Llama 3.3 70B by default) extraction.

    Uses the OpenAI-compatible chat completions API. Returns None if
    unavailable or fails. Status surfaced via LLM_STATUS for /api/llm_health.
    """
    import time
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    model_name = os.environ.get("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
    LLM_STATUS["provider"] = "openrouter"
    LLM_STATUS["model"] = model_name
    LLM_STATUS["key_set"] = True

    try:
        from openai import OpenAI  # type: ignore
        LLM_STATUS["sdk_importable"] = True
    except ImportError:
        LLM_STATUS["sdk_importable"] = False
        LLM_STATUS["last_call_status"] = "sdk_missing"
        return None

    LLM_STATUS["calls_total"] += 1
    started = time.perf_counter()
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                # OpenRouter recommends these headers for free-tier identification.
                "HTTP-Referer": "https://github.com/team-aurdinary/careersight",
                "X-Title": "CareerSight",
            },
        )
        prompt = _build_extraction_prompt(text)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system",
                 "content": "You are a strict JSON extractor. Return only valid JSON matching the requested schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
        data = json.loads(raw)
        profile = _coerce_profile(data, source_text=text)
        # Override the institute_name placeholder if the LLM returned one
        profile.institute_name = data.get("institute_name") or "Parsed via OpenRouter"
        LLM_STATUS["calls_ok"] += 1
        LLM_STATUS["last_call_status"] = "ok"
        LLM_STATUS["last_call_at"] = time.time()
        LLM_STATUS["last_call_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        LLM_STATUS["last_error"] = None
        return profile
    except Exception as exc:
        LLM_STATUS["last_call_at"] = time.time()
        LLM_STATUS["last_call_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        LLM_STATUS["last_call_status"] = _classify_error(exc)
        LLM_STATUS["last_error"] = str(exc)[:240]
        return None


def _gemini_parse(text: str) -> Optional[StudentProfile]:
    """Try Gemini Flash extraction. Returns None if unavailable or fails."""
    import time
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai  # type: ignore
        LLM_STATUS["sdk_importable"] = True
    except ImportError:
        LLM_STATUS["sdk_importable"] = False
        LLM_STATUS["last_call_status"] = "sdk_missing"
        return None

    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    LLM_STATUS["provider"] = "gemini"
    LLM_STATUS["model"] = model_name
    LLM_STATUS["key_set"] = True
    LLM_STATUS["calls_total"] += 1
    started = time.perf_counter()
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(_build_extraction_prompt(text))
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).rstrip("`").strip()
        data = json.loads(raw)
        profile = _coerce_profile(data, source_text=text)
        profile.institute_name = data.get("institute_name") or "Parsed via Gemini Flash"
        LLM_STATUS["calls_ok"] += 1
        LLM_STATUS["last_call_status"] = "ok"
        LLM_STATUS["last_call_at"] = time.time()
        LLM_STATUS["last_call_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        LLM_STATUS["last_error"] = None
        return profile
    except Exception as exc:
        LLM_STATUS["last_call_at"] = time.time()
        LLM_STATUS["last_call_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        LLM_STATUS["last_call_status"] = _classify_error(exc)
        LLM_STATUS["last_error"] = str(exc)[:240]
        return None


def parse_resume_pdf(pdf_bytes: bytes) -> tuple[StudentProfile, str]:
    """Returns (profile, parser_used).

    Cascade: OpenRouter (free-tier Llama 3.3 by default) -> Gemini Flash -> heuristic.
    The first provider with a key set is tried first; on rate-limit / auth /
    parse failure we automatically fall through to the next.

    Uses the column-aware pdf_extract module so two-column resume layouts
    don't get garbled into interleaved-line gibberish.
    """
    from .pdf_extract import extract_pdf
    extraction = extract_pdf(pdf_bytes)
    text = extraction.text
    if not text.strip():
        return StudentProfile(), "empty"
    profile, parser = parse_resume_text(text)
    # Tag layout info onto the profile so the UI can surface it.
    if profile.field_confidence is None:
        profile.field_confidence = {}
    profile.field_confidence["_layout"] = extraction.layout
    profile.field_confidence["_page_count"] = extraction.page_count
    profile.field_confidence["_duplicate_pages"] = extraction.duplicate_page_count
    return profile, parser


def parse_resume_text(text: str) -> tuple[StudentProfile, str]:
    """Try LLMs first (OpenRouter -> Gemini), then heuristic. Used by
    /api/score_from_text — handy for testing without a PDF."""
    if os.environ.get("OPENROUTER_API_KEY"):
        profile = _openrouter_parse(text)
        if profile is not None:
            return profile, "openrouter"
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        profile = _gemini_parse(text)
        if profile is not None:
            return profile, "gemini"
    return _heuristic_parse(text), "heuristic"
