"""PDF text extraction with multi-column awareness.

The default `page.extract_text()` walks words in render order, which on
two-column resumes (Canva templates, "modern" CV layouts) interleaves the
left and right columns line-by-line and produces garbled text.

This module:
  1. Pulls the per-word bounding-box list from pdfplumber.
  2. Clusters word x-coordinates with a simple two-mean split.
  3. If the two clusters are well-separated AND each holds a meaningful
     share of the words, the page is multi-column: extract the left column
     top-to-bottom, then the right column top-to-bottom.
  4. Otherwise extract the page as plain text.

Latency cost vs the default extractor: ~50-80ms per page. Acceptable in
the critical path for the ~20% of resumes that need it; the layout flag
is also surfaced to the judge as part of the streaming reveal.

No GPU, no VLM, no model download. Just bounding-box arithmetic.
"""
from __future__ import annotations

import io
from dataclasses import dataclass


@dataclass
class PageExtraction:
    text: str
    layout: str           # "single" | "multicolumn"
    n_words: int
    n_columns: int
    col_centers: list[float]   # x-coordinate of each detected column center (page coords)


@dataclass
class PDFExtraction:
    text: str
    pages: list[PageExtraction]
    layout: str           # majority layout across pages
    page_count: int
    has_pages_with_no_text: bool
    duplicate_page_count: int


def _split_two_means(xs: list[float], iters: int = 8) -> tuple[float, float, list[int]]:
    """Tiny 1D 2-means. Returns (left_mean, right_mean, assignments)."""
    if not xs:
        return 0.0, 0.0, []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-6:
        return lo, hi, [0] * len(xs)
    left = lo + (hi - lo) * 0.25
    right = lo + (hi - lo) * 0.75
    assigns = [0] * len(xs)
    for _ in range(iters):
        # assign each point to nearest center
        for i, x in enumerate(xs):
            assigns[i] = 0 if abs(x - left) <= abs(x - right) else 1
        # update means; if a cluster is empty, leave its center
        l_pts = [x for x, a in zip(xs, assigns) if a == 0]
        r_pts = [x for x, a in zip(xs, assigns) if a == 1]
        if l_pts: left = sum(l_pts) / len(l_pts)
        if r_pts: right = sum(r_pts) / len(r_pts)
    return left, right, assigns


def _extract_page(page, *, gap_threshold: float = 150.0,
                  share_threshold: float = 0.30) -> PageExtraction:
    """Decide single vs multi-column on one page; return ordered text."""
    # Use pdfplumber's words with word-level x0, top.
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False) or []
    if not words:
        return PageExtraction(text="", layout="single", n_words=0,
                              n_columns=1, col_centers=[])

    xs = [w["x0"] for w in words]
    left_c, right_c, assigns = _split_two_means(xs)

    # Decide multi-column iff the two cluster centers are well-separated
    # AND both clusters hold at least share_threshold of the words.
    n = len(words)
    n_left = sum(1 for a in assigns if a == 0)
    n_right = n - n_left
    is_multi = (
        abs(right_c - left_c) >= gap_threshold
        and n_left / n >= share_threshold
        and n_right / n >= share_threshold
    )

    if not is_multi:
        text = page.extract_text() or ""
        return PageExtraction(text=text, layout="single", n_words=n,
                              n_columns=1, col_centers=[left_c])

    # Two-column path: emit left col top-to-bottom, then right col t-to-b.
    left_words = sorted(
        [w for w, a in zip(words, assigns) if a == 0],
        key=lambda w: (round(w["top"], 1), w["x0"]),
    )
    right_words = sorted(
        [w for w, a in zip(words, assigns) if a == 1],
        key=lambda w: (round(w["top"], 1), w["x0"]),
    )

    def _join(words_sorted: list[dict]) -> str:
        # Group words into lines by y; insert space inside line, newline between.
        if not words_sorted:
            return ""
        lines: list[list[str]] = []
        cur_top = words_sorted[0]["top"]
        cur_line: list[str] = []
        for w in words_sorted:
            if abs(w["top"] - cur_top) > 4:  # new line if y jumps > 4 pt
                if cur_line:
                    lines.append(cur_line)
                cur_line = []
                cur_top = w["top"]
            cur_line.append(w["text"])
        if cur_line:
            lines.append(cur_line)
        return "\n".join(" ".join(line) for line in lines)

    text = _join(left_words) + "\n\n" + _join(right_words)
    return PageExtraction(text=text, layout="multicolumn", n_words=n,
                          n_columns=2, col_centers=[left_c, right_c])


def extract_pdf(pdf_bytes: bytes, *, max_pages: int = 5) -> PDFExtraction:
    """Open the PDF, extract text per page with multi-column awareness."""
    import pdfplumber  # lazy
    pages: list[PageExtraction] = []
    page_count = 0
    has_blanks = False

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            pe = _extract_page(page)
            if not pe.text.strip():
                has_blanks = True
            pages.append(pe)

    # Duplicate-page detection: hash text per page
    seen: set[int] = set()
    dup = 0
    for pe in pages:
        h = hash(pe.text.strip())
        if h in seen:
            dup += 1
        else:
            seen.add(h)

    full_text = "\n\n".join(pe.text for pe in pages if pe.text)
    layout_majority = "multicolumn" if sum(1 for pe in pages if pe.layout == "multicolumn") > len(pages) / 2 else "single"

    return PDFExtraction(
        text=full_text, pages=pages, layout=layout_majority,
        page_count=page_count, has_pages_with_no_text=has_blanks,
        duplicate_page_count=dup,
    )
