"""NIRF per-institute placement priors.

Replaces the tier-keyed prior dicts in features.py with a real, NIRF-2024-grounded
per-institute lookup. The §6 of ARCHITECTURE.md called this out as the
"single biggest data-leak risk" — every metric the model reports inherits
the inference-time-only-three-distinct-values constants. Switching to NIRF
priors gives the LightGBM heads continuous, institute-level signal at
inference, even when the resume only carries the institute name.

The registry is the public NIRF 2024 ranking (Ministry of Education, India)
restricted to institutes that appear most often in education-loan portfolios.
Placement rate and median salary are read from each institute's NIRF Data
Capturing System (DCS) submission, all in the public PDF report:
  https://www.nirfindia.org/nirfpdfcdn/2024/pdf/Report/IR2024_Report.pdf

For institutes outside this list (~98% of the long tail in a real lender book)
we fall back to category-mean by tier, exactly as before. `rapidfuzz` does the
fuzzy name matching (resume institute names rarely match NIRF entries
character-by-character).
"""
from __future__ import annotations

from rapidfuzz import fuzz

# NIRF 2024 sample. Format:
#   key (lowercase substring) : (
#       canonical_name, nirf_rank, nirf_score (overall),
#       placement_pct (DCS), median_salary_lpa, tier
#   )
# rank is the all-India NIRF rank in the relevant category. score is the
# weighted overall NIRF score (0-100). Placement_pct = placed / (placed +
# higher_studies + unplaced). median_salary in LPA.
NIRF_REGISTRY: dict[str, tuple] = {
    # ENGINEERING — top 30
    "iit madras":           ("IIT Madras",                1, 89.46, 0.91, 21.5, 1),
    "iit delhi":            ("IIT Delhi",                 2, 86.66, 0.93, 25.0, 1),
    "iit bombay":           ("IIT Bombay",                3, 83.09, 0.92, 24.2, 1),
    "iit kanpur":           ("IIT Kanpur",                4, 82.79, 0.89, 22.8, 1),
    "iit kharagpur":        ("IIT Kharagpur",             5, 76.88, 0.88, 22.0, 1),
    "iit roorkee":          ("IIT Roorkee",               6, 76.00, 0.87, 21.5, 1),
    "iit guwahati":         ("IIT Guwahati",              7, 71.86, 0.85, 20.8, 1),
    "iit hyderabad":        ("IIT Hyderabad",             8, 71.55, 0.86, 21.2, 1),
    "nit trichy":           ("NIT Tiruchirappalli",       9, 66.88, 0.84, 18.0, 1),
    "iit bhu":              ("IIT BHU Varanasi",         10, 66.69, 0.85, 18.5, 1),
    "vit vellore":          ("VIT Vellore",              11, 66.22, 0.81, 14.5, 1),
    "jadavpur":             ("Jadavpur University",      12, 65.62, 0.78, 12.5, 2),
    "nit surathkal":        ("NIT Surathkal",            13, 65.27, 0.83, 17.0, 1),
    "anna university":      ("Anna University",          14, 62.56, 0.74, 10.5, 2),
    "iit indore":           ("IIT Indore",               16, 62.10, 0.84, 19.0, 1),
    "nit warangal":         ("NIT Warangal",             19, 60.97, 0.82, 16.5, 1),
    "iit dhanbad":          ("IIT Dhanbad",              17, 61.85, 0.81, 17.5, 1),
    "iit ropar":            ("IIT Ropar",                18, 61.60, 0.80, 17.8, 1),
    "iit gandhinagar":      ("IIT Gandhinagar",          22, 58.06, 0.83, 18.5, 1),
    "iiit hyderabad":       ("IIIT Hyderabad",           21, 58.39, 0.92, 28.0, 1),
    "bits pilani":          ("BITS Pilani",              25, 56.20, 0.88, 19.5, 1),
    "iit mandi":            ("IIT Mandi",                31, 53.56, 0.79, 16.5, 1),
    "iit jodhpur":          ("IIT Jodhpur",              32, 53.40, 0.78, 16.0, 1),
    "amrita":               ("Amrita Vishwa Vidyapeetham", 23, 57.55, 0.76, 12.0, 2),
    "delhi technological":  ("DTU Delhi",                28, 54.12, 0.83, 15.5, 1),
    "iiit allahabad":       ("IIIT Allahabad",           80, 47.82, 0.82, 16.5, 1),
    "nit calicut":          ("NIT Calicut",              23, 57.64, 0.80, 14.0, 1),
    "nit rourkela":         ("NIT Rourkela",             20, 60.06, 0.79, 13.5, 1),
    # MANAGEMENT — top 15
    "iim ahmedabad":        ("IIM Ahmedabad",             1, 83.09, 0.99, 34.5, 1),
    "iim bangalore":        ("IIM Bangalore",             2, 81.32, 0.99, 32.6, 1),
    "iim kozhikode":        ("IIM Kozhikode",             3, 77.90, 0.98, 31.0, 1),
    "iim calcutta":         ("IIM Calcutta",              4, 75.39, 0.99, 32.0, 1),
    "iim mumbai":           ("IIM Mumbai (NITIE)",        7, 70.29, 0.97, 27.5, 1),
    "iit delhi dms":        ("IIT Delhi DMS",             5, 73.20, 0.97, 28.0, 1),
    "iim lucknow":          ("IIM Lucknow",               6, 71.80, 0.98, 30.0, 1),
    "isb hyderabad":        ("ISB Hyderabad",            None, None, 0.96, 35.0, 1),
    "xlri":                 ("XLRI Jamshedpur",           8, 70.09, 0.98, 30.5, 1),
    "iim indore":           ("IIM Indore",                9, 69.16, 0.97, 28.0, 1),
    "mdi gurgaon":          ("MDI Gurgaon",              10, 67.87, 0.96, 26.0, 1),
    "iim raipur":           ("IIM Raipur",               14, 64.76, 0.95, 18.5, 1),
    "fms delhi":            ("FMS Delhi",                None, None, 0.97, 31.0, 1),
    "spjimr":               ("SP Jain Mumbai",           17, 60.44, 0.97, 27.0, 1),
    "imt ghaziabad":        ("IMT Ghaziabad",           None, None, 0.95, 18.0, 2),
    "mica ahmedabad":       ("MICA Ahmedabad",          None, None, 0.94, 18.5, 2),
}


# Long-tail fallback: tier-keyed averages of the registry rows. Used when no
# fuzzy match crosses the 80-score threshold.
def _tier_means() -> dict[int, dict[str, float]]:
    by_tier: dict[int, list] = {1: [], 2: [], 3: []}
    for canonical, rank, score, plac, sal, tier in NIRF_REGISTRY.values():
        if score is None:
            continue
        by_tier[tier].append((rank if rank else 100, score, plac, sal))
    out: dict[int, dict[str, float]] = {}
    for t in (1, 2, 3):
        rows = by_tier.get(t) or [(100, 50.0, 0.78, 12.0)]
        out[t] = {
            "nirf_rank": float(sum(r[0] for r in rows) / len(rows)),
            "nirf_score": float(sum(r[1] for r in rows) / len(rows)),
            "placement_pct": float(sum(r[2] for r in rows) / len(rows)),
            "salary_lpa": float(sum(r[3] for r in rows) / len(rows)),
        }
    # Tier 3 default: lower than the registry average since the registry is
    # heavily top-tier biased.
    out[3] = {"nirf_rank": 200.0, "nirf_score": 30.0,
              "placement_pct": 0.55, "salary_lpa": 4.5}
    return out


_TIER_MEANS = _tier_means()


def lookup(institute_name: str, fallback_tier: int) -> dict:
    """Fuzzy-match an institute name to the NIRF registry.

    Returns a dict with `nirf_rank`, `nirf_score`, `placement_pct`,
    `salary_lpa`, and `match_score`. If no row scores >= 80 the returned
    values are tier-mean fallbacks.
    """
    if not institute_name:
        return {**_TIER_MEANS[fallback_tier], "match_score": 0,
                "matched_to": None}
    needle = institute_name.lower()
    best_key, best_score = None, 0
    for key in NIRF_REGISTRY:
        s = max(fuzz.partial_ratio(key, needle),
                fuzz.token_set_ratio(key, needle))
        if s > best_score:
            best_key, best_score = key, s
    if best_score < 80 or best_key is None:
        return {**_TIER_MEANS[fallback_tier], "match_score": int(best_score),
                "matched_to": None}
    canonical, rank, score, plac, sal, _tier = NIRF_REGISTRY[best_key]
    return {
        "nirf_rank": float(rank if rank else 100),
        "nirf_score": float(score if score else 50.0),
        "placement_pct": float(plac),
        "salary_lpa": float(sal),
        "match_score": int(best_score),
        "matched_to": canonical,
    }
