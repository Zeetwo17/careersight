# CareerSight — End-to-End Architecture

**Status:** Research-grade hackathon MVP, post-IPR rebuild. The system now obeys the prompt's primary architectural call: institute-specific empirical placement distributions anchor every salary and risk prediction; NIRF rank is a level-4 cluster input only, never a direct salary proxy.

> **TL;DR.** A FastAPI service ingests a PDF resume or JSON profile, runs it through a **5-stage pipeline** — (1) rule-based validity gate that rejects non-resumes before any scoring, (2) column-aware PDF text extraction, (3) 3-tier LLM-or-heuristic parser cascade with per-field confidence, (4) Institute Placement Registry (IPR) lookup with 5-level fallback, (5) anchored scoring (salary p10–p90 anchored to institute IPR, placement probabilities blended with the institute's actual `placement_rate_6m`). Streaming SSE emits one event per finding so the judge sees forensic discovery, not a JSON dump. Three LightGBM heads, Beta calibration, DeepHit survival, SHAP top-5 drivers, Thompson Sampling Co-Pilot, and CRC-guaranteed risk threshold all run in <600ms after the IPR anchor is applied.

---

## 1. Problem framing

Hackathon Problem Statement 1 — predict placement timeline (3/6/12 months), salary range, and delayed-repayment risk for education-loan borrowers, with explainable AI and lender-actionable alerts. **Not credit automation.** The deck targets the ₹35,100 Cr at-risk slice of India's ₹4.5 L Cr education-loan book.

The architecture obeys five non-negotiable constraints from the prompt synthesis:

1. **No resume → no score.** A validity gate runs before any parsing or feature extraction. Marksheets, brochures, and academic papers are rejected with explicit evidence.
2. **Institute data anchors everything.** A pre-built IPR keyed by `(institute_slug, degree, branch, year_bin)` is the source of truth for salary distributions and placement rates. NIRF only feeds level-4 tier-cluster construction.
3. **No hallucinated fields.** Every parsed field carries a 0–1 confidence. Missing fields stay missing (with imputation flagged) — they never get fabricated.
4. **Heavy work is offline.** The model bundle, IPR dict, and SHAP background dataset all load at server startup. Inference is dict lookups + small tree calls.
5. **Streaming reveal over JSON dump.** Every stage emits an SSE event so the judge watches the system reason from evidence.

---

## 2. Repository layout

```
CareerSight/
├── backend/app/
│   ├── validity_gate.py     ★ NEW — rule-based "is this a resume?" gate (8 +ve / 2 -ve signals, <50ms)
│   ├── pdf_extract.py       ★ NEW — column-aware text extraction (handles two-column layouts)
│   ├── ipr.py               ★ NEW — Institute Placement Registry, 5-level fallback ladder
│   ├── stream.py            ★ NEW — SSE event generator for the forensic-discovery flow
│   ├── resume_parser.py     ★ UPDATED — 3-tier cascade + per-field confidence + elite-outlier rule
│   ├── predict.py           ★ UPDATED — IPR-anchored salary + placement, elite-outlier branch
│   ├── schema.py            ★ UPDATED — added field_confidence, is_elite_outlier, elite_outlier_reasons
│   ├── main.py              ★ UPDATED — added /api/score_from_resume_stream + validity-gate guard
│   │
│   ├── features.py          # StudentProfile -> 60-dim vector, NIRF prior fill-in
│   ├── nirf.py              # NIRF 2024 fuzzy registry (legacy — used at L4 cluster construction)
│   ├── train.py             # End-to-end training pipeline (~30 min)
│   ├── calibration.py       # Beta calibration (Kull 2017) + ECE/Brier
│   ├── crc.py               # MAPIE Learn-Then-Test BCC (Bates 2024)
│   ├── survival.py          # DeepHitSingle on {3, 6, 12} grid (Lee 2018)
│   ├── bandit.py            # Beta-Bernoulli Thompson Sampling Co-Pilot
│   ├── counterfactual.py    # EconML DR ATE + placebo refuter (per action)
│   ├── causal.py            # PC algorithm + Markov blanket (causal-learn)
│   ├── federated.py         # Federated SHAP across 2 shards with Gaussian DP
│   ├── edge_model.py        # Born-Again single tree (Vidal 2020), int16 JSON
│   ├── fairness.py          # Fairlearn audit (DPDP §10 + EEOC four-fifths)
│   ├── drift.py             # PSI + KS Bonferroni-corrected
│   ├── ood.py               # Kaggle real-label OOD validation fold
│   ├── portfolio.py         # Lender book + Naukri-JobSpeak-anchored PRI
│   ├── demo_profiles.py     # Priya/Arjun/Meera/Rahul personas
│   └── synth_data.py        # 15k synthetic borrowers (training corpus)
├── frontend/
│   ├── index.html           ★ UPDATED — added discovery-panel for the SSE forensic flow
│   ├── app.js               ★ UPDATED — SSE consumer + IPR / elite / anchor card renderers
│   └── style.css            ★ UPDATED — discovery-panel + ipr-card + elite-banner + anchor-card CSS
├── data/
│   ├── synthetic_students.csv      # 15k rows, 60 cols
│   ├── ood/                        # Drop Kaggle CSVs here for real-label OOD
│   ├── bandit_state.json           # Persisted Beta(α, β) posteriors
│   └── models/bundle.joblib (~10.8 MB)
├── ARCHITECTURE.md          # this file
├── README.md
├── prompt.txt               # The prompt-synthesis design doc this rebuild implements
└── ...
```

---

## 3. End-to-end request flow

```
Browser (frontend/app.js)
   │  POST multipart  /api/score_from_resume_stream
   ▼
FastAPI (backend/app/main.py)
   │
   ▼
stream_pdf(pdf_bytes)  ── async generator, yields one SSE event per finding
   │
   ├─ Stage 0 — Pre-flight
   │     ├─ Magic-bytes PDF check, size cap (10 MB)
   │     └─ event: upload         {filename, size_kb}
   │
   ├─ Stage 1 — Validity gate (validity_gate.py, <50ms)
   │     ├─ pdfplumber extract first 2 pages → text
   │     ├─ Score 8 positive + 2 negative structural signals → p_resume ∈ [0,1]
   │     ├─ event: stage          {name: "validity"}
   │     ├─ event: validity_signal × N (one per found signal, paced at 100ms)
   │     └─ event: validity_result {p_resume, band, signals[]}
   │
   │  IF p_resume < 0.40  →  event: rejected {reason}; event: done; STOP
   │
   ├─ Stage 2 — Column-aware text extraction (pdf_extract.py)
   │     ├─ pdfplumber extract_words() per page
   │     ├─ 1D 2-means on word x0 → detect multi-column layout
   │     ├─ If multi-column: emit left col top-to-bottom, then right col
   │     └─ Duplicate-page detection (text hash)
   │
   ├─ Stage 3 — Parser cascade (resume_parser.py)
   │     ├─ event: stage          {name: "parse"}
   │     ├─ Try OpenRouter (free-tier GPT-OSS-120B) with strict-JSON response_format
   │     ├─ Fallback Gemini Flash 2.0 on rate-limit / auth / parse error
   │     ├─ Final fallback: heuristic regex parser (no API needed, always works)
   │     ├─ Output: StudentProfile + per-field confidence dict
   │     ├─ event: parsed_field × ~6  (name, course, institute, CGPA, internships, skills)
   │     │              {field, value, confidence, imputed: bool}
   │     ├─ Elite-outlier detection: 5-signal rule, requires 2+ to fire
   │     └─ event: parse_result   {parser_used, layout, page_count,
   │                               is_elite_outlier, elite_outlier_reasons[]}
   │
   ├─ Stage 4 — Institute Placement Registry lookup (ipr.py, <5ms)
   │     ├─ event: stage          {name: "ipr"}
   │     ├─ Normalise institute name → fuzzy match against alias map (rapidfuzz ≥ 80)
   │     ├─ 5-level fallback ladder:
   │     │     L1  exact (slug, degree, branch)            ← "high" data quality
   │     │     L2  same institute + degree, all branches   ← "medium"
   │     │     L3  same institute, degree-aggregate        ← "low"
   │     │     L4  cluster (tier × degree × branch_family) ← "medium" if n ≥ 1000
   │     │     L5  national baseline by degree             ← "baseline"
   │     └─ event: ipr_card       {canonical_name, p10..p90, placement_rate_3/6/12m,
   │                               sample_size, source, fallback_level, data_quality}
   │
   └─ Stage 5 — Anchored scoring (predict.py, <600ms)
         ├─ event: stage          {name: "score"}
         ├─ Feature vector: profile_to_vector(profile) → 60-dim float32
         ├─ Raw model outputs (parallel):
         │     · 3 LightGBM classifiers → raw_p_3m / 6m / 12m
         │     · 3 LightGBM CQR regressors → raw_salary_low / med / high (10/50/90 quantile)
         │     · Beta calibration applied (netcal) — Kull 2017
         │
         ├─ Anchor weight: α = {0.70, 0.55, 0.40, 0.20} for {high, medium, low, baseline}
         │     · L1 / L2 high-quality IPR dominates
         │     · L4 cluster / L5 baseline → model dominates
         │
         ├─ Anchored placement probability:
         │     p_6m = α · ipr.placement_rate_6m + (1-α) · raw_p_6m
         │
         ├─ Anchored salary:
         │     deviation        = clamp(raw_med / ipr.p50, 0.6, 2.0)
         │     anchored_p50     = α · ipr.p50 + (1-α) · raw_med
         │     scale            = anchored_p50 / ipr.p50
         │     low_band         = ipr.p10 · scale
         │     high_band        = ipr.p90 · scale
         │
         ├─ Elite-outlier branch (if profile.is_elite_outlier):
         │     median_floor     = max(anchored_p50, ipr.p75)
         │     low_band         = max(ipr.p50, anchored_p50 · 0.85)
         │     high_band        = max(ipr.p90 · 1.3, median · 2.0)
         │     (lower bound jumps to institute median — can't underprice ICPC + Google)
         │
         ├─ Confidence widening:
         │     parse_conf_mean  = mean(field_confidence values)
         │     widen_factor     = (1.20 if parse_conf_mean < 0.65 else 1.0)
         │                      · (1.15 if ipr.fallback_level >= 4 else 1.0)
         │     range            *= widen_factor
         │
         ├─ Hard floors / ceilings (preserve invariants low ≤ med ≤ high)
         │
         ├─ DeepHitSingle survival curve (Lee 2018) at cuts {3, 6, 12} → 13 monthly points
         ├─ SHAP top-5 (TreeExplainer, 200-row pre-warmed background, <400ms)
         ├─ Thompson-sampled Co-Pilot recommendation (Beta posteriors loaded at startup)
         ├─ CRC threshold from MAPIE BCC (Bates 2024): FNR ≤ 10% w.p. ≥ 90%
         │
         ├─ event: salary_card    {salary_band_lpa, ipr_p50, anchor_weight, is_elite_outlier}
         ├─ event: placement_card {placement_probabilities, ipr_placement_rate, risk,
         │                         survival_curve, survival_method}
         ├─ event: drivers_card   {top_drivers, explanation}
         └─ event: copilot_card   {copilot_actions}

         ├─ event: result {full ~6 KB JSON payload}     ← idempotent re-render anchor
         └─ event: done   {latency_ms, parser_used}
```

Warm latency: ~3.5–5s end-to-end (paced for the demo; the actual computation is ~600ms — the rest is `await asyncio.sleep` between events to make the reveal humanly readable). Cold first-call: ~3s extra for DeepHit module rebuild.

---

## 4. Module-by-module reference

### `validity_gate.py` ★ NEW

Rule-based scorer. Reads first 2 pages (or pre-extracted text). Logistic aggregator over **8 positive structural signals** (email, phone, profile URL, ≥2 section headers, date timeline, bullet ratio, short-line ratio, text density) and **2 negative signals** (`not_marksheet` matching 12 telltales like "subject code", "grade sheet", "abstract...introduction...references"; `image_heavy` for <150 chars).

Produces `ValidityReport(p_resume, band, signals[], page_count, char_count, image_heavy)`. Three bands:
- `p_resume ≥ 0.70` → **accept**
- `0.40 ≤ p_resume < 0.70` → **warn**
- `p_resume < 0.40` → **reject** (hard stop, `rejection_message()` builds plain-English explanation)

**Why rule-based** (vs. the prompt's XGBoost suggestion): structural signals ARE the resume signal; rule-based is <50ms, deterministic, debuggable, and surfaces interpretable evidence for the judge.

### `pdf_extract.py` ★ NEW

Multi-column-aware text extractor. Per page:
1. `page.extract_words()` → word boxes
2. 1D 2-means on `x0` coordinates (8 iterations, ~1ms)
3. Multi-column iff cluster centers >150px apart AND each cluster ≥30% of words
4. If multi-column: emit left col sorted by `(top, x0)` then right col, joined into lines by y-jump >4pt
5. Else fall through to plain `extract_text()`

Also: duplicate-page detection by text hash; flags `has_pages_with_no_text` for the validity gate's image-heavy path. Output: `PDFExtraction(text, pages[], layout, page_count, duplicate_page_count)`.

### `ipr.py` ★ NEW

Institute Placement Registry. **In-process Python dict, loaded at module import** (no Redis, no disk read at inference, no TTL). Three data structures:

- **`INSTITUTE_PRIORS`** — 21 hand-curated institutes × 31 (degree, branch) entries with full salary percentile distribution `(p10, p25, p50, p75, p90)`, `placement_rate_3m/6m/12m`, `sample_size`, `year_bin`, `source`. Sources: NIRF 2024 DCS submissions, public placement reports from institute placement cells (2023–2024), College Dunia / Shiksha aggregates.
- **`ALIASES`** — 58 alias entries mapping common name spellings ("IIITA", "IIIT Prayagraj", "Indian Institute of Information Technology Allahabad") → canonical slug.
- **`CLUSTER_PRIORS`** — 14 cluster keys `(tier, degree, branch_family)` (computing / electronics / core / finance / general) with percentile distributions for the long-tail institutes.
- **`NATIONAL_BASELINE`** — 6 baselines by degree only (BTech / MTech / MBA / MCA / BSc / BCom).

`lookup(institute_name, course_type, tier_hint)` runs the **5-level fallback ladder** in order, returns an `IPRResult` with full provenance (`fallback_level`, `data_quality`, `source`, `sample_size`, `year_bin`).

**Specific cases the prompt called out, now correct:**
- IIIT Allahabad CSE → L1 exact (n=287, p50=₹18L, 6m=84%) — was undersold by NIRF rank 80
- VIT Vellore CSE → L1 exact (n=1240, p50=₹12L, 6m=78%) — was overinflated by NIRF rank 11
- Tier-3 unknown college → L4 cluster (tier-3 BTech computing, n=5800, p50=₹5.5L)
- Empty / unmatched institute → L5 national baseline (degree-only)

### `resume_parser.py` ★ UPDATED

Three-tier cascade: **OpenRouter → Gemini Flash → heuristic regex parser**. All three populate the same `StudentProfile` schema and `field_confidence` dict.

- **OpenRouter** — OpenAI-compatible chat completions, `openai/gpt-oss-120b:free`, `response_format={"type": "json_object"}`, temp 0.0, max_tokens 900. System: "You are a strict JSON extractor."
- **Gemini Flash** — `gemini-2.0-flash`, `generate_content()`, single user prompt (no system role).
- **Heuristic** — deterministic regex extractors for course / tier / region / name / CGPA / institute / internships / skills / certifications / hackathons / publications / leadership. Always works offline. Per-field confidence is **differentiated** (CGPA from `CGPA: X.X` → 0.95 vs. `X/10` → 0.70 vs. missing → 0.0).

**Elite-outlier detection** runs after parsing — `_detect_elite_outlier(text, profile)`. Triggers if **any 2+** of:
1. Top-tier internship from explicit `_FAANG_TIER` set (~30 companies)
2. ≥1 paper publication
3. ≥2 hackathon wins
4. CGPA ≥ 9.0
5. ≥500 coding problems

Bonus signal (always listed if found, doesn't count toward 2/5): ICPC / Putnam / Google Code Jam.

LLM status (success/rate-limit/auth-failed/parse-failed) is written to a module-level `LLM_STATUS` dict; `/api/llm_health` reads it for the UI status pill.

### `predict.py` ★ UPDATED

Inference path. Loads `data/models/bundle.joblib` lazily once. Steps:

1. `profile_to_vector(profile)` → 60-dim float32
2. `ipr.lookup(institute, course, tier_hint)` → `IPRResult`
3. `anchor_w = {0.70, 0.55, 0.40, 0.20}[ipr.data_quality]`
4. Raw model outputs (LightGBM × Beta calibration):
   - `raw_p_3m, raw_p_6m, raw_p_12m`
   - `raw_salary_low, raw_salary_med, raw_salary_high` (CQR α=0.10/0.50/0.90)
5. **Anchor placement probabilities**: `p_6m = α · ipr.placement_rate_6m + (1-α) · raw_p_6m`
6. **Anchor salary**: deviation, scale, anchored_p50, range from `ipr.p10..p90` scaled by deviation
7. **Elite branch**: if `profile.is_elite_outlier`, lower bound jumps to institute p50 / p75 (the "ICPC at IIT Bombay" case lands at ₹27–137L instead of synthetic-mean ₹15L)
8. **Confidence widening**: × 1.20 if parser confidence mean < 0.65; × 1.15 if IPR fallback level ≥ 4
9. Hard floors/ceilings (preserve `low ≤ med ≤ high`)
10. DeepHitSingle survival curve (lazy module rebuild from bundle state-dict; falls back to monotonic interpolation if torch is unavailable)
11. SHAP top-5 drivers (TreeExplainer on the calibrated 6m head)
12. Thompson-sampled Co-Pilot actions (Beta posteriors loaded once, dict lookup at inference)
13. CRC threshold from MAPIE BCC

Returns ~6 KB JSON with `profile`, `ipr` (full IPR result + anchor weight + raw_model + parse confidence), `elite_outlier`, `risk`, `placement_probabilities`, `salary_band_lpa`, `top_drivers`, `explanation`, `copilot_actions`, `survival_curve`, `early_warning_days`, `model_meta`.

### `stream.py` ★ NEW

Async SSE generator. Two entry points: `stream_pdf(pdf_bytes)` and `stream_text(text)`. Each yields one event per finding, with `await asyncio.sleep(PACE_FAST=100ms or PACE_SLOW=350ms)` between events for human-readable pacing.

Event types: `upload`, `stage`, `validity_signal`, `validity_result`, `rejected`, `parsed_field`, `parse_result`, `ipr_card`, `salary_card`, `placement_card`, `drivers_card`, `copilot_card`, `result`, `done`, `error`. The `result` event carries the full prediction payload so the UI can re-render idempotently.

### `schema.py` ★ UPDATED

`StudentProfile` dataclass (the wire format). Added 4 metadata fields:
- `field_confidence: dict | None` — `{field_name: 0..1}` plus `_layout`, `_page_count`, `_duplicate_pages`
- `is_elite_outlier: bool`
- `elite_outlier_reasons: list | None`
- `branch: str | None`, `degree: str | None` — optional explicit (degree, branch) for IPR L1 lookup

Backwards-compatible: defaults preserve pre-update behaviour.

### `main.py` ★ UPDATED

FastAPI app. Now:
- Bootstrap stubs for torch / pycox / torchtuples on Windows machines where torch DLLs fail (so the bundle still unpickles)
- `/api/score_from_resume` and `/api/score_from_text` now run the **validity gate first**; non-resumes get `{rejected: true, reason: ..., validity: {...}}` with no fabricated score
- `/api/score_from_resume_stream` and `/api/score_from_text_stream` — **new SSE endpoints** for the forensic-discovery flow
- `/api/health` — adds `ipr` stats (institutes, branches, clusters, baselines, aliases counts)
- `/api/ipr_stats` — sample IPR entries for the Architecture tab

---

## 5. Data sources

| Source | Used for | Provenance |
|---|---|---|
| **IPR** (`ipr.INSTITUTE_PRIORS`) | L1–L3 institute-specific salary p10–p90 + placement rates | Hand-curated from NIRF 2024 DCS submissions + public institute placement reports (2023–2024) + College Dunia / Shiksha aggregates |
| **NIRF 2024** (`nirf.NIRF_REGISTRY`) | L4 tier-cluster construction signal (one input among many — never the salary proxy) | NIRF 2024 official report (nirfindia.org) |
| **Synthetic borrowers** (`data/synthetic_students.csv`) | LightGBM training corpus (15k rows × 60 features) | Generated by `synth_data.py` with realistic Weibull survival DGP, course/tier multipliers, sector demand shocks |
| **OpenRouter free models** | LLM extraction primary path | `openai/gpt-oss-120b:free` (default), Llama 3.3 70B, etc. |
| **Gemini Flash 2.0** | LLM extraction fallback | Google AI Studio free tier |
| **Naukri JobSpeak Index** | PRI macro anchor | Info Edge monthly publication |
| **Kaggle** (optional) | Real-label OOD validation fold | Roshan + Tejashvi campus-placement datasets in `data/ood/` |

---

## 6. Scoring layer detail

### 6.1 Salary anchoring math

Given parsed profile and `IPRResult` from level L:

```
α = ANCHOR_WEIGHT_BY_QUALITY[ipr.data_quality]
        = 0.70 if "high"  (L1, n ≥ 100)
        = 0.55 if "medium" (L1 n 30-99, L2)
        = 0.40 if "low"   (L3, L4 small cluster)
        = 0.20 if "baseline" (L5)

raw_med = LightGBM_CQR_α=0.50.predict(feature_vector)
deviation = clamp(raw_med / ipr.p50, 0.6, 2.0)

anchored_p50 = α · ipr.p50 + (1-α) · raw_med
scale = anchored_p50 / ipr.p50

low_band  = ipr.p10 · scale
high_band = ipr.p90 · scale

if profile.is_elite_outlier:
    median_floor = max(anchored_p50, ipr.p75)
    low_band     = max(ipr.p50, anchored_p50 · 0.85)
    high_band    = max(ipr.p90 · 1.3, median_floor · 2.0)

widen = 1.0
if parse_conf_mean < 0.65: widen *= 1.20
if ipr.fallback_level >= 4: widen *= 1.15
range *= widen

# Hard floors/ceilings
low  = max(2.0, low)
high = min(high, ipr.p90 · (3.0 if elite else 1.5))
```

### 6.2 Placement probability anchoring

Three independent LightGBM classifiers (3m / 6m / 12m heads), each Beta-calibrated (Kull 2017). Each is blended with the IPR's measured rate:

```
p_horizon = α · ipr.placement_rate_horizon + (1-α) · calibrated_model_prob_horizon
```

When the IPR has high-quality data (e.g., L1 with 287 students), the institute's actual rate dominates (70%). When the IPR is sparse (L4 cluster or L5 national), the model's per-student prediction dominates (60–80%).

### 6.3 Risk score and CRC threshold

```
risk_score = round((1 - p_6m) * 100)   ∈ [0, 100]
risk_tier  = HIGH if score ≥ 60 else MEDIUM if ≥ 35 else LOW
crc_threshold = bundle["crc"]["risk_score_threshold"]   ← Bates 2024 Learn-Then-Test BCC
                                                          guarantees FNR ≤ 10% w.p. ≥ 90%
```

### 6.4 Top drivers (SHAP)

`shap.TreeExplainer` over the calibrated 6m head's underlying LightGBM. Top-5 features by `|SHAP value|`, sign-inverted so positive contributions = increase in risk. Background dataset: 200 rows pre-warmed at server startup → typical compute <400ms.

### 6.5 Co-Pilot (Thompson Sampling)

Beta-Bernoulli posteriors per `(action × segment)` — 6 actions (skill_cert, internship, mock_interview, portal_activity, resume_clinic, system_design) × ~12 segments (tier × branch × risk_tier). Posteriors are warm-started from EconML DR ATEs trained offline, then updated nightly via `/api/record_outcome` callbacks. At inference: dict lookup + posterior mean / 95% credible interval — no live Thompson sampling on the critical path.

---

## 7. SSE event catalog

| Event | Payload | When |
|---|---|---|
| `upload` | `{filename, size_bytes, size_kb}` | Immediately after PDF received |
| `stage` | `{name: validity\|parse\|ipr\|score, label}` | Stage transition |
| `validity_signal` | `{name, weight, found, detail}` | Per validity signal that fires (positive or negative) |
| `validity_result` | `{p_resume, band, signals[], page_count, char_count}` | After validity gate completes |
| `rejected` | `{reason, p_resume}` | If validity gate rejects (terminates stream) |
| `parsed_field` | `{field, label, value, confidence, imputed}` | One per parsed field (name, course, institute, CGPA, internships, skills, certifications) |
| `parse_result` | `{parser_used, layout, page_count, duplicate_pages, is_elite_outlier, elite_outlier_reasons[]}` | After parser cascade |
| `ipr_card` | Full `IPRResult.to_dict()` | After IPR lookup |
| `salary_card` | `{salary_band_lpa, ipr_p50, anchor_weight, is_elite_outlier, elite_reasons[]}` | After scoring |
| `placement_card` | `{placement_probabilities, ipr_placement_rate, risk, survival_curve, survival_method, early_warning_days}` | After scoring |
| `drivers_card` | `{top_drivers, explanation}` | After SHAP |
| `copilot_card` | `{copilot_actions}` | After bandit lookup |
| `result` | Full prediction payload (~6 KB) | Idempotent re-render anchor |
| `done` | `{latency_ms, parser_used}` | Stream ends |
| `error` | `{stage, message}` | Any stage failure (stream continues for non-fatal errors, terminates otherwise) |

---

## 8. API surface

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/score_from_resume` | POST | One-shot JSON scoring (validity-gated) |
| `/api/score_from_resume_stream` ★ | POST | **SSE forensic flow** — judge-facing |
| `/api/score_from_text` | POST | One-shot JSON over pasted text (validity-gated) |
| `/api/score_from_text_stream` ★ | POST | SSE over pasted text |
| `/api/predict` | POST | Score arbitrary JSON `ProfileBody` (no validity gate — for programmatic calls) |
| `/api/demo_profile/{key}` | GET | Score Priya / Arjun / Meera / Rahul personas |
| `/api/health` | GET | AUCs, ECE, salary MAE, DeepHit C-index, **IPR stats** |
| `/api/llm_health?probe=true` | GET | LLM cascade status (live indicator with optional 4-token probe) |
| `/api/ipr_stats` ★ | GET | IPR coverage sample for Architecture tab |
| `/api/portfolio?limit=&risk_tier=` | GET | 200-borrower book (filterable) |
| `/api/pri` | GET | Naukri-JobSpeak-anchored 13-month PRI series |
| `/api/record_outcome` | POST | Update Beta(α, β) bandit posterior with 0/1 outcome |
| `/api/bandit_state` | GET | Bandit posteriors per (segment × action) |
| `/api/causal` | GET | PC-discovered CPDAG + Markov blanket |
| `/api/federated_shap` | GET | DP-aggregated SHAP across 2 shards |
| `/api/counterfactual` | GET | DR ATE per Co-Pilot action with placebo refuter |
| `/api/edge_model` | GET | Born-Again tree as JSON for offline deployment |
| `/api/fairness` | GET | Fairlearn audit results |
| `/api/drift` | GET | PSI + KS per feature with industry thresholds |
| `/api/ood` | GET | OOD validation reports (real-label or fallback) |

★ = new in this rebuild.

---

## 9. Latency budget

| Stage | Cold | Warm | Notes |
|---|---|---|---|
| Validity gate | 50ms | 30ms | pdfplumber first 2 pages + 10 regex signals |
| Column-aware extraction | 80ms / page | 50ms / page | Plus 1ms for the 2-means |
| LLM parse (OpenRouter) | 3–8s | 2–6s | External network — fallback to heuristic if slow |
| LLM parse (Gemini) | 2–5s | 1.5–4s | Same |
| Heuristic parse | 30ms | 15ms | Pure regex, no I/O |
| IPR lookup | 5ms | <1ms | Dict lookup + rapidfuzz fuzzy on 58 aliases |
| Feature vector | 5ms | 3ms | StudentProfile → 60-dim float32 |
| LightGBM × 3 + CQR × 3 | 80ms | 30ms | Fast tree inference |
| Beta calibration × 3 | 5ms | 2ms | Closed-form transform |
| DeepHit survival | 2.5s | 50ms | Cold = first lazy module rebuild from state-dict |
| SHAP top-5 | 600ms | 300ms | TreeExplainer with 200-row pre-warmed background |
| Bandit lookup | <1ms | <1ms | Pre-loaded posteriors |
| **Total compute** | **~10s** | **~600ms warm** (no LLM) or **~3–6s warm** (with LLM) |

The streaming UI **adds ~3.5s of intentional pacing** (`asyncio.sleep` between events) so the judge sees the discovery happen one bullet at a time. The actual inference is ~600ms warm — the rest is theatre.

---

## 10. Failure modes and fallbacks

| Failure | What happens | Why it's OK |
|---|---|---|
| Non-resume PDF (marksheet, paper) | Validity gate rejects with `p_resume < 0.40`; SSE emits `rejected` with explicit evidence; no scoring | Prompt's #1 requirement: "If not a resume, stop early and explain why" |
| Image-only / scanned PDF | Validity gate flags `image_heavy`; Stage 2 returns empty text; parser gets `(StudentProfile(), "empty")` | Honest about being unable to score; no hallucination |
| OpenRouter rate-limited | Status `rate_limited` in `LLM_STATUS`; cascade falls through to Gemini | UI pill turns amber; user sees no error |
| Both LLMs fail | Heuristic regex parser runs (always works) | Lower extraction quality, but typed `StudentProfile` is still produced |
| Multi-column resume | `pdf_extract.extract_pdf()` clusters word x-coords; emits left col then right col | Handles ~80% of multi-column layouts; UI surfaces "multicolumn detected" |
| Institute name not in IPR | 5-level fallback ladder lands at L4 cluster or L5 national; `data_quality = "low"` or `"baseline"` | Salary range gets widened by 1.15× to honestly reflect uncertainty |
| Missing CGPA | Default 7.0 used in feature vector but `field_confidence["cgpa"] = 0.0`; UI shows "imputed" badge; salary widened by 1.20× | Never fabricated — just imputed with provenance |
| torch DLL load fails on Windows | `main.py` bootstraps a stub `torch` module so netcal can unpickle Beta calibrators; DeepHit falls back to interpolated survival | System still runs end-to-end; survival curve is monotonic interpolation instead of DeepHit |
| Low parser confidence (<0.65 mean) | Salary range × 1.20 widening; UI shows amber confidence pills per field | Honest uncertainty, not fake precision |
| IPR fallback level ≥ 4 | Salary range × 1.15 widening; UI shows L4 / L5 badge in IPR card | Judge sees explicitly that institute data is sparse |

---

## 11. What deliberately stays OFF the critical path

These all run offline / batch / nightly, never at inference:

- BHM training (Bayesian hierarchical model for institute distributions) — empirical IPR is the practical equivalent
- Causal DAG discovery (PC algorithm)
- Counterfactual ATE estimation (EconML LinearDRLearner)
- Federated SHAP across shards
- Fairlearn audit
- PSI / KS drift detection
- OOD validation
- Born-Again tree distillation
- Bandit posterior MCMC updates (run nightly via outcome ingestion)
- Multi-shot LLM refinement (the cascade does at most ONE LLM call per request; no back-and-forth loops)

Each is exposed via its own GET endpoint for the Architecture tab. None of them block the live demo.

---

## 12. Citations index

| Component | Paper |
|---|---|
| Beta calibration | Kull, Silva Filho, Flach — AISTATS 2017 |
| Conformal Risk Control | Bates, Angelopoulos, Lei, Malik, Jordan — JACM 2024 (arXiv:2101.02703) |
| Thompson Sampling | Russo, Van Roy et al. — FnT-ML 2018 (arXiv:1707.02038) |
| DeepHit | Lee et al. — AAAI 2018 |
| PC algorithm | Spirtes & Glymour — Causation, Prediction, and Search (2000) |
| Markov-blanket selection | Yu et al. — Causal Learner Toolbox — arXiv:2103.06544 |
| Born-Again Tree | Vidal & Schiffer — ICML 2020 (arXiv:2003.11132) |
| DR estimation | Coston, Mishler, Kennedy, Chouldechova — FAccT 2020 (arXiv:1909.00066) |
| Federated SHAP + DP | Saifullah et al. — Frontiers in AI 2024 |
| Differential privacy | Dwork & Roth — Foundations & Trends 2014 |
| Drift detection | Rabanser, Günnemann, Lipton — NeurIPS 2019 (arXiv:1810.11953) |
| Fair credit scoring | Kozodoi, Jacob, Lessmann — EJOR 2022 (arXiv:2103.01907) |
| DPDP Act 2023 §10 | meity.gov.in |
| NIRF 2024 priors (L4 cluster construction only) | nirfindia.org |

---

## 13. What's still honest about being incomplete

- **IPR coverage is 21 institutes, 31 institute-branches.** Long tail (~98% of institutes a real lender would see) lands at L4 cluster fallback. Production deployment would replace this dict with a lender's full institute database and fresh annual placement-report scraping.
- **Synthetic training corpus.** LightGBM heads were trained on 15k synthetic borrowers. The IPR anchor partially compensates by overriding salary/placement output for high-data-quality institutes, but full retraining on real lender data would tighten the residual.
- **Elite-outlier rule is a fixed 5-signal heuristic.** Production version should learn the trigger boundary from labelled data.
- **Multi-column extraction handles ~80% of layouts.** Heavily-designed Canva templates can still trip the column detector; those land back on the LLM cascade which is robust to garbled text.
- **Fairness audit reports tier-disparate impact at DI = 0.15.** Surfaced honestly in `/api/fairness`. Mitigation path on display: `fairlearn.reductions.ExponentiatedGradient(DemographicParity)`.

---

*Last updated 2026-05-02. The build that produced this document is reproducible end-to-end: `pip install -r backend/requirements.txt && python -m uvicorn backend.app.main:app --port 8000`. The IPR is hand-curated; everything else is generated from working code, not specs.*
