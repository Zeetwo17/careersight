# CareerSight

**AI · Education Loan Risk Intelligence**
Team Aurdinary · IIIT Allahabad · Hridyesh · Harsh

CareerSight links education loans to career outcomes — predicting placement
timelines, salary ranges, and delayed-repayment risk so lenders can intervene
90+ days early and students get targeted support. **Not credit automation.**

Research-grade hackathon MVP. Every "world-first" innovation in the deck is
implemented end-to-end with off-the-shelf-library code and a paper citation.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the full module-by-module breakdown.

---

## What's in here

```
CareerSight/
├── backend/app/
│   ├── schema.py            # 60-feature schema (45 numeric + NIRF + 12 one-hot)
│   ├── synth_data.py        # 15k synthetic borrowers
│   ├── features.py          # StudentProfile -> vector + NIRF prior lookup
│   ├── nirf.py              # NIRF 2024 institute registry + rapidfuzz matching
│   ├── train.py             # End-to-end pipeline: LGBM + Beta cal + DeepHit + CRC + ...
│   ├── predict.py           # Inference (Beta-calibrated, DeepHit survival, TS bandit)
│   ├── calibration.py       # Beta calibration (Kull 2017) + ECE/Brier
│   ├── crc.py               # MAPIE Learn-Then-Test BCC (Bates 2024)
│   ├── survival.py          # DeepHitSingle on {3, 6, 12} cuts (Lee 2018)
│   ├── bandit.py            # Beta-Bernoulli Thompson Sampling Co-Pilot
│   ├── counterfactual.py    # EconML DR ATE per action + placebo refuter
│   ├── causal.py            # PC algorithm causal DAG + Markov blanket
│   ├── federated.py         # Federated SHAP + Gaussian DP across 2 shards
│   ├── edge_model.py        # Born-Again single tree (Vidal 2020), int16 JSON
│   ├── fairness.py          # Fairlearn audit (DPDP §10 + EEOC four-fifths)
│   ├── drift.py             # PSI + KS Bonferroni-corrected
│   ├── ood.py               # Kaggle real-label OOD validation fold
│   ├── portfolio.py         # Lender book + Naukri-JobSpeak PRI
│   ├── demo_profiles.py     # Priya / Arjun / Meera / Rahul personas
│   ├── resume_parser.py     # PDF -> StudentProfile (Gemini Flash + heuristic fallback)
│   └── main.py              # FastAPI app — 16 endpoints + static SPA
├── frontend/
│   ├── index.html           # Single-page dashboard
│   ├── style.css            # Navy + orange branding (matches pitch deck)
│   └── app.js               # Tabs, charts, all card renderers
├── data/
│   ├── synthetic_students.csv
│   ├── ood/                 # Drop Kaggle CSVs here for real-label OOD
│   ├── bandit_state.json    # Persisted Beta(α, β) posteriors
│   └── models/bundle.joblib (~10.8 MB)
├── ARCHITECTURE.md
├── README.md
├── CareerSight.pdf          # Pitch deck (11 slides)
└── ...
```

---

## Quick start

### 1. Install dependencies

```bash
cd CareerSight
pip install -r backend/requirements.txt
```

The full stack uses: `lightgbm`, `shap`, `netcal` (calibration), `mapie` (CRC),
`fairlearn`, `causal-learn` (PC algorithm), `econml`+`dowhy` (counterfactual),
`pycox`+`torch`+`torchtuples` (DeepHit), `rapidfuzz` (NIRF matching),
`pdfplumber`, `fastapi`+`uvicorn`.

### 2. Generate the synthetic dataset (one-time)

```bash
python -m backend.app.synth_data --n 15000 --out data/synthetic_students.csv
```

15,000 borrowers with 60 features and placement outcomes at 3 / 6 / 12 months.

### 3. Train the bundle (one-time, ~3-5 min)

```bash
python -m backend.app.train
```

Pipeline:

- 3× LightGBM placement-timeline classifiers (3 / 6 / 12-month).
- 3× LightGBM Conformal Quantile Regression salary models (10 / 50 / 90 percentile).
- Beta calibrators per horizon (Kull 2017); reports ECE pre/post.
- MAPIE Learn-Then-Test CRC controller (Bates 2024) → λ* threshold.
- DeepHitSingle survival on cuts {3, 6, 12} → Antolini C-index.
- EconML LinearDRLearner per Co-Pilot action → ATE + placebo.
- Born-Again single tree distillation → ~10 KB JSON edge model.
- Federated SHAP across 2 shards + Gaussian DP (ε=1.0, δ=1e-5).
- PC algorithm causal DAG + Markov blanket.
- Fairlearn audit on tier × is_metro × course_type.
- Kaggle OOD validation (or covariate-shift fallback).
- SHAP TreeExplainer for the inference path.

Reported metrics on the current bundle: AUCs 0.85 / 0.86 / 0.79, ECE 0.027,
DeepHit C-index 0.769, edge model 10.7 KB, federated Spearman ρ 0.94.

### 4. Run the API + dashboard

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>.

The dashboard has four tabs:

- **Live Demo** — drop a PDF or click a persona; full risk surface in <200 ms warm.
- **Lender Portfolio** — 200-borrower book scored against the live model.
- **PRI Index** — Naukri-JobSpeak-anchored 13-month PlacementRisk Index.
- **Architecture** — pipeline diagram, **14 LIVE innovations**, model diagnostics, calibration reliability, OOD card, drift PSI, fairness audit, causal DAG, Federated SHAP, counterfactual ATEs, edge-model card.

---

## Optional: enable Gemini Flash resume parsing

```bash
export GEMINI_API_KEY=your_key_here   # macOS / Linux
$env:GEMINI_API_KEY="your_key_here"   # Windows PowerShell
```

Without a key, a deterministic rule-based parser kicks in (fuzzy course/tier
detection, regex CGPA extraction, top-tier-company keyword catalogue, etc.).

## Optional: enable real-label OOD validation

Drop these public CC0 CSVs into `data/ood/`:

- `campus_placement_roshan.csv` — <https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement>
- `engineering_placements_tejashvi.csv` — <https://www.kaggle.com/datasets/tejashvi14/engineering-placements-prediction>

Re-run `python -m backend.app.train`. The Architecture tab's OOD card will
show real-label AUC + ECE instead of the covariate-shift fallback.

---

## API surface

| Endpoint                                | Method | Purpose                                                  |
| --------------------------------------- | ------ | -------------------------------------------------------- |
| `/api/health`                           | GET    | Model AUCs, ECE, salary MAE, DeepHit C-index             |
| `/api/demo_profile/{key}`               | GET    | Score Priya / Arjun / Meera / Rahul                      |
| `/api/predict`                          | POST   | Score arbitrary `ProfileBody` JSON                       |
| `/api/score_from_resume`                | POST   | Multipart PDF → full risk surface                        |
| `/api/score_from_text`                  | POST   | Pasted resume text → full risk surface                   |
| `/api/portfolio?limit=&risk_tier=`      | GET    | 200-borrower book                                        |
| `/api/pri`                              | GET    | Naukri-JobSpeak-anchored 13-month PRI                    |
| `/api/record_outcome`                   | POST   | Update Beta(α, β) bandit posterior                       |
| `/api/bandit_state`                     | GET    | Bandit posteriors per (segment × action)                 |
| `/api/causal`                           | GET    | PC-discovered CPDAG + Markov blanket                     |
| `/api/federated_shap`                   | GET    | DP-aggregated SHAP across 2 shards                       |
| `/api/counterfactual`                   | GET    | DR ATE per Co-Pilot action + placebo                     |
| `/api/edge_model`                       | GET    | Born-Again tree as JSON                                  |
| `/api/fairness`                         | GET    | Fairlearn audit                                          |
| `/api/drift`                            | GET    | PSI + KS                                                 |
| `/api/ood`                              | GET    | OOD validation reports                                   |

OpenAPI docs auto-generated at <http://127.0.0.1:8000/docs>.

---

## License

Hackathon submission — Team Aurdinary, IIIT Allahabad. All rights reserved.
