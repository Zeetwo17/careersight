# OOD validation data

Drop one or both of these files in this directory to enable real-label
out-of-distribution validation. Both are public CC0 datasets.

## campus_placement_roshan.csv (215 rows)

Source: <https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement>

Columns used: `degree_p`, `ssc_p`, `mba_p`, `workex`, `specialisation`, `status`.

To download with the Kaggle CLI:

```bash
kaggle datasets download -d benroshan/factors-affecting-campus-placement -p ./
unzip factors-affecting-campus-placement.zip
mv Placement_Data_Full_Class.csv campus_placement_roshan.csv
```

## engineering_placements_tejashvi.csv (~2,966 rows)

Source: <https://www.kaggle.com/datasets/tejashvi14/engineering-placements-prediction>

Columns used: `Stream`, `Internships`, `CGPA`, `HistoryOfBacklogs`, `PlacedOrNot`.

```bash
kaggle datasets download -d tejashvi14/engineering-placements-prediction -p ./
unzip engineering-placements-prediction.zip
mv collegePlace.csv engineering_placements_tejashvi.csv
```

## Without Kaggle CLI

`backend/app/ood.py` falls back to a covariate-shifted slice of our own
synthetic data. AUC reported on that fallback is conservative — real
Kaggle labels typically score higher because the schema overlap is partial.

After dropping the CSVs here, retrain to refresh `data/models/bundle.joblib`:

```bash
python -m backend.app.train
```
