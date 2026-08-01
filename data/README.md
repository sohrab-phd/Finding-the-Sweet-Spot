# Data directory

## `raw/`

SemEval-2010 Task 8 files are downloaded automatically by:

```bash
python reproduce.py --stage data
```

Do not commit the zip or extracted `TRAIN_FILE.TXT` / `TEST_FILE_FULL.TXT` files.

## `processed/`

- `subsets/SE.*.json` — training-instance indices for Algorithm 1 (kept in the repository for reproducibility)
- `train.json` / `test.json` / `vocab/` — generated locally; gitignored
