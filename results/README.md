# Results

This directory stores experimental outputs.

| Path | Contents |
|------|----------|
| `metrics/` | Per-run Accuracy / Precision / Recall / F-score JSON |
| `tables/` | Table I and Table II (CSV, Markdown, LaTeX) |
| `figures/` | Figures 1–2 (PNG, PDF) |
| `checkpoints/` | Trained model weights (local only; not tracked by git) |

Regenerate tables and figures from metrics:

```bash
python reproduce.py --stage tables
```
