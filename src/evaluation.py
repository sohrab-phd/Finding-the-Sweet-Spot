"""Evaluation metrics, table export, and comparison vs paper-reported numbers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.utils import PROJECT_ROOT, ensure_dirs, save_json

logger = logging.getLogger(__name__)

# Paper Table II reported values
PAPER_TABLE_II: Dict[str, Dict[str, Dict[str, float]]] = {
    "Att-CNN": {
        "Accuracy": {"SE.1": 0.22, "SE.2": 0.26, "SE.3": 0.37, "SE.4": 0.48, "SE.5": 0.58, "SE.6": 0.66, "SE.7": 0.73, "SE.8": 0.77},
        "Precision": {"SE.1": 0.21, "SE.2": 0.16, "SE.3": 0.26, "SE.4": 0.41, "SE.5": 0.57, "SE.6": 0.64, "SE.7": 0.74, "SE.8": 0.79},
        "Recall": {"SE.1": 0.20, "SE.2": 0.25, "SE.3": 0.30, "SE.4": 0.41, "SE.5": 0.59, "SE.6": 0.69, "SE.7": 0.74, "SE.8": 0.79},
        "F-score": {"SE.1": 0.16, "SE.2": 0.19, "SE.3": 0.27, "SE.4": 0.40, "SE.5": 0.56, "SE.6": 0.66, "SE.7": 0.74, "SE.8": 0.79},
    },
    "Att-BiLSTM": {
        "Accuracy": {"SE.1": 0.30, "SE.2": 0.45, "SE.3": 0.46, "SE.4": 0.60, "SE.5": 0.66, "SE.6": 0.72, "SE.7": 0.75, "SE.8": 0.79},
        "Precision": {"SE.1": 0.34, "SE.2": 0.42, "SE.3": 0.42, "SE.4": 0.59, "SE.5": 0.65, "SE.6": 0.70, "SE.7": 0.76, "SE.8": 0.81},
        "Recall": {"SE.1": 0.33, "SE.2": 0.39, "SE.3": 0.42, "SE.4": 0.60, "SE.5": 0.73, "SE.6": 0.76, "SE.7": 0.77, "SE.8": 0.80},
        "F-score": {"SE.1": 0.25, "SE.2": 0.37, "SE.3": 0.38, "SE.4": 0.57, "SE.5": 0.65, "SE.6": 0.71, "SE.7": 0.75, "SE.8": 0.80},
    },
    "R-BERT": {
        "Accuracy": {"SE.1": 0.20, "SE.2": 0.26, "SE.3": 0.41, "SE.4": 0.65, "SE.5": 0.73, "SE.6": 0.79, "SE.7": 0.82, "SE.8": 0.84},
        "Precision": {"SE.1": 0.24, "SE.2": 0.26, "SE.3": 0.34, "SE.4": 0.65, "SE.5": 0.71, "SE.6": 0.75, "SE.7": 0.77, "SE.8": 0.79},
        "Recall": {"SE.1": 0.13, "SE.2": 0.12, "SE.3": 0.28, "SE.4": 0.56, "SE.5": 0.69, "SE.6": 0.77, "SE.7": 0.81, "SE.8": 0.83},
        "F-score": {"SE.1": 0.07, "SE.2": 0.11, "SE.3": 0.27, "SE.4": 0.54, "SE.5": 0.69, "SE.6": 0.76, "SE.7": 0.79, "SE.8": 0.81},
    },
    "MTB": {
        "Accuracy": {"SE.1": 0.19, "SE.2": 0.22, "SE.3": 0.38, "SE.4": 0.59, "SE.5": 0.71, "SE.6": 0.78, "SE.7": 0.83, "SE.8": 0.85},
        "Precision": {"SE.1": 0.19, "SE.2": 0.23, "SE.3": 0.32, "SE.4": 0.58, "SE.5": 0.69, "SE.6": 0.74, "SE.7": 0.76, "SE.8": 0.79},
        "Recall": {"SE.1": 0.11, "SE.2": 0.12, "SE.3": 0.25, "SE.4": 0.51, "SE.5": 0.66, "SE.6": 0.73, "SE.7": 0.78, "SE.8": 0.80},
        "F-score": {"SE.1": 0.03, "SE.2": 0.09, "SE.3": 0.20, "SE.4": 0.51, "SE.5": 0.67, "SE.6": 0.74, "SE.7": 0.78, "SE.8": 0.80},
    },
}

MODEL_DISPLAY = {
    "att_cnn": "Att-CNN",
    "att_bilstm": "Att-BiLSTM",
    "rbert": "R-BERT",
    "mtb": "MTB",
}

SUBSET_ORDER = [f"SE.{i}" for i in range(1, 9)]
METRIC_ORDER = ["Accuracy", "Precision", "Recall", "F-score"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> Dict[str, float]:
    """Compute Accuracy, Precision, Recall, F-score as reported in the paper."""
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "Accuracy": round(acc, 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F-score": round(float(f1), 4),
    }


def compute_official_semeval_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Approximate official SemEval-2010 Task 8 macro-F1:
    macro-average over 18 directed relations, excluding Other.
    """
    labels = list(range(18))
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return float(f1)


def export_table_i(
    subset_sizes: Dict[str, int],
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Regenerate Table I."""
    out_dir = Path(out_dir or PROJECT_ROOT / "results" / "tables")
    ensure_dirs(out_dir)
    fractions = {
        "SE.1": "1/128",
        "SE.2": "1/64",
        "SE.3": "1/32",
        "SE.4": "1/16",
        "SE.5": "1/8",
        "SE.6": "1/4",
        "SE.7": "1/2",
        "SE.8": "1",
    }
    rows = []
    for name in SUBSET_ORDER:
        rows.append(
            {
                "Dataset Notation": name,
                "Portion": fractions[name],
                "Number of instances": subset_sizes.get(name, ""),
            }
        )
    df = pd.DataFrame(rows)
    _write_table(df, out_dir / "table_i_subsets")
    return df


def metrics_dict_to_table_ii(all_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Build Table II-style dataframe.

    all_metrics: {display_model: {subset: {Accuracy, Precision, Recall, F-score}}}
    """
    rows = []
    for model in ["Att-CNN", "Att-BiLSTM", "R-BERT", "MTB"]:
        if model not in all_metrics:
            continue
        for metric in METRIC_ORDER:
            row = {"Classification Model": model, "Performance Metric": metric}
            for se in SUBSET_ORDER:
                val = all_metrics.get(model, {}).get(se, {}).get(metric, None)
                row[se] = val
            rows.append(row)
    return pd.DataFrame(rows)


def export_table_ii(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    out_dir = Path(out_dir or PROJECT_ROOT / "results" / "tables")
    ensure_dirs(out_dir)
    df = metrics_dict_to_table_ii(all_metrics)
    _write_table(df, out_dir / "table_ii_performance")
    return df


def export_paper_table_ii(out_dir: Optional[Path] = None) -> pd.DataFrame:
    """Export the paper-reported Table II for reference / comparison."""
    out_dir = Path(out_dir or PROJECT_ROOT / "results" / "tables")
    ensure_dirs(out_dir)
    nested = {
        model: {
            se: {m: PAPER_TABLE_II[model][m][se] for m in METRIC_ORDER}
            for se in SUBSET_ORDER
        }
        for model in PAPER_TABLE_II
    }
    df = metrics_dict_to_table_ii(nested)
    _write_table(df, out_dir / "table_ii_paper_reported")
    return df


def _write_table(df: pd.DataFrame, stem: Path) -> None:
    stem = Path(stem)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    df.to_markdown(stem.with_suffix(".md"), index=False)
    try:
        tex = df.to_latex(index=False, float_format="%.2f")
    except Exception:  # noqa: BLE001
        tex = df.to_string(index=False)
    stem.with_suffix(".tex").write_text(tex, encoding="utf-8")
    logger.info("Wrote table %s.{csv,md,tex}", stem.name)


def compare_with_paper(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Build absolute and percentage differences vs paper Table II."""
    out_dir = Path(out_dir or PROJECT_ROOT / "results" / "tables")
    ensure_dirs(out_dir)
    rows = []
    for model_key, display in MODEL_DISPLAY.items():
        # accept either key form
        model_metrics = all_metrics.get(display) or all_metrics.get(model_key) or {}
        for metric in METRIC_ORDER:
            for se in SUBSET_ORDER:
                reported = PAPER_TABLE_II[display][metric][se]
                reproduced = model_metrics.get(se, {}).get(metric)
                if reproduced is None:
                    continue
                abs_diff = float(reproduced) - float(reported)
                pct = (abs_diff / reported * 100.0) if reported != 0 else float("nan")
                rows.append(
                    {
                        "Model": display,
                        "Metric": metric,
                        "Subset": se,
                        "Reported": reported,
                        "Reproduced": reproduced,
                        "AbsDiff": round(abs_diff, 4),
                        "PctDiff": round(pct, 2) if pct == pct else None,
                    }
                )
    df = pd.DataFrame(rows)
    _write_table(df, out_dir / "comparison_vs_paper")
    save_json(rows, PROJECT_ROOT / "results" / "metrics" / "comparison_vs_paper.json")
    return df


def save_run_metrics(
    model: str,
    subset: str,
    metrics: Dict[str, float],
    out_dir: Optional[Path] = None,
) -> None:
    out_dir = Path(out_dir or PROJECT_ROOT / "results" / "metrics")
    ensure_dirs(out_dir)
    path = out_dir / f"{model}_{subset}.json"
    save_json({"model": model, "subset": subset, **metrics}, path)


def load_all_run_metrics(metrics_dir: Optional[Path] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load results/metrics/*_SE.*.json into nested Table-II structure."""
    import json
    import re

    metrics_dir = Path(metrics_dir or PROJECT_ROOT / "results" / "metrics")
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    skip_names = {"all_metrics.json", "comparison_vs_paper.json", "summary.txt"}
    run_pattern = re.compile(r"^.+_SE\.\d+\.json$", re.IGNORECASE)

    for path in sorted(metrics_dir.glob("*.json")):
        if path.name in skip_names or path.name.startswith("comparison"):
            continue
        # Prefer explicit per-run files: model_SE.N.json
        if not run_pattern.match(path.name):
            logger.debug("Skipping non-run metrics file: %s", path.name)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not all(k in data for k in ("model", "subset", "Accuracy", "Precision", "Recall", "F-score")):
            logger.warning("Skipping incomplete metrics file: %s", path.name)
            continue
        model = data["model"]
        subset = data["subset"]
        display = MODEL_DISPLAY.get(model, model)
        result.setdefault(display, {})
        result[display][subset] = {
            "Accuracy": data["Accuracy"],
            "Precision": data["Precision"],
            "Recall": data["Recall"],
            "F-score": data["F-score"],
        }
    return result
