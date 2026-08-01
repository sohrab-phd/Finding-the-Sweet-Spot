"""Reproduce Figures 1 and 2 from the paper."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation import MODEL_DISPLAY, PAPER_TABLE_II, SUBSET_ORDER
from src.utils import PROJECT_ROOT, ensure_dirs

logger = logging.getLogger(__name__)

FRACTIONS = {
    "SE.1": 1 / 128,
    "SE.2": 1 / 64,
    "SE.3": 1 / 32,
    "SE.4": 1 / 16,
    "SE.5": 1 / 8,
    "SE.6": 1 / 4,
    "SE.7": 1 / 2,
    "SE.8": 1.0,
}

# Visual style matching paper Figures 1–2
STYLE = {
    "Att-CNN": {"color": "#1f77b4", "marker": "o", "label": "Att-CNN"},
    "Att-BiLSTM": {"color": "#ff7f0e", "marker": "*", "label": "Att-BLSTM", "markersize": 10},
    "R-BERT": {"color": "#2ca02c", "marker": "+", "label": "R-BERT", "markersize": 10},
    "MTB": {"color": "#d62728", "marker": "x", "label": "MTB", "markersize": 8},
}


def _series(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    model: str,
    metric: str,
) -> tuple:
    xs, ys = [], []
    for se in SUBSET_ORDER:
        if se in all_metrics.get(model, {}) and metric in all_metrics[model][se]:
            xs.append(FRACTIONS[se])
            ys.append(all_metrics[model][se][metric])
    return np.array(xs), np.array(ys)


def plot_learning_curve(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    metric: str,
    ylabel: str,
    out_path: Path,
    threshold: float = 0.5,
    title: Optional[str] = None,
) -> Path:
    ensure_dirs(out_path.parent)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

    for model, style in STYLE.items():
        xs, ys = _series(all_metrics, model, metric)
        if len(xs) == 0:
            continue
        ax.plot(
            xs,
            ys,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            linewidth=1.5,
            markersize=style.get("markersize", 6),
        )

    ax.axvline(x=threshold, color="green", linestyle=":", linewidth=1.5)
    ax.set_xlabel("Training size")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    logger.info("Wrote figure %s", out_path)
    return out_path


def plot_figure1(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Optional[Path] = None,
) -> Path:
    out_dir = Path(out_dir or PROJECT_ROOT / "results" / "figures")
    return plot_learning_curve(
        all_metrics,
        metric="Accuracy",
        ylabel="Accuracy",
        out_path=out_dir / "figure1_accuracy.png",
        title="Impact of training data size on Accuracy",
    )


def plot_figure2(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    out_dir: Optional[Path] = None,
) -> Path:
    out_dir = Path(out_dir or PROJECT_ROOT / "results" / "figures")
    return plot_learning_curve(
        all_metrics,
        metric="F-score",
        ylabel="F score",
        out_path=out_dir / "figure2_fscore.png",
        title="Impact of training data size on F-score",
    )


def plot_paper_reference_figures(out_dir: Optional[Path] = None) -> None:
    """Plot Figures 1–2 from paper-reported Table II (reference curves)."""
    out_dir = Path(out_dir or PROJECT_ROOT / "results" / "figures")
    nested = {
        model: {
            se: {m: PAPER_TABLE_II[model][m][se] for m in ["Accuracy", "Precision", "Recall", "F-score"]}
            for se in SUBSET_ORDER
        }
        for model in PAPER_TABLE_II
    }
    plot_learning_curve(
        nested,
        "Accuracy",
        "Accuracy",
        out_dir / "figure1_accuracy_paper_reference.png",
        title="Figure 1 (paper-reported values)",
    )
    plot_learning_curve(
        nested,
        "F-score",
        "F score",
        out_dir / "figure2_fscore_paper_reference.png",
        title="Figure 2 (paper-reported values)",
    )
