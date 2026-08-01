#!/usr/bin/env python
"""
Single-command reproduction entry point for:
  Finding the Sweet Spot: An Empirical Study on Dataset Size,
  Performance, and Efficiency in Relation Extraction

Usage examples:
  python reproduce.py                     # full pipeline
  python reproduce.py --stage data        # download + partition only
  python reproduce.py --stage tables      # regenerate tables/figures from saved metrics
  python reproduce.py --models att_cnn --subsets SE.8
  python reproduce.py --quick             # fewer epochs (smoke test)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import get_subset_examples, load_processed, prepare_dataset
from src.evaluation import (
    MODEL_DISPLAY,
    compare_with_paper,
    compute_metrics,
    export_paper_table_ii,
    export_table_i,
    export_table_ii,
    load_all_run_metrics,
    save_run_metrics,
)
from src.preprocessing import build_vocabularies, load_word_embeddings
from src.training import run_experiment
from src.utils import ensure_dirs, load_config, save_json, set_seed, setup_logging
from src.visualization import plot_figure1, plot_figure2, plot_paper_reference_figures

logger = logging.getLogger("reproduce")

ALL_MODELS = ["att_cnn", "att_bilstm", "rbert", "mtb"]
ALL_SUBSETS = [f"SE.{i}" for i in range(1, 9)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce Sweet Spot RE experiments")
    p.add_argument(
        "--stage",
        choices=["all", "data", "train", "tables", "report"],
        default="all",
        help="Pipeline stage to run",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        help="Models to train (att_cnn att_bilstm rbert mtb)",
    )
    p.add_argument(
        "--subsets",
        nargs="+",
        default=ALL_SUBSETS,
        help="Subsets to train (SE.1 … SE.8)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Reduce epochs for a smoke-test run",
    )
    p.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip downloading GoogleNews embeddings (random init)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="torch device: cuda (default), auto, cpu, or cuda:0",
    )
    return p.parse_args()


def stage_data(cfg: dict) -> dict:
    logger.info("=== Stage: data download / parse / Algorithm 1 partitioning ===")
    set_seed(int(cfg.get("seed", 42)))
    data = prepare_dataset(
        seed=int(cfg.get("seed", 42)),
        subset_fractions=cfg.get("subset_fractions"),
        subset_sizes=cfg.get("subset_sizes"),
    )
    export_table_i(data["meta"]["subsets"])
    export_paper_table_ii()
    plot_paper_reference_figures()
    return data


def _apply_quick(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["epochs"] = min(int(cfg.get("epochs", 5)), 2)
    cfg["batch_size"] = min(int(cfg.get("batch_size", 16)), 8)
    return cfg


def stage_train(
    models: List[str],
    subsets: List[str],
    seed: int,
    device: str,
    quick: bool = False,
    skip_embeddings: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    logger.info("=== Stage: training ===")
    set_seed(seed)
    data = load_processed()
    train, test, subset_idx = data["train"], data["test"], data["subsets"]

    # Shared vocab / embeddings for neural models
    need_neural = any(m in models for m in ("att_cnn", "att_bilstm"))
    word_vocab = pos_vocab = emb_matrix = None
    if need_neural:
        vocabs = build_vocabularies(train)
        word_vocab, pos_vocab = vocabs["word"], vocabs["pos"]
        neural_cfg = load_config("att_cnn")
        if skip_embeddings or quick:
            import numpy as np

            dim = int(neural_cfg.get("embedding_dim", 300))
            emb_matrix = np.random.uniform(-0.05, 0.05, size=(len(word_vocab), dim)).astype("float32")
            emb_matrix[0] = 0.0
            logger.info("Using random embeddings (skip_embeddings/quick)")
        else:
            emb_matrix = load_word_embeddings(
                word_vocab,
                embedding_dim=int(neural_cfg.get("embedding_dim", 300)),
                embedding_name=neural_cfg.get("embedding_name", "word2vec-google-news-300"),
            )

    all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model_name in models:
        cfg = load_config(model_name)
        cfg["seed"] = seed
        cfg["device"] = device
        if quick:
            cfg = _apply_quick(cfg)

        display = MODEL_DISPLAY[model_name]
        all_metrics.setdefault(display, {})

        for subset in subsets:
            if subset not in subset_idx:
                raise KeyError(f"Missing subset indices for {subset}; run --stage data first")
            train_subset = get_subset_examples(train, subset_idx[subset])
            t0 = time.time()
            result = run_experiment(
                model_name=model_name,
                subset_name=subset,
                train_examples=train_subset,
                test_examples=test,
                cfg=cfg,
                word_vocab=word_vocab,
                pos_vocab=pos_vocab,
                embedding_matrix=emb_matrix,
            )
            metrics = compute_metrics(result["labels"], result["preds"], average=cfg.get("average", "macro"))
            save_run_metrics(model_name, subset, metrics)
            all_metrics[display][subset] = metrics
            logger.info(
                "%s %s → Acc=%.4f P=%.4f R=%.4f F=%.4f (%.1fs)",
                display,
                subset,
                metrics["Accuracy"],
                metrics["Precision"],
                metrics["Recall"],
                metrics["F-score"],
                time.time() - t0,
            )
            # free GPU memory between runs
            del result
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    save_json(all_metrics, ROOT / "results" / "metrics" / "all_metrics.json")
    return all_metrics


def stage_tables(all_metrics: Optional[dict] = None) -> None:
    logger.info("=== Stage: tables & figures ===")
    if all_metrics is None:
        all_metrics = load_all_run_metrics()
        if not all_metrics:
            # Fall back to paper reference so artifacts always exist
            logger.warning("No run metrics found — generating paper-reference tables/figures only")
            export_paper_table_ii()
            plot_paper_reference_figures()
            data = load_processed() if (ROOT / "data" / "processed" / "meta.json").exists() else None
            if data:
                export_table_i({k: len(v) for k, v in data["subsets"].items()})
            return

    meta_path = ROOT / "data" / "processed" / "meta.json"
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        export_table_i(meta.get("subsets", {}))

    export_table_ii(all_metrics)
    export_paper_table_ii()
    plot_figure1(all_metrics)
    plot_figure2(all_metrics)
    plot_paper_reference_figures()
    compare_with_paper(all_metrics)


def stage_report() -> None:
    """Write a short metrics comparison summary under results/metrics/."""
    logger.info("=== Stage: validation report refresh ===")
    all_metrics = load_all_run_metrics()
    if not all_metrics:
        logger.warning("No metrics to report yet")
        return
    df = compare_with_paper(all_metrics)
    summary_path = ROOT / "results" / "metrics" / "summary.txt"
    lines = [
        "Reproduction vs paper (absolute difference summary)",
        "=" * 60,
    ]
    if len(df):
        lines.append(f"Mean |AbsDiff|: {df['AbsDiff'].abs().mean():.4f}")
        lines.append(f"Max  |AbsDiff|: {df['AbsDiff'].abs().max():.4f}")
        lines.append("")
        lines.append(df.to_string(index=False))
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", summary_path)


def main() -> None:
    args = parse_args()
    setup_logging()
    ensure_dirs(
        ROOT / "data" / "raw",
        ROOT / "data" / "processed",
        ROOT / "results" / "figures",
        ROOT / "results" / "tables",
        ROOT / "results" / "metrics",
    )

    # Resolve / validate GPU early so BERT training does not silently fall back to CPU
    from src.utils import get_device

    try:
        resolved = get_device(args.device, require_cuda=(args.device.lower().startswith("cuda")))
        logger.info("Runtime device resolved to: %s", resolved)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    cfg = load_config()
    cfg["seed"] = args.seed
    cfg["device"] = args.device
    set_seed(args.seed)

    models = [m.lower().replace("-", "_") for m in args.models]
    # normalize aliases
    alias = {
        "attcnn": "att_cnn",
        "att_cnn": "att_cnn",
        "attbilstm": "att_bilstm",
        "att_bilstm": "att_bilstm",
        "att_blstm": "att_bilstm",
        "r_bert": "rbert",
        "rbert": "rbert",
        "mtb": "mtb",
    }
    models = [alias.get(m, m) for m in models]

    all_metrics = None
    if args.stage in ("all", "data"):
        stage_data(cfg)

    if args.stage in ("all", "train"):
        all_metrics = stage_train(
            models=models,
            subsets=args.subsets,
            seed=args.seed,
            device=args.device,
            quick=args.quick,
            skip_embeddings=args.skip_embeddings,
        )

    if args.stage in ("all", "tables", "train"):
        stage_tables(all_metrics)

    if args.stage in ("all", "report", "tables"):
        stage_report()

    logger.info("Done. Results under %s", ROOT / "results")


if __name__ == "__main__":
    main()
