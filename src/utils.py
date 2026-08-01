"""Shared utilities: seeding, config loading, paths, logging."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    return PROJECT_ROOT


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("reproduce")


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(model: Optional[str] = None) -> Dict[str, Any]:
    """Load default config, optionally merged with a model-specific config."""
    cfg = load_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    if model:
        model_path = PROJECT_ROOT / "configs" / f"{model}.yaml"
        if not model_path.exists():
            # allow aliases
            aliases = {
                "att-cnn": "att_cnn",
                "att_cnn": "att_cnn",
                "att-bilstm": "att_bilstm",
                "att_bilstm": "att_bilstm",
                "att-blstm": "att_bilstm",
                "r-bert": "rbert",
                "rbert": "rbert",
                "mtb": "mtb",
            }
            key = aliases.get(model.lower().replace("-", "_"), model.lower())
            model_path = PROJECT_ROOT / "configs" / f"{key}.yaml"
        model_cfg = load_yaml(model_path)
        cfg = {**cfg, **model_cfg}
    return cfg


def set_seed(seed: int = 42) -> None:
    """Make experiments as deterministic as possible."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    worker_seed = torch_initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_initial_seed() -> int:
    try:
        import torch

        return torch.initial_seed()
    except Exception:
        return 42


def get_device(preference: str = "auto", require_cuda: bool = False):
    """
    Resolve the torch device.

    preference:
      - "auto"  → CUDA if available, else CPU
      - "cuda"  → CUDA (raises if unavailable when require_cuda, else warns + CPU)
      - "cpu"   → force CPU
      - "cuda:N"→ specific GPU index
    """
    import logging

    import torch

    logger = logging.getLogger(__name__)
    pref = (preference or "auto").lower().strip()

    if pref == "cpu":
        logger.info("Using device: cpu (forced)")
        return torch.device("cpu")

    cuda_ok = torch.cuda.is_available()
    if pref.startswith("cuda"):
        if not cuda_ok:
            msg = (
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch, e.g.:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )
            if require_cuda or pref == "cuda":
                # Strict for explicit --device cuda
                raise RuntimeError(msg)
            logger.warning("%s Falling back to CPU.", msg)
            return torch.device("cpu")
        device = torch.device(pref if ":" in pref else "cuda")
        _log_cuda_device(device, logger)
        return device

    # auto
    if cuda_ok:
        device = torch.device("cuda")
        _log_cuda_device(device, logger)
        return device
    logger.warning(
        "CUDA not available (CPU-only PyTorch or no driver). Using CPU. "
        "For GPU: pip install torch --index-url https://download.pytorch.org/whl/cu124"
    )
    return torch.device("cpu")


def _log_cuda_device(device, logger) -> None:
    import torch

    idx = device.index if device.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
    logger.info(
        "Using device: %s | GPU: %s | Memory: %.1f GiB | CUDA: %s | torch: %s",
        device,
        name,
        mem_gb,
        torch.version.cuda,
        torch.__version__,
    )


def batch_to_device(batch: Dict[str, Any], device, non_blocking: bool = True) -> Dict[str, Any]:
    """Move all tensor values in a batch dict to `device`."""
    import torch

    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=non_blocking and device.type == "cuda")
        else:
            out[k] = v
    return out


def ensure_dirs(*paths: Union[str, Path]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# SemEval-2010 Task 8 relation labels (19-way)
RELATION_LABELS = [
    "Cause-Effect(e1,e2)",
    "Cause-Effect(e2,e1)",
    "Component-Whole(e1,e2)",
    "Component-Whole(e2,e1)",
    "Content-Container(e1,e2)",
    "Content-Container(e2,e1)",
    "Entity-Destination(e1,e2)",
    "Entity-Destination(e2,e1)",
    "Entity-Origin(e1,e2)",
    "Entity-Origin(e2,e1)",
    "Instrument-Agency(e1,e2)",
    "Instrument-Agency(e2,e1)",
    "Member-Collection(e1,e2)",
    "Member-Collection(e2,e1)",
    "Message-Topic(e1,e2)",
    "Message-Topic(e2,e1)",
    "Product-Producer(e1,e2)",
    "Product-Producer(e2,e1)",
    "Other",
]

LABEL2ID = {label: i for i, label in enumerate(RELATION_LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
