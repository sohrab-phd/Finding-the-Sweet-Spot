"""SemEval-2010 Task 8 download, parsing, and Algorithm 1 subset partitioning."""

from __future__ import annotations

import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from src.utils import (
    LABEL2ID,
    PROJECT_ROOT,
    ensure_dirs,
    load_json,
    save_json,
)

logger = logging.getLogger(__name__)

# Community mirror commonly used by RE reproductions
SEMEVAL_ZIP_URLS = [
    "https://github.com/sahitya0000/Relation-Classification/raw/master/corpus/SemEval2010_task8_all_data.zip",
    "https://raw.githubusercontent.com/sahitya0000/Relation-Classification/master/corpus/SemEval2010_task8_all_data.zip",
]

TRAIN_FILE_CANDIDATES = [
    "SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT",
    "SemEval2010_task8_training/TRAIN_FILE.TXT",
    "TRAIN_FILE.TXT",
]
TEST_FILE_CANDIDATES = [
    "SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT",
    "SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT",
    "TEST_FILE_FULL.TXT",
]


def _find_member(names: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    name_set = set(names)
    for c in candidates:
        if c in name_set:
            return c
    # fuzzy: endswith
    for c in candidates:
        suffix = c.split("/")[-1]
        for n in names:
            if n.endswith(suffix) and "train" in n.lower() and "TRAIN" in c:
                return n
            if n.endswith(suffix) and ("test" in n.lower() or "TEST" in c):
                return n
    for n in names:
        if n.upper().endswith("TRAIN_FILE.TXT"):
            return n
    for n in names:
        if n.upper().endswith("TEST_FILE_FULL.TXT"):
            return n
    return None


def download_semeval(raw_dir: Optional[Path] = None) -> Path:
    """Download and extract SemEval-2010 Task 8 corpus into data/raw."""
    raw_dir = Path(raw_dir or PROJECT_ROOT / "data" / "raw")
    ensure_dirs(raw_dir)
    marker = raw_dir / ".semeval_ready"
    train_out = raw_dir / "TRAIN_FILE.TXT"
    test_out = raw_dir / "TEST_FILE_FULL.TXT"
    if marker.exists() and train_out.exists() and test_out.exists():
        logger.info("SemEval data already present at %s", raw_dir)
        return raw_dir

    zip_path = raw_dir / "SemEval2010_task8_all_data.zip"
    if not zip_path.exists():
        last_err: Optional[Exception] = None
        for url in SEMEVAL_ZIP_URLS:
            try:
                logger.info("Downloading SemEval-2010 Task 8 from %s", url)
                resp = requests.get(url, timeout=120, allow_redirects=True)
                resp.raise_for_status()
                zip_path.write_bytes(resp.content)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                logger.warning("Download failed for %s: %s", url, exc)
        if last_err is not None and not zip_path.exists():
            raise RuntimeError(
                "Could not download SemEval-2010 Task 8. "
                "Place TRAIN_FILE.TXT and TEST_FILE_FULL.TXT under data/raw/ "
                f"manually. Last error: {last_err}"
            )

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        train_member = _find_member(names, TRAIN_FILE_CANDIDATES)
        test_member = _find_member(names, TEST_FILE_CANDIDATES)
        if train_member is None or test_member is None:
            # extract all and search on disk
            zf.extractall(raw_dir)
        else:
            train_out.write_bytes(zf.read(train_member))
            test_out.write_bytes(zf.read(test_member))

    if not train_out.exists() or not test_out.exists():
        # search recursively
        trains = list(raw_dir.rglob("TRAIN_FILE.TXT"))
        tests = list(raw_dir.rglob("TEST_FILE_FULL.TXT"))
        if not trains or not tests:
            raise FileNotFoundError(
                "Could not locate TRAIN_FILE.TXT / TEST_FILE_FULL.TXT after extract"
            )
        train_out.write_bytes(trains[0].read_bytes())
        test_out.write_bytes(tests[0].read_bytes())

    marker.write_text("ok", encoding="utf-8")
    logger.info("SemEval files ready: %s, %s", train_out, test_out)
    return raw_dir


_ENTITY_RE = re.compile(
    r"(.*?)<e1>(.*?)</e1>(.*?)<e2>(.*?)</e2>(.*?)$"
    r"|"
    r"(.*?)<e2>(.*?)</e2>(.*?)<e1>(.*?)</e1>(.*?)$",
    re.DOTALL,
)


def _parse_sentence_with_entities(sentence: str) -> Dict:
    """Parse a SemEval sentence with <e1>/<e2> markers into tokens and spans."""
    # Strip wrapping quotes if present
    s = sentence.strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]

    # Find entity character spans in the tagged string, then build clean text
    e1_start_tag = s.find("<e1>")
    e1_end_tag = s.find("</e1>")
    e2_start_tag = s.find("<e2>")
    e2_end_tag = s.find("</e2>")
    if min(e1_start_tag, e1_end_tag, e2_start_tag, e2_end_tag) < 0:
        raise ValueError(f"Missing entity markers in: {sentence}")

    e1_text = s[e1_start_tag + 4 : e1_end_tag]
    e2_text = s[e2_start_tag + 4 : e2_end_tag]

    # Build clean sentence and track entity token indices via whitespace tokenization
    # Replace tags while preserving entity tokens.
    clean = (
        s.replace("<e1>", " ")
        .replace("</e1>", " ")
        .replace("<e2>", " ")
        .replace("</e2>", " ")
    )
    # Normalize whitespace
    tokens = clean.split()

    # Re-tokenize from tagged form more carefully to recover entity token spans
    tokens, e1_span, e2_span = _tokenize_with_entity_spans(s)

    return {
        "tokens": tokens,
        "e1_span": e1_span,  # inclusive [start, end]
        "e2_span": e2_span,
        "e1_text": e1_text,
        "e2_text": e2_text,
        "sentence_tagged": s,
        "sentence": " ".join(tokens),
    }


def _tokenize_with_entity_spans(tagged: str) -> Tuple[List[str], Tuple[int, int], Tuple[int, int]]:
    """Whitespace-tokenize while recording e1/e2 token index spans."""
    # Walk the string, splitting on whitespace outside/inside tags.
    pieces: List[Tuple[str, Optional[str]]] = []  # (text, entity_tag or None)
    i = 0
    current_entity: Optional[str] = None
    buf: List[str] = []

    def flush_buf() -> None:
        nonlocal buf
        if buf:
            text = "".join(buf)
            for tok in text.split():
                pieces.append((tok, current_entity))
            buf = []

    while i < len(tagged):
        if tagged.startswith("<e1>", i):
            flush_buf()
            current_entity = "e1"
            i += 4
            continue
        if tagged.startswith("</e1>", i):
            flush_buf()
            current_entity = None
            i += 5
            continue
        if tagged.startswith("<e2>", i):
            flush_buf()
            current_entity = "e2"
            i += 4
            continue
        if tagged.startswith("</e2>", i):
            flush_buf()
            current_entity = None
            i += 5
            continue
        buf.append(tagged[i])
        i += 1
    flush_buf()

    tokens = [t for t, _ in pieces]
    e1_idxs = [idx for idx, (_, ent) in enumerate(pieces) if ent == "e1"]
    e2_idxs = [idx for idx, (_, ent) in enumerate(pieces) if ent == "e2"]
    if not e1_idxs or not e2_idxs:
        raise ValueError(f"Failed to locate entity tokens in: {tagged}")
    e1_span = (e1_idxs[0], e1_idxs[-1])
    e2_span = (e2_idxs[0], e2_idxs[-1])
    return tokens, e1_span, e2_span


def parse_semeval_file(path: Path) -> List[Dict]:
    """Parse TRAIN_FILE.TXT / TEST_FILE_FULL.TXT format."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    # Records: id\tsentence\nrelation\nComment:\n\n
    blocks = re.split(r"\n\s*\n", text.strip())
    examples: List[Dict] = []
    for block in blocks:
        lines = [ln for ln in block.strip().splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            continue
        # First line: ID\t"sentence"
        m = re.match(r'^(\d+)\s+"?(.*)"?\s*$', lines[0])
        if not m:
            # sometimes: 1\t"....."
            parts = lines[0].split("\t", 1)
            if len(parts) != 2:
                continue
            ex_id, sent = parts[0].strip(), parts[1].strip()
        else:
            ex_id, sent = m.group(1), m.group(2)
        sent = sent.strip()
        if sent.startswith('"') and sent.endswith('"'):
            sent = sent[1:-1]
        relation = lines[1].strip()
        if relation not in LABEL2ID:
            # Normalize occasional spacing
            relation = relation.replace(" ", "")
            # try restore parentheses form
            if relation not in LABEL2ID:
                logger.warning("Unknown relation '%s' for id %s — skipping", lines[1], ex_id)
                continue
        parsed = _parse_sentence_with_entities(sent)
        examples.append(
            {
                "id": int(ex_id),
                "relation": relation,
                "label": LABEL2ID[relation],
                **parsed,
            }
        )
    return examples


def prepare_dataset(
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    seed: int = 42,
    subset_fractions: Optional[Dict[str, float]] = None,
    subset_sizes: Optional[Dict[str, int]] = None,
) -> Dict:
    """Download, parse, partition (Algorithm 1), and save processed data."""
    raw_dir = Path(raw_dir or PROJECT_ROOT / "data" / "raw")
    processed_dir = Path(processed_dir or PROJECT_ROOT / "data" / "processed")
    ensure_dirs(processed_dir, processed_dir / "subsets")

    download_semeval(raw_dir)
    train = parse_semeval_file(raw_dir / "TRAIN_FILE.TXT")
    test = parse_semeval_file(raw_dir / "TEST_FILE_FULL.TXT")

    logger.info("Parsed train=%d test=%d", len(train), len(test))
    if len(train) != 8000:
        logger.warning("Expected 8000 train examples, got %d", len(train))
    if len(test) != 2717:
        logger.warning("Expected 2717 test examples, got %d", len(test))

    # Cache POS tags once (used by Att-CNN / Att-BiLSTM)
    from src.preprocessing import attach_pos_tags

    logger.info("Caching POS tags for train/test (one-time)…")
    attach_pos_tags(train)
    attach_pos_tags(test)

    save_json(train, processed_dir / "train.json")
    save_json(test, processed_dir / "test.json")

    subsets = create_subsets(
        n_train=len(train),
        seed=seed,
        subset_fractions=subset_fractions,
        subset_sizes=subset_sizes,
        out_dir=processed_dir / "subsets",
    )
    meta = {
        "n_train": len(train),
        "n_test": len(test),
        "subsets": {k: len(v) for k, v in subsets.items()},
    }
    save_json(meta, processed_dir / "meta.json")
    return {"train": train, "test": test, "subsets": subsets, "meta": meta}


def create_subsets(
    n_train: int = 8000,
    seed: int = 42,
    subset_fractions: Optional[Dict[str, float]] = None,
    subset_sizes: Optional[Dict[str, int]] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, List[int]]:
    """
    Algorithm 1 — Dataset Partitioning for Incremental Evaluation.

    For each fraction X in [1/128, ..., 1], draw a uniformly random sample of
    size X * |SE| from the training set, with a fixed random seed.
    """
    if subset_fractions is None:
        subset_fractions = {
            "SE.1": 1 / 128,
            "SE.2": 1 / 64,
            "SE.3": 1 / 32,
            "SE.4": 1 / 16,
            "SE.5": 1 / 8,
            "SE.6": 1 / 4,
            "SE.7": 1 / 2,
            "SE.8": 1.0,
        }
    # Prefer explicit Table I sizes when provided
    if subset_sizes is None:
        subset_sizes = {
            name: int(round(frac * n_train)) for name, frac in subset_fractions.items()
        }
        # Force Table I published counts when n_train == 8000
        if n_train == 8000:
            subset_sizes = {
                "SE.1": 62,
                "SE.2": 125,
                "SE.3": 250,
                "SE.4": 500,
                "SE.5": 1000,
                "SE.6": 2000,
                "SE.7": 4000,
                "SE.8": 8000,
            }

    rng = np.random.RandomState(seed)
    all_indices = np.arange(n_train)
    subsets: Dict[str, List[int]] = {}
    out_dir = Path(out_dir or PROJECT_ROOT / "data" / "processed" / "subsets")
    ensure_dirs(out_dir)

    for name in sorted(subset_fractions.keys(), key=lambda x: subset_fractions[x]):
        size = min(int(subset_sizes[name]), n_train)
        # Independent uniform sample per Algorithm 1
        chosen = rng.choice(all_indices, size=size, replace=False)
        chosen_list = sorted(int(i) for i in chosen)
        subsets[name] = chosen_list
        (out_dir / f"{name}.json").write_text(
            json.dumps(chosen_list), encoding="utf-8"
        )
        logger.info("Created %s with %d instances", name, size)

    return subsets


def load_processed(processed_dir: Optional[Path] = None) -> Dict:
    processed_dir = Path(processed_dir or PROJECT_ROOT / "data" / "processed")
    train = load_json(processed_dir / "train.json")
    test = load_json(processed_dir / "test.json")
    subsets = {}
    for path in sorted((processed_dir / "subsets").glob("SE.*.json")):
        subsets[path.stem] = load_json(path)
    return {"train": train, "test": test, "subsets": subsets}


def get_subset_examples(train: List[Dict], subset_indices: List[int]) -> List[Dict]:
    return [train[i] for i in subset_indices]
