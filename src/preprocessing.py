"""Preprocessing utilities for Att-CNN / Att-BiLSTM and BERT-family models."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.utils import PROJECT_ROOT, ensure_dirs, save_json

logger = logging.getLogger(__name__)

# Coarse POS category mapping inspired by Att-CNN (15 tags)
COARSE_POS = {
    "NN": "NOUN",
    "NNS": "NOUN",
    "NNP": "NOUN",
    "NNPS": "NOUN",
    "VB": "VERB",
    "VBD": "VERB",
    "VBG": "VERB",
    "VBN": "VERB",
    "VBP": "VERB",
    "VBZ": "VERB",
    "JJ": "ADJ",
    "JJR": "ADJ",
    "JJS": "ADJ",
    "RB": "ADV",
    "RBR": "ADV",
    "RBS": "ADV",
    "PRP": "PRON",
    "PRP$": "PRON",
    "WP": "PRON",
    "WP$": "PRON",
    "DT": "DET",
    "IN": "ADP",
    "TO": "ADP",
    "CC": "CONJ",
    "CD": "NUM",
    "MD": "VERB",
    "EX": "DET",
    "FW": "X",
    "LS": "X",
    "PDT": "DET",
    "POS": "PRT",
    "RP": "PRT",
    "SYM": "X",
    "UH": "X",
    "WDT": "DET",
    "WRB": "ADV",
    ",": ".",
    ".": ".",
    ":": ".",
    "``": ".",
    "''": ".",
    "-LRB-": ".",
    "-RRB-": ".",
}

COARSE_POS_LIST = [
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PRON",
    "DET",
    "ADP",
    "CONJ",
    "NUM",
    "PRT",
    "X",
    ".",
    "OTHER",
    "PAD",
    "UNK",
]


def ensure_nltk() -> None:
    import nltk

    for pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def pos_tag_tokens(tokens: Sequence[str]) -> List[str]:
    ensure_nltk()
    from nltk import pos_tag

    tagged = pos_tag(list(tokens))
    coarse = []
    for _, tag in tagged:
        coarse.append(COARSE_POS.get(tag, "OTHER"))
    return coarse


def relative_positions(
    seq_len: int, e1_span: Tuple[int, int], e2_span: Tuple[int, int], max_pos: int = 100
) -> Tuple[List[int], List[int]]:
    """Relative distance from each token to e1 / e2 (clamped), shifted to >=0 for embedding."""
    e1_center = (e1_span[0] + e1_span[1]) // 2
    e2_center = (e2_span[0] + e2_span[1]) // 2
    pos1, pos2 = [], []
    for i in range(seq_len):
        d1 = max(-max_pos, min(max_pos, i - e1_center))
        d2 = max(-max_pos, min(max_pos, i - e2_center))
        # shift to [0, 2*max_pos]
        pos1.append(d1 + max_pos)
        pos2.append(d2 + max_pos)
    return pos1, pos2


class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.token2id: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.id2token: Dict[int, str] = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts: Sequence[Sequence[str]]) -> None:
        counter: Counter = Counter()
        for toks in texts:
            counter.update(toks)
        for tok, freq in counter.most_common():
            if freq < self.min_freq:
                continue
            if tok not in self.token2id:
                idx = len(self.token2id)
                self.token2id[tok] = idx
                self.id2token[idx] = tok

    def encode(self, tokens: Sequence[str], max_len: Optional[int] = None) -> List[int]:
        ids = [self.token2id.get(t, 1) for t in tokens]
        if max_len is not None:
            ids = ids[:max_len]
            ids = ids + [0] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.token2id)

    def save(self, path: Path) -> None:
        save_json({"token2id": self.token2id}, path)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        import json

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = cls()
        vocab.token2id = {k: int(v) for k, v in data["token2id"].items()}
        vocab.id2token = {int(v): k for k, v in vocab.token2id.items()}
        return vocab


def build_vocabularies(train_examples: List[Dict], out_dir: Optional[Path] = None) -> Dict:
    out_dir = Path(out_dir or PROJECT_ROOT / "data" / "processed" / "vocab")
    ensure_dirs(out_dir)

    word_vocab = Vocabulary()
    word_vocab.build([ex["tokens"] for ex in train_examples])
    word_vocab.save(out_dir / "word_vocab.json")

    pos_vocab = Vocabulary()
    pos_vocab.token2id = {p: i for i, p in enumerate(COARSE_POS_LIST)}
    pos_vocab.id2token = {i: p for p, i in pos_vocab.token2id.items()}
    pos_vocab.save(out_dir / "pos_vocab.json")

    logger.info("Word vocab size=%d", len(word_vocab))
    return {"word": word_vocab, "pos": pos_vocab}


def load_word_embeddings(
    vocab: Vocabulary,
    embedding_dim: int = 300,
    embedding_name: str = "word2vec-google-news-300",
    project_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Load pretrained embeddings aligned to `vocab`.
    Falls back to random init if download fails.
    """
    target_dim = project_dim or embedding_dim
    matrix = np.random.uniform(-0.05, 0.05, size=(len(vocab), target_dim)).astype(np.float32)
    matrix[0] = 0.0  # PAD

    vectors = None
    try:
        import gensim.downloader as api

        logger.info("Loading embeddings via gensim: %s (may take a while)", embedding_name)
        vectors = api.load(embedding_name)
    except Exception as exc:
        logger.warning("gensim download failed (%s); trying local fallbacks", exc)

    if vectors is None:
        # Try common local paths
        candidates = [
            PROJECT_ROOT / "data" / "raw" / "GoogleNews-vectors-negative300.bin",
            PROJECT_ROOT / "data" / "raw" / "glove.6B.300d.txt",
            PROJECT_ROOT / "data" / "raw" / "glove.6B.100d.txt",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                from gensim.models import KeyedVectors

                if path.suffix == ".bin":
                    vectors = KeyedVectors.load_word2vec_format(str(path), binary=True)
                else:
                    vectors = KeyedVectors.load_word2vec_format(str(path), binary=False, no_header=True)
                logger.info("Loaded embeddings from %s", path)
                break
            except Exception as exc:
                logger.warning("Failed loading %s: %s", path, exc)

    if vectors is None:
        logger.warning("No pretrained embeddings available — using random initialization")
        return matrix

    src_dim = int(vectors.vector_size)
    hit = 0
    for token, idx in vocab.token2id.items():
        if token in ("<PAD>", "<UNK>"):
            continue
        key = token
        if key not in vectors:
            # try lowercase
            key = token.lower()
        if key in vectors:
            vec = np.asarray(vectors[key], dtype=np.float32)
            if project_dim and src_dim != target_dim:
                # truncate or pad
                if src_dim >= target_dim:
                    vec = vec[:target_dim]
                else:
                    padded = np.zeros(target_dim, dtype=np.float32)
                    padded[:src_dim] = vec
                    vec = padded
            elif src_dim != target_dim:
                if src_dim >= target_dim:
                    vec = vec[:target_dim]
                else:
                    padded = np.zeros(target_dim, dtype=np.float32)
                    padded[:src_dim] = vec
                    vec = padded
            matrix[idx] = vec
            hit += 1
    logger.info("Embedding coverage: %d / %d tokens", hit, len(vocab) - 2)
    return matrix


def add_entity_markers(example: Dict, style: str = "rbert") -> str:
    """
    Insert entity marker tokens into the whitespace-tokenized sentence.
    Returns a string suitable for BERT tokenization.
    """
    tokens = list(example["tokens"])
    e1_s, e1_e = example["e1_span"]
    e2_s, e2_e = example["e2_span"]

    # Insert from the right so indices stay valid
    markers = [
        (e1_s, "[E1]"),
        (e1_e + 1, "[/E1]"),
        (e2_s, "[E2]"),
        (e2_e + 1, "[/E2]"),
    ]
    # Sort descending by position
    for pos, mk in sorted(markers, key=lambda x: x[0], reverse=True):
        tokens.insert(pos, mk)
    return " ".join(tokens)


def attach_pos_tags(examples: List[Dict]) -> List[Dict]:
    """POS-tag all examples once and store coarse tags on each example."""
    ensure_nltk()
    for i, ex in enumerate(examples):
        if "pos_tags" not in ex:
            ex["pos_tags"] = pos_tag_tokens(ex["tokens"])
        if (i + 1) % 1000 == 0:
            logger.info("POS-tagged %d / %d examples", i + 1, len(examples))
    return examples


def preprocess_neural_example(
    example: Dict,
    word_vocab: Vocabulary,
    pos_vocab: Vocabulary,
    max_len: int = 100,
    max_pos: int = 100,
) -> Dict:
    tokens = example["tokens"]
    pos_tags = example.get("pos_tags") or pos_tag_tokens(tokens)
    pos1, pos2 = relative_positions(len(tokens), tuple(example["e1_span"]), tuple(example["e2_span"]), max_pos)
    length = min(len(tokens), max_len)
    word_ids = word_vocab.encode(tokens, max_len)
    pos_ids = [pos_vocab.token2id.get(p, pos_vocab.token2id["UNK"]) for p in pos_tags]
    pos_ids = pos_ids[:max_len] + [pos_vocab.token2id["PAD"]] * (max_len - len(pos_ids[:max_len]))
    pos1 = pos1[:max_len] + [0] * (max_len - len(pos1[:max_len]))
    pos2 = pos2[:max_len] + [0] * (max_len - len(pos2[:max_len]))
    return {
        "word_ids": word_ids,
        "pos_ids": pos_ids,
        "pos1": pos1,
        "pos2": pos2,
        "length": length,
        "label": example["label"],
        "e1_span": example["e1_span"],
        "e2_span": example["e2_span"],
    }
