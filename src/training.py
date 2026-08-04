"""PyTorch datasets and training loops for all four RE models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing import (
    Vocabulary,
    add_entity_markers,
    preprocess_neural_example,
)
from src.utils import (
    PROJECT_ROOT,
    batch_to_device,
    ensure_dirs,
    get_device,
    seed_worker,
    set_seed,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class NeuralREDataset(Dataset):
    """Dataset for Att-CNN / Att-BiLSTM."""

    def __init__(
        self,
        examples: List[Dict],
        word_vocab: Vocabulary,
        pos_vocab: Vocabulary,
        max_len: int = 100,
        max_pos: int = 100,
        use_pos: bool = True,
    ):
        self.examples = examples
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.max_len = max_len
        self.max_pos = max_pos
        self.use_pos = use_pos
        self.cache = [
            preprocess_neural_example(ex, word_vocab, pos_vocab, max_len, max_pos)
            for ex in examples
        ]

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.cache[idx]
        e1 = item["e1_span"]
        e2 = item["e2_span"]
        # clamp spans to max_len-1
        e1 = (min(e1[0], self.max_len - 1), min(e1[1], self.max_len - 1))
        e2 = (min(e2[0], self.max_len - 1), min(e2[1], self.max_len - 1))
        return {
            "word_ids": torch.tensor(item["word_ids"], dtype=torch.long),
            "pos_ids": torch.tensor(item["pos_ids"], dtype=torch.long),
            "pos1": torch.tensor(item["pos1"], dtype=torch.long),
            "pos2": torch.tensor(item["pos2"], dtype=torch.long),
            "length": torch.tensor(item["length"], dtype=torch.long),
            "e1_span": torch.tensor(e1, dtype=torch.long),
            "e2_span": torch.tensor(e2, dtype=torch.long),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


class BertREDataset(Dataset):
    """Dataset for R-BERT / MTB with entity markers."""

    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_len: int = 128,
        mode: str = "rbert",
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        self.encoded = [self._encode(ex) for ex in examples]

    def _encode(self, example: Dict) -> Dict:
        marked = add_entity_markers(example)
        # Tokenize; keep track of marker positions
        encoding = self.tokenizer(
            marked,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=False,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        e1_start_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        e1_end_id = self.tokenizer.convert_tokens_to_ids("[/E1]")
        e2_start_id = self.tokenizer.convert_tokens_to_ids("[E2]")
        e2_end_id = self.tokenizer.convert_tokens_to_ids("[/E2]")

        ids_list = input_ids.tolist()
        try:
            e1_s = ids_list.index(e1_start_id)
            e1_e = ids_list.index(e1_end_id)
            e2_s = ids_list.index(e2_start_id)
            e2_e = ids_list.index(e2_end_id)
        except ValueError:
            # Truncation removed a marker — fall back to CLS-adjacent positions
            e1_s, e1_e, e2_s, e2_e = 1, 1, 2, 2

        e1_mask = torch.zeros(self.max_len, dtype=torch.long)
        e2_mask = torch.zeros(self.max_len, dtype=torch.long)
        # entity tokens between markers (exclusive of markers), inclusive inner tokens
        if e1_e > e1_s + 1:
            e1_mask[e1_s + 1 : e1_e] = 1
        else:
            e1_mask[e1_s] = 1
        if e2_e > e2_s + 1:
            e2_mask[e2_s + 1 : e2_e] = 1
        else:
            e2_mask[e2_s] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "e1_mask": e1_mask,
            "e2_mask": e2_mask,
            "e1_start": torch.tensor(e1_s, dtype=torch.long),
            "e2_start": torch.tensor(e2_s, dtype=torch.long),
            "label": torch.tensor(example["label"], dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.encoded[idx]


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    device=None,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    if pin_memory is None:
        # Pin host memory when training on CUDA for faster H2D copies
        if device is not None:
            pin_memory = getattr(device, "type", str(device)) == "cuda"
        else:
            pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=g,
        pin_memory=bool(pin_memory),
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _optimizer(model: torch.nn.Module, cfg: Dict[str, Any]):
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("learning_rate", 1e-3))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if name == "adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=1e-5)
    weight_decay = float(cfg.get("weight_decay", 0.01))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def predict_neural(model, loader, device, model_type: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        labels.append(batch["label"].numpy())
        batch = batch_to_device(batch, device)
        if model_type == "att_cnn":
            logits = model(
                batch["word_ids"],
                batch["pos_ids"],
                batch["pos1"],
                batch["pos2"],
                batch["length"],
                batch["e1_span"],
                batch["e2_span"],
            )
        else:
            logits = model(batch["word_ids"], batch["length"])
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels)


@torch.no_grad()
def predict_bert(model, loader, device, model_type: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for batch in loader:
        labels.append(batch["label"].numpy())
        batch = batch_to_device(batch, device)
        if model_type == "rbert":
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                e1_mask=batch["e1_mask"],
                e2_mask=batch["e2_mask"],
            )
        else:
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                e1_start=batch["e1_start"],
                e2_start=batch["e2_start"],
            )
        preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(labels)


def train_att_cnn(
    train_examples: List[Dict],
    test_examples: List[Dict],
    cfg: Dict[str, Any],
    word_vocab: Vocabulary,
    pos_vocab: Vocabulary,
    embedding_matrix: Optional[np.ndarray] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from models.att_cnn import AttCNN

    device = get_device(cfg.get("device", "auto"), require_cuda=(str(cfg.get("device", "auto")).lower() == "cuda"))
    set_seed(int(cfg.get("seed", 42)))
    logger.info("Att-CNN training on %s", device)

    train_ds = NeuralREDataset(
        train_examples, word_vocab, pos_vocab, cfg.get("max_seq_len", 100), cfg.get("max_position", 100)
    )
    test_ds = NeuralREDataset(
        test_examples, word_vocab, pos_vocab, cfg.get("max_seq_len", 100), cfg.get("max_position", 100)
    )
    train_loader = make_loader(
        train_ds, cfg.get("batch_size", 50), True, cfg.get("seed", 42), device=device
    )
    test_loader = make_loader(
        test_ds, cfg.get("batch_size", 50), False, cfg.get("seed", 42), device=device
    )

    pretrained = None
    if embedding_matrix is not None:
        pretrained = torch.tensor(embedding_matrix, dtype=torch.float32)

    model = AttCNN(
        vocab_size=len(word_vocab),
        num_classes=int(cfg.get("num_relations", 19)),
        embedding_dim=int(cfg.get("embedding_dim", 300)),
        pos_vocab_size=len(pos_vocab),
        pos_embedding_dim=int(cfg.get("pos_embedding_dim", 10)),
        position_embedding_dim=int(cfg.get("position_embedding_dim", 5)),
        max_position=int(cfg.get("max_position", 100)),
        num_filters=int(cfg.get("num_filters", 100)),
        filter_size=int(cfg.get("filter_size", 3)),
        hidden_dim=int(cfg.get("hidden_dim", 100)),
        dropout=float(cfg.get("dropout", 0.5)),
        pretrained_embeddings=pretrained,
    ).to(device)

    opt = _optimizer(model, cfg)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = int(cfg.get("epochs", 20))
    clip = float(cfg.get("grad_clip", 5.0))

    best_state = None
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            logits = model(
                batch["word_ids"],
                batch["pos_ids"],
                batch["pos1"],
                batch["pos2"],
                batch["length"],
                batch["e1_span"],
                batch["e2_span"],
            )
            loss = criterion(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total += loss.item()
        avg = total / max(len(train_loader), 1)
        logger.info("Att-CNN epoch %d/%d loss=%.4f [%s]", epoch, epochs, avg, device)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    preds, labels = predict_neural(model, test_loader, device, "att_cnn")

    if checkpoint_dir:
        ensure_dirs(checkpoint_dir)
        torch.save(model.state_dict(), Path(checkpoint_dir) / "model.pt")

    return {"preds": preds, "labels": labels, "model": model}


def train_att_bilstm(
    train_examples: List[Dict],
    test_examples: List[Dict],
    cfg: Dict[str, Any],
    word_vocab: Vocabulary,
    pos_vocab: Vocabulary,
    embedding_matrix: Optional[np.ndarray] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from models.att_bilstm import AttBiLSTM

    device = get_device(cfg.get("device", "auto"), require_cuda=(str(cfg.get("device", "auto")).lower() == "cuda"))
    set_seed(int(cfg.get("seed", 42)))
    logger.info("Att-BiLSTM training on %s", device)

    emb_dim = int(cfg.get("embedding_project_dim", cfg.get("embedding_dim", 100)))
    train_ds = NeuralREDataset(
        train_examples, word_vocab, pos_vocab, cfg.get("max_seq_len", 100)
    )
    test_ds = NeuralREDataset(
        test_examples, word_vocab, pos_vocab, cfg.get("max_seq_len", 100)
    )
    train_loader = make_loader(
        train_ds, cfg.get("batch_size", 10), True, cfg.get("seed", 42), device=device
    )
    test_loader = make_loader(
        test_ds, cfg.get("batch_size", 10), False, cfg.get("seed", 42), device=device
    )

    pretrained = None
    if embedding_matrix is not None:
        # ensure dim matches
        if embedding_matrix.shape[1] != emb_dim:
            if embedding_matrix.shape[1] > emb_dim:
                embedding_matrix = embedding_matrix[:, :emb_dim]
            else:
                pad = np.zeros((embedding_matrix.shape[0], emb_dim), dtype=np.float32)
                pad[:, : embedding_matrix.shape[1]] = embedding_matrix
                embedding_matrix = pad
        pretrained = torch.tensor(embedding_matrix, dtype=torch.float32)

    model = AttBiLSTM(
        vocab_size=len(word_vocab),
        num_classes=int(cfg.get("num_relations", 19)),
        embedding_dim=emb_dim,
        hidden_dim=int(cfg.get("hidden_dim", 100)),
        dropout=float(cfg.get("dropout", 0.5)),
        pretrained_embeddings=pretrained,
    ).to(device)

    opt = _optimizer(model, cfg)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = int(cfg.get("epochs", 30))
    clip = float(cfg.get("grad_clip", 5.0))

    best_state = None
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch["word_ids"], batch["length"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total += loss.item()
        avg = total / max(len(train_loader), 1)
        logger.info("Att-BiLSTM epoch %d/%d loss=%.4f [%s]", epoch, epochs, avg, device)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    preds, labels = predict_neural(model, test_loader, device, "att_bilstm")

    if checkpoint_dir:
        ensure_dirs(checkpoint_dir)
        torch.save(model.state_dict(), Path(checkpoint_dir) / "model.pt")

    return {"preds": preds, "labels": labels, "model": model}


def _load_pretrained_with_fallback(loader, name: str, **kwargs):
    """
    Load a HuggingFace artifact preferring the local cache.

    Newer transformers versions probe the Hub (e.g. list_repo_templates) even when
    files are cached, which fails on flaky networks. Try local_files_only first.
    """
    try:
        return loader(name, local_files_only=True, **kwargs)
    except Exception as local_exc:
        logger.warning(
            "Local load of %s failed (%s); retrying with Hub download…",
            name,
            local_exc,
        )
        return loader(name, local_files_only=False, **kwargs)


def _build_bert_tokenizer(cfg: Dict[str, Any]):
    from transformers import BertTokenizer

    name = cfg.get("pretrained_model", "bert-base-uncased")
    tokenizer = _load_pretrained_with_fallback(BertTokenizer.from_pretrained, name)
    special = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    tokenizer.add_special_tokens({"additional_special_tokens": special})
    return tokenizer


def train_rbert(
    train_examples: List[Dict],
    test_examples: List[Dict],
    cfg: Dict[str, Any],
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from transformers import get_linear_schedule_with_warmup

    from models.rbert import RBERT

    device = get_device(cfg.get("device", "auto"), require_cuda=(str(cfg.get("device", "auto")).lower() == "cuda"))
    set_seed(int(cfg.get("seed", 42)))
    logger.info("R-BERT training on %s", device)
    tokenizer = _build_bert_tokenizer(cfg)

    train_ds = BertREDataset(train_examples, tokenizer, cfg.get("max_seq_len", 128), "rbert")
    test_ds = BertREDataset(test_examples, tokenizer, cfg.get("max_seq_len", 128), "rbert")
    train_loader = make_loader(
        train_ds, cfg.get("batch_size", 16), True, cfg.get("seed", 42), device=device
    )
    test_loader = make_loader(
        test_ds, cfg.get("batch_size", 16), False, cfg.get("seed", 42), device=device
    )

    pretrained = cfg.get("pretrained_model", "bert-base-uncased")
    model = RBERT.from_pretrained_bert(
        pretrained_model=pretrained,
        num_classes=int(cfg.get("num_relations", 19)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    model.bert.resize_token_embeddings(len(tokenizer))
    model.to(device)

    opt = _optimizer(model, cfg)
    epochs = int(cfg.get("epochs", 5))
    total_steps = max(len(train_loader) * epochs, 1)
    warmup = int(total_steps * float(cfg.get("warmup_ratio", 0.1)))
    scheduler = get_linear_schedule_with_warmup(opt, warmup, total_steps)
    criterion = torch.nn.CrossEntropyLoss()
    clip = float(cfg.get("grad_clip", 1.0))

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                e1_mask=batch["e1_mask"],
                e2_mask=batch["e2_mask"],
            )
            loss = criterion(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            scheduler.step()
            total += loss.item()
        logger.info(
            "R-BERT epoch %d/%d loss=%.4f [%s]",
            epoch,
            epochs,
            total / max(len(train_loader), 1),
            device,
        )

    preds, labels = predict_bert(model, test_loader, device, "rbert")
    if checkpoint_dir:
        ensure_dirs(checkpoint_dir)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
    return {"preds": preds, "labels": labels, "model": model}


def train_mtb(
    train_examples: List[Dict],
    test_examples: List[Dict],
    cfg: Dict[str, Any],
    checkpoint_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    from transformers import get_linear_schedule_with_warmup

    from models.mtb import MTBRelationClassifier

    device = get_device(cfg.get("device", "auto"), require_cuda=(str(cfg.get("device", "auto")).lower() == "cuda"))
    set_seed(int(cfg.get("seed", 42)))
    logger.info("MTB training on %s", device)
    tokenizer = _build_bert_tokenizer(cfg)

    train_ds = BertREDataset(train_examples, tokenizer, cfg.get("max_seq_len", 128), "mtb")
    test_ds = BertREDataset(test_examples, tokenizer, cfg.get("max_seq_len", 128), "mtb")
    train_loader = make_loader(
        train_ds, cfg.get("batch_size", 16), True, cfg.get("seed", 42), device=device
    )
    test_loader = make_loader(
        test_ds, cfg.get("batch_size", 16), False, cfg.get("seed", 42), device=device
    )

    pretrained = cfg.get("pretrained_model", "bert-base-uncased")
    model = MTBRelationClassifier.from_pretrained_bert(
        pretrained_model=pretrained,
        num_classes=int(cfg.get("num_relations", 19)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    model.bert.resize_token_embeddings(len(tokenizer))
    model.to(device)

    opt = _optimizer(model, cfg)
    epochs = int(cfg.get("epochs", 5))
    total_steps = max(len(train_loader) * epochs, 1)
    warmup = int(total_steps * float(cfg.get("warmup_ratio", 0.1)))
    scheduler = get_linear_schedule_with_warmup(opt, warmup, total_steps)
    criterion = torch.nn.CrossEntropyLoss()
    clip = float(cfg.get("grad_clip", 1.0))

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                e1_start=batch["e1_start"],
                e2_start=batch["e2_start"],
            )
            loss = criterion(logits, batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            scheduler.step()
            total += loss.item()
        logger.info(
            "MTB epoch %d/%d loss=%.4f [%s]",
            epoch,
            epochs,
            total / max(len(train_loader), 1),
            device,
        )

    preds, labels = predict_bert(model, test_loader, device, "mtb")
    if checkpoint_dir:
        ensure_dirs(checkpoint_dir)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
    return {"preds": preds, "labels": labels, "model": model}


def run_experiment(
    model_name: str,
    subset_name: str,
    train_examples: List[Dict],
    test_examples: List[Dict],
    cfg: Dict[str, Any],
    word_vocab: Optional[Vocabulary] = None,
    pos_vocab: Optional[Vocabulary] = None,
    embedding_matrix: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Train one model on one subset and return predictions."""
    ckpt = PROJECT_ROOT / "results" / "checkpoints" / model_name / subset_name
    logger.info("=== Experiment %s @ %s (n_train=%d) ===", model_name, subset_name, len(train_examples))

    if model_name == "att_cnn":
        assert word_vocab is not None and pos_vocab is not None
        return train_att_cnn(
            train_examples, test_examples, cfg, word_vocab, pos_vocab, embedding_matrix, ckpt
        )
    if model_name == "att_bilstm":
        assert word_vocab is not None and pos_vocab is not None
        return train_att_bilstm(
            train_examples, test_examples, cfg, word_vocab, pos_vocab, embedding_matrix, ckpt
        )
    if model_name == "rbert":
        return train_rbert(train_examples, test_examples, cfg, ckpt)
    if model_name == "mtb":
        return train_mtb(train_examples, test_examples, cfg, ckpt)
    raise ValueError(f"Unknown model: {model_name}")
