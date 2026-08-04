"""Matching the Blanks / BERT Entity Markers (Soares et al., ACL 2019)."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class MTBRelationClassifier(nn.Module):
    """
    BERT entity-marker classifier (Soares et al., ACL 2019).

    Uses concat of entity-start states; supervised fine-tuning only
    (no Wikipedia Matching-the-Blanks pretraining).
    """

    def __init__(
        self,
        config: BertConfig | None = None,
        num_classes: int = 19,
        dropout: float = 0.1,
        pretrained_model: str = "bert-base-uncased",
    ):
        super().__init__()
        if config is None:
            config = BertConfig.from_pretrained(pretrained_model)
        self.config = config
        self.bert = BertModel(config)
        hidden = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    @classmethod
    def from_pretrained_bert(
        cls,
        pretrained_model: str = "bert-base-uncased",
        num_classes: int = 19,
        dropout: float = 0.1,
    ) -> "MTBRelationClassifier":
        try:
            config = BertConfig.from_pretrained(pretrained_model, local_files_only=True)
            bert = BertModel.from_pretrained(pretrained_model, local_files_only=True)
        except Exception:
            config = BertConfig.from_pretrained(pretrained_model, local_files_only=False)
            bert = BertModel.from_pretrained(pretrained_model, local_files_only=False)
        model = cls(config=config, num_classes=num_classes, dropout=dropout)
        model.bert = bert
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_start: torch.Tensor,
        e2_start: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq = outputs.last_hidden_state  # (B, L, H)
        bsz = seq.size(0)
        h_e1 = seq[torch.arange(bsz, device=seq.device), e1_start]
        h_e2 = seq[torch.arange(bsz, device=seq.device), e2_start]
        feat = self.dropout(torch.cat([h_e1, h_e2], dim=-1))
        logits = self.classifier(feat)
        return logits

    def save_pretrained(self, path) -> None:
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        self.config.save_pretrained(path)
