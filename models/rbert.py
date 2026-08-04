"""R-BERT: Enriching BERT with entity information (Wu & He, CIKM 2019)."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class RBERT(nn.Module):
    """R-BERT (Wu & He, CIKM 2019): BERT + entity markers + span averages."""

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
        self.fc_cls = nn.Linear(hidden, hidden)
        self.fc_e1 = nn.Linear(hidden, hidden)
        self.fc_e2 = nn.Linear(hidden, hidden)
        self.classifier = nn.Linear(hidden * 3, num_classes)
        self._init_classifier_weights()

    def _init_classifier_weights(self) -> None:
        for module in (self.fc_cls, self.fc_e1, self.fc_e2, self.classifier):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @classmethod
    def from_pretrained_bert(
        cls,
        pretrained_model: str = "bert-base-uncased",
        num_classes: int = 19,
        dropout: float = 0.1,
    ) -> "RBERT":
        try:
            config = BertConfig.from_pretrained(pretrained_model, local_files_only=True)
            bert = BertModel.from_pretrained(pretrained_model, local_files_only=True)
        except Exception:
            config = BertConfig.from_pretrained(pretrained_model, local_files_only=False)
            bert = BertModel.from_pretrained(pretrained_model, local_files_only=False)
        model = cls(config=config, num_classes=num_classes, dropout=dropout)
        model.bert = bert
        return model

    @staticmethod
    def _entity_average(
        sequence_output: torch.Tensor, e_mask: torch.Tensor
    ) -> torch.Tensor:
        """Average hidden states where e_mask == 1."""
        e_mask = e_mask.unsqueeze(-1).float()
        summed = torch.sum(sequence_output * e_mask, dim=1)
        denom = torch.clamp(e_mask.sum(dim=1), min=1e-9)
        return summed / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_mask: torch.Tensor,
        e2_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq = outputs.last_hidden_state
        cls = seq[:, 0]
        e1 = self._entity_average(seq, e1_mask)
        e2 = self._entity_average(seq, e2_mask)

        cls = self.dropout(torch.tanh(self.fc_cls(cls)))
        e1 = self.dropout(torch.tanh(self.fc_e1(e1)))
        e2 = self.dropout(torch.tanh(self.fc_e2(e2)))
        feat = torch.cat([cls, e1, e2], dim=-1)
        logits = self.classifier(self.dropout(feat))
        return logits

    def save_pretrained(self, path) -> None:
        from pathlib import Path

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        self.config.save_pretrained(path)
