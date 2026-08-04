"""Attention-Based BiLSTM for relation classification (Zhou et al., ACL 2016)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttBiLSTM(nn.Module):
    """Att-BiLSTM (Zhou et al., ACL 2016): BiLSTM + attention over hidden states."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 19,
        embedding_dim: int = 100,
        hidden_dim: int = 100,
        dropout: float = 0.5,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.word_embed.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.word_embed.weight.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        # Attention parameters (Zhou et al.)
        self.att_w = nn.Linear(hidden_dim, hidden_dim)
        self.att_u = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, word_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.word_embed(word_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # Element-wise sum of forward / backward states
        hidden = out[:, :, : self.lstm.hidden_size] + out[:, :, self.lstm.hidden_size :]

        # Attention
        u = torch.tanh(self.att_w(hidden))
        scores = self.att_u(u).squeeze(-1)
        seqlen = scores.size(1)
        idx = torch.arange(seqlen, device=word_ids.device).unsqueeze(0)
        mask = idx < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=-1)
        sent = torch.bmm(alpha.unsqueeze(1), hidden).squeeze(1)
        sent = self.dropout(sent)
        logits = self.fc(sent)
        return logits
