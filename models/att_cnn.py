"""Attention-Based CNN for semantic relation extraction (Shen & Huang, COLING 2016)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttCNN(nn.Module):
    """Att-CNN (Shen & Huang, COLING 2016): CNN + entity-aware attention."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 19,
        embedding_dim: int = 300,
        pos_vocab_size: int = 15,
        pos_embedding_dim: int = 10,
        position_embedding_dim: int = 5,
        max_position: int = 100,
        num_filters: int = 100,
        filter_size: int = 3,
        hidden_dim: int = 100,
        dropout: float = 0.5,
        pretrained_embeddings: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.position_embedding_dim = position_embedding_dim

        self.word_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.word_embed.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.word_embed.weight.requires_grad = False

        self.pos_embed = nn.Embedding(pos_vocab_size, pos_embedding_dim, padding_idx=0)
        # relative positions in [0, 2*max_position]
        self.pos1_embed = nn.Embedding(2 * max_position + 1, position_embedding_dim)
        self.pos2_embed = nn.Embedding(2 * max_position + 1, position_embedding_dim)

        input_dim = embedding_dim + pos_embedding_dim + 2 * position_embedding_dim
        self.conv = nn.Conv1d(input_dim, num_filters, kernel_size=filter_size, padding=filter_size // 2)

        # Attention MLP: concat(word, entity) → score
        att_in = embedding_dim * 2
        self.att_fc1 = nn.Linear(att_in, hidden_dim)
        self.att_fc2 = nn.Linear(hidden_dim, 1)

        # Final MLP: conv_feat + ctx_e1 + ctx_e2
        feat_dim = num_filters + 2 * embedding_dim
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def _entity_vector(self, word_emb: torch.Tensor, span: torch.Tensor) -> torch.Tensor:
        """Average word embeddings over entity token span. span: (B, 2)."""
        bsz, seqlen, dim = word_emb.size()
        device = word_emb.device
        idx = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        start = span[:, 0].unsqueeze(1)
        end = span[:, 1].unsqueeze(1)
        mask = (idx >= start) & (idx <= end)
        mask_f = mask.unsqueeze(-1).float()
        summed = (word_emb * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _attention_context(
        self, word_emb: torch.Tensor, entity_vec: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Word-level attention w.r.t. one entity (§II-A.1 / Shen & Huang)."""
        bsz, seqlen, dim = word_emb.size()
        ent = entity_vec.unsqueeze(1).expand(-1, seqlen, -1)
        h = torch.cat([word_emb, ent], dim=-1)
        u = self.att_fc2(torch.tanh(self.att_fc1(h))).squeeze(-1)  # (B, L)
        # mask pads
        device = word_emb.device
        idx = torch.arange(seqlen, device=device).unsqueeze(0)
        mask = idx < lengths.unsqueeze(1)
        u = u.masked_fill(~mask, -1e9)
        alpha = torch.softmax(u, dim=-1)
        ctx = torch.bmm(alpha.unsqueeze(1), word_emb).squeeze(1)
        return ctx

    def forward(
        self,
        word_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        pos1: torch.Tensor,
        pos2: torch.Tensor,
        lengths: torch.Tensor,
        e1_span: torch.Tensor,
        e2_span: torch.Tensor,
    ) -> torch.Tensor:
        word = self.word_embed(word_ids)
        pos = self.pos_embed(pos_ids)
        p1 = self.pos1_embed(pos1)
        p2 = self.pos2_embed(pos2)
        x = torch.cat([word, pos, p1, p2], dim=-1)  # (B, L, D)
        x = self.dropout(x)

        # Convolution over time
        conv_in = x.transpose(1, 2)  # (B, D, L)
        conv_out = torch.tanh(self.conv(conv_in))  # (B, F, L)
        # mask before max-pool
        bsz, nfilt, seqlen = conv_out.size()
        idx = torch.arange(seqlen, device=word_ids.device).unsqueeze(0)
        mask = (idx < lengths.unsqueeze(1)).unsqueeze(1)
        conv_out = conv_out.masked_fill(~mask, -1e9)
        sent_vec = conv_out.max(dim=2).values  # (B, F)

        e1_vec = self._entity_vector(word, e1_span)
        e2_vec = self._entity_vector(word, e2_span)
        ctx1 = self._attention_context(word, e1_vec, lengths)
        ctx2 = self._attention_context(word, e2_vec, lengths)

        feat = torch.cat([sent_vec, ctx1, ctx2], dim=-1)
        feat = self.dropout(torch.relu(self.fc1(feat)))
        logits = self.fc2(feat)
        return logits
