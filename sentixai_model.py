"""BiLSTM with additive attention for SentixAI."""

from __future__ import annotations

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """Masked additive attention that returns both context and token weights."""

    def __init__(self, feature_size: int, attention_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(feature_size, attention_size)
        self.score = nn.Linear(attention_size, 1, bias=False)

    def forward(
        self, sequence_features: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # sequence_features: [batch, tokens, features]
        attention_scores = self.score(torch.tanh(self.projection(sequence_features))).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~padding_mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), sequence_features).squeeze(1)
        return context, attention_weights


class BiLSTMAttentionClassifier(nn.Module):
    """Tokens -> Embedding -> BiLSTM -> Attention -> Dropout -> Dense logits."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.35,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self.padding_idx = padding_idx

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = token_ids.ne(self.padding_idx)
        embeddings = self.embedding_dropout(self.embedding(token_ids))
        sequence_features, _ = self.bilstm(embeddings)
        context, attention_weights = self.attention(sequence_features, padding_mask)
        logits = self.classifier(self.dropout(context))
        return logits, attention_weights
