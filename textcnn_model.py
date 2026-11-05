"""TextCNN (Convolutional Neural Network for Text Classification) for SentixAI.

Architecture:
  Token IDs -> Embedding -> 1D Convolutions with Kernels (2, 3, 4) -> ReLU
  -> 1D Global Max Pooling over time -> Concatenation -> Dropout -> Dense -> Logits
"""

from __future__ import annotations

import torch
from torch import nn


class TextCNNClassifier(nn.Module):
    """Multiple 1D Convolutions with kernel sizes (2, 3, 4) + Global Max Pooling."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 128,
        num_filters: int = 64,
        kernel_sizes: tuple[int, ...] = (2, 3, 4),
        hidden_dim: int = 64,
        dropout: float = 0.35,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout)

        # Multiple 1D Convolutions with different n-gram kernel window sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=0,
            )
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        total_filter_dim = num_filters * len(kernel_sizes)

        # Dense classification head
        self.fc = nn.Sequential(
            nn.Linear(total_filter_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.padding_idx = padding_idx

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # token_ids: [batch, seq_len]
        # embeddings: [batch, seq_len, embedding_dim]
        embeddings = self.embedding_dropout(self.embedding(token_ids))

        # Permute for 1D convolution: [batch, embedding_dim, seq_len]
        x = embeddings.permute(0, 2, 1)

        # Convolve, apply ReLU, and global max pool across sequence length dimension
        pooled_outputs = []
        for conv in self.convs:
            # conv(x): [batch, num_filters, seq_len - k + 1]
            c = torch.relu(conv(x))
            # Global max-pool over time: [batch, num_filters]
            p = torch.max(c, dim=2)[0]
            pooled_outputs.append(p)

        # Concatenate features from all kernel sizes: [batch, num_filters * len(kernel_sizes)]
        features = torch.cat(pooled_outputs, dim=1)

        # Logits: [batch, num_classes]
        logits = self.fc(features)
        return logits, features
