"""Neural architectures used in SentixAI's controlled comparison study."""

from __future__ import annotations

import torch
from torch import nn

from sentixai_model import AdditiveAttention


class MLPClassifier(nn.Module):
    """Embedding-average MLP baseline; no recurrent sequence encoder."""
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=128, dropout=0.35, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.padding_idx = padding_idx

    def forward(self, token_ids):
        mask = token_ids.ne(self.padding_idx).unsqueeze(-1)
        vectors = self.embedding(token_ids) * mask
        pooled = vectors.sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(pooled), None


class LSTMClassifier(nn.Module):
    """Unidirectional LSTM with final valid hidden state."""
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=128, dropout=0.35, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
        self.padding_idx = padding_idx

    def forward(self, token_ids):
        output, _ = self.lstm(self.embedding_dropout(self.embedding(token_ids)))
        last = token_ids.ne(self.padding_idx).sum(1).clamp(min=1).sub(1)
        features = output[torch.arange(token_ids.size(0), device=token_ids.device), last]
        return self.classifier(features), None


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM without attention."""
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=128, dropout=0.35, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, num_classes))
        self.padding_idx = padding_idx

    def forward(self, token_ids):
        output, _ = self.lstm(self.embedding_dropout(self.embedding(token_ids)))
        mask = token_ids.ne(self.padding_idx).unsqueeze(-1)
        pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.classifier(pooled), None


class BiLSTMAttentionComparison(nn.Module):
    """BiLSTM + additive attention for the controlled comparison."""
    def __init__(self, vocab_size, num_classes, embedding_dim=128, hidden_dim=128, dropout=0.35, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = AdditiveAttention(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, num_classes))
        self.padding_idx = padding_idx

    def forward(self, token_ids):
        features, _ = self.lstm(self.embedding_dropout(self.embedding(token_ids)))
        context, weights = self.attention(features, token_ids.ne(self.padding_idx))
        return self.classifier(context), weights


MODELS = {"mlp": MLPClassifier, "lstm": LSTMClassifier, "bilstm": BiLSTMClassifier, "bilstm_attention": BiLSTMAttentionComparison}
