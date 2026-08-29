"""Reusable inference for saved SentixAI classifiers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from predict_sentixai import load_checkpoint
from sentixai_data import clean_and_tokenize, encode_tokens
from sentixai_model import BiLSTMAttentionClassifier


class SavedTextClassifier:
    """Loads one checkpoint once and produces actual model probabilities/attention."""

    def __init__(self, model_dir: str | Path, model_name: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model_dir = Path(model_dir)
        checkpoint = load_checkpoint(self.model_dir / model_name, self.device)
        self.vocabulary = json.loads((self.model_dir / "vocabulary.json").read_text(encoding="utf-8"))
        self.class_names = checkpoint["class_names"]
        self.max_length = checkpoint["max_length"]
        self.model = BiLSTMAttentionClassifier(**checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def predict(self, text: str) -> dict:
        tokens = clean_and_tokenize(text)
        if not tokens:
            raise ValueError("Enter at least one letter or number so the model has text to classify.")
        encoded = encode_tokens(tokens, self.vocabulary, self.max_length)
        with torch.no_grad():
            logits, weights = self.model(torch.tensor([encoded], dtype=torch.long, device=self.device))
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
            predicted_id = int(logits.argmax(dim=1).item())
            attention = weights.squeeze(0).cpu().tolist()[: min(len(tokens), self.max_length)]
        attention_tokens = [
            {"token": token, "weight": float(weight)}
            for token, weight in zip(tokens[: self.max_length], attention)
        ]
        return {
            "label": self.class_names[predicted_id],
            "confidence": float(probabilities[predicted_id]),
            "probabilities": dict(zip(self.class_names, probabilities)),
            "attention_tokens": attention_tokens,
        }

    def predict_probabilities(self, texts: list[str]) -> np.ndarray:
        """LIME/SHAP-compatible classifier function using this exact checkpoint."""
        encoded = []
        for text in texts:
            tokens = clean_and_tokenize(text) or ["[empty]"]
            encoded.append(encode_tokens(tokens, self.vocabulary, self.max_length))
        with torch.no_grad():
            token_ids = torch.tensor(encoded, dtype=torch.long, device=self.device)
            logits, _ = self.model(token_ids)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        return np.asarray(probabilities)

    def predict_many(self, texts: list[str], batch_size: int = 128) -> list[dict]:
        """Efficient batch prediction from the same saved model checkpoint."""
        token_lists = [clean_and_tokenize(text) for text in texts]
        if any(not tokens for tokens in token_lists):
            raise ValueError("Each review must contain at least one letter or number.")
        all_probabilities = []
        with torch.no_grad():
            for start in range(0, len(token_lists), batch_size):
                encoded = [encode_tokens(tokens, self.vocabulary, self.max_length)
                           for tokens in token_lists[start:start + batch_size]]
                token_ids = torch.tensor(encoded, dtype=torch.long, device=self.device)
                logits, _ = self.model(token_ids)
                all_probabilities.extend(torch.softmax(logits, dim=1).cpu().tolist())
        return [
            {"label": self.class_names[int(np.argmax(probabilities))],
             "confidence": float(max(probabilities)),
             "probabilities": dict(zip(self.class_names, probabilities))}
            for probabilities in all_probabilities
        ]
