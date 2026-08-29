"""Run saved sentiment and emotion models on one text without retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from predict_sentixai import load_checkpoint
from sentixai_data import clean_and_tokenize, encode_tokens
from sentixai_model import BiLSTMAttentionClassifier


def predict(text: str, model_dir: Path, model_name: str, device: torch.device) -> dict:
    checkpoint = load_checkpoint(model_dir / model_name, device)
    vocabulary = json.loads((model_dir / "vocabulary.json").read_text(encoding="utf-8"))
    model = BiLSTMAttentionClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    tokens = clean_and_tokenize(text)
    encoded = encode_tokens(tokens, vocabulary, checkpoint["max_length"])
    with torch.no_grad():
        logits, weights = model(torch.tensor([encoded], dtype=torch.long, device=device))
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        label_index = int(logits.argmax(dim=1).item())
        attention = weights.squeeze(0).cpu().tolist()[:len(tokens)]
    ranked = sorted(zip(tokens[:len(attention)], attention), key=lambda item: item[1], reverse=True)
    return {
        "label": checkpoint["class_names"][label_index],
        "confidence": float(probabilities[label_index]),
        "probabilities": dict(zip(checkpoint["class_names"], probabilities)),
        "attention_tokens": [{"token": token, "weight": float(weight)} for token, weight in ranked[:10]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--sentiment-model-dir", default="outputs/artifacts")
    parser.add_argument("--emotion-model-dir", default="outputs/emotion/artifacts")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    output = {
        "sentiment": predict(args.text, Path(args.sentiment_model_dir), "bilstm_attention.pt", device),
        "emotion": predict(args.text, Path(args.emotion_model_dir), "emotion_bilstm_attention.pt", device),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
