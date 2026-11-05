"""Predict sentiment for text using the trained TextCNN model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from textcnn_inference import SavedTextCNNClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Input text to classify")
    parser.add_argument("--model-dir", default="outputs/textcnn/artifacts", help="Model artifacts directory")
    parser.add_argument("--model-name", default="textcnn.pt", help="Checkpoint filename")
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()

    classifier = SavedTextCNNClassifier(Path(args.model_dir), args.model_name, device=args.device)
    prediction = classifier.predict(args.text)
    print(json.dumps(prediction, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
