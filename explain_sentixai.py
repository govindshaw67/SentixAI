"""Generate actual attention, LIME, and/or SHAP explanations from a saved model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sentixai_inference import SavedTextClassifier
from sentixai_xai import attention_explanation, lime_explanation, shap_explanation

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", choices=["sentiment", "emotion"], default="sentiment")
    parser.add_argument("--methods", nargs="+", choices=["attention", "lime", "shap"], default=["attention"])
    parser.add_argument("--lime-samples", type=int, default=500)
    parser.add_argument("--shap-samples", type=int, default=160)
    args = parser.parse_args()
    if args.model == "sentiment":
        classifier = SavedTextClassifier(ROOT / "outputs/artifacts", "bilstm_attention.pt")
    else:
        classifier = SavedTextClassifier(ROOT / "outputs/emotion/artifacts", "emotion_bilstm_attention.pt")
    prediction = classifier.predict(args.text)
    explanations = {}
    if "attention" in args.methods:
        explanations["attention"] = attention_explanation(prediction)
    if "lime" in args.methods:
        explanations["lime"] = lime_explanation(classifier, args.text, num_samples=args.lime_samples)
    if "shap" in args.methods:
        explanations["shap"] = shap_explanation(classifier, args.text, nsamples=args.shap_samples)
    print(json.dumps({"prediction": prediction, "explanation": explanations}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
