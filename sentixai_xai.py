"""Attention, LIME, and optional SHAP explanations for saved SentixAI models."""

from __future__ import annotations

from typing import Any

import numpy as np

from sentixai_data import clean_and_tokenize
from sentixai_inference import SavedTextClassifier


def attention_explanation(prediction: dict) -> dict[str, Any]:
    return {
        "method": "attention",
        "important_words": sorted(prediction["attention_tokens"], key=lambda item: item["weight"], reverse=True),
        "note": "Weights come from the model's attention layer and sum to approximately 1 over processed tokens.",
    }


def lime_explanation(classifier: SavedTextClassifier, text: str, num_features: int = 10, num_samples: int = 500) -> dict[str, Any]:
    from lime.lime_text import LimeTextExplainer

    prediction = classifier.predict(text)
    predicted_id = classifier.class_names.index(prediction["label"])
    explainer = LimeTextExplainer(class_names=classifier.class_names, random_state=42)
    explanation = explainer.explain_instance(
        text, classifier.predict_probabilities, labels=(predicted_id,),
        num_features=num_features, num_samples=num_samples,
    )
    return {
        "method": "lime",
        "predicted_class": prediction["label"],
        "important_words": [
            {"token": token, "contribution": float(weight)}
            for token, weight in explanation.as_list(label=predicted_id)
        ],
        "note": "Positive values locally support the displayed class; negative values locally oppose it.",
        "num_samples": num_samples,
    }


def shap_explanation(classifier: SavedTextClassifier, text: str, max_tokens: int = 16, nsamples: int = 160) -> dict[str, Any]:
    """Kernel SHAP on binary word-presence masks; intentionally optional and bounded."""
    import shap

    prediction = classifier.predict(text)
    predicted_id = classifier.class_names.index(prediction["label"])
    tokens = clean_and_tokenize(text)[:max_tokens]
    if not tokens:
        raise ValueError("SHAP needs at least one processed token.")

    def masked_predict(masks: np.ndarray) -> np.ndarray:
        texts = [" ".join(token for token, enabled in zip(tokens, mask) if enabled >= 0.5) or "[empty]" for mask in masks]
        return classifier.predict_probabilities(texts)

    np.random.seed(42)
    background = np.zeros((1, len(tokens)))
    explainer = shap.KernelExplainer(masked_predict, background)
    shap_values = explainer.shap_values(np.ones((1, len(tokens))), nsamples=nsamples)
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[predicted_id])[0]
    else:
        values = np.asarray(shap_values)
        values = values[0, :, predicted_id] if values.ndim == 3 else values[0]
    important_words = [
        {"token": token, "contribution": float(value)}
        for token, value in sorted(zip(tokens, values), key=lambda item: abs(item[1]), reverse=True)
    ]
    return {
        "method": "shap",
        "predicted_class": prediction["label"],
        "important_words": important_words,
        "note": "Positive SHAP values support the displayed class versus an all-words-masked baseline; negative values oppose it.",
        "nsamples": nsamples,
        "processed_tokens": len(tokens),
        "input_truncated": len(clean_and_tokenize(text)) > len(tokens),
    }
