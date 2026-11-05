"""Attention, LIME, and optional SHAP explanations for saved SentixAI models."""

from __future__ import annotations

from typing import Any

import numpy as np

from sentixai_data import clean_and_tokenize
from sentixai_inference import SavedTextClassifier

try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    LimeTextExplainer = None

try:
    import shap
except ImportError:
    shap = None


def attention_explanation(prediction: dict) -> dict[str, Any]:
    return {
        "method": "attention",
        "important_words": sorted(prediction["attention_tokens"], key=lambda item: item["weight"], reverse=True),
        "note": "Weights come from the model's attention layer and sum to approximately 1 over processed tokens.",
    }


def lime_explanation(
    classifier: SavedTextClassifier,
    text: str,
    num_features: int = 10,
    num_samples: int = 100,
) -> dict[str, Any]:
    """Local Interpretable Model-agnostic Explanations with bounded perturbations."""
    if LimeTextExplainer is None:
        raise ImportError("LIME library is not installed in the current environment.")

    tokens = clean_and_tokenize(text)
    if not tokens:
        return {
            "method": "lime",
            "predicted_class": "",
            "important_words": [],
            "note": "Input text contains no tokens to explain.",
            "num_samples": 0,
        }

    prediction = classifier.predict(text)
    predicted_label = prediction["label"]
    predicted_id = classifier.class_names.index(predicted_label)

    # Use split expression matching clean_and_tokenize for consistent tokenization
    explainer = LimeTextExplainer(
        class_names=classifier.class_names,
        random_state=42,
        split_expression=r"\W+",
    )
    explanation = explainer.explain_instance(
        text,
        classifier.predict_probabilities,
        labels=(predicted_id,),
        num_features=num_features,
        num_samples=num_samples,
    )
    raw_list = explanation.as_list(label=predicted_id)
    return {
        "method": "lime",
        "predicted_class": predicted_label,
        "important_words": [
            {"token": str(token), "contribution": float(weight)}
            for token, weight in raw_list
        ],
        "note": "Positive values locally support the displayed class; negative values locally oppose it.",
        "num_samples": num_samples,
    }


def shap_explanation(
    classifier: SavedTextClassifier,
    text: str,
    max_tokens: int = 16,
    nsamples: int = 60,
) -> dict[str, Any]:
    """Kernel SHAP on binary word-presence masks; bounded and silent."""
    if shap is None:
        raise ImportError("SHAP library is not installed in the current environment.")

    tokens = clean_and_tokenize(text)[:max_tokens]
    if not tokens:
        return {
            "method": "shap",
            "predicted_class": "",
            "important_words": [],
            "note": "Input text contains no tokens to explain.",
            "nsamples": 0,
            "processed_tokens": 0,
            "input_truncated": False,
        }

    prediction = classifier.predict(text)
    predicted_label = prediction["label"]
    predicted_id = classifier.class_names.index(predicted_label)

    def masked_predict(masks: np.ndarray) -> np.ndarray:
        texts = [
            " ".join(token for token, enabled in zip(tokens, mask) if enabled >= 0.5) or "[empty]"
            for mask in masks
        ]
        return classifier.predict_probabilities(texts)

    np.random.seed(42)
    background = np.zeros((1, len(tokens)))
    explainer = shap.KernelExplainer(masked_predict, background)

    # Avoid requesting more evaluations than the combinatorial space for short texts
    actual_samples = min(nsamples, 2 ** len(tokens)) if len(tokens) <= 8 else nsamples
    shap_values = explainer.shap_values(
        np.ones((1, len(tokens))),
        nsamples=actual_samples,
        silent=True,
    )

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
        "predicted_class": predicted_label,
        "important_words": important_words,
        "note": "Positive SHAP values support the displayed class versus an all-words-masked baseline; negative values oppose it.",
        "nsamples": actual_samples,
        "processed_tokens": len(tokens),
        "input_truncated": len(clean_and_tokenize(text)) > len(tokens),
    }
