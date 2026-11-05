"""Local SentixAI dashboard and prediction API."""

from __future__ import annotations

import os

# Limit thread pool allocations to prevent memory crashes on 512MB RAM cloud tiers
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
from functools import lru_cache
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
import torch

torch.set_num_threads(1)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sentixai_inference import SavedTextClassifier
from sentixai_xai import attention_explanation, lime_explanation, shap_explanation
from textcnn_inference import SavedTextCNNClassifier

ROOT = Path(__file__).resolve().parent
DOWNLOADS = ROOT / "outputs" / "batch_results"
app = FastAPI(title="SentixAI", version="4.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    methods: list[str] = Field(default_factory=lambda: ["attention"])
    model_type: str = Field(default="bilstm", description="Sentiment model architecture: 'bilstm' or 'textcnn'")
    include_emotion: bool = True
    lime_samples: int = Field(default=60, ge=10, le=1000)
    shap_samples: int = Field(default=40, ge=10, le=300)


@lru_cache
def sentiment_model() -> SavedTextClassifier:
    return SavedTextClassifier(ROOT / "outputs/artifacts", "bilstm_attention.pt")


@lru_cache
def textcnn_sentiment_model() -> SavedTextCNNClassifier | None:
    path = ROOT / "outputs/textcnn/artifacts/textcnn.pt"
    if path.exists():
        return SavedTextCNNClassifier(ROOT / "outputs/textcnn/artifacts", "textcnn.pt")
    return None


@lru_cache
def emotion_model() -> SavedTextClassifier:
    return SavedTextClassifier(ROOT / "outputs/emotion/artifacts", "emotion_bilstm_attention.pt")


def explain(classifier: SavedTextClassifier, prediction: dict, text: str, methods: list[str], request: PredictionRequest) -> dict:
    selected = set(methods)
    invalid = selected - {"attention", "lime", "shap"}
    if invalid:
        raise ValueError(f"Unsupported explanation method(s): {', '.join(sorted(invalid))}")
    result = {}
    if "attention" in selected:
        try:
            result["attention"] = attention_explanation(prediction)
        except Exception as err:
            result["attention"] = {"method": "attention", "important_words": [], "note": f"Attention unavailable: {err}"}
    if "lime" in selected:
        try:
            result["lime"] = lime_explanation(classifier, text, num_samples=request.lime_samples)
        except Exception as err:
            result["lime"] = {
                "method": "lime",
                "predicted_class": prediction.get("label", ""),
                "important_words": [],
                "note": f"LIME unavailable: {err}",
            }
    if "shap" in selected:
        try:
            result["shap"] = shap_explanation(classifier, text, nsamples=request.shap_samples)
        except Exception as err:
            result["shap"] = {
                "method": "shap",
                "predicted_class": prediction.get("label", ""),
                "important_words": [],
                "note": f"SHAP unavailable: {err}",
            }
    return result


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(ROOT / "static/index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "sentiment_model_bilstm": (ROOT / "outputs/artifacts/bilstm_attention.pt").exists(),
        "sentiment_model_textcnn": (ROOT / "outputs/textcnn/artifacts/textcnn.pt").exists(),
        "emotion_model": (ROOT / "outputs/emotion/artifacts/emotion_bilstm_attention.pt").exists(),
    }


@app.get("/api/models")
def list_models() -> dict:
    return {
        "sentiment_models": [
            {
                "id": "bilstm",
                "name": "BiLSTM + Additive Attention",
                "type": "Recurrent with Softmax Attention",
                "ready": (ROOT / "outputs/artifacts/bilstm_attention.pt").exists(),
            },
            {
                "id": "textcnn",
                "name": "TextCNN (1D-CNN)",
                "type": "Convolutional with Kernels (2, 3, 4)",
                "ready": (ROOT / "outputs/textcnn/artifacts/textcnn.pt").exists(),
            },
        ]
    }


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/analytics")
def analytics() -> dict:
    """Serve only metrics and curves created by the real training runs."""
    try:
        sentiment = load_results(ROOT / "outputs/results/evaluation.json")
        emotion = load_results(ROOT / "outputs/emotion/results/evaluation.json")
        models_data = {
            "sentiment": {
                "accuracy": sentiment["test_metrics"]["accuracy"],
                "f1": sentiment["test_metrics"]["f1_weighted"],
                "history": sentiment["history"],
                "classes": sentiment["class_names"],
            },
            "emotion": {
                "accuracy": emotion["test_metrics"]["accuracy"],
                "f1": emotion["test_metrics"]["f1_weighted"],
                "history": emotion["history"],
                "classes": emotion["class_names"],
            },
        }

        # Include TextCNN analytics if available
        textcnn_eval_path = ROOT / "outputs/textcnn/results/evaluation.json"
        if textcnn_eval_path.exists():
            textcnn_eval = load_results(textcnn_eval_path)
            models_data["textcnn"] = {
                "accuracy": textcnn_eval["test_metrics"]["accuracy"],
                "f1": textcnn_eval["test_metrics"]["f1_weighted"],
                "history": textcnn_eval["history"],
                "classes": textcnn_eval["class_names"],
            }

        return {
            "models": models_data,
            "images": {
                "sentiment_confusion_matrix": "/assets/sentiment/confusion-matrix",
                "textcnn_confusion_matrix": "/assets/textcnn/confusion-matrix",
                "emotion_confusion_matrix": "/assets/emotion/confusion-matrix",
            },
        }
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"Unable to load training analytics: {error}") from error


@app.get("/assets/{model}/confusion-matrix", include_in_schema=False)
def confusion_matrix_image(model: str) -> FileResponse:
    paths = {
        "sentiment": ROOT / "outputs/results/confusion_matrix.png",
        "textcnn": ROOT / "outputs/textcnn/results/confusion_matrix.png",
        "emotion": ROOT / "outputs/emotion/results/confusion_matrix.png",
    }
    if model not in paths or not paths[model].exists():
        raise HTTPException(status_code=404, detail="Confusion matrix is not available.")
    return FileResponse(paths[model], media_type="image/png")


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        model_choice = (request.model_type or "bilstm").lower()
        if model_choice == "textcnn":
            textcnn_clf = textcnn_sentiment_model()
            if textcnn_clf is None:
                # Graceful fallback to BiLSTM if TextCNN weights are not present
                sentiment_classifier = sentiment_model()
                sentiment = sentiment_classifier.predict(request.text)
                sentiment["model_used"] = "BiLSTM (Fallback: TextCNN weights not found)"
            else:
                sentiment = textcnn_clf.predict(request.text)
                sentiment["model_used"] = "TextCNN"
                sentiment["attention_tokens"] = []  # CNN uses global max-pooling instead of attention
                sentiment_classifier = textcnn_clf
        else:
            sentiment_classifier = sentiment_model()
            sentiment = sentiment_classifier.predict(request.text)
            sentiment["model_used"] = "BiLSTM + Additive Attention"

        response = {
            "sentiment": sentiment,
            "sentiment_confidence": sentiment["confidence"],
            "model_type": sentiment.get("model_used", "BiLSTM"),
            "explanation": {"sentiment": explain(sentiment_classifier, sentiment, request.text, request.methods, request)},
        }
        if request.include_emotion:
            emotion_classifier = emotion_model()
            emotion = emotion_classifier.predict(request.text)
            response["emotion"] = emotion
            response["emotion_confidence"] = emotion["confidence"]
            response["explanation"]["emotion"] = explain(emotion_classifier, emotion, request.text, request.methods, request)
        return response
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/batch")
async def batch_predict(file: UploadFile = File(...)) -> dict:
    """Validate review CSV, run both saved classifiers, and persist download output."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")
    try:
        uploaded = pd.read_csv(file.file)
    except Exception as error:
        raise HTTPException(status_code=400, detail="The uploaded file could not be read as a CSV.") from error
    if "review" not in uploaded.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a column named 'review'.")
    if uploaded.empty:
        raise HTTPException(status_code=400, detail="CSV contains no review rows.")
    if len(uploaded) > 5000:
        raise HTTPException(status_code=400, detail="CSV is limited to 5,000 reviews per upload.")
    reviews = uploaded["review"].fillna("").astype(str).str.strip().tolist()
    invalid_rows = [str(index + 2) for index, review in enumerate(reviews) if not review]
    if invalid_rows:
        preview = ", ".join(invalid_rows[:8])
        raise HTTPException(status_code=400, detail=f"Review column has empty value(s) at CSV row(s): {preview}.")
    try:
        sentiment_predictions = sentiment_model().predict_many(reviews)
        emotion_predictions = emotion_model().predict_many(reviews)
    except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=f"Unable to process reviews: {error}") from error

    results = pd.DataFrame({
        "review": reviews,
        "sentiment": [item["label"] for item in sentiment_predictions],
        "sentiment_confidence": [item["confidence"] for item in sentiment_predictions],
        "emotion": [item["label"] for item in emotion_predictions],
        "emotion_confidence": [item["confidence"] for item in emotion_predictions],
    })
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    batch_id = uuid4().hex
    try:
        results.to_csv(DOWNLOADS / f"{batch_id}.csv", index=False)
    except OSError as error:
        raise HTTPException(status_code=500, detail="Unable to create the processed CSV download.") from error
    sentiment_labels = ["negative", "neutral", "positive"]
    emotion_labels = ["joy", "anger", "sadness", "fear", "surprise", "love", "disappointment", "frustration"]
    return {
        "batch_id": batch_id,
        "total_reviews": len(results),
        "sentiment_counts": {label: int((results["sentiment"] == label).sum()) if label in sentiment_model().class_names else None for label in sentiment_labels},
        "emotion_counts": {label: int((results["emotion"] == label).sum()) if label in emotion_model().class_names else None for label in emotion_labels},
        "preview": results.head(100).to_dict(orient="records"),
        "download_url": f"/api/batch/{batch_id}/download",
    }


@app.get("/api/batch/{batch_id}/download")
def download_batch(batch_id: str) -> FileResponse:
    if not batch_id.isalnum():
        raise HTTPException(status_code=404, detail="Batch result not found.")
    path = DOWNLOADS / f"{batch_id}.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Batch result not found. Run the upload again.")
    return FileResponse(path, media_type="text/csv", filename="sentixai_batch_results.csv")
