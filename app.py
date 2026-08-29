"""Local SentixAI dashboard and prediction API."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from sentixai_inference import SavedTextClassifier
from sentixai_xai import attention_explanation, lime_explanation, shap_explanation

ROOT = Path(__file__).resolve().parent
DOWNLOADS = ROOT / "outputs" / "batch_results"
app = FastAPI(title="SentixAI", version="4.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    methods: list[str] = Field(default_factory=lambda: ["attention"])
    include_emotion: bool = True
    lime_samples: int = Field(default=500, ge=100, le=2000)
    shap_samples: int = Field(default=160, ge=50, le=500)


@lru_cache
def sentiment_model() -> SavedTextClassifier:
    return SavedTextClassifier(ROOT / "outputs/artifacts", "bilstm_attention.pt")


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
        result["attention"] = attention_explanation(prediction)
    if "lime" in selected:
        result["lime"] = lime_explanation(classifier, text, num_samples=request.lime_samples)
    if "shap" in selected:
        result["shap"] = shap_explanation(classifier, text, nsamples=request.shap_samples)
    return result


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    return FileResponse(ROOT / "static/index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "sentiment_model": (ROOT / "outputs/artifacts/bilstm_attention.pt").exists(),
            "emotion_model": (ROOT / "outputs/emotion/artifacts/emotion_bilstm_attention.pt").exists()}


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/analytics")
def analytics() -> dict:
    """Serve only metrics and curves created by the real training runs."""
    try:
        sentiment = load_results(ROOT / "outputs/results/evaluation.json")
        emotion = load_results(ROOT / "outputs/emotion/results/evaluation.json")
        return {
            "models": {
                "sentiment": {"accuracy": sentiment["test_metrics"]["accuracy"], "f1": sentiment["test_metrics"]["f1_weighted"],
                              "history": sentiment["history"], "classes": sentiment["class_names"]},
                "emotion": {"accuracy": emotion["test_metrics"]["accuracy"], "f1": emotion["test_metrics"]["f1_weighted"],
                            "history": emotion["history"], "classes": emotion["class_names"]},
            },
            "images": {
                "sentiment_confusion_matrix": "/assets/sentiment/confusion-matrix",
                "emotion_confusion_matrix": "/assets/emotion/confusion-matrix",
            },
        }
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"Unable to load training analytics: {error}") from error


@app.get("/assets/{model}/confusion-matrix", include_in_schema=False)
def confusion_matrix_image(model: str) -> FileResponse:
    paths = {"sentiment": ROOT / "outputs/results/confusion_matrix.png", "emotion": ROOT / "outputs/emotion/results/confusion_matrix.png"}
    if model not in paths or not paths[model].exists():
        raise HTTPException(status_code=404, detail="Confusion matrix is not available.")
    return FileResponse(paths[model], media_type="image/png")


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        sentiment_classifier = sentiment_model()
        sentiment = sentiment_classifier.predict(request.text)
        response = {
            "sentiment": sentiment,
            "sentiment_confidence": sentiment["confidence"],
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
