"""VAE inference helper for the Streamlit dashboard.

Loads a saved ReviewVAE checkpoint, TF-IDF vectorizer, and scaler,
then provides a simple API for the 'Unusual Review Detection' tab.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vae_model import ReviewVAE


class VAEAnomalyDetector:
    """Wraps a trained ReviewVAE checkpoint for dashboard inference.

    Args:
        artifact_dir: Directory containing ``vae_model.pt``,
                      ``vae_vectorizer.pkl``, and ``vae_scaler.pkl``.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        artifact_dir = Path(artifact_dir)

        # Load vectorizer + scaler
        with open(artifact_dir / "vae_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(artifact_dir / "vae_scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        # Load model
        checkpoint = torch.load(
            artifact_dir / "vae_model.pt",
            map_location="cpu",
            weights_only=False,
        )
        cfg = checkpoint["model_config"]
        self.model = ReviewVAE(**cfg)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.threshold: float = checkpoint.get("threshold", 0.0)
        self.anomaly_top_k_pct: float = checkpoint.get("anomaly_top_k_pct", 5.0)

    # ------------------------------------------------------------------
    # Vectorisation
    # ------------------------------------------------------------------

    def _vectorise(self, texts: list[str]) -> torch.Tensor:
        X = self.vectorizer.transform(texts).toarray().astype(np.float32)
        X = self.scaler.transform(X)
        return torch.tensor(X, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_single(self, text: str) -> dict[str, float | bool]:
        """Return anomaly score and flag for one review."""
        x = self._vectorise([text])
        score = float(self.model.reconstruction_error(x)[0].item())
        return {
            "anomaly_score": round(score, 6),
            "is_unusual":    score >= self.threshold,
            "threshold":     round(self.threshold, 6),
        }

    def score_batch(self, texts: list[str]) -> pd.DataFrame:
        """Score a list of reviews and return a DataFrame with results."""
        x = self._vectorise(texts)
        scores = self.model.reconstruction_error(x).numpy()
        return pd.DataFrame(
            {
                "review":        texts,
                "anomaly_score": scores.round(6),
                "is_unusual":    scores >= self.threshold,
            }
        ).sort_values("anomaly_score", ascending=False)

    def latent_coords(self, texts: list[str]) -> np.ndarray:
        """Return (N, latent_dim) latent mean vectors for 2-D visualisation."""
        x = self._vectorise(texts)
        return self.model.encode(x).numpy()
