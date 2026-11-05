"""Train the ReviewVAE on a CSV of unlabelled review text.

Module 5 – Generative Models (VAE-based anomaly detection)

Usage
-----
python train_vae.py --csv data/sample_batch_reviews.csv
python train_vae.py --csv reviews.csv --epochs 30 --latent-dim 32 --beta 0.5
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from sentixai_data import clean_and_tokenize
from vae_model import ReviewVAE


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_reviews(csv_path: str, text_column: str) -> list[str]:
    """Load and clean raw review text from a CSV file."""
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in {csv_path}. "
                         f"Available columns: {df.columns.tolist()}")
    texts = df[text_column].fillna("").astype(str).str.strip()
    texts = texts[texts.str.len() > 0].tolist()
    if not texts:
        raise ValueError("No non-empty review texts found.")
    return texts


def build_tfidf_features(
    texts: list[str],
    max_features: int = 5000,
) -> tuple[np.ndarray, TfidfVectorizer, MinMaxScaler]:
    """Tokenise, TF-IDF vectorise, and min-max normalise to [0, 1]."""
    vectorizer = TfidfVectorizer(
        tokenizer=clean_and_tokenize,
        max_features=max_features,
        min_df=1,
        token_pattern=None,     # tokenizer= overrides token_pattern
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, vectorizer, scaler


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model: ReviewVAE,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = recon_sum = kl_sum = 0.0
    n = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar, _ = model(batch_x)
            loss, recon_loss, kl_loss = model.loss(recon, batch_x, mu, logvar)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            recon_sum  += recon_loss.item() * bs
            kl_sum     += kl_loss.item() * bs
            n += bs
    return {"loss": total_loss / n, "recon": recon_sum / n, "kl": kl_sum / n}


def save_loss_plot(history: dict[str, list[float]], out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train total")
    if history.get("val_loss"):
        axes[0].plot(epochs, history["val_loss"], label="Val total")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Total ELBO loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_recon"], label="Train recon")
    axes[1].plot(epochs, history["train_kl"],    label="Train KL")
    if history.get("val_recon"):
        axes[1].plot(epochs, history["val_recon"], label="Val recon", linestyle="--")
        axes[1].plot(epochs, history["val_kl"],    label="Val KL",    linestyle="--")
    axes[1].set(xlabel="Epoch", ylabel="Loss component", title="Reconstruction vs KL")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",          required=True,         help="CSV file with review text")
    parser.add_argument("--text-column",  default="review",      help="Column name for review text")
    parser.add_argument("--output-dir",   default="outputs/vae", help="Directory for VAE artifacts")
    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF vocabulary size")
    parser.add_argument("--hidden-dim",   type=int, default=256,  help="Hidden layer width")
    parser.add_argument("--latent-dim",   type=int, default=32,   help="Latent space dimension")
    parser.add_argument("--beta",         type=float, default=1.0, help="KL weight (beta-VAE)")
    parser.add_argument("--batch-size",   type=int, default=64)
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split",    type=float, default=0.1,
                        help="Fraction of data held out for validation")
    parser.add_argument("--anomaly-top-k", type=float, default=5.0,
                        help="Percentage of reviews flagged as anomalies (default: top 5%%)")
    parser.add_argument("--patience",     type=int, default=5)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--device",       choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps"  if torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Directories
    out_dir = Path(args.output_dir)
    artifacts_dir = out_dir / "artifacts"
    results_dir   = out_dir / "results"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("Loading and vectorising reviews …")
    texts = load_reviews(args.csv, args.text_column)
    X, vectorizer, scaler = build_tfidf_features(texts, args.max_features)
    input_dim = X.shape[1]
    print(f"  {len(texts)} reviews | TF-IDF dim: {input_dim}")

    # Train / validation split
    n_val = max(1, int(len(texts) * args.val_split))
    n_train = len(texts) - n_val
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(texts))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    X_val   = torch.tensor(X[val_idx],   dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train), batch_size=args.batch_size,
        shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(X_val), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )

    # Model
    model = ReviewVAE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        beta=args.beta,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # Training
    history: dict[str, list[float]] = {
        "train_loss": [], "train_recon": [], "train_kl": [],
        "val_loss":   [], "val_recon":   [], "val_kl":   [],
    }
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"\nTraining ReviewVAE for up to {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, device, optimizer)
        val_m   = run_epoch(model, val_loader,   device)

        history["train_loss"].append(train_m["loss"])
        history["train_recon"].append(train_m["recon"])
        history["train_kl"].append(train_m["kl"])
        history["val_loss"].append(val_m["loss"])
        history["val_recon"].append(val_m["recon"])
        history["val_kl"].append(val_m["kl"])

        print(f"Epoch {epoch:03d} | "
              f"train loss {train_m['loss']:.5f} "
              f"(recon {train_m['recon']:.5f} | kl {train_m['kl']:.5f})  |  "
              f"val loss {val_m['loss']:.5f} "
              f"(recon {val_m['recon']:.5f} | kl {val_m['kl']:.5f})")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)

    # Anomaly scoring on ALL reviews
    print("\nScoring all reviews for anomalies …")
    X_all = torch.tensor(X, dtype=torch.float32)
    scores = model.reconstruction_error(X_all.to(device)).cpu().numpy()

    threshold = np.percentile(scores, 100 - args.anomaly_top_k)
    is_anomaly = scores >= threshold

    anomaly_df = pd.DataFrame({
        "review":             texts,
        "anomaly_score":      scores.round(6),
        "is_unusual":         is_anomaly,
    }).sort_values("anomaly_score", ascending=False)

    anomaly_csv = results_dir / "unusual_reviews.csv"
    anomaly_df.to_csv(anomaly_csv, index=False)
    print(f"  Flagged {is_anomaly.sum()} reviews as unusual (top {args.anomaly_top_k}%)")
    print(f"  Anomaly report saved -> {anomaly_csv}")

    # Save artifacts
    checkpoint_path = artifacts_dir / "vae_model.pt"
    torch.save(
        {
            "model_state":  best_state,
            "model_config": {
                "input_dim":  input_dim,
                "hidden_dim": args.hidden_dim,
                "latent_dim": args.latent_dim,
                "beta":       args.beta,
            },
            "threshold":        float(threshold),
            "anomaly_top_k_pct": args.anomaly_top_k,
        },
        checkpoint_path,
    )
    with open(artifacts_dir / "vae_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(artifacts_dir / "vae_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    results = {
        "best_val_loss":    float(best_val_loss),
        "anomaly_threshold": float(threshold),
        "total_reviews":    len(texts),
        "flagged_reviews":  int(is_anomaly.sum()),
        "history":          history,
        "config": vars(args),
    }
    (results_dir / "vae_evaluation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    save_loss_plot(history, results_dir / "vae_training_curves.png")

    print(f"\nCheckpoint saved at {checkpoint_path}")
    print(f"Loss curves saved at {results_dir / 'vae_training_curves.png'}")
    print(f"Results saved at {results_dir / 'vae_evaluation.json'}")


if __name__ == "__main__":
    main()
