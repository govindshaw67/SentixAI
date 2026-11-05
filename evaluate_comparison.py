"""Controlled comparative evaluation between BiLSTM and TextCNN on held-out test data."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from predict_sentixai import load_checkpoint
from sentixai_data import SentimentDataset, load_and_split_data
from sentixai_model import BiLSTMAttentionClassifier
from textcnn_model import TextCNNClassifier


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    labels, preds = [], []
    with torch.inference_mode():
        for token_ids, targets in loader:
            token_ids, targets = token_ids.to(device), targets.to(device)
            logits, _ = model(token_ids)
            loss = criterion(logits, targets)
            total_loss += loss.item() * targets.size(0)
            labels.extend(targets.cpu().tolist())
            preds.extend(logits.argmax(dim=1).cpu().tolist())

    acc = float(accuracy_score(labels, preds))
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    return {
        "loss": float(total_loss / len(loader.dataset)),
        "accuracy": acc,
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "f1_macro": float(f1_m),
    }


def measure_latency(model, loader, device, num_samples: int = 1000) -> float:
    model.eval()
    total_time = 0.0
    processed = 0
    with torch.inference_mode():
        for token_ids, _ in loader:
            token_ids = token_ids.to(device)
            t0 = time.perf_counter()
            _ = model(token_ids)
            total_time += time.perf_counter() - t0
            processed += token_ids.size(0)
            if processed >= num_samples:
                break
    return (total_time / processed) * 1000.0  # ms per sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="/Users/govindprasadshaw/Downloads/DATASETS/IMDB Dataset.csv")
    parser.add_argument("--bilstm-dir", default="outputs/artifacts")
    parser.add_argument("--bilstm-name", default="bilstm_attention.pt")
    parser.add_argument("--textcnn-dir", default="outputs/textcnn/artifacts")
    parser.add_argument("--textcnn-name", default="textcnn.pt")
    parser.add_argument("--output-file", default="outputs/textcnn/results/model_comparison.json")
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else args.device)
    print(f"Evaluation device: {device}")

    # Load dataset
    print(f"Loading test split from: {args.csv}")
    prepared = load_and_split_data(args.csv, seed=args.seed)
    print(f"Held-out test set size: {len(prepared.test_labels):,} reviews")

    # 1. Evaluate BiLSTM
    bilstm_path = Path(args.bilstm_dir) / args.bilstm_name
    bilstm_vocab_path = Path(args.bilstm_dir) / "vocabulary.json"
    print(f"\nEvaluating BiLSTM from: {bilstm_path}")
    bilstm_ckpt = load_checkpoint(bilstm_path, device)
    bilstm_vocab = json.loads(bilstm_vocab_path.read_text(encoding="utf-8"))
    bilstm_model = BiLSTMAttentionClassifier(**bilstm_ckpt["model_config"]).to(device)
    bilstm_model.load_state_dict(bilstm_ckpt["model_state"])

    bilstm_dataset = SentimentDataset(prepared.test_tokens, prepared.test_labels, bilstm_vocab, bilstm_ckpt["max_length"])
    bilstm_loader = DataLoader(bilstm_dataset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    bilstm_metrics = evaluate_model(bilstm_model, bilstm_loader, criterion, device)
    bilstm_latency = measure_latency(bilstm_model, bilstm_loader, device)
    bilstm_params = sum(p.numel() for p in bilstm_model.parameters())
    bilstm_size_mb = bilstm_path.stat().st_size / (1024 * 1024)

    # 2. Evaluate TextCNN
    textcnn_path = Path(args.textcnn_dir) / args.textcnn_name
    textcnn_vocab_path = Path(args.textcnn_dir) / "vocabulary.json"
    print(f"Evaluating TextCNN from: {textcnn_path}")
    textcnn_ckpt = load_checkpoint(textcnn_path, device)
    textcnn_vocab = json.loads(textcnn_vocab_path.read_text(encoding="utf-8"))
    textcnn_model = TextCNNClassifier(**textcnn_ckpt["model_config"]).to(device)
    textcnn_model.load_state_dict(textcnn_ckpt["model_state"])

    textcnn_dataset = SentimentDataset(prepared.test_tokens, prepared.test_labels, textcnn_vocab, textcnn_ckpt["max_length"])
    textcnn_loader = DataLoader(textcnn_dataset, batch_size=64, shuffle=False)

    textcnn_metrics = evaluate_model(textcnn_model, textcnn_loader, criterion, device)
    textcnn_latency = measure_latency(textcnn_model, textcnn_loader, device)
    textcnn_params = sum(p.numel() for p in textcnn_model.parameters())
    textcnn_size_mb = textcnn_path.stat().st_size / (1024 * 1024)

    comparison_results = {
        "dataset": args.csv,
        "test_samples": len(prepared.test_labels),
        "models": {
            "BiLSTM_Attention": {
                "accuracy": bilstm_metrics["accuracy"],
                "precision_weighted": bilstm_metrics["precision_weighted"],
                "recall_weighted": bilstm_metrics["recall_weighted"],
                "f1_weighted": bilstm_metrics["f1_weighted"],
                "f1_macro": bilstm_metrics["f1_macro"],
                "parameters": bilstm_params,
                "model_size_mb": round(bilstm_size_mb, 2),
                "latency_ms_per_sample": round(bilstm_latency, 3),
            },
            "TextCNN": {
                "accuracy": textcnn_metrics["accuracy"],
                "precision_weighted": textcnn_metrics["precision_weighted"],
                "recall_weighted": textcnn_metrics["recall_weighted"],
                "f1_weighted": textcnn_metrics["f1_weighted"],
                "f1_macro": textcnn_metrics["f1_macro"],
                "parameters": textcnn_params,
                "model_size_mb": round(textcnn_size_mb, 2),
                "latency_ms_per_sample": round(textcnn_latency, 3),
            },
        },
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison_results, indent=2), encoding="utf-8")

    # Display comparison table
    df = pd.DataFrame([
        {
            "Model": "BiLSTM + Additive Attention",
            "Accuracy": f"{bilstm_metrics['accuracy'] * 100:.2f}%",
            "Precision (W)": f"{bilstm_metrics['precision_weighted'] * 100:.2f}%",
            "Recall (W)": f"{bilstm_metrics['recall_weighted'] * 100:.2f}%",
            "F1-Score (W)": f"{bilstm_metrics['f1_weighted'] * 100:.2f}%",
            "Macro F1": f"{bilstm_metrics['f1_macro'] * 100:.2f}%",
            "Parameters": f"{bilstm_params:,}",
            "Model Size": f"{bilstm_size_mb:.1f} MB",
            "Latency": f"{bilstm_latency:.2f} ms/sample",
        },
        {
            "Model": "TextCNN (Kernels 2, 3, 4)",
            "Accuracy": f"{textcnn_metrics['accuracy'] * 100:.2f}%",
            "Precision (W)": f"{textcnn_metrics['precision_weighted'] * 100:.2f}%",
            "Recall (W)": f"{textcnn_metrics['recall_weighted'] * 100:.2f}%",
            "F1-Score (W)": f"{textcnn_metrics['f1_weighted'] * 100:.2f}%",
            "Macro F1": f"{textcnn_metrics['f1_macro'] * 100:.2f}%",
            "Parameters": f"{textcnn_params:,}",
            "Model Size": f"{textcnn_size_mb:.1f} MB",
            "Latency": f"{textcnn_latency:.2f} ms/sample",
        },
    ])

    print("\n" + "=" * 90)
    print("CONTROLLED MODEL COMPARISON SUMMARY (Same 4,959 Test Reviews)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
