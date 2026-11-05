"""Train and evaluate an independent TextCNN sentiment classifier for SentixAI."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from sentixai_data import SentimentDataset, build_vocabulary, load_and_split_data
from textcnn_model import TextCNNClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(requested)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels, predictions = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for token_ids, targets in loader:
            token_ids, targets = token_ids.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, _ = model(token_ids)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * targets.size(0)
            labels.extend(targets.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / len(loader.dataset), labels, predictions


def metrics_dict(labels, predictions) -> dict[str, float]:
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(precision_m),
        "recall_macro": float(recall_m),
        "f1_macro": float(f1_m),
    }


def save_plots(history, labels, predictions, class_names, results_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, history["train_loss"], label="Training loss", color="#2563eb", marker="o")
    axes[0].plot(epochs, history["validation_loss"], label="Validation loss", color="#ea580c", marker="o")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="TextCNN Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(epochs, history["train_accuracy"], label="Training accuracy", color="#2563eb", marker="o")
    axes[1].plot(epochs, history["validation_accuracy"], label="Validation accuracy", color="#ea580c", marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="TextCNN Accuracy")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    figure.tight_layout()
    figure.savefig(results_dir / "training_curves.png", dpi=160)
    plt.close(figure)

    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axis)
    axis.set(xlabel="Predicted Label", ylabel="Actual Label", title="TextCNN Test Confusion Matrix")
    figure.tight_layout()
    figure.savefig(results_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="/Users/govindprasadshaw/Downloads/DATASETS/IMDB Dataset.csv", help="Dataset CSV")
    parser.add_argument("--text-column", default="review", help="Text column name")
    parser.add_argument("--label-column", default="sentiment", help="Label column name")
    parser.add_argument("--output-dir", default="outputs/textcnn", help="Target output directory")
    parser.add_argument("--model-name", default="textcnn.pt", help="Checkpoint filename")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--max-vocab-size", type=int, default=20000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-filters", type=int, default=64)
    parser.add_argument("--kernel-sizes", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using compute device: {device}")

    output_dir = Path(args.output_dir)
    artifacts_dir, results_dir = output_dir / "artifacts", output_dir / "results"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading and splitting dataset: {args.csv}")
    prepared = load_and_split_data(args.csv, args.seed, args.text_column, args.label_column)
    print(f"Dataset split sizes: Train={len(prepared.train_labels):,}, "
          f"Validation={len(prepared.validation_labels):,}, Test={len(prepared.test_labels):,}")
    print(f"Classes: {prepared.class_names}")

    vocabulary = build_vocabulary(prepared.train_tokens, args.min_frequency, args.max_vocab_size)
    print(f"Fitted vocabulary size: {len(vocabulary):,} tokens")

    train_dataset = SentimentDataset(prepared.train_tokens, prepared.train_labels, vocabulary, args.max_length)
    validation_dataset = SentimentDataset(prepared.validation_tokens, prepared.validation_labels, vocabulary, args.max_length)
    test_dataset = SentimentDataset(prepared.test_tokens, prepared.test_labels, vocabulary, args.max_length)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_config = {
        "vocab_size": len(vocabulary),
        "num_classes": len(prepared.class_names),
        "embedding_dim": args.embedding_dim,
        "num_filters": args.num_filters,
        "kernel_sizes": tuple(args.kernel_sizes),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "padding_idx": vocabulary["<PAD>"],
    }
    model = TextCNNClassifier(**model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TextCNN Trainable Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    best_state, best_validation_loss, remaining_patience = None, float("inf"), args.patience

    start_train_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_labels, train_predictions = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, validation_labels, validation_predictions = run_epoch(model, validation_loader, criterion, device)
        train_metrics = metrics_dict(train_labels, train_predictions)
        validation_metrics = metrics_dict(validation_labels, validation_predictions)

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["validation_accuracy"].append(validation_metrics["accuracy"])

        epoch_duration = time.perf_counter() - epoch_start
        print(f"Epoch {epoch:02d} ({epoch_duration:.1f}s) | "
              f"Train Loss {train_loss:.4f}, Acc {train_metrics['accuracy']:.4f} | "
              f"Val Loss {validation_loss:.4f}, Acc {validation_metrics['accuracy']:.4f}")

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            remaining_patience = args.patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                print("Early stopping triggered by validation loss.")
                break

    total_training_time = time.perf_counter() - start_train_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    model.load_state_dict(best_state)

    # Held-out Test Set Evaluation
    test_start = time.perf_counter()
    test_loss, test_labels, test_predictions = run_epoch(model, test_loader, criterion, device)
    test_eval_duration = time.perf_counter() - test_start

    test_metrics = metrics_dict(test_labels, test_predictions)
    report = classification_report(
        test_labels, test_predictions, target_names=prepared.class_names, zero_division=0, output_dict=True
    )

    # Save Checkpoint
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model_config,
        "class_names": prepared.class_names,
        "max_length": args.max_length,
        "training_time_seconds": round(total_training_time, 2),
        "total_parameters": total_params,
    }
    torch.save(checkpoint, artifacts_dir / args.model_name)
    (artifacts_dir / "vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")

    # Save Evaluation
    results = {
        "model": "TextCNN",
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "classification_report": report,
        "class_names": prepared.class_names,
        "split_sizes": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
            "test": len(test_dataset),
        },
        "device": str(device),
        "history": history,
        "training_time_seconds": round(total_training_time, 2),
        "test_inference_time_seconds": round(test_eval_duration, 4),
        "total_parameters": total_params,
        "model_size_bytes": (artifacts_dir / args.model_name).stat().st_size,
    }
    (results_dir / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_plots(history, test_labels, test_predictions, prepared.class_names, results_dir)

    print("\n" + "=" * 50)
    print("TextCNN Held-Out Test Evaluation Results:")
    print("=" * 50)
    print(f"Accuracy:          {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Weighted F1:       {test_metrics['f1_weighted'] * 100:.2f}%")
    print(f"Macro F1:          {test_metrics['f1_macro'] * 100:.2f}%")
    print(f"Weighted Precision:{test_metrics['precision_weighted'] * 100:.2f}%")
    print(f"Weighted Recall:   {test_metrics['recall_weighted'] * 100:.2f}%")
    print(f"Training Time:     {total_training_time:.2f}s")
    print(f"Inference Latency: {(test_eval_duration / len(test_dataset)) * 1000:.3f} ms / sample")
    print(f"Saved Checkpoint:  {artifacts_dir / args.model_name}")
    print(f"Saved Evaluation:  {results_dir / 'evaluation.json'}")
    print("=" * 50)


if __name__ == "__main__":
    main()
