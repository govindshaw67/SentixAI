"""Train and evaluate the SentixAI BiLSTM-attention sentiment classifier."""

from __future__ import annotations

import argparse
import copy
import json
import random
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
from sentixai_model import BiLSTMAttentionClassifier


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
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }


def save_plots(history, labels, predictions, class_names, results_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, history["train_loss"], label="Training loss")
    axes[0].plot(epochs, history["validation_loss"], label="Validation loss")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="Training accuracy")
    axes[1].plot(epochs, history["validation_accuracy"], label="Validation accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(results_dir / "training_curves.png", dpi=160)
    plt.close(figure)

    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axis)
    axis.set(xlabel="Predicted", ylabel="Actual", title="Test-set confusion matrix")
    figure.tight_layout()
    figure.savefig(results_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="CSV file for this classification task")
    parser.add_argument("--text-column", default="review", help="Text column name")
    parser.add_argument("--label-column", default="sentiment", help="Class-label column name")
    parser.add_argument("--output-dir", default="outputs", help="Directory for model artifacts and results")
    parser.add_argument("--model-name", default="bilstm_attention.pt", help="Checkpoint filename")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=300)
    parser.add_argument("--max-vocab-size", type=int, default=30000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    artifacts_dir, results_dir = output_dir / "artifacts", output_dir / "results"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    prepared = load_and_split_data(args.csv, args.seed, args.text_column, args.label_column)
    vocabulary = build_vocabulary(prepared.train_tokens, args.min_frequency, args.max_vocab_size)
    train_dataset = SentimentDataset(prepared.train_tokens, prepared.train_labels, vocabulary, args.max_length)
    validation_dataset = SentimentDataset(prepared.validation_tokens, prepared.validation_labels, vocabulary, args.max_length)
    test_dataset = SentimentDataset(prepared.test_tokens, prepared.test_labels, vocabulary, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_config = {
        "vocab_size": len(vocabulary), "num_classes": len(prepared.class_names),
        "embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim,
        "dropout": args.dropout, "padding_idx": vocabulary["<PAD>"],
    }
    model = BiLSTMAttentionClassifier(**model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = {"train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": []}
    best_state, best_validation_loss, remaining_patience = None, float("inf"), args.patience
    for epoch in range(1, args.epochs + 1):
        train_loss, train_labels, train_predictions = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, validation_labels, validation_predictions = run_epoch(model, validation_loader, criterion, device)
        train_metrics = metrics_dict(train_labels, train_predictions)
        validation_metrics = metrics_dict(validation_labels, validation_predictions)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["validation_accuracy"].append(validation_metrics["accuracy"])
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f}, accuracy {train_metrics['accuracy']:.4f} | "
              f"validation loss {validation_loss:.4f}, accuracy {validation_metrics['accuracy']:.4f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            remaining_patience = args.patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    test_loss, test_labels, test_predictions = run_epoch(model, test_loader, criterion, device)
    test_metrics = metrics_dict(test_labels, test_predictions)
    report = classification_report(test_labels, test_predictions, target_names=prepared.class_names, zero_division=0, output_dict=True)
    checkpoint = {"model_state": model.state_dict(), "model_config": model_config,
                  "class_names": prepared.class_names, "max_length": args.max_length}
    torch.save(checkpoint, artifacts_dir / args.model_name)
    (artifacts_dir / "vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")
    results = {"test_loss": test_loss, "test_metrics": test_metrics, "classification_report": report,
               "class_names": prepared.class_names, "split_sizes": {"train": len(train_dataset), "validation": len(validation_dataset), "test": len(test_dataset)},
               "device": str(device), "history": history}
    (results_dir / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_plots(history, test_labels, test_predictions, prepared.class_names, results_dir)
    print("\nTest results:")
    print(json.dumps({"loss": test_loss, **test_metrics}, indent=2))
    print(classification_report(test_labels, test_predictions, target_names=prepared.class_names, zero_division=0))
    print(f"Saved model to {artifacts_dir / args.model_name}")


if __name__ == "__main__":
    main()
