"""Download and prepare exact-match GoEmotions labels for SentixAI."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, load_dataset

# These names are the original GoEmotions annotation labels, not inferred labels.
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]
TARGET_EMOTIONS = {"joy", "anger", "sadness", "fear", "surprise", "love", "disappointment"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/goemotions_target_emotions.csv")
    args = parser.parse_args()

    dataset = load_dataset("AiLab-IMCS-UL/go_emotions-en")
    all_splits = concatenate_datasets([dataset[split] for split in dataset.keys()])
    records = []
    for row in all_splits:
        labels = row["labels"]
        # Keep only unambiguous, single-label examples with an exact requested label.
        if len(labels) != 1:
            continue
        emotion = GOEMOTIONS_LABELS[labels[0]]
        if emotion in TARGET_EMOTIONS:
            records.append({"text": row["text"], "emotion": emotion, "source_label": emotion})

    prepared = pd.DataFrame(records).dropna().drop_duplicates().reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False)
    print(f"Saved {len(prepared):,} examples to {output_path}")
    print("Classes:", ", ".join(sorted(prepared["emotion"].unique())))
    print(prepared["emotion"].value_counts().sort_index().to_string())
    print("Frustration is intentionally not included: GoEmotions has no source label named 'frustration'.")


if __name__ == "__main__":
    main()
