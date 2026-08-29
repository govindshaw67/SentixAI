"""Dataset preparation, cleaning, vocabulary, and batching for SentixAI."""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")


def clean_and_tokenize(text: str) -> list[str]:
    """Reuse the useful parts of baseline cleaning without English stemming."""
    text = html.unescape(str(text)).lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    return TOKEN_PATTERN.findall(text)


def build_vocabulary(token_lists: list[list[str]], min_frequency: int, max_vocab_size: int) -> dict[str, int]:
    counts = Counter(token for tokens in token_lists for token in tokens)
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    candidates = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    for token, count in candidates:
        if count < min_frequency or len(vocabulary) >= max_vocab_size:
            continue
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_tokens(tokens: list[str], vocabulary: dict[str, int], max_length: int) -> list[int]:
    ids = [vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in tokens[:max_length]]
    return ids + [vocabulary[PAD_TOKEN]] * (max_length - len(ids))


class SentimentDataset(Dataset):
    def __init__(self, token_lists: list[list[str]], labels: list[int], vocabulary: dict[str, int], max_length: int) -> None:
        self.examples = [encode_tokens(tokens, vocabulary, max_length) for tokens in token_lists]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.examples[index], dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)


@dataclass
class PreparedData:
    train_tokens: list[list[str]]
    validation_tokens: list[list[str]]
    test_tokens: list[list[str]]
    train_labels: list[int]
    validation_labels: list[int]
    test_labels: list[int]
    class_names: list[str]


def load_and_split_data(
    csv_path: str,
    seed: int,
    text_column: str = "review",
    label_column: str = "sentiment",
) -> PreparedData:
    data = pd.read_csv(csv_path)
    required_columns = {text_column, label_column}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"CSV must include columns {sorted(required_columns)}; found {data.columns.tolist()}.")

    data = data.loc[:, [text_column, label_column]].dropna().drop_duplicates().copy()
    data[label_column] = data[label_column].astype(str).str.strip().str.lower()
    class_names = sorted(data[label_column].unique().tolist())
    if len(class_names) < 2:
        raise ValueError("At least two sentiment classes are required.")

    label_to_id = {label: index for index, label in enumerate(class_names)}
    data["label"] = data[label_column].map(label_to_id)
    data["tokens"] = data[text_column].map(clean_and_tokenize)
    data = data.loc[data["tokens"].map(bool)].reset_index(drop=True)

    train_data, holdout_data = train_test_split(
        data, test_size=0.20, random_state=seed, stratify=data["label"]
    )
    validation_data, test_data = train_test_split(
        holdout_data, test_size=0.50, random_state=seed, stratify=holdout_data["label"]
    )

    def values(frame: pd.DataFrame) -> tuple[list[list[str]], list[int]]:
        return frame["tokens"].tolist(), frame["label"].astype(int).tolist()

    train_tokens, train_labels = values(train_data)
    validation_tokens, validation_labels = values(validation_data)
    test_tokens, test_labels = values(test_data)
    return PreparedData(
        train_tokens, validation_tokens, test_tokens,
        train_labels, validation_labels, test_labels, class_names,
    )
