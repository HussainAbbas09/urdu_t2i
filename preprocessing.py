"""
Improved preprocessing pipeline for Urdu text classification.

Optimized for:
✔ Large Urdu vocabulary
✔ Better semantic retention
✔ Better LSTM performance
✔ Better XLM-R training
✔ Cleaner normalization
✔ HuggingFace / Colab compatibility
"""

import re
import os
import pickle
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    MAX_TOKENS,
    VOCAB_SIZE,
    LABEL2ID,
    MODELS_DIR,
)

# ─────────────────────────────────────────────
# Unicode normalization
# ─────────────────────────────────────────────

NORM_MAP = str.maketrans({

    "ﻱ": "ی",
    "ﻰ": "ی",
    "ي": "ی",

    "ك": "ک",

    "ة": "ت",

    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",

    "ؤ": "و",
    "ئ": "ی",
})

URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+"
)

MULTI_SPACE_PAT = re.compile(r"\s+")

# Keep Urdu + English + digits
VALID_TEXT_PATTERN = re.compile(
    r"[^\u0600-\u06FFa-zA-Z0-9\s]"
)

# ─────────────────────────────────────────────
# Basic cleaning helpers
# ─────────────────────────────────────────────

def normalize_urdu(text: str) -> str:

    return text.translate(NORM_MAP)


def remove_urls(text: str) -> str:

    return URL_PATTERN.sub(" ", text)


def clean_whitespace(text: str) -> str:

    return MULTI_SPACE_PAT.sub(
        " ",
        text
    ).strip()

# ─────────────────────────────────────────────
# Main preprocessing
# ─────────────────────────────────────────────

def preprocess_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    # lowercase English only
    text = text.lower()

    # remove urls
    text = remove_urls(text)

    # normalize unicode
    text = normalize_urdu(text)

    # remove unwanted symbols
    text = VALID_TEXT_PATTERN.sub(
        " ",
        text
    )

    # clean spaces
    text = clean_whitespace(text)

    return text

# ─────────────────────────────────────────────
# Dataset Loading
# ─────────────────────────────────────────────

def load_dataset(
    path: str = DATA_PATH
) -> pd.DataFrame:

    df = pd.read_csv(path)

    print(
        f"[INFO] Loaded {len(df):,} rows "
        f"from '{path}'"
    )

    # ── Required columns ──

    assert "text" in df.columns, \
        "Missing 'text' column"

    assert "label" in df.columns, \
        "Missing 'label' column"

    # ── Remove nulls ──

    df = df.dropna(
        subset=["text", "label"]
    )

    # ── Remove empty text ──

    df = df[
        df["text"].astype(str).str.strip() != ""
    ]

    df = df.reset_index(drop=True)

    print("[INFO] Applying preprocessing...")

    # ── Clean text ──

    df["clean_text"] = df["text"].apply(
        preprocess_text
    )

    # ── Encode labels ──

    df["label_id"] = df["label"].map(
        LABEL2ID
    )

    # remove unknown labels
    df = df.dropna(subset=["label_id"])

    df["label_id"] = df["label_id"].astype(int)

    print(
        f"[INFO] {len(df):,} rows after cleaning"
    )

    print(
        "\n[INFO] Category distribution:\n"
    )

    print(
        df["label"]
        .value_counts()
        .to_string()
    )

    return df

# ─────────────────────────────────────────────
# Train / Validation / Test Split
# ─────────────────────────────────────────────

def split_data(df: pd.DataFrame):

    test_size = TEST_RATIO

    val_size = VAL_RATIO / (
        TRAIN_RATIO + VAL_RATIO
    )

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label_id"],
        random_state=RANDOM_SEED,
    )

    train, val = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val["label_id"],
        random_state=RANDOM_SEED,
    )

    print(
        f"\n[INFO] Split → "
        f"Train: {len(train):,} | "
        f"Val: {len(val):,} | "
        f"Test: {len(test):,}"
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )

# ─────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def build_vocab(
    texts,
    max_vocab: int = VOCAB_SIZE
):

    counter = Counter()

    for text in texts:

        counter.update(
            text.split()
        )

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }

    for word, _ in counter.most_common(
        max_vocab - 2
    ):
        vocab[word] = len(vocab)

    print(
        f"[INFO] Vocabulary size: "
        f"{len(vocab):,} tokens"
    )

    return vocab

# ─────────────────────────────────────────────
# Text → Token IDs
# ─────────────────────────────────────────────

def text_to_ids(
    text: str,
    vocab: dict,
    max_len: int = MAX_TOKENS
):

    unk_id = vocab[UNK_TOKEN]
    pad_id = vocab[PAD_TOKEN]

    tokens = text.split()[:max_len]

    ids = [
        vocab.get(token, unk_id)
        for token in tokens
    ]

    # padding
    ids += [pad_id] * (
        max_len - len(ids)
    )

    return ids


def encode_dataset(
    texts,
    vocab,
    max_len: int = MAX_TOKENS
):

    return np.array(

        [
            text_to_ids(
                text,
                vocab,
                max_len
            )
            for text in texts
        ],

        dtype=np.int64
    )

# ─────────────────────────────────────────────
# Save / Load Vocabulary
# ─────────────────────────────────────────────

def save_vocab(
    vocab: dict,
    path: str = None
):

    if path is None:

        path = os.path.join(
            MODELS_DIR,
            "vocab.pkl"
        )

    with open(path, "wb") as f:

        pickle.dump(vocab, f)

    print(
        f"[INFO] Vocabulary saved → {path}"
    )


def load_vocab(
    path: str = None
):

    if path is None:

        path = os.path.join(
            MODELS_DIR,
            "vocab.pkl"
        )

    with open(path, "rb") as f:

        vocab = pickle.load(f)

    print(
        f"[INFO] Vocabulary loaded "
        f"({len(vocab):,} tokens)"
    )

    return vocab

# ─────────────────────────────────────────────
# EDA
# ─────────────────────────────────────────────

def compute_eda_stats(
    df: pd.DataFrame
):

    token_lengths = df["clean_text"].apply(
        lambda x: len(x.split())
    )

    return {

        "total_samples":
            len(df),

        "num_classes":
            df["label"].nunique(),

        "class_distribution":
            df["label"]
            .value_counts()
            .to_dict(),

        "avg_token_length":
            round(
                token_lengths.mean(),
                2
            ),

        "median_token_length":
            int(
                token_lengths.median()
            ),

        "max_token_length":
            int(
                token_lengths.max()
            ),

        "min_token_length":
            int(
                token_lengths.min()
            ),
    }

# ─────────────────────────────────────────────
# Quick Self Test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("URDU DATASET PREPROCESSING")
    print("=" * 60)

    df = load_dataset()

    stats = compute_eda_stats(df)

    print("\n── DATASET STATS ──")

    for k, v in stats.items():

        print(f"{k}: {v}")

    train, val, test = split_data(df)

    vocab = build_vocab(
        train["clean_text"].tolist()
    )

    save_vocab(vocab)

    sample = encode_dataset(
        train["clean_text"].tolist()[:5],
        vocab
    )

    print(
        f"\nEncoded shape: {sample.shape}"
    )

    print("\nDone ✓")