"""
config.py

Central configuration for the Urdu Text-to-Image Classification project.

Optimized for:
✔ New Group140 dataset
✔ Urdu NLP
✔ XLM-RoBERTa
✔ Colab GPU training
✔ HuggingFace Spaces deployment
"""

import os
import torch

# ─────────────────────────────────────────────
# Dataset Path
# ─────────────────────────────────────────────

# IMPORTANT:
# Must exactly match uploaded CSV filename

DATA_PATH = "dataset/urdu_text_to_image_dataset_group140.csv"

# ─────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────

MODELS_DIR = "saved_models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────
# Random Seed
# ─────────────────────────────────────────────

RANDOM_SEED = 42

# ─────────────────────────────────────────────
# Train / Validation / Test Split
# ─────────────────────────────────────────────

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ─────────────────────────────────────────────
# Dataset Categories
# MUST EXACTLY MATCH CSV LABELS
# ─────────────────────────────────────────────

CATEGORIES = [
    "animals",
    "nature",
    "technology"
]

NUM_CLASSES = len(CATEGORIES)

LABEL2ID = {
    label: idx
    for idx, label in enumerate(CATEGORIES)
}

ID2LABEL = {
    idx: label
    for label, idx in LABEL2ID.items()
}

# ─────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────

# Larger context for Urdu sentences
MAX_TOKENS = 256

# Larger vocabulary for LSTM
VOCAB_SIZE = 50000

# ─────────────────────────────────────────────
# TF-IDF + Logistic Regression
# ─────────────────────────────────────────────

TFIDF_MAX_FEATURES = 100000

# Better Urdu phrase understanding
TFIDF_NGRAM_RANGE = (1, 3)

LR_C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
]

LR_MAX_ITER = 5000

# ─────────────────────────────────────────────
# BiLSTM
# ─────────────────────────────────────────────

EMBEDDING_DIM = 300
HIDDEN_DIM    = 512

LSTM_LAYERS  = 2
LSTM_DROPOUT = 0.4

LSTM_EPOCHS     = 20
LSTM_BATCH_SIZE = 64

LSTM_LR = 1e-3

# ─────────────────────────────────────────────
# XLM-RoBERTa
# ─────────────────────────────────────────────

# Best multilingual model for Urdu
BERT_MODEL_NAME = "xlm-roberta-base"

BERT_MAX_LEN = 256

BERT_EPOCHS = 5

# Reduce if GPU memory issue
BERT_BATCH_SIZE = 16

BERT_LR = 2e-5

BERT_WARMUP_RATIO = 0.1

# ─────────────────────────────────────────────
# Training Controls
# ─────────────────────────────────────────────

EARLY_STOPPING_PATIENCE = 3

GRADIENT_CLIP_VALUE = 1.0

# ─────────────────────────────────────────────
# Optional Debug
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("CONFIG LOADED")
    print("=" * 60)

    print(f"Dataset Path : {DATA_PATH}")
    print(f"Device       : {DEVICE}")
    print(f"Classes      : {NUM_CLASSES}")
    print(f"Vocabulary   : {VOCAB_SIZE}")
    print(f"Max Tokens   : {MAX_TOKENS}")
    print(f"BERT Model   : {BERT_MODEL_NAME}")

    print("\nCategories:")
    for i, category in enumerate(CATEGORIES):
        print(f"{i}: {category}")

    print("\nDone ✓")