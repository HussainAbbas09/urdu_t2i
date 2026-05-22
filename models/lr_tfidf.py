"""
Improved lr_tfidf.py
Better optimized for Urdu NLP classification
"""

import os
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from config import (
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    LR_C_VALUES,
    LR_MAX_ITER,
    MODELS_DIR,
    RANDOM_SEED
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "lr_tfidf_model.pkl"
)

# ─────────────────────────────────────────────
# Build Pipeline
# ─────────────────────────────────────────────

def build_pipeline():

    pipeline = Pipeline([

        (
            "tfidf",

            TfidfVectorizer(

                max_features=TFIDF_MAX_FEATURES,

                ngram_range=TFIDF_NGRAM_RANGE,

                analyzer="word",

                token_pattern=r"(?u)\b\w+\b",

                lowercase=False,

                strip_accents=None,

                sublinear_tf=True,

                min_df=1,

                max_df=0.98,

                norm="l2",

                smooth_idf=True,

                use_idf=True,
            )
        ),

        (
            "clf",

            LogisticRegression(

                max_iter=LR_MAX_ITER,

                C=1.0,

                solver="saga",

                random_state=RANDOM_SEED,

                n_jobs=-1,

                class_weight="balanced",

                multi_class="multinomial",

                verbose=0,
            )
        )
    ])

    return pipeline


# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────

def train(

    train_texts,
    train_labels,

    val_texts,
    val_labels
):

    print("\n[LR] Starting training...")

    best_acc = 0
    best_c = None
    best_model = None

    for c_value in LR_C_VALUES:

        print(f"\n[LR] Testing C = {c_value}")

        model = build_pipeline()

        model.set_params(
            clf__C=c_value
        )

        model.fit(
            train_texts,
            train_labels
        )

        preds = model.predict(val_texts)

        acc = accuracy_score(
            val_labels,
            preds
        )

        print(f"[LR] Validation Accuracy: {acc:.4f}")

        if acc > best_acc:

            best_acc = acc

            best_c = c_value

            best_model = model

    print("\n[LR] Best Hyperparameter")
    print(f"[LR] Best C = {best_c}")
    print(f"[LR] Best Accuracy = {best_acc:.4f}")

    # ─────────────────────────────────────────
    # Final training on train + validation
    # ─────────────────────────────────────────

    print("\n[LR] Re-training on Train + Validation")

    final_texts = (
        list(train_texts)
        + list(val_texts)
    )

    final_labels = (
        list(train_labels)
        + list(val_labels)
    )

    final_model = build_pipeline()

    final_model.set_params(
        clf__C=best_c
    )

    final_model.fit(
        final_texts,
        final_labels
    )

    print("[LR] Final model trained ✓")

    return final_model


# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────

def predict(model, texts):

    preds = model.predict(texts)

    probs = model.predict_proba(texts)

    return (
        np.array(preds),
        np.array(probs)
    )


# ─────────────────────────────────────────────
# Save model
# ─────────────────────────────────────────────

def save_model(

    model,
    path=MODEL_PATH
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(path, "wb") as f:

        pickle.dump(model, f)

    print(f"[LR] Model saved → {path}")


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────

def load_model(path=MODEL_PATH):

    with open(path, "rb") as f:

        model = pickle.load(f)

    print(f"[LR] Model loaded ← {path}")

    return model


# ─────────────────────────────────────────────
# Check if saved
# ─────────────────────────────────────────────

def is_saved(path=MODEL_PATH):

    return os.path.exists(path)


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    sample_texts = [

        "خوبصورت پہاڑوں کا منظر",

        "جدید کمپیوٹر اور ٹیکنالوجی",

        "فٹبال کھیلتا ہوا کھلاڑی",

        "مزیدار پاکستانی کھانا",

        "قدرتی جنگل اور درخت"
    ]

    sample_labels = [0, 1, 2, 3, 4]

    model = build_pipeline()

    model.fit(
        sample_texts,
        sample_labels
    )

    preds, probs = predict(
        model,
        sample_texts
    )

    print("\nPredictions:")
    print(preds)

    print("\nProbabilities:")
    print(probs)

    print("\nDone ✓")