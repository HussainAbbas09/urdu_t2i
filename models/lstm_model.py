"""
Improved lstm_model.py
Optimized for Urdu NLP classification
Supports large Urdu vocabulary datasets
"""

import os
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    VOCAB_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    LSTM_LAYERS,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_LR,
    NUM_CLASSES,
    MODELS_DIR,
    RANDOM_SEED,
    GRADIENT_CLIP_VALUE,
    EARLY_STOPPING_PATIENCE
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "lstm_model.pt"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class UrduDataset(Dataset):

    def __init__(self, sequences, labels):

        self.sequences = torch.tensor(
            sequences,
            dtype=torch.long
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        return (
            self.sequences[idx],
            self.labels[idx]
        )


# ─────────────────────────────────────────────
# Attention Layer
# ─────────────────────────────────────────────

class Attention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attention = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, outputs):

        weights = self.attention(outputs)

        weights = torch.softmax(
            weights,
            dim=1
        )

        context = torch.sum(
            weights * outputs,
            dim=1
        )

        return context


# ─────────────────────────────────────────────
# BiLSTM Model
# ─────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size=VOCAB_SIZE
    ):

        super().__init__()

        self.embedding = nn.Embedding(

            num_embeddings=vocab_size,

            embedding_dim=EMBEDDING_DIM,

            padding_idx=0
        )

        self.embedding_dropout = nn.Dropout(0.2)

        self.lstm = nn.LSTM(

            input_size=EMBEDDING_DIM,

            hidden_size=HIDDEN_DIM,

            num_layers=LSTM_LAYERS,

            dropout=LSTM_DROPOUT,

            batch_first=True,

            bidirectional=True
        )

        self.attention = Attention(HIDDEN_DIM)

        self.dropout = nn.Dropout(
            LSTM_DROPOUT
        )

        self.fc1 = nn.Linear(
            HIDDEN_DIM * 2,
            256
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            256,
            NUM_CLASSES
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.embedding_dropout(x)

        outputs, _ = self.lstm(x)

        context = self.attention(outputs)

        x = self.dropout(context)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        logits = self.fc2(x)

        return logits


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train(

    train_seqs,
    train_labels,

    val_seqs,
    val_labels,

    vocab_size=VOCAB_SIZE
):

    print(f"\n[LSTM] Training on: {DEVICE}")

    model = BiLSTMClassifier(
        vocab_size=vocab_size
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LSTM_LR,

        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()

    scheduler = ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=1
    )

    train_loader = DataLoader(

        UrduDataset(
            train_seqs,
            train_labels
        ),

        batch_size=LSTM_BATCH_SIZE,

        shuffle=True,

        pin_memory=True
    )

    val_loader = DataLoader(

        UrduDataset(
            val_seqs,
            val_labels
        ),

        batch_size=LSTM_BATCH_SIZE,

        shuffle=False,

        pin_memory=True
    )

    history = {

        "train_loss": [],
        "val_loss": [],

        "train_acc": [],
        "val_acc": []
    }

    best_acc = 0
    patience_counter = 0

    # ─────────────────────────────────────────
    # Epoch loop
    # ─────────────────────────────────────────

    for epoch in range(LSTM_EPOCHS):

        # ───────────── TRAIN ─────────────

        model.train()

        total_loss = 0

        correct = 0

        total = 0

        for seqs, labels in train_loader:

            seqs = seqs.to(DEVICE)

            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(seqs)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                GRADIENT_CLIP_VALUE
            )

            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

        train_acc = correct / total

        train_loss = (
            total_loss / len(train_loader)
        )

        # ───────────── VALIDATION ─────────────

        model.eval()

        val_loss = 0

        val_correct = 0

        val_total = 0

        with torch.no_grad():

            for seqs, labels in val_loader:

                seqs = seqs.to(DEVICE)

                labels = labels.to(DEVICE)

                outputs = model(seqs)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += loss.item()

                preds = torch.argmax(
                    outputs,
                    dim=1
                )

                val_correct += (
                    preds == labels
                ).sum().item()

                val_total += labels.size(0)

        val_acc = val_correct / val_total

        avg_val_loss = (
            val_loss / len(val_loader)
        )

        scheduler.step(val_acc)

        history["train_loss"].append(
            round(train_loss, 4)
        )

        history["val_loss"].append(
            round(avg_val_loss, 4)
        )

        history["train_acc"].append(
            round(train_acc, 4)
        )

        history["val_acc"].append(
            round(val_acc, 4)
        )

        print(
            f"Epoch {epoch+1}/{LSTM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # ─────────────────────────────────────
        # Save best model
        # ─────────────────────────────────────

        if val_acc > best_acc:

            best_acc = val_acc

            patience_counter = 0

            save_model(model)

            print(
                f"[LSTM] Best model saved "
                f"(Val Acc: {val_acc:.4f})"
            )

        else:

            patience_counter += 1

            if (
                patience_counter
                >= EARLY_STOPPING_PATIENCE
            ):

                print(
                    "\n[LSTM] Early stopping triggered"
                )

                break

    model = load_model(
        vocab_size=vocab_size
    )

    return model, history


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────

def predict(

    model,
    sequences,

    batch_size=128
):

    model.eval()

    dataset = UrduDataset(

        sequences,

        np.zeros(
            len(sequences),
            dtype=np.int64
        )
    )

    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False
    )

    all_preds = []

    all_probs = []

    with torch.no_grad():

        for seqs, _ in loader:

            seqs = seqs.to(DEVICE)

            outputs = model(seqs)

            probs = torch.softmax(
                outputs,
                dim=1
            )

            preds = torch.argmax(
                probs,
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_probs.extend(
                probs.cpu().numpy()
            )

    return (

        np.array(all_preds),

        np.array(all_probs)
    )


# ─────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────

def save_model(

    model,

    path=MODEL_PATH
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    torch.save(

        model.state_dict(),

        path
    )

    print(f"[LSTM] Model saved → {path}")


def load_model(

    path=MODEL_PATH,

    vocab_size=VOCAB_SIZE
):

    model = BiLSTMClassifier(
        vocab_size=vocab_size
    ).to(DEVICE)

    model.load_state_dict(

        torch.load(
            path,
            map_location=DEVICE
        )
    )

    model.eval()

    print(f"[LSTM] Model loaded ← {path}")

    return model


def is_saved(path=MODEL_PATH):

    return os.path.exists(path)


# ─────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":

    sample_x = np.random.randint(
        0,
        1000,
        (8, 50)
    )

    sample_y = np.random.randint(
        0,
        NUM_CLASSES,
        8
    )

    model = BiLSTMClassifier(
        vocab_size=1000
    )

    outputs = model(
        torch.tensor(
            sample_x,
            dtype=torch.long
        )
    )

    print(outputs.shape)

    print("Done ✓")