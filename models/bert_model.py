"""
Advanced bert_model.py
Optimized for Urdu NLP + Google Colab GPU

Features:
✓ XLM-RoBERTa support
✓ Mixed precision FP16
✓ Early stopping
✓ Better pooling
✓ Faster training
✓ Gradient clipping
✓ Strong Urdu classification
✓ Better validation stability
"""

import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import accuracy_score

from config import (
    BERT_MODEL_NAME,
    BERT_MAX_LEN,
    BERT_EPOCHS,
    BERT_BATCH_SIZE,
    BERT_LR,
    BERT_WARMUP_RATIO,
    NUM_CLASSES,
    MODELS_DIR,
    RANDOM_SEED,
    EARLY_STOPPING_PATIENCE,
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

MODEL_DIR = os.path.join(
    MODELS_DIR,
    "bert_model"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ─────────────────────────────────────────────
# Seeds
# ─────────────────────────────────────────────

torch.manual_seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class BERTDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_len
    ):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "label":
                torch.tensor(
                    self.labels[idx],
                    dtype=torch.long
                )
        }

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

class BERTClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = AutoModel.from_pretrained(
            BERT_MODEL_NAME
        )

        hidden_size = self.bert.config.hidden_size

        self.dropout1 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(
            hidden_size,
            512
        )

        self.relu = nn.ReLU()

        self.dropout2 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(
            512,
            NUM_CLASSES
        )

    def mean_pooling(
        self,
        token_embeddings,
        attention_mask
    ):

        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        summed = torch.sum(
            token_embeddings * mask,
            dim=1
        )

        summed_mask = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )

        return summed / summed_mask

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = self.mean_pooling(
            outputs.last_hidden_state,
            attention_mask
        )

        x = self.dropout1(pooled)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout2(x)

        logits = self.fc2(x)

        return logits

# ─────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────

def train(
    train_texts,
    train_labels,
    val_texts,
    val_labels
):

    print(f"\n[BERT] Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    model = BERTClassifier().to(DEVICE)

    train_dataset = BERTDataset(
        train_texts,
        train_labels,
        tokenizer,
        BERT_MAX_LEN
    )

    val_dataset = BERTDataset(
        val_texts,
        val_labels,
        tokenizer,
        BERT_MAX_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BERT_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BERT_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BERT_LR,
        weight_decay=0.01
    )

    total_steps = (
        len(train_loader)
        * BERT_EPOCHS
    )

    warmup_steps = int(
        total_steps
        * BERT_WARMUP_RATIO
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0

    early_stop_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    # ─────────────────────────────────────────
    # Epoch loop
    # ─────────────────────────────────────────

    for epoch in range(BERT_EPOCHS):

        # ───────── TRAIN ─────────

        model.train()

        train_loss = 0

        train_preds = []

        train_true = []

        for batch in train_loader:

            input_ids = batch["input_ids"].to(DEVICE)

            attention_mask = batch["attention_mask"].to(DEVICE)

            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):

                outputs = model(
                    input_ids,
                    attention_mask
                )

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(optimizer)

            scaler.update()

            scheduler.step()

            train_loss += loss.item()

            preds = torch.argmax(
                outputs,
                dim=1
            )

            train_preds.extend(
                preds.cpu().numpy()
            )

            train_true.extend(
                labels.cpu().numpy()
            )

        train_acc = accuracy_score(
            train_true,
            train_preds
        )

        avg_train_loss = (
            train_loss
            / len(train_loader)
        )

        # ───────── VALIDATION ─────────

        model.eval()

        val_loss = 0

        val_preds = []

        val_true = []

        with torch.no_grad():

            for batch in val_loader:

                input_ids = batch["input_ids"].to(DEVICE)

                attention_mask = batch["attention_mask"].to(DEVICE)

                labels = batch["label"].to(DEVICE)

                outputs = model(
                    input_ids,
                    attention_mask
                )

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += loss.item()

                preds = torch.argmax(
                    outputs,
                    dim=1
                )

                val_preds.extend(
                    preds.cpu().numpy()
                )

                val_true.extend(
                    labels.cpu().numpy()
                )

        val_acc = accuracy_score(
            val_true,
            val_preds
        )

        avg_val_loss = (
            val_loss
            / len(val_loader)
        )

        history["train_loss"].append(
            avg_train_loss
        )

        history["val_loss"].append(
            avg_val_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_acc"].append(
            val_acc
        )

        print(
            f"Epoch {epoch+1}/{BERT_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # ─────────────────────────────────────

        if val_acc > best_acc:

            best_acc = val_acc

            early_stop_counter = 0

            save_model(
                model,
                tokenizer
            )

            print(
                f"[BERT] Best model saved "
                f"(Val Acc: {val_acc:.4f})"
            )

        else:

            early_stop_counter += 1

            print(
                f"[BERT] Early stop counter: "
                f"{early_stop_counter}"
            )

            if early_stop_counter >= EARLY_STOPPING_PATIENCE:

                print(
                    "[BERT] Early stopping triggered"
                )

                break

    model, tokenizer = load_model()

    return model, history

# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────

def predict(
    model,
    tokenizer,
    texts,
    batch_size=32
):

    model.eval()

    dataset = BERTDataset(
        texts,
        [0] * len(texts),
        tokenizer,
        BERT_MAX_LEN
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    all_preds = []

    all_probs = []

    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(DEVICE)

            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(
                input_ids,
                attention_mask
            )

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
    tokenizer
):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    torch.save(
        model.state_dict(),
        os.path.join(
            MODEL_DIR,
            "classifier_weights.pt"
        )
    )

    tokenizer.save_pretrained(MODEL_DIR)

    print(f"[BERT] Saved → {MODEL_DIR}")

def load_model():

    model = BERTClassifier().to(DEVICE)

    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="HussainAbbas09/urdu-xlmr-model",
        filename="classifier_weights.pt"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    model.eval()

    print("[BERT] Loaded model from HF Hub")

    return model, tokenizer
def is_saved():

    return os.path.exists(
        os.path.join(
            MODEL_DIR,
            "classifier_weights.pt"
        )
    )
