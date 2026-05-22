"""
app.py — Hugging Face Spaces Deployment
Urdu Text Classification using:
• Logistic Regression + TF-IDF
• BiLSTM + Attention
• XLM-RoBERTa

Optimized for:
✓ Hugging Face Spaces
✓ Gradio 4+
✓ CPU deployment
✓ Stable startup
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Core Imports
# ─────────────────────────────────────────────

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import gradio as gr

# ─────────────────────────────────────────────
# Project Path
# ─────────────────────────────────────────────

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# ─────────────────────────────────────────────
# Config Imports
# ─────────────────────────────────────────────

from config import (
    CATEGORIES,
    NUM_CLASSES,
    ID2LABEL,
    MODELS_DIR
)

from preprocessing import (
    preprocess_text,
    load_vocab,
    text_to_ids
)

# ─────────────────────────────────────────────
# Global Model Storage
# ─────────────────────────────────────────────

loaded = {}

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────

DEVICE = "cpu"

# ─────────────────────────────────────────────
# Load Logistic Regression
# ─────────────────────────────────────────────

def try_load_lr():

    try:

        from models import lr_tfidf

        if lr_tfidf.is_saved():

            loaded["lr"] = lr_tfidf.load_model()

            print("[APP] LR model loaded ✓")

        else:

            print("[APP] LR model NOT found")

    except Exception as e:

        print(f"[APP] LR loading failed: {e}")

# ─────────────────────────────────────────────
# Load BiLSTM
# ─────────────────────────────────────────────

def try_load_lstm():

    try:

        from models import lstm_model

        vocab_path = os.path.join(
            MODELS_DIR,
            "vocab.pkl"
        )

        if lstm_model.is_saved() and os.path.exists(vocab_path):

            vocab = load_vocab()

            model = lstm_model.load_model(
                vocab_size=len(vocab)
            )

            loaded["lstm"] = model
            loaded["vocab"] = vocab

            print("[APP] BiLSTM model loaded ✓")

        else:

            print("[APP] BiLSTM model NOT found")

    except Exception as e:

        print(f"[APP] BiLSTM loading failed: {e}")

# ─────────────────────────────────────────────
# Load XLM-R
# ─────────────────────────────────────────────

def try_load_bert():

    try:

        from models import bert_model

        if bert_model.is_saved():

            model, tokenizer = bert_model.load_model()

            loaded["bert"] = model
            loaded["bert_tok"] = tokenizer

            print("[APP] XLM-R model loaded ✓")

        else:

            print("[APP] XLM-R model NOT found")

    except Exception as e:

        print(f"[APP] XLM-R loading failed: {e}")

# ─────────────────────────────────────────────
# Load All Models
# ─────────────────────────────────────────────

print("\n[INFO] Loading models...\n")

try_load_lr()
try_load_lstm()
try_load_bert()

print("\n[INFO] Startup complete.\n")

# ─────────────────────────────────────────────
# Prediction Helpers
# ─────────────────────────────────────────────

def predict_lr(clean_text):

    if "lr" not in loaded:
        return None, None

    try:

        from models import lr_tfidf

        preds, probs = lr_tfidf.predict(
            loaded["lr"],
            [clean_text]
        )

        return int(preds[0]), probs[0].tolist()

    except Exception as e:

        print(f"LR prediction error: {e}")
        return None, None

# ─────────────────────────────────────────────

def predict_lstm(clean_text):

    if "lstm" not in loaded:
        return None, None

    try:

        from models import lstm_model

        vocab = loaded["vocab"]

        seq = np.array(
            [text_to_ids(clean_text, vocab)],
            dtype=np.int64
        )

        preds, probs = lstm_model.predict(
            loaded["lstm"],
            seq
        )

        return int(preds[0]), probs[0].tolist()

    except Exception as e:

        print(f"LSTM prediction error: {e}")
        return None, None

# ─────────────────────────────────────────────

def predict_bert(clean_text):

    if "bert" not in loaded:
        return None, None

    try:

        from models import bert_model

        preds, probs = bert_model.predict(
            loaded["bert"],
            loaded["bert_tok"],
            [clean_text]
        )

        return int(preds[0]), probs[0].tolist()

    except Exception as e:

        print(f"BERT prediction error: {e}")
        return None, None

# ─────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────

COLORS = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860"
]

def make_bar_chart(probs, model_name, pred_idx):

    fig, ax = plt.subplots(figsize=(6, 3.5))

    probs_percent = [p * 100 for p in probs]

    colors = [
        "#E74C3C" if i == pred_idx else COLORS[i % len(COLORS)]
        for i in range(len(probs))
    ]

    bars = ax.barh(
        CATEGORIES,
        probs_percent,
        color=colors
    )

    ax.set_xlim(0, 100)

    ax.set_xlabel("Confidence (%)")
    ax.set_title(model_name)

    ax.invert_yaxis()

    for bar, value in zip(bars, probs_percent):

        ax.text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center"
        )

    plt.tight_layout()

    return fig

# ─────────────────────────────────────────────
# Main Prediction
# ─────────────────────────────────────────────

def classify_urdu_text(text):

    if not text or not text.strip():

        empty = "⚠️ Please enter Urdu text"

        return (
            empty, "", None,
            empty, "", None,
            empty, "", None,
            ""
        )

    clean = preprocess_text(text)

    # LR
    lr_idx, lr_probs = predict_lr(clean)

    if lr_idx is not None:

        lr_label = f"🏷️ {ID2LABEL[lr_idx]}"
        lr_conf = f"{lr_probs[lr_idx] * 100:.1f}%"

        lr_chart = make_bar_chart(
            lr_probs,
            "LR + TF-IDF",
            lr_idx
        )

    else:

        lr_label = "❌ Model unavailable"
        lr_conf = ""
        lr_chart = None

    # LSTM
    lstm_idx, lstm_probs = predict_lstm(clean)

    if lstm_idx is not None:

        lstm_label = f"🏷️ {ID2LABEL[lstm_idx]}"
        lstm_conf = f"{lstm_probs[lstm_idx] * 100:.1f}%"

        lstm_chart = make_bar_chart(
            lstm_probs,
            "BiLSTM",
            lstm_idx
        )

    else:

        lstm_label = "❌ Model unavailable"
        lstm_conf = ""
        lstm_chart = None

    # BERT
    bert_idx, bert_probs = predict_bert(clean)

    if bert_idx is not None:

        bert_label = f"🏷️ {ID2LABEL[bert_idx]}"
        bert_conf = f"{bert_probs[bert_idx] * 100:.1f}%"

        bert_chart = make_bar_chart(
            bert_probs,
            "XLM-RoBERTa",
            bert_idx
        )

    else:

        bert_label = "❌ Model unavailable"
        bert_conf = ""
        bert_chart = None

    # Summary Table
    summary_html = f"""
    <table style='width:100%; border-collapse:collapse'>
        <tr style='background:#1f2937; color:white'>
            <th style='padding:10px'>Model</th>
            <th>Prediction</th>
            <th>Confidence</th>
        </tr>

        <tr>
            <td style='padding:8px'>LR + TF-IDF</td>
            <td>{lr_label}</td>
            <td>{lr_conf}</td>
        </tr>

        <tr>
            <td style='padding:8px'>BiLSTM</td>
            <td>{lstm_label}</td>
            <td>{lstm_conf}</td>
        </tr>

        <tr>
            <td style='padding:8px'>XLM-R</td>
            <td>{bert_label}</td>
            <td>{bert_conf}</td>
        </tr>
    </table>
    """

    return (
        lr_label,
        lr_conf,
        lr_chart,

        lstm_label,
        lstm_conf,
        lstm_chart,

        bert_label,
        bert_conf,
        bert_chart,

        summary_html
    )

# ─────────────────────────────────────────────
# Example Inputs
# ─────────────────────────────────────────────

EXAMPLES = [
    ["مسافر پہاڑوں میں ٹریکنگ کر رہا ہے"],
    ["استاد بچوں کو کلاس میں پڑھا رہا ہے"],
    ["کھلاڑی فٹبال کھیل رہا ہے"],
    ["مزیدار کھانا میز پر رکھا ہے"],
    ["شہر کی سڑکوں پر گاڑیاں چل رہی ہیں"],
    ["کمپیوٹر اور جدید ٹیکنالوجی"]
]

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

with gr.Blocks(
    title="Urdu Text Classification",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # 🖼️ Urdu Text Classification
        
        Compare predictions from:
        - Logistic Regression + TF-IDF
        - BiLSTM + Attention
        - XLM-RoBERTa
        """
    )

    with gr.Tab("Prediction"):

        input_text = gr.Textbox(
            label="Enter Urdu Text",
            lines=4,
            placeholder="یہاں اردو متن درج کریں..."
        )

        predict_btn = gr.Button(
            "🚀 Predict",
            variant="primary"
        )

        gr.Examples(
            examples=EXAMPLES,
            inputs=input_text
        )

        summary_out = gr.HTML()

        with gr.Row():

            with gr.Column():

                gr.Markdown("## LR + TF-IDF")

                lr_label = gr.Markdown()
                lr_conf = gr.Markdown()
                lr_plot = gr.Plot()

            with gr.Column():

                gr.Markdown("## BiLSTM")

                lstm_label = gr.Markdown()
                lstm_conf = gr.Markdown()
                lstm_plot = gr.Plot()

            with gr.Column():

                gr.Markdown("## XLM-R")

                bert_label = gr.Markdown()
                bert_conf = gr.Markdown()
                bert_plot = gr.Plot()

        predict_btn.click(
            fn=classify_urdu_text,
            inputs=input_text,
            outputs=[
                lr_label,
                lr_conf,
                lr_plot,

                lstm_label,
                lstm_conf,
                lstm_plot,

                bert_label,
                bert_conf,
                bert_plot,

                summary_out
            ]
        )

    with gr.Tab("Model Status"):

        status_html = "<h2>Loaded Models</h2><ul>"

        status_html += (
            "<li>✅ LR + TF-IDF</li>"
            if "lr" in loaded
            else "<li>❌ LR not loaded</li>"
        )

        status_html += (
            "<li>✅ BiLSTM</li>"
            if "lstm" in loaded
            else "<li>❌ BiLSTM not loaded</li>"
        )

        status_html += (
            "<li>✅ XLM-R</li>"
            if "bert" in loaded
            else "<li>❌ XLM-R not loaded</li>"
        )

        status_html += "</ul>"

        gr.HTML(status_html)

# ─────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )