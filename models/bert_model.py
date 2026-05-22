def load_model():

    model = BERTClassifier().to(DEVICE)

    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id="HussainAbbas09/urdu-xlmr-classifier"
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

    return True
