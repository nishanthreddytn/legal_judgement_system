import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF


class HybridLegalNER(nn.Module):

    def __init__(
        self,
        base_model,
        num_labels,
        lstm_hidden_size=256,
        dropout=0.2,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(base_model)

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(
            lstm_hidden_size * 2,
            num_labels,
        )

        self.crf = CRF(
            num_labels,
            batch_first=True,
        )

    def forward(self, input_ids, attention_mask):

        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        x = bert_output.last_hidden_state

        x = self.dropout(x)

        x, _ = self.lstm(x)

        x = self.dropout(x)

        emissions = self.classifier(x)

        return emissions


def load_labels():

    path = Path("data/ner/label_config.json")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return config["labels"]


def main():

    base_model = "law-ai/InLegalBERT"

    checkpoint_path = Path(
        "models/legal_ner/checkpoints/checkpoint-7000/checkpoint.pt"
    )

    labels = load_labels()

    num_labels = len(labels)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print("Checkpoint:", checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model
    )

    print("Loading model...")

    model = HybridLegalNER(
        base_model=base_model,
        num_labels=num_labels,
        lstm_hidden_size=256,
        dropout=0.2,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")
    print()

    text = input(
        "Enter legal text: "
    ).strip()

    if not text:
        print("No text entered.")
        return

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():

        emissions = model(
            input_ids,
            attention_mask,
        )

        predictions = model.crf.decode(
            emissions,
            mask=attention_mask.bool(),
        )[0]

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids[0]
    )

    print()
    print("=" * 70)
    print("NER RESULTS")
    print("=" * 70)

    for token, label_id in zip(
        tokens,
        predictions,
    ):

        label = labels[label_id]

        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        print(
            f"{token:30} -> {label}"
        )


if __name__ == "__main__":
    main()