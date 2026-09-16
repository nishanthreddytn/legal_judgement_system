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

        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        x = output.last_hidden_state

        x = self.dropout(x)

        x, _ = self.lstm(x)

        x = self.dropout(x)

        return self.classifier(x)


class LegalNER:

    def __init__(self):

        self.base_model = "law-ai/InLegalBERT"

        self.checkpoint = Path(
            "models/legal_ner/checkpoints/checkpoint-7000/checkpoint.pt"
        )

        self.label_config = Path(
            "data/ner/label_config.json"
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        with self.label_config.open(
            "r",
            encoding="utf-8",
        ) as f:
            config = json.load(f)

        self.labels = config["labels"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model
        )

        self.model = HybridLegalNER(
            base_model=self.base_model,
            num_labels=len(self.labels),
            lstm_hidden_size=256,
            dropout=0.2,
        )

        checkpoint = torch.load(
            self.checkpoint,
            map_location="cpu",
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        input_ids = encoded["input_ids"].to(
            self.device
        )

        attention_mask = encoded[
            "attention_mask"
        ].to(self.device)

        with torch.no_grad():

            emissions = self.model(
                input_ids,
                attention_mask,
            )

            prediction = self.model.crf.decode(
                emissions,
                mask=attention_mask.bool(),
            )[0]

        tokens = self.tokenizer.convert_ids_to_tokens(
            input_ids[0]
        )

        entities = []

        current_entity = None

        for token, label_id in zip(
            tokens,
            prediction,
        ):

            label = self.labels[label_id]

            if token in [
                "[CLS]",
                "[SEP]",
                "[PAD]",
            ]:
                continue

            if label.startswith("B-"):

                if current_entity:
                    entities.append(
                        current_entity
                    )

                current_entity = {
                    "label": label[2:],
                    "tokens": [token],
                }

            elif label.startswith("I-"):

                if current_entity:

                    current_entity[
                        "tokens"
                    ].append(token)

            else:

                if current_entity:
                    entities.append(
                        current_entity
                    )

                    current_entity = None

        if current_entity:
            entities.append(
                current_entity
            )

        results = []

        for entity in entities:

            tokens = entity["tokens"]

            text_value = ""

            for token in tokens:

                if token.startswith("##"):
                    text_value += token[2:]
                else:

                    if text_value:
                        text_value += " "

                    text_value += token

            results.append(
                {
                    "text": text_value,
                    "label": entity["label"],
                }
            )

        return results


ner_model = LegalNER()