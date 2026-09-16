from pathlib import Path
import json
import re

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "legal_ner"

_model = None
_tokenizer = None

_device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def available():
    return (
        (MODEL_DIR / "hybrid_model.pt").exists()
        and (MODEL_DIR / "config.json").exists()
    )


def load():

    global _model
    global _tokenizer

    if _model is not None:
        return

    if not available():
        return

    from app.ml.hybrid_model import HybridLegalNER

    config_path = MODEL_DIR / "config.json"

    cfg = json.loads(
        config_path.read_text(
            encoding="utf-8"
        )
    )

    _tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"]
    )

    _model = HybridLegalNER(
        cfg["base_model"],
        cfg["num_labels"],
        cfg["lstm_hidden_size"],
        cfg["dropout"],
    )

    checkpoint = torch.load(
        MODEL_DIR / "hybrid_model.pt",
        map_location=_device,
    )

    _model.load_state_dict(
        checkpoint
    )

    _model.id2label = {
        int(k): v
        for k, v in cfg["id2label"].items()
    }

    _model.to(_device)
    _model.eval()


def _merge_entities(text, entities):
    """
    Merge adjacent/overlapping entities of the same type.
    """

    if not entities:
        return []

    entities = sorted(
        entities,
        key=lambda x: (
            x["start"],
            x["end"],
        ),
    )

    merged = []

    for entity in entities:

        if not merged:
            merged.append(entity)
            continue

        previous = merged[-1]

        # Same entity type and overlapping/adjacent
        if (
            previous["label"] == entity["label"]
            and entity["start"] <= previous["end"] + 2
        ):
            previous["end"] = max(
                previous["end"],
                entity["end"],
            )

            previous["text"] = text[
                previous["start"]:previous["end"]
            ]

        else:
            merged.append(entity)

    return merged


def _clean_entity_text(text):
    """
    Clean whitespace and tokenizer artifacts.
    """

    text = text.replace("##", "")
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip(" ,.;:()[]")


def _repair_provision_entities(text, entities):
    """
    Repair incomplete PROVISION entities.

    Example:
        'section'
    becomes:
        'section 302'

    when the source text contains it.
    """

    repaired = []

    for entity in entities:

        if entity["label"] != "PROVISION":
            repaired.append(entity)
            continue

        entity_text = entity["text"].strip()

        # If model only detected "section", inspect following text.
        if entity_text.lower() in {
            "section",
            "sec",
            "s",
        }:

            following = text[
                entity["end"]:
                entity["end"] + 50
            ]

            match = re.match(
                r"\s*(\d+[A-Za-z]?(?:\s*\(\d+\))?)",
                following,
                flags=re.IGNORECASE,
            )

            if match:

                entity["end"] += match.end()

                entity["text"] = text[
                    entity["start"]:
                    entity["end"]
                ]

        repaired.append(entity)

    return repaired


def predict(text):

    load()

    if _model is None:
        return []

    if not text or not text.strip():
        return []

    enc = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    offsets = enc.pop(
        "offset_mapping"
    )[0].tolist()

    input_ids = enc[
        "input_ids"
    ].to(_device)

    attention_mask = enc[
        "attention_mask"
    ].to(_device)

    with torch.no_grad():

        predictions = _model.predict(
            input_ids,
            attention_mask,
        )[0]

    labels = {
        int(k): v
        for k, v in _model.id2label.items()
    }

    entities = []

    current = None

    for i, prediction in enumerate(predictions):

        if i >= len(offsets):
            break

        start, end = offsets[i]

        # Ignore [CLS], [SEP], padding etc.
        if start == end:
            continue

        prediction_id = int(
            prediction
        )

        label = labels.get(
            prediction_id,
            "O",
        )

        # Outside
        if label == "O":

            if current is not None:
                entities.append(current)
                current = None

            continue

        # Beginning of entity
        if label.startswith("B-"):

            if current is not None:
                entities.append(current)

            current = {
                "label": label[2:],
                "start": start,
                "end": end,
            }

            continue

        # Inside entity
        if label.startswith("I-"):

            entity_type = label[2:]

            if (
                current is not None
                and current["label"] == entity_type
            ):
                current["end"] = end

            else:

                if current is not None:
                    entities.append(current)

                current = {
                    "label": entity_type,
                    "start": start,
                    "end": end,
                }

    if current is not None:
        entities.append(current)

    # Convert offsets to text
    cleaned_entities = []

    for entity in entities:

        entity_text = text[
            entity["start"]:
            entity["end"]
        ]

        entity_text = _clean_entity_text(
            entity_text
        )

        if not entity_text:
            continue

        entity["text"] = entity_text

        cleaned_entities.append(
            entity
        )

    # Merge pieces
    cleaned_entities = _merge_entities(
        text,
        cleaned_entities,
    )

    # Repair incomplete provisions
    cleaned_entities = _repair_provision_entities(
        text,
        cleaned_entities,
    )

    # Final cleanup
    final_entities = []

    for entity in cleaned_entities:

        entity["text"] = _clean_entity_text(
            entity["text"]
        )

        if not entity["text"]:
            continue

        final_entities.append(
            {
                "text": entity["text"],
                "label": entity["label"],
            }
        )

    return final_entities