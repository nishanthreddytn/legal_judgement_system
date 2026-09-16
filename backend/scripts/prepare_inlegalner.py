from pathlib import Path
import json

from datasets import load_dataset
from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "law-ai/InLegalBERT"
DATASET_NAME = "opennyaiorg/InLegalNER"

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "ner"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LABELS
# ============================================================

EXPECTED_LABELS = [
    "O",
    "LAWYER",
    "COURT",
    "JUDGE",
    "PETITIONER",
    "RESPONDENT",
    "CASE_NUMBER",
    "GPE",
    "DATE",
    "ORG",
    "STATUTE",
    "WITNESS",
    "PRECEDENT",
    "PROVISION",
    "OTHER_PERSON",
]


# ============================================================
# CONVERT CHARACTER ANNOTATIONS -> BIO WORD LABELS
# ============================================================

def convert_example(example):
    text = example["data"]["text"]

    labels = ["O"] * len(text)

    annotations = example.get("annotations", [])

    for annotation_group in annotations:
        for result in annotation_group.get("result", []):

            value = result.get("value", {})

            start = value.get("start")
            end = value.get("end")
            entity_labels = value.get("labels", [])

            if start is None or end is None or not entity_labels:
                continue

            entity_type = entity_labels[0]

            if entity_type not in EXPECTED_LABELS:
                raise ValueError(
                    f"Unknown entity label: {entity_type}"
                )

            if start < 0 or end > len(text) or start >= end:
                raise ValueError(
                    f"Invalid annotation span: "
                    f"start={start}, end={end}, text_length={len(text)}"
                )

            # Check for overlapping annotations.
            for position in range(start, end):
                if labels[position] != "O":
                    raise ValueError(
                        f"Overlapping annotation detected in text: "
                        f"{text[start:end]!r}"
                    )

            # Character-level temporary marking.
            labels[start] = f"B-{entity_type}"

            for position in range(start + 1, end):
                labels[position] = f"I-{entity_type}"

    return text, labels


# ============================================================
# CONVERT CHARACTER BIO -> TOKEN BIO
# ============================================================

def tokenize_example(text, char_labels, tokenizer):
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offsets = encoding["offset_mapping"]

    token_labels = []

    for start, end in offsets:

        # Special tokens such as [CLS] and [SEP]
        if start == end:
            token_labels.append(-100)
            continue

        # Ignore whitespace-only tokens.
        token_text = text[start:end]

        if token_text.strip() == "":
            token_labels.append(-100)
            continue

        # Find labels inside this token.
        covered = char_labels[start:end]

        entity_labels = [
            label
            for label in covered
            if label != "O"
        ]

        if not entity_labels:
            token_labels.append("O")
            continue

        # Determine entity type.
        first_entity = entity_labels[0]

        if first_entity.startswith("B-"):
            entity_type = first_entity[2:]
            token_labels.append(f"B-{entity_type}")
        else:
            entity_type = first_entity[2:]
            token_labels.append(f"I-{entity_type}")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": token_labels,
    }


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(dataset_split, tokenizer, split_name):

    processed = []

    entity_count = 0

    for index, example in enumerate(dataset_split):

        text, char_labels = convert_example(example)

        result = tokenize_example(
            text,
            char_labels,
            tokenizer,
        )

        # Convert labels to integer IDs later.
        processed.append(
            {
                "input_ids": result["input_ids"],
                "attention_mask": result["attention_mask"],
                "labels": result["labels"],
            }
        )

        entity_count += sum(
            1
            for label in char_labels
            if label.startswith("B-")
        )

        if (index + 1) % 1000 == 0:
            print(
                f"{split_name}: "
                f"{index + 1}/{len(dataset_split)} processed"
            )

    print(
        f"{split_name}: completed "
        f"({len(processed)} examples, "
        f"{entity_count} entities)"
    )

    return processed


# ============================================================
# CONVERT STRING LABELS TO INTEGER LABELS
# ============================================================

def build_label_map(processed_splits):

    labels = {"O"}

    for split in processed_splits.values():
        for example in split:
            for label in example["labels"]:
                if label != -100:
                    labels.add(label)

    labels = sorted(labels)

    # Make O the first label.
    labels.remove("O")
    labels.insert(0, "O")

    label2id = {
        label: index
        for index, label in enumerate(labels)
    }

    id2label = {
        index: label
        for label, index in label2id.items()
    }

    return label2id, id2label


def convert_label_strings(processed_splits, label2id):

    for split_name, split in processed_splits.items():

        for example in split:

            example["labels"] = [
                -100 if label == -100 else label2id[label]
                for label in example["labels"]
            ]

    return processed_splits


# ============================================================
# SAVE JSONL
# ============================================================

def save_split(split, path):

    with path.open("w", encoding="utf-8") as f:

        for example in split:
            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("InLegalNER preparation")
    print("=" * 70)

    print(f"Dataset : {DATASET_NAME}")
    print(f"Model   : {BASE_MODEL}")
    print(f"Output  : {OUTPUT_DIR}")
    print()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading InLegalNER...")

    dataset = load_dataset(DATASET_NAME)

    print(dataset)
    print()

    # --------------------------------------------------------
    # Verify official splits
    # --------------------------------------------------------

    required_splits = ["train", "dev", "test"]

    for split in required_splits:
        if split not in dataset:
            raise RuntimeError(
                f"Missing required dataset split: {split}"
            )

    print("Dataset split sizes:")

    for split in required_splits:
        print(
            f"  {split}: {len(dataset[split])}"
        )

    print()

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    print("Loading InLegalBERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    print(
        f"Tokenizer: {tokenizer.__class__.__name__}"
    )

    print()

    # --------------------------------------------------------
    # Process dataset
    # --------------------------------------------------------

    processed = {}

    for split in required_splits:

        processed[split] = process_split(
            dataset[split],
            tokenizer,
            split,
        )

    # --------------------------------------------------------
    # Build label map
    # --------------------------------------------------------

    print()
    print("Building label map...")

    label2id, id2label = build_label_map(
        processed
    )

    print()
    print("Labels discovered:")

    for label, index in label2id.items():
        print(
            f"  {index:2d} -> {label}"
        )

    print()

    # --------------------------------------------------------
    # Convert labels to IDs
    # --------------------------------------------------------

    processed = convert_label_strings(
        processed,
        label2id,
    )

    # --------------------------------------------------------
    # Save splits
    # --------------------------------------------------------

    print("Saving prepared datasets...")

    save_split(
        processed["train"],
        OUTPUT_DIR / "train.jsonl",
    )

    save_split(
        processed["dev"],
        OUTPUT_DIR / "dev.jsonl",
    )

    save_split(
        processed["test"],
        OUTPUT_DIR / "test.jsonl",
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = {
        "dataset": DATASET_NAME,
        "base_model": BASE_MODEL,
        "max_length": 512,
        "labels": list(label2id.keys()),
        "label2id": label2id,
        "id2label": {
            str(k): v
            for k, v in id2label.items()
        },
    }

    with (
        OUTPUT_DIR / "label_config.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    for split_name in required_splits:

        split = processed[split_name]

        print()
        print(
            f"{split_name.upper()}: "
            f"{len(split)} examples"
        )

        for i in range(min(3, len(split))):

            example = split[i]

            assert len(
                example["input_ids"]
            ) == len(
                example["attention_mask"]
            )

            assert len(
                example["input_ids"]
            ) == len(
                example["labels"]
            )

        print("  First 3 examples: OK")

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)

    print()
    print("Generated files:")

    print(
        f"  {OUTPUT_DIR / 'train.jsonl'}"
    )

    print(
        f"  {OUTPUT_DIR / 'dev.jsonl'}"
    )

    print(
        f"  {OUTPUT_DIR / 'test.jsonl'}"
    )

    print(
        f"  {OUTPUT_DIR / 'label_config.json'}"
    )

    print()
    print("Official dataset split preserved:")
    print(f"  Train: {len(dataset['train'])}")
    print(f"  Dev:   {len(dataset['dev'])}")
    print(f"  Test:  {len(dataset['test'])}")

    print()
    print("No model was trained.")
    print("Dataset preparation and validation succeeded.")


if __name__ == "__main__":
    main()