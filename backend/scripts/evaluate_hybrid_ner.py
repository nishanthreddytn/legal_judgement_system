"""Evaluate InLegalBERT -> BiLSTM -> Linear -> CRF legal NER model.

Dataset format:
    data/ner/test.jsonl

Each JSONL record contains:
    {
        "input_ids": [...],
        "attention_mask": [...],
        "labels": [...]
    }

Run from backend directory:
    python -m scripts.evaluate_hybrid_ner
"""

import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from transformers import AutoModel
from torchcrf import CRF


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models" / "legal_ner"

CHECKPOINT = (
    MODEL_DIR
    / "checkpoints"
    / "checkpoint-7000"
    / "checkpoint.pt"
)

TEST_FILE = BASE_DIR / "data" / "ner" / "test.jsonl"

BASE_MODEL = "law-ai/InLegalBERT"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LABELS
# ============================================================

LABELS = [
    "O",

    "B-CASE_NUMBER",
    "B-COURT",
    "B-DATE",
    "B-GPE",
    "B-JUDGE",
    "B-LAWYER",
    "B-ORG",
    "B-OTHER_PERSON",
    "B-PETITIONER",
    "B-PRECEDENT",
    "B-PROVISION",
    "B-RESPONDENT",
    "B-STATUTE",
    "B-WITNESS",

    "I-CASE_NUMBER",
    "I-COURT",
    "I-DATE",
    "I-GPE",
    "I-JUDGE",
    "I-LAWYER",
    "I-ORG",
    "I-OTHER_PERSON",
    "I-PETITIONER",
    "I-PRECEDENT",
    "I-PROVISION",
    "I-RESPONDENT",
    "I-STATUTE",
    "I-WITNESS",
]

LABEL2ID = {
    label: i
    for i, label in enumerate(LABELS)
}

ID2LABEL = {
    i: label
    for i, label in enumerate(LABELS)
}

ENTITY_TYPES = [
    label[2:]
    for label in LABELS
    if label.startswith("B-")
]


# ============================================================
# MODEL
# ============================================================

class HybridLegalNER(nn.Module):

    def __init__(self):

        super().__init__()

        print("Loading InLegalBERT...")

        self.bert = AutoModel.from_pretrained(
            BASE_MODEL
        )

        hidden_size = self.bert.config.hidden_size

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(
            512,
            len(LABELS)
        )

        self.crf = CRF(
            len(LABELS),
            batch_first=True
        )


    def emissions(
        self,
        input_ids,
        attention_mask
    ):

        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = bert_output.last_hidden_state

        lstm_output, _ = self.lstm(
            sequence_output
        )

        lstm_output = self.dropout(
            lstm_output
        )

        emissions = self.classifier(
            lstm_output
        )

        return emissions


    def decode(
        self,
        input_ids,
        attention_mask
    ):

        emissions = self.emissions(
            input_ids,
            attention_mask
        )

        mask = attention_mask.bool().clone()

        # torchcrf requires the first timestep
        # to be active for every sequence.
        mask[:, 0] = True

        predictions = self.crf.decode(
            emissions,
            mask=mask
        )

        return predictions


# ============================================================
# LOAD DATASET
# ============================================================

def load_jsonl(path):

    examples = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            examples.append(
                json.loads(line)
            )

    return examples


# ============================================================
# CHECK DATASET
# ============================================================

def validate_example(example, index):

    required = [
        "input_ids",
        "attention_mask",
        "labels"
    ]

    for field in required:

        if field not in example:

            raise ValueError(
                f"Example {index} is missing field: {field}"
            )

    input_ids = example["input_ids"]
    attention_mask = example["attention_mask"]
    labels = example["labels"]

    if not isinstance(input_ids, list):

        raise ValueError(
            f"Example {index}: input_ids is not a list"
        )

    if not isinstance(attention_mask, list):

        raise ValueError(
            f"Example {index}: attention_mask is not a list"
        )

    if not isinstance(labels, list):

        raise ValueError(
            f"Example {index}: labels is not a list"
        )

    if len(input_ids) != len(attention_mask):

        raise ValueError(
            f"Example {index}: "
            f"input_ids length {len(input_ids)} != "
            f"attention_mask length {len(attention_mask)}"
        )

    if len(input_ids) != len(labels):

        raise ValueError(
            f"Example {index}: "
            f"input_ids length {len(input_ids)} != "
            f"labels length {len(labels)}"
        )


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(labels):

    """
    Convert BIO label sequence into entity spans.

    Example:

        B-COURT
        I-COURT
        I-COURT
        O

    becomes:

        ("COURT", start, end)
    """

    result = []

    current_type = None
    start = None

    for index, label in enumerate(labels):

        if label.startswith("B-"):

            if current_type is not None:

                result.append(
                    (
                        current_type,
                        start,
                        index
                    )
                )

            current_type = label[2:]
            start = index

        elif label.startswith("I-"):

            entity_type = label[2:]

            if current_type != entity_type:

                if current_type is not None:

                    result.append(
                        (
                            current_type,
                            start,
                            index
                        )
                    )

                current_type = entity_type
                start = index

        else:

            if current_type is not None:

                result.append(
                    (
                        current_type,
                        start,
                        index
                    )
                )

            current_type = None
            start = None

    if current_type is not None:

        result.append(
            (
                current_type,
                start,
                len(labels)
            )
        )

    return result


# ============================================================
# ENTITY METRICS
# ============================================================

def calculate_entity_metrics(
    true_sequences,
    predicted_sequences
):

    true_entities = set()
    predicted_entities = set()

    per_type_true = defaultdict(set)
    per_type_predicted = defaultdict(set)

    for example_index, (
        true_labels,
        predicted_labels
    ) in enumerate(
        zip(
            true_sequences,
            predicted_sequences
        )
    ):

        true_spans = extract_entities(
            true_labels
        )

        predicted_spans = extract_entities(
            predicted_labels
        )

        for entity_type, start, end in true_spans:

            item = (
                example_index,
                entity_type,
                start,
                end
            )

            true_entities.add(item)

            per_type_true[entity_type].add(
                item
            )

        for entity_type, start, end in predicted_spans:

            item = (
                example_index,
                entity_type,
                start,
                end
            )

            predicted_entities.add(item)

            per_type_predicted[entity_type].add(
                item
            )

    true_positive = len(
        true_entities &
        predicted_entities
    )

    total_true = len(true_entities)

    total_predicted = len(predicted_entities)

    precision = (
        true_positive / total_predicted
        if total_predicted > 0
        else 0.0
    )

    recall = (
        true_positive / total_true
        if total_true > 0
        else 0.0
    )

    if precision + recall > 0:

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0


    rows = []

    for entity_type in ENTITY_TYPES:

        true_set = per_type_true[
            entity_type
        ]

        predicted_set = per_type_predicted[
            entity_type
        ]

        tp = len(
            true_set &
            predicted_set
        )

        entity_precision = (
            tp / len(predicted_set)
            if predicted_set
            else 0.0
        )

        entity_recall = (
            tp / len(true_set)
            if true_set
            else 0.0
        )

        if (
            entity_precision +
            entity_recall
        ) > 0:

            entity_f1 = (
                2
                * entity_precision
                * entity_recall
                / (
                    entity_precision
                    + entity_recall
                )
            )

        else:

            entity_f1 = 0.0

        rows.append(
            (
                entity_type,
                entity_precision,
                entity_recall,
                entity_f1,
                len(true_set)
            )
        )

    return (
        precision,
        recall,
        f1,
        rows
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(model):

    print()
    print("Loading checkpoint:")
    print(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

            print(
                "Checkpoint format: "
                "FULL TRAINING CHECKPOINT"
            )

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

            print(
                "Checkpoint format: "
                "state dict"
            )

        else:

            state_dict = checkpoint

            print(
                "Checkpoint format: "
                "raw model state dict"
            )

        if "global_step" in checkpoint:

            print(
                "Checkpoint global step:",
                checkpoint["global_step"]
            )

        if "epoch" in checkpoint:

            print(
                "Checkpoint epoch:",
                checkpoint["epoch"]
            )

        if "loss" in checkpoint:

            print(
                "Checkpoint training loss:",
                checkpoint["loss"]
            )

    else:

        state_dict = checkpoint

        print(
            "Checkpoint format: "
            "raw state dict"
        )


    # Remove DataParallel prefix if present.

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value


    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )


    if missing:

        print()
        print(
            "WARNING: Missing model parameters:",
            len(missing)
        )

        for key in missing[:10]:

            print(
                "  ",
                key
            )


    if unexpected:

        print()
        print(
            "WARNING: Unexpected checkpoint parameters:",
            len(unexpected)
        )

        for key in unexpected[:10]:

            print(
                "  ",
                key
            )


    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HYBRID LEGAL NER EVALUATION")
    print("=" * 70)

    print()
    print("Model directory :", MODEL_DIR)
    print("Checkpoint      :", CHECKPOINT)
    print("Test dataset    :", TEST_FILE)
    print("Number of labels:", len(LABELS))

    print()
    print("Architecture:")
    print("InLegalBERT -> BiLSTM -> Linear -> CRF")


    # --------------------------------------------------------
    # FILE CHECKS
    # --------------------------------------------------------

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_FILE}"
        )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}"
        )


    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print()
    print("Loading test dataset...")

    examples = load_jsonl(
        TEST_FILE
    )

    print(
        "Test examples:",
        len(examples)
    )

    if len(examples) == 0:

        raise RuntimeError(
            "Test dataset contains 0 examples."
        )


    # --------------------------------------------------------
    # VALIDATE FIRST EXAMPLE
    # --------------------------------------------------------

    print()
    print("Validating dataset format...")

    validate_example(
        examples[0],
        1
    )

    print(
        "Dataset format confirmed:"
    )

    print(
        "input_ids + attention_mask + labels"
    )


    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    print()
    print("Device:", DEVICE)


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("Loading trained hybrid model...")

    model = HybridLegalNER()

    model = load_checkpoint(
        model
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print()
    print(
        "Trained model loaded successfully."
    )


    # --------------------------------------------------------
    # EVALUATION STORAGE
    # --------------------------------------------------------

    true_sequences = []

    predicted_sequences = []

    token_correct = 0

    token_total = 0


    # --------------------------------------------------------
    # RUN EVALUATION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RUNNING TEST EVALUATION")
    print("=" * 70)


    with torch.no_grad():

        for index, example in enumerate(
            examples,
            start=1
        ):

            validate_example(
                example,
                index
            )

            input_ids = torch.tensor(
                [example["input_ids"]],
                dtype=torch.long,
                device=DEVICE
            )

            attention_mask = torch.tensor(
                [example["attention_mask"]],
                dtype=torch.long,
                device=DEVICE
            )

            labels = torch.tensor(
                [example["labels"]],
                dtype=torch.long,
                device=DEVICE
            )


            # ------------------------------------------------
            # CRF DECODING
            # ------------------------------------------------

            predictions = model.decode(
                input_ids,
                attention_mask
            )


            predicted_ids = predictions[0]


            # ------------------------------------------------
            # Build valid positions.
            #
            # labels == -100 are ignored.
            # ------------------------------------------------

            true_ids = labels[0].tolist()

            attention = (
                attention_mask[0]
                .tolist()
            )


            valid_true = []

            valid_pred = []


            pred_position = 0


            for position in range(
                len(true_ids)
            ):

                if attention[position] != 1:

                    continue

                true_label_id = (
                    true_ids[position]
                )


                # CRF produces predictions
                # only for active positions.

                if pred_position >= len(
                    predicted_ids
                ):

                    break


                predicted_label_id = int(
                    predicted_ids[
                        pred_position
                    ]
                )

                pred_position += 1


                # Ignore special-token / padding
                # labels marked as -100.

                if true_label_id == -100:

                    continue


                # Safety check.

                if not (
                    0 <= true_label_id
                    < len(LABELS)
                ):

                    continue


                if not (
                    0 <= predicted_label_id
                    < len(LABELS)
                ):

                    continue


                valid_true.append(
                    ID2LABEL[
                        true_label_id
                    ]
                )

                valid_pred.append(
                    ID2LABEL[
                        predicted_label_id
                    ]
                )


            # ------------------------------------------------
            # Store example
            # ------------------------------------------------

            if valid_true:

                true_sequences.append(
                    valid_true
                )

                predicted_sequences.append(
                    valid_pred
                )


                # Token accuracy.

                for true_label, predicted_label in zip(
                    valid_true,
                    valid_pred
                ):

                    token_total += 1

                    if true_label == predicted_label:

                        token_correct += 1


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if index % 100 == 0:

                print(
                    f"Processed "
                    f"{index}/{len(examples)} examples"
                )


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if len(true_sequences) == 0:

        raise RuntimeError(
            "\n"
            "Evaluation produced 0 valid examples.\n"
            "This means the dataset/checkpoint format "
            "is still incompatible.\n"
        )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    precision, recall, f1, rows = (
        calculate_entity_metrics(
            true_sequences,
            predicted_sequences
        )
    )


    token_accuracy = (
        token_correct / token_total
        if token_total > 0
        else 0.0
    )


    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print()

    print(
        f"Entity Precision : "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Entity Recall    : "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"Entity F1        : "
        f"{f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"Token Accuracy   : "
        f"{token_accuracy:.4f} "
        f"({token_accuracy * 100:.2f}%)"
    )


    # --------------------------------------------------------
    # PER ENTITY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PER-ENTITY PERFORMANCE")
    print("=" * 70)

    print()

    print(
        f"{'Entity':<18}"
        f"{'Precision':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'Support':>10}"
    )

    print("-" * 58)


    for (
        entity_type,
        entity_precision,
        entity_recall,
        entity_f1,
        support
    ) in rows:

        print(
            f"{entity_type:<18}"
            f"{entity_precision:>10.4f}"
            f"{entity_recall:>10.4f}"
            f"{entity_f1:>10.4f}"
            f"{support:>10}"
        )


    # --------------------------------------------------------
    # MACRO AVERAGE
    # --------------------------------------------------------

    macro_precision = (
        sum(
            row[1]
            for row in rows
        )
        / len(rows)
    )

    macro_recall = (
        sum(
            row[2]
            for row in rows
        )
        / len(rows)
    )

    macro_f1 = (
        sum(
            row[3]
            for row in rows
        )
        / len(rows)
    )

    total_support = sum(
        row[4]
        for row in rows
    )


    print()

    print(
        f"{'Macro avg':<18}"
        f"{macro_precision:>10.4f}"
        f"{macro_recall:>10.4f}"
        f"{macro_f1:>10.4f}"
        f"{total_support:>10}"
    )


    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Evaluated "
        f"{len(true_sequences)} "
        f"test examples."
    )

    print()

    print(
        "Model used: "
        "InLegalBERT -> BiLSTM -> Linear -> CRF"
    )

    print(
        "Checkpoint used:"
    )

    print(
        CHECKPOINT
    )

    print()

    print(
        "Entity F1 is the primary NER metric."
    )

    print(
        "Token Accuracy is reported separately."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()