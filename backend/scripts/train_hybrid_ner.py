import argparse
import json
import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup
from torchcrf import CRF


# ============================================================
# HYBRID MODEL
# InLegalBERT -> BiLSTM -> Linear -> CRF
# ============================================================

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

    def emissions(
        self,
        input_ids,
        attention_mask,
    ):

        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        x = bert_output.last_hidden_state

        x = self.dropout(x)

        x, _ = self.lstm(x)

        x = self.dropout(x)

        return self.classifier(x)

    def loss(
        self,
        input_ids,
        attention_mask,
        labels,
    ):

        emissions = self.emissions(
            input_ids,
            attention_mask,
        )

        safe_labels = labels.clone()

        safe_labels[safe_labels == -100] = 0

        crf_mask = attention_mask.bool()

        log_likelihood = self.crf(
            emissions,
            safe_labels,
            mask=crf_mask,
            reduction="mean",
        )

        return -log_likelihood

    def predict(
        self,
        input_ids,
        attention_mask,
    ):

        emissions = self.emissions(
            input_ids,
            attention_mask,
        )

        crf_mask = attention_mask.bool()

        return self.crf.decode(
            emissions,
            mask=crf_mask,
        )


# ============================================================
# DATASET
# ============================================================

class TokenizedNERDataset(Dataset):

    def __init__(self, path):

        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.path}"
            )

        self.rows = []

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if line:
                    self.rows.append(
                        json.loads(line)
                    )

        if not self.rows:
            raise ValueError(
                f"No examples found in {self.path}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):

        row = self.rows[index]

        return {
            "input_ids": torch.tensor(
                row["input_ids"],
                dtype=torch.long,
            ),

            "attention_mask": torch.tensor(
                row["attention_mask"],
                dtype=torch.long,
            ),

            "labels": torch.tensor(
                row["labels"],
                dtype=torch.long,
            ),
        }


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):

    max_length = max(
        len(item["input_ids"])
        for item in batch
    )

    input_ids_batch = []
    attention_batch = []
    labels_batch = []

    for item in batch:

        length = len(item["input_ids"])

        padding = max_length - length

        input_ids = torch.cat(
            [
                item["input_ids"],
                torch.zeros(
                    padding,
                    dtype=torch.long,
                ),
            ]
        )

        attention = torch.cat(
            [
                item["attention_mask"],
                torch.zeros(
                    padding,
                    dtype=torch.long,
                ),
            ]
        )

        labels = torch.cat(
            [
                item["labels"],
                torch.full(
                    (padding,),
                    -100,
                    dtype=torch.long,
                ),
            ]
        )

        input_ids_batch.append(input_ids)
        attention_batch.append(attention)
        labels_batch.append(labels)

    return {
        "input_ids": torch.stack(
            input_ids_batch
        ),

        "attention_mask": torch.stack(
            attention_batch
        ),

        "labels": torch.stack(
            labels_batch
        ),
    }


# ============================================================
# LABEL CONFIG
# ============================================================

def load_label_config():

    path = Path(
        "data/ner/label_config.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing label configuration: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# DEV LOSS
# ============================================================

def evaluate_loss(
    model,
    loader,
    device,
):

    model.eval()

    total_loss = 0.0
    count = 0

    with torch.no_grad():

        for batch in loader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            loss = model.loss(
                input_ids,
                attention_mask,
                labels,
            )

            total_loss += loss.item()

            count += 1

    model.train()

    if count == 0:
        return 0.0

    return total_loss / count


# ============================================================
# SAFE CHECKPOINT SAVE
#
# IMPORTANT:
# We save to a temporary file first.
#
# If saving fails because the disk is full,
# the previous valid checkpoint remains untouched.
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    output_dir,
    epoch,
    step,
    global_step,
    train_loss,
):

    checkpoint_dir = (
        Path(output_dir)
        / "checkpoints"
        / f"checkpoint-{global_step}"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_path = (
        checkpoint_dir
        / "checkpoint.pt"
    )

    temp_path = (
        checkpoint_dir
        / "checkpoint.tmp.pt"
    )

    # Remove old incomplete temporary file
    if temp_path.exists():

        try:
            temp_path.unlink()
        except Exception:
            pass

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "epoch":
            epoch,

        "step":
            step,

        "global_step":
            global_step,

        "train_loss":
            train_loss,
    }

    try:

        print()
        print(
            f"Saving checkpoint {global_step}..."
        )

        torch.save(
            checkpoint,
            temp_path,
        )

        # Atomic replacement
        os.replace(
            temp_path,
            final_path,
        )

        print(
            f"CHECKPOINT SAVED: {final_path}"
        )

        return True

    except Exception as e:

        print()
        print(
            "WARNING: CHECKPOINT SAVE FAILED"
        )

        print(
            f"Reason: {e}"
        )

        print(
            "The training model in memory is still valid."
        )

        print(
            "The previous successful checkpoint remains untouched."
        )

        # Delete corrupted temporary file
        if temp_path.exists():

            try:
                temp_path.unlink()
            except Exception:
                pass

        return False


# ============================================================
# SAVE FINAL MODEL
# ============================================================

def save_final_model(
    model,
    output_dir,
    config,
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        output_dir
        / "hybrid_model.pt"
    )

    config_path = (
        output_dir
        / "config.json"
    )

    temp_model_path = (
        output_dir
        / "hybrid_model.tmp.pt"
    )

    print()
    print(
        "Saving final model..."
    )

    try:

        if temp_model_path.exists():

            temp_model_path.unlink()

        torch.save(
            model.state_dict(),
            temp_model_path,
        )

        os.replace(
            temp_model_path,
            model_path,
        )

        config_path.write_text(
            json.dumps(
                config,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "Final model saved:"
        )

        print(
            " ",
            model_path,
        )

        print()
        print(
            "Configuration saved:"
        )

        print(
            " ",
            config_path,
        )

        return True

    except Exception as e:

        print()
        print(
            "WARNING: FINAL MODEL SAVE FAILED"
        )

        print(
            f"Reason: {e}"
        )

        if temp_model_path.exists():

            try:
                temp_model_path.unlink()
            except Exception:
                pass

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Continue training InLegalBERT + BiLSTM + CRF "
            "Legal NER model."
        )
    )

    parser.add_argument(
        "--train-data",
        default="data/ner/train.jsonl",
    )

    parser.add_argument(
        "--dev-data",
        default="data/ner/dev.jsonl",
    )

    parser.add_argument(
        "--base-model",
        default="law-ai/InLegalBERT",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--output",
        default="models/legal_ner",
    )

    # ========================================================
    # IMPORTANT:
    # We will now load checkpoint-4000 directly.
    # ========================================================

    parser.add_argument(
        "--resume-checkpoint",
        default=(
            "models/legal_ner/"
            "checkpoints/checkpoint-4000/"
            "checkpoint.pt"
        ),
    )

    parser.add_argument(
        "--start-step",
        type=int,
        default=4000,
        help=(
            "Global step represented by the "
            "resume checkpoint."
        ),
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 70)
    print("HYBRID LEGAL NER CONTINUED TRAINING")
    print("=" * 70)

    print()

    print("Architecture:")
    print("  InLegalBERT")
    print("      -> BiLSTM")
    print("      -> Linear classifier")
    print("      -> CRF")

    print()

    print("Train dataset:", args.train_data)
    print("Dev dataset  :", args.dev_data)
    print("Base model   :", args.base_model)

    print(
        "Resume checkpoint:",
        args.resume_checkpoint,
    )

    print(
        "Starting step:",
        args.start_step,
    )

    print(
        "Epochs:",
        args.epochs,
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Learning rate:",
        args.lr,
    )

    print(
        "Checkpoint every:",
        args.checkpoint_every,
        "steps",
    )

    # ========================================================
    # LABELS
    # ========================================================

    label_config = load_label_config()

    labels = label_config["labels"]

    num_labels = len(labels)

    print()
    print(
        "Number of labels:",
        num_labels,
    )

    print()
    print("Labels:")

    for i, label in enumerate(labels):

        print(
            f"  {i:2d} -> {label}"
        )

    # ========================================================
    # DATASETS
    # ========================================================

    print()
    print("Loading datasets...")

    train_dataset = TokenizedNERDataset(
        args.train_data
    )

    dev_dataset = TokenizedNERDataset(
        args.dev_data
    )

    print(
        "Train examples:",
        len(train_dataset),
    )

    print(
        "Dev examples:",
        len(dev_dataset),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    else:

        print()
        print(
            "WARNING: CUDA is not available."
        )

        print(
            "Training InLegalBERT on CPU can be slow."
        )

    # ========================================================
    # CHECK DISK SPACE
    # ========================================================

    print()
    print(
        "Checking disk space..."
    )

    disk_usage = shutil.disk_usage(
        Path.cwd().anchor
    )

    free_gb = (
        disk_usage.free
        / (1024 ** 3)
    )

    print(
        f"Free disk space: {free_gb:.2f} GB"
    )

    if free_gb < 5:

        raise RuntimeError(
            "Less than 5 GB of free disk space. "
            "Free some space before continuing."
        )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Loading InLegalBERT..."
    )

    model = HybridLegalNER(
        base_model=args.base_model,
        num_labels=num_labels,
        lstm_hidden_size=256,
        dropout=0.2,
    )

    # ========================================================
    # LOAD CHECKPOINT (OPTIONAL)
    # ========================================================

    checkpoint_path = Path(
        args.resume_checkpoint
    )

    if checkpoint_path.exists():
        print()
        print(
            "Loading checkpoint..."
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )


    # ========================================================
    # DETERMINE CHECKPOINT FORMAT
    # ========================================================

    if checkpoint_path.exists():
        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            print(
                "Checkpoint format: full training checkpoint"
            )

            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )

        else:

            print(
                "Checkpoint format: plain model state_dict"
            )

            model.load_state_dict(
                checkpoint,
                strict=True,
            )

        print(
            "Model weights loaded successfully."
        )
    else:
        print("Training from scratch. No checkpoint loaded.")


    model.to(device)

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
    )

    # ========================================================
    # TRAINING STEPS
    # ========================================================

    steps_per_epoch = len(
        train_loader
    )

    total_new_steps = (
        steps_per_epoch
        * args.epochs
    )

    final_global_step = (
        args.start_step
        + total_new_steps
    )

    # We create a scheduler for the continuation period.
    #
    # IMPORTANT:
    # Do NOT use the old 0->5498 scheduler because
    # we are continuing from step 4000.
    #
    # This scheduler is designed for the new training run.
    # ========================================================

    warmup_steps = max(
        1,
        int(
            total_new_steps * 0.1
        ),
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_new_steps,
        )
    )

    print()
    print(
        "Steps per epoch:",
        steps_per_epoch,
    )

    print(
        "New training steps:",
        total_new_steps,
    )

    print(
        "Starting global step:",
        args.start_step,
    )

    print(
        "Final global step:",
        final_global_step,
    )

    print(
        "Warmup steps:",
        warmup_steps,
    )

    # ========================================================
    # CRF SMOKE TEST
    # ========================================================

    print()
    print("=" * 70)
    print("RUNNING CRF SMOKE TEST")
    print("=" * 70)

    test_batch = next(
        iter(train_loader)
    )

    test_input_ids = test_batch[
        "input_ids"
    ].to(device)

    test_attention = test_batch[
        "attention_mask"
    ].to(device)

    test_labels = test_batch[
        "labels"
    ].to(device)

    print()

    print(
        "Batch shape:",
        tuple(
            test_input_ids.shape
        ),
    )

    print(
        "CRF first timestep mask:",
        test_attention[:, 0].tolist(),
    )

    if not torch.all(
        test_attention[:, 0].bool()
    ):

        raise RuntimeError(
            "CRF smoke test failed: "
            "first timestep mask is not active."
        )

    with torch.no_grad():

        smoke_loss = model.loss(
            test_input_ids,
            test_attention,
            test_labels,
        )

    print(
        "Smoke-test loss:",
        f"{smoke_loss.item():.4f}",
    )

    print()
    print(
        "CRF smoke test PASSED."
    )

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING CONTINUED TRAINING")
    print("=" * 70)

    model.train()

    global_step = args.start_step

    for epoch in range(
        args.epochs
    ):

        total_loss = 0.0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels_tensor = batch[
                "labels"
            ].to(device)

            optimizer.zero_grad()

            loss = model.loss(
                input_ids,
                attention_mask,
                labels_tensor,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            scheduler.step()

            global_step += 1

            total_loss += loss.item()

            # =================================================
            # PROGRESS
            # =================================================

            if step % 100 == 0:

                average_loss = (
                    total_loss / step
                )

                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"| Step {step}/{len(train_loader)} "
                    f"| Global step {global_step} "
                    f"| Loss {average_loss:.4f}"
                )

            # =================================================
            # PERIODIC CHECKPOINT
            # =================================================

            if (
                global_step
                % args.checkpoint_every
                == 0
            ):

                average_loss = (
                    total_loss / step
                )

                success = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    output_dir=args.output,
                    epoch=epoch,
                    step=step,
                    global_step=global_step,
                    train_loss=average_loss,
                )

                if not success:

                    print()
                    print(
                        "Training will continue."
                    )

                    print(
                        "No valid checkpoint was created "
                        "for this step."
                    )

        # =====================================================
        # EPOCH COMPLETE
        # =====================================================

        train_loss = (
            total_loss
            / len(train_loader)
        )

        print()
        print(
            f"Epoch {epoch + 1} complete"
        )

        print(
            f"  Train loss: {train_loss:.4f}"
        )

        # =====================================================
        # DEV LOSS
        # =====================================================

        print()
        print(
            "Evaluating on development set..."
        )

        dev_loss = evaluate_loss(
            model,
            dev_loader,
            device,
        )

        print(
            f"  Dev loss: {dev_loss:.4f}"
        )

    # ========================================================
    # FINAL MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("CONTINUED TRAINING COMPLETE")
    print("=" * 70)

    config = {

        "base_model":
            args.base_model,

        "num_labels":
            num_labels,

        "labels":
            labels,

        "id2label":
        {
            str(i): label
            for i, label
            in enumerate(labels)
        },

        "label2id":
        {
            label: i
            for i, label
            in enumerate(labels)
        },

        "lstm_hidden_size":
            256,

        "dropout":
            0.2,

        "architecture":
        [
            "InLegalBERT",
            "BiLSTM",
            "Linear",
            "CRF",
        ],

        "continued_from":
            str(checkpoint_path),

        "started_from_step":
            args.start_step,

        "total_new_steps":
            total_new_steps,

        "total_training_steps":
            global_step,

    }

    success = save_final_model(
        model=model,
        output_dir=args.output,
        config=config,
    )

    if success:

        print()
        print("=" * 70)
        print("READY FOR TEST EVALUATION")
        print("=" * 70)

    else:

        print()
        print(
            "Training finished, but the final model "
            "could not be saved."
        )


if __name__ == "__main__":
    main()