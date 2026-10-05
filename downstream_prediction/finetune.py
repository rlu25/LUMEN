#!/usr/bin/env python3 -u
"""Fine-tune the pretrained fusion encoder for Y0/Y2/Y4 to Y6 prediction.

Unlike ``train.py``, which consumes frozen, pre-extracted subject vectors,
this trainer reloads the original modality/visit tokens and propagates the
supervised loss through the pretrained fusion projectors and Transformer
encoder. It does not load a separate phenotype-specific genetic embedding;
genetics enter only through the generic fusion modality tokens. Each
phenotype gets an independent copy of the pretrained model. The fusion
reconstruction decoder is not used or updated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

from modality_pretraining.common import seed_everything
from multimodal_fusion.data import MultiModalDataset, collate_tokens
from multimodal_fusion.model import MultimodalFusionMAE
from downstream_prediction.data import (
    DEFAULT_LABELS,
    DEFAULT_SPLIT_FILE,
    TASKS,
    load_master_split,
    load_y6_labels,
)
from downstream_prediction.model import DownstreamHead


INPUT_VISITS = ("Y0", "Y2", "Y4")
DEFAULT_FUSION_CHECKPOINT = ROOT / "multimodal_fusion" / "checkpoints" / "best.pt"
DEFAULT_REFERENCE_OUTPUT = HERE / "checkpoints"
DEFAULT_OUTPUT = HERE / "finetuned_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the pretrained fusion encoder on prospective outcomes."
    )
    parser.add_argument("tasks", nargs="*", metavar="TASK")
    parser.add_argument("--fusion-checkpoint", type=Path, default=DEFAULT_FUSION_CHECKPOINT)
    parser.add_argument(
        "--reference-output-dir",
        type=Path,
        default=DEFAULT_REFERENCE_OUTPUT,
        help=(
            "completed frozen downstream output whose split, seed, scaler, and "
            "prediction-head checkpoint define the comparison contract"
        ),
    )
    parser.add_argument("--embedding-dir", type=Path)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--head-init",
        choices=("fresh", "warm-start"),
        default="fresh",
        help=(
            "fresh initializes a new fully trainable task predictor; warm-start "
            "reproduces the historical behavior that loads the frozen predictor"
        ),
    )
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--min-relative-improvement", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    invalid = sorted(set(args.tasks) - set(TASKS))
    if invalid:
        parser.error(
            f"invalid tasks: {', '.join(invalid)} "
            f"(choose from {', '.join(TASKS)})"
        )
    if (
        args.epochs < 1
        or args.patience < 1
        or args.batch_size < 1
        or args.checkpoint_every < 1
    ):
        parser.error(
            "--epochs, --patience, --batch-size, and --checkpoint-every "
            "must be positive"
        )
    if args.workers < 0:
        parser.error("--workers cannot be negative")
    if args.encoder_lr <= 0 or args.head_lr <= 0:
        parser.error("--encoder-lr and --head-lr must be positive")
    if not 0 <= args.min_relative_improvement < 1:
        parser.error("--min-relative-improvement must be in [0,1)")
    args.input_visits = INPUT_VISITS
    args.target_visit = "Y6"
    return args


class FineTuneDataset(Dataset):
    """Label-aligned prospective tokens for one task and master split."""

    def __init__(
        self,
        task: str,
        split: str,
        token_dataset: MultiModalDataset,
        split_map: dict[str, str],
        label_map: dict[str, float],
    ):
        self.task = task
        self.split = split
        self.samples = []
        self.subjects = []
        self.targets = []
        self.task_genetics = []
        self.genetic_present = 0
        for subid, sample in zip(token_dataset.subjects, token_dataset.samples):
            if split_map.get(subid) != split or subid not in label_map:
                continue
            self.subjects.append(subid)
            self.samples.append(sample)
            self.targets.append(float(label_map[subid]))
            self.task_genetics.append(np.empty(0, dtype=np.float32))
        self.targets = np.asarray(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        raw, mod_ids, visit_ids, slot_ids = self.samples[index]
        return (
            raw.copy(),
            mod_ids.copy(),
            visit_ids.copy(),
            slot_ids.copy(),
            self.task_genetics[index].copy(),
            self.targets[index],
            self.subjects[index],
        )


def normalize_token_samples(dataset: MultiModalDataset, normalizer: dict):
    """Apply the fusion training-only normalizer once to the shared tokens."""
    from multimodal_fusion.data import MODALITY_DIMS, MODALITY_IDX

    normalized = []
    for raw, mod_ids, visit_ids, slot_ids in dataset.samples:
        raw = raw.copy()
        for modality, mod_idx in MODALITY_IDX.items():
            selected = mod_ids == mod_idx
            if not selected.any() or modality not in normalizer:
                continue
            dim = MODALITY_DIMS[modality]
            raw[selected, :dim] = (
                raw[selected, :dim] - normalizer[modality]["mean"]
            ) / normalizer[modality]["std"]
        normalized.append((raw, mod_ids, visit_ids, slot_ids))
    dataset.samples = normalized


def collate_finetune(batch):
    tokens = collate_tokens([item[:4] for item in batch])
    task_genetic = torch.from_numpy(np.stack([item[4] for item in batch]))
    targets = torch.as_tensor([item[5] for item in batch], dtype=torch.float32)
    subids = [item[6] for item in batch]
    return (*tokens, task_genetic, targets, subids)


class FineTunedPredictor(nn.Module):
    def __init__(
        self,
        backbone: MultimodalFusionMAE,
        reference_checkpoint: dict,
        head_init: str = "warm-start",
    ):
        super().__init__()
        if head_init not in {"fresh", "warm-start"}:
            raise ValueError(f"Unsupported head initialization: {head_init}")
        self.backbone = backbone
        head_config = reference_checkpoint["model_config"]
        if int(head_config["stage2_dim"]) != backbone.d_model:
            raise ValueError(
                "Frozen head and fusion checkpoint dimensions disagree: "
                f"{head_config['stage2_dim']} != {backbone.d_model}"
            )
        if int(head_config.get("task_genetic_dim", 0)) != 0:
            raise ValueError("Reference checkpoint unexpectedly uses task genetics")
        self.head = DownstreamHead(**head_config)
        if head_init == "warm-start":
            self.head.load_state_dict(reference_checkpoint["model"])
            if self.head.linear_skip is None:
                raise ValueError("Reference checkpoint must use the residual linear head")
            # Historical mode: preserve the frozen-run linear anchor. Gradients
            # still propagate through it into the fine-tuned fusion representation.
            self.head.linear_skip.requires_grad_(False)
        scaler = reference_checkpoint["stage2_scaler"]
        self.register_buffer(
            "stage2_mean", torch.as_tensor(scaler["mean"], dtype=torch.float32)
        )
        self.register_buffer(
            "stage2_std", torch.as_tensor(scaler["std"], dtype=torch.float32)
        )

    def forward(self, raw, mod_ids, visit_ids, slot_ids, padding, task_genetic):
        subject = self.backbone.embed_subject(
            raw, mod_ids, visit_ids, slot_ids, padding
        )
        subject = (subject - self.stage2_mean) / self.stage2_std
        genetic = task_genetic if self.head.task_genetic_dim else None
        return self.head(subject, genetic)


def disable_unused_decoder(backbone: MultimodalFusionMAE):
    """The reconstruction decoder is irrelevant to supervised prediction."""
    prefixes = (
        "mask_token",
        "decoder_projection",
        "decoder.",
        "decoder_norm",
        "reconstruction_heads.",
    )
    for name, parameter in backbone.named_parameters():
        if name == "mask_token" or name.startswith(prefixes[1:]):
            parameter.requires_grad_(False)


def fit_task_genetic_scaler(dataset: FineTuneDataset):
    if not dataset.task_genetics or not len(dataset.task_genetics[0]):
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    values = np.stack(dataset.task_genetics)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    # Do not center or scale the binary presence indicator.
    mean[-1], std[-1] = 0.0, 1.0
    return mean, std


def apply_task_genetic_scaler(dataset: FineTuneDataset, mean, std):
    if not len(mean):
        return
    dataset.task_genetics = [
        ((value - mean) / std).astype(np.float32)
        for value in dataset.task_genetics
    ]


def regression_metrics(target, prediction):
    mse = float(mean_squared_error(target, prediction))
    correlation = (
        float(np.corrcoef(target, prediction)[0, 1])
        if len(target) > 1 and np.std(target) > 0 and np.std(prediction) > 0
        else float("nan")
    )
    return {
        "pearson_r": correlation,
        "r2": float(r2_score(target, prediction)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(mean_absolute_error(target, prediction)),
    }


def classification_metrics(target, probability):
    result = {"positive_rate": float(np.mean(target))}
    if len(np.unique(target)) == 2:
        result.update({
            "roc_auc": float(roc_auc_score(target, probability)),
            "average_precision": float(average_precision_score(target, probability)),
        })
    else:
        result.update({"roc_auc": float("nan"), "average_precision": float("nan")})
    return result


def move_batch(batch, device):
    raw, mod_ids, visit_ids, slot_ids, padding, genetic, target, subids = batch
    return (
        raw.to(device, non_blocking=True),
        mod_ids.to(device, non_blocking=True),
        visit_ids.to(device, non_blocking=True),
        slot_ids.to(device, non_blocking=True),
        padding.to(device, non_blocking=True),
        genetic.to(device, non_blocking=True),
        target.to(device, non_blocking=True),
        subids,
    )


def evaluate_loss(model, loader, loss_fn, target_transform, device):
    model.eval()
    total = count = 0
    with torch.no_grad():
        for batch in loader:
            raw, mod_ids, visit_ids, slot_ids, padding, genetic, target, _ = move_batch(
                batch, device
            )
            transformed = target_transform(target)
            prediction = model(raw, mod_ids, visit_ids, slot_ids, padding, genetic)
            loss = loss_fn(prediction, transformed)
            total += float(loss) * len(target)
            count += len(target)
    return total / max(count, 1)


def predict(model, loader, spec, y_mean, y_std, device):
    model.eval()
    targets, predictions, subids = [], [], []
    with torch.no_grad():
        for batch in loader:
            raw, mod_ids, visit_ids, slot_ids, padding, genetic, target, batch_ids = move_batch(
                batch, device
            )
            output = model(raw, mod_ids, visit_ids, slot_ids, padding, genetic)
            if spec.kind == "classification":
                output = torch.sigmoid(output)
            else:
                output = output * y_std + y_mean
            targets.append(target.cpu().numpy())
            predictions.append(output.cpu().numpy())
            subids.extend(batch_ids)
    return (
        np.concatenate(targets),
        np.concatenate(predictions),
        subids,
    )


def build_token_dataset(args, checkpoint, split_map=None, split=None):
    """Load prospective tokens once; all phenotype trainers share this store."""
    embedding_dir = args.embedding_dir or Path(checkpoint["embedding_dir"])
    token_dataset = MultiModalDataset(
        embedding_dir=embedding_dir,
        split_map=split_map,
        split=split,
        modalities=tuple(checkpoint["modalities"]),
        visits=args.input_visits,
        min_tokens=2,
        normalizer=None,
    )
    normalize_token_samples(token_dataset, checkpoint["normalizer"])
    return token_dataset, embedding_dir


def build_datasets(task, args, token_dataset, split_map):
    label_frame = load_y6_labels(args.labels, task)
    label_map = dict(zip(label_frame["subid"], label_frame["target"]))
    datasets = {
        split: FineTuneDataset(
            task,
            split,
            token_dataset,
            split_map,
            label_map,
        )
        for split in ("train", "val", "test")
    }
    if any(not dataset for dataset in datasets.values()):
        raise RuntimeError(
            f"Empty split for {task}: "
            + ", ".join(f"{k}={len(v)}" for k, v in datasets.items())
        )
    genetic_mean, genetic_std = fit_task_genetic_scaler(datasets["train"])
    for dataset in datasets.values():
        apply_task_genetic_scaler(dataset, genetic_mean, genetic_std)
    return datasets, genetic_mean, genetic_std


def make_loaders(datasets, args):
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_finetune,
        "persistent_workers": args.workers > 0,
    }
    return {
        "train": DataLoader(datasets["train"], shuffle=True, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def save_checkpoint(checkpoint: dict, path: Path):
    """Atomically replace a checkpoint so an interrupted write keeps the old file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def resolve_recorded_path(value: str | Path) -> Path:
    """Resolve repository-relative paths saved by the frozen downstream run."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_reference_checkpoint(
    task: str, args: argparse.Namespace, datasets: dict[str, FineTuneDataset]
) -> tuple[dict, Path]:
    """Load and verify the frozen-run contract for a fair fine-tuning comparison."""
    reference_root = args.reference_output_dir
    if not (reference_root / ".complete").is_file():
        raise FileNotFoundError(
            f"Reference frozen run is incomplete: {reference_root / '.complete'}"
        )
    checkpoint_path = reference_root / task / "best.pt"
    predictions_path = reference_root / task / "predictions.csv"
    cohort_path = reference_root / task / "cohort.json"
    for path in (checkpoint_path, predictions_path, cohort_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen-run reference artifact: {path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "task", "model", "model_config", "stage2_scaler", "target_scaler",
        "split_file", "labels", "subject_dir", "stage2_representation",
        "optimization",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"{checkpoint_path}: missing keys {sorted(missing)}")
    if checkpoint["task"] != task:
        raise ValueError(f"{checkpoint_path}: task is {checkpoint['task']!r}, not {task!r}")
    if (
        args.head_init == "warm-start"
        and int(checkpoint["optimization"].get("seed", -1)) != args.seed
    ):
        raise ValueError(
            f"{checkpoint_path}: frozen seed={checkpoint['optimization'].get('seed')} "
            f"but fine-tuning seed={args.seed}"
        )
    reference_optimization = checkpoint["optimization"]
    matched_optimization = {
        "learning_rate": args.head_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "min_relative_improvement": args.min_relative_improvement,
    }
    if args.head_init == "warm-start":
        for name, value in matched_optimization.items():
            recorded = reference_optimization.get(name)
            if recorded is None or not np.isclose(float(recorded), float(value)):
                raise ValueError(
                    f"{checkpoint_path}: frozen {name}={recorded}, "
                    f"but fine-tuning uses {value}"
                )
    if resolve_recorded_path(checkpoint["split_file"]) != args.split_file.resolve():
        raise ValueError(f"{checkpoint_path}: split file does not match --split-file")
    if resolve_recorded_path(checkpoint["labels"]) != args.labels.resolve():
        raise ValueError(f"{checkpoint_path}: labels do not match --labels")
    recorded_stage2 = checkpoint["stage2_representation"].get("checkpoint")
    if not recorded_stage2 or Path(recorded_stage2).resolve() != args.fusion_checkpoint.resolve():
        raise ValueError(
            f"{checkpoint_path}: fusion checkpoint does not match --fusion-checkpoint"
        )

    with cohort_path.open() as handle:
        reference_cohort = json.load(handle)
    observed_cohort = {split: len(dataset) for split, dataset in datasets.items()}
    for split, count in observed_cohort.items():
        if int(reference_cohort.get(split, -1)) != count:
            raise ValueError(
                f"{task}: {split} cohort differs from frozen run: "
                f"{count} != {reference_cohort.get(split)}"
            )

    reference_predictions = pd.read_csv(predictions_path, usecols=["subid"])
    reference_ids = reference_predictions["subid"].astype(str).tolist()
    test_ids = datasets["test"].subjects
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError(f"{predictions_path}: duplicate test subject IDs")
    if set(reference_ids) != set(test_ids):
        raise ValueError(f"{task}: test subjects differ from the frozen run")
    return checkpoint, checkpoint_path


def train_task(
    task, args, pretrained_checkpoint, token_dataset, embedding_dir, split_map, device
):
    # Make a task produce the same initialization and shuffle order whether it
    # is run alone or as part of the five-task command.
    seed_everything(args.seed)
    spec = TASKS[task]
    datasets, genetic_mean, genetic_std = build_datasets(
        task, args, token_dataset, split_map
    )
    reference_checkpoint, reference_checkpoint_path = load_reference_checkpoint(
        task, args, datasets
    )
    loaders = make_loaders(datasets, args)

    backbone = MultimodalFusionMAE(**pretrained_checkpoint["model_config"])
    backbone.load_state_dict(pretrained_checkpoint["model"])
    disable_unused_decoder(backbone)
    model = FineTunedPredictor(
        backbone, reference_checkpoint, head_init=args.head_init
    ).to(device)

    encoder_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    head_parameters = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    train_targets = datasets["train"].targets
    if spec.kind == "regression":
        y_mean = float(train_targets.mean())
        y_std = max(float(train_targets.std()), 1e-6)
        target_transform = lambda value: (value - y_mean) / y_std
        loss_fn = nn.MSELoss()
    else:
        y_mean, y_std = 0.0, 1.0
        positives = max(float(train_targets.sum()), 1.0)
        pos_weight = torch.tensor(
            [(len(train_targets) - positives) / positives], device=device
        )
        target_transform = lambda value: value
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    reference_target_scaler = reference_checkpoint["target_scaler"]
    if not (
        np.isclose(y_mean, float(reference_target_scaler["mean"]), rtol=0, atol=1e-6)
        and np.isclose(y_std, float(reference_target_scaler["std"]), rtol=0, atol=1e-6)
    ):
        raise ValueError(
            f"{task}: target scaling differs from frozen run: "
            f"({y_mean}, {y_std}) != "
            f"({reference_target_scaler['mean']}, {reference_target_scaler['std']})"
        )

    checkpoint_metadata = {
        "task": task,
        "kind": spec.kind,
        "stage2_model_config": pretrained_checkpoint["model_config"],
        "stage2_checkpoint": str(args.fusion_checkpoint.resolve()),
        "pretrained_stage2_epoch": pretrained_checkpoint.get("epoch"),
        "modalities": tuple(pretrained_checkpoint["modalities"]),
        "input_visits": args.input_visits,
        "target_visit": args.target_visit,
        "embedding_dir": str(embedding_dir.resolve()),
        "task_genetic_dir": None,
        "task_genetic_scaler": {"mean": genetic_mean, "std": genetic_std},
        "target_scaler": {"mean": y_mean, "std": y_std},
        "split_file": str(args.split_file.resolve()),
        "labels": str(args.labels.resolve()),
        "evaluation_split": "test",
        "seed": args.seed,
        "reference_frozen_output": str(args.reference_output_dir.resolve()),
        "reference_frozen_checkpoint": str(reference_checkpoint_path.resolve()),
        "reference_head_model_config": reference_checkpoint["model_config"],
        "head_initialization": args.head_init,
        "selection_includes_epoch_0": args.head_init == "warm-start",
        "stage2_scaler": reference_checkpoint["stage2_scaler"],
        "encoder_lr": args.encoder_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "min_relative_improvement": args.min_relative_improvement,
        "checkpoint_every": args.checkpoint_every,
    }
    print(
        f"  cohort={{'train': {len(datasets['train'])}, 'val': {len(datasets['val'])}, "
        f"'test': {len(datasets['test'])}, "
        f"'task_genetic': {sum(d.genetic_present for d in datasets.values())}}}",
        flush=True,
    )
    print(
        f"  trainable encoder parameters={sum(p.numel() for p in encoder_parameters):,}; "
        f"head parameters={sum(p.numel() for p in head_parameters):,}; "
        f"reference={reference_checkpoint_path}",
        flush=True,
    )

    initial_loss = evaluate_loss(
        model, loaders["val"], loss_fn, target_transform, device
    )
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    if args.head_init == "warm-start":
        reference_val_loss = float(reference_checkpoint["val_loss"])
        if not np.isclose(initial_loss, reference_val_loss, rtol=1e-4, atol=1e-5):
            raise ValueError(
                f"{task}: online fusion plus frozen head does not reproduce the "
                f"frozen validation loss: {initial_loss} != {reference_val_loss}"
            )
        required_loss = initial_loss * (1.0 - args.min_relative_improvement)
        best_loss = initial_loss
        observed_best_loss = initial_loss
        best_epoch, stale = 0, 0
        best_state = initial_state
    else:
        # The random epoch-0 predictor is diagnostic only. Selection begins
        # after the first supervised training epoch.
        required_loss = None
        best_loss = float("inf")
        observed_best_loss = float("inf")
        best_epoch, stale = None, 0
        best_state = None
    task_dir = args.output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_dir / "loss_log.csv"
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "encoder_lr", "head_lr",
            "is_best", "seconds",
        ])
        writer.writerow([
            0, "", initial_loss, args.encoder_lr, args.head_lr,
            int(args.head_init == "warm-start"), 0.0,
        ])
        log_file.flush()
        initial_checkpoint = {
            **checkpoint_metadata,
            "model": initial_state,
            "optimizer": optimizer.state_dict(),
            "epoch": 0,
            "train_loss": None,
            "val_loss": initial_loss,
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "stale_epochs": 0,
            "initial_validation_loss": initial_loss,
            "initial_frozen_val_loss": (
                initial_loss if args.head_init == "warm-start" else None
            ),
            "required_finetuned_val_loss": required_loss,
        }
        save_checkpoint(initial_checkpoint, task_dir / "latest.pt")
        if args.head_init == "warm-start":
            save_checkpoint(initial_checkpoint, task_dir / "best.pt")
            print(
                f"  epoch=0000 train=NA val={initial_loss:.6f} "
                f"(frozen-reference initialization; fine-tuned selection "
                f"requires val<={required_loss:.6f})",
                flush=True,
            )
        else:
            print(
                f"  epoch=0000 train=NA val={initial_loss:.6f} "
                "(fresh random predictor; diagnostic only, not eligible for selection)",
                flush=True,
            )
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            model.train()
            total = count = 0
            for batch in loaders["train"]:
                raw, mod_ids, visit_ids, slot_ids, padding, genetic, target, _ = move_batch(
                    batch, device
                )
                transformed = target_transform(target)
                prediction = model(raw, mod_ids, visit_ids, slot_ids, padding, genetic)
                loss = loss_fn(prediction, transformed)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [*encoder_parameters, *head_parameters], max_norm=1.0
                )
                optimizer.step()
                total += float(loss.detach()) * len(target)
                count += len(target)
            train_loss = total / max(count, 1)
            val_loss = evaluate_loss(
                model, loaders["val"], loss_fn, target_transform, device
            )
            observed_improvement = val_loss < observed_best_loss - 1e-6
            if observed_improvement:
                observed_best_loss, stale = val_loss, 0
            else:
                stale += 1
            is_meaningful = (
                True if args.head_init == "fresh" else val_loss <= required_loss
            )
            is_best = is_meaningful and val_loss < best_loss - 1e-6
            current_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            if is_best:
                best_loss, best_epoch = val_loss, epoch
                best_state = current_state
            checkpoint = {
                **checkpoint_metadata,
                "model": current_state,
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_epoch": best_epoch,
                "best_val_loss": best_loss,
                "stale_epochs": stale,
                "initial_validation_loss": initial_loss,
                "initial_frozen_val_loss": (
                    initial_loss if args.head_init == "warm-start" else None
                ),
                "required_finetuned_val_loss": required_loss,
            }
            save_checkpoint(checkpoint, task_dir / "latest.pt")
            if is_best:
                save_checkpoint(checkpoint, task_dir / "best.pt")
            if epoch % args.checkpoint_every == 0:
                periodic_path = task_dir / f"epoch_{epoch:04d}.pt"
                save_checkpoint(checkpoint, periodic_path)
                print(f"  saved periodic checkpoint: {periodic_path.name}", flush=True)
            elapsed = time.time() - started
            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                optimizer.param_groups[0]["lr"],
                optimizer.param_groups[1]["lr"],
                int(is_best),
                elapsed,
            ])
            log_file.flush()
            if epoch == 1 or is_best or epoch % 10 == 0:
                print(
                    f"  epoch={epoch:04d} train={train_loss:.6f} "
                    f"val={val_loss:.6f} best={best_loss:.6f} "
                    f"stale={stale}/{args.patience}",
                    flush=True,
                )
            if stale >= args.patience:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {task}")
    model.load_state_dict(best_state)
    target, prediction, subids = predict(
        model, loaders["test"], spec, y_mean, y_std, device
    )
    metrics = (
        classification_metrics(target, prediction)
        if spec.kind == "classification"
        else regression_metrics(target, prediction)
    )
    pd.DataFrame({
        "subid": subids,
        "target": target,
        "prediction": prediction,
    }).to_csv(task_dir / "predictions.csv", index=False)
    pd.DataFrame([{
        "task": task,
        "model": f"finetuned_stage2_{args.head_init.replace('-', '_')}_head",
        "evaluation_split": "test",
        **metrics,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
    }]).to_csv(task_dir / "metrics.csv", index=False)

    cohort = {
        split: len(dataset) for split, dataset in datasets.items()
    }
    cohort["task_genetic"] = sum(
        dataset.genetic_present for dataset in datasets.values()
    )
    with (task_dir / "cohort.json").open("w") as handle:
        json.dump(cohort, handle, indent=2)
    print(pd.DataFrame([{"task": task, **metrics}]).to_string(index=False), flush=True)
    return {
        "task": task,
        "model": f"finetuned_stage2_{args.head_init.replace('-', '_')}_head",
        "evaluation_split": "test",
        **metrics,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
    }


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.output_dir.resolve() == args.reference_output_dir.resolve():
        raise ValueError("--output-dir must differ from --reference-output-dir")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty output directory: {args.output_dir}"
        )
    if not args.fusion_checkpoint.exists():
        raise FileNotFoundError(f"Missing fusion checkpoint: {args.fusion_checkpoint}")
    pretrained_checkpoint = torch.load(
        args.fusion_checkpoint, map_location="cpu", weights_only=False
    )
    required = {"model", "model_config", "modalities", "normalizer", "embedding_dir"}
    missing = required - set(pretrained_checkpoint)
    if missing:
        raise ValueError(
            f"{args.fusion_checkpoint} is missing checkpoint keys: {sorted(missing)}"
        )
    if pretrained_checkpoint["model_config"].get("pooling_mode") != "fusion_token":
        raise ValueError(
            f"{args.fusion_checkpoint} is not a fusion-token checkpoint"
        )
    tasks = tuple(args.tasks) or tuple(TASKS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Fusion fine-tuning device={device}; tasks={tasks}; "
        f"head_init={args.head_init}; "
        f"input_visits={args.input_visits}; target_visit={args.target_visit}",
        flush=True,
    )
    print(
        f"Loading shared {'/'.join(args.input_visits)} modality-token store ...",
        flush=True,
    )
    token_dataset, embedding_dir = build_token_dataset(args, pretrained_checkpoint)
    split_frame = load_master_split(args.split_file)
    split_map = dict(zip(split_frame["subid"], split_frame["split"]))
    print(
        f"Loaded {len(token_dataset):,} subjects once for all requested tasks",
        flush=True,
    )
    rows = []
    for task in tasks:
        print(f"\nTask: {task}", flush=True)
        rows.append(train_task(
            task,
            args,
            pretrained_checkpoint,
            token_dataset,
            embedding_dir,
            split_map,
            device,
        ))
    pd.DataFrame(rows).to_csv(args.output_dir / "all_tasks_metrics.csv", index=False)
    (args.output_dir / ".complete").write_text("complete\n")
    print(f"\nSaved fine-tuning outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
