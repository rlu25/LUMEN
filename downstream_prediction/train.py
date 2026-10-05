#!/usr/bin/env python3 -u
"""Train task heads from prospective shared subject embeddings only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

HERE = Path(__file__).resolve().parent
from modality_pretraining.common import seed_everything

from downstream_prediction.data import (
    DEFAULT_LABELS,
    DEFAULT_SPLIT_FILE,
    DEFAULT_SUBJECT_DIR,
    TASKS,
    DownstreamDataset,
)
from downstream_prediction.model import DownstreamHead


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="*", metavar="TASK")
    parser.add_argument("--subject-dir", type=Path, default=DEFAULT_SUBJECT_DIR)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "runs")
    parser.add_argument(
        "--evaluation-split", choices=("val", "test"), default="test",
        help=(
            "partition used for reported predictions and metrics; use val for "
            "architecture selection and test only after selecting the architecture"
        ),
    )
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--hidden-dims", nargs="*", type=int, default=[])
    parser.add_argument(
        "--min-relative-improvement", type=float, default=0.005,
        help=(
            "minimum relative validation-loss reduction required before the "
            "nonlinear residual replaces the epoch-0 linear baseline"
        ),
    )
    parser.add_argument(
        "--layer-norm", action=argparse.BooleanOptionalAction, default=False,
        help="apply per-subject LayerNorm after train-only feature standardization",
    )
    parser.add_argument(
        "--residual-linear", action=argparse.BooleanOptionalAction, default=True,
        help="add a direct linear path so the nonlinear head can refine it",
    )
    parser.add_argument(
        "--allow-existing-output", action="store_true",
        help="allow writing into a non-empty output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    invalid = sorted(set(args.tasks) - set(TASKS))
    if invalid:
        parser.error(f"invalid tasks: {', '.join(invalid)} (choose from {', '.join(TASKS)})")
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("--epochs, --patience, and --batch-size must be positive")
    if args.projection_dim < 1 or any(width < 1 for width in args.hidden_dims):
        parser.error("--projection-dim and every --hidden-dims value must be positive")
    if not 0 <= args.dropout < 1 or args.weight_decay < 0:
        parser.error("--dropout must be in [0,1) and --weight-decay cannot be negative")
    if not 0 <= args.min_relative_improvement < 1:
        parser.error("--min-relative-improvement must be in [0,1)")
    return args


def fit_scaler(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = value.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = value.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def transform(value, mean, std):
    return ((value - mean) / std).astype(np.float32)


def regression_metrics(target, prediction):
    if len(target) > 1 and np.std(target) > 0 and np.std(prediction) > 0:
        correlation = float(np.corrcoef(target, prediction)[0, 1])
    else:
        correlation = float("nan")
    mse = float(mean_squared_error(target, prediction))
    return {
        "pearson_r": correlation,
        "r2": float(r2_score(target, prediction)),
        "mse": mse,
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": math.sqrt(mse),
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


def score(spec, target, prediction):
    return (classification_metrics(target, prediction) if spec.kind == "classification"
            else regression_metrics(target, prediction))


def baseline(task, spec, arrays, output_dir, evaluation_split):
    train_x, train_g, train_y, _ = arrays["train"]
    val_x, val_g, val_y, _ = arrays["val"]
    eval_x, eval_g, eval_y, eval_ids = arrays[evaluation_split]
    train_features = np.concatenate([train_x, train_g], axis=1)
    val_features = np.concatenate([val_x, val_g], axis=1)
    eval_features = np.concatenate([eval_x, eval_g], axis=1)
    mean, std = fit_scaler(train_features)
    train_features = transform(train_features, mean, std)
    val_features = transform(val_features, mean, std)
    eval_features = transform(eval_features, mean, std)

    if spec.kind == "classification":
        candidates = (0.01, 0.1, 1.0, 10.0)
        positives = max(float(train_y.sum()), 1.0)
        pos_weight = (len(train_y) - positives) / positives
        best = None
        for c in candidates:
            model = LogisticRegression(C=c, class_weight="balanced", max_iter=3000,
                                       random_state=42)
            model.fit(train_features, train_y)
            probability = np.clip(model.predict_proba(val_features)[:, 1], 1e-7, 1 - 1e-7)
            loss = float(-np.mean(
                pos_weight * val_y * np.log(probability)
                + (1 - val_y) * np.log(1 - probability)
            ))
            if best is None or loss < best[0]:
                best = (loss, c, model)
        _, selected, model = best
        prediction = model.predict_proba(eval_features)[:, 1]
        parameter = {"C": selected}
    else:
        candidates = np.logspace(-3, 3, 13)
        best = None
        for alpha in candidates:
            model = Ridge(alpha=float(alpha), solver="svd")
            model.fit(train_features, train_y)
            loss = float(mean_squared_error(val_y, model.predict(val_features)))
            if best is None or loss < best[0]:
                best = (loss, float(alpha), model)
        _, selected, model = best
        prediction = model.predict(eval_features)
        parameter = {"alpha": selected}

    prediction_name = (
        "baseline_predictions.csv" if evaluation_split == "test"
        else "baseline_validation_predictions.csv"
    )
    pd.DataFrame({"subid": eval_ids, "target": eval_y, "prediction": prediction}).to_csv(
        output_dir / prediction_name, index=False
    )
    linear_init = {
        "weight": np.asarray(model.coef_, dtype=np.float32).reshape(-1),
        "bias": float(np.asarray(model.intercept_).reshape(-1)[0]),
    }
    return score(spec, eval_y, prediction), parameter, linear_init


def train_head(
    task, spec, arrays, args, output_dir, device, linear_init, stage2_metadata
):
    train_x, train_g, train_y, _ = arrays["train"]
    val_x, val_g, val_y, _ = arrays["val"]
    eval_x, eval_g, eval_y, eval_ids = arrays[args.evaluation_split]

    x_mean, x_std = fit_scaler(train_x)
    train_x, val_x, eval_x = [transform(x, x_mean, x_std) for x in (train_x, val_x, eval_x)]
    if train_g.shape[1]:
        g_mean, g_std = fit_scaler(train_g)
        # Keep the final presence indicator interpretable instead of centering it.
        g_mean[-1], g_std[-1] = 0.0, 1.0
        train_g, val_g, eval_g = [transform(g, g_mean, g_std) for g in (train_g, val_g, eval_g)]
    else:
        g_mean = g_std = np.empty(0, dtype=np.float32)

    if spec.kind == "regression":
        y_mean = float(train_y.mean())
        y_std = max(float(train_y.std()), 1e-6)
        train_target = (train_y - y_mean) / y_std
        val_target = (val_y - y_mean) / y_std
        loss_fn = nn.MSELoss()
    else:
        y_mean, y_std = 0.0, 1.0
        train_target, val_target = train_y, val_y
        positives = max(float(train_y.sum()), 1.0)
        pos_weight = torch.tensor([(len(train_y) - positives) / positives], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model_config = {
        "stage2_dim": int(train_x.shape[1]),
        "task_genetic_dim": int(train_g.shape[1]),
        "hidden_dims": tuple(args.hidden_dims),
        "dropout": args.dropout,
        "projection_dim": args.projection_dim,
        "use_layer_norm": args.layer_norm,
        "residual_linear": args.residual_linear,
    }
    model = DownstreamHead(**model_config).to(device)
    if model.linear_skip is None:
        raise RuntimeError("The overfit-resistant head requires --residual-linear")
    skip_weight = linear_init["weight"].copy()
    skip_bias = float(linear_init["bias"])
    if spec.kind == "regression":
        skip_weight /= y_std
        skip_bias = (skip_bias - y_mean) / y_std
    with torch.no_grad():
        model.linear_skip.weight.copy_(
            torch.from_numpy(skip_weight).to(device).reshape(1, -1)
        )
        model.linear_skip.bias.fill_(skip_bias)
        final_layer = next(
            layer for layer in reversed(model.head) if isinstance(layer, nn.Linear)
        )
        final_layer.weight.zero_()
        final_layer.bias.zero_()
    model.linear_skip.requires_grad_(False)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tensors = {
        "train_x": torch.from_numpy(train_x).to(device),
        "train_g": torch.from_numpy(train_g).to(device),
        "train_y": torch.from_numpy(train_target.astype(np.float32)).to(device),
        "val_x": torch.from_numpy(val_x).to(device),
        "val_g": torch.from_numpy(val_g).to(device),
        "val_y": torch.from_numpy(val_target.astype(np.float32)).to(device),
    }
    rng = np.random.default_rng(args.seed)
    model.eval()
    with torch.no_grad():
        val_genetic = tensors["val_g"] if train_g.shape[1] else None
        linear_val_loss = float(
            loss_fn(model(tensors["val_x"], val_genetic), tensors["val_y"])
        )
    best_loss = linear_val_loss
    observed_best_loss = linear_val_loss
    required_loss = linear_val_loss * (1.0 - args.min_relative_improvement)
    best_epoch, stale = 0, 0
    best_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    log_path = output_dir / "loss_log.csv"
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "is_best", "seconds"])
        writer.writerow([0, "", best_loss, 1, 0.0])
        log_file.flush()
        print(
            f"  epoch=0000 train=NA val={best_loss:.6f} "
            "(validation-selected linear initialization; "
            f"nonlinear selection requires val<={required_loss:.6f})",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            model.train()
            permutation = rng.permutation(len(train_y))
            total = 0.0
            for start in range(0, len(permutation), args.batch_size):
                index = torch.as_tensor(permutation[start:start + args.batch_size], device=device)
                genetic = tensors["train_g"][index] if train_g.shape[1] else None
                prediction = model(tensors["train_x"][index], genetic)
                loss = loss_fn(prediction, tensors["train_y"][index])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(index)
            train_loss = total / len(train_y)
            model.eval()
            with torch.no_grad():
                val_genetic = tensors["val_g"] if val_g.shape[1] else None
                val_loss = float(loss_fn(model(tensors["val_x"], val_genetic), tensors["val_y"]))
            observed_improvement = val_loss < observed_best_loss - 1e-6
            if observed_improvement:
                observed_best_loss, stale = val_loss, 0
            else:
                stale += 1
            is_meaningful = val_loss <= required_loss
            is_best = is_meaningful and val_loss < best_loss - 1e-6
            if is_best:
                best_loss, best_epoch = val_loss, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            elapsed = time.time() - started
            writer.writerow([epoch, train_loss, val_loss, int(is_best), elapsed])
            log_file.flush()
            if epoch == 1 or is_best or epoch % 20 == 0:
                print(f"  epoch={epoch:04d} train={train_loss:.6f} val={val_loss:.6f} "
                      f"best={best_loss:.6f} stale={stale}/{args.patience}", flush=True)
            if stale >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("No downstream checkpoint was selected")
    model.load_state_dict(best_state)
    checkpoint = {
        "task": task,
        "target_column": spec.target_column,
        "kind": spec.kind,
        "model": best_state,
        "model_config": model_config,
        "epoch": best_epoch,
        "val_loss": best_loss,
        "stage2_scaler": {"mean": x_mean, "std": x_std},
        "task_genetic_scaler": {"mean": g_mean, "std": g_std},
        "target_scaler": {"mean": y_mean, "std": y_std},
        "split_file": str(args.split_file),
        "labels": str(args.labels),
        "subject_dir": str(args.subject_dir),
        "stage2_representation": stage2_metadata,
        "task_genetic_dir": None,
        "evaluation_split": args.evaluation_split,
        "trainable_parameters": trainable_parameters,
        "optimization": {
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
            "linear_val_loss": linear_val_loss,
            "min_relative_improvement": args.min_relative_improvement,
            "required_nonlinear_val_loss": required_loss,
            "linear_skip_initialized_from_baseline": True,
            "linear_skip_frozen": True,
        },
    }
    torch.save(checkpoint, output_dir / "best.pt")

    model.to(device).eval()
    with torch.no_grad():
        eval_x_tensor = torch.from_numpy(eval_x).to(device)
        eval_g_tensor = torch.from_numpy(eval_g).to(device) if eval_g.shape[1] else None
        prediction = model(eval_x_tensor, eval_g_tensor).cpu().numpy()
    if spec.kind == "classification":
        prediction = 1.0 / (1.0 + np.exp(-prediction))
    else:
        prediction = prediction * y_std + y_mean
    prediction_name = (
        "predictions.csv" if args.evaluation_split == "test"
        else "validation_predictions.csv"
    )
    pd.DataFrame({"subid": eval_ids, "target": eval_y, "prediction": prediction}).to_csv(
        output_dir / prediction_name, index=False
    )
    return score(spec, eval_y, prediction), best_epoch, best_loss


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = tuple(args.tasks) or tuple(TASKS)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_existing_output:
        raise FileExistsError(
            f"Refusing to write into non-empty output directory: {args.output_dir}\n"
            "Choose a new --output-dir or pass --allow-existing-output."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    print(
        f"Downstream prediction device={device}; tasks={tasks}; "
        f"evaluation_split={args.evaluation_split}"
    )
    for task in tasks:
        print(f"\nTask: {task}", flush=True)
        dataset = DownstreamDataset(
            task, args.subject_dir, args.labels, args.split_file,
            task_genetic_dir=None,
        )
        arrays = {split: dataset.arrays(split) for split in ("train", "val", "test")}
        summary = dataset.summary()
        print(f"  cohort={summary}", flush=True)
        task_dir = args.output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        baseline_metrics, baseline_parameter, linear_init = baseline(
            task, dataset.spec, arrays, task_dir, args.evaluation_split
        )
        head_metrics, best_epoch, best_val_loss = train_head(
            task,
            dataset.spec,
            arrays,
            args,
            task_dir,
            device,
            linear_init,
            dataset.subject_metadata,
        )
        rows = []
        for model_name, metrics, extra in (
            ("linear_baseline", baseline_metrics, baseline_parameter),
            ("mlp_head", head_metrics, {"best_epoch": best_epoch, "best_val_loss": best_val_loss}),
        ):
            row = {
                "task": task,
                "model": model_name,
                "evaluation_split": args.evaluation_split,
                **metrics,
                **extra,
            }
            rows.append(row)
            all_metrics.append(row)
        pd.DataFrame(rows).to_csv(task_dir / "metrics.csv", index=False)
        with (task_dir / "cohort.json").open("w") as handle:
            json.dump(summary, handle, indent=2)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    pd.DataFrame(all_metrics).to_csv(args.output_dir / "all_tasks_metrics.csv", index=False)
    (args.output_dir / ".complete").write_text("complete\n")
    print(f"\nSaved downstream outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
