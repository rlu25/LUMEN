#!/usr/bin/env python3 -u
"""Pretrain general genetic tokens with masked regional reconstruction."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ARTIFACTS, CHECKPOINTS, N_LATENTS, REGION_MODEL_DIM
from .data import RegionalDataset, fit_normalizer
from .model import GeneticRegionMAE
from .utils import seed_everything


def evaluate(model, loader, device):
    model.eval(); total = count = 0
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for regions in loader:
            regions = regions.to(device)
            loss = model(regions, generator=generator)
            total += float(loss) * len(regions); count += len(regions)
    return total / max(count, 1)


def save_checkpoint(checkpoint, path: Path):
    """Atomically replace a checkpoint so interrupted writes do not corrupt it."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regional-dir", type=Path, default=ARTIFACTS / "regional")
    parser.add_argument("--output-dir", type=Path, default=CHECKPOINTS)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.40)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-latents", type=int, default=N_LATENTS)
    parser.add_argument("--d-model", type=int, default=REGION_MODEL_DIM)
    parser.add_argument("--latent-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()
    if not 0 < args.mask_ratio < 1:
        raise ValueError("--mask-ratio must lie between zero and one")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    seed_everything(args.seed)
    feature_path = args.regional_dir / "regional_features.npy"
    sample_path = args.regional_dir / "samples.csv"
    raw_train = RegionalDataset(feature_path, sample_path, "train")
    mean, std = fit_normalizer(raw_train)
    train_ds = RegionalDataset(feature_path, sample_path, "train", mean, std)
    val_ds = RegionalDataset(feature_path, sample_path, "val", mean, std)
    if not train_ds or not val_ds:
        raise RuntimeError(f"Empty genetic split: train={len(train_ds)}, val={len(val_ds)}")
    n_regions, input_dim = train_ds.shape[1:]
    config = {
        "n_regions": n_regions,
        "input_dim": input_dim,
        "d_model": args.d_model,
        "n_latents": args.n_latents,
        "latent_depth": args.latent_depth,
        "mask_ratio": args.mask_ratio,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeneticRegionMAE(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Genetic MAE: train={len(train_ds):,}, val={len(val_ds):,}, "
          f"regions={n_regions:,}, input_dim={input_dim}, device={device}")

    best, stale = float("inf"), 0
    with (args.output_dir / "loss_log.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate", "is_best", "seconds"])
        stream.flush()
        for epoch in range(1, args.epochs + 1):
            started = time.time(); model.train(); total = count = 0
            learning_rate = optimizer.param_groups[0]["lr"]
            for regions in train_loader:
                regions = regions.to(device)
                loss = model(regions)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(regions); count += len(regions)
            scheduler.step()
            train_loss = total / max(count, 1)
            val_loss = evaluate(model, val_loader, device)
            is_best = val_loss < best - 1e-6
            elapsed = time.time() - started
            writer.writerow([epoch, train_loss, val_loss, learning_rate, int(is_best), elapsed])
            stream.flush()
            if is_best:
                best, stale = val_loss, 0
            else:
                stale += 1

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "model_config": config,
                "normalizer_mean": mean,
                "normalizer_std": std,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best,
                "stale_epochs": stale,
                "epochs_requested": args.epochs,
                "regional_dir": str(args.regional_dir),
            }
            save_checkpoint(checkpoint, args.output_dir / "latest.pt")
            if is_best:
                save_checkpoint(checkpoint, args.output_dir / "best.pt")
            if epoch % args.save_every == 0:
                save_checkpoint(
                    checkpoint,
                    args.output_dir / f"epoch_{epoch:04d}.pt",
                )
            print(f"epoch={epoch:04d} train={train_loss:.6f} val={val_loss:.6f} "
                  f"best={best:.6f} stale={stale}/{args.patience} ({elapsed:.0f}s)", flush=True)
            if stale >= args.patience:
                print(f"Early stop at epoch {epoch}"); break
    print(f"Done. Best validation loss={best:.6f}")


if __name__ == "__main__":
    main()
