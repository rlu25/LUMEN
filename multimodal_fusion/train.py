#!/usr/bin/env python3 -u
"""Train the leakage-controlled masked multimodal fusion autoencoder."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
from modality_pretraining.common import load_split_map, seed_everything

from multimodal_fusion.data import (
    DEFAULT_EMBEDDING_DIR, MODALITY_DIMS, MultiModalDataset, collate_tokens,
    fit_normalizer,
)
from multimodal_fusion.model import MultimodalFusionMAE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--output-dir", type=Path, default=HERE / "checkpoints")
    parser.add_argument("--modalities", nargs="+", choices=tuple(MODALITY_DIMS),
                        default=list(MODALITY_DIMS))
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.60)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--min-tokens", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-decoder", type=int, default=128)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--decoder-depth", type=int, default=4)
    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()
    total = count = 0
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for raw, mod_ids, visit_ids, slot_ids, padding in loader:
            raw, mod_ids = raw.to(device), mod_ids.to(device)
            visit_ids, slot_ids = visit_ids.to(device), slot_ids.to(device)
            padding = padding.to(device)
            loss = model(raw, mod_ids, visit_ids, slot_ids, padding, generator=generator)
            total += float(loss) * len(raw)
            count += len(raw)
    return total / max(count, 1)


def main():
    args = parse_args()
    if not 0 < args.mask_ratio < 1:
        raise ValueError("--mask-ratio must be between zero and one")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to write into non-empty output directory: {args.output_dir}\n"
            "Choose a new --output-dir."
        )
    seed_everything(args.seed)
    split_map = load_split_map()
    modalities = tuple(args.modalities)

    unnormalized_train = MultiModalDataset(
        args.embedding_dir, split_map, "train", modalities,
        min_tokens=args.min_tokens,
    )
    normalizer = fit_normalizer(unnormalized_train)
    train_ds = MultiModalDataset(
        args.embedding_dir, split_map, "train", modalities,
        min_tokens=args.min_tokens, normalizer=normalizer,
    )
    val_ds = MultiModalDataset(
        args.embedding_dir, split_map, "val", modalities,
        min_tokens=args.min_tokens, normalizer=normalizer,
    )
    if not train_ds or not val_ds:
        raise RuntimeError(f"Empty fusion split: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, args.batch_size, shuffle=True, drop_last=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_tokens,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_tokens,
        persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        "d_model": args.d_model,
        "d_decoder": args.d_decoder,
        "encoder_depth": args.encoder_depth,
        "decoder_depth": args.decoder_depth,
        "mask_ratio": args.mask_ratio,
        "masking_mode": "modality",
        "pooling_mode": "fusion_token",
        "fusion_mode": "modality_summary",
    }
    model = MultimodalFusionMAE(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "loss_log.csv"

    print(f"Fusion: train={len(train_ds):,}, val={len(val_ds):,}, device={device}")
    print(f"Modalities: {modalities}")
    print(
        "Foundation contract: one learned summary per modality, one shared "
        "fusion-token bottleneck, and reconstruction only through that bottleneck"
    )
    print(f"Train token counts: {train_ds.token_counts}")
    print(f"Validation token counts: {val_ds.token_counts}")

    best, stale = float("inf"), 0
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate", "is_best", "seconds"])
        log_file.flush()
        for epoch in range(1, args.epochs + 1):
            started = time.time()
            learning_rate = optimizer.param_groups[0]["lr"]
            model.train()
            total = count = 0
            for raw, mod_ids, visit_ids, slot_ids, padding in train_loader:
                raw, mod_ids = raw.to(device), mod_ids.to(device)
                visit_ids, slot_ids = visit_ids.to(device), slot_ids.to(device)
                padding = padding.to(device)
                loss = model(raw, mod_ids, visit_ids, slot_ids, padding)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(raw)
                count += len(raw)
            scheduler.step()
            train_loss = total / max(count, 1)
            val_loss = evaluate(model, val_loader, device)
            is_best = val_loss < best - 1e-6
            elapsed = time.time() - started
            writer.writerow([epoch, train_loss, val_loss, learning_rate, int(is_best), elapsed])
            log_file.flush()

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "model_config": model_config,
                "modalities": modalities,
                "visits": ("Y0", "Y2", "Y4", "Y6"),
                "normalizer": normalizer,
                "embedding_dir": str(args.embedding_dir),
                "split_file": str(HERE.parent / "data_splits" / "master_subject_split.csv"),
            }
            if is_best:
                best, stale = val_loss, 0
            else:
                stale += 1
            checkpoint["best_val_loss"] = best
            if is_best:
                torch.save(checkpoint, args.output_dir / "best.pt")
                torch.save(model.state_dict(), args.output_dir / "model.pt")
            if epoch % args.checkpoint_every == 0:
                periodic_path = args.output_dir / f"epoch_{epoch:04d}.pt"
                torch.save(checkpoint, periodic_path)
                print(f"Saved periodic checkpoint: {periodic_path}", flush=True)
            print(
                f"epoch={epoch:04d} train={train_loss:.6f} val={val_loss:.6f} "
                f"best={best:.6f} stale={stale}/{args.patience} ({elapsed:.0f}s)",
                flush=True,
            )
            if stale >= args.patience:
                print(f"Early stop at epoch {epoch}", flush=True)
                break
    (args.output_dir / ".complete").write_text("complete\n")
    print(f"Done. Best validation loss={best:.6f}; checkpoint={args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
