#!/usr/bin/env python3 -u
"""Label-free BOLD MAE pretraining on all Y0/Y2/Y4/Y6 development scans.

Contrastive positives come from the same scan, preventing the encoder from
being explicitly trained to erase developmental differences across visits.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from modality_pretraining.common import (
    ALLOWED_VISITS, canonical_subid, load_split_map, seed_everything,
)
from modality_pretraining.bold.model import (
    BOLDMAEDecoder, BOLDMAEEncoder, MASK_RATIO, T_WIN, load_window,
    mae_loss, nt_xent_loss,
)


BOLD_DIR = Path(os.environ.get("LUMEN_BOLD_DIR", "data/bold"))


class SameScanPairDataset(Dataset):
    def __init__(self, files, seed=42):
        self.files = files
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        return (torch.from_numpy(load_window(fp, self.rng)),
                torch.from_numpy(load_window(fp, self.rng)))


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = random.Random(info.seed)


def inventory(split_map, split):
    files = []
    for fp in BOLD_DIR.glob("sub-*_Y*.npy"):
        visit = fp.stem.split("_")[-1]
        subid = canonical_subid(fp.stem.split("_")[0])
        if visit not in ALLOWED_VISITS or split_map.get(subid) != split:
            continue
        # Padding is not represented in the current MAE attention/loss masks;
        # exclude short or malformed scans rather than training on fake frames.
        try:
            shape = np.load(fp, mmap_mode="r").shape
        except Exception:
            continue
        if len(shape) == 2 and shape[0] == 366 and shape[1] >= T_WIN:
            files.append(fp)
    return sorted(files)


@torch.no_grad()
def validation_loss(encoder, decoder, projector, loader, device, mask_ratio, lambda_cl):
    encoder.eval(); decoder.eval(); projector.eval(); total = n = 0
    for a, b in loader:
        a, b = a.to(device), b.to(device)
        z, visible, keep, restore = encoder(a, mask_ratio)
        recon = decoder(visible, keep, restore)
        loss = mae_loss(recon, a, keep)
        z2, _, _, _ = encoder(b, mask_ratio)
        loss = loss + lambda_cl * nt_xent_loss(projector(z), projector(z2))
        total += float(loss) * len(a); n += len(a)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mask-ratio", type=float, default=MASK_RATIO)
    ap.add_argument("--lambda-cl", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_map = load_split_map()
    train_ds = SameScanPairDataset(inventory(split_map, "train"), 42)
    val_ds = SameScanPairDataset(inventory(split_map, "val"), 43)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, worker_init_fn=worker_init,
                              persistent_workers=args.workers > 0, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, drop_last=False,
                            num_workers=0, pin_memory=True)
    encoder, decoder = BOLDMAEEncoder().to(device), BOLDMAEDecoder().to(device)
    # The contrastive head is discarded after pretraining so the final CLS
    # representation is not forced directly into the contrastive geometry.
    projector = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 64)).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    out = Path(__file__).resolve().parent; out.mkdir(parents=True, exist_ok=True)
    loss_log = (out / "loss_log.csv").open("w", newline="")
    loss_writer = csv.writer(loss_log)
    loss_writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate", "is_best"])
    loss_log.flush()
    print(f"BOLD scans: train={len(train_ds):,} val={len(val_ds):,}; device={device}")

    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        learning_rate = opt.param_groups[0]["lr"]
        encoder.train(); decoder.train(); projector.train(); losses = []
        for a, b in train_loader:
            a, b = a.to(device), b.to(device)
            z, visible, keep, restore = encoder(a, args.mask_ratio)
            loss_mae = mae_loss(decoder(visible, keep, restore), a, keep)
            z2, _, _, _ = encoder(b, args.mask_ratio)
            loss = loss_mae + args.lambda_cl * nt_xent_loss(projector(z), projector(z2))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); losses.append(float(loss.detach()))
        scheduler.step()
        val = validation_loss(encoder, decoder, projector, val_loader, device,
                              args.mask_ratio, args.lambda_cl)
        train_loss = float(np.mean(losses))
        is_best = val < best - 1e-5
        loss_writer.writerow([epoch, train_loss, val, learning_rate, int(is_best)])
        loss_log.flush()
        print(f"epoch={epoch:04d} train={train_loss:.6f} val={val:.6f}", flush=True)
        if is_best:
            best, stale = val, 0
            torch.save(encoder.state_dict(), out / "encoder.pt")
            torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(), "val_loss": val,
                        "projection": projector.state_dict(),
                        "visits": ALLOWED_VISITS, "positive_pair": "same_scan"}, out / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    loss_log.close()
    print(f"Best validation objective: {best:.6f}")


if __name__ == "__main__":
    main()
