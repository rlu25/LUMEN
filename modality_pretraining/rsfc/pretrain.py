#!/usr/bin/env python3 -u
"""Prospective, label-free rsFC masked-reconstruction pretraining.

All available Y0/Y2/Y4/Y6 scans from master-split training families update weights.
Checkpoint selection uses reconstruction loss on validation families.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from modality_pretraining.common import (
    ALLOWED_VISITS, canonical_subid, load_split_map, seed_everything,
)
from modality_pretraining.rsfc.model import FCTransformerEncoder, N_ROIS, make_mask


RSFC = Path(os.environ.get("LUMEN_RSFC_DIR", "data/rsfc"))


class SymmetricFCMAE(nn.Module):
    """Hide both row and column for every masked ROI before reconstruction."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        d, n = encoder.d_model, encoder.n_rois
        self.decoder = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, n))

    def forward(self, x, mask):
        corrupted = x.masked_fill(mask.unsqueeze(-1), 0.0)
        corrupted = corrupted.masked_fill(mask.unsqueeze(1), 0.0)
        emb, tokens = self.encoder(corrupted, mask_roi=mask)
        recon = self.decoder(tokens)
        weight = mask.unsqueeze(-1).float()
        loss = ((recon - x) ** 2 * weight).sum() / (weight.sum() * x.shape[-1]).clamp(1)
        return loss, emb


def build_index(split_map):
    arrays, rows = {}, []
    for visit in ALLOWED_VISITS:
        key = visit.lower()
        x = np.load(RSFC / f"{key}_X.npy", mmap_mode="r")
        pids = np.load(RSFC / f"{key}_pids.npy", allow_pickle=True)
        arrays[visit] = x
        for i, pid in enumerate(pids):
            subid = canonical_subid(pid)
            split = split_map.get(subid)
            if split in {"train", "val"} and not np.isnan(x[i]).all():
                rows.append((visit, i, subid, split))
    return arrays, rows


def fit_edge_means(arrays, rows, chunk_size=256):
    n_edges = next(iter(arrays.values())).shape[1]
    sums = np.zeros(n_edges, dtype=np.float64)
    counts = np.zeros(n_edges, dtype=np.int64)
    train = [(v, i) for v, i, _, s in rows if s == "train"]
    for start in range(0, len(train), chunk_size):
        block = np.stack([np.asarray(arrays[v][i], dtype=np.float32) for v, i in train[start:start + chunk_size]])
        finite = np.isfinite(block)
        sums += np.where(finite, block, 0).sum(axis=0)
        counts += finite.sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return means.astype(np.float32)


class RSFCDataset(Dataset):
    def __init__(self, arrays, rows, edge_means, roi_pairs):
        self.arrays, self.rows, self.edge_means = arrays, rows, edge_means
        self.ii, self.jj = roi_pairs[:, 0], roi_pairs[:, 1]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        visit, row, _, _ = self.rows[index]
        flat = np.asarray(self.arrays[visit][row], dtype=np.float32).copy()
        bad = ~np.isfinite(flat)
        flat[bad] = self.edge_means[bad]
        flat = np.arctanh(np.clip(flat, -0.9999, 0.9999))
        mat = np.zeros((N_ROIS, N_ROIS), dtype=np.float32)
        mat[self.ii, self.jj] = flat
        mat[self.jj, self.ii] = flat
        mean, std = mat.mean(), max(float(mat.std()), 1e-6)
        return torch.from_numpy((mat - mean) / std)


@torch.no_grad()
def validation_loss(model, loader, device, mask_ratio):
    model.eval()
    total, n = 0.0, 0
    for x in loader:
        x = x.to(device)
        loss, _ = model(x, make_mask(len(x), N_ROIS, mask_ratio, device))
        total += float(loss) * len(x)
        n += len(x)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mask-ratio", type=float, default=0.30)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_map = load_split_map()
    arrays, rows = build_index(split_map)
    roi_pairs = np.load(RSFC / "roi_pairs.npy")
    means = fit_edge_means(arrays, rows)
    train_rows = [r for r in rows if r[3] == "train"]
    val_rows = [r for r in rows if r[3] == "val"]
    train_ds = RSFCDataset(arrays, train_rows, means, roi_pairs)
    val_ds = RSFCDataset(arrays, val_rows, means, roi_pairs)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    encoder = FCTransformerEncoder().to(device)
    model = SymmetricFCMAE(encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    out = Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "normalizer.npz", edge_mean=means, visits=np.array(ALLOWED_VISITS))
    loss_log = (out / "loss_log.csv").open("w", newline="")
    loss_writer = csv.writer(loss_log)
    loss_writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate", "is_best"])
    loss_log.flush()
    print(f"rsFC scans: train={len(train_ds):,} val={len(val_ds):,}; device={device}")

    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        learning_rate = opt.param_groups[0]["lr"]
        model.train(); losses = []
        for x in train_loader:
            x = x.to(device)
            loss, _ = model(x, make_mask(len(x), N_ROIS, args.mask_ratio, device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(float(loss.detach()))
        scheduler.step()
        val = validation_loss(model, val_loader, device, args.mask_ratio)
        train_loss = float(np.mean(losses))
        is_best = val < best - 1e-5
        loss_writer.writerow([epoch, train_loss, val, learning_rate, int(is_best)])
        loss_log.flush()
        print(f"epoch={epoch:04d} train={train_loss:.6f} val={val:.6f}", flush=True)
        if is_best:
            best, stale = val, 0
            torch.save(encoder.state_dict(), out / "encoder.pt")
            torch.save({"epoch": epoch, "encoder": encoder.state_dict(),
                        "decoder": model.decoder.state_dict(), "val_loss": val,
                        "visits": ALLOWED_VISITS}, out / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    loss_log.close()
    print(f"Best validation reconstruction loss: {best:.6f}")


if __name__ == "__main__":
    main()
