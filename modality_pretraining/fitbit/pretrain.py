#!/usr/bin/env python3 -u
"""Leakage-controlled Fitbit contrastive pretraining.

Each positive pair is two corruptions of the same real-time 168-hour window.
Timestamp gaps are retained as missing hours instead of being compressed away.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from modality_pretraining.common import canonical_subid, load_split_map, seed_everything
from modality_pretraining.fitbit.model import (
    D_IN, N_FEAT, WIN_HOURS, FitbitWBM, infonce_loss,
)


CACHE = Path(os.environ.get("LUMEN_FITBIT_CACHE", "data/fitbit_hourly_cache"))
FITBIT_TABLE = Path(os.environ.get("LUMEN_FITBIT_TABLE", "data/nt_y_fitb_act_d.csv"))
EVENT_TO_VISIT = {
    "baseline_year_1_arm_1": "Y0",
    "2_year_follow_up_y_arm_1": "Y2",
    "4_year_follow_up_y_arm_1": "Y4",
}
MIN_VALID_DAYS = 4
MIN_OBSERVED_HOURS_PER_DAY = 10


def protocol_start_hour(value) -> int:
    """Convert a timezone-naive protocol date to the cache's epoch-hour convention."""
    return int(pd.Timestamp(value).value // (3600 * 10**9))


def build_visit_manifest(
    split_map=None,
    min_valid_days=MIN_VALID_DAYS,
    min_observed_hours_per_day=MIN_OBSERVED_HOURS_PER_DAY,
):
    """Audit every protocol visit against real cache timestamps and wear coverage."""
    dates = pd.read_csv(
        FITBIT_TABLE,
        usecols=["src_subject_id", "eventname", "fit_ss_protocol_startdate"],
        low_memory=False,
    )
    dates = dates[dates["eventname"].isin(EVENT_TO_VISIT)].copy()
    dates["date"] = pd.to_datetime(
        dates["fit_ss_protocol_startdate"], errors="coerce", format="mixed"
    )
    dates = dates.dropna(subset=["date"]).drop_duplicates(["src_subject_id", "eventname"])
    rows = []
    for row in dates.itertuples(index=False):
        pid = canonical_subid(row.src_subject_id)
        visit = EVENT_TO_VISIT[row.eventname]
        start_hour = protocol_start_hour(row.date)
        feat_path, time_path = CACHE / f"{pid}.npy", CACHE / f"{pid}_ts.npy"
        observed_hours = valid_days = hr_hours = sleep_hours = 0
        accepted = False
        reason = "missing_cache"
        if feat_path.exists() and time_path.exists():
            try:
                feat = np.load(feat_path, mmap_mode="r")
                timestamps = np.asarray(np.load(time_path), dtype=np.int64)
                if feat.ndim != 2 or feat.shape[1] != N_FEAT:
                    reason = "invalid_feature_shape"
                elif timestamps.ndim != 1 or len(timestamps) != len(feat):
                    reason = "timestamp_length_mismatch"
                else:
                    hours = timestamps // 3600
                    inside = (hours >= start_hour) & (hours < start_hour + WIN_HOURS)
                    relative = np.unique(hours[inside]) - start_hour
                    observed_hours = int(len(relative))
                    if observed_hours:
                        hours_per_day = np.bincount(
                            relative // 24, minlength=WIN_HOURS // 24
                        )
                        valid_days = int(
                            np.sum(hours_per_day >= min_observed_hours_per_day)
                        )
                        hr_hours = int(np.isfinite(feat[inside, 4]).sum())
                        sleep_hours = int(np.isfinite(feat[inside, 5:]).any(axis=1).sum())
                        accepted = valid_days >= min_valid_days
                        reason = "accepted" if accepted else "insufficient_valid_days"
                    else:
                        reason = "no_timestamp_overlap"
            except (OSError, ValueError) as exc:
                reason = f"cache_read_error:{type(exc).__name__}"
        rows.append({
            "subject_id": pid,
            "split": split_map.get(pid, "unassigned") if split_map is not None else "unassigned",
            "visit": visit,
            "protocol_start": row.date.isoformat(),
            "start_hour": start_hour,
            "observed_hours": observed_hours,
            "valid_days": valid_days,
            "hr_hours": hr_hours,
            "sleep_hours": sleep_hours,
            "accepted": accepted,
            "reason": reason,
        })
    return pd.DataFrame(rows).sort_values(["subject_id", "visit"]).reset_index(drop=True)


def manifest_entries(manifest, split=None):
    selected = manifest[manifest["accepted"]]
    if split is not None:
        selected = selected[selected["split"] == split]
    return sorted(
        (row.subject_id, row.visit, int(row.start_hour))
        for row in selected.itertuples(index=False)
    )


def visit_entries(
    split_map,
    split,
    min_valid_days=MIN_VALID_DAYS,
    min_observed_hours_per_day=MIN_OBSERVED_HOURS_PER_DAY,
):
    manifest = build_visit_manifest(
        split_map, min_valid_days, min_observed_hours_per_day
    )
    return manifest_entries(manifest, split)


def fit_normalizer(entries):
    sums = np.zeros(N_FEAT, dtype=np.float64)
    sqs = np.zeros(N_FEAT, dtype=np.float64)
    counts = np.zeros(N_FEAT, dtype=np.int64)
    fixed_rng = np.random.default_rng(0)
    for pid, _, start_hour in entries:
        feat = np.load(CACHE / f"{pid}.npy").astype(np.float32)
        ts = np.load(CACHE / f"{pid}_ts.npy")
        x = dense_window(feat, ts, fixed_rng, start_hour=start_hour).astype(np.float64)
        ok = np.isfinite(x)
        sums += np.where(ok, x, 0).sum(0)
        sqs += np.where(ok, x * x, 0).sum(0)
        counts += ok.sum(0)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    var = np.divide(sqs, counts, out=np.ones_like(sqs), where=counts > 0) - means ** 2
    return means.astype(np.float32), np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)


def dense_window(feat, timestamps, rng, deterministic=False, start_hour=None):
    """Return a true hourly grid; absent/non-wear hours remain NaN."""
    ts = np.asarray(timestamps, dtype=np.int64)
    order = np.argsort(ts)
    ts, feat = ts[order], feat[order]
    if len(ts) == 0:
        return np.full((WIN_HOURS, N_FEAT), np.nan, dtype=np.float32)
    if start_hour is None:
        anchor_idx = len(ts) // 2 if deterministic else int(rng.integers(len(ts)))
        offset = WIN_HOURS // 2 if deterministic else int(rng.integers(WIN_HOURS))
        start = int(ts[anchor_idx] // 3600) - offset
    else:
        start = int(start_hour)
    wanted = np.arange(start, start + WIN_HOURS, dtype=np.int64)
    observed = ts // 3600
    pos = np.searchsorted(observed, wanted)
    valid = (pos < len(observed)) & (observed[np.minimum(pos, len(observed) - 1)] == wanted)
    out = np.full((WIN_HOURS, N_FEAT), np.nan, dtype=np.float32)
    out[valid] = feat[pos[valid]]
    return out


def tokenize(window, means, stds):
    missing = ~np.isfinite(window)
    values = (window - means) / stds
    values[missing] = 0.0
    return np.concatenate([values, missing.astype(np.float32)], axis=-1).astype(np.float32)


def augment(tokens, rng, drop_rate):
    x = tokens.copy()
    drop = rng.random(len(x)) < drop_rate
    x[drop, :N_FEAT] = 0.0
    x[drop, N_FEAT:] = 1.0
    return x


class FitbitDataset(Dataset):
    def __init__(self, entries, means, stds, drop_rate, seed, deterministic=False):
        self.entries, self.means, self.stds = entries, means, stds
        self.drop_rate, self.seed, self.deterministic = drop_rate, seed, deterministic
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        pid, _, start_hour = self.entries[idx]
        feat = np.load(CACHE / f"{pid}.npy").astype(np.float32)
        ts = np.load(CACHE / f"{pid}_ts.npy")
        rng = np.random.default_rng(self.seed + idx) if self.deterministic else self.rng
        base = tokenize(dense_window(feat, ts, rng, self.deterministic,
                                     start_hour=start_hour), self.means, self.stds)
        return (torch.from_numpy(augment(base, rng, self.drop_rate)),
                torch.from_numpy(augment(base, rng, self.drop_rate)))


def worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed)


@torch.no_grad()
def validation_loss(model, loader, device, temperature):
    model.eval(); total = n = 0
    for a, b in loader:
        a, b = a.to(device), b.to(device)
        loss = infonce_loss(model(a), model(b), temperature)
        total += float(loss) * len(a); n += len(a)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--drop-rate", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Checkpoint directory; use a new directory to preserve an older run.",
    )
    ap.add_argument("--min-valid-days", type=int, default=MIN_VALID_DAYS)
    ap.add_argument(
        "--min-observed-hours-per-day",
        type=int,
        default=MIN_OBSERVED_HOURS_PER_DAY,
    )
    args = ap.parse_args()
    if not 1 <= args.min_valid_days <= WIN_HOURS // 24:
        ap.error("--min-valid-days must be between 1 and 7")
    if not 1 <= args.min_observed_hours_per_day <= 24:
        ap.error("--min-observed-hours-per-day must be between 1 and 24")
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_map = load_split_map()
    manifest = build_visit_manifest(
        split_map, args.min_valid_days, args.min_observed_hours_per_day
    )
    train_entries = manifest_entries(manifest, "train")
    val_entries = manifest_entries(manifest, "val")
    means, stds = fit_normalizer(train_entries)
    train_ds = FitbitDataset(train_entries, means, stds, args.drop_rate, 42)
    val_ds = FitbitDataset(val_entries, means, stds, args.drop_rate, 43, deterministic=True)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, worker_init_fn=worker_init,
                              persistent_workers=args.workers > 0, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, drop_last=True,
                            num_workers=args.workers, worker_init_fn=worker_init,
                            persistent_workers=args.workers > 0, pin_memory=True)

    model = FitbitWBM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "visit_manifest.csv", index=False)
    np.savez(out / "normalizer.npz", means=means, stds=stds,
             window_hours=WIN_HOURS, timestamp_unit="seconds",
             visits=np.array(sorted(manifest.loc[manifest.accepted, "visit"].unique())),
             min_valid_days=args.min_valid_days,
             min_observed_hours_per_day=args.min_observed_hours_per_day)
    accepted_counts = manifest.loc[manifest.accepted].groupby(["split", "visit"]).size()
    print("Accepted Fitbit visit-windows by split and visit:")
    print(accepted_counts.to_string())
    print(f"Fitbit visit-windows: train={len(train_ds):,} val={len(val_ds):,}; device={device}")
    loss_log = (out / "loss_log.csv").open("w", newline="")
    loss_writer = csv.writer(loss_log)
    loss_writer.writerow(["epoch", "train_loss", "val_loss", "learning_rate", "is_best"])
    loss_log.flush()

    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        learning_rate = opt.param_groups[0]["lr"]
        model.train(); losses = []
        for a, b in train_loader:
            a, b = a.to(device), b.to(device)
            loss = infonce_loss(model(a), model(b), args.temperature)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(float(loss.detach()))
        scheduler.step()
        val = validation_loss(model, val_loader, device, args.temperature)
        train_loss = float(np.mean(losses))
        is_best = val < best - 1e-5
        loss_writer.writerow([epoch, train_loss, val, learning_rate, int(is_best)])
        loss_log.flush()
        print(f"epoch={epoch:04d} train={train_loss:.6f} val={val:.6f}", flush=True)
        if is_best:
            best, stale = val, 0
            torch.save(model.encoder.state_dict(), out / "encoder.pt")
            torch.save({"epoch": epoch, "encoder": model.encoder.state_dict(),
                        "projection": model.proj.state_dict(), "val_loss": val}, out / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    loss_log.close()
    print(f"Best validation contrastive loss: {best:.6f}")


if __name__ == "__main__":
    main()
