#!/usr/bin/env python3 -u
"""Extract frozen modality embeddings into the multimodal-fusion contract."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

from modality_pretraining.common import ALLOWED_VISITS, canonical_subid
from modality_pretraining.bold.model import BOLDMAEEncoder, T_WIN, load_window
from modality_pretraining.fitbit.pretrain import (
    CACHE,
    MIN_OBSERVED_HOURS_PER_DAY,
    MIN_VALID_DAYS,
    build_visit_manifest,
    dense_window,
    manifest_entries,
    tokenize,
)
from modality_pretraining.fitbit.model import FitbitEncoder
from modality_pretraining.rsfc.model import FCTransformerEncoder, N_ROIS


MODALITY_ROOT = ROOT / "modality_pretraining"
RSFC_DIR = Path(os.environ.get("LUMEN_RSFC_DIR", "data/rsfc"))
BOLD_DIR = Path(os.environ.get("LUMEN_BOLD_DIR", "data/bold"))
SMRI_DIR = Path(os.environ.get("LUMEN_SMRI_EMBEDDING_DIR", "data/smri_embeddings"))


def save_batch(paths, values):
    for path, value in zip(paths, values):
        np.save(path, np.asarray(value, dtype=np.float32))


def extract_rsfc(output: Path, device, batch_size: int):
    destination = output / "rsfc"
    destination.mkdir(parents=True, exist_ok=True)
    encoder = FCTransformerEncoder().to(device)
    encoder.load_state_dict(torch.load(MODALITY_ROOT / "rsfc" / "encoder.pt", map_location=device))
    encoder.eval()
    stats = np.load(MODALITY_ROOT / "rsfc" / "normalizer.npz")
    edge_mean = stats["edge_mean"].astype(np.float32)
    roi_pairs = np.load(RSFC_DIR / "roi_pairs.npy")
    ii, jj = roi_pairs[:, 0], roi_pairs[:, 1]

    for visit in ALLOWED_VISITS:
        key = visit.lower()
        values = np.load(RSFC_DIR / f"{key}_X.npy", mmap_mode="r")
        subjects = np.load(RSFC_DIR / f"{key}_pids.npy", allow_pickle=True)
        batch, paths = [], []

        def flush():
            if not batch:
                return
            tensor = torch.from_numpy(np.stack(batch)).to(device)
            with torch.no_grad():
                embeddings, _ = encoder(tensor, mask_roi=None)
            save_batch(paths, embeddings.cpu().numpy())
            batch.clear(); paths.clear()

        for index, subject in enumerate(subjects):
            subid = canonical_subid(subject)
            path = destination / f"{subid}_{visit}.npy"
            if path.exists():
                continue
            flat = np.asarray(values[index], dtype=np.float32).copy()
            if np.isnan(flat).all():
                continue
            bad = ~np.isfinite(flat)
            flat[bad] = edge_mean[bad]
            flat = np.arctanh(np.clip(flat, -0.9999, 0.9999))
            matrix = np.zeros((N_ROIS, N_ROIS), dtype=np.float32)
            matrix[ii, jj] = flat
            matrix[jj, ii] = flat
            matrix = (matrix - matrix.mean()) / max(float(matrix.std()), 1e-6)
            batch.append(matrix); paths.append(path)
            if len(batch) >= batch_size:
                flush()
        flush()
        print(f"rsFC {visit}: {len(list(destination.glob(f'*_{visit}.npy'))):,} embeddings")


def extract_fitbit(output: Path, device, batch_size: int, checkpoint_dir: Path):
    destination = output / "fitbit"
    destination.mkdir(parents=True, exist_ok=True)
    encoder = FitbitEncoder().to(device)
    encoder.load_state_dict(torch.load(checkpoint_dir / "encoder.pt", map_location=device))
    encoder.eval()
    stats = np.load(checkpoint_dir / "normalizer.npz")
    means, stds = stats["means"], stats["stds"]
    min_valid_days = int(stats["min_valid_days"]) \
        if "min_valid_days" in stats.files else MIN_VALID_DAYS
    min_hours = int(stats["min_observed_hours_per_day"]) \
        if "min_observed_hours_per_day" in stats.files else MIN_OBSERVED_HOURS_PER_DAY
    manifest = build_visit_manifest(
        min_valid_days=min_valid_days,
        min_observed_hours_per_day=min_hours,
    )
    entries = manifest_entries(manifest)
    manifest.to_csv(destination / "visit_manifest.csv", index=False)
    batch, paths = [], []

    def flush():
        if not batch:
            return
        tensor = torch.from_numpy(np.stack(batch)).to(device)
        with torch.no_grad():
            embeddings = encoder(tensor)
        save_batch(paths, embeddings.cpu().numpy())
        batch.clear(); paths.clear()

    for index, (subid, visit, start_hour) in enumerate(entries):
        path = destination / f"{subid}_{visit}.npy"
        if path.exists():
            continue
        feat_path, time_path = CACHE / f"{subid}.npy", CACHE / f"{subid}_ts.npy"
        if not feat_path.exists() or not time_path.exists():
            continue
        window = dense_window(
            np.load(feat_path).astype(np.float32), np.load(time_path),
            np.random.default_rng(index), deterministic=True,
            start_hour=start_hour,
        )
        batch.append(tokenize(window, means, stds)); paths.append(path)
        if len(batch) >= batch_size:
            flush()
    flush()
    print(
        f"Fitbit accepted visits: {len(entries):,}; "
        f"embeddings in destination: {len(list(destination.glob('NDAR_INV*.npy'))):,}"
    )


def extract_bold(output: Path, device, batch_size: int):
    destination = output / "bold"
    destination.mkdir(parents=True, exist_ok=True)
    encoder = BOLDMAEEncoder().to(device)
    encoder.load_state_dict(torch.load(MODALITY_ROOT / "bold" / "encoder.pt", map_location=device))
    encoder.eval()
    for visit in ALLOWED_VISITS:
        batch, paths = [], []

        def flush():
            if not batch:
                return
            tensor = torch.from_numpy(np.stack(batch)).to(device)
            with torch.no_grad():
                embeddings = encoder.encode(tensor)
            save_batch(paths, embeddings.cpu().numpy())
            batch.clear(); paths.clear()

        for source in sorted(BOLD_DIR.glob(f"sub-*_{visit}.npy")):
            subid = canonical_subid(source.stem.split("_")[0])
            path = destination / f"{subid}_{visit}.npy"
            if path.exists():
                continue
            shape = np.load(source, mmap_mode="r").shape
            if len(shape) != 2 or shape[0] != 366 or shape[1] < T_WIN:
                continue
            value = load_window(source, rng=None)
            if not np.isfinite(value).all():
                continue
            batch.append(value); paths.append(path)
            if len(batch) >= batch_size:
                flush()
        flush()
        print(f"BOLD {visit}: {len(list(destination.glob(f'*_{visit}.npy'))):,} embeddings")


def link_smri(output: Path):
    destination = output / "smri"
    destination.mkdir(parents=True, exist_ok=True)
    linked = 0
    for source in sorted(SMRI_DIR.glob("NDAR_INV*.npy")):
        parts = source.stem.split("_")
        if len(parts) != 3 or parts[2] not in ALLOWED_VISITS:
            continue
        target = destination / source.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source)
        linked += 1
    print(f"sMRI: linked {linked:,}; total={len(list(destination.glob('NDAR_INV*.npy'))):,}")


def main():
    parser = argparse.ArgumentParser()
    available_modalities = ("rsfc", "fitbit", "bold", "smri")
    parser.add_argument("modalities", nargs="*", metavar="MODALITY")
    parser.add_argument("--output-dir", type=Path, default=HERE / "embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--fitbit-checkpoint-dir",
        type=Path,
        default=MODALITY_ROOT / "fitbit",
        help="Fitbit checkpoint directory used when extracting Fitbit embeddings.",
    )
    args = parser.parse_args()
    invalid = sorted(set(args.modalities) - set(available_modalities))
    if invalid:
        parser.error(
            f"invalid modalities: {', '.join(invalid)} "
            f"(choose from {', '.join(available_modalities)})"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; output={args.output_dir}")
    actions = {
        "rsfc": lambda: extract_rsfc(args.output_dir, device, args.batch_size),
        "fitbit": lambda: extract_fitbit(
            args.output_dir, device, args.batch_size, args.fitbit_checkpoint_dir
        ),
        # Full BOLD inference attends over 3,660 patch tokens; batch one avoids
        # the quadratic attention matrix exhausting GPU memory.
        "bold": lambda: extract_bold(args.output_dir, device, 1),
        "smri": lambda: link_smri(args.output_dir),
    }
    modalities = args.modalities or tuple(actions)
    for modality in modalities:
        actions[modality]()


if __name__ == "__main__":
    main()
