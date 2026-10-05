#!/usr/bin/env python3 -u
"""Export frozen `(K, 256)` genome-wide tokens for multimodal fusion."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ARTIFACTS, CHECKPOINTS, FUSION_GENETIC_DIR
from .data import RegionalDataset
from .model import GeneticRegionMAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINTS / "best.pt")
    parser.add_argument("--regional-dir", type=Path, default=ARTIFACTS / "regional")
    parser.add_argument("--output-dir", type=Path, default=FUSION_GENETIC_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--include-unassigned", action="store_true")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = RegionalDataset(
        args.regional_dir / "regional_features.npy",
        args.regional_dir / "samples.csv",
        split=None,
        mean=checkpoint["normalizer_mean"], std=checkpoint["normalizer_std"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeneticRegionMAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    loader = DataLoader(dataset, args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    offset = saved = 0
    with (args.output_dir / "manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subid", "split", "shape", "file"])
        with torch.no_grad():
            for regions in loader:
                tokens = model.encode(regions.to(device)).cpu().numpy().astype(np.float32)
                rows = dataset.samples.iloc[offset:offset + len(tokens)]
                for row, value in zip(rows.itertuples(index=False), tokens):
                    if row.split not in {"train", "val", "test"} and not args.include_unassigned:
                        continue
                    filename = f"{row.subid}.npy"
                    np.save(args.output_dir / filename, value)
                    writer.writerow([row.subid, row.split, "x".join(map(str, value.shape)), filename])
                    saved += 1
                offset += len(tokens)
    print(f"Saved {saved:,} genetic token files shaped "
          f"({model.n_latents}, {model.d_model}) to {args.output_dir}")


if __name__ == "__main__":
    main()
