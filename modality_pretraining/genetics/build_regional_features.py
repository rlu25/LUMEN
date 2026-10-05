#!/usr/bin/env python3 -u
"""PCA-compress SNP delta vectors and pool standardized dosages by region."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin
from sklearn.decomposition import IncrementalPCA

from .config import ARTIFACTS, PCA_DIM, PLINK_PREFIX
from .utils import save_json


def delta_paths(directory: Path):
    paths = [directory / f"chr{i}.npz" for i in range(1, 23)]
    return [path for path in paths if path.exists()]


def fit_pca(paths, n_components: int, batch_size: int):
    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    fitted = 0
    for path in paths:
        values = np.load(path, allow_pickle=True)["delta"]
        for start in range(0, len(values), batch_size):
            batch = values[start:start + batch_size].astype(np.float32)
            if len(batch) < n_components:
                continue
            pca.partial_fit(batch)
            fitted += len(batch)
        print(f"PCA fit: {path.stem}, cumulative={fitted:,}", flush=True)
    if fitted == 0:
        raise RuntimeError("No delta batch was large enough to fit PCA")
    return pca, fitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=ARTIFACTS / "panel.csv")
    parser.add_argument("--samples", type=Path, default=ARTIFACTS / "samples.csv")
    parser.add_argument("--delta-dir", type=Path, default=ARTIFACTS / "delta")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "regional")
    parser.add_argument("--plink-prefix", type=Path, default=PLINK_PREFIX)
    parser.add_argument("--pca-dim", type=int, default=PCA_DIM)
    parser.add_argument("--pca-batch-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "regional_features.npy"
    if feature_path.exists() and not args.overwrite:
        raise FileExistsError(f"{feature_path} exists; pass --overwrite to rebuild")

    paths = delta_paths(args.delta_dir)
    if not paths:
        raise FileNotFoundError(f"No chr*.npz delta files in {args.delta_dir}")
    pca, n_pca_fit = fit_pca(paths, args.pca_dim, args.pca_batch_size)
    np.savez(
        args.output_dir / "delta_pca.npz",
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance=pca.explained_variance_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )

    panel = pd.read_csv(args.panel).set_index("variant_index", drop=False)
    sample_table = pd.read_csv(args.samples)
    genotype = read_plink1_bin(
        str(args.plink_prefix) + ".bed",
        str(args.plink_prefix) + ".bim",
        str(args.plink_prefix) + ".fam",
        verbose=False,
    )
    if len(genotype.sample) != len(sample_table):
        raise ValueError("samples.csv no longer matches the PLINK sample axis")

    active_regions = sorted({
        int(region)
        for path in paths
        for region in np.load(path, allow_pickle=True)["region_id"]
    })
    region_slot = {region: slot for slot, region in enumerate(active_regions)}
    features = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float16,
        shape=(len(sample_table), len(active_regions), args.pca_dim),
    )
    features[:] = 0
    region_rows = []

    for path in paths:
        archive = np.load(path, allow_pickle=True)
        variant_indices = archive["variant_index"].astype(np.int64)
        region_ids = archive["region_id"].astype(np.int64)
        dosage_flip = archive["dosage_flip"].astype(bool)
        delta = archive["delta"].astype(np.float32)
        projected = (delta - pca.mean_.astype(np.float32)) @ pca.components_.astype(np.float32).T
        for region in np.unique(region_ids):
            selected = np.flatnonzero(region_ids == region)
            variants = variant_indices[selected]
            metadata = panel.loc[variants]
            dosage = np.asarray(genotype.isel(variant=variants).values, dtype=np.float32)
            means = metadata["dosage_mean"].to_numpy(np.float32)
            stds = metadata["dosage_std"].to_numpy(np.float32)
            dosage = np.where(np.isfinite(dosage), dosage, means)
            flips = dosage_flip[selected]
            dosage[:, flips] = 2.0 - dosage[:, flips]
            means = np.where(flips, 2.0 - means, means)
            dosage = (dosage - means) / stds
            value = dosage @ projected[selected] / np.sqrt(max(len(selected), 1))
            slot = region_slot[int(region)]
            features[:, slot, :] = value.astype(np.float16)
            region_rows.append({
                "region_slot": slot,
                "panel_region_id": int(region),
                "chrom": int(metadata["chrom"].iloc[0]),
                "start_grch37": int(metadata["pos_grch37"].min()),
                "end_grch37": int(metadata["pos_grch37"].max()),
                "n_embedded_snps": len(selected),
            })
        features.flush()
        print(f"Regional pooling: {path.stem}", flush=True)

    regions = pd.DataFrame(region_rows).sort_values("region_slot").drop_duplicates("region_slot")
    regions.to_csv(args.output_dir / "regions.csv", index=False)
    sample_table.to_csv(args.output_dir / "samples.csv", index=False)
    save_json(args.output_dir / "metadata.json", {
        "shape": list(features.shape),
        "dtype": "float16",
        "pca_dim": args.pca_dim,
        "pca_fit_snps": n_pca_fit,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "n_active_regions": len(active_regions),
        "region_definition": "chromosome-local consecutive QC SNP groups",
        "dosage_normalization_split": "train",
    })
    print(f"Saved regional features {features.shape} -> {feature_path}")


if __name__ == "__main__":
    main()
