#!/usr/bin/env python3 -u
"""Create a phenotype-free, training-family-QC SNP panel from the PLINK array."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin

from .config import ARTIFACTS, AUTOSOMES, PLINK_PREFIX, SPLIT_FILE
from .utils import canonical_subid, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plink-prefix", type=Path, default=PLINK_PREFIX)
    parser.add_argument("--split-file", type=Path, default=SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--maf-min", type=float, default=0.01)
    parser.add_argument("--missingness-max", type=float, default=0.05)
    parser.add_argument("--snps-per-region", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    genotype = read_plink1_bin(
        str(args.plink_prefix) + ".bed",
        str(args.plink_prefix) + ".bim",
        str(args.plink_prefix) + ".fam",
        verbose=False,
    )
    split_df = pd.read_csv(
        args.split_file, usecols=["subid", "split"], dtype={"subid": str, "split": str}
    )
    split_map = dict(zip(split_df["subid"], split_df["split"]))
    subids = np.array([canonical_subid(x) for x in genotype.sample.values])
    splits = np.array([split_map.get(x, "unassigned") for x in subids])
    train_indices = np.flatnonzero(splits == "train")
    if not len(train_indices):
        raise RuntimeError("No PLINK subjects belong to the master training split")

    samples = pd.DataFrame({
        "sample_index": np.arange(len(subids)),
        "iid": genotype.sample.values.astype(str),
        "subid": subids,
        "split": splits,
    })
    samples.to_csv(args.output_dir / "samples.csv", index=False)

    chromosomes = genotype.chrom.values.astype(str)
    candidate = np.flatnonzero(np.isin(chromosomes, AUTOSOMES))
    kept = []
    print(f"QC on {len(train_indices):,} training subjects and {len(candidate):,} autosomal SNPs")
    for start in range(0, len(candidate), args.chunk_size):
        indices = candidate[start:start + args.chunk_size]
        block = np.asarray(
            genotype.isel(sample=train_indices, variant=indices).values,
            dtype=np.float32,
        )
        finite = np.isfinite(block)
        counts = finite.sum(axis=0)
        means = np.divide(
            np.where(finite, block, 0).sum(axis=0), counts,
            out=np.full(len(indices), np.nan), where=counts > 0,
        )
        missingness = 1.0 - counts / len(train_indices)
        frequency = means / 2.0
        maf = np.minimum(frequency, 1.0 - frequency)
        centered = np.where(finite, block - means, 0.0)
        std = np.sqrt(np.divide(
            np.square(centered).sum(axis=0), counts,
            out=np.zeros(len(indices)), where=counts > 0,
        ))
        valid = (
            (missingness <= args.missingness_max)
            & (maf >= args.maf_min)
            & np.isfinite(means)
            & (std > 1e-6)
        )
        for local in np.flatnonzero(valid):
            variant_index = int(indices[local])
            kept.append({
                "variant_index": variant_index,
                "chrom": chromosomes[variant_index],
                "pos_grch37": int(genotype.pos.values[variant_index]),
                "rsid": str(genotype.snp.values[variant_index]),
                "plink_a0": str(genotype.a0.values[variant_index]).upper(),
                "plink_a1": str(genotype.a1.values[variant_index]).upper(),
                "dosage_mean": float(means[local]),
                "dosage_std": float(std[local]),
                "missingness": float(missingness[local]),
                "maf": float(maf[local]),
            })
        print(f"  {min(start + args.chunk_size, len(candidate)):,}/{len(candidate):,}; kept={len(kept):,}", flush=True)

    panel = pd.DataFrame(kept)
    panel["chrom_int"] = panel["chrom"].astype(int)
    panel = panel.sort_values(["chrom_int", "pos_grch37", "variant_index"]).reset_index(drop=True)
    region_ids = np.empty(len(panel), dtype=np.int64)
    next_region = 0
    for _, indices in panel.groupby("chrom_int", sort=True).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        local_regions = np.arange(len(positions)) // args.snps_per_region + next_region
        region_ids[positions] = local_regions
        next_region = int(local_regions[-1]) + 1
    panel["region_id"] = region_ids
    panel = panel.drop(columns="chrom_int")
    panel.to_csv(args.output_dir / "panel.csv", index=False)
    save_json(args.output_dir / "panel_metadata.json", {
        "n_plink_subjects": len(subids),
        "n_training_subjects": len(train_indices),
        "n_autosomal_candidates": len(candidate),
        "n_qc_snps": len(panel),
        "n_regions_before_sequence_filter": int(panel["region_id"].nunique()),
        "snps_per_region": args.snps_per_region,
        "maf_min": args.maf_min,
        "missingness_max": args.missingness_max,
        "qc_split": "train",
    })
    print(f"Saved {len(panel):,} SNPs in {panel['region_id'].nunique():,} regions")


if __name__ == "__main__":
    main()
