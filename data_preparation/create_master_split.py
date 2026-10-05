#!/usr/bin/env python3
"""Create the family-grouped/site-balanced split used throughout the model.

The test families must never participate in encoder pretraining, normalizer
fitting, checkpoint selection, SNP selection, fusion pretraining, or model
selection. Unknown-family subjects are retained as singleton groups.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from modality_pretraining.common import ROOT, SEED, canonical_subid

SPLIT_ROOT = ROOT / "data_splits"


DEMO = Path(os.environ.get("LUMEN_DEMOGRAPHICS_CSV", "data/abcd_y_lt.csv"))
RSFC = Path(os.environ.get("LUMEN_RSFC_DIR", "data/rsfc"))
BOLD = Path(os.environ.get("LUMEN_BOLD_DIR", "data/bold"))
FITBIT = Path(os.environ.get("LUMEN_FITBIT_CACHE", "data/fitbit_hourly_cache"))


def inventory_subjects() -> set[str]:
    subjects: set[str] = set()
    labels = ROOT / "data_preparation" / "labels.csv"
    if labels.exists():
        subjects.update(pd.read_csv(labels, usecols=["subid"])["subid"].dropna())

    for visit in ("y0", "y2", "y4", "y6"):
        path = RSFC / f"{visit}_pids.npy"
        if path.exists():
            subjects.update(canonical_subid(x) for x in np.load(path, allow_pickle=True))

    subjects.update(canonical_subid(p.stem.split("_")[0]) for p in BOLD.glob("sub-*_Y[0246].npy"))
    subjects.update(p.stem for p in FITBIT.glob("NDAR_INV*.npy") if not p.stem.endswith("_ts"))

    for modality in ("smri", "genetic"):
        directory = ROOT / "multimodal_fusion" / "embeddings" / modality
        for path in directory.glob("NDAR_INV*.npy"):
            parts = path.stem.split("_")
            if len(parts) >= 2:
                subjects.add("_".join(parts[:2]))
    return {canonical_subid(s) for s in subjects}


def load_demographics() -> pd.DataFrame:
    demo = pd.read_csv(
        DEMO,
        usecols=["src_subject_id", "eventname", "site_id_l", "rel_family_id"],
        low_memory=False,
    )
    baseline = demo[demo["eventname"] == "baseline_year_1_arm_1"].copy()
    baseline = baseline.drop_duplicates("src_subject_id")
    baseline["subid"] = baseline["src_subject_id"].map(canonical_subid)
    return baseline[["subid", "site_id_l", "rel_family_id"]]


def first_fold(strata: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    # StratifiedGroupKFold warns (and cannot meaningfully balance) a stratum
    # represented fewer than n_splits times. Fold only those isolated records
    # into the largest stratum for assignment; the real site_id remains in the
    # saved file for later reporting.
    strata = np.asarray(strata, dtype=object).copy()
    values, counts = np.unique(strata, return_counts=True)
    largest = values[np.argmax(counts)]
    rare = set(values[counts < n_splits])
    if rare:
        strata[np.isin(strata, list(rare))] = largest
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(cv.split(np.zeros(len(strata)), strata, groups))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=SPLIT_ROOT / "master_subject_split.csv")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    subjects = pd.DataFrame({"subid": sorted(inventory_subjects())})
    df = subjects.merge(load_demographics(), on="subid", how="left")
    df["site_id"] = df["site_id_l"].fillna("UNKNOWN").astype(str)
    missing_family = df["rel_family_id"].isna()
    df["family_id"] = df["rel_family_id"].astype(object)
    df.loc[missing_family, "family_id"] = "SINGLETON:" + df.loc[missing_family, "subid"]
    df["family_id"] = df["family_id"].astype(str)

    remaining, test = first_fold(
        df["site_id"].to_numpy(), df["family_id"].to_numpy(), 10, args.seed
    )
    rem = df.iloc[remaining]
    train_local, val_local = first_fold(
        rem["site_id"].to_numpy(), rem["family_id"].to_numpy(), 9, args.seed + 1
    )

    split = np.full(len(df), "test", dtype=object)
    split[remaining[train_local]] = "train"
    split[remaining[val_local]] = "val"
    df["split"] = split

    # A family must occur in exactly one split.
    family_nsplit = df.groupby("family_id")["split"].nunique()
    if family_nsplit.max() != 1:
        raise RuntimeError("Family leakage detected while creating master split")

    out = df[["subid", "family_id", "site_id", "split"]].sort_values("subid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Saved {len(out):,} subjects -> {args.output}")
    print(out["split"].value_counts().to_string())
    print("\nFamilies:")
    print(out.groupby("split")["family_id"].nunique().to_string())
    print("\nSite x split:")
    print(pd.crosstab(out["site_id"], out["split"]).to_string())


if __name__ == "__main__":
    main()
