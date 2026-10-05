#!/usr/bin/env python3
"""Create a phenotype-independent, family-safe, site-disjoint split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data_splits" / "master_subject_split.csv"
DEFAULT_OUTPUT = ROOT / "data_splits" / "master_site_holdout_split.csv"
DEFAULT_VAL_SITES = ("site04", "site10", "site16", "site22")
DEFAULT_TEST_SITES = ("site03", "site09", "site15", "site21")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-sites", nargs="+", default=list(DEFAULT_VAL_SITES))
    parser.add_argument("--test-sites", nargs="+", default=list(DEFAULT_TEST_SITES))
    args = parser.parse_args()

    val_sites = set(args.val_sites)
    test_sites = set(args.test_sites)
    if val_sites & test_sites:
        parser.error("--val-sites and --test-sites must be disjoint")

    frame = pd.read_csv(
        args.source,
        usecols=["subid", "family_id", "site_id"],
        dtype={"subid": str, "family_id": str, "site_id": str},
    )
    if frame["subid"].duplicated().any():
        raise ValueError(f"Duplicate subjects in {args.source}")
    if frame[["family_id", "site_id"]].isna().any().any():
        raise ValueError(f"Missing family/site values in {args.source}")

    observed_sites = set(frame["site_id"]) - {"UNKNOWN"}
    missing = (val_sites | test_sites) - observed_sites
    if missing:
        raise ValueError(f"Requested sites are absent: {sorted(missing)}")

    unknown_subjects = frame.loc[frame["site_id"].eq("UNKNOWN"), "subid"].tolist()
    frame = frame.loc[frame["site_id"].ne("UNKNOWN")].copy()
    frame["split"] = "train"
    frame.loc[frame["site_id"].isin(val_sites), "split"] = "val"
    frame.loc[frame["site_id"].isin(test_sites), "split"] = "test"

    family_nsplit = frame.groupby("family_id")["split"].nunique()
    crossing_families = sorted(family_nsplit[family_nsplit > 1].index)
    crossing_subjects = frame.loc[
        frame["family_id"].isin(crossing_families), "subid"
    ].tolist()
    frame = frame.loc[~frame["family_id"].isin(crossing_families)].copy()
    frame = frame[["subid", "family_id", "site_id", "split"]].sort_values("subid")

    if set(frame["split"]) != {"train", "val", "test"}:
        raise RuntimeError("Site assignment produced an empty split")
    if (frame.groupby("site_id")["split"].nunique() != 1).any():
        raise RuntimeError("Site leakage detected")
    if (frame.groupby("family_id")["split"].nunique() != 1).any():
        raise RuntimeError("Family leakage detected")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    metadata = {
        "definition": "phenotype-independent family-safe site holdout",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "output": str(args.output.resolve()),
        "val_sites": sorted(val_sites),
        "test_sites": sorted(test_sites),
        "train_sites": sorted(observed_sites - val_sites - test_sites),
        "counts": {k: int(v) for k, v in frame["split"].value_counts().items()},
        "excluded_unknown_site_subjects": unknown_subjects,
        "excluded_cross_partition_families": crossing_families,
        "excluded_cross_partition_subjects": crossing_subjects,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved {len(frame):,} subjects -> {args.output}")
    print(frame["split"].value_counts().to_string())
    print("\nSite x split:")
    print(pd.crosstab(frame["site_id"], frame["split"]).to_string())
    print(f"\nMetadata -> {metadata_path}")


if __name__ == "__main__":
    main()
