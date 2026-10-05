"""Shared utilities for leakage-controlled modality pretraining."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data_splits" / "master_subject_split.csv"
# Modality pretraining is general-purpose representation learning, so it uses every
# available wave from development families. The prospective Y0/Y2/Y4 -> Y6
# restriction applies later when downstream prediction constructs inputs.
ALLOWED_VISITS = ("Y0", "Y2", "Y4", "Y6")
SEED = 42


def canonical_subid(value: str) -> str:
    """Convert ABCD ``sub-*`` IDs to the project's ``NDAR_INV*`` convention."""
    value = str(value)
    if value.startswith("NDAR_INV"):
        return value
    if value.startswith("sub-"):
        return "NDAR_INV" + value[4:]
    return value


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_map(path: Path = SPLIT_FILE) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Master split not found: {path}\n"
            "Run data_preparation/create_master_split.py first."
        )
    df = pd.read_csv(path, usecols=["subid", "split"])
    if df["subid"].duplicated().any():
        raise ValueError(f"Duplicate subjects in {path}")
    valid = {"train", "val", "test"}
    if not set(df["split"]).issubset(valid):
        raise ValueError(f"Unexpected split value in {path}")
    return dict(zip(df["subid"], df["split"]))


def split_of(subid: str, split_map: dict[str, str]) -> str | None:
    return split_map.get(canonical_subid(subid))
