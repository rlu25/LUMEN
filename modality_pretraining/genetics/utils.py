"""Shared utilities for genetic preprocessing and training."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np


def canonical_subid(value: str) -> str:
    value = str(value)
    if value.startswith("NDAR_INV"):
        return value
    if value.startswith("sub-"):
        return "NDAR_INV" + value[4:]
    return value


def seed_everything(seed: int = 42):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        json.dump(value, stream, indent=2)
