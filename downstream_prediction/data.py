"""Contracts for prospective Y0/Y2/Y4 to Y6 downstream prediction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_SUBJECT_DIR = ROOT / "multimodal_fusion" / "subject_embeddings"
DEFAULT_LABELS = ROOT / "data_preparation" / "labels.csv"
DEFAULT_SPLIT_FILE = ROOT / "data_splits" / "master_subject_split.csv"
TASK_GENETIC_DIM = 512


@dataclass(frozen=True)
class TaskSpec:
    name: str
    target_column: str
    kind: str


TASKS = {
    # BMI z-score is age/sex standardized and matches the existing downstream
    # convention. The task-specific genetic filename remains ``*_bmi.npy``.
    "bmi": TaskSpec("bmi", "bmi_z", "regression"),
    "g_factor": TaskSpec("g_factor", "g_factor", "regression"),
    "internalizing": TaskSpec("internalizing", "internalizing", "regression"),
    "externalizing": TaskSpec("externalizing", "externalizing", "regression"),
    "sui": TaskSpec("sui", "sui", "classification"),
}


@lru_cache(maxsize=None)
def _load_vector(path: Path, expected_dim: int) -> np.ndarray:
    value = np.asarray(np.load(path), dtype=np.float32)
    if value.shape != (expected_dim,):
        raise ValueError(f"{path}: expected ({expected_dim},), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{path}: contains NaN or infinity")
    return value


def load_master_split(path: Path = DEFAULT_SPLIT_FILE) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["subid", "family_id", "site_id", "split"])
    if frame["subid"].duplicated().any():
        raise ValueError(f"Duplicate subjects in master split: {path}")
    if not set(frame["split"]).issubset({"train", "val", "test"}):
        raise ValueError(f"Unexpected split values in {path}")
    if frame["family_id"].isna().any():
        raise ValueError(f"Missing family IDs in {path}")
    family_splits = frame.groupby("family_id")["split"].nunique()
    if (family_splits > 1).any():
        raise ValueError(f"At least one family crosses splits in {path}")
    return frame


def validate_subject_metadata(subject_dir: Path) -> dict:
    completion_marker = subject_dir / ".complete"
    if not completion_marker.exists():
        raise FileNotFoundError(
            f"Incomplete fusion-embedding extraction: {completion_marker} is missing"
        )
    path = subject_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prospective extraction metadata: {path}\n"
            "Rerun: python multimodal_fusion/extract_subject_embeddings.py"
        )
    with path.open() as handle:
        metadata = json.load(handle)
    if tuple(metadata.get("input_visits", ())) != ("Y0", "Y2", "Y4"):
        raise ValueError(
            f"{path}: expected input_visits ['Y0', 'Y2', 'Y4'], "
            f"got {metadata.get('input_visits')}"
        )
    output_dim = metadata.get("output_dim")
    if not isinstance(output_dim, int) or output_dim < 1:
        raise ValueError(f"{path}: output_dim must be a positive integer, got {output_dim!r}")
    if metadata.get("representation") != "shared_foundation_embedding":
        raise ValueError(
            f"{path}: expected representation='shared_foundation_embedding', "
            f"got {metadata.get('representation')!r}"
        )
    if metadata.get("subject_pooling") != "learned_fusion_token":
        raise ValueError(
            f"{path}: expected subject_pooling='learned_fusion_token', "
            f"got {metadata.get('subject_pooling')!r}"
        )
    return metadata


def load_y6_labels(path: Path, task: str) -> pd.DataFrame:
    spec = TASKS[task]
    required = {"subid", "visit", spec.target_column}
    labels = pd.read_csv(path)
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    y6 = labels.loc[
        labels["visit"].eq("Y6"), ["subid", spec.target_column]
    ].dropna(subset=[spec.target_column]).copy()
    if y6.empty:
        raise ValueError(
            f"{path} has no non-missing Y6 labels for {spec.target_column}. "
            "Rebuild it with: python -m data_preparation.build_labels"
        )
    if y6["subid"].duplicated().any():
        duplicated = y6.loc[y6["subid"].duplicated(), "subid"].iloc[0]
        raise ValueError(f"Duplicate Y6 label for {duplicated} in {path}")
    y6 = y6.rename(columns={spec.target_column: "target"})
    y6["target"] = pd.to_numeric(y6["target"], errors="coerce")
    y6 = y6.dropna(subset=["target"])
    if spec.kind == "classification" and not set(y6["target"].unique()).issubset({0.0, 1.0}):
        raise ValueError(f"{spec.target_column} must contain only 0/1 labels")
    return y6


class DownstreamDataset:
    """Aligned frozen subject embeddings, Y6 labels, and master split metadata."""

    def __init__(
        self,
        task: str,
        subject_dir: Path | str = DEFAULT_SUBJECT_DIR,
        labels_path: Path | str = DEFAULT_LABELS,
        split_file: Path | str = DEFAULT_SPLIT_FILE,
        task_genetic_dir: Path | str | None = None,
    ):
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}; choose from {tuple(TASKS)}")
        self.task = task
        self.spec = TASKS[task]
        self.subject_dir = Path(subject_dir)
        if not self.subject_dir.exists():
            raise FileNotFoundError(
                f"Fusion subject embeddings not found: {self.subject_dir}\n"
                "Run: python multimodal_fusion/extract_subject_embeddings.py"
            )
        self.subject_metadata = validate_subject_metadata(self.subject_dir)
        self.stage2_dim = int(self.subject_metadata["output_dim"])
        labels = load_y6_labels(Path(labels_path), task)
        split = load_master_split(Path(split_file))
        frame = split.merge(labels, on="subid", how="inner", validate="one_to_one")
        frame["embedding_file"] = frame["subid"].map(
            lambda subid: str(self.subject_dir / f"{subid}.npy")
        )
        frame = frame[frame["embedding_file"].map(lambda value: Path(value).exists())].copy()

        self.task_genetic_dir = Path(task_genetic_dir) if task_genetic_dir else None
        if self.task_genetic_dir is not None:
            frame["task_genetic_file"] = frame["subid"].map(
                lambda subid: str(self.task_genetic_dir / f"{subid}_{task}.npy")
            )
            frame["has_task_genetic"] = frame["task_genetic_file"].map(
                lambda value: Path(value).exists()
            )
        else:
            frame["task_genetic_file"] = ""
            frame["has_task_genetic"] = False

        self.frame = frame.sort_values(["split", "subid"]).reset_index(drop=True)
        for name in ("train", "val", "test"):
            if not self.frame["split"].eq(name).any():
                raise RuntimeError(f"Empty {name} cohort for task {task}")

    def arrays(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        rows = self.frame[self.frame["split"].eq(split)]
        stage2 = np.stack([
            _load_vector(Path(path), self.stage2_dim) for path in rows["embedding_file"]
        ])
        target = rows["target"].to_numpy(dtype=np.float32)
        if self.task_genetic_dir is None:
            task_genetic = np.empty((len(rows), 0), dtype=np.float32)
        else:
            task_genetic = np.zeros((len(rows), TASK_GENETIC_DIM + 1), dtype=np.float32)
            for index, (_, row) in enumerate(rows.iterrows()):
                if row["has_task_genetic"]:
                    task_genetic[index, :TASK_GENETIC_DIM] = _load_vector(
                        Path(row["task_genetic_file"]), TASK_GENETIC_DIM
                    )
                    task_genetic[index, -1] = 1.0
        return stage2, task_genetic, target, rows["subid"].tolist()

    def summary(self) -> dict[str, int]:
        result = {
            split: int(self.frame["split"].eq(split).sum())
            for split in ("train", "val", "test")
        }
        result["task_genetic"] = int(self.frame["has_task_genetic"].sum())
        return result
