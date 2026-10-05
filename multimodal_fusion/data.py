"""Embedding contracts and datasets for leakage-controlled multimodal fusion."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_DIR = Path(__file__).resolve().parent / "embeddings"
SPLIT_FILE = ROOT / "data_splits" / "master_subject_split.csv"

MODALITY_DIMS = {
    "rsfc": 256,
    "fitbit": 128,
    "smri": 512,
    "bold": 256,
    # A general-genetic file is (256,) or (K, 256), where K is the number
    # of genome-wide latent tokens. Phenotype-specific 512-d files are not
    # valid fusion inputs; they are reserved for phenotype-specific analysis.
    "genetic": 256,
}
MODALITY_IDX = {name: i for i, name in enumerate(MODALITY_DIMS)}
VISITS = ("Y0", "Y2", "Y4", "Y6", "NA")
VISIT_IDX = {name: i for i, name in enumerate(VISITS)}
TASK_TAGS = {"bmi", "g_factor", "internalizing", "externalizing", "sui"}
D_MAX = max(MODALITY_DIMS.values())
MAX_GENETIC_TOKENS = 128


def _parse_visit_file(path: Path) -> tuple[str, str] | None:
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    subid = "_".join(parts[:2])
    tag = "_".join(parts[2:])
    if not subid.startswith("NDAR_INV") or tag not in VISIT_IDX or tag == "NA":
        return None
    return subid, tag


def _parse_genetic_file(path: Path) -> str | None:
    """Accept only ``NDAR_INVxxx.npy`` as a generic genetic representation."""
    if not path.stem.startswith("NDAR_INV"):
        return None
    # A canonical subject ID contains one underscore. Any additional suffix
    # is task-conditioned (or otherwise violates this fusion contract).
    if len(path.stem.split("_")) != 2:
        return None
    return path.stem


def _load_embedding(path: Path, dim: int, allow_tokens: bool = False) -> np.ndarray:
    value = np.asarray(np.load(path), dtype=np.float32)
    if allow_tokens and value.ndim == 1:
        value = value[None, :]
    expected = ("K", dim) if allow_tokens else (dim,)
    valid_shape = value.ndim == 2 and value.shape[0] > 0 and value.shape[1] == dim \
        if allow_tokens else value.shape == (dim,)
    if not valid_shape:
        raise ValueError(f"{path}: expected shape {expected}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{path}: embedding contains NaN or infinity")
    return value


class MultiModalDataset(Dataset):
    """One variable-length sequence of modality/visit tokens per subject."""

    def __init__(
        self,
        embedding_dir: Path | str = DEFAULT_EMBEDDING_DIR,
        split_map: dict[str, str] | None = None,
        split: str | None = None,
        modalities: tuple[str, ...] = tuple(MODALITY_DIMS),
        visits: tuple[str, ...] = VISITS[:-1],
        min_tokens: int = 2,
        normalizer: dict[str, dict[str, np.ndarray]] | None = None,
        require_modalities: bool = True,
    ):
        self.embedding_dir = Path(embedding_dir)
        self.modalities = tuple(modalities)
        self.visits = tuple(visits)
        self.normalizer = normalizer
        unknown = set(self.modalities) - set(MODALITY_DIMS)
        if unknown:
            raise ValueError(f"Unknown modalities: {sorted(unknown)}")

        cache: dict[str, list[tuple[int, int, int, np.ndarray]]] = {}
        counts = Counter()
        rejected_task_genetics = 0
        for modality in self.modalities:
            directory = self.embedding_dir / modality
            if not directory.exists():
                if require_modalities:
                    raise FileNotFoundError(f"Missing requested modality directory: {directory}")
                continue
            dim = MODALITY_DIMS[modality]
            if modality == "genetic":
                for path in sorted(directory.glob("NDAR_INV*.npy")):
                    subid = _parse_genetic_file(path)
                    if subid is None:
                        suffix = "_".join(path.stem.split("_")[2:])
                        rejected_task_genetics += int(suffix in TASK_TAGS)
                        continue
                    if split_map is not None and split_map.get(subid) != split:
                        continue
                    values = _load_embedding(path, dim, allow_tokens=True)
                    if len(values) > MAX_GENETIC_TOKENS:
                        raise ValueError(
                            f"{path}: {len(values)} genetic tokens exceeds "
                            f"MAX_GENETIC_TOKENS={MAX_GENETIC_TOKENS}"
                        )
                    for slot, value in enumerate(values):
                        cache.setdefault(subid, []).append(
                            (MODALITY_IDX[modality], VISIT_IDX["NA"], slot, value)
                        )
                        counts[modality] += 1
                continue

            for path in sorted(directory.glob("NDAR_INV*.npy")):
                parsed = _parse_visit_file(path)
                if parsed is None:
                    continue
                subid, visit = parsed
                if visit not in self.visits:
                    continue
                if split_map is not None and split_map.get(subid) != split:
                    continue
                value = _load_embedding(path, dim)
                cache.setdefault(subid, []).append(
                    (MODALITY_IDX[modality], VISIT_IDX[visit], 0, value)
                )
                counts[modality] += 1

        if "genetic" in self.modalities and rejected_task_genetics and counts["genetic"] == 0:
            raise ValueError(
                f"{self.embedding_dir / 'genetic'} contains phenotype-specific genetic files "
                "but no generic NDAR_INV<id>.npy files. Task-specific genetics must enter "
                "through a phenotype-specific adapter, not generic fusion."
            )

        self.subjects = []
        self.samples = []
        for subid in sorted(cache):
            entries = sorted(cache[subid], key=lambda item: (item[0], item[1], item[2]))
            if len(entries) < min_tokens:
                continue
            raw = np.zeros((len(entries), D_MAX), dtype=np.float32)
            mod_ids = np.empty(len(entries), dtype=np.int64)
            visit_ids = np.empty(len(entries), dtype=np.int64)
            slot_ids = np.empty(len(entries), dtype=np.int64)
            for i, (mod_idx, visit_idx, slot_idx, value) in enumerate(entries):
                modality = tuple(MODALITY_DIMS)[mod_idx]
                raw[i, : MODALITY_DIMS[modality]] = value
                mod_ids[i], visit_ids[i] = mod_idx, visit_idx
                slot_ids[i] = slot_idx
            self.subjects.append(subid)
            self.samples.append((raw, mod_ids, visit_ids, slot_ids))
        self.token_counts = dict(counts)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        raw, mod_ids, visit_ids, slot_ids = self.samples[index]
        raw = raw.copy()
        if self.normalizer is not None:
            for modality, mod_idx in MODALITY_IDX.items():
                positions = mod_ids == mod_idx
                if not positions.any() or modality not in self.normalizer:
                    continue
                dim = MODALITY_DIMS[modality]
                stats = self.normalizer[modality]
                raw[positions, :dim] = (
                    raw[positions, :dim] - stats["mean"]
                ) / stats["std"]
        return raw, mod_ids.copy(), visit_ids.copy(), slot_ids.copy()


def fit_normalizer(dataset: MultiModalDataset) -> dict[str, dict[str, np.ndarray]]:
    """Fit per-modality, per-dimension statistics on master-training tokens."""
    sums = {m: np.zeros(d, dtype=np.float64) for m, d in MODALITY_DIMS.items()}
    squares = {m: np.zeros(d, dtype=np.float64) for m, d in MODALITY_DIMS.items()}
    counts = Counter()
    for raw, mod_ids, _, _ in dataset.samples:
        for modality, mod_idx in MODALITY_IDX.items():
            selected = raw[mod_ids == mod_idx, : MODALITY_DIMS[modality]].astype(np.float64)
            if not len(selected):
                continue
            sums[modality] += selected.sum(axis=0)
            squares[modality] += np.square(selected).sum(axis=0)
            counts[modality] += len(selected)
    result = {}
    for modality in dataset.modalities:
        if counts[modality] == 0:
            continue
        mean = sums[modality] / counts[modality]
        var = squares[modality] / counts[modality] - np.square(mean)
        result[modality] = {
            "mean": mean.astype(np.float32),
            "std": np.sqrt(np.maximum(var, 1e-6)).astype(np.float32),
        }
    return result


def collate_tokens(batch):
    batch_size = len(batch)
    max_tokens = max(len(item[0]) for item in batch)
    raw = np.zeros((batch_size, max_tokens, D_MAX), dtype=np.float32)
    mod_ids = np.zeros((batch_size, max_tokens), dtype=np.int64)
    visit_ids = np.zeros((batch_size, max_tokens), dtype=np.int64)
    slot_ids = np.zeros((batch_size, max_tokens), dtype=np.int64)
    padding = np.ones((batch_size, max_tokens), dtype=bool)
    for i, (values, modalities, visits, slots) in enumerate(batch):
        n = len(values)
        raw[i, :n] = values
        mod_ids[i, :n] = modalities
        visit_ids[i, :n] = visits
        slot_ids[i, :n] = slots
        padding[i, :n] = False
    return tuple(torch.from_numpy(x) for x in (raw, mod_ids, visit_ids, slot_ids, padding))
