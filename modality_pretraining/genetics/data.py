"""Memory-mapped regional genetic dataset and train-only normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class RegionalDataset(Dataset):
    def __init__(self, feature_path: Path, samples_path: Path, split: str | None,
                 mean=None, std=None):
        self.feature_path = Path(feature_path)
        self.samples = pd.read_csv(samples_path)
        self.indices = np.flatnonzero(self.samples["split"].to_numpy() == split) \
            if split is not None else np.arange(len(self.samples))
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self._features = None

    @property
    def features(self):
        if self._features is None:
            self._features = np.load(self.feature_path, mmap_mode="r")
        return self._features

    @property
    def shape(self):
        return self.features.shape

    @property
    def subject_ids(self):
        return self.samples.iloc[self.indices]["subid"].astype(str).tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        value = np.asarray(self.features[self.indices[index]], dtype=np.float32)
        if self.mean is not None:
            value = (value - self.mean) / self.std
        return value

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_features"] = None
        return state


def fit_normalizer(dataset: RegionalDataset, subject_chunk=16):
    sums = np.zeros(dataset.shape[-1], dtype=np.float64)
    squares = np.zeros(dataset.shape[-1], dtype=np.float64)
    count = 0
    for start in range(0, len(dataset.indices), subject_chunk):
        indices = dataset.indices[start:start + subject_chunk]
        values = np.asarray(dataset.features[indices], dtype=np.float32)
        sums += values.sum(axis=(0, 1), dtype=np.float64)
        squares += np.square(values).sum(axis=(0, 1), dtype=np.float64)
        count += values.shape[0] * values.shape[1]
    mean = sums / max(count, 1)
    variance = squares / max(count, 1) - np.square(mean)
    return mean.astype(np.float32), np.sqrt(np.maximum(variance, 1e-6)).astype(np.float32)
