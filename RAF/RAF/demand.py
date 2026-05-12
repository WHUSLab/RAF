from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GroupLabeler:
    values: list[Any]
    _value_to_index: dict[Any, int] = None

    def __post_init__(self):
        if self._value_to_index is None:
            self._value_to_index = {value: idx for idx, value in enumerate(self.values)}

    def num_groups(self) -> int:
        return len(self.values)

    def get_group(self, sample: dict[str, Any], sensitive_col: str) -> int | None:
        value = sample.get(sensitive_col, sample.get("is_english_name"))
        return self._value_to_index.get(value)


def build_group_labeler(df: pd.DataFrame, sensitive_col: str) -> GroupLabeler:
    values = [value for value in pd.unique(df[sensitive_col]) if not pd.isna(value)]
    if not values:
        raise ValueError(f"No valid values found in '{sensitive_col}'.")
    return GroupLabeler(values=list(values))


def recompute_q_from_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2:
        raise ValueError("counts must be a 2-D matrix.")
    return np.maximum(0.0, np.max(counts, axis=1, keepdims=True) - counts)


def recompute_group_level_q_from_values(sensitive_values: np.ndarray, group_values: list[Any]) -> np.ndarray:
    counts = np.zeros((1, len(group_values)), dtype=float)
    group_index = {value: idx for idx, value in enumerate(group_values)}

    # Vectorized counting using np.unique and indexing
    unique_vals, unique_counts = np.unique(sensitive_values, return_counts=True)
    for val, cnt in zip(unique_vals, unique_counts):
        idx = group_index.get(val)
        if idx is not None:
            counts[0, idx] = float(cnt)

    return np.maximum(0.0, np.max(counts, axis=1, keepdims=True) - counts)


def build_counts_from_labels(labels: np.ndarray, sensitive_values: np.ndarray, group_values: list[Any]) -> np.ndarray:
    n_clusters = int(np.max(labels)) + 1 if labels.size > 0 else 0
    n_groups = len(group_values)
    counts = np.zeros((n_clusters, n_groups), dtype=int)
    group_index = {value: idx for idx, value in enumerate(group_values)}

    # Map sensitive_values to group indices, -1 for unknown
    group_ids = np.array([group_index.get(v, -1) for v in sensitive_values.tolist()], dtype=np.int64)

    # Filter valid entries (valid cluster and valid group)
    valid = (labels >= 0) & (group_ids >= 0)
    valid_labels = labels[valid]
    valid_groups = group_ids[valid]

    if valid_labels.size > 0:
        np.add.at(counts, (valid_labels, valid_groups), 1)

    return counts
