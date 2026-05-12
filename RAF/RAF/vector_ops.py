from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def ensure_embedding_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "embedding" not in out.columns and "features" in out.columns:
        out["embedding"] = out["features"]
    if "features" not in out.columns and "embedding" in out.columns:
        out["features"] = out["embedding"]
    if "embedding" not in out.columns and "features" not in out.columns:
        raise KeyError("DataFrame must contain 'embedding' or 'features'.")
    return out


def extract_embeddings(df: pd.DataFrame, *, feature_col: Optional[str] = None) -> np.ndarray:
    if df.empty:
        return np.zeros((0, 0), dtype=np.float32)
    if feature_col is None:
        if "embedding" in df.columns:
            feature_col = "embedding"
        elif "features" in df.columns:
            feature_col = "features"
        else:
            raise KeyError("DataFrame must contain 'embedding' or 'features'.")
    if feature_col not in df.columns:
        raise KeyError(f"Feature column '{feature_col}' not found.")
    return np.stack(df[feature_col].to_numpy()).astype(np.float32, copy=False)


def pairwise_similarity(a: np.ndarray, b: np.ndarray, *, metric: str) -> np.ndarray:
    metric = metric.strip().lower()
    if metric != "euclidean":
        raise ValueError("Only 'euclidean' metric is currently supported.")
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.sum(a * a, axis=1, dtype=np.float32)
    b_norm = np.sum(b * b, axis=1, dtype=np.float32)
    dots = a @ b.T
    return -(a_norm[:, None] + b_norm[None, :] - 2.0 * dots)


def compute_maxsim(
    major: np.ndarray,
    minor: np.ndarray,
    *,
    metric: str = "euclidean",
    batch_size: int = 4096,
    return_argmax: bool = True,
) -> dict[str, np.ndarray | float]:
    major = np.asarray(major, dtype=np.float32)
    minor = np.asarray(minor, dtype=np.float32)
    if major.ndim != 2 or minor.ndim != 2:
        raise ValueError("major and minor embeddings must be 2-D arrays.")

    n = major.shape[0]
    max_sims = np.zeros((n,), dtype=np.float32)
    argmax_idx = np.full((n,), -1, dtype=np.int64)
    if major.size == 0 or minor.size == 0:
        result: dict[str, np.ndarray | float] = {
            "max_sims": max_sims,
            "total": 0.0,
        }
        if return_argmax:
            result["argmax_idx"] = argmax_idx
        return result

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = pairwise_similarity(major[start:end], minor, metric=metric)
        local_idx = np.argmax(sims, axis=1)
        local_best = sims[np.arange(sims.shape[0]), local_idx]
        max_sims[start:end] = local_best.astype(np.float32, copy=False)
        argmax_idx[start:end] = local_idx.astype(np.int64, copy=False)

    result = {"max_sims": max_sims, "total": float(np.sum(max_sims, dtype=np.float64))}
    if return_argmax:
        result["argmax_idx"] = argmax_idx
    return result
