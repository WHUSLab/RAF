from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


@dataclass
class ClusterSet:
    centers: np.ndarray
    radii: np.ndarray

    def assign_cluster(self, embedding: np.ndarray) -> Optional[int]:
        embedding = np.asarray(embedding, dtype=np.float32)
        dists = np.linalg.norm(self.centers - embedding, axis=1)
        inside = dists <= self.radii
        if not np.any(inside):
            return None
        masked = np.where(inside, dists, np.inf)
        return int(np.argmin(masked))

    def assign_clusters(self, embeddings: np.ndarray, *, batch_size: int = 4096) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32)
        labels = np.full((x.shape[0],), -1, dtype=np.int32)
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            chunk = x[start:end]
            dists = np.linalg.norm(chunk[:, None, :] - self.centers[None, :, :], axis=2)
            inside = dists <= self.radii[None, :]
            valid = np.any(inside, axis=1)
            if not np.any(valid):
                continue
            labels[start:end][valid] = np.argmin(np.where(inside[valid], dists[valid], np.inf), axis=1)
        return labels


def _resolve_subsample_size(
    n_samples: int,
    subsample_ratio: Optional[float],
    subsample_size: Optional[int],
    n_clusters: int,
) -> Optional[int]:
    if subsample_size is not None:
        if subsample_size <= 0 or subsample_size >= n_samples or subsample_size < n_clusters:
            return None
        return int(subsample_size)
    if subsample_ratio is None or subsample_ratio <= 0 or subsample_ratio >= 1:
        return None
    resolved = int(n_samples * subsample_ratio)
    if resolved < n_clusters:
        return None
    return resolved


def fit_clusters(
    embeddings: np.ndarray,
    n_clusters: int,
    *,
    random_state: int = 42,
    subsample_ratio: Optional[float] = None,
    subsample_size: Optional[int] = None,
    assign_batch_size: int = 4096,
    use_minibatch: bool = True,
) -> tuple[ClusterSet, np.ndarray]:
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("embeddings must be a 2-D array.")
    if x.shape[0] < n_clusters:
        raise ValueError("n_clusters must not exceed the number of samples.")

    resolved_subsample = _resolve_subsample_size(
        x.shape[0],
        subsample_ratio,
        subsample_size,
        n_clusters,
    )
    if use_minibatch:
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=min(max(256, n_clusters * 8), max(256, x.shape[0])),
            n_init="auto",
        )
    else:
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto",
        )

    if resolved_subsample is not None:
        rng = np.random.default_rng(random_state)
        train_idx = rng.choice(x.shape[0], size=resolved_subsample, replace=False)
        model.fit(x[train_idx])
        centers = np.asarray(model.cluster_centers_, dtype=np.float32)
        labels = np.empty((x.shape[0],), dtype=np.int32)
        min_dist = np.empty((x.shape[0],), dtype=np.float32)
        for start in range(0, x.shape[0], assign_batch_size):
            end = min(start + assign_batch_size, x.shape[0])
            chunk = x[start:end]
            dists = np.linalg.norm(chunk[:, None, :] - centers[None, :, :], axis=2)
            labels[start:end] = np.argmin(dists, axis=1).astype(np.int32)
            min_dist[start:end] = np.min(dists, axis=1).astype(np.float32)
    else:
        labels = model.fit_predict(x).astype(np.int32)
        centers = np.asarray(model.cluster_centers_, dtype=np.float32)
        min_dist = np.linalg.norm(x - centers[labels], axis=1).astype(np.float32)

    radii = np.zeros((n_clusters,), dtype=np.float32)
    for cid in range(n_clusters):
        mask = labels == cid
        if not np.any(mask):
            continue
        radii[cid] = float(np.percentile(min_dist[mask], 95))

    return ClusterSet(centers=centers, radii=radii), labels
