from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .clustering import fit_clusters
from .fair_external_eval import run_external_fair_relax_merge
from .vector_ops import compute_maxsim, ensure_embedding_columns, extract_embeddings


@dataclass
class EvalMetrics:
    rows: int
    sse: float
    avg_sse: float
    avg_radius: float
    max_radius: float
    balance_gap: float
    fairness_gap: float
    maxsim_total: float
    clustering_method: str = "kmeans"
    clustering_objective: Optional[float] = None


def _compute_sse(embeddings: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    valid = labels >= 0
    if not np.any(valid):
        return 0.0
    diff = embeddings[valid] - centers[labels[valid]]
    return float(np.sum(diff * diff, dtype=np.float64))


def _compute_radius_stats(embeddings: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> tuple[float, float]:
    valid = labels >= 0
    labels = labels[valid]
    embeddings = embeddings[valid]
    if labels.size == 0:
        return 0.0, 0.0
    radii: list[float] = []
    for cid in np.unique(labels):
        mask = labels == cid
        if not np.any(mask):
            continue
        diff = embeddings[mask] - centers[int(cid)]
        radius = float(np.sqrt(np.max(np.sum(diff * diff, axis=1, dtype=np.float64))))
        radii.append(radius)
    if not radii:
        return 0.0, 0.0
    return float(np.mean(radii)), float(np.max(radii))


def _fairness_gap(labels: np.ndarray, sensitive_values: np.ndarray) -> float:
    valid = labels >= 0
    labels = labels[valid]
    sensitive_values = sensitive_values[valid]
    if labels.size == 0:
        return 0.0
    values, freqs = np.unique(sensitive_values, return_counts=True)
    overall = {
        value: float(freq) / float(np.sum(freqs))
        for value, freq in zip(values.tolist(), freqs.tolist())
    }
    gaps = []
    for cid in np.unique(labels):
        mask = labels == cid
        size = int(np.sum(mask))
        if size <= 0:
            continue
        cluster_values = sensitive_values[mask]
        gap = 0.0
        for value in values:
            cluster_ratio = float(np.sum(cluster_values == value) / size)
            gap += abs(cluster_ratio - overall[value])
        gaps.append(0.5 * gap)
    return float(np.mean(gaps)) if gaps else 0.0


def _balance_gap(labels: np.ndarray, sensitive_values: np.ndarray) -> float:
    valid = labels >= 0
    labels = labels[valid]
    sensitive_values = sensitive_values[valid]
    if labels.size == 0:
        return 0.0
    values = np.unique(sensitive_values)
    if values.size <= 1:
        return 0.0
    target_ratio = 1.0 / float(values.size)
    gaps = []
    for cid in np.unique(labels):
        mask = labels == cid
        size = int(np.sum(mask))
        if size <= 0:
            continue
        cluster_values = sensitive_values[mask]
        gap = 0.0
        for value in values:
            cluster_ratio = float(np.sum(cluster_values == value) / size)
            gap += abs(cluster_ratio - target_ratio)
        gaps.append(0.5 * gap)
    return float(np.mean(gaps)) if gaps else 0.0


def _stratified_subsample_df(
    df: pd.DataFrame,
    *,
    sensitive_col: str,
    sample_size: Optional[int],
    random_state: int,
) -> pd.DataFrame:
    if sample_size is None:
        return df
    sample_size = int(sample_size)
    if sample_size <= 0 or len(df) <= sample_size:
        return df

    strat_cols = []
    if "cluster_id" in df.columns:
        strat_cols.append("cluster_id")
    if sensitive_col in df.columns:
        strat_cols.append(sensitive_col)
    if not strat_cols:
        return df.sample(n=sample_size, random_state=int(random_state), replace=False).reset_index(drop=True)

    rng = np.random.default_rng(int(random_state))
    grouped = list(df.groupby(strat_cols, dropna=False, sort=False))
    total = len(df)
    raw_targets: list[float] = []
    base_counts: list[int] = []
    remainders: list[tuple[float, int]] = []
    picked_parts: list[pd.DataFrame] = []

    for idx, (_, group) in enumerate(grouped):
        target = float(sample_size) * float(len(group)) / float(total)
        base = min(len(group), int(np.floor(target)))
        raw_targets.append(target)
        base_counts.append(base)
        remainders.append((target - base, idx))

    assigned = sum(base_counts)
    need = max(0, sample_size - assigned)
    for _, idx in sorted(remainders, key=lambda item: (-item[0], item[1]))[:need]:
        if base_counts[idx] < len(grouped[idx][1]):
            base_counts[idx] += 1

    for idx, (_, group) in enumerate(grouped):
        take = int(base_counts[idx])
        if take <= 0:
            continue
        if take >= len(group):
            picked_parts.append(group)
            continue
        local_seed = int(rng.integers(0, 2**31 - 1))
        picked_parts.append(group.sample(n=take, random_state=local_seed, replace=False))

    if not picked_parts:
        return df.sample(n=sample_size, random_state=int(random_state), replace=False).reset_index(drop=True)
    sampled = pd.concat(picked_parts, ignore_index=False)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=int(random_state), replace=False)
    elif len(sampled) < sample_size:
        remaining = df.drop(index=sampled.index, errors="ignore")
        if len(remaining) > 0:
            extra_n = min(sample_size - len(sampled), len(remaining))
            extra = remaining.sample(n=extra_n, random_state=int(random_state), replace=False)
            sampled = pd.concat([sampled, extra], ignore_index=False)
    return sampled.reset_index(drop=True)


def evaluate_augmented_dataset(
    df: pd.DataFrame,
    *,
    sensitive_col: str,
    feature_col: Optional[str],
    n_clusters: int,
    random_state: int,
    batch_size: int,
    clustering_method: str = "kmeans",
    eval_subsample_size: Optional[int] = None,
    eval_subsample_random_state: Optional[int] = None,
    external_fair_algo_dir: Optional[str] = None,
    external_fair_delta: float = 0.2,
    external_fair_rounding: bool = True,
) -> EvalMetrics:
    df = ensure_embedding_columns(df)
    eval_df = _stratified_subsample_df(
        df,
        sensitive_col=sensitive_col,
        sample_size=eval_subsample_size,
        random_state=random_state if eval_subsample_random_state is None else int(eval_subsample_random_state),
    )
    embeddings = extract_embeddings(eval_df, feature_col=feature_col)
    if clustering_method == "kmeans":
        cluster_set, labels = fit_clusters(
            embeddings,
            n_clusters,
            random_state=random_state,
            subsample_ratio=None,
            subsample_size=None,
        )
        centers = cluster_set.centers
        sse = _compute_sse(embeddings, labels, centers)
        avg_radius = float(np.mean(cluster_set.radii)) if cluster_set.radii.size > 0 else 0.0
        max_radius = float(np.max(cluster_set.radii)) if cluster_set.radii.size > 0 else 0.0
        clustering_objective: Optional[float] = None
    elif clustering_method == "external_fair_relax_merge":
        external = run_external_fair_relax_merge(
            eval_df,
            sensitive_col=sensitive_col,
            feature_col=feature_col,
            n_clusters=n_clusters,
            delta=external_fair_delta,
            algo_dir=external_fair_algo_dir,
            rounding=external_fair_rounding,
            random_state=random_state,
        )
        labels = external.labels
        centers = external.centers
        sse = _compute_sse(embeddings, labels, centers)
        avg_radius, max_radius = _compute_radius_stats(embeddings, labels, centers)
        clustering_objective = float(external.objective)
    else:
        raise ValueError(f"Unsupported clustering_method: {clustering_method}")
    balance_gap = _balance_gap(labels, eval_df[sensitive_col].to_numpy())
    fairness_gap = _fairness_gap(labels, eval_df[sensitive_col].to_numpy())

    counts = eval_df[sensitive_col].value_counts(dropna=True)
    if counts.empty or len(counts) <= 1:
        maxsim_total = 0.0
    else:
        major_value = counts.idxmax()
        major_emb = extract_embeddings(eval_df[eval_df[sensitive_col] == major_value], feature_col=feature_col)
        minor_emb = extract_embeddings(eval_df[eval_df[sensitive_col] != major_value], feature_col=feature_col)
        maxsim_total = float(
            compute_maxsim(
                major_emb,
                minor_emb,
                metric="euclidean",
                batch_size=batch_size,
                return_argmax=False,
            )["total"]
        )

    return EvalMetrics(
        rows=int(len(eval_df)),
        sse=sse,
        avg_sse=sse / float(len(eval_df)) if len(eval_df) > 0 else 0.0,
        avg_radius=avg_radius,
        max_radius=max_radius,
        balance_gap=balance_gap,
        fairness_gap=fairness_gap,
        maxsim_total=maxsim_total,
        clustering_method=clustering_method,
        clustering_objective=clustering_objective,
    )
