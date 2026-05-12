from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .vector_ops import ensure_embedding_columns, extract_embeddings


@dataclass
class ExternalFairEvalResult:
    labels: np.ndarray
    centers: np.ndarray
    objective: float
    method: str
    rounded: bool


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {module_name!r} from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _encode_sensitive(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    unique = list(pd.unique(values))
    mapping = {str(value): idx for idx, value in enumerate(unique)}
    encoded = np.asarray([mapping[str(value)] for value in values], dtype=np.int32)
    return encoded, mapping


def _build_color_constraints(
    encoded: np.ndarray,
) -> tuple[dict[str, list[int]], dict[str, dict[int, float]], dict[str, dict[int, float]]]:
    values, counts = np.unique(encoded, return_counts=True)
    representation = {int(value): float(count / len(encoded)) for value, count in zip(values, counts)}
    color_flag = {"color": encoded.tolist()}
    # For final evaluation we use the external solver's native cluster-fairness form:
    # each cluster should match the dataset-wide group proportions, not a 50/50 target.
    alpha = {"color": {int(value): representation[int(value)] for value in values}}
    beta = {"color": {int(value): representation[int(value)] for value in values}}
    return color_flag, alpha, beta


def _prepare_numeric_frame(df: pd.DataFrame, *, feature_col: str, sensitive_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = ensure_embedding_columns(df)
    embeddings = extract_embeddings(df, feature_col=feature_col)
    encoded, _ = _encode_sensitive(df[sensitive_col].to_numpy())
    numeric = pd.DataFrame(embeddings, columns=[f"x{i}" for i in range(embeddings.shape[1])])
    numeric["color"] = encoded
    return numeric, encoded


def _build_initial_center_pool(
    points: np.ndarray,
    *,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    n_points, dim = points.shape
    if n_points <= 0:
        raise RuntimeError("Cannot build initial centers from an empty dataset.")
    if n_points <= n_clusters:
        return np.asarray(points, dtype=np.float64)

    rng = np.random.default_rng(int(random_state))
    subsample_size = int(min(n_points, max(500, 20 * int(n_clusters))))
    if subsample_size < n_points:
        subsample_idx = rng.choice(n_points, size=subsample_size, replace=False)
        subsample = points[subsample_idx]
    else:
        subsample = points

    kmeans_pool_size = int(min(len(subsample), max(2 * int(n_clusters), int(n_clusters) + 4)))
    random_pool_size = int(min(n_points, max(int(n_clusters), 8)))

    kmeans = KMeans(
        n_clusters=kmeans_pool_size,
        random_state=int(random_state),
        n_init=10,
    )
    kmeans.fit(subsample)
    center_parts = [np.asarray(kmeans.cluster_centers_, dtype=np.float64)]

    random_idx = rng.choice(n_points, size=random_pool_size, replace=False)
    center_parts.append(np.asarray(points[random_idx], dtype=np.float64))

    pool = np.vstack(center_parts)
    pool = np.unique(np.round(pool, decimals=12), axis=0)
    if pool.ndim == 1:
        pool = pool.reshape(1, dim)
    if pool.shape[0] < n_clusters:
        extra_idx = rng.choice(n_points, size=min(n_points, n_clusters), replace=False)
        pool = np.vstack([pool, points[extra_idx]])
        pool = np.unique(np.round(pool, decimals=12), axis=0)
    return np.asarray(pool, dtype=np.float64)


def run_external_fair_relax_merge(
    df: pd.DataFrame,
    *,
    sensitive_col: str,
    feature_col: str | None,
    n_clusters: int,
    delta: float,
    algo_dir: str | Path | None,
    rounding: bool = True,
    random_state: int = 42,
) -> ExternalFairEvalResult:
    algo_root = Path(algo_dir) if algo_dir else Path(__file__).resolve().parent / "fair_algorithms_for_clustering-master(2)"
    algo_root = algo_root.resolve()
    if not algo_root.exists():
        raise RuntimeError(f"Fair clustering algorithm directory does not exist: {algo_root}")

    solver_path = algo_root / "gurobi_fair_assignment_lp_solver.py"
    if not solver_path.exists():
        raise RuntimeError(f"gurobi_fair_assignment_lp_solver.py not found under: {algo_root}")

    if str(algo_root) not in sys.path:
        sys.path.insert(0, str(algo_root))

    try:
        solver_module = _load_module("raf_ext_gurobi_fair_assignment_lp_solver", solver_path)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The external fair clustering backend requires extra dependencies that are not available in the current "
            f"environment: {exc}. Install the package requirements for fair_algorithms_for_clustering first."
        ) from exc

    round_new = None
    if rounding:
        rounding_path = algo_root / "rounding.py"
        if not rounding_path.exists():
            raise RuntimeError(f"rounding.py not found under: {algo_root}")
        try:
            rounding_module = _load_module("raf_ext_rounding", rounding_path)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The external fair clustering rounding step requires extra dependencies that are not available in "
                f"the current environment: {exc}. Install the package requirements for fair_algorithms_for_clustering "
                "or disable rounding with --no-external_fair_rounding."
            ) from exc
        round_new = rounding_module.Round_new

    numeric_df, encoded = _prepare_numeric_frame(df, feature_col=feature_col, sensitive_col=sensitive_col)
    feature_df = numeric_df[[col for col in numeric_df.columns if col != "color"]].copy()
    color_flag, alpha, beta = _build_color_constraints(encoded)
    points = feature_df.to_numpy(dtype=np.float64, copy=False)
    cluster_centers = _build_initial_center_pool(
        points,
        n_clusters=int(n_clusters),
        random_state=int(random_state),
    )

    fair_partial_assignment = solver_module.fair_partial_assignment
    partial_res = fair_partial_assignment(feature_df, cluster_centers, alpha, beta, color_flag, False)
    assignment_matrix = np.asarray(partial_res["assignment"], dtype=np.float64).reshape(len(feature_df), cluster_centers.shape[0])
    center_weight = np.maximum(assignment_matrix.sum(axis=0), 0.0)

    kmeans = KMeans(n_clusters=int(n_clusters), random_state=int(random_state), n_init=10)
    kmeans.fit(cluster_centers, sample_weight=center_weight)
    final_centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    final_res = fair_partial_assignment(feature_df, final_centers, alpha, beta, color_flag, False)

    if rounding:
        assert round_new is not None
        objective, rounded_assignment = round_new(
            feature_df,
            final_centers,
            color_flag,
            np.asarray(final_res["assignment"], dtype=np.float64),
        )
        labels = np.asarray(np.argmax(rounded_assignment, axis=1), dtype=np.int32)
        rounded = True
    else:
        objective = float(final_res["objective"])
        labels = np.asarray(np.argmax(np.asarray(final_res["assignment"], dtype=np.float64), axis=1), dtype=np.int32)
        rounded = False

    return ExternalFairEvalResult(
        labels=labels,
        centers=final_centers,
        objective=float(objective),
        method="external_fair_relax_merge_kmeans_init",
        rounded=rounded,
    )
