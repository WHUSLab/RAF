from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .bandit import (
    BanditState,
    EpsilonGreedyPolicy,
    ExploreExploitPolicy,
    FairnessWeightedEpsGreedyPolicy,
    RandomPolicy,
    normalize_policy_name,
)
from .clustering import ClusterSet, fit_clusters
from .config import RAFConfig, SourceSpec
from .data import SourceManager, build_sources_from_specs, normalize_record
from .demand import (
    GroupLabeler,
    build_counts_from_labels,
    build_group_labeler,
    recompute_group_level_q_from_values,
    recompute_q_from_counts,
)
from .evaluation import EvalMetrics, evaluate_augmented_dataset
from .selection import (
    SelectState,
    build_initial_select_states,
    clone_select_state,
    split_major_minor,
    try_accept_baseline_exact,
    try_accept_incremental,
    try_accept_incremental_hybrid,
)
from .vector_ops import ensure_embedding_columns, extract_embeddings


_QUERY_DF_CACHE: dict[str, pd.DataFrame] = {}
_PREPARED_STATE_CACHE: dict[str, "PreparedState"] = {}


@dataclass
class PreparedState:
    query_df: pd.DataFrame
    cluster_set: Optional[ClusterSet]
    group_labeler: GroupLabeler
    demand: np.ndarray
    q_flat: np.ndarray
    cluster_group_demand: np.ndarray
    major_df: pd.DataFrame
    minor_dfs: dict[object, pd.DataFrame]
    major_value: object
    major_vectors: np.ndarray
    major_norm_sq: np.ndarray
    select_states: dict[object, SelectState]
    init_profile: dict[str, float]
    major_cluster_labels: Optional[np.ndarray]
    minor_cluster_labels_by_group: dict[object, np.ndarray]


@dataclass
class RAFRunResult:
    evaluated: int
    sampled: int
    candidates: int
    accepted: int
    skipped_not_minor: int
    skipped_gap_zero: int
    total_gain: float
    total_cost: float
    elapsed_s: float
    init_s: float
    query_load_s: float
    cluster_s: float
    select_state_build_s: float
    source_load_s: float
    loop_s: float
    policy_s: float
    select_s: float
    avg_internal_candidates: Optional[float]
    select_profile: Optional[dict[str, float]]
    valuation_mode: str
    source_selected: list[int]
    source_sampled: list[int]
    source_minority_sampled: list[int]
    source_candidates: list[int]
    source_names: list[str]
    accepted_records: list[dict[str, Any]]
    init_profile: dict[str, float]
    initial_eval: Optional[EvalMetrics]
    final_eval: Optional[EvalMetrics]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.final_eval is not None:
            payload["final_eval"] = asdict(self.final_eval)
        return payload


@dataclass
class CandidateCollection:
    evaluated: int
    sampled: int
    candidates: int
    skipped_not_minor: int
    skipped_gap_zero: int
    total_cost: float
    source_selected: list[int]
    source_sampled: list[int]
    source_minority_sampled: list[int]
    source_candidates: list[int]
    source_names: list[str]
    candidate_records: list[dict[str, Any]]
    policy_s: float
    candidate_loop_s: float


class RAFPipeline:
    def __init__(
        self,
        config: RAFConfig,
        *,
        source_specs: Optional[list[SourceSpec]] = None,
        progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
        progress_every: int = 1000,
    ) -> None:
        self.config = config
        self.source_specs = list(source_specs) if source_specs is not None else None
        self.progress_callback = progress_callback
        self.progress_every = max(1, int(progress_every))

    def _emit_progress(self, event: str, **payload: Any) -> None:
        if self.progress_callback is None:
            return
        body = {
            "event": event,
            "policy": self.config.policy,
            "valuation_mode": self.config.valuation_mode,
        }
        body.update(payload)
        self.progress_callback(body)

    @staticmethod
    def _nonzero_source_counts(names: list[str], counts: np.ndarray) -> dict[str, int]:
        items: list[tuple[str, int]] = []
        for idx, count in enumerate(np.asarray(counts, dtype=np.int64).tolist()):
            count = int(count)
            if count <= 0:
                continue
            items.append((str(names[idx]), count))
        items.sort(key=lambda item: (-item[1], item[0]))
        return dict(items)

    def _load_query_df(self) -> pd.DataFrame:
        if not self.config.query_path:
            raise ValueError("query_path is required unless query_df is passed directly.")
        cache_key = str(pd.io.common.stringify_path(self.config.query_path))
        if cache_key in _QUERY_DF_CACHE:
            self._emit_progress("query_cache_hit", query_path=self.config.query_path)
            return _QUERY_DF_CACHE[cache_key].copy()
        self._emit_progress("query_load_start", query_path=self.config.query_path)
        try:
            df = pd.read_parquet(self.config.query_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read query parquet '{self.config.query_path}'. Install a parquet engine such as pyarrow."
            ) from exc
        _QUERY_DF_CACHE[cache_key] = df
        self._emit_progress("query_load_done", query_path=self.config.query_path, rows=int(len(df)))
        return df.copy()

    def _query_cache_key(self) -> Optional[str]:
        if not self.config.query_path:
            return None
        path = pd.io.common.stringify_path(self.config.query_path)
        resolved = str(Path(path).resolve())
        try:
            stat = Path(resolved).stat()
            return f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
        except OSError:
            return resolved

    def _prepare_config_fingerprint(self) -> str:
        payload = {
            "policy": normalize_policy_name(self.config.policy),
            "sensitive_col": self.config.sensitive_col,
            "feature_col": self.config.feature_col,
            "n_clusters": self.config.n_clusters,
            "cluster_random_state": (
                self.config.cluster_random_state
                if self.config.cluster_random_state is not None
                else self.config.random_state
            ),
            "cluster_subsample_ratio": self.config.cluster_subsample_ratio,
            "cluster_subsample_size": self.config.cluster_subsample_size,
            "cluster_assign_batch_size": self.config.cluster_assign_batch_size,
            "valuation_mode": self.config.valuation_mode,
            "metric": self.config.metric,
            "batch_size": self.config.batch_size,
            "hnsw_single_index": self.config.hnsw_single_index,
            "hnsw_delta_capacity": self.config.hnsw_delta_capacity,
            "hnsw_rebuild_delta_threshold": self.config.hnsw_rebuild_delta_threshold,
            "hnsw_ef_search": self.config.hnsw_ef_search,
            "hnsw_m": self.config.hnsw_m,
            "hnsw_ef_construction": self.config.hnsw_ef_construction,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cluster_random_state(self) -> int:
        if self.config.cluster_random_state is not None:
            return int(self.config.cluster_random_state)
        return int(self.config.random_state)

    def _eval_random_state(self) -> int:
        if self.config.eval_random_state is not None:
            return int(self.config.eval_random_state)
        return self._cluster_random_state()

    def _evaluate_dataset(self, df: pd.DataFrame) -> EvalMetrics:
        return evaluate_augmented_dataset(
            df,
            sensitive_col=self.config.sensitive_col,
            feature_col=self.config.feature_col,
            n_clusters=self.config.eval_n_clusters,
            random_state=self._eval_random_state(),
            batch_size=self.config.batch_size,
            clustering_method=self.config.eval_clustering_method,
            eval_subsample_size=self.config.eval_subsample_size,
            eval_subsample_random_state=self.config.eval_subsample_random_state,
            external_fair_algo_dir=self.config.external_fair_algo_dir,
            external_fair_delta=self.config.external_fair_delta,
            external_fair_rounding=self.config.external_fair_rounding,
        )

    def _prepare_cache_key(self) -> Optional[str]:
        query_key = self._query_cache_key()
        if query_key is None:
            return None
        return f"{query_key}|{self._prepare_config_fingerprint()}"

    def _clone_prepared_state(self, prepared: PreparedState) -> PreparedState:
        return PreparedState(
            query_df=prepared.query_df.copy(deep=True),
            cluster_set=None
            if prepared.cluster_set is None
            else ClusterSet(
                centers=np.asarray(prepared.cluster_set.centers, dtype=np.float32).copy(),
                radii=np.asarray(prepared.cluster_set.radii, dtype=np.float32).copy(),
            ),
            group_labeler=GroupLabeler(values=list(prepared.group_labeler.values)),
            demand=np.asarray(prepared.demand, dtype=np.float64).copy(),
            q_flat=np.asarray(prepared.q_flat, dtype=np.float64).copy(),
            cluster_group_demand=np.asarray(prepared.cluster_group_demand, dtype=np.float64).copy(),
            major_df=prepared.major_df.copy(deep=True),
            minor_dfs={group: df.copy(deep=True) for group, df in prepared.minor_dfs.items()},
            major_value=prepared.major_value,
            major_vectors=np.asarray(prepared.major_vectors, dtype=np.float32).copy(),
            major_norm_sq=np.asarray(prepared.major_norm_sq, dtype=np.float32).copy(),
            select_states={group: clone_select_state(state) for group, state in prepared.select_states.items()},
            init_profile=dict(prepared.init_profile),
            major_cluster_labels=None
            if prepared.major_cluster_labels is None
            else np.asarray(prepared.major_cluster_labels, dtype=np.int32).copy(),
            minor_cluster_labels_by_group={
                group: np.asarray(labels, dtype=np.int32).copy()
                for group, labels in prepared.minor_cluster_labels_by_group.items()
            },
        )

    def _build_source_manager(self) -> SourceManager:
        if self.source_specs is None:
            raise ValueError("Standalone RAFPipeline requires explicit source_specs.")
        specs = self.source_specs
        if self.config.limit_sources is not None:
            specs = specs[: int(self.config.limit_sources)]
        self._emit_progress("source_manager_build_start", num_specs=int(len(specs)))
        manager = build_sources_from_specs(specs)
        if len(manager) <= 0:
            raise ValueError("No sources configured.")
        self._emit_progress("source_manager_build_done", num_sources=int(len(manager)))
        return manager

    def _build_policy(self, bandit_state: BanditState):
        policy_name = normalize_policy_name(self.config.policy)
        if policy_name == "random":
            return RandomPolicy(seed=self.config.random_state)
        if policy_name == "epsilon_greedy":
            return EpsilonGreedyPolicy(
                alpha=self.config.epsilon_alpha,
                min_epsilon=self.config.epsilon_min,
                seed=self.config.random_state,
            )
        if policy_name == "fair_eps_greedy":
            return FairnessWeightedEpsGreedyPolicy(
                alpha_eps=self.config.fair_alpha_eps,
                eps_min=self.config.fair_eps_min,
                kappa=self.config.fair_kappa,
                epsilon_p=self.config.fair_epsilon_p,
                enable_pruning=self.config.fair_enable_pruning,
                prune_budget_interval=self.config.fair_prune_budget_interval,
                prune_fraction=self.config.fair_prune_fraction,
                min_active_sources=self.config.fair_min_active_sources,
                prune_use_uniqueness=self.config.fair_prune_use_uniqueness,
                seed=self.config.random_state,
            )
        if policy_name == "explore_exploit":
            return ExploreExploitPolicy(
                explore_budget_fraction=self.config.explore_budget_fraction,
                seed=self.config.random_state,
            )
        raise ValueError(f"Unsupported policy '{self.config.policy}'.")

    def _valuate_sample(
        self,
        state: SelectState,
        candidate_vector: np.ndarray,
        *,
        cluster_set: ClusterSet,
        major_vectors: np.ndarray,
        major_norm_sq: np.ndarray,
        candidate_cluster: Optional[int],
    ):
        valuation_mode = self.config.valuation_mode.strip().lower()
        if valuation_mode == "baseline":
            return try_accept_baseline_exact(
                state,
                major_vectors,
                candidate_vector,
                major_norm_sq=major_norm_sq,
                eps=self.config.eps,
                metric=self.config.metric,
                batch_size=self.config.batch_size,
            )
        if valuation_mode == "incremental":
            return try_accept_incremental(
                state,
                major_vectors,
                candidate_vector,
                major_norm_sq=major_norm_sq,
                eps=self.config.eps,
                metric=self.config.metric,
                candidate_cluster=candidate_cluster,
            )
        if valuation_mode == "incremental_hybrid":
            return try_accept_incremental_hybrid(
                state,
                major_vectors,
                candidate_vector,
                major_norm_sq=major_norm_sq,
                eps=self.config.eps,
                metric=self.config.metric,
                neighbor_k=self.config.hybrid_neighbor_k,
                candidate_cluster=candidate_cluster,
                allowed_clusters=None,
            )
        raise ValueError(f"Unsupported valuation_mode '{self.config.valuation_mode}'.")

    def _allowed_cluster_ids(self, cluster_set: ClusterSet, candidate_cluster: int) -> set[int]:
        expand = max(0, int(self.config.hybrid_cluster_expand))
        if expand <= 0:
            return {int(candidate_cluster)}
        centers = np.asarray(cluster_set.centers, dtype=np.float32)
        if candidate_cluster < 0 or candidate_cluster >= centers.shape[0]:
            return {int(candidate_cluster)}
        diffs = centers - centers[int(candidate_cluster)]
        dists = np.sum(diffs * diffs, axis=1, dtype=np.float32)
        order = np.argsort(dists)
        allowed: list[int] = []
        for cid in order.tolist():
            cid = int(cid)
            allowed.append(cid)
            if len(allowed) >= expand + 1:
                break
        return set(allowed)

    def _policy_uses_cluster_group_gap(self) -> bool:
        policy_name = normalize_policy_name(self.config.policy)
        return policy_name in {"random", "fair_eps_greedy"}

    def _policy_requires_prevaluation_cluster_gate(self) -> bool:
        policy_name = normalize_policy_name(self.config.policy)
        return policy_name in {"epsilon_greedy", "explore_exploit"}

    def _policy_uses_warmup(self) -> bool:
        policy_name = normalize_policy_name(self.config.policy)
        return policy_name in {"epsilon_greedy", "fair_eps_greedy"}

    def _select_warmup_source(
        self,
        *,
        bandit_state: BanditState,
        available_mask: np.ndarray,
    ) -> int | None:
        warmup_rounds = max(0, int(self.config.policy_warmup_rounds))
        if warmup_rounds <= 0 or not self._policy_uses_warmup():
            return None
        pending = np.flatnonzero(available_mask & (bandit_state.N_i < warmup_rounds))
        if pending.size == 0:
            return None
        min_seen = int(np.min(bandit_state.N_i[pending]))
        candidates = pending[bandit_state.N_i[pending] == min_seen]
        return int(candidates[bandit_state.t % len(candidates)])

    def _build_initial_demand(
        self,
        *,
        labels: np.ndarray,
        sensitive_values: np.ndarray,
        group_values: list[Any],
    ) -> np.ndarray:
        if self._policy_uses_cluster_group_gap():
            counts = build_counts_from_labels(labels, sensitive_values, group_values)
            return recompute_q_from_counts(counts)
        return recompute_group_level_q_from_values(sensitive_values, group_values)

    def prepare(self, query_df: Optional[pd.DataFrame] = None) -> PreparedState:
        cache_key = None if query_df is not None else self._prepare_cache_key()
        if cache_key is not None and cache_key in _PREPARED_STATE_CACHE:
            self._emit_progress("prepare_cache_hit")
            return self._clone_prepared_state(_PREPARED_STATE_CACHE[cache_key])

        start = time.perf_counter()
        init_profile: dict[str, float] = {}
        self._emit_progress("prepare_start")
        query_load_start = time.perf_counter()
        query_df = self._load_query_df() if query_df is None else query_df.copy()
        init_profile["query_load"] = time.perf_counter() - query_load_start

        stage = time.perf_counter()
        query_df = ensure_embedding_columns(query_df)
        query_df = query_df[query_df[self.config.sensitive_col].notna()].reset_index(drop=True)
        init_profile["prepare_query"] = time.perf_counter() - stage
        self._emit_progress("prepare_query_done", rows=int(len(query_df)))

        stage = time.perf_counter()
        self._emit_progress("prepare_cluster_start", rows=int(len(query_df)), n_clusters=int(self.config.n_clusters))
        group_labeler = build_group_labeler(query_df, self.config.sensitive_col)
        cluster_set = None
        labels = np.zeros((len(query_df),), dtype=np.int32)
        embeddings = extract_embeddings(query_df, feature_col=self.config.feature_col)
        cluster_set, labels = fit_clusters(
            embeddings,
            self.config.n_clusters,
            random_state=self._cluster_random_state(),
            subsample_ratio=self.config.cluster_subsample_ratio,
            subsample_size=self.config.cluster_subsample_size,
            assign_batch_size=self.config.cluster_assign_batch_size,
        )
        counts = build_counts_from_labels(labels, query_df[self.config.sensitive_col].to_numpy(), group_labeler.values)
        cluster_group_demand = recompute_q_from_counts(counts)
        uses_cluster_group_gap = self._policy_uses_cluster_group_gap()
        demand = self._build_initial_demand(
            labels=labels,
            sensitive_values=query_df[self.config.sensitive_col].to_numpy(),
            group_values=group_labeler.values,
        )
        init_profile["cluster_and_demand"] = time.perf_counter() - stage
        self._emit_progress(
            "prepare_cluster_done",
            groups=int(len(group_labeler.values)),
            demand_rows=int(demand.shape[0]),
            demand_cols=int(demand.shape[1]),
            clustered=bool(uses_cluster_group_gap),
            stage_elapsed_s=float(init_profile["cluster_and_demand"]),
        )

        stage = time.perf_counter()
        major_df, minor_dfs, major_value = split_major_minor(query_df, self.config.sensitive_col)
        self._emit_progress(
            "prepare_select_state_start",
            major_rows=int(len(major_df)),
            minor_groups=int(len(minor_dfs)),
            minor_rows=int(sum(len(df) for df in minor_dfs.values())),
        )
        major_embeddings_full = extract_embeddings(major_df, feature_col=self.config.feature_col)
        major_cluster_labels = None
        if cluster_set is not None:
            major_cluster_labels = cluster_set.assign_clusters(
                major_embeddings_full,
                batch_size=self.config.cluster_assign_batch_size,
            )
        minor_cluster_labels_by_group: dict[object, np.ndarray] = {}
        if cluster_set is not None:
            for group_value, minor_df in minor_dfs.items():
                minor_embeddings = extract_embeddings(minor_df, feature_col=self.config.feature_col)
                minor_cluster_labels_by_group[group_value] = cluster_set.assign_clusters(
                    minor_embeddings,
                    batch_size=self.config.cluster_assign_batch_size,
                )
        major_vectors, major_norm_sq, select_states = build_initial_select_states(
            major_df,
            minor_dfs,
            feature_col=self.config.feature_col,
            metric=self.config.metric,
            batch_size=self.config.batch_size,
            major_cluster_labels=major_cluster_labels,
            minor_cluster_labels_by_group=minor_cluster_labels_by_group,
            use_hnsw=self.config.valuation_mode.strip().lower() == "incremental_hybrid",
            single_hnsw_index=self.config.hnsw_single_index,
            delta_capacity=self.config.hnsw_delta_capacity,
            hnsw_rebuild_delta_threshold=self.config.hnsw_rebuild_delta_threshold,
            hnsw_ef_search=self.config.hnsw_ef_search,
            hnsw_m=self.config.hnsw_m,
            hnsw_ef_construction=self.config.hnsw_ef_construction,
        )
        init_profile["build_select_state"] = time.perf_counter() - stage
        init_profile["total_prepare"] = time.perf_counter() - start
        self._emit_progress(
            "prepare_select_state_done",
            stage_elapsed_s=float(init_profile["build_select_state"]),
            total_prepare_s=float(init_profile["total_prepare"]),
        )

        prepared = PreparedState(
            query_df=query_df,
            cluster_set=cluster_set,
            group_labeler=group_labeler,
            demand=demand,
            q_flat=demand.reshape(-1).astype(float, copy=False),
            cluster_group_demand=cluster_group_demand,
            major_df=major_df,
            minor_dfs=minor_dfs,
            major_value=major_value,
            major_vectors=major_vectors,
            major_norm_sq=major_norm_sq,
            select_states=select_states,
            init_profile=init_profile,
            major_cluster_labels=major_cluster_labels,
            minor_cluster_labels_by_group=minor_cluster_labels_by_group,
        )
        if cache_key is not None:
            _PREPARED_STATE_CACHE[cache_key] = prepared
            self._emit_progress("prepare_done", cached=True, total_prepare_s=float(init_profile["total_prepare"]))
            return self._clone_prepared_state(prepared)
        self._emit_progress("prepare_done", cached=False, total_prepare_s=float(init_profile["total_prepare"]))
        return prepared

    def collect_candidates(
        self,
        *,
        query_df: Optional[pd.DataFrame] = None,
        source_manager: Optional[SourceManager] = None,
    ) -> CandidateCollection:
        self._emit_progress("candidate_collection_start")
        prepared = self.prepare(query_df=query_df)
        source_manager = self._build_source_manager() if source_manager is None else source_manager
        costs = source_manager.costs()
        bandit_state = BanditState(
            len(source_manager),
            prepared.demand.size,
            num_groups=prepared.demand.shape[1],
        )
        policy = self._build_policy(bandit_state)

        evaluated = 0
        sampled = 0
        candidates = 0
        skipped_not_minor = 0
        skipped_gap_zero = 0
        total_cost = 0.0
        policy_s = 0.0
        source_selected = np.zeros((len(source_manager),), dtype=np.int64)
        source_sampled = np.zeros((len(source_manager),), dtype=np.int64)
        source_minority_sampled = np.zeros((len(source_manager),), dtype=np.int64)
        source_candidates = np.zeros((len(source_manager),), dtype=np.int64)
        candidate_records: list[dict[str, Any]] = []

        available_mask = np.ones((len(source_manager),), dtype=bool)
        loop_start = time.perf_counter()
        stop_reason = "completed"
        try:
            while evaluated < self.config.max_steps:
                if self.config.max_cost is not None and total_cost >= self.config.max_cost:
                    stop_reason = "max_cost"
                    break

                for si in range(len(source_manager.sources)):
                    if available_mask[si] and source_manager.sources[si].exhausted:
                        available_mask[si] = False
                if not np.any(available_mask):
                    stop_reason = "no_available_sources"
                    break

                policy_start = time.perf_counter()
                warmup_source_idx = self._select_warmup_source(
                    bandit_state=bandit_state,
                    available_mask=available_mask,
                )
                if warmup_source_idx is not None:
                    source_idx = int(warmup_source_idx)
                else:
                    source_idx = policy.select_source(
                        state=bandit_state,
                        q_flat=prepared.q_flat,
                        costs=costs,
                        available_mask=available_mask,
                        current_cost=float(total_cost),
                        max_cost=self.config.max_cost,
                    )
                policy_s += time.perf_counter() - policy_start
                source_selected[source_idx] += 1

                sample = source_manager.get(source_idx).sample_one()
                if sample is None:
                    evaluated += 1
                    continue

                sampled += 1
                source_sampled[source_idx] += 1

                uses_cluster_group_gap = self._policy_uses_cluster_group_gap()
                requires_prevaluation_cluster_gate = self._policy_requires_prevaluation_cluster_gate()
                group_value = sample.get(self.config.sensitive_col, sample.get("is_english_name"))
                if group_value in prepared.select_states:
                    source_minority_sampled[source_idx] += 1
                group_id = prepared.group_labeler.get_group(sample, self.config.sensitive_col)
                cluster_id = None
                if prepared.cluster_set is None:
                    raise RuntimeError("cluster_set is required for candidate processing.")
                cluster_id = prepared.cluster_set.assign_cluster(np.asarray(sample["embedding"], dtype=np.float32))
                if group_id is None:
                    combo_idx = None
                elif uses_cluster_group_gap and cluster_id is not None:
                    combo_idx = int(cluster_id * prepared.demand.shape[1] + group_id)
                elif not uses_cluster_group_gap:
                    combo_idx = int(group_id)
                else:
                    combo_idx = None
                bandit_state.update_observation(source_idx, combo_idx)

                if group_value not in prepared.select_states:
                    skipped_not_minor += 1
                    total_cost += float(costs[source_idx])
                    evaluated += 1
                    continue

                if group_id is None or (uses_cluster_group_gap and cluster_id is None):
                    skipped_gap_zero += 1
                    total_cost += float(costs[source_idx])
                    evaluated += 1
                    continue

                demand_row = int(cluster_id) if uses_cluster_group_gap else 0
                if prepared.demand[demand_row, group_id] <= 0:
                    skipped_gap_zero += 1
                    total_cost += float(costs[source_idx])
                    evaluated += 1
                    continue

                candidates += 1
                source_candidates[source_idx] += 1
                bandit_state.mark_effective(source_idx)
                demand_decay = max(0.0, float(self.config.demand_decay_per_candidate))
                prepared.demand[demand_row, group_id] = max(
                    0.0,
                    prepared.demand[demand_row, group_id] - demand_decay,
                )
                prepared.q_flat[combo_idx] = prepared.demand[demand_row, group_id]

                if requires_prevaluation_cluster_gate:
                    if cluster_id is None:
                        skipped_gap_zero += 1
                        total_cost += float(costs[source_idx])
                        evaluated += 1
                        continue
                    gate_cluster_id = int(cluster_id)
                    if prepared.cluster_group_demand[gate_cluster_id, group_id] <= 0:
                        skipped_gap_zero += 1
                        total_cost += float(costs[source_idx])
                        evaluated += 1
                        continue
                    prepared.cluster_group_demand[gate_cluster_id, group_id] = max(
                        0.0,
                        prepared.cluster_group_demand[gate_cluster_id, group_id] - demand_decay,
                    )
                candidate_records.append(dict(sample))

                total_cost += float(costs[source_idx])
                evaluated += 1

                if self.config.use_select_feedback:
                    policy.on_observation(bandit_state, source_idx)
            else:
                stop_reason = "max_steps"
        finally:
            source_manager.close_all()

        candidate_loop_s = time.perf_counter() - loop_start
        self._emit_progress(
            "candidate_collection_done",
            stop_reason=stop_reason,
            evaluated=int(evaluated),
            candidates=int(candidates),
            total_cost=float(total_cost),
            candidate_loop_s=float(candidate_loop_s),
        )
        return CandidateCollection(
            evaluated=evaluated,
            sampled=sampled,
            candidates=candidates,
            skipped_not_minor=skipped_not_minor,
            skipped_gap_zero=skipped_gap_zero,
            total_cost=float(total_cost),
            source_selected=source_selected.tolist(),
            source_sampled=source_sampled.tolist(),
            source_minority_sampled=source_minority_sampled.tolist(),
            source_candidates=source_candidates.tolist(),
            source_names=[source.name for source in source_manager.sources],
            candidate_records=candidate_records,
            policy_s=float(policy_s),
            candidate_loop_s=float(candidate_loop_s),
        )

    def replay_candidates(
        self,
        candidates: CandidateCollection | list[dict[str, Any]],
        *,
        query_df: Optional[pd.DataFrame] = None,
        return_final_eval: bool = True,
        evaluate_outputs: bool = True,
    ) -> RAFRunResult:
        prepared = self.prepare(query_df=query_df)
        init_s = float(prepared.init_profile.get("total_prepare", 0.0))
        if isinstance(candidates, CandidateCollection):
            candidate_records = list(candidates.candidate_records)
            evaluated = int(candidates.evaluated)
            sampled = int(candidates.sampled)
            candidate_count = int(candidates.candidates)
            skipped_not_minor = int(candidates.skipped_not_minor)
            skipped_gap_zero = int(candidates.skipped_gap_zero)
            total_cost = float(candidates.total_cost)
            source_selected = list(candidates.source_selected)
            source_sampled = list(candidates.source_sampled)
            source_minority_sampled = list(candidates.source_minority_sampled)
            source_candidates = list(candidates.source_candidates)
            source_names = list(candidates.source_names)
        else:
            candidate_records = list(candidates)
            evaluated = len(candidate_records)
            sampled = len(candidate_records)
            candidate_count = len(candidate_records)
            skipped_not_minor = 0
            skipped_gap_zero = 0
            total_cost = float(candidate_count)
            source_selected = []
            source_sampled = []
            source_minority_sampled = []
            source_candidates = []
            source_names = []

        accepted = 0
        total_gain = 0.0
        select_s = 0.0
        cand_total = 0
        cand_samples = 0
        select_profile_totals: dict[str, float] = {}
        accepted_records: list[dict[str, Any]] = []

        self._emit_progress(
            "candidate_replay_start",
            valuation_mode=self.config.valuation_mode,
            candidates=int(len(candidate_records)),
        )
        loop_start = time.perf_counter()
        for sample in candidate_records:
            group_value = sample.get(self.config.sensitive_col, sample.get("is_english_name"))
            state = prepared.select_states.get(group_value)
            if state is None:
                continue
            cluster_id = None
            if prepared.cluster_set is not None:
                cluster_id = prepared.cluster_set.assign_cluster(np.asarray(sample["embedding"], dtype=np.float32))
            select_start = time.perf_counter()
            result = self._valuate_sample(
                state,
                np.asarray(sample["embedding"], dtype=np.float32),
                cluster_set=prepared.cluster_set,
                major_vectors=prepared.major_vectors,
                major_norm_sq=prepared.major_norm_sq,
                candidate_cluster=cluster_id,
            )
            select_s += time.perf_counter() - select_start
            if result.accept:
                accepted += 1
                total_gain += float(result.gain)
                accepted_records.append(dict(sample))
            if result.num_candidates is not None:
                cand_total += int(result.num_candidates)
                cand_samples += 1
            if result.profile:
                for key, value in result.profile.items():
                    select_profile_totals[key] = select_profile_totals.get(key, 0.0) + float(value)
        loop_s = time.perf_counter() - loop_start

        initial_eval = self._evaluate_dataset(prepared.query_df) if evaluate_outputs else None
        final_eval = None
        if return_final_eval and evaluate_outputs:
            frames = [prepared.query_df]
            if accepted_records:
                frames.append(pd.DataFrame(accepted_records))
            augmented_df = pd.concat(frames, ignore_index=True)
            final_eval = self._evaluate_dataset(augmented_df)
        self._emit_progress(
            "candidate_replay_done",
            valuation_mode=self.config.valuation_mode,
            candidates=int(len(candidate_records)),
            accepted=int(accepted),
            select_s=float(select_s),
            loop_s=float(loop_s),
        )
        return RAFRunResult(
            evaluated=evaluated,
            sampled=sampled,
            candidates=candidate_count,
            accepted=accepted,
            skipped_not_minor=skipped_not_minor,
            skipped_gap_zero=skipped_gap_zero,
            total_gain=float(total_gain),
            total_cost=float(total_cost),
            elapsed_s=float(loop_s),
            init_s=float(init_s),
            query_load_s=float(prepared.init_profile.get("query_load", 0.0)),
            cluster_s=float(prepared.init_profile.get("cluster_and_demand", 0.0)),
            select_state_build_s=float(prepared.init_profile.get("build_select_state", 0.0)),
            source_load_s=0.0,
            loop_s=float(loop_s),
            policy_s=0.0,
            select_s=float(select_s),
            avg_internal_candidates=None if cand_samples <= 0 else float(cand_total / cand_samples),
            select_profile=dict(select_profile_totals) if select_profile_totals else None,
            valuation_mode=self.config.valuation_mode,
            source_selected=source_selected,
            source_sampled=source_sampled,
            source_minority_sampled=source_minority_sampled,
            source_candidates=source_candidates,
            source_names=source_names,
            accepted_records=accepted_records,
            init_profile=prepared.init_profile,
            initial_eval=initial_eval,
            final_eval=final_eval,
        )

    def run(
        self,
        *,
        query_df: Optional[pd.DataFrame] = None,
        source_manager: Optional[SourceManager] = None,
        return_final_eval: bool = True,
        evaluate_outputs: bool = True,
    ) -> RAFRunResult:
        self._emit_progress("run_start", return_final_eval=bool(return_final_eval))
        init_start = time.perf_counter()
        prepared = self.prepare(query_df=query_df)
        init_s = time.perf_counter() - init_start
        self._emit_progress("run_prepare_done", init_s=float(init_s))

        source_load_start = time.perf_counter()
        source_manager = self._build_source_manager() if source_manager is None else source_manager
        source_load_s = time.perf_counter() - source_load_start
        costs = source_manager.costs()
        bandit_state = BanditState(
            len(source_manager),
            prepared.demand.size,
            num_groups=prepared.demand.shape[1],
        )
        policy = self._build_policy(bandit_state)
        accepted_records: list[dict[str, Any]] = []

        evaluated = 0
        sampled = 0
        candidates = 0
        accepted = 0
        skipped_not_minor = 0
        skipped_gap_zero = 0
        total_gain = 0.0
        total_cost = 0.0
        policy_s = 0.0
        select_s = 0.0
        cand_total = 0
        cand_samples = 0
        select_profile_totals: dict[str, float] = {}
        source_selected = np.zeros((len(source_manager),), dtype=np.int64)
        source_sampled = np.zeros((len(source_manager),), dtype=np.int64)
        source_minority_sampled = np.zeros((len(source_manager),), dtype=np.int64)
        source_candidates = np.zeros((len(source_manager),), dtype=np.int64)
        snapshot_fraction = self.config.budget_snapshot_fraction
        next_budget_snapshot_cost: float | None = None
        last_snapshot_selected = np.zeros((len(source_manager),), dtype=np.int64)
        last_snapshot_candidates = np.zeros((len(source_manager),), dtype=np.int64)
        if (
            snapshot_fraction is not None
            and self.config.max_cost is not None
            and float(snapshot_fraction) > 0.0
        ):
            next_budget_snapshot_cost = float(self.config.max_cost) * float(snapshot_fraction)

        available_mask = np.ones((len(source_manager),), dtype=bool)
        loop_start = time.perf_counter()
        self._emit_progress(
            "loop_start",
            num_sources=int(len(source_manager)),
            max_steps=int(self.config.max_steps),
            max_cost=None if self.config.max_cost is None else float(self.config.max_cost),
            max_accepted=None if self.config.max_accepted is None else int(self.config.max_accepted),
        )
        stop_reason = "completed"
        try:
            while evaluated < self.config.max_steps:
                if self.config.max_cost is not None and total_cost >= self.config.max_cost:
                    stop_reason = "max_cost"
                    break
                if self.config.max_accepted is not None and accepted >= self.config.max_accepted:
                    stop_reason = "max_accepted"
                    break

                # Refresh exhausted flags for all currently-available sources
                for si in range(len(source_manager.sources)):
                    if available_mask[si] and source_manager.sources[si].exhausted:
                        available_mask[si] = False
                if not np.any(available_mask):
                    stop_reason = "no_available_sources"
                    break

                policy_start = time.perf_counter()
                warmup_source_idx = self._select_warmup_source(
                    bandit_state=bandit_state,
                    available_mask=available_mask,
                )
                if warmup_source_idx is not None:
                    source_idx = int(warmup_source_idx)
                    self._emit_progress(
                        "warmup_select",
                        source_idx=int(source_idx),
                        source_name=str(source_manager.sources[source_idx].name),
                        source_seen=int(bandit_state.N_i[source_idx]),
                        warmup_rounds=int(self.config.policy_warmup_rounds),
                    )
                else:
                    source_idx = policy.select_source(
                        state=bandit_state,
                        q_flat=prepared.q_flat,
                        costs=costs,
                        available_mask=available_mask,
                        current_cost=float(total_cost),
                        max_cost=self.config.max_cost,
                    )
                policy_s += time.perf_counter() - policy_start
                pruned_sources = getattr(policy, "last_pruned_sources", None)
                if pruned_sources:
                    disabled_mask = getattr(policy, "disabled", np.zeros_like(available_mask, dtype=bool))
                    self._emit_progress(
                        "policy_prune",
                        pruned_sources=[int(idx) for idx in pruned_sources],
                        active_sources=int(np.sum(available_mask & (~disabled_mask))),
                    )
                source_selected[source_idx] += 1

                sample = source_manager.get(source_idx).sample_one()
                if sample is None:
                    evaluated += 1
                    continue
                sampled += 1
                source_sampled[source_idx] += 1
                # sample is already normalized by InMemorySource/ParquetSource

                uses_cluster_group_gap = self._policy_uses_cluster_group_gap()
                requires_prevaluation_cluster_gate = self._policy_requires_prevaluation_cluster_gate()
                group_value = sample.get(self.config.sensitive_col, sample.get("is_english_name"))
                if group_value in prepared.select_states:
                    source_minority_sampled[source_idx] += 1
                group_id = prepared.group_labeler.get_group(sample, self.config.sensitive_col)
                cluster_id = None
                if prepared.cluster_set is None:
                    raise RuntimeError("cluster_set is required for candidate processing.")
                cluster_id = prepared.cluster_set.assign_cluster(np.asarray(sample["embedding"], dtype=np.float32))
                if group_id is None:
                    combo_idx = None
                elif uses_cluster_group_gap and cluster_id is not None:
                    combo_idx = int(cluster_id * prepared.demand.shape[1] + group_id)
                elif not uses_cluster_group_gap:
                    combo_idx = int(group_id)
                else:
                    combo_idx = None
                bandit_state.update_observation(source_idx, combo_idx)

                if group_value not in prepared.select_states:
                    skipped_not_minor += 1
                    total_cost += float(costs[source_idx])
                    evaluated += 1
                    continue

                if group_id is None or (uses_cluster_group_gap and cluster_id is None):
                    skipped_gap_zero += 1
                    total_cost += float(costs[source_idx])
                    evaluated += 1
                    continue

                demand_row = int(cluster_id) if uses_cluster_group_gap else 0
                if prepared.demand[demand_row, group_id] <= 0:
                    skipped_gap_zero += 1
                    total_cost += float(costs[source_idx])
                    evaluated += 1
                    continue

                candidates += 1
                source_candidates[source_idx] += 1
                bandit_state.mark_effective(source_idx)
                demand_decay = max(0.0, float(self.config.demand_decay_per_candidate))
                prepared.demand[demand_row, group_id] = max(
                    0.0,
                    prepared.demand[demand_row, group_id] - demand_decay,
                )
                prepared.q_flat[combo_idx] = prepared.demand[demand_row, group_id]

                if requires_prevaluation_cluster_gate:
                    if cluster_id is None:
                        skipped_gap_zero += 1
                        total_cost += float(costs[source_idx])
                        evaluated += 1
                        continue
                    gate_cluster_id = int(cluster_id)
                    if prepared.cluster_group_demand[gate_cluster_id, group_id] <= 0:
                        skipped_gap_zero += 1
                        total_cost += float(costs[source_idx])
                        evaluated += 1
                        continue
                    prepared.cluster_group_demand[gate_cluster_id, group_id] = max(
                        0.0,
                        prepared.cluster_group_demand[gate_cluster_id, group_id] - demand_decay,
                    )

                select_start = time.perf_counter()
                result = self._valuate_sample(
                    prepared.select_states[group_value],
                    np.asarray(sample["embedding"], dtype=np.float32),
                    cluster_set=prepared.cluster_set,
                    major_vectors=prepared.major_vectors,
                    major_norm_sq=prepared.major_norm_sq,
                    candidate_cluster=cluster_id,
                )
                select_s += time.perf_counter() - select_start

                if result.accept:
                    accepted += 1
                    total_gain += float(result.gain)
                    accepted_records.append(dict(sample))
                if result.num_candidates is not None:
                    cand_total += int(result.num_candidates)
                    cand_samples += 1
                if result.profile:
                    for key, value in result.profile.items():
                        select_profile_totals[key] = select_profile_totals.get(key, 0.0) + float(value)

                total_cost += float(costs[source_idx])
                evaluated += 1

                if self.config.use_select_feedback:
                    policy.on_observation(bandit_state, source_idx)

                while next_budget_snapshot_cost is not None and total_cost >= next_budget_snapshot_cost:
                    interval_selected = source_selected - last_snapshot_selected
                    interval_candidates = source_candidates - last_snapshot_candidates
                    self._emit_progress(
                        "budget_source_checkpoint",
                        evaluated=int(evaluated),
                        accepted=int(accepted),
                        total_cost=float(total_cost),
                        budget_checkpoint_cost=float(next_budget_snapshot_cost),
                        budget_fraction=float(next_budget_snapshot_cost / float(self.config.max_cost)),
                        interval_source_selected=self._nonzero_source_counts(
                            [source.name for source in source_manager.sources],
                            interval_selected,
                        ),
                        interval_source_candidates=self._nonzero_source_counts(
                            [source.name for source in source_manager.sources],
                            interval_candidates,
                        ),
                        cumulative_source_selected=self._nonzero_source_counts(
                            [source.name for source in source_manager.sources],
                            source_selected,
                        ),
                        cumulative_source_candidates=self._nonzero_source_counts(
                            [source.name for source in source_manager.sources],
                            source_candidates,
                        ),
                    )
                    last_snapshot_selected = source_selected.copy()
                    last_snapshot_candidates = source_candidates.copy()
                    next_budget_snapshot_cost += float(self.config.max_cost) * float(snapshot_fraction)

                if evaluated % self.progress_every == 0:
                    self._emit_progress(
                        "loop_checkpoint",
                        evaluated=int(evaluated),
                        sampled=int(sampled),
                        candidates=int(candidates),
                        accepted=int(accepted),
                        skipped_not_minor=int(skipped_not_minor),
                        skipped_gap_zero=int(skipped_gap_zero),
                        total_cost=float(total_cost),
                        select_s=float(select_s),
                        policy_s=float(policy_s),
                        loop_elapsed_s=float(time.perf_counter() - loop_start),
                    )
            else:
                stop_reason = "max_steps"
        finally:
            source_manager.close_all()
        self._emit_progress(
            "loop_done",
            stop_reason=stop_reason,
            evaluated=int(evaluated),
            sampled=int(sampled),
            candidates=int(candidates),
            accepted=int(accepted),
            skipped_not_minor=int(skipped_not_minor),
            skipped_gap_zero=int(skipped_gap_zero),
            total_cost=float(total_cost),
            select_s=float(select_s),
            policy_s=float(policy_s),
            loop_elapsed_s=float(time.perf_counter() - loop_start),
        )
        loop_s = time.perf_counter() - loop_start

        initial_eval = self._evaluate_dataset(prepared.query_df) if evaluate_outputs else None
        final_eval = None
        if return_final_eval and evaluate_outputs:
            self._emit_progress("final_eval_start", accepted_records=int(len(accepted_records)))
            frames = [prepared.query_df]
            if accepted_records:
                frames.append(pd.DataFrame(accepted_records))
            augmented_df = pd.concat(frames, ignore_index=True)
            final_eval = self._evaluate_dataset(augmented_df)
            self._emit_progress(
                "final_eval_done",
                rows=int(final_eval.rows),
                avg_sse=float(final_eval.avg_sse),
                balance_gap=float(final_eval.balance_gap),
                fairness_gap=float(final_eval.fairness_gap),
                maxsim_total=float(final_eval.maxsim_total),
            )
        else:
            self._emit_progress("final_eval_skipped")

        result = RAFRunResult(
            evaluated=evaluated,
            sampled=sampled,
            candidates=candidates,
            accepted=accepted,
            skipped_not_minor=skipped_not_minor,
            skipped_gap_zero=skipped_gap_zero,
            total_gain=float(total_gain),
            total_cost=float(total_cost),
            elapsed_s=float(loop_s),
            init_s=float(init_s),
            query_load_s=float(prepared.init_profile.get("query_load", 0.0)),
            cluster_s=float(prepared.init_profile.get("cluster_and_demand", 0.0)),
            select_state_build_s=float(prepared.init_profile.get("build_select_state", 0.0)),
            source_load_s=float(source_load_s),
            loop_s=float(loop_s),
            policy_s=float(policy_s),
            select_s=float(select_s),
            avg_internal_candidates=None if cand_samples <= 0 else float(cand_total / cand_samples),
            select_profile=dict(select_profile_totals) if select_profile_totals else None,
            valuation_mode=self.config.valuation_mode,
            source_selected=source_selected.tolist(),
            source_sampled=source_sampled.tolist(),
            source_minority_sampled=source_minority_sampled.tolist(),
            source_candidates=source_candidates.tolist(),
            source_names=[source.name for source in source_manager.sources],
            accepted_records=accepted_records,
            init_profile=prepared.init_profile,
            initial_eval=initial_eval,
            final_eval=final_eval,
        )
        self._emit_progress(
            "run_done",
            evaluated=int(result.evaluated),
            accepted=int(result.accepted),
            total_cost=float(result.total_cost),
            init_s=float(result.init_s),
            elapsed_s=float(result.elapsed_s),
            query_load_s=float(result.query_load_s),
            cluster_s=float(result.cluster_s),
            select_state_build_s=float(result.select_state_build_s),
            source_load_s=float(result.source_load_s),
            loop_s=float(result.loop_s),
        )
        return result
