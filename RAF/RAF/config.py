from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class SourceSpec:
    name: str
    path: Optional[str] = None
    cost: float = 1.0
    feature_col: str = "embedding"
    sensitive_col: str = "is_english_name"
    extra_cols: tuple[str, ...] = field(default_factory=tuple)
    sampler_mode: str = "random"
    with_replacement: bool = False
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceSpec":
        return cls(
            name=str(payload["name"]),
            path=payload.get("path"),
            cost=float(payload.get("cost", 1.0)),
            feature_col=str(payload.get("feature_col", "embedding")),
            sensitive_col=str(payload.get("sensitive_col", "is_english_name")),
            extra_cols=tuple(payload.get("extra_cols", ()) or ()),
            sampler_mode=str(payload.get("sampler_mode", "random")),
            with_replacement=bool(payload.get("with_replacement", False)),
            seed=payload.get("seed"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["extra_cols"] = list(self.extra_cols)
        return payload


@dataclass
class RAFConfig:
    query_path: Optional[str] = None
    sensitive_col: str = "is_english_name"
    feature_col: Optional[str] = None
    n_clusters: int = 20
    eval_n_clusters: int = 20
    eval_clustering_method: str = "kmeans"
    eval_subsample_size: Optional[int] = None
    eval_subsample_random_state: Optional[int] = None
    external_fair_algo_dir: Optional[str] = None
    external_fair_delta: float = 0.2
    external_fair_rounding: bool = True
    random_state: int = 42
    cluster_random_state: Optional[int] = None
    eval_random_state: Optional[int] = None
    cluster_subsample_ratio: Optional[float] = 0.1
    cluster_subsample_size: Optional[int] = None
    cluster_assign_batch_size: int = 4096
    policy: str = "fair_eps_greedy"
    valuation_mode: str = "incremental_hybrid"
    fair_kappa: float = 0.0
    fair_epsilon_p: float = 1e-6
    epsilon_alpha: float = 1.0
    epsilon_min: float = 0.1
    fair_alpha_eps: float = 1.0
    fair_eps_min: float = 0.1
    fair_enable_pruning: bool = False
    fair_prune_budget_interval: float = 0.1
    fair_prune_fraction: float = 0.1
    fair_min_active_sources: int = 1
    fair_prune_use_uniqueness: bool = True
    policy_warmup_rounds: int = 0
    metric: str = "euclidean"
    eps: float = 0.0
    batch_size: int = 4096
    hybrid_neighbor_k: int = 50
    hybrid_cluster_expand: int = 0
    hnsw_single_index: bool = False
    hnsw_delta_capacity: int = 4096
    hnsw_rebuild_delta_threshold: int = 2048
    hnsw_ef_search: int = 100
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    limit_sources: Optional[int] = 20
    max_steps: int = 200_000
    max_cost: Optional[float] = 20_000.0
    max_accepted: Optional[int] = None
    explore_budget_fraction: float = 0.1
    budget_snapshot_fraction: Optional[float] = None
    demand_decay_per_candidate: float = 0.2
    use_select_feedback: bool = True
    verbose: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "RAFConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
