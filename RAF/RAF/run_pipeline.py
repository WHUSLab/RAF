from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    import select as _stdlib_select  # noqa: F401
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from RAF.bandit import normalize_policy_name
    from RAF.config import RAFConfig
    from RAF.data import build_source_specs_from_directory, load_source_specs_json
    from RAF.pipeline import RAFPipeline
else:
    from .bandit import normalize_policy_name
    from .config import RAFConfig
    from .data import build_source_specs_from_directory, load_source_specs_json
    from .pipeline import RAFPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RAF pipeline.")
    policy_choices = ["random", "epsilon_greedy", "fair_eps_greedy", "explore_exploit", "ratio"]
    parser.add_argument(
        "--query_path",
        type=str,
        default=None,
    )
    parser.add_argument("--source_config", type=str, default=None)
    parser.add_argument("--source_dir", type=str, default=None)
    parser.add_argument("--source_glob", type=str, default="*.parquet")
    parser.add_argument("--policy", type=str, default="fair_eps_greedy", choices=policy_choices)
    parser.add_argument("--valuation_mode", type=str, default="incremental_hybrid")
    parser.add_argument("--epsilon_alpha", type=float, default=1.0)
    parser.add_argument("--epsilon_min", type=float, default=0.1)
    parser.add_argument("--fair_kappa", type=float, default=0.0)
    parser.add_argument("--fair_epsilon_p", type=float, default=1e-6)
    parser.add_argument("--fair_alpha_eps", type=float, default=1.0)
    parser.add_argument("--fair_eps_min", type=float, default=0.1)
    parser.add_argument("--fair_enable_pruning", action="store_true")
    parser.add_argument("--fair_prune_budget_interval", type=float, default=0.1)
    parser.add_argument("--fair_prune_fraction", type=float, default=0.1)
    parser.add_argument("--fair_min_active_sources", type=int, default=1)
    parser.add_argument("--fair_disable_uniqueness_pruning", action="store_true")
    parser.add_argument("--policy_warmup_rounds", type=int, default=50)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--eval_n_clusters", type=int, default=20)
    parser.add_argument(
        "--eval_clustering_method",
        type=str,
        default="kmeans",
        choices=("kmeans", "external_fair_relax_merge"),
    )
    parser.add_argument("--eval_subsample_size", type=int, default=None)
    parser.add_argument("--eval_subsample_random_state", type=int, default=None)
    parser.add_argument("--external_fair_algo_dir", type=str, default=None)
    parser.add_argument("--external_fair_delta", type=float, default=0.2)
    parser.add_argument(
        "--external_fair_rounding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--hybrid_neighbor_k", type=int, default=50)
    parser.add_argument("--hybrid_cluster_expand", type=int, default=0)
    parser.add_argument("--hnsw_single_index", action="store_true")
    parser.add_argument("--hnsw_delta_capacity", type=int, default=4096)
    parser.add_argument("--hnsw_rebuild_delta_threshold", type=int, default=2048)
    parser.add_argument("--hnsw_ef_search", type=int, default=100)
    parser.add_argument("--hnsw_m", type=int, default=16)
    parser.add_argument("--hnsw_ef_construction", type=int, default=200)
    parser.add_argument("--limit_sources", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--max_cost", type=float, default=20000.0)
    parser.add_argument("--max_accepted", type=int, default=None)
    parser.add_argument("--explore_budget_fraction", type=float, default=0.1)
    parser.add_argument("--explore_ratio", dest="explore_budget_fraction", type=float, default=None)
    parser.add_argument("--explore_alpha", dest="explore_budget_fraction", type=float, default=None)
    parser.add_argument("--budget_snapshot_fraction", type=float, default=None)
    parser.add_argument("--demand_decay_per_candidate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster_random_state", type=int, default=42)
    parser.add_argument("--eval_random_state", type=int, default=42)
    parser.add_argument("--progress_every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_config and not args.source_dir:
        raise ValueError("Provide either --source_config or --source_dir.")
    config = RAFConfig(
        query_path=args.query_path,
        policy=normalize_policy_name(args.policy),
        valuation_mode=args.valuation_mode,
        epsilon_alpha=args.epsilon_alpha,
        epsilon_min=args.epsilon_min,
        fair_kappa=args.fair_kappa,
        fair_epsilon_p=args.fair_epsilon_p,
        fair_alpha_eps=args.fair_alpha_eps,
        fair_eps_min=args.fair_eps_min,
        fair_enable_pruning=args.fair_enable_pruning,
        fair_prune_budget_interval=args.fair_prune_budget_interval,
        fair_prune_fraction=args.fair_prune_fraction,
        fair_min_active_sources=args.fair_min_active_sources,
        fair_prune_use_uniqueness=not args.fair_disable_uniqueness_pruning,
        policy_warmup_rounds=args.policy_warmup_rounds,
        n_clusters=args.n_clusters,
        eval_n_clusters=args.eval_n_clusters,
        eval_clustering_method=args.eval_clustering_method,
        eval_subsample_size=args.eval_subsample_size,
        eval_subsample_random_state=args.eval_subsample_random_state,
        external_fair_algo_dir=args.external_fair_algo_dir,
        external_fair_delta=args.external_fair_delta,
        external_fair_rounding=args.external_fair_rounding,
        cluster_random_state=args.cluster_random_state,
        eval_random_state=args.eval_random_state,
        hybrid_neighbor_k=args.hybrid_neighbor_k,
        hybrid_cluster_expand=args.hybrid_cluster_expand,
        hnsw_single_index=args.hnsw_single_index,
        hnsw_delta_capacity=args.hnsw_delta_capacity,
        hnsw_rebuild_delta_threshold=args.hnsw_rebuild_delta_threshold,
        hnsw_ef_search=args.hnsw_ef_search,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        limit_sources=args.limit_sources,
        max_steps=args.max_steps,
        max_cost=args.max_cost,
        max_accepted=args.max_accepted,
        explore_budget_fraction=0.1 if args.explore_budget_fraction is None else args.explore_budget_fraction,
        budget_snapshot_fraction=args.budget_snapshot_fraction,
        demand_decay_per_candidate=args.demand_decay_per_candidate,
        random_state=args.seed,
    )
    if args.source_config:
        source_specs = load_source_specs_json(args.source_config)
    else:
        source_specs = build_source_specs_from_directory(
            args.source_dir,
            pattern=args.source_glob,
            extra_cols=("cluster_id", "source_id", "group_label"),
        )
    start_time = time.perf_counter()

    def progress_callback(event: dict[str, object]) -> None:
        elapsed_s = float(time.perf_counter() - start_time)
        event_name = str(event.get("event", "unknown"))
        print(f"[progress] elapsed_s={elapsed_s:.1f} stage={event_name} payload={event}", flush=True)

    result = RAFPipeline(
        config,
        source_specs=source_specs,
        progress_callback=progress_callback,
        progress_every=args.progress_every,
    ).run()
    print(
        "evaluated={evaluated} sampled={sampled} candidates={candidates} accepted={accepted} "
        "total_gain={total_gain:.6f} total_cost={total_cost:.3f} elapsed_s={elapsed_s:.3f}".format(
            **result.to_dict()
        )
    )
    print(f"valuation_mode={result.valuation_mode}")
    print(
        "skipped_not_minor={skipped_not_minor} skipped_gap_zero={skipped_gap_zero} "
        "init_s={init_s:.3f} policy_s={policy_s:.3f} select_s={select_s:.3f}".format(
            **result.to_dict()
        )
    )
    if result.avg_internal_candidates is not None:
        print(f"avg_internal_candidates={result.avg_internal_candidates:.3f}")
    if result.select_profile:
        profile_parts = " ".join(f"{key}={value:.6f}" for key, value in sorted(result.select_profile.items()))
        print(f"select_profile {profile_parts}")
    print(
        "timing query_load_s={query_load_s:.3f} cluster_s={cluster_s:.3f} "
        "select_state_build_s={select_state_build_s:.3f} source_load_s={source_load_s:.3f} "
        "loop_s={loop_s:.3f}".format(
            **result.to_dict()
        )
    )
    for name, sampled, minority_sampled in zip(
        result.source_names,
        result.source_sampled,
        result.source_minority_sampled,
    ):
        sampled = int(sampled)
        minority_sampled = int(minority_sampled)
        minority_ratio = float(minority_sampled / sampled) if sampled > 0 else 0.0
        print(
            f"source_samples source={name} sampled={sampled} "
            f"minority_sampled={minority_sampled} minority_ratio={minority_ratio:.4f}"
        )
    if result.final_eval is not None:
        initial_payload = result.to_dict()["initial_eval"]
        final_payload = result.to_dict()["final_eval"]
        print(
            "initial_eval rows={rows} avg_sse={avg_sse:.6f} avg_radius={avg_radius:.6f} max_radius={max_radius:.6f} balance_gap={balance_gap:.6f} fairness_gap={fairness_gap:.6f} clustering_method={clustering_method}".format(
                **initial_payload
            )
        )
        if initial_payload.get("clustering_objective") is not None:
            print(f"initial_eval_clustering_objective={float(initial_payload['clustering_objective']):.6f}")
        print(
            "final_eval rows={rows} avg_sse={avg_sse:.6f} avg_radius={avg_radius:.6f} max_radius={max_radius:.6f} balance_gap={balance_gap:.6f} fairness_gap={fairness_gap:.6f} maxsim_total={maxsim_total:.6f} clustering_method={clustering_method}".format(
                **final_payload
            )
        )
        if final_payload.get("clustering_objective") is not None:
            print(f"final_eval_clustering_objective={float(final_payload['clustering_objective']):.6f}")


if __name__ == "__main__":
    main()
