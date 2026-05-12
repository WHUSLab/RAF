from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    import select as _stdlib_select  # noqa: F401
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from RAF.bandit import normalize_policy_name
    from RAF.config import RAFConfig
    from RAF.data import build_source_specs_from_directory, load_source_specs_json
    from RAF.experiments import (
        run_parameter_sweep,
        run_policy_comparison,
        run_shared_candidate_valuation_comparison,
        save_rows_csv,
    )
else:
    from .bandit import normalize_policy_name
    from .config import RAFConfig
    from .data import build_source_specs_from_directory, load_source_specs_json
    from .experiments import (
        run_parameter_sweep,
        run_policy_comparison,
        run_shared_candidate_valuation_comparison,
        save_rows_csv,
    )


def _resolve_range(start: int | None, end: int | None, step: int | None) -> tuple[int, int, int] | None:
    if start is None and end is None and step is None:
        return None
    if start is None or end is None or step is None:
        raise ValueError("Sweep requires start, end, and step.")
    return int(start), int(end), int(step)


def _iter_sweep_values(start: int, end: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("Sweep step must be positive.")
    if end < start:
        raise ValueError("Sweep end must be >= start.")
    return list(range(int(start), int(end) + 1, int(step)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAF experiments.")
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
    parser.add_argument("--policy_warmup_rounds", type=int, default=0)
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
    parser.add_argument("--demand_decay_per_candidate", type=float, default=0.2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--cluster_random_state", type=int, default=42)
    parser.add_argument("--eval_random_state", type=int, default=42)
    parser.add_argument("--trim_ratio", type=float, default=0.0)
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--policies", type=str, default=None)
    parser.add_argument("--valuation_modes", type=str, default="baseline,incremental,incremental_hybrid")
    parser.add_argument("--shared_candidates_across_valuations", action="store_true")
    parser.add_argument("--estimate_baseline_runtime", action="store_true")
    parser.add_argument("--baseline_estimate_candidates", type=int, default=200)
    parser.add_argument("--skip_final_eval", action="store_true")
    parser.add_argument("--show_top_selected_source", action="store_true")
    parser.add_argument("--progress_every", type=int, default=1000)
    parser.add_argument("--max_cost_start", type=int, default=None)
    parser.add_argument("--max_cost_end", type=int, default=None)
    parser.add_argument("--max_cost_step", type=int, default=None)
    parser.add_argument("--limit_sources_start", type=int, default=None)
    parser.add_argument("--limit_sources_end", type=int, default=None)
    parser.add_argument("--limit_sources_step", type=int, default=None)
    parser.add_argument("--n_clusters_start", type=int, default=None)
    parser.add_argument("--n_clusters_end", type=int, default=None)
    parser.add_argument("--n_clusters_step", type=int, default=None)
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
        demand_decay_per_candidate=args.demand_decay_per_candidate,
        cluster_random_state=args.cluster_random_state,
        eval_random_state=args.eval_random_state,
        verbose=False,
    )
    if args.source_config:
        source_specs = load_source_specs_json(args.source_config)
    else:
        source_specs = build_source_specs_from_directory(
            args.source_dir,
            pattern=args.source_glob,
            extra_cols=("cluster_id", "source_id", "group_label"),
        )

    def progress_callback(event: dict[str, object]) -> None:
        event_name = str(event.get("event", "unknown"))
        if event_name == "sweep_run_start":
            print(
                "[progress] policy={policy} run={run_idx}/{runs} {param_name}={param_value} status=start elapsed_s={elapsed_s:.1f}".format(
                    **event
                ),
                flush=True,
            )
            return
        if event_name == "sweep_run_end":
            print(
                "[progress] policy={policy} run={run_idx}/{runs} {param_name}={param_value} status=done elapsed_s={elapsed_s:.1f} run_elapsed_s={run_elapsed_s:.1f} evaluated={evaluated:.0f} accepted={accepted:.0f} select_s={select_s:.3f}".format(
                    **event
                ),
                flush=True,
            )
            if args.show_top_selected_source:
                top_source_name = event.get("top_source_name")
                top_source_count = int(event.get("top_source_count", 0) or 0)
                if top_source_name and top_source_count > 0:
                    print(
                        f"[top_source] policy={event['policy']} run={event['run_idx']}/{event['runs']} "
                        f"{event['param_name']}={event['param_value']} source={top_source_name} count={top_source_count}",
                        flush=True,
                    )
            accepted_cluster_counts = event.get("accepted_cluster_counts")
            if accepted_cluster_counts:
                cluster_parts = ", ".join(
                    f"cluster_{key}={value}" for key, value in sorted(dict(accepted_cluster_counts).items())
                )
                print(
                    f"[accepted_clusters] policy={event['policy']} run={event['run_idx']}/{event['runs']} "
                    f"{event['param_name']}={event['param_value']} {cluster_parts}",
                    flush=True,
            )
            return
        if event_name == "loop_checkpoint":
            evaluated = event.get("evaluated", "?")
            accepted = event.get("accepted", "?")
            elapsed_s = event.get("elapsed_s")
            base = (
                f"[progress] policy={event.get('policy')} run={event.get('run_idx')}/{event.get('runs')} "
                f"{event.get('param_name')}={event.get('param_value')} status=checkpoint "
                f"evaluated={evaluated} accepted={accepted}"
            )
            if elapsed_s is not None:
                base += f" elapsed_s={float(elapsed_s):.1f}"
            print(base, flush=True)
            return
        if event_name.startswith(("query_", "prepare_", "source_manager_", "loop_", "final_eval")):
            extras = []
            for key in ("rows", "num_sources", "cached", "reason", "query_path"):
                if key in event:
                    extras.append(f"{key}={event[key]}")
            suffix = f" {' '.join(extras)}" if extras else ""
            print(
                f"[progress] policy={event.get('policy')} run={event.get('run_idx')}/{event.get('runs')} "
                f"{event.get('param_name')}={event.get('param_value')} stage={event_name}{suffix}",
                flush=True,
            )

    sweeps = {
        "max_cost": _resolve_range(args.max_cost_start, args.max_cost_end, args.max_cost_step),
        "limit_sources": _resolve_range(
            args.limit_sources_start,
            args.limit_sources_end,
            args.limit_sources_step,
        ),
        "n_clusters": _resolve_range(
            args.n_clusters_start,
            args.n_clusters_end,
            args.n_clusters_step,
        ),
    }

    rows: list[dict[str, object]] = []
    if args.shared_candidates_across_valuations:
        valuation_modes = [item.strip() for item in args.valuation_modes.split(",") if item.strip()]
        rows.extend(
            run_shared_candidate_valuation_comparison(
                config,
                valuation_modes=valuation_modes,
                runs=args.runs,
                seed_base=args.seed_base,
                trim_ratio=args.trim_ratio,
                source_specs=source_specs,
                return_final_eval=not args.skip_final_eval,
                estimate_baseline_runtime=args.estimate_baseline_runtime,
                baseline_estimate_candidates=args.baseline_estimate_candidates,
            )
        )
    elif args.policies:
        policies = [item.strip() for item in args.policies.split(",") if item.strip()]
        active_sweeps = [(param_name, value_range) for param_name, value_range in sweeps.items() if value_range is not None]
        if not active_sweeps:
            rows.extend(
                run_policy_comparison(
                    config,
                    policies=policies,
                    runs=args.runs,
                    seed_base=args.seed_base,
                    trim_ratio=args.trim_ratio,
                    source_specs=source_specs,
                    return_final_eval=not args.skip_final_eval,
                    progress_callback=progress_callback,
                    progress_every=args.progress_every,
                )
            )
        else:
            for param_name, value_range in active_sweeps:
                assert value_range is not None
                start, end, step = value_range
                for value in _iter_sweep_values(start, end, step):
                    sweep_config = RAFConfig(**config.to_dict())
                    setattr(sweep_config, param_name, value)
                    comparison_rows = run_policy_comparison(
                        sweep_config,
                        policies=policies,
                        runs=args.runs,
                        seed_base=args.seed_base,
                        trim_ratio=args.trim_ratio,
                        source_specs=source_specs,
                        return_final_eval=not args.skip_final_eval,
                        progress_callback=progress_callback,
                        progress_every=args.progress_every,
                    )
                    for row in comparison_rows:
                        row["param_name"] = param_name
                        row["value"] = value
                    rows.extend(comparison_rows)
    else:
        for param_name, value_range in sweeps.items():
            if value_range is None:
                continue
            start, end, step = value_range
            rows.extend(
                [
                    row.to_dict()
                    for row in run_parameter_sweep(
                        config,
                        param_name=param_name,
                        start=start,
                        end=end,
                        step=step,
                        runs=args.runs,
                        seed_base=args.seed_base,
                        trim_ratio=args.trim_ratio,
                        source_specs=source_specs,
                        return_final_eval=not args.skip_final_eval,
                        progress_callback=progress_callback,
                        progress_every=args.progress_every,
                    )
                ]
            )

    for row in rows:
        print(row)
    if args.out_csv:
        save_rows_csv(args.out_csv, rows)


if __name__ == "__main__":
    main()





