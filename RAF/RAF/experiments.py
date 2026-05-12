from __future__ import annotations

import csv
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from .bandit import normalize_policy_name
from .config import RAFConfig, SourceSpec
from .data import SourceManager
from .pipeline import CandidateCollection, RAFPipeline


@dataclass
class SweepResult:
    valuation_mode: str
    param_name: str
    value: Any
    runs: int
    evaluated: float
    candidates: float
    accepted: Optional[float]
    total_gain: Optional[float]
    total_cost: float
    elapsed_s: float
    init_s: float
    policy_s: float
    select_s: float
    avg_internal_candidates: Optional[float]
    select_profile: Optional[dict[str, float]]
    balance_gap: Optional[float]
    fairness_gap: Optional[float]
    avg_sse: Optional[float]
    avg_radius: Optional[float]
    max_radius: Optional[float]
    initial_balance_gap: Optional[float]
    initial_fairness_gap: Optional[float]
    initial_avg_sse: Optional[float]
    initial_avg_radius: Optional[float]
    initial_max_radius: Optional[float]
    initial_clustering_method: Optional[str]
    initial_clustering_objective: Optional[float]
    final_clustering_method: Optional[str]
    final_clustering_objective: Optional[float]
    maxsim_total: Optional[float]
    accepted_cluster_counts: Optional[dict[str, float]]
    timing_estimated: bool = False
    timing_estimate_sample_size: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _top_selected_source(result) -> tuple[Optional[str], int]:
    if not result.source_selected or not result.source_names:
        return None, 0
    counts = list(result.source_selected)
    if not counts:
        return None, 0
    top_idx = max(range(len(counts)), key=lambda idx: counts[idx])
    top_count = int(counts[top_idx])
    if top_count <= 0:
        return None, 0
    return str(result.source_names[top_idx]), top_count


def _accepted_cluster_counts(result) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in result.accepted_records or []:
        if "cluster_id" not in record:
            continue
        cluster_value = record.get("cluster_id")
        if cluster_value is None:
            continue
        try:
            cluster_key = str(int(cluster_value))
        except Exception:
            cluster_key = str(cluster_value)
        counter[cluster_key] += 1
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def _source_sample_profiles(result) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = list(result.source_names or [])
    sampled = list(result.source_sampled or [])
    minority_sampled = list(getattr(result, "source_minority_sampled", []) or [])
    total = max(len(names), len(sampled), len(minority_sampled))
    for idx in range(total):
        name = str(names[idx]) if idx < len(names) else f"source_{idx}"
        sampled_count = int(sampled[idx]) if idx < len(sampled) else 0
        minority_count = int(minority_sampled[idx]) if idx < len(minority_sampled) else 0
        minority_ratio = float(minority_count / sampled_count) if sampled_count > 0 else 0.0
        rows.append(
            {
                "source_name": name,
                "sampled": sampled_count,
                "minority_sampled": minority_count,
                "minority_ratio": minority_ratio,
            }
        )
    rows.sort(key=lambda row: (-int(row["sampled"]), str(row["source_name"])))
    return rows


def _mean_dict_counts(values: list[dict[str, int]], trim_ratio: float) -> Optional[dict[str, float]]:
    present = [value for value in values if value]
    if not present:
        return None
    all_keys = sorted({key for value in present for key in value.keys()}, key=lambda item: (len(item), item))
    result: dict[str, float] = {}
    for key in all_keys:
        key_values = [float(value.get(key, 0.0)) for value in values]
        result[key] = _trimmed_mean(key_values, trim_ratio)
    return result


def _mean_dict_floats(values: list[dict[str, float]], trim_ratio: float) -> Optional[dict[str, float]]:
    present = [value for value in values if value]
    if not present:
        return None
    all_keys = sorted({key for value in present for key in value.keys()})
    result: dict[str, float] = {}
    for key in all_keys:
        key_values = [float(value.get(key, 0.0)) for value in values]
        result[key] = _trimmed_mean(key_values, trim_ratio)
    return result


def _trim(values: list[float], trim_ratio: float) -> list[float]:
    if trim_ratio <= 0 or len(values) < 3:
        return list(values)
    cut = int(len(values) * trim_ratio)
    if cut <= 0 or cut * 2 >= len(values):
        return list(values)
    return sorted(values)[cut : len(values) - cut]


def _trimmed_mean(values: list[float], trim_ratio: float) -> float:
    trimmed = _trim(values, trim_ratio)
    if not trimmed:
        return 0.0
    return float(sum(trimmed) / len(trimmed))


def _trimmed_mean_optional(values: list[Optional[float]], trim_ratio: float) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return _trimmed_mean(present, trim_ratio)


def _scale_profile(profile: Optional[dict[str, float]], scale: float) -> Optional[dict[str, float]]:
    if not profile:
        return None
    return {str(key): float(value) * float(scale) for key, value in profile.items()}


def run_parameter_sweep(
    base_config: RAFConfig,
    *,
    param_name: str,
    start: int,
    end: int,
    step: int,
    runs: int,
    seed_base: int = 42,
    trim_ratio: float = 0.0,
    query_df: Optional[pd.DataFrame] = None,
    source_specs: Optional[Sequence[SourceSpec]] = None,
    source_manager_factory: Optional[Callable[[], SourceManager]] = None,
    return_final_eval: bool = True,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    progress_every: int = 1000,
) -> list[SweepResult]:
    if not hasattr(base_config, param_name):
        raise AttributeError(f"RAFConfig has no field named '{param_name}'.")
    rows: list[SweepResult] = []
    sweep_start = time.perf_counter()
    for value_idx, value in enumerate(range(start, end + 1, step), start=1):
        metrics: dict[str, list[float]] = {
            "evaluated": [],
            "candidates": [],
            "accepted": [],
            "total_gain": [],
            "total_cost": [],
            "elapsed_s": [],
            "init_s": [],
            "policy_s": [],
            "select_s": [],
            "avg_internal_candidates": [],
            "balance_gap": [],
            "fairness_gap": [],
            "avg_sse": [],
            "avg_radius": [],
            "max_radius": [],
            "initial_balance_gap": [],
            "initial_fairness_gap": [],
            "initial_avg_sse": [],
            "initial_avg_radius": [],
            "initial_max_radius": [],
            "initial_clustering_method": [],
            "initial_clustering_objective": [],
            "final_clustering_method": [],
            "final_clustering_objective": [],
            "maxsim_total": [],
        }
        accepted_cluster_counts_runs: list[dict[str, int]] = []
        select_profile_runs: list[dict[str, float]] = []
        for run_idx in range(runs):
            run_number = run_idx + 1
            def _emit(payload: dict[str, Any]) -> None:
                if progress_callback is None:
                    return
                enriched = {
                    "policy": normalize_policy_name(base_config.policy),
                    "valuation_mode": base_config.valuation_mode,
                    "param_name": param_name,
                    "param_value": value,
                    "run_idx": run_number,
                    "runs": runs,
                    "sweep_idx": value_idx,
                }
                enriched.update(payload)
                progress_callback(enriched)

            if progress_callback is not None:
                _emit(
                    {
                        "event": "sweep_run_start",
                        "elapsed_s": float(time.perf_counter() - sweep_start),
                    }
                )
            run_config = replace(base_config, random_state=seed_base + run_idx)
            setattr(run_config, param_name, value)
            manager = source_manager_factory() if source_manager_factory is not None else None
            run_start = time.perf_counter()
            result = RAFPipeline(
                run_config,
                source_specs=list(source_specs) if source_specs is not None else None,
                progress_callback=_emit if progress_callback is not None else None,
                progress_every=progress_every,
            ).run(
                query_df=query_df,
                source_manager=manager,
                return_final_eval=return_final_eval,
            )
            if progress_callback is not None:
                top_source_name, top_source_count = _top_selected_source(result)
                accepted_cluster_counts = _accepted_cluster_counts(result)
                _emit(
                    {
                        "event": "sweep_run_end",
                        "elapsed_s": float(time.perf_counter() - sweep_start),
                        "run_elapsed_s": float(time.perf_counter() - run_start),
                        "evaluated": float(result.evaluated),
                        "accepted": float(result.accepted),
                        "select_s": float(result.select_s),
                        "top_source_name": top_source_name,
                        "top_source_count": int(top_source_count),
                        "accepted_cluster_counts": accepted_cluster_counts,
                        "source_sample_profiles": _source_sample_profiles(result),
                    }
                )
            metrics["evaluated"].append(float(result.evaluated))
            metrics["candidates"].append(float(result.candidates))
            metrics["accepted"].append(float(result.accepted))
            metrics["total_gain"].append(float(result.total_gain))
            metrics["total_cost"].append(float(result.total_cost))
            metrics["elapsed_s"].append(float(result.elapsed_s))
            metrics["init_s"].append(float(result.init_s))
            metrics["policy_s"].append(float(result.policy_s))
            metrics["select_s"].append(float(result.select_s))
            metrics["avg_internal_candidates"].append(
                None if result.avg_internal_candidates is None else float(result.avg_internal_candidates)
            )
            metrics["fairness_gap"].append(
                None if result.final_eval is None else float(result.final_eval.fairness_gap)
            )
            metrics["balance_gap"].append(
                None if result.final_eval is None else float(result.final_eval.balance_gap)
            )
            metrics["avg_sse"].append(None if result.final_eval is None else float(result.final_eval.avg_sse))
            metrics["avg_radius"].append(
                None if result.final_eval is None else float(result.final_eval.avg_radius)
            )
            metrics["max_radius"].append(
                None if result.final_eval is None else float(result.final_eval.max_radius)
            )
            metrics["initial_balance_gap"].append(
                None if result.initial_eval is None else float(result.initial_eval.balance_gap)
            )
            metrics["initial_fairness_gap"].append(
                None if result.initial_eval is None else float(result.initial_eval.fairness_gap)
            )
            metrics["initial_avg_sse"].append(
                None if result.initial_eval is None else float(result.initial_eval.avg_sse)
            )
            metrics["initial_avg_radius"].append(
                None if result.initial_eval is None else float(result.initial_eval.avg_radius)
            )
            metrics["initial_max_radius"].append(
                None if result.initial_eval is None else float(result.initial_eval.max_radius)
            )
            metrics["initial_clustering_method"].append(
                None if result.initial_eval is None else str(result.initial_eval.clustering_method)
            )
            metrics["initial_clustering_objective"].append(
                None
                if result.initial_eval is None or result.initial_eval.clustering_objective is None
                else float(result.initial_eval.clustering_objective)
            )
            metrics["final_clustering_method"].append(
                None if result.final_eval is None else str(result.final_eval.clustering_method)
            )
            metrics["final_clustering_objective"].append(
                None
                if result.final_eval is None or result.final_eval.clustering_objective is None
                else float(result.final_eval.clustering_objective)
            )
            metrics["maxsim_total"].append(
                None if result.final_eval is None else float(result.final_eval.maxsim_total)
            )
            accepted_cluster_counts_runs.append(_accepted_cluster_counts(result))
            if result.select_profile:
                select_profile_runs.append(dict(result.select_profile))

        rows.append(
            SweepResult(
                valuation_mode=base_config.valuation_mode,
                param_name=param_name,
                value=value,
                runs=runs,
                evaluated=_trimmed_mean(metrics["evaluated"], trim_ratio),
                candidates=_trimmed_mean(metrics["candidates"], trim_ratio),
                accepted=_trimmed_mean(metrics["accepted"], trim_ratio),
                total_gain=_trimmed_mean(metrics["total_gain"], trim_ratio),
                total_cost=_trimmed_mean(metrics["total_cost"], trim_ratio),
                elapsed_s=_trimmed_mean(metrics["elapsed_s"], trim_ratio),
                init_s=_trimmed_mean(metrics["init_s"], trim_ratio),
                policy_s=_trimmed_mean(metrics["policy_s"], trim_ratio),
                select_s=_trimmed_mean(metrics["select_s"], trim_ratio),
                avg_internal_candidates=_trimmed_mean_optional(metrics["avg_internal_candidates"], trim_ratio),
                select_profile=_mean_dict_floats(select_profile_runs, trim_ratio),
                balance_gap=_trimmed_mean_optional(metrics["balance_gap"], trim_ratio),
                fairness_gap=_trimmed_mean_optional(metrics["fairness_gap"], trim_ratio),
                avg_sse=_trimmed_mean_optional(metrics["avg_sse"], trim_ratio),
                avg_radius=_trimmed_mean_optional(metrics["avg_radius"], trim_ratio),
                max_radius=_trimmed_mean_optional(metrics["max_radius"], trim_ratio),
                initial_balance_gap=_trimmed_mean_optional(metrics["initial_balance_gap"], trim_ratio),
                initial_fairness_gap=_trimmed_mean_optional(metrics["initial_fairness_gap"], trim_ratio),
                initial_avg_sse=_trimmed_mean_optional(metrics["initial_avg_sse"], trim_ratio),
                initial_avg_radius=_trimmed_mean_optional(metrics["initial_avg_radius"], trim_ratio),
                initial_max_radius=_trimmed_mean_optional(metrics["initial_max_radius"], trim_ratio),
                initial_clustering_method=next(
                    (str(value) for value in metrics["initial_clustering_method"] if value is not None),
                    None,
                ),
                initial_clustering_objective=_trimmed_mean_optional(
                    metrics["initial_clustering_objective"], trim_ratio
                ),
                final_clustering_method=next(
                    (str(value) for value in metrics["final_clustering_method"] if value is not None),
                    None,
                ),
                final_clustering_objective=_trimmed_mean_optional(
                    metrics["final_clustering_objective"], trim_ratio
                ),
                maxsim_total=_trimmed_mean_optional(metrics["maxsim_total"], trim_ratio),
                accepted_cluster_counts=_mean_dict_counts(accepted_cluster_counts_runs, trim_ratio),
            )
        )
    return rows


def run_policy_comparison(
    base_config: RAFConfig,
    *,
    policies: Sequence[str],
    runs: int,
    seed_base: int = 42,
    trim_ratio: float = 0.0,
    query_df: Optional[pd.DataFrame] = None,
    source_specs: Optional[Sequence[SourceSpec]] = None,
    source_manager_factory: Optional[Callable[[], SourceManager]] = None,
    return_final_eval: bool = True,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    progress_every: int = 1000,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fixed_budget = int(base_config.max_cost or 0)
    for policy in policies:
        policy_rows = run_parameter_sweep(
            replace(base_config, policy=normalize_policy_name(policy)),
            param_name="max_cost",
            start=fixed_budget,
            end=fixed_budget,
            step=1,
            runs=runs,
            seed_base=seed_base,
            trim_ratio=trim_ratio,
            query_df=query_df,
            source_specs=source_specs,
            source_manager_factory=source_manager_factory,
            return_final_eval=return_final_eval,
            progress_callback=progress_callback,
            progress_every=progress_every,
        )
        row = policy_rows[0].to_dict()
        row["policy"] = normalize_policy_name(policy)
        rows.append(row)
    return rows


def run_shared_candidate_valuation_comparison(
    base_config: RAFConfig,
    *,
    valuation_modes: Sequence[str],
    runs: int,
    seed_base: int = 42,
    trim_ratio: float = 0.0,
    query_df: Optional[pd.DataFrame] = None,
    source_specs: Optional[Sequence[SourceSpec]] = None,
    source_manager_factory: Optional[Callable[[], SourceManager]] = None,
    return_final_eval: bool = True,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    estimate_baseline_runtime: bool = False,
    baseline_estimate_candidates: int = 200,
) -> list[dict[str, Any]]:
    normalized_modes = [str(mode).strip().lower() for mode in valuation_modes if str(mode).strip()]
    if not normalized_modes:
        raise ValueError("valuation_modes must contain at least one mode.")

    metrics_by_mode: dict[str, dict[str, list[float]]] = {
        mode: {
            "evaluated": [],
            "candidates": [],
            "accepted": [],
            "total_gain": [],
            "total_cost": [],
            "elapsed_s": [],
            "init_s": [],
            "policy_s": [],
            "select_s": [],
            "avg_internal_candidates": [],
            "balance_gap": [],
            "fairness_gap": [],
            "avg_sse": [],
            "avg_radius": [],
            "max_radius": [],
            "initial_balance_gap": [],
            "initial_fairness_gap": [],
            "initial_avg_sse": [],
            "initial_avg_radius": [],
            "initial_max_radius": [],
            "initial_clustering_method": [],
            "initial_clustering_objective": [],
            "final_clustering_method": [],
            "final_clustering_objective": [],
            "maxsim_total": [],
        }
        for mode in normalized_modes
    }
    accepted_cluster_counts_by_mode: dict[str, list[dict[str, int]]] = {mode: [] for mode in normalized_modes}
    select_profiles_by_mode: dict[str, list[dict[str, float]]] = {mode: [] for mode in normalized_modes}

    for run_idx in range(runs):
        run_number = run_idx + 1
        candidate_config = replace(base_config, random_state=seed_base + run_idx)
        manager = source_manager_factory() if source_manager_factory is not None else None
        collector = RAFPipeline(
            candidate_config,
            source_specs=list(source_specs) if source_specs is not None else None,
        )
        candidate_collection: CandidateCollection = collector.collect_candidates(
            query_df=query_df,
            source_manager=manager,
        )

        for mode in normalized_modes:
            replay_config = replace(candidate_config, valuation_mode=mode)
            timing_estimated = False
            timing_estimate_sample_size: Optional[int] = None
            if mode == "baseline" and estimate_baseline_runtime:
                sample_size = min(
                    max(0, int(baseline_estimate_candidates)),
                    len(candidate_collection.candidate_records),
                )
                timing_estimated = True
                timing_estimate_sample_size = sample_size
                if sample_size > 0:
                    sample_result = RAFPipeline(
                        replay_config,
                        source_specs=list(source_specs) if source_specs is not None else None,
                    ).replay_candidates(
                        candidate_collection.candidate_records[:sample_size],
                        query_df=query_df,
                        return_final_eval=False,
                    )
                    scale = float(candidate_collection.candidates) / float(sample_size)
                    estimated_select_s = float(sample_result.select_s) * scale
                    estimated_elapsed_s = float(sample_result.elapsed_s) * scale
                    result = sample_result
                    result = replace(
                        result,
                        evaluated=int(candidate_collection.evaluated),
                        sampled=int(candidate_collection.sampled),
                        candidates=int(candidate_collection.candidates),
                        accepted=0,
                        total_gain=0.0,
                        total_cost=float(candidate_collection.total_cost),
                        elapsed_s=float(estimated_elapsed_s),
                        policy_s=0.0,
                        select_s=float(estimated_select_s),
                        avg_internal_candidates=sample_result.avg_internal_candidates,
                        select_profile=_scale_profile(sample_result.select_profile, scale),
                        source_selected=list(candidate_collection.source_selected),
                        source_sampled=list(candidate_collection.source_sampled),
                        source_minority_sampled=list(candidate_collection.source_minority_sampled),
                        source_candidates=list(candidate_collection.source_candidates),
                        source_names=list(candidate_collection.source_names),
                        accepted_records=[],
                        final_eval=None,
                    )
                else:
                    result = RAFPipeline(
                        replay_config,
                        source_specs=list(source_specs) if source_specs is not None else None,
                    ).replay_candidates(
                        [],
                        query_df=query_df,
                        return_final_eval=False,
                    )
                    result = replace(
                        result,
                        evaluated=int(candidate_collection.evaluated),
                        sampled=int(candidate_collection.sampled),
                        candidates=int(candidate_collection.candidates),
                        accepted=0,
                        total_gain=0.0,
                        total_cost=float(candidate_collection.total_cost),
                        elapsed_s=0.0,
                        policy_s=0.0,
                        select_s=0.0,
                        avg_internal_candidates=None,
                        select_profile=None,
                        source_selected=list(candidate_collection.source_selected),
                        source_sampled=list(candidate_collection.source_sampled),
                        source_minority_sampled=list(candidate_collection.source_minority_sampled),
                        source_candidates=list(candidate_collection.source_candidates),
                        source_names=list(candidate_collection.source_names),
                        accepted_records=[],
                        final_eval=None,
                    )
            else:
                result = RAFPipeline(
                    replay_config,
                    source_specs=list(source_specs) if source_specs is not None else None,
                ).replay_candidates(
                    candidate_collection,
                    query_df=query_df,
                    return_final_eval=return_final_eval,
                )
            metrics = metrics_by_mode[mode]
            metrics["evaluated"].append(float(result.evaluated))
            metrics["candidates"].append(float(result.candidates))
            metrics["accepted"].append(None if timing_estimated else float(result.accepted))
            metrics["total_gain"].append(None if timing_estimated else float(result.total_gain))
            metrics["total_cost"].append(float(result.total_cost))
            metrics["elapsed_s"].append(float(result.elapsed_s))
            metrics["init_s"].append(float(result.init_s))
            metrics["policy_s"].append(float(result.policy_s))
            metrics["select_s"].append(float(result.select_s))
            metrics["avg_internal_candidates"].append(
                None if result.avg_internal_candidates is None else float(result.avg_internal_candidates)
            )
            metrics["fairness_gap"].append(
                None if timing_estimated or result.final_eval is None else float(result.final_eval.fairness_gap)
            )
            metrics["balance_gap"].append(
                None if timing_estimated or result.final_eval is None else float(result.final_eval.balance_gap)
            )
            metrics["avg_sse"].append(
                None if timing_estimated or result.final_eval is None else float(result.final_eval.avg_sse)
            )
            metrics["avg_radius"].append(
                None if timing_estimated or result.final_eval is None else float(result.final_eval.avg_radius)
            )
            metrics["max_radius"].append(
                None if timing_estimated or result.final_eval is None else float(result.final_eval.max_radius)
            )
            metrics["initial_balance_gap"].append(
                None if result.initial_eval is None else float(result.initial_eval.balance_gap)
            )
            metrics["initial_fairness_gap"].append(
                None if result.initial_eval is None else float(result.initial_eval.fairness_gap)
            )
            metrics["initial_avg_sse"].append(
                None if result.initial_eval is None else float(result.initial_eval.avg_sse)
            )
            metrics["initial_avg_radius"].append(
                None if result.initial_eval is None else float(result.initial_eval.avg_radius)
            )
            metrics["initial_max_radius"].append(
                None if result.initial_eval is None else float(result.initial_eval.max_radius)
            )
            metrics["initial_clustering_method"].append(
                None if result.initial_eval is None else str(result.initial_eval.clustering_method)
            )
            metrics["initial_clustering_objective"].append(
                None
                if result.initial_eval is None or result.initial_eval.clustering_objective is None
                else float(result.initial_eval.clustering_objective)
            )
            metrics["final_clustering_method"].append(
                None if timing_estimated or result.final_eval is None else str(result.final_eval.clustering_method)
            )
            metrics["final_clustering_objective"].append(
                None
                if timing_estimated or result.final_eval is None or result.final_eval.clustering_objective is None
                else float(result.final_eval.clustering_objective)
            )
            metrics["maxsim_total"].append(
                None if timing_estimated or result.final_eval is None else float(result.final_eval.maxsim_total)
            )
            accepted_cluster_counts_by_mode[mode].append({} if timing_estimated else _accepted_cluster_counts(result))
            if result.select_profile:
                select_profiles_by_mode[mode].append(dict(result.select_profile))
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "shared_valuation_run_end",
                        "policy": normalize_policy_name(base_config.policy),
                        "valuation_mode": mode,
                        "run_idx": run_number,
                        "runs": runs,
                        "evaluated": float(result.evaluated),
                        "candidates": float(result.candidates),
                        "accepted": None if timing_estimated else float(result.accepted),
                        "select_s": float(result.select_s),
                        "timing_estimated": bool(timing_estimated),
                        "timing_estimate_sample_size": timing_estimate_sample_size,
                    }
                )

    rows: list[dict[str, Any]] = []
    for mode in normalized_modes:
        metrics = metrics_by_mode[mode]
        timing_estimated = bool(mode == "baseline" and estimate_baseline_runtime)
        row = SweepResult(
            valuation_mode=mode,
            param_name="valuation_mode",
            value=mode,
            runs=runs,
            evaluated=_trimmed_mean(metrics["evaluated"], trim_ratio),
            candidates=_trimmed_mean(metrics["candidates"], trim_ratio),
            accepted=_trimmed_mean_optional(metrics["accepted"], trim_ratio),
            total_gain=_trimmed_mean_optional(metrics["total_gain"], trim_ratio),
            total_cost=_trimmed_mean(metrics["total_cost"], trim_ratio),
            elapsed_s=_trimmed_mean(metrics["elapsed_s"], trim_ratio),
            init_s=_trimmed_mean(metrics["init_s"], trim_ratio),
            policy_s=_trimmed_mean(metrics["policy_s"], trim_ratio),
            select_s=_trimmed_mean(metrics["select_s"], trim_ratio),
            avg_internal_candidates=_trimmed_mean_optional(metrics["avg_internal_candidates"], trim_ratio),
            select_profile=_mean_dict_floats(select_profiles_by_mode[mode], trim_ratio),
            balance_gap=_trimmed_mean_optional(metrics["balance_gap"], trim_ratio),
            fairness_gap=_trimmed_mean_optional(metrics["fairness_gap"], trim_ratio),
            avg_sse=_trimmed_mean_optional(metrics["avg_sse"], trim_ratio),
            avg_radius=_trimmed_mean_optional(metrics["avg_radius"], trim_ratio),
            max_radius=_trimmed_mean_optional(metrics["max_radius"], trim_ratio),
            initial_balance_gap=_trimmed_mean_optional(metrics["initial_balance_gap"], trim_ratio),
            initial_fairness_gap=_trimmed_mean_optional(metrics["initial_fairness_gap"], trim_ratio),
            initial_avg_sse=_trimmed_mean_optional(metrics["initial_avg_sse"], trim_ratio),
            initial_avg_radius=_trimmed_mean_optional(metrics["initial_avg_radius"], trim_ratio),
            initial_max_radius=_trimmed_mean_optional(metrics["initial_max_radius"], trim_ratio),
            initial_clustering_method=next(
                (str(value) for value in metrics["initial_clustering_method"] if value is not None),
                None,
            ),
            initial_clustering_objective=_trimmed_mean_optional(
                metrics["initial_clustering_objective"], trim_ratio
            ),
            final_clustering_method=next(
                (str(value) for value in metrics["final_clustering_method"] if value is not None),
                None,
            ),
            final_clustering_objective=_trimmed_mean_optional(
                metrics["final_clustering_objective"], trim_ratio
            ),
            maxsim_total=_trimmed_mean_optional(metrics["maxsim_total"], trim_ratio),
            accepted_cluster_counts=_mean_dict_counts(accepted_cluster_counts_by_mode[mode], trim_ratio),
            timing_estimated=timing_estimated,
            timing_estimate_sample_size=(
                None if not timing_estimated else min(max(0, int(baseline_estimate_candidates)), int(_trimmed_mean(metrics["candidates"], trim_ratio)))
            ),
        ).to_dict()
        row["policy"] = normalize_policy_name(base_config.policy)
        row["candidate_policy"] = normalize_policy_name(base_config.policy)
        rows.append(row)
    return rows


def save_rows_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
