from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .config import SourceSpec


Sample = dict[str, Any]
_PARQUET_RECORDS_CACHE: dict[tuple[str, str, str, tuple[str, ...]], list[Sample]] = {}


def normalize_record(
    sample: dict[str, Any],
    *,
    feature_col: str = "embedding",
    sensitive_col: str = "is_english_name",
) -> Sample:
    record = dict(sample)
    if feature_col in record and "embedding" not in record:
        record["embedding"] = record[feature_col]
    if feature_col in record and "features" not in record:
        record["features"] = record[feature_col]
    if "embedding" in record and "features" not in record:
        record["features"] = record["embedding"]
    if "features" in record and "embedding" not in record:
        record["embedding"] = record["features"]
    if sensitive_col in record and "is_english_name" not in record:
        record["is_english_name"] = record[sensitive_col]
    return record


class DataSource(abc.ABC):
    def __init__(self, name: str, cost: float) -> None:
        self.name = name
        self.cost = float(cost)
        self.exhausted = False
        self.num_samples_drawn = 0

    @abc.abstractmethod
    def sample_one(self) -> Optional[Sample]:
        raise NotImplementedError

    def reset(self) -> None:
        self.exhausted = False
        self.num_samples_drawn = 0

    def close(self) -> None:
        return


class InMemorySource(DataSource):
    def __init__(
        self,
        name: str,
        cost: float,
        records: Sequence[dict[str, Any]],
        *,
        sampler_mode: str = "random",
        with_replacement: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name, cost)
        self._records = [normalize_record(item) for item in records]
        self._sampler_mode = sampler_mode.strip().lower()
        self._with_replacement = with_replacement
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._remaining: list[int] = []
        self._cursor = 0
        self.reset()

    def sample_one(self) -> Optional[Sample]:
        if self.exhausted:
            return None
        if not self._records:
            self.exhausted = True
            return None

        if self._with_replacement:
            if self._sampler_mode == "random":
                idx = int(self._rng.integers(0, len(self._records)))
            else:
                idx = self._cursor % len(self._records)
                self._cursor += 1
        else:
            if not self._remaining:
                self.exhausted = True
                return None
            if self._sampler_mode == "random":
                # O(1) swap-with-last instead of O(n) pop from middle
                pick = int(self._rng.integers(0, len(self._remaining)))
                idx = self._remaining[pick]
                self._remaining[pick] = self._remaining[-1]
                self._remaining.pop()
            else:
                idx = self._remaining.pop(0)

        self.num_samples_drawn += 1
        return dict(self._records[idx])

    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self._seed)
        self._remaining = list(range(len(self._records)))
        if self._sampler_mode == "random" and not self._with_replacement:
            self._rng.shuffle(self._remaining)
        self._cursor = 0


class ParquetSource(InMemorySource):
    def __init__(self, spec: SourceSpec, *, use_cache: bool = True) -> None:
        records = load_parquet_source_records(spec, use_cache=use_cache)
        super().__init__(
            spec.name,
            spec.cost,
            records,
            sampler_mode=spec.sampler_mode,
            with_replacement=spec.with_replacement,
            seed=spec.seed,
        )
        self.path = spec.path


class SourceManager:
    def __init__(self, sources: Sequence[DataSource]) -> None:
        self.sources = list(sources)

    def __len__(self) -> int:
        return len(self.sources)

    def get(self, idx: int) -> DataSource:
        return self.sources[idx]

    def costs(self) -> np.ndarray:
        return np.asarray([source.cost for source in self.sources], dtype=float)

    def reset_all(self) -> None:
        for source in self.sources:
            source.reset()

    def close_all(self) -> None:
        for source in self.sources:
            source.close()


def _source_cache_key(spec: SourceSpec) -> tuple[str, str, str, tuple[str, ...]]:
    if not spec.path:
        raise ValueError(f"Source '{spec.name}' is missing a parquet path.")
    return (
        str(Path(spec.path).resolve()),
        str(spec.feature_col),
        str(spec.sensitive_col),
        tuple(spec.extra_cols),
    )


def load_parquet_source_records(spec: SourceSpec, *, use_cache: bool = True) -> list[Sample]:
    cache_key = _source_cache_key(spec)
    if use_cache and cache_key in _PARQUET_RECORDS_CACHE:
        return _PARQUET_RECORDS_CACHE[cache_key]

    try:
        df = pd.read_parquet(spec.path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read parquet source '{spec.path}'. Install a parquet engine such as pyarrow."
        ) from exc

    needed = {spec.feature_col, spec.sensitive_col, *spec.extra_cols}
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise KeyError(f"Source '{spec.name}' is missing columns: {missing}")

    records: list[Sample] = []
    for row in df[list(needed)].to_dict("records"):
        records.append(
            normalize_record(
                row,
                feature_col=spec.feature_col,
                sensitive_col=spec.sensitive_col,
            )
        )
    if use_cache:
        _PARQUET_RECORDS_CACHE[cache_key] = records
    return records


def clear_source_cache() -> None:
    _PARQUET_RECORDS_CACHE.clear()


def build_sources_from_specs(specs: Sequence[SourceSpec], *, use_cache: bool = True) -> SourceManager:
    return SourceManager([ParquetSource(spec, use_cache=use_cache) for spec in specs])


def load_source_specs_json(path: str | Path) -> list[SourceSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("sources", payload)
    if not isinstance(entries, list):
        raise ValueError("Source config JSON must be a list or {'sources': [...]} object.")
    return [SourceSpec.from_dict(item) for item in entries]


def build_source_specs_from_directory(
    directory: str | Path,
    *,
    pattern: str = "*.parquet",
    cost: float = 1.0,
    feature_col: str = "embedding",
    sensitive_col: str = "is_english_name",
    extra_cols: Optional[Sequence[str]] = None,
    sampler_mode: str = "random",
    with_replacement: bool = False,
) -> list[SourceSpec]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {root}")
    specs: list[SourceSpec] = []
    for idx, path in enumerate(sorted(root.glob(pattern)), start=1):
        specs.append(
            SourceSpec(
                name=f"source_{idx:02d}",
                path=str(path),
                cost=cost,
                feature_col=feature_col,
                sensitive_col=sensitive_col,
                extra_cols=tuple(extra_cols or ()),
                sampler_mode=sampler_mode,
                with_replacement=with_replacement,
            )
        )
    if not specs:
        raise FileNotFoundError(f"No files matched pattern '{pattern}' in {root}")
    return specs
