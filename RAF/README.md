# RAF

This repository contains a cleaned public-facing subset of the RAF project focused only on RAF runtime and experiment execution.

Included:

- `RAF/` core RAF runtime package
- `RAF/run_pipeline.py` generic single-run CLI
- `RAF/run_experiments.py` generic sweep / experiment CLI
- `source_config.example.json` example source manifest
- `requirements.txt` minimal runtime dependencies
- `pyproject.toml` package metadata and CLI entry points

Excluded on purpose:

- `query_builder/`
- `source_builder/`
- `tests/`
- `results/`
- dataset-specific PowerShell wrappers
- analysis / comparison / summarization scripts
- bundled third-party fair clustering repository

## Layout

```text
RAF/
  README.md
  pyproject.toml
  requirements.txt
  source_config.example.json
  RAF/
    __init__.py
    bandit.py
    clustering.py
    config.py
    data.py
    demand.py
    evaluation.py
    experiments.py
    fair_external_eval.py
    pipeline.py
    run_experiments.py
    run_pipeline.py
    selection.py
    vector_ops.py
```

## Install

```powershell
python -m pip install -r .\requirements.txt
```

or install as a package:

```powershell
python -m pip install -e .
```

`hnswlib` is only required for `incremental_hybrid`.

If you need `incremental_hybrid`, install the optional extra:

```powershell
python -m pip install -e ".[hnsw]"
```

## Run RAF Once

```powershell
python -m RAF.run_pipeline `
  --query_path D:\path\to\query.parquet `
  --source_dir D:\path\to\sources `
  --source_glob source_*.parquet `
  --policy fair_eps_greedy `
  --valuation_mode incremental_hybrid `
  --limit_sources 20
```

You can also pass `--source_config .\source_config.example.json` instead of `--source_dir`.

If installed with `pip install -e .`, you can also run:

```powershell
raf-run `
  --query_path D:\path\to\query.parquet `
  --source_dir D:\path\to\sources
```

## Run Experiments

```powershell
python -m RAF.run_experiments `
  --query_path D:\path\to\query.parquet `
  --source_dir D:\path\to\sources `
  --source_glob source_*.parquet `
  --policy fair_eps_greedy `
  --valuation_mode incremental_hybrid `
  --max_cost_start 6000 `
  --max_cost_end 20000 `
  --max_cost_step 2000 `
  --runs 5
```

If installed with `pip install -e .`, you can also run `raf-experiments`.

## Data Expectations

- Query and source inputs are parquet files.
- Embedding column defaults to `embedding`.
- Sensitive attribute column defaults to `is_english_name`.
- Source files must contain the embedding column, the sensitive column, and any requested extra columns.

## Optional External Fair Evaluation

`fair_external_eval.py` is kept for compatibility, but the bundled external solver repository is not included here.

If you use `--eval_clustering_method external_fair_relax_merge`, provide an external algorithm directory via `--external_fair_algo_dir` and install that solver's own dependencies separately.
