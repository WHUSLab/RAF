# RAF: Retrieval-Augmented Dataset Assembly for Fair Clustering

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#installation)
[![License](https://img.shields.io/badge/License-TBD-lightgrey.svg)](#license)
[![Status](https://img.shields.io/badge/Status-Research%20Artifact-orange.svg)](#repository-status)

RAF is a retrieval-augmented dataset assembly framework for downstream fair clustering. Instead of treating fairness only as a constraint imposed on a fixed input dataset, RAF improves the data foundation before clustering by selectively acquiring real samples from external data sources.

The key idea is simple: when a minority group is under-represented in some semantic regions of the data space, fair clustering algorithms may have too few relevant samples to produce both useful and fair clusters. RAF addresses this limitation by retrieving and accepting external samples that are not only from under-represented groups, but also semantically aligned with the majority-group distribution of the target dataset.

This repository contains the public-facing RAF runtime and experiment execution code.

---

## Table of Contents

- [Why RAF?](#why-raf)
- [Method Overview](#method-overview)
- [Repository Status](#repository-status)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Data](#data)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [External Fair-Clustering Evaluation](#external-fair-clustering-evaluation)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)
- [License](#license)

---

## Why RAF?

Fair clustering aims to partition data while reducing group-level disparities with respect to a sensitive attribute. Existing methods usually operate on a fixed dataset. This setting is limiting when the dataset itself is imbalanced: if minority samples are missing in some regions of the embedding space, a clustering algorithm can only rebalance the samples that already exist. In such cases, fairness constraints may force semantically misaligned assignments and degrade clustering utility.

RAF takes a data-centric view. It asks a different question:

> Given an initial dataset, a set of external data sources, and a limited acquisition budget, which external samples should be assembled to improve downstream fair clustering?

RAF is designed for this retrieval-augmented setting. It first identifies where the initial dataset lacks minority-group support, then allocates budget to promising external sources, and finally values each retrieved candidate by its marginal contribution to cross-group distributional alignment.

---

## Method Overview

RAF follows a two-stage selection--valuation workflow.

### 1. Demand-aware source selection

RAF first pre-clusters the query dataset and builds a demand matrix over cluster--group combinations. Each entry reflects how many additional samples are needed for a sensitive group in a specific semantic region. This demand signal is used to guide a multi-armed-bandit source selection policy, so that RAF spends more budget on sources that are more likely to return useful candidates.

### 2. MaxSim-based candidate valuation

After a candidate is retrieved, RAF estimates whether the sample reduces the distributional gap between majority and minority groups in the embedding space. RAF uses a MaxSim-style vector-set similarity objective to measure the marginal contribution of each candidate. A sample is accepted only when it improves the current majority--minority alignment.

### 3. Efficient incremental valuation

Naively recomputing the full MaxSim gain for every candidate is expensive. RAF therefore supports incremental valuation and an optional hybrid-index mode. The hybrid mode combines nearest-neighbor search with inverted lists to reduce the number of majority samples that must be rechecked for each candidate.

The recommended runtime setting is:

```text
policy         = fair_eps_greedy
valuation_mode = incremental_hybrid
```

---

## Repository Status

This repository is a cleaned public-facing subset of the RAF project. It focuses on runtime execution and reproducible experiment sweeps.

Included:

- `RAF/`: core RAF runtime package
- `RAF/run_pipeline.py`: generic single-run CLI
- `RAF/run_experiments.py`: generic sweep / experiment CLI
- `source_config.example.json`: example source manifest
- `requirements.txt`: minimal runtime dependencies
- `pyproject.toml`: package metadata and CLI entry points

Excluded on purpose:

- dataset construction scripts
- query/source builders
- tests
- result folders
- dataset-specific PowerShell wrappers
- analysis, comparison, and summarization scripts
- bundled third-party fair-clustering repositories

The public release assumes that RAF-ready query and source files have already been prepared.

---

## Repository Layout

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

---

## Installation

### Option 1: Install runtime dependencies

```bash
python -m pip install -r requirements.txt
```

### Option 2: Install RAF as an editable package

```bash
python -m pip install -e .
```

### Optional: install HNSW support

`hnswlib` is only required when using `incremental_hybrid` valuation.

```bash
python -m pip install -e ".[hnsw]"
```

On Windows PowerShell, use:

```powershell
python -m pip install -e ".[hnsw]"
```

---

## Data

### Input format

RAF expects preprocessed query and source files in Parquet format.

| Column | Default name | Description |
| --- | --- | --- |
| Embedding | `embedding` | Vector representation of each sample. |
| Sensitive attribute | `is_english_name` | Sensitive group label used for fair dataset assembly. |
| Extra columns | user-defined | Optional metadata columns preserved for analysis or evaluation. |

Each source file must contain at least the embedding column and the sensitive attribute column.

### Passing source files

RAF supports two source-loading modes.

#### Mode A: source directory

```bash
--source_dir /path/to/sources \
--source_glob "source_*.parquet"
```

#### Mode B: source manifest

```bash
--source_config source_config.example.json
```

Use the provided `source_config.example.json` as the template for declaring source files and source-level metadata.

### Original datasets

The experiments in the RAF paper are designed around multi-modal datasets. Raw datasets are not bundled in this repository. Please follow the corresponding dataset licenses and access policies.

| Dataset | Modality | Original source |
| --- | --- | --- |
| SciSciNet-v2 | scientific publication metadata / text-derived embeddings | https://www.openresearchbeacon.org/project/sciscinet/ |
| Folktables | tabular census data | https://github.com/socialfoundations/folktables |
| FairFace | face images | https://github.com/joojs/fairface |
| Argoverse 2 Motion Forecasting | trajectory / autonomous-driving scenarios | https://argoverse.github.io/user-guide/ |

### Processed RAF-ready datasets

Processed datasets will be released separately.

| Dataset | RAF-ready query/source files | Status |
| --- | --- | --- |
| SciSciNet-v2 | https://huggingface.co/datasets/pengyueli/RAF_SciSciNet | Placeholder |
| Folktables | `TBD` | Placeholder |
| FairFace | https://huggingface.co/datasets/pengyueli/RAF_FairFace | Placeholder |
| Argoverse 2 | `TBD` | Placeholder |

---

## Quick Start

### Run RAF once

```bash
python -m RAF.run_pipeline \
  --query_path /path/to/query.parquet \
  --source_dir /path/to/sources \
  --source_glob "source_*.parquet" \
  --policy fair_eps_greedy \
  --valuation_mode incremental_hybrid \
  --limit_sources 20
```

Windows PowerShell version:

```powershell
python -m RAF.run_pipeline `
  --query_path D:\path\to\query.parquet `
  --source_dir D:\path\to\sources `
  --source_glob source_*.parquet `
  --policy fair_eps_greedy `
  --valuation_mode incremental_hybrid `
  --limit_sources 20
```

You can pass a source manifest instead of a source directory:

```bash
python -m RAF.run_pipeline \
  --query_path /path/to/query.parquet \
  --source_config source_config.example.json \
  --policy fair_eps_greedy \
  --valuation_mode incremental_hybrid
```

If RAF is installed as an editable package, the console entry point is also available:

```bash
raf-run \
  --query_path /path/to/query.parquet \
  --source_dir /path/to/sources
```

To inspect all available command-line options:

```bash
python -m RAF.run_pipeline --help
```

---

## Running Experiments

Use `run_experiments.py` for repeated runs and budget sweeps.

```bash
python -m RAF.run_experiments \
  --query_path /path/to/query.parquet \
  --source_dir /path/to/sources \
  --source_glob "source_*.parquet" \
  --policy fair_eps_greedy \
  --valuation_mode incremental_hybrid \
  --max_cost_start 6000 \
  --max_cost_end 20000 \
  --max_cost_step 2000 \
  --runs 5
```

Windows PowerShell version:

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

If installed with `pip install -e .`, you can also run:

```bash
raf-experiments \
  --query_path /path/to/query.parquet \
  --source_dir /path/to/sources
```

To inspect all experiment options:

```bash
python -m RAF.run_experiments --help
```

---

## External Fair-Clustering Evaluation

`fair_external_eval.py` is kept for compatibility with external fair-clustering solvers. The external solver repository is not bundled in this public release.

If you use:

```text
--eval_clustering_method external_fair_relax_merge
```

then you must also provide:

```text
--external_fair_algo_dir /path/to/external/solver
```

Install the external solver's dependencies separately and follow its license.

---

## Reproducibility Notes

- RAF does not bundle raw datasets or processed query/source files.
- Public code starts from RAF-ready Parquet inputs.
- The default embedding column is `embedding`.
- The default sensitive attribute column is `is_english_name`.
- `incremental_hybrid` requires `hnswlib`.
- External fair-clustering solvers are optional and must be installed separately.
- Dataset-specific preprocessing should be documented together with the processed dataset release.

---

## Citation

If you use RAF in your research, please cite the corresponding paper.

```bibtex
@article{raf2026,
  title   = {Retrieval-Augmented Dataset Assembly for Fair Clustering},
  author  = {Anonymous Authors},
  journal = {TBD},
  year    = {2026}
}
```

Please also cite the original datasets used in your experiments according to their official citation instructions.

---

## License

The license for this repository is `TBD`.

Raw datasets and external solvers are governed by their own licenses and terms of use.

---

## Contact

For questions about RAF, please open an issue in this repository or contact the authors.
