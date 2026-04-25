# Multi-objective HotpotQA preprocessing

## Overview

`Multi-objective HotpotQA` starts from the original HotpotQA train and dev
files, filters `bridge` questions with an LLM-based validation pass,
combines single-objective questions into multi-objective tasks with
$k \in \{1, 2, 3, 4, 5\}$, annotates ground-truth DAGs, retrieves
held-out demonstrations, and computes task-characterization metrics.

This is the heaviest preprocessing pipeline in the repository.

- `KQA Pro` remains the recommended full walkthrough.
- Artifact evaluators are generally expected to inspect the prepared
    `Multi-objective HotpotQA` artifacts on this machine instead of rerunning
    the paid API phases.

## Prerequisites

- Complete the code setup in [`docs/setup/code.md`](../setup/code.md).
- Use [`data/multiobj_hotpotqa/README.md`](../../data/multiobj_hotpotqa/README.md)
    as the local landing page.
- Add `OPENAI_API_KEY` to the repository root `.env` file or export it in
    your shell before running the validation or DAG-annotation stages.

### External requirements

| Requirement | Needed for | Notes |
| --- | --- | --- |
| Raw HotpotQA files | All phases | `download.sh` downloads `hotpot_train_v1.1.json` and `hotpot_dev_distractor_v1.json` |
| OpenAI Batch API | Bridge-question validation | Paid API step; the output directory is timestamped |
| OpenAI chat/responses API | DAG annotation | Paid API step; uses `gpt-5.2-2025-12-11` in the current artifacts |
| `BAAI/bge-base-en-v1.5` | Retrieval | Downloaded automatically on first use |
| Pyserini prebuilt indexes | Downstream search-agent experiments | Not required to generate the JSON preprocessing artifacts, but required for the actual `search` tool used in the `multiobj` experiments |
| Java/Lucene runtime | Pyserini-backed search experiments | Managed through the Pyserini setup documented in [`docs/setup/code.md`](../setup/code.md) |

For the Pyserini indexes used by the downstream experiments, see the
`Pyserini` section in [`docs/setup/code.md`](../setup/code.md). The two
configured prebuilt indexes are:

- `beir-v1.0.0-hotpotqa.bge-base-en-v1.5`
- `beir-v1.0.0-hotpotqa.flat`

### Raw inputs and scripts

| Path | Produced by | Used for |
| --- | --- | --- |
| `data/multiobj_hotpotqa/download.sh` | Repository source | Downloads the raw HotpotQA train and dev files |
| `data/multiobj_hotpotqa/hotpot_train_v1.1.json` | `download.sh` | Train split source |
| `data/multiobj_hotpotqa/hotpot_dev_distractor_v1.json` | `download.sh` | Dev split source, reused as test input |
| `data/multiobj_hotpotqa/scripts/batch_validate_submit.py` | Repository source | Submit bridge-question validation jobs to the OpenAI Batch API |
| `data/multiobj_hotpotqa/scripts/batch_validate_download.py` | Repository source | Download batch results and produce `validated.jsonl` |
| `data/multiobj_hotpotqa/scripts/preprocess_multiobj_hotpotqa.py` | Repository source | Sample train/test multi-objective examples and build the held-out pool |
| `data/multiobj_hotpotqa/scripts/annotate_dag.py` | Repository source | Fill the `dag` field with model-generated component DAGs and deterministic merges |
| `data/multiobj_hotpotqa/scripts/retrieve_multiobj_hotpotqa_examples.py` | Repository source | Attach nearest-neighbor demonstrations |
| `src/planning/task_characterization/scripts/compute_metrics_multiobj_hotpotqa.py` | Repository source | Produce `hotpotqa_values.v1.csv` |

## Usage

There is intentionally no single shell wrapper for the full pipeline.
Validation and DAG annotation both depend on timestamped batch/output
directories and paid API calls, so the reproducible path is stage-by-stage.

### Phase 1: download the raw HotpotQA files

```shell
bash data/multiobj_hotpotqa/download.sh
```

### Phase 2: validate `bridge` questions with the OpenAI Batch API

Use the submit script first. It creates a timestamped output directory.

```shell
uv run python data/multiobj_hotpotqa/scripts/batch_validate_submit.py --input data/multiobj_hotpotqa/hotpot_train_v1.1.json --output data/multiobj_hotpotqa/batch_validation --model gpt-4.1-2025-04-14
uv run python data/multiobj_hotpotqa/scripts/batch_validate_submit.py --input data/multiobj_hotpotqa/hotpot_dev_distractor_v1.json --output data/multiobj_hotpotqa/batch_validation --model gpt-4.1-2025-04-14
```

To estimate the validation input-token cost without making any API calls,
use `--dry-run`:

```shell
uv run python data/multiobj_hotpotqa/scripts/batch_validate_submit.py --input data/multiobj_hotpotqa/hotpot_train_v1.1.json --output data/multiobj_hotpotqa/batch_validation --model gpt-4.1-2025-04-14 --dry-run
```

After the batches finish, download and merge the results. On this machine,
the existing validation outputs live in the following directories:

```shell
uv run python data/multiobj_hotpotqa/scripts/batch_validate_download.py --batch-dir data/multiobj_hotpotqa/batch_validation/gpt-4.1-2025-04-14/2025-12-15-23-35-48
uv run python data/multiobj_hotpotqa/scripts/batch_validate_download.py --batch-dir data/multiobj_hotpotqa/batch_validation/gpt-4.1-2025-04-14/2025-12-15-23-02-19
```

### Phase 3: sample the train/test multi-objective benchmark files

```shell
uv run python data/multiobj_hotpotqa/scripts/preprocess_multiobj_hotpotqa.py --validated-bridge data/multiobj_hotpotqa/batch_validation/gpt-4.1-2025-04-14/2025-12-15-23-35-48/validated.jsonl --original-hotpotqa data/multiobj_hotpotqa/hotpot_train_v1.1.json --output data/multiobj_hotpotqa/processed/train.v1.json --split train --samples-per-k 200 --heldout-pool data/multiobj_hotpotqa/processed/train_heldout_pool.v1.json
uv run python data/multiobj_hotpotqa/scripts/preprocess_multiobj_hotpotqa.py --validated-bridge data/multiobj_hotpotqa/batch_validation/gpt-4.1-2025-04-14/2025-12-15-23-02-19/validated.jsonl --original-hotpotqa data/multiobj_hotpotqa/hotpot_dev_distractor_v1.json --output data/multiobj_hotpotqa/processed/test.v1.json --split test --samples-per-k 200
```

### Phase 4: annotate DAGs

```shell
uv run python data/multiobj_hotpotqa/scripts/annotate_dag.py --input data/multiobj_hotpotqa/processed/train.v1.json --output data/multiobj_hotpotqa/processed/train.v1.annotated.json --workers 50
uv run python data/multiobj_hotpotqa/scripts/annotate_dag.py --input data/multiobj_hotpotqa/processed/test.v1.json --output data/multiobj_hotpotqa/processed/test.v1.annotated.json --workers 50
uv run python data/multiobj_hotpotqa/scripts/annotate_dag.py --input data/multiobj_hotpotqa/processed/train_heldout_pool.v1.json --output data/multiobj_hotpotqa/processed/train_heldout_pool.v1.annotated.json --workers 50
```

### Phase 5: retrieve held-out demonstrations

```shell
uv run python data/multiobj_hotpotqa/scripts/retrieve_multiobj_hotpotqa_examples.py
```

### Phase 6: compute task-characterization metrics

```shell
uv run python src/planning/task_characterization/scripts/compute_metrics_multiobj_hotpotqa.py --data-dir data/multiobj_hotpotqa
```

## Optional inspection commands

Use the viewers below if you want to inspect the prepared artifacts rather
than rerun the expensive API-backed stages.

```shell
uv run streamlit run data/multiobj_hotpotqa/scripts/multiobj_dag_visualizer.py
uv run streamlit run data/multiobj_hotpotqa/scripts/view_retrieval_results.py
```

## Output files

The current artifacts on this machine produced the following outputs.

| Path | Produced by | Contents | Observed output in this session |
| --- | --- | --- | --- |
| `data/multiobj_hotpotqa/batch_validation/gpt-4.1-2025-04-14/2025-12-15-23-35-48/validated.jsonl` | Validation download | Train `bridge` questions with `valid_reasoning_structure` | `72991` validated records |
| `data/multiobj_hotpotqa/batch_validation/gpt-4.1-2025-04-14/2025-12-15-23-02-19/validated.jsonl` | Validation download | Dev `bridge` questions with `valid_reasoning_structure` | `5918` validated records |
| `data/multiobj_hotpotqa/processed/train.v1.json` | Sampling | Train split before DAG annotation | `1000` examples |
| `data/multiobj_hotpotqa/processed/test.v1.json` | Sampling | Test split before DAG annotation | `1000` examples |
| `data/multiobj_hotpotqa/processed/train_heldout_pool.v1.json` | Sampling | Held-out demonstration pool before DAG annotation | `2000` examples |
| `data/multiobj_hotpotqa/processed/train.v1.annotated.json` | DAG annotation | Train split with merged DAGs | `1000` examples |
| `data/multiobj_hotpotqa/processed/test.v1.annotated.json` | DAG annotation | Test split with merged DAGs | `1000` examples |
| `data/multiobj_hotpotqa/processed/train_heldout_pool.v1.annotated.json` | DAG annotation | Held-out pool with merged DAGs | `2000` examples |
| `data/multiobj_hotpotqa/processed/train.v1.annotated.50nn.json` | Retrieval | Train split with `50` retrieved candidates per example | `1000` examples |
| `data/multiobj_hotpotqa/processed/test.v1.annotated.50nn.json` | Retrieval | Test split with `50` retrieved candidates per example | `1000` examples |
| `data/multiobj_hotpotqa/processed/hotpotqa_values.v1.csv` | Metrics | Per-example task-characterization metrics over train and test | `2000` data rows expected |

## Sanity checks

### After download

- Confirm that both raw files exist:
    - `data/multiobj_hotpotqa/hotpot_train_v1.1.json`
    - `data/multiobj_hotpotqa/hotpot_dev_distractor_v1.json`

### After validation

- Each batch directory should contain:
    - `batch_info.json`
    - `validated.jsonl`
    - `validation_stats.json`
- On this machine, the current validation stats are:
    - train bridge questions: `34627 / 72991` valid (`47.4%`)
    - dev bridge questions: `2711 / 5918` valid (`45.8%`)

### After sampling

- `train.v1.json` should contain `1000` examples.
- `test.v1.json` should contain `1000` examples.
- `train_heldout_pool.v1.json` should contain `2000` examples.
- The benchmark construction targets `200` examples for each `k` value.

### After DAG annotation

- `train.v1.annotated.json`, `test.v1.annotated.json`, and
    `train_heldout_pool.v1.annotated.json` should all exist.
- The current annotated files record `gpt-5.2-2025-12-11` in
    `metadata.dag_annotation.model`.
- The `dag` field should be non-empty for annotated examples.

### After retrieval

- `train.v1.annotated.50nn.json` and `test.v1.annotated.50nn.json` should
    both exist.
- The retrieval viewer should open without missing-file errors.

### After metrics

- `data/multiobj_hotpotqa/processed/hotpotqa_values.v1.csv` should exist.
- The CSV should contain `2000` data rows for the current annotated train
    and test files combined.

## Runtime, storage, and API-cost notes

- The validation and DAG-annotation phases both require a paid OpenAI API.
- The exact end-to-end cost is unavailable. Use caution and, when in doubt,
    run the validation stage with `--dry-run` before submitting a batch.
- Because this pipeline subsamples the benchmark and uses the OpenAI Batch
    API (50% discount) for the validation phase, the total preprocessing cost
    should remain well below `$100` in practice.
- The retrieval stage downloads `BAAI/bge-base-en-v1.5` on first use.
- The Pyserini indexes referenced in the code setup guide are separate from
    this JSON preprocessing pipeline; they are required for the downstream
    `search` tool used in the actual `multiobj` experiments and can consume
    substantial extra disk under `~/.cache/pyserini/indexes/`.
