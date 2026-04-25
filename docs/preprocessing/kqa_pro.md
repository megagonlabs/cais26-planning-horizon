# KQA Pro preprocessing

## Overview

`KQA Pro` is the primary preprocessing walkthrough for this repository.
The pipeline validates gold KoPL programs against the provided knowledge
base, converts the programs to DAG form, rebalances the benchmark by
workflow length, and retrieves similar held-out examples for few-shot
prompting.

The implementation targets `200` examples per complexity bin for each
split. With the raw files downloaded by
`data/kopl_kbqa/kqa_pro/download.sh`, the repository produced the
following outputs in this session:

| Split | Target | Actual output | Why |
| --- | --- | --- | --- |
| `train` | `1000` examples | `1000` examples | All five bins had at least `200` validated examples |
| `val` | `1000` examples | `929` examples | Bin `4` (`10+` DAG nodes) had only `129` validated examples |

## Prerequisites

- Complete the code setup described in
  [`docs/setup/code.md`](../setup/code.md).
- Download the raw data as described in
  [`data/kopl_kbqa/kqa_pro/README.md`](../../data/kopl_kbqa/kqa_pro/README.md).

| Path | Produced by | Used for |
| --- | --- | --- |
| `data/kopl_kbqa/kqa_pro/train.json` | `download.sh` | Raw training questions and KoPL programs |
| `data/kopl_kbqa/kqa_pro/val.json` | `download.sh` | Raw validation questions and KoPL programs |
| `data/kopl_kbqa/kqa_pro/kb.json` | `download.sh` | Knowledge base loaded by `KoPLEngine` |
| `data/kopl_kbqa/kqa_pro/scripts/embed_kb.py` | Repository source | Generates KoPL grounding embeddings for the demo and worker agents |
| `data/kopl_kbqa/kqa_pro/scripts/preprocess_kqa_pro.py` | Repository source | Validation, DAG conversion, and balanced sampling |
| `data/kopl_kbqa/kqa_pro/scripts/retrieve_kqa_pro_examples.py` | Repository source | Similarity-based demonstration retrieval |
| `data/kopl_kbqa/kqa_pro/scripts/run_preprocessing.sh` | Repository source | KQA Pro-only batch wrapper for the walkthrough |

## Usage

Run the following commands to reproduce the standard walkthrough:

```shell
bash data/kopl_kbqa/kqa_pro/download.sh
bash data/kopl_kbqa/kqa_pro/scripts/run_preprocessing.sh
```

`run_preprocessing.sh` now includes the embedding step before the dataset
preprocessing stages. The standalone command remains useful when you want to
prepare only the demo and worker-agent prerequisites.

## Breakdown of the setup flow

### Stage 1: generate KoPL grounding embeddings

```shell
uv run python data/kopl_kbqa/kqa_pro/scripts/embed_kb.py --kb_path data/kopl_kbqa/kqa_pro/kb.json
```

This step creates:

- `data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5/entity_embeddings.pkl`
- `data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5/key_embeddings.pkl`
- `data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5/value_embeddings.pkl`

## Breakdown of `run_preprocessing.sh`

`run_preprocessing.sh` runs four steps in order:

1. `embed_kb.py`
2. `preprocess_kqa_pro.py`
3. `retrieve_kqa_pro_examples.py`
4. `src/planning/task_characterization/scripts/compute_metrics_kqa_pro.py`

Use the manual commands below if you want to separate experiment inputs
from the optional analysis output.

### Stage 2: validation, DAG conversion, and balanced sampling

```shell
uv run python data/kopl_kbqa/kqa_pro/scripts/preprocess_kqa_pro.py
```

### Stage 3: held-out example retrieval for agent experiments

```shell
uv run python data/kopl_kbqa/kqa_pro/scripts/retrieve_kqa_pro_examples.py
```

### Stage 4: task-characterization metrics for topology analysis

```shell
uv run python src/planning/task_characterization/scripts/compute_metrics_kqa_pro.py --data-dir data/kopl_kbqa/kqa_pro
```

## Optional: inspect the retrieval outputs in Streamlit

Run the following command if you want to inspect the retrieved
demonstration candidates in a browser:

```shell
uv run streamlit run data/kopl_kbqa/kqa_pro/scripts/view_retrieval_results.py
```

## Output files

| Path | Produced by | Contents | Observed output in this session |
| --- | --- | --- | --- |
| `data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5/{entity,key,value}_embeddings.pkl` | Stage 1 | KoPL grounding embeddings for the demo and worker agents | Three pickle files expected |
| `data/kopl_kbqa/kqa_pro/processed/train.v1.json` | Stage 2 | Balanced train split with DAGs and metadata | `1000` examples |
| `data/kopl_kbqa/kqa_pro/processed/val.v1.json` | Stage 2 | Balanced validation split with DAGs and metadata | `929` examples |
| `data/kopl_kbqa/kqa_pro/processed/train_heldout_pool.v1.json` | Stage 2 | Validated train examples not selected for the benchmark | `92908` examples |
| `data/kopl_kbqa/kqa_pro/processed/train.v1.50nn.json` | Stage 3 | Train split with `50` retrieved demonstration candidates per example | `1000` examples |
| `data/kopl_kbqa/kqa_pro/processed/val.v1.50nn.json` | Stage 3 | Validation split with `50` retrieved demonstration candidates per example | `929` examples |
| `data/kopl_kbqa/kqa_pro/processed/kqa_pro_values.v1.csv` | Stage 4 | Per-example task-characterization metrics over the preprocessed train and val splits | `1929` data rows expected |

## Sanity checks

### After download

- Confirm that the three raw files exist:
  - `data/kopl_kbqa/kqa_pro/train.json`
  - `data/kopl_kbqa/kqa_pro/val.json`
  - `data/kopl_kbqa/kqa_pro/kb.json`

### After Stage 1

- Confirm that the following files exist:
  - `data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5/entity_embeddings.pkl`
  - `data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5/key_embeddings.pkl`
  - `data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5/value_embeddings.pkl`

### After Stage 2

- The script should report:
  - `Selected 1000 examples for train`
  - `Selected 929 examples for val`
  - `Warning: Bin 4 has only 129 examples (need 200)`
- The processed JSON files listed above should exist.

### After Stage 3

- The script should report:
  - `Loaded 92908 heldout examples`
  - `Saved 1000 examples with 50 candidates each`
  - `Saved 929 examples with 50 candidates each`
- The retrieval viewer should load without a missing-file error.

### After Stage 4

- `data/kopl_kbqa/kqa_pro/processed/kqa_pro_values.v1.csv` should exist.
- The CSV should contain one row per example from `train.v1.json` and
  `val.v1.json`, so `1929` data rows are expected for the current raw
  files.

## Runtime, storage, and API-cost notes

- No paid API is used anywhere in the `KQA Pro` preprocessing pipeline.
- `data/kopl_kbqa/kqa_pro/scripts/embed_kb.py` downloads the
  `BAAI/bge-base-en-v1.5` model on the
  first run and writes three embedding pickle files under `embeddings/`.
- Execution time (example):
  - validation and balanced sampling took about `3m29s`
    (`3m05s` for `train`, `0m24s` for `val`)
  - retrieval took about `4m15s` after the held-out pool was ready
    (`3m43s` to embed the held-out pool, then about `0m31s` for the train
    and val queries)
- The first retrieval run downloaded the
  `BAAI/bge-base-en-v1.5` model weights (`~400 MB`).
- The wrapper's final CSV step is lightweight compared with the validation
  and retrieval stages.
