# Atomic KBQA preprocessing

## Overview

`Atomic KBQA` covers three Freebase-based datasets:

- `GrailQA`
- `WebQSP`
- `GraphQ`

The raw JSON files in this repository are already in the `KBQA-o1`
processed format: each example includes a question, answer set,
SPARQL query, S-expression, and linearized `function_list`.

This repository adds four preprocessing stages on top of those files:

1. deduplicate by `sexpr`
2. filter out nested relation expressions that the worker tools do not support
3. build balanced benchmark splits and convert the `function_list` to DAG form
4. retrieve similar held-out examples and compute task-characterization metrics

After the shared prerequisites are in place, the canonical wrapper is:

```shell
bash data/atomic_kbqa/scripts/run_preprocessing.sh
```

## Prerequisites

- Complete the code setup in [`docs/setup/code.md`](../setup/code.md).
- Use [`data/atomic_kbqa/README.md`](../../data/atomic_kbqa/README.md)
    as the local landing page.
- Ensure the Freebase/Virtuoso service required by `KBQA-o1` is already
    running and reachable via ODBC.

### Shared Freebase and KBQA-o1 requirements

The raw files in this repository assume the upstream preparation described
in [`vendor/KBQA-o1/README.md`](../../vendor/KBQA-o1/README.md):

- download and preprocess the original `GrailQA`, `WebQSP`, and `GraphQ`
    datasets with `KBQA-o1`
- host Freebase with Virtuoso
- download the ontology files `fb_roles`, `fb_types`, and
    `reverse_properties`

The default local Freebase settings in this repository are:

| Setting | Value | Source |
| --- | --- | --- |
| ODBC port | `13002` | `src/planning/tools/freebase/default_config.py` |
| SPARQL endpoint | `http://localhost:13002/sparql` | `src/planning/tools/freebase/default_config.py` |
| ODBC driver path | `vendor/KBQA-o1/utils/lib/virtodbc.so` | `src/planning/tools/freebase/default_config.py` |

Use the ODBC sanity check in [`docs/setup/code.md`](../setup/code.md) if
you want to verify the connection before running any dataset script.

### Shared local files and scripts

| Path | Produced by | Used for |
| --- | --- | --- |
| `data/atomic_kbqa/grailqa/GrailQA_train.json` | Upstream `KBQA-o1` preprocessing | GrailQA train split input |
| `data/atomic_kbqa/grailqa/GrailQA_test.json` | Upstream `KBQA-o1` preprocessing | GrailQA test split input |
| `data/atomic_kbqa/webqsp/WebQSP_train.json` | Upstream `KBQA-o1` preprocessing | WebQSP train split input |
| `data/atomic_kbqa/webqsp/WebQSP_test.json` | Upstream `KBQA-o1` preprocessing | WebQSP test split input |
| `data/atomic_kbqa/graphq/GraphQ_train.json` | Upstream `KBQA-o1` preprocessing | GraphQ train split input |
| `data/atomic_kbqa/graphq/GraphQ_test.json` | Upstream `KBQA-o1` preprocessing | GraphQ test split input |
| `data/atomic_kbqa/freebase/fb_roles` | GrailQA ontology download | Ontology-derived relation extraction |
| `data/atomic_kbqa/freebase/fb_types` | GrailQA ontology download | Freebase type resources for the worker stack |
| `data/atomic_kbqa/freebase/reverse_properties` | GrailQA ontology download | Reverse-relation metadata |
| `data/atomic_kbqa/freebase/relation_list.txt` | Existing local file or refresh commands below | Relation inventory used by the atomic KBQA tooling |
| `data/atomic_kbqa/freebase/literal_relation_list.txt` | Existing local file or refresh commands below | Literal-valued relation inventory |
| `data/atomic_kbqa/scripts/run_preprocessing.sh` | Repository source | Canonical wrapper for all three datasets |
| `data/atomic_kbqa/scripts/validate_sparql.py` | Repository source | Optional post-preprocessing validation against the live Freebase service |

### Optional: refresh the Freebase relation lists

These commands are only needed if you are rebuilding the local Freebase
resources or want to verify that the relation inventories still match the
current processed JSON files.

```shell
uv run python data/atomic_kbqa/scripts/extract_ontology_relations.py
uv run python data/atomic_kbqa/scripts/verify_and_update_relation_list.py
uv run python data/atomic_kbqa/scripts/generate_literal_relation_list.py
```

`generate_literal_relation_list.py` queries the live Freebase endpoint and
can take several minutes.

## Usage

Run the shared wrapper after the prerequisites above are ready:

```shell
bash data/atomic_kbqa/scripts/run_preprocessing.sh
```

## Breakdown of `run_preprocessing.sh`

The wrapper runs the following stages for each dataset:

1. dataset-specific preprocessing (`preprocess_*.py`)
2. held-out example retrieval (`retrieve_*.py`)
3. task-characterization metrics (`compute_metrics_atomic_kbqa.py`)

Use the per-dataset commands below if you want to rerun only one dataset.

### GrailQA

```shell
uv run python data/atomic_kbqa/grailqa/scripts/preprocess_grailqa.py
uv run python data/atomic_kbqa/grailqa/scripts/retrieve_grailqa_examples.py
uv run python src/planning/task_characterization/scripts/compute_metrics_atomic_kbqa.py --data-dir data/atomic_kbqa/grailqa --dataset grailqa
```

Optional inspection:

```shell
uv run streamlit run data/atomic_kbqa/grailqa/scripts/view_retrieval_results.py
```

### WebQSP

```shell
uv run python data/atomic_kbqa/webqsp/scripts/preprocess_webqsp.py
uv run python data/atomic_kbqa/webqsp/scripts/retrieve_webqsp_examples.py
uv run python src/planning/task_characterization/scripts/compute_metrics_atomic_kbqa.py --data-dir data/atomic_kbqa/webqsp --dataset webqsp
```

Optional inspection:

```shell
uv run streamlit run data/atomic_kbqa/webqsp/scripts/view_retrieval_results.py
```

### GraphQ

```shell
uv run python data/atomic_kbqa/graphq/scripts/preprocess_graphq.py
uv run python data/atomic_kbqa/graphq/scripts/retrieve_graphq_examples.py
uv run python src/planning/task_characterization/scripts/compute_metrics_atomic_kbqa.py --data-dir data/atomic_kbqa/graphq --dataset graphq
```

Optional inspection:

```shell
uv run streamlit run data/atomic_kbqa/graphq/scripts/view_retrieval_results.py
```

### Optional: validate the generated SPARQL programs

If you want an extra end-to-end check against the live Freebase service,
run `validate_sparql.py` on any processed split.

```shell
uv run python data/atomic_kbqa/scripts/validate_sparql.py --input data/atomic_kbqa/grailqa/processed/grailqa_train.v1.json
uv run python data/atomic_kbqa/scripts/validate_sparql.py --input data/atomic_kbqa/webqsp/processed/webqsp_train.v1.json
uv run python data/atomic_kbqa/scripts/validate_sparql.py --input data/atomic_kbqa/graphq/processed/graphq_train.v1.json
```

## Output files

The current artifacts on this machine produced the following outputs.

| Path | Produced by | Contents | Observed output in this session |
| --- | --- | --- | --- |
| `data/atomic_kbqa/grailqa/processed/grailqa_train.v1.json` | GrailQA preprocessing | Balanced GrailQA train split with DAGs | `500` examples |
| `data/atomic_kbqa/grailqa/processed/grailqa_test.v1.json` | GrailQA preprocessing | Balanced GrailQA test split with DAGs | `478` examples |
| `data/atomic_kbqa/grailqa/processed/grailqa_train_heldout_pool.v1.json` | GrailQA preprocessing | Held-out GrailQA pool for retrieval | `31915` examples |
| `data/atomic_kbqa/grailqa/processed/grailqa_train.v1.50nn.json` | GrailQA retrieval | Train split with `50` candidates per example | `500` examples |
| `data/atomic_kbqa/grailqa/processed/grailqa_test.v1.50nn.json` | GrailQA retrieval | Test split with `50` candidates per example | `478` examples |
| `data/atomic_kbqa/grailqa/processed/grailqa_values.v1.csv` | GrailQA metrics | Per-example task-characterization metrics | `978` data rows expected |
| `data/atomic_kbqa/webqsp/processed/webqsp_train.v1.json` | WebQSP preprocessing | Balanced WebQSP train split with DAGs | `486` examples |
| `data/atomic_kbqa/webqsp/processed/webqsp_test.v1.json` | WebQSP preprocessing | Balanced WebQSP test split with DAGs | `422` examples |
| `data/atomic_kbqa/webqsp/processed/WebQSP_train_heldout_pool.v1.json` | WebQSP preprocessing | Held-out WebQSP pool for retrieval | `1952` examples |
| `data/atomic_kbqa/webqsp/processed/webqsp_train.v1.50nn.json` | WebQSP retrieval | Train split with `50` candidates per example | `486` examples |
| `data/atomic_kbqa/webqsp/processed/webqsp_test.v1.50nn.json` | WebQSP retrieval | Test split with `50` candidates per example | `422` examples |
| `data/atomic_kbqa/webqsp/processed/webqsp_values.v1.csv` | WebQSP metrics | Per-example task-characterization metrics | `908` data rows expected |
| `data/atomic_kbqa/graphq/processed/graphq_train.v1.json` | GraphQ preprocessing | Balanced GraphQ train split with DAGs | `212` examples |
| `data/atomic_kbqa/graphq/processed/graphq_test.v1.json` | GraphQ preprocessing | Balanced GraphQ test split with DAGs | `215` examples |
| `data/atomic_kbqa/graphq/processed/graphq_train_heldout_pool.v1.json` | GraphQ preprocessing | Held-out GraphQ pool for retrieval | `0` examples |
| `data/atomic_kbqa/graphq/processed/graphq_test.v1.50nn.json` | GraphQ retrieval | Test split with retrieved candidates | `215` examples |
| `data/atomic_kbqa/graphq/processed/graphq_values.v1.csv` | GraphQ metrics | Per-example task-characterization metrics | `427` data rows expected |

## Sanity checks

### Shared prerequisite checks

- Confirm that all six raw dataset files listed above exist.
- Confirm that the following Freebase resource files exist:
    - `data/atomic_kbqa/freebase/fb_roles`
    - `data/atomic_kbqa/freebase/fb_types`
    - `data/atomic_kbqa/freebase/reverse_properties`
    - `data/atomic_kbqa/freebase/relation_list.txt`
    - `data/atomic_kbqa/freebase/literal_relation_list.txt`
- If preprocessing fails immediately, re-run the ODBC sanity check from
    [`docs/setup/code.md`](../setup/code.md).

### After GrailQA preprocessing

- `grailqa_train.v1.json` should contain `500` examples.
- `grailqa_test.v1.json` should contain `478` examples.
- `grailqa_train_heldout_pool.v1.json` should contain `31915` examples.

### After WebQSP preprocessing

- `webqsp_train.v1.json` should contain `486` examples.
- `webqsp_test.v1.json` should contain `422` examples.
- `WebQSP_train_heldout_pool.v1.json` should contain `1952` examples.

### After GraphQ preprocessing

- `graphq_train.v1.json` should contain `212` examples.
- `graphq_test.v1.json` should contain `215` examples.
- `graphq_train_heldout_pool.v1.json` is expected to be empty in the
    current artifact because all available train examples were selected into
    the benchmark.

### After retrieval

- GrailQA and WebQSP should create both `*.v1.50nn.json` files.
- GraphQ should create `graphq_test.v1.50nn.json`.
- `graphq_train.v1.50nn.json` is not expected in the current artifact
    because the GraphQ held-out pool is empty, so the train retrieval stage
    exits early with a warning.

### After metrics

- `grailqa_values.v1.csv` should contain `978` data rows.
- `webqsp_values.v1.csv` should contain `908` data rows.
- `graphq_values.v1.csv` should contain `427` data rows.

### Optional SPARQL validation

- `validate_sparql.py` is the fastest way to confirm that the generated
    S-expressions still execute correctly against the local Freebase service.

## Runtime, storage, and API-cost notes

- No paid API is used anywhere in the `Atomic KBQA` preprocessing pipeline.
- The expensive dependency is external infrastructure, not model cost:
    the upstream `KBQA-o1` instructions reference a `53 GB+` Virtuoso DB
    download and recommend a high-memory host for Freebase.
- The preprocessing scripts query Freebase over ODBC to recover entity
    labels and can take several minutes per dataset.
- The retrieval stage downloads the `BAAI/bge-base-en-v1.5` model weights
    on first use (`~400 MB`).
- `GraphQ` is the main edge case in the current artifact: the balanced
    train benchmark already uses all available train examples, so the held-out
    pool is empty and the train retrieval output is omitted.
