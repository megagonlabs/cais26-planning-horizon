# Multi-objective HotpotQA dataset

This directory contains the local files for the `Multi-objective HotpotQA`
track.

For the canonical preprocessing workflow and evaluator guidance, see:

- [`docs/setup/data.md`](../../docs/setup/data.md)
- [`docs/preprocessing/multiobj_hotpotqa.md`](../../docs/preprocessing/multiobj_hotpotqa.md)

## Directory layout

| Path | Purpose |
| --- | --- |
| `download.sh` | Downloads the raw HotpotQA train and dev files |
| `hotpot_train_v1.1.json` | Raw HotpotQA train split |
| `hotpot_dev_distractor_v1.json` | Raw HotpotQA dev split reused as test input |
| `batch_validation/` | Timestamped validation outputs from the OpenAI Batch API |
| `processed/` | Sampled benchmark files, annotated DAGs, retrieval outputs, and metrics |
| `scripts/` | Validation, sampling, annotation, retrieval, and inspection utilities |

## Expected local files

After running `bash data/multiobj_hotpotqa/download.sh`, this directory
should contain:

- `hotpot_train_v1.1.json`
- `hotpot_dev_distractor_v1.json`

The current artifact on this machine also includes populated
`batch_validation/` and `processed/` directories.

## Quick start

```shell
uv run streamlit run data/multiobj_hotpotqa/scripts/multiobj_dag_visualizer.py
uv run streamlit run data/multiobj_hotpotqa/scripts/view_retrieval_results.py
```

`multiobj_dag_visualizer.py` lets evaluators inspect the annotated DAGs that
ship with this artifact.

`view_retrieval_results.py` lets evaluators inspect the retrieved
demonstration candidates attached to the processed train and test files.

See the canonical preprocessing guide for the full staged workflow,
including the paid OpenAI-backed validation and DAG-annotation steps.
