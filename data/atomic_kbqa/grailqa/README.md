# GrailQA dataset

This directory contains the local `GrailQA` files used by the
`Atomic KBQA` experiments.

For the canonical preprocessing workflow and evaluator guidance, see:

- [`docs/setup/data.md`](../../../docs/setup/data.md)
- [`docs/preprocessing/atomic_kbqa.md`](../../../docs/preprocessing/atomic_kbqa.md)

## Directory layout

| Path | Purpose |
| --- | --- |
| `GrailQA_train.json` | Raw train split prepared with `KBQA-o1` |
| `GrailQA_test.json` | Raw test split prepared with `KBQA-o1` |
| `scripts/` | GrailQA-specific preprocessing, retrieval, and inspection scripts |
| `processed/` | Generated benchmark files, held-out pool, retrieval outputs, and metrics |

## Expected local files

Before running the GrailQA-specific scripts, this directory should contain:

- `GrailQA_train.json`
- `GrailQA_test.json`

The generated outputs are written to `processed/`.

## Quick start

```shell
uv run python data/atomic_kbqa/grailqa/scripts/preprocess_grailqa.py
uv run python data/atomic_kbqa/grailqa/scripts/retrieve_grailqa_examples.py
uv run python src/planning/task_characterization/scripts/compute_metrics_atomic_kbqa.py --data-dir data/atomic_kbqa/grailqa --dataset grailqa
```

`preprocess_grailqa.py` creates the balanced benchmark splits and held-out
pool.

`retrieve_grailqa_examples.py` attaches nearest-neighbor demonstrations to
the train and test splits.

`compute_metrics_atomic_kbqa.py` produces the per-example DAG metrics CSV.

See the shared Atomic KBQA guide for expected counts, ODBC prerequisites,
and optional validation commands.
