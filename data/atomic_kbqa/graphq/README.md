# GraphQ dataset

This directory contains the local `GraphQ` files used by the
`Atomic KBQA` experiments.

For the canonical preprocessing workflow and evaluator guidance, see:

- [`docs/setup/data.md`](../../../docs/setup/data.md)
- [`docs/preprocessing/atomic_kbqa.md`](../../../docs/preprocessing/atomic_kbqa.md)

## Directory layout

| Path | Purpose |
| --- | --- |
| `GraphQ_train.json` | Raw train split prepared with `KBQA-o1` |
| `GraphQ_test.json` | Raw test split prepared with `KBQA-o1` |
| `scripts/` | GraphQ-specific preprocessing, retrieval, and inspection scripts |
| `processed/` | Generated benchmark files, held-out pool, retrieval outputs, and metrics |

## Expected local files

Before running the GraphQ-specific scripts, this directory should contain:

- `GraphQ_train.json`
- `GraphQ_test.json`

The generated outputs are written to `processed/`.

## Quick start

```shell
uv run python data/atomic_kbqa/graphq/scripts/preprocess_graphq.py
uv run python data/atomic_kbqa/graphq/scripts/retrieve_graphq_examples.py
uv run python src/planning/task_characterization/scripts/compute_metrics_atomic_kbqa.py --data-dir data/atomic_kbqa/graphq --dataset graphq
```

`preprocess_graphq.py` creates the balanced benchmark splits.

`retrieve_graphq_examples.py` attaches nearest-neighbor demonstrations.
In the current artifact, the GraphQ held-out pool is empty, so the train
retrieval output is not written and only the test retrieval file is produced.

`compute_metrics_atomic_kbqa.py` produces the per-example DAG metrics CSV.

See the shared Atomic KBQA guide for expected counts, ODBC prerequisites,
and optional validation commands.
