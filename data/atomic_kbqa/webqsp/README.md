# WebQSP dataset

This directory contains the local `WebQSP` files used by the
`Atomic KBQA` experiments.

For the canonical preprocessing workflow and evaluator guidance, see:

- [`docs/setup/data.md`](../../../docs/setup/data.md)
- [`docs/preprocessing/atomic_kbqa.md`](../../../docs/preprocessing/atomic_kbqa.md)

## Directory layout

| Path | Purpose |
| --- | --- |
| `WebQSP_train.json` | Raw train split prepared with `KBQA-o1` |
| `WebQSP_test.json` | Raw test split prepared with `KBQA-o1` |
| `scripts/` | WebQSP-specific preprocessing, retrieval, and inspection scripts |
| `processed/` | Generated benchmark files, held-out pool, retrieval outputs, and metrics |

## Expected local files

Before running the WebQSP-specific scripts, this directory should contain:

- `WebQSP_train.json`
- `WebQSP_test.json`

The generated outputs are written to `processed/`.

## Quick start

```shell
uv run python data/atomic_kbqa/webqsp/scripts/preprocess_webqsp.py
uv run python data/atomic_kbqa/webqsp/scripts/retrieve_webqsp_examples.py
uv run python src/planning/task_characterization/scripts/compute_metrics_atomic_kbqa.py --data-dir data/atomic_kbqa/webqsp --dataset webqsp
```

`preprocess_webqsp.py` creates the balanced benchmark splits and held-out
pool.

`retrieve_webqsp_examples.py` attaches nearest-neighbor demonstrations to
the train and test splits.

`compute_metrics_atomic_kbqa.py` produces the per-example DAG metrics CSV.

See the shared Atomic KBQA guide for expected counts, ODBC prerequisites,
and optional validation commands.
