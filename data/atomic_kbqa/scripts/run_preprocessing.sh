#!/usr/bin/env bash

set -euo pipefail

# Run the reproducible preprocessing steps for all Atomic KBQA datasets.
# Prerequisites:
# - Freebase/Virtuoso is already running and reachable via ODBC.
# - Raw dataset JSON files already exist under data/atomic_kbqa/{grailqa,webqsp,graphq}/.

## GrailQA
uv run python data/atomic_kbqa/grailqa/scripts/preprocess_grailqa.py
uv run python data/atomic_kbqa/grailqa/scripts/retrieve_grailqa_examples.py
uv run python src/planning/task_characterization/scripts/compute_metrics_atomic_kbqa.py --data-dir data/atomic_kbqa/grailqa --dataset grailqa

## WebQSP
uv run python data/atomic_kbqa/webqsp/scripts/preprocess_webqsp.py
uv run python data/atomic_kbqa/webqsp/scripts/retrieve_webqsp_examples.py
uv run python src/planning/task_characterization/scripts/compute_metrics_atomic_kbqa.py --data-dir data/atomic_kbqa/webqsp --dataset webqsp

## GraphQ
uv run python data/atomic_kbqa/graphq/scripts/preprocess_graphq.py
uv run python data/atomic_kbqa/graphq/scripts/retrieve_graphq_examples.py
uv run python src/planning/task_characterization/scripts/compute_metrics_atomic_kbqa.py --data-dir data/atomic_kbqa/graphq --dataset graphq
