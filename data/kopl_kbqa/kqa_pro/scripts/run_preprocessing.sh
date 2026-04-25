#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../../../" && pwd)"

cd "$repo_root"

uv run python data/kopl_kbqa/kqa_pro/scripts/embed_kb.py --kb_path data/kopl_kbqa/kqa_pro/kb.json
uv run python data/kopl_kbqa/kqa_pro/scripts/preprocess_kqa_pro.py
uv run python data/kopl_kbqa/kqa_pro/scripts/retrieve_kqa_pro_examples.py
uv run python src/planning/task_characterization/scripts/compute_metrics_kqa_pro.py --data-dir data/kopl_kbqa/kqa_pro
