#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Downloading KQA Pro dataset..."
wget -O "$script_dir/train.json" \
	https://huggingface.co/datasets/drt/kqa_pro/resolve/main/train.json
wget -O "$script_dir/val.json" \
	https://huggingface.co/datasets/drt/kqa_pro/resolve/main/val.json

echo "Downloading the KG file..."
wget -O "$script_dir/kb.json" \
	https://github.com/data-iitd/symKGQA/raw/refs/heads/main/QUACK/preprocessed_kb/kb.json
