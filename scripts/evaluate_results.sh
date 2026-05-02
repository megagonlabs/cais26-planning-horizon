#!/bin/bash

# Check if a directory argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <directory> [--base-url <url>] [--max-tokens <n>] [--model <model>]"
    exit 1
fi

DIR="$1"
shift

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist."
    exit 1
fi

echo "Evaluating results in directory: $DIR"

# Find all result.jsonl files recursively and run the evaluation script for each
find "$DIR" -type f -name "result.jsonl" | while read -r file; do
    echo "Evaluating: $file"
    uv run python scripts/evaluate_result.py -i "$file" --workers 50 "$@"
done
