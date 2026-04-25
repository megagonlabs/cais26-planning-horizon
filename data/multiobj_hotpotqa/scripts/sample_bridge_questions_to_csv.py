"""
Sample HotpotQA bridge questions and write to CSV for manual annotation.

This script samples a specified number of bridge questions from a HotpotQA JSON file
and writes their _id, question, answer, and formatted supporting facts to a CSV file.

Usage:
    uv run python data/multiobj_hotpotqa/scripts/sample_bridge_questions_to_csv.py \
        --input data/multiobj_hotpotqa/hotpot_train_v1.1.json \
        --num-examples 100 \
        --seed 42 \
        --output data/multiobj_hotpotqa/manual_bridge_sample.csv
"""

from pathlib import Path
import argparse
import csv
import random
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.multiobj_hotpotqa.scripts.utils import (  # noqa: E402
    load_json,
    filter_by_type,
    format_supporting_sentences,
)

def main():
    parser = argparse.ArgumentParser(
        description="Sample HotpotQA bridge questions and write to CSV for manual annotation"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input HotpotQA JSON file",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of bridge questions to sample (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path",
    )
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from: {args.input}")
    all_records = load_json(args.input)
    print(f"Loaded {len(all_records)} records")

    # Filter bridge questions
    bridge_examples = filter_by_type(all_records, "bridge")
    print(f"Found {len(bridge_examples)} bridge questions")

    if len(bridge_examples) < args.num_examples:
        print(f"Warning: Requested {args.num_examples} but only {len(bridge_examples)} bridge questions available.")
        num_examples = len(bridge_examples)
    else:
        num_examples = args.num_examples

    # Sample random examples
    random.seed(args.seed)
    sampled = random.sample(bridge_examples, num_examples)
    print(f"Sampled {num_examples} bridge questions (seed={args.seed})")

    # Write to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["_id", "question", "answer", "supporting_facts"])
        for ex in sampled:
            formatted_facts = format_supporting_sentences(ex)
            answer = ex["answer"] if "answer" in ex else ex.get("golden_answers", [""])[0]
            writer.writerow([
                ex["_id"],
                ex["question"],
                answer,
                formatted_facts,
            ])
    print(f"Wrote {num_examples} bridge questions to: {args.output}")

if __name__ == "__main__":
    main()
