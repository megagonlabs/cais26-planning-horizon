"""
Sample HotpotQA questions from processed multi-objective dataset for manual verification.

This script extracts component questions (bridge and comparison) from a processed
multi-objective dataset file and writes a balanced sample to CSV for manual inspection.

The processed file contains multi-objective examples with component metadata.
This script extracts the original component questions and formats them for review.

Usage:
    # Sample from test split (50 bridge + 50 comparison = 100 total)
    uv run python data/multiobj_hotpotqa/scripts/sample_questions_to_csv.py \
        --input data/multiobj_hotpotqa/processed/test.v1.json \
        --num-examples 50 \
        --seed 42 \
        --output data/multiobj_hotpotqa/manual_verification_sample.csv

    # Sample from train split
    uv run python data/multiobj_hotpotqa/scripts/sample_questions_to_csv.py \
        --input data/multiobj_hotpotqa/processed/train.v1.json \
        --num-examples 100 \
        --seed 42 \
        --output data/multiobj_hotpotqa/manual_verification_train_sample.csv
"""

from pathlib import Path
from typing import Any
import argparse
import csv
import json
import random


def load_json(file_path: Path) -> list[dict[str, Any]]:
    """Load data from JSON file (list format)."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_supporting_facts(component_meta: dict[str, Any]) -> str:
    """Format supporting facts for CSV output."""
    lines = []
    for sent in component_meta.get("supporting_sentences", []):
        lines.append(f"[{sent['title']}] {sent['text']}")
    return "\n".join(lines) if lines else "(No supporting facts)"


def extract_component_questions(
    processed_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Extract component questions from processed multi-objective examples.

    Args:
        processed_data: List of processed examples from processed JSON file

    Returns:
        List of component questions with: id, type, question, answer, supporting_facts
    """
    components = []

    for example in processed_data:
        k = example["metadata"]["k"]
        answers = example["answers"]
        component_metas = example["metadata"]["components"]
        question_text = example["question"]

        # Parse questions from the combined question text
        if k == 1:
            # Single question: use as-is
            questions = [question_text]
        else:
            # Multi-objective: parse numbered questions ("1. Q1\n2. Q2\n...")
            lines = question_text.split("\n")
            questions = []
            for line in lines:
                if line.strip() and ". " in line:
                    # Remove numbering: "1. Q" -> "Q"
                    q = line.split(". ", 1)[1] if ". " in line else line
                    questions.append(q)
                else:
                    questions.append(line.strip())

        # Create component entries
        for i, comp_meta in enumerate(component_metas):
            question = questions[i] if i < len(questions) else ""
            answer = answers[i] if i < len(answers) else ""

            component = {
                "id": comp_meta["id"],
                "type": comp_meta["type"],
                "question": question,
                "answer": answer,
                "supporting_facts": format_supporting_facts(comp_meta),
            }
            components.append(component)

    return components


def main():
    parser = argparse.ArgumentParser(
        description="Sample HotpotQA questions from processed dataset for manual verification"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to processed multi-objective dataset JSON file",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=50,
        help="Number of bridge and comparison examples to sample (default: 50 each)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output CSV file path",
    )
    args = parser.parse_args()

    # Load processed data
    print(f"Loading processed dataset from: {args.input}")
    processed_data = load_json(args.input)
    print(f"Loaded {len(processed_data)} multi-objective examples")

    # Extract component questions
    print("Extracting component questions...")
    all_components = extract_component_questions(processed_data)
    print(f"Total component questions: {len(all_components)}")

    # Filter by type
    bridge_components = [c for c in all_components if c["type"] == "bridge"]
    comparison_components = [c for c in all_components if c["type"] == "comparison"]
    print(f"  Bridge: {len(bridge_components)}")
    print(f"  Comparison: {len(comparison_components)}")

    # Check availability
    num_examples = args.num_examples
    if len(bridge_components) < num_examples:
        print(
            f"Warning: Requested {num_examples} bridge but only {len(bridge_components)} available"
        )
        num_bridge = len(bridge_components)
    else:
        num_bridge = num_examples

    if len(comparison_components) < num_examples:
        print(
            f"Warning: Requested {num_examples} comparison but only {len(comparison_components)} available"
        )
        num_comparison = len(comparison_components)
    else:
        num_comparison = num_examples

    # Sample
    random.seed(args.seed)
    sampled_bridge = random.sample(bridge_components, num_bridge)
    sampled_comparison = random.sample(comparison_components, num_comparison)
    sampled = sampled_bridge + sampled_comparison

    print(f"Sampled {num_bridge} bridge + {num_comparison} comparison = {len(sampled)} total (seed={args.seed})")

    # Write to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "type", "question", "answer", "supporting_facts"])
        for comp in sampled:
            writer.writerow([
                comp["id"],
                comp["type"],
                comp["question"],
                comp["answer"],
                comp["supporting_facts"],
            ])

    print(f"Wrote sample to: {args.output}")


if __name__ == "__main__":
    main()
