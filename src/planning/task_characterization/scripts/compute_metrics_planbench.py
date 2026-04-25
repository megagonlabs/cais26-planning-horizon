"""
compute_metrics_planbench.py

Script to compute task characterization metrics for PlanBench dataset.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from planning.task_characterization.metrics_utils import compute_dag_complexity


def load_planbench_data(json_path: Path) -> list[dict]:
    """
    Load preprocessed PlanBench data from JSON file.

    Args:
        json_path: Path to preprocessed JSON file

    Returns:
        List of problem instances
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def compute_input_length(question: str, goal: str) -> int:
    """
    Compute the token length of the problem description (initial state + goal).

    Args:
        question: Initial state description
        goal: Goal state description

    Returns:
        Token count (whitespace tokenization)
    """
    combined = f"{question} Goal: {goal}"
    return len(combined.split())


def compute_workflow_length(dag: list[dict]) -> int:
    """
    Compute the number of steps in the DAG (sequential plan).

    Args:
        dag: List of DAG nodes

    Returns:
        Number of nodes in the DAG
    """
    return len(dag)


def main(args):
    """
    Main function to compute metrics for PlanBench dataset variants.
    """
    data_dir = Path(args.data_dir)
    processed_dir = data_dir / "processed"
    output_dir = processed_dir

    # Find all preprocessed JSON files
    json_files = sorted(processed_dir.glob("*.v1.json"))

    if not json_files:
        print(f"ERROR: No preprocessed JSON files found in {processed_dir}")
        return

    print("=" * 80)
    print("PLANBENCH TASK CHARACTERIZATION METRICS")
    print("=" * 80)

    for json_path in json_files:
        # Extract domain and variant from filename (e.g., blocksworld_basic.v1.json)
        filename = json_path.stem.replace(".v1", "")
        output_path = output_dir / f"{filename}_metrics.v1.csv"

        print(f"\nProcessing: {json_path.name}")
        print(f"Output: {output_path}")

        # Load data
        data = load_planbench_data(json_path)

        if not data:
            print(f"  WARNING: No instances found in {json_path}")
            continue

        # Compute metrics for each instance
        rows = []
        min_input_len = float("inf")
        max_input_len = 0
        min_input_item = None
        max_input_item = None
        min_workflow_len = float("inf")
        max_workflow_len = 0
        min_workflow_item = None
        max_workflow_item = None

        for item in tqdm(data, desc=f"  {filename}", leave=False):
            question = item["question"]
            goal = item["goal"]
            dag = item["dag"]

            input_len = compute_input_length(question, goal)
            workflow_len = compute_workflow_length(dag)

            # Track min/max items for input length
            if input_len < min_input_len:
                min_input_len = input_len
                min_input_item = item
            if input_len > max_input_len:
                max_input_len = input_len
                max_input_item = item

            # Track min/max items for workflow length
            if workflow_len < min_workflow_len:
                min_workflow_len = workflow_len
                min_workflow_item = item
            if workflow_len > max_workflow_len:
                max_workflow_len = workflow_len
                max_workflow_item = item

            # Format DAG for DAG complexity computation
            nodes = []
            for j, step in enumerate(dag):
                node = {"index": j, "parents": step["dependencies"], "children": []}
                nodes.append(node)

            dag_complexity = compute_dag_complexity(nodes, ignore_input_node=False)

            rows.append(
                {
                    "id": item["id"],
                    "domain": item["domain"],
                    "split": item["split"],
                    "input_len": input_len,
                    "workflow_len": workflow_len,
                }
            )
            if dag_complexity:
                rows[-1].update(dag_complexity)

        # Write results to CSV
        df = pd.DataFrame.from_records(rows)
        df.to_csv(output_path, index=False)
        print(f"  Wrote {len(rows)} rows to {output_path}")

        # Show summary statistics
        arr_input = np.array([row["input_len"] for row in rows])
        arr_workflow = np.array([row["workflow_len"] for row in rows])

        print(f"\n  Summary for {filename}:")
        print(f"    Count: {len(rows)}")
        print(
            f"    Input Length - Min: {arr_input.min()} Max: {arr_input.max()} "
            f"Mean: {arr_input.mean():.2f} Median: {np.median(arr_input):.2f} "
            f"Std: {arr_input.std():.2f}"
        )
        print(
            f"    Workflow Length - Min: {arr_workflow.min()} Max: {arr_workflow.max()} "
            f"Mean: {arr_workflow.mean():.2f} Median: {np.median(arr_workflow):.2f} "
            f"Std: {arr_workflow.std():.2f}"
        )

        # Show examples
        if min_input_item:
            print(f"\n  Min Input Length ({min_input_len} tokens):")
            print(f"    ID: {min_input_item['id']}")
            print(f"    Question: {min_input_item['question'][:100]}...")
        if max_input_item:
            print(f"\n  Max Input Length ({max_input_len} tokens):")
            print(f"    ID: {max_input_item['id']}")
            print(f"    Question: {max_input_item['question'][:100]}...")
        if min_workflow_item:
            print(f"\n  Min Workflow Length ({min_workflow_len} steps):")
            print(f"    ID: {min_workflow_item['id']}")
            print(f"    Plan: {min_workflow_item['dag']}")
        if max_workflow_item:
            print(f"\n  Max Workflow Length ({max_workflow_len} steps):")
            print(f"    ID: {max_workflow_item['id']}")
            print(f"    Plan: {max_workflow_item['dag'][:5]}... ({len(max_workflow_item['dag'])} steps total)")

    print("\n" + "=" * 80)
    print("Metrics computation complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PlanBench task metrics.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/planbench",
        help="Directory containing PlanBench processed/ subdirectory",
    )
    args = parser.parse_args()

    main(args)
