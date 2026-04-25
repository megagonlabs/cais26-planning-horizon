"""
compute_metrics_multiobj_hotpotqa.py

Script to compute task characterization metrics for multi-objective HotpotQA dataset.
Supports varying complexity levels (k = 1, 2, 3, 4 sub-questions per example).
"""

import argparse
import json
import os
import sys

from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from planning.task_characterization.metrics_utils import (
    compute_dag_complexity,
)


def load_multiobj_hotpotqa_data(data_dir: str) -> list[dict]:
    """
    Loads preprocessed multi-objective HotpotQA data from local JSON files.

    Args:
        data_dir (str): Directory containing processed/ subdirectory

    Returns:
        list[dict]: Combined list of preprocessed examples with DAGs
    """
    processed_dir = os.path.join(data_dir, "processed")
    train_path = os.path.join(processed_dir, "train.v1.annotated.json")
    test_path = os.path.join(processed_dir, "test.v1.annotated.json")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise ValueError(f"Required files not found in {processed_dir}")

    data = []
    with open(train_path, "r") as f:
        for item in json.load(f):
            item["split"] = "train"
            data.append(item)
    with open(test_path, "r") as f:
        for item in json.load(f):
            item["split"] = "test"
            data.append(item)
    return data


def compute_input_length(question: str) -> int:
    """Computes token length of question using whitespace tokenization."""
    return len(question.split())


def compute_workflow_length(dag: list[dict]) -> int:
    """Computes number of steps in the DAG."""
    return len(dag)


def get_question_complexity(metadata: dict) -> int:
    """Extracts the k value (number of sub-questions) from metadata."""
    return metadata.get("k", 1)


def get_component_types(metadata: dict) -> tuple[str, ...]:
    """
    Extracts component types from metadata.
    Returns tuple of types (e.g., ('bridge', 'comparison')).
    """
    components = metadata.get("components", [])
    return tuple(comp.get("type", "unknown") for comp in components)


def format_dag_for_complexity(dag: list[dict]) -> list[dict]:
    """
    Convert DAG from preprocessed format to complexity computation format.

    Expected DAG format: list of nodes with 'function', 'inputs', 'dependencies' fields
    """
    nodes = []
    for j, step in enumerate(dag):
        node = {
            "index": j,
            "parents": step.get("dependencies", []),
            "children": [],  # not used in complexity computation
        }
        nodes.append(node)
    return nodes


def main(args, output_name: str = "hotpotqa_values.v1.csv"):
    """
    Main function to load multi-objective HotpotQA data, compute metrics,
    and write per-instance results to CSV.
    """
    # Load dataset
    data = load_multiobj_hotpotqa_data(args.data_dir)

    if not data:
        print(f"Error: No data found in {os.path.join(args.data_dir, 'processed')}")
        return

    print(f"Loaded {len(data)} examples")

    # Prepare output directory
    output_dir = os.path.join(args.data_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

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

    for i, item in tqdm(enumerate(data), desc="Processing instances", total=len(data)):
        question = item.get("question", "")
        dag = item.get("dag", [])
        metadata = item.get("metadata", {})

        # Skip items with missing data
        if not question or not dag:
            print(
                f"Skipping instance {i}: missing required fields (question, dag)"
            )
            continue

        input_len = compute_input_length(question)
        workflow_len = compute_workflow_length(dag)
        k_value = get_question_complexity(metadata)
        component_types = sorted(get_component_types(metadata))

        # Track min/max items
        if input_len < min_input_len:
            min_input_len = input_len
            min_input_item = item
        if input_len > max_input_len:
            max_input_len = input_len
            max_input_item = item

        if workflow_len < min_workflow_len:
            min_workflow_len = workflow_len
            min_workflow_item = item
        if workflow_len > max_workflow_len:
            max_workflow_len = workflow_len
            max_workflow_item = item

        # Compute DAG complexity
        nodes = format_dag_for_complexity(dag)
        dag_complexity = compute_dag_complexity(nodes, ignore_input_node=False)

        row = {
            "id": item["id"],
            "dataset": "multiobj_hotpotqa",
            "split": item.get("split", "unknown"),
            "k": k_value,
            "component_types": ",".join(component_types),
            "input_len": input_len,
            "workflow_len": workflow_len,
        }
        if dag_complexity:
            row.update(dag_complexity)
        rows.append(row)

    if not rows:
        print("Error: No valid instances to process")
        return

    # Write results to CSV
    print(f"Writing {len(rows)} rows to {output_path}")
    df = pd.DataFrame.from_records(rows)
    df.to_csv(output_path, index=False)

    # Show summary statistics
    arr_input = np.array([row["input_len"] for row in rows])
    arr_workflow = np.array([row["workflow_len"] for row in rows])

    print("\n" + "=" * 80)
    print("MULTI-OBJECTIVE HOTPOTQA TASK CHARACTERIZATION METRICS")
    print("=" * 80)
    print("Dataset: multiobj_hotpotqa")
    print(f"Total Count: {len(rows)}")

    # Count by split
    for split in set(row.get("split", "unknown") for row in rows):
        count = sum(1 for row in rows if row.get("split") == split)
        print(f"  {split}: {count}")

    # Count by k value
    print("\nDistribution by k (number of sub-questions):")
    for k in sorted(set(row.get("k", 1) for row in rows)):
        count = sum(1 for row in rows if row.get("k") == k)
        print(f"  k={k}: {count}")

    print("\n--- INPUT LENGTH STATISTICS ---")
    print(
        f"Input Length - Min: {arr_input.min()} Max: {arr_input.max()} "
        f"Mean: {arr_input.mean():.2f} Median: {np.median(arr_input):.2f} "
        f"Std: {arr_input.std():.2f}"
    )

    print("\n--- WORKFLOW LENGTH STATISTICS (DAG) ---")
    print(
        f"Workflow Length (DAG) - Min: {arr_workflow.min()} Max: {arr_workflow.max()} "
        f"Mean: {arr_workflow.mean():.2f} Median: {np.median(arr_workflow):.2f} "
        f"Std: {arr_workflow.std():.2f}"
    )

    # Statistics by k value
    print("\n--- STATISTICS BY K VALUE ---")
    for k in sorted(set(row.get("k", 1) for row in rows)):
        k_rows = [row for row in rows if row.get("k") == k]
        k_input = np.array([row["input_len"] for row in k_rows])
        k_workflow = np.array([row["workflow_len"] for row in k_rows])
        print(f"\nk={k} (n={len(k_rows)}):")
        print(
            f"  Input Length - Mean: {k_input.mean():.2f} "
            f"Median: {np.median(k_input):.2f}"
        )
        print(
            f"  Workflow Length - Mean: {k_workflow.mean():.2f} "
            f"Median: {np.median(k_workflow):.2f}"
        )

    print("\n--- EXAMPLE INSTANCES ---")
    if min_input_item:
        print(f"\nMin Input Length Question ({min_input_len} tokens):")
        print(f"  {min_input_item['question'][:150]}...")
    if max_input_item:
        print(f"\nMax Input Length Question ({max_input_len} tokens):")
        print(f"  {max_input_item['question'][:150]}...")
    if min_workflow_item:
        min_k = min_workflow_item.get("metadata", {}).get("k", 1)
        print(
            f"\nMin Workflow Length (k={min_k}, {min_workflow_len} DAG steps):"
        )
        print(f"  Question: {min_workflow_item['question'][:150]}...")
    if max_workflow_item:
        max_k = max_workflow_item.get("metadata", {}).get("k", 1)
        print(
            f"\nMax Workflow Length (k={max_k}, {max_workflow_len} DAG steps):"
        )
        print(f"  Question: {max_workflow_item['question'][:150]}...")

    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute task characterization metrics for multi-objective HotpotQA dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/multiobj_hotpotqa",
        help="Directory containing processed/ subdirectory with train.v1.annotated.json and test.v1.annotated.json",
    )
    args = parser.parse_args()

    main(args)
