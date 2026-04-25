"""
compute_metrics_atomic_kbqa.py

Generic script to compute task characterization metrics for KBQA datasets (GrailQA, WebQSP, GraphQ, etc.)
Supports any KBQA dataset with preprocessed JSON files containing 'question', 'dag', and 'function_list' fields.
"""

from collections import defaultdict
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


def load_kbqa_data(data_dir: str, dataset_name: str) -> list[dict]:
    """
    Loads preprocessed KBQA data from local JSON files.
    Auto-detects dataset naming conventions (e.g., grailqa_*, webqsp_*, graphq_*).

    Args:
        data_dir (str): Directory containing processed/ subdirectory
        dataset_name (str): Name of dataset (e.g., 'grailqa', 'webqsp', 'graphq')

    Returns:
        list[dict]: Combined list of preprocessed KBQA examples with DAGs
    """
    processed_dir = os.path.join(data_dir, "processed")

    if not os.path.exists(processed_dir):
        raise ValueError(f"processed/ directory not found: {processed_dir}")

    data = []

    # Find all v1.json files matching the dataset prefix
    for filename in sorted(os.listdir(processed_dir)):
        if filename.endswith(".v1.json") and filename.startswith(dataset_name):
            # Skip demonstration and heldout pool files
            if "50nn" in filename or "heldout" in filename:
                continue

            filepath = os.path.join(processed_dir, filename)

            # Extract split from filename (e.g., "grailqa_train.v1.json" -> "train")
            split_name = filename.replace(f"{dataset_name}_", "").replace(
                ".v1.json", ""
            )

            print(f"Loading {filename}...")
            with open(filepath, "r") as f:
                for item in json.load(f):
                    item["split"] = split_name
                    data.append(item)

    return data


def compute_input_length(question: str) -> int:
    """Computes token length of question using whitespace tokenization."""
    return len(question.split())


def compute_workflow_length(dag: list[dict]) -> int:
    """Computes number of steps in deduplicated DAG."""
    return len(dag)


def compute_function_list_length(function_list: list[str]) -> int:
    """Computes number of steps in original function list."""
    return len(function_list)


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


def main(args, output_name: str | None = None):
    """
    Main function to load KBQA data, compute metrics, and write per-instance results to CSV.
    """
    if output_name is None:
        output_name = f"{args.dataset}_values.v1.csv"

    # Load dataset
    data = load_kbqa_data(args.data_dir, args.dataset)

    if not data:
        print(f"Error: No data found in {os.path.join(args.data_dir, 'processed')}")
        print(
            f"Expected files matching pattern: {args.dataset}_*.v1.json (excluding *50nn* and *heldout*)"
        )
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
    min_function_len = float("inf")
    max_function_len = 0
    max_function_item = None
    last_func_counts = defaultdict(int)

    for i, item in tqdm(enumerate(data), desc="Processing instances", total=len(data)):
        question = item.get("question", "")
        dag = item.get("dag", [])
        function_list = item.get("function_list", [])

        # Skip items with missing data
        if not question or not dag or not function_list:
            print(
                f"Skipping instance {i}: missing required fields (question, dag, function_list)"
            )
            continue

        input_len = compute_input_length(question)
        workflow_len = compute_workflow_length(dag)
        function_len = compute_function_list_length(function_list)

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

        if function_len < min_function_len:
            min_function_len = function_len
        if function_len > max_function_len:
            max_function_len = function_len
            max_function_item = item

        # Compute DAG complexity
        nodes = format_dag_for_complexity(dag)
        dag_complexity = compute_dag_complexity(nodes, ignore_input_node=False)

        # Record final function before 'finish' step
        last_step = dag[-1]
        assert last_step["function"] == "finish"
        assert (
            len(last_step["dependencies"]) == 1
        )  # This should be True for atomic KBQA DAGs
        last_function = dag[last_step["dependencies"][0]]["function"]
        last_func_counts[last_function] += 1

        # Prepare row for CSV
        row = {
            "id": item["id"],
            "dataset": args.dataset,
            "split": item["split"],
            "input_len": input_len,
            "workflow_len": workflow_len,
            "function_len": function_len,
            "last_step": last_function,
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
    arr_function = np.array([row["function_len"] for row in rows])

    print("\n" + "=" * 80)
    print(f"{args.dataset.upper()} TASK CHARACTERIZATION METRICS")
    print("=" * 80)
    print(f"Dataset: {args.dataset.upper()} (Preprocessed with deduplicated DAGs)")
    print(f"Total Count: {len(rows)}")

    # Count by split
    for split in set(row.get("split", "unknown") for row in rows):
        count = sum(1 for row in rows if row.get("split") == split)
        print(f"  {split}: {count}")

    print("\n--- INPUT LENGTH STATISTICS ---")
    print(
        f"Input Length - Min: {arr_input.min()} Max: {arr_input.max()} Mean: {arr_input.mean():.2f} Median: {np.median(arr_input):.2f} Std: {arr_input.std():.2f}"
    )

    print("\n--- FUNCTION LIST LENGTH STATISTICS---")
    print(
        f"Function List Length - Min: {arr_function.min()} Max: {arr_function.max()} Mean: {arr_function.mean():.2f} Median: {np.median(arr_function):.2f} Std: {arr_function.std():.2f}"
    )

    print("\n--- LAST STEP FUNCTION FREQUENCIES ---")
    for func, count in sorted(
        last_func_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {func}: {count}")

    print("\n--- EXAMPLE INSTANCES ---")
    if min_input_item:
        print(f"\nMin Input Length Question ({min_input_len} tokens):")
        print(f"  {min_input_item['question'][:100]}...")
    if max_input_item:
        print(f"\nMax Input Length Question ({max_input_len} tokens):")
        print(f"  {max_input_item['question'][:100]}...")
    if min_workflow_item:
        min_workflow_len_actual = compute_workflow_length(min_workflow_item["dag"])
        min_function_len_actual = compute_function_list_length(
            min_workflow_item["function_list"]
        )
        print(
            f"\nMin Workflow Length Query ({min_workflow_len_actual} DAG steps vs {min_function_len_actual} function steps):"
        )
        print(f"  First 3 steps: {min_workflow_item['function_list'][:3]}")
    if max_workflow_item:
        max_workflow_len_actual = compute_workflow_length(max_workflow_item["dag"])
        max_function_len_actual = compute_function_list_length(
            max_workflow_item["function_list"]
        )
        print(
            f"\nMax Workflow Length Query ({max_workflow_len_actual} DAG steps vs {max_function_len_actual} function steps):"
        )
        print(f"  First 3 steps: {max_workflow_item['function_list'][:3]}")
    if max_function_item:
        max_function_len_orig = compute_function_list_length(
            max_function_item["function_list"]
        )
        max_dag_len = compute_workflow_length(max_function_item["dag"])
        compression = max_function_len_orig / max_dag_len if max_dag_len > 0 else 1.0
        print(
            f"\nMax Function List Length ({max_function_len_orig} function steps vs {max_dag_len} DAG steps, {compression:.2f}x compression):"
        )
        print(f"  First 3 steps: {max_function_item['function_list'][:3]}")

    print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute task characterization metrics for KBQA datasets (GrailQA, WebQSP, GraphQ, etc.)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="grailqa",
        help="Dataset name (e.g., 'grailqa', 'webqsp', 'graphq')",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing processed/ subdirectory. Defaults to data/atomic_kbqa/{dataset}",
    )
    args = parser.parse_args()

    # Auto-set data_dir if not provided
    if args.data_dir is None:
        args.data_dir = f"data/atomic_kbqa/{args.dataset}"

    main(args)
