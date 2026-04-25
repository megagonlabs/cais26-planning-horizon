"""
compute_metrics_kqa_pro.py

Main script to load KQA Pro dataset, compute input length metric, workflow length, and DAG complexity.
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


def load_kqa_pro_data(data_dir: str) -> list[dict]:
    """
    Loads preprocessed KQA Pro data from local JSON files.
    Uses deduplicated DAG representations (train.v1.json and val.v1.json).

    Args:
        data_dir (str): Directory containing processed/ subdirectory

    Returns:
        list[dict]: Combined list of preprocessed KQA Pro examples with DAGs
    """
    processed_dir = os.path.join(data_dir, "processed")
    train_path = os.path.join(processed_dir, "train.v1.json")
    val_path = os.path.join(processed_dir, "val.v1.json")

    data = []
    with open(train_path, "r") as f:
        for item in json.load(f):
            item["split"] = "train"
            data.append(item)
    with open(val_path, "r") as f:
        for item in json.load(f):
            item["split"] = "val"
            data.append(item)
    return data


def compute_input_length(question: str) -> int:
    """
    Computes the token length of the question using whitespace tokenization.
    Args:
        question (str): The question string.
    Returns:
        int: Token count of the question.
    """
    return len(question.split())


def compute_workflow_length(dag: list[dict]) -> int:
    """
    Computes the number of steps in the deduplicated DAG representation.

    Args:
        dag (list[dict]): List of DAG nodes (deduplicated program steps).

    Returns:
        int: Number of nodes in the deduplicated DAG.
    """
    return len(dag)


def compute_program_length(program: list[dict]) -> int:
    """
    Computes the number of steps in the original (non-deduplicated) program.

    Args:
        program (list[dict]): List of original program steps.

    Returns:
        int: Number of nodes in the original program.
    """
    return len(program)


def main(args, output_name: str = "kqa_pro_values.v1.csv"):
    """
    Main function to load KQA Pro data, compute metrics, and write per-instance results to CSV.
    """
    # Load dataset
    data = load_kqa_pro_data(args.data_dir)

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
    min_program_len = float("inf")
    max_program_len = 0
    max_program_item = None
    last_func_counts = defaultdict(int)

    for i, item in tqdm(enumerate(data), desc="Processing instances"):
        question = item["question"]
        dag = item["dag"]
        program = item["program"]
        input_len = compute_input_length(question)
        workflow_len = compute_workflow_length(dag)
        program_len = compute_program_length(program)

        # Track min/max items for input length
        if input_len < min_input_len:
            min_input_len = input_len
            min_input_item = item
        if input_len > max_input_len:
            max_input_len = input_len
            max_input_item = item

        # Track min/max items for workflow length (DAG)
        if workflow_len < min_workflow_len:
            min_workflow_len = workflow_len
            min_workflow_item = item
        if workflow_len > max_workflow_len:
            max_workflow_len = workflow_len
            max_workflow_item = item

        # Track min/max items for program length (original)
        if program_len < min_program_len:
            min_program_len = program_len
        if program_len > max_program_len:
            max_program_len = program_len
            max_program_item = item

        # Format DAG for DAG complexity computation
        nodes = []
        for j, step in enumerate(dag):
            node = {
                "index": j,
                "parents": step["dependencies"],
                "children": [],  # not used
            }
            nodes.append(node)

        # Compute DAG complexity
        dag_complexity = compute_dag_complexity(nodes, ignore_input_node=False)

        # Record final function (note: KQA Pro DAGs don't have 'finish' at the end)
        last_step = dag[-1]
        last_function = last_step["function"]
        last_func_counts[last_function] += 1

        # Prepare row for CSV
        rows.append(
            {
                "id": item["id"],
                "dataset": "kqa_pro",
                "input_len": input_len,
                "workflow_len": workflow_len,
                "program_len": program_len,
                "last_step": last_function,
            }
        )
        if dag_complexity:
            rows[-1].update(dag_complexity)

    # Write results to CSV
    print(f"Writing {len(rows)} rows to {output_path}")
    df = pd.DataFrame.from_records(rows)
    df.to_csv(output_path, index=False)

    # Show summary statistics
    arr_input = np.array([row["input_len"] for row in rows])
    arr_workflow = np.array([row["workflow_len"] for row in rows])
    arr_program = np.array([row["program_len"] for row in rows])

    print("\n" + "=" * 80)
    print("KQA Pro TASK CHARACTERIZATION METRICS")
    print("=" * 80)
    print("Dataset: kqa_pro (Preprocessed with deduplicated DAGs)")
    print(f"Count: {len(rows)}")

    print("\n--- INPUT LENGTH STATISTICS ---")
    print(
        f"Input Length - Min: {arr_input.min()} Max: {arr_input.max()} Mean: {arr_input.mean():.2f} Median: {np.median(arr_input):.2f} Std: {arr_input.std():.2f}"
    )

    print("\n--- WORKFLOW LENGTH STATISTICS (DEDUPLICATED DAG) ---")
    print(
        f"Workflow Length (DAG) - Min: {arr_workflow.min()} Max: {arr_workflow.max()} Mean: {arr_workflow.mean():.2f} Median: {np.median(arr_workflow):.2f} Std: {arr_workflow.std():.2f}"
    )

    print("\n--- PROGRAM LENGTH STATISTICS (ORIGINAL) ---")
    print(
        f"Program Length (Original) - Min: {arr_program.min()} Max: {arr_program.max()} Mean: {arr_program.mean():.2f} Median: {np.median(arr_program):.2f} Std: {arr_program.std():.2f}"
    )

    print("\n--- DAG COMPRESSION RATIO ---")
    compression_ratios = arr_program / arr_workflow
    print(
        f"Compression Ratio (Program / DAG) - Min: {compression_ratios.min():.2f}x Max: {compression_ratios.max():.2f}x Mean: {compression_ratios.mean():.2f}x Median: {np.median(compression_ratios):.2f}x"
    )

    print("\n--- LAST STEP FUNCTION FREQUENCIES ---")
    for func, count in sorted(
        last_func_counts.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {func}: {count}")

    print("\n--- EXAMPLE INSTANCES ---")
    if min_input_item:
        print(f"\nMin Input Length Question ({min_input_len} tokens):")
        print(f"  {min_input_item['question']}")
    if max_input_item:
        print(f"\nMax Input Length Question ({max_input_len} tokens):")
        print(f"  {max_input_item['question']}")
    if min_workflow_item:
        min_workflow_len_actual = compute_workflow_length(min_workflow_item["dag"])
        min_program_len_actual = compute_program_length(min_workflow_item["program"])
        print(
            f"\nMin Workflow Length Program ({min_workflow_len_actual} DAG steps vs {min_program_len_actual} original steps):"
        )
        print(f"  Program: {min_workflow_item['program']}")
    if max_workflow_item:
        max_workflow_len_actual = compute_workflow_length(max_workflow_item["dag"])
        max_program_len_actual = compute_program_length(max_workflow_item["program"])
        print(
            f"\nMax Workflow Length Program ({max_workflow_len_actual} DAG steps vs {max_program_len_actual} original steps):"
        )
        print(f"  Program: {max_workflow_item['program']}")
    if max_program_item:
        max_program_len_orig = compute_program_length(max_program_item["program"])
        max_dag_len = compute_workflow_length(max_program_item["dag"])
        compression = max_program_len_orig / max_dag_len
        print(
            f"\nMax Original Program Length ({max_program_len_orig} original steps vs {max_dag_len} DAG steps, {compression:.2f}x compression):"
        )
        print(f"  Program: {max_program_item['program']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute KQA Pro task metrics.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/kopl_kbqa/kqa_pro",
        help="Directory containing KQA Pro train.json and val.json",
    )
    args = parser.parse_args()

    main(args)
