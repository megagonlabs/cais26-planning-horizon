"""
compute_metrics.py

Main script to load AgentBank subset, compute input length metric, and show distribution.
"""


from collections import defaultdict
from pathlib import Path
from typing import Any, Optional
import argparse
import json
import os

from tqdm import tqdm
import numpy as np
import pandas as pd

from datasets import load_dataset
from planning.task_characterization.metrics_utils import compute_dag_complexity


def load_agentbank_subset(subset: str) -> list[dict]:
    """
    Loads the specified AgentBank subset from Huggingface Hub.
    Args:
        subset (str): Subset name (e.g., "gsm8k", "math", etc.)
    Returns:
        list[dict]: List of trajectory dicts.
    """
    ds = load_dataset("Solaris99/AgentBank", subset)
    return ds["train"]


def compute_input_length(conversation: list[dict]) -> int:
    """
    Computes the token length of the initial user input using whitespace tokenization.
    Args:
        conversation (list[dict]): List of conversation turns.
    Returns:
        int: Token count of the first user input.
    """
    for turn in conversation:
        if turn["from"] == "human":
            return len(turn["value"].split())
    return 0

def compute_workflow_length(conversation: list[dict]) -> int:
    """
    Computes the number of action steps in the trajectory.
    Args:
        conversation (list[dict]): List of conversation turns.
    Returns:
        int: Number of action steps ("Action" from "gpt").
    """
    count = 0
    for turn in conversation:
        if turn["from"] == "gpt" and "Action:" in turn["value"].strip():
            count += 1
    return count




def main(args):
    """
    Main function to load AgentBank subset, compute axis metrics, and write per-instance results to CSV.
    """
    # Load dataset
    data = load_agentbank_subset(args.subset)

    id2dag = {}
    if args.dag_dir:
        # Load DAG files from the specified directory
        for filename in os.listdir(args.dag_dir):
            if not filename.endswith(".json"):
                continue
            dag_path = os.path.join(args.dag_dir, filename)
            with open(dag_path, "r") as f:
                dag_data = json.load(f)
                id2dag[dag_data["id"]] = dag_data
        print(f"Loaded {len(id2dag)} DAGs from {args.dag_dir}")

    # Prepare output directory
    output_dir = os.path.join("data", "agentbank", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.subset}_values.v1.csv")

    # Compute metrics for each instance
    rows = []
    for item in tqdm(data, desc="Processing instances"):
        conversations = item["conversations"]
        input_len = compute_input_length(conversations)
        workflow_len = compute_workflow_length(conversations)
        dag_complexity = compute_dag_complexity(id2dag.get(item["id"], {}).get("dag", []), ignore_input_node=True)
        rows.append({
            "id": item.get("id", ""),
            "subset": args.subset,
            "input_len": input_len,
            "workflow_len": workflow_len,
        })
        if dag_complexity:
            rows[-1].update(dag_complexity)

    # Write results to CSV
    print(f"Writing {len(rows)} rows to {output_path}")
    df = pd.DataFrame.from_records(rows)
    df.to_csv(output_path, index=False)

    # Show summary statistics for input_len and workflow_len
    arr_input = np.array([row["input_len"] for row in rows])
    arr_workflow = np.array([row["workflow_len"] for row in rows])
    print(f"Subset: {args.subset}")
    print(f"Count: {len(rows)}")
    print(f"Input Length - Min: {arr_input.min()} Max: {arr_input.max()} Mean: {arr_input.mean():.2f} Median: {np.median(arr_input):.2f} Std: {arr_input.std():.2f}")
    print(f"Workflow Length - Min: {arr_workflow.min()} Max: {arr_workflow.max()} Mean: {arr_workflow.mean():.2f} Median: {np.median(arr_workflow):.2f} Std: {arr_workflow.std():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute AgentBank task metrics.")
    parser.add_argument("--subset", type=str, required=True, help="Subset name (e.g., gsm8k, math, etc.)")
    parser.add_argument("--dag-dir", type=Path, help="Path to a directory containing DAG files")
    args = parser.parse_args()

    main(args)
