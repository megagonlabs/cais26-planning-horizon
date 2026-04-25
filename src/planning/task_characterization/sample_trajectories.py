"""
sample_trajectories.py

Script to sample trajectories from AgentBank datasets for manual DAG creation.
Creates individual JSON files for each sampled trajectory following the specification:
data/agentbank/processed/dags/gold.v1/<dataset>/<id>.json
"""

import argparse
import json
import os
import random

from datasets import load_dataset

from planning.task_characterization.dag_from_trajectory import convert_trajectory_to_dag


def sample_trajectories_from_dataset(dataset_name: str, num_samples: int = 2, seed: int = 42) -> list[dict]:
    """
    Sample trajectories from a specific AgentBank dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., "gsm8k", "math", etc.)
        num_samples (int): Number of samples to take
        seed (int): Random seed for reproducibility

    Returns:
        list[dict]: List of sampled trajectory dictionaries
    """
    print(f"Loading dataset: {dataset_name}")

    try:
        ds = load_dataset("Solaris99/AgentBank", dataset_name)
        data = ds["train"]

        print(f"Dataset {dataset_name} loaded with {len(data)} examples")

        # Set random seed for reproducibility
        random.seed(seed)

        # Sample indices
        total_examples = len(data)
        if num_samples > total_examples:
            print(f"Warning: Requested {num_samples} samples but only {total_examples} available")
            num_samples = total_examples

        sampled_indices = random.sample(range(total_examples), num_samples)

        # Extract sampled examples
        sampled_trajectories = []
        for idx in sampled_indices:
            trajectory = {
                "original_index": idx,
                "id": data[idx]["id"],
                "conversations": data[idx]["conversations"]
            }
            sampled_trajectories.append(trajectory)

        print(f"Sampled {len(sampled_trajectories)} trajectories from {dataset_name}")
        return sampled_trajectories

    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return []


def save_samples_to_individual_files(trajectories: list[dict], dataset_name: str, output_dir: str):
    """
    Save each sampled trajectory to an individual JSON file with the following format:
    {
        "id": <trajectory_id>,
        "query": <query>,"
        "original_index": <index in dataset>,
        "original_trajectory": <list of conversation turns>,
        "dag": <generated DAG structure>,  # Now automatically generated
        "metadata": {
            "num_steps": <number of GPT turns>,
            "num_actions": <number of action steps>,
            "created_for_manual_annotation": true,
            "dag_auto_generated": true
        }
    }

    Files are saved to: data/agentbank/processed/dags/gold.v<version>/<dataset>/<id>.json

    Args:
        trajectories (list[Dict]): List of trajectory dictionaries
        dataset_name (str): Name of the dataset
        output_dir (str): Output directory path
        version (int): Version number for the gold set (default: 1)
    """
    def _get_output_dir(output_dir, version, dataset_name):
        return os.path.join(output_dir, "dags", f"gold.v{version}", dataset_name.lower())

    version = save_samples_to_individual_files.version if hasattr(save_samples_to_individual_files, 'version') else 1
    dataset_output_dir = _get_output_dir(output_dir, version, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    saved_files = []

    for trajectory in trajectories:
        trajectory_id = trajectory["id"]
        conversations = trajectory["conversations"]

        # Convert trajectory to DAG format
        dag = convert_trajectory_to_dag(conversations)

        file_data = {
            "id": trajectory_id,
            "query": conversations[0]["value"],
            "original_index": trajectory["original_index"],
            "original_trajectory": conversations,
            "dag": dag,  # Now contains the generated DAG structure
            "metadata": {
                "num_steps": len([turn for turn in conversations if turn["from"] == "gpt"]),
                "num_actions": sum(1 for turn in conversations
                                 if turn["from"] == "gpt" and "Action:" in turn["value"])
            }
        }

        output_path = os.path.join(dataset_output_dir, f"{trajectory_id}.json")
        # if the file already exists, skip
        if os.path.exists(output_path):
            print(f"File already exists, skipping: {output_path}")
            continue

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(file_data, f, indent=2, ensure_ascii=False)

        saved_files.append(output_path)

    print(f"Saved {len(trajectories)} individual files to {dataset_output_dir}")
    for file_path in saved_files:
        print(f"  - {file_path}")

    return saved_files


def create_in_context_examples_list(output_dir: str, datasets: list[str]):
    """
    Create the in-context.v<version>.json file listing the IDs of sampled trajectories
    for in-context examples as specified in the roadmap.

    Args:
        output_dir (str): Base output directory
        datasets (list[str]): List of dataset names
        version (int): Version number for the gold set (default: 1)
    """
    version = create_in_context_examples_list.version if hasattr(create_in_context_examples_list, 'version') else 1
    in_context_examples = {}

    for dataset_name in datasets:
        dataset_dir = os.path.join(output_dir, "dags", f"gold.v{version}", dataset_name.lower())
        if os.path.exists(dataset_dir):
            json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]
            example_ids = [os.path.splitext(f)[0] for f in json_files]
            in_context_examples[dataset_name] = example_ids

    in_context_file = os.path.join(output_dir, "dags", f"gold.v{version}", f"in-context.v{version}.json")
    os.makedirs(os.path.dirname(in_context_file), exist_ok=True)

    with open(in_context_file, 'w', encoding='utf-8') as f:
        json.dump(in_context_examples, f, indent=2, ensure_ascii=False)

    print(f"\nCreated in-context examples list: {in_context_file}")
    print("In-context examples:")
    for dataset, ids in in_context_examples.items():
        print(f"  {dataset}: {ids}")

    return in_context_file


def main(args):
    datasets = ["gsm8k", "math", "mathqa", "hotpotqa", "strategyqa"]
    output_dir = os.path.join("data", "agentbank", "processed")
    version = args.version

    # Set version for downstream functions
    save_samples_to_individual_files.version = version
    create_in_context_examples_list.version = version


    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")

        # Sample trajectories
        trajectories = sample_trajectories_from_dataset(
            dataset_name,
            num_samples=args.num_samples,
            seed=args.seed
        )

        if trajectories:
            # Show each extracted trajectory to stdout
            for i, traj in enumerate(trajectories):
                print(f"\n--- Extracted trajectory {i+1} for {dataset_name} ---")
                print(json.dumps(traj, indent=2, ensure_ascii=False))

            # Save to individual files following specification format
            save_samples_to_individual_files(trajectories, dataset_name, output_dir)

        print()

    # Create the in-context examples list after processing all datasets
    create_in_context_examples_list(output_dir, datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample trajectories from AgentBank datasets for manual DAG creation.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of samples per dataset (default: 2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Gold set version number (default: 1)"
    )

    args = parser.parse_args()
    main(args)
