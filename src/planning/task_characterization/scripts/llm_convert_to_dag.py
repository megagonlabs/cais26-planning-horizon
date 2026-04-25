"""
CLI: Convert AgentBank linear trajectories to DAGs using an LLM prompt.

This script loads trajectories from a selected AgentBank subset, builds a
system/user prompt from a yaml file, calls the OpenAI API in parallel to
annotate dependencies (with structured outputs), validates the returned DAGs,
and writes one JSON per example into the specified output directory.

Usage (via uv):
    uv run python -m planning.task_characterization.scripts.llm_convert_to_dag \
        --dataset gsm8k \
        --output_directory data/agentbank/processed/dags/auto.v1/gsm8k \
        --prompt src/planning/task_characterization/prompts/dag_conversion.v1.yaml \
        --model gpt-5-mini-2025-08-07 \
        --n_workers 50
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import argparse
import json

import yaml
from datasets import load_dataset as hf_load_dataset

from planning.services.openai import batch_completion
from planning.services.openai import OpenAIClient
from planning.task_characterization.dag_from_trajectory import (
    convert_trajectory_to_dag,
    validate_dag,
)


def load_agentbank_dataset(
    dataset_name: str,
    exclude_ids: Optional[set[str]] = None,
    include_ids: Optional[set[str]] = None,
) -> list[dict]:
    """
    Load and process trajectories from a specific AgentBank dataset split.

    Args:
        dataset_name (str): AgentBank subset name (e.g., "gsm8k").
        exclude_ids (Optional[set[str]]): IDs to exclude.
        include_ids (Optional[set[str]]): If provided, only include these IDs.

    Returns:
        list[dict]: Items with keys: "id", "query", "original_index",
            "original_trajectory", "dag" (pre-parsed nodes without deps), and "metadata".
    """
    print(f"Loading dataset: {dataset_name}")

    try:
        ds = hf_load_dataset("Solaris99/AgentBank", dataset_name)
        _data = ds["train"]
        # Reformat the trajectories
        data = [
            {
                "id": item["id"],
                "query": item["conversations"][0]["value"],
                "original_index": idx,
                "original_trajectory": item["conversations"],
            }
            for idx, item in enumerate(_data)
        ]

        # Filter the trajectories if IDs are provided
        if exclude_ids:
            num0 = len(data)
            data = [item for item in data if item["id"] not in exclude_ids]
            print(
                f"Excluded {num0 - len(data)} examples (#exclude_ids = {len(exclude_ids)})"
            )
        if include_ids:
            data = [item for item in data if item["id"] in include_ids]
            print(f"Included {len(data)} examples (#include_ids = {len(include_ids)})")

        print(f"Dataset {dataset_name} loaded with {len(data)} examples")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return []

    for traj in data:
        dag = convert_trajectory_to_dag(
            traj["original_trajectory"], include_dependencies=False
        )
        traj["dag"] = dag
        traj["metadata"] = {
            "num_steps": len(
                [turn for turn in traj["original_trajectory"] if turn["from"] == "gpt"]
            ),
            "num_actions": sum(
                1
                for turn in traj["original_trajectory"]
                if turn["from"] == "gpt" and "Action:" in turn["value"]
            ),
        }

    return data


def load_prompt_yaml(path_prompt: Path, dataset: str) -> tuple[str, str]:
    """
    Load system prompt and dataset-specific examples from YAML prompt file.

    Args:
        path_prompt (Path): Path to the prompt YAML file.
        dataset (str): Dataset key used to select examples.

    Returns:
        tuple[str, str]: (base_system_prompt, examples_text)
    """
    with open(path_prompt, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    base_prompt: str = y.get("base_prompt", "").strip()
    ex_map: dict[str, str] = y.get("examples", {}) or {}
    examples_text: str = ex_map.get(dataset, "").strip()
    return base_prompt, examples_text


def build_system_message(base_prompt: str, examples_text: str) -> str:
    """
    Render the system prompt by filling placeholders.

    Args:
        base_prompt (str): Prompt template containing {examples} and optionally {additional_instructions}.
        examples_text (str): Dataset-specific examples block.

    Returns:
        str: Rendered system message.
    """
    # Keep additional_instructions empty for now.
    return (
        base_prompt.replace("{examples}", examples_text)
        .replace("{additional_instructions}", "")
        .strip()
    )


def build_model_inputs(
    items: list[dict],
    system_message: str,
) -> list[dict]:
    """
    Construct model inputs for the API from trajectories and pre-nodes.

    Args:
        items (list[dict]): Items containing query, trajectory and pre-parsed nodes.
        system_message (str): Rendered system message.

    Returns:
        list[dict]: List of model input dictionaries for the API.
    """
    model_inputs: list[dict] = []
    count_single_turn_trajectories = 0
    for it in items:
        if it["metadata"]["num_actions"] == 1:
            # Skip single-turn trajectories
            count_single_turn_trajectories += 1
            continue
        user_content = "Query: {query}\n\n```json\n{dag}\n```".format(
            query=it["query"], dag=json.dumps(it["dag"], ensure_ascii=False)
        )
        model_inputs.append(
            {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content},
                ],
                "response_format": _build_response_format_schema(it["id"]),
            }
        )
    print(f"Skipped {count_single_turn_trajectories} single-turn trajectories.")
    print(f"Created a batch of {len(model_inputs)} messages.")
    return model_inputs


def _build_response_format_schema(example_id: str) -> dict[str, Any]:
    """
    Build a JSON Schema for OpenAI structured outputs representing a list of DAG nodes.

    Returns:
        dict[str, Any]: Response format payload compatible with OpenAI structured outputs.
    """
    node_props = {
        "index": {
            "type": "integer",
            "description": "Unique step index starting from 0.",
        },
        "action": {
            "type": "string",
            "description": "Type of action (e.g., math, search, finish).",
        },
        "args": {"type": "string", "description": "Arguments or input for the action."},
        "observation": {
            "type": "string",
            "description": "Output from the action.",
        },
        "parents": {
            "type": "array",
            "description": "Indices of steps this action depends on. -1 represents the input query. This field should not be empty.",
            "items": {"type": "integer"},
        },
    }
    schema = {
        "type": "array",
        "description": "List of DAG nodes to represent the reasoning process.",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["index", "action", "args", "observation", "parents"],
            "properties": node_props,
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "trajectory_dag_nodes",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "nodes"],
                "properties": {
                    "id": {
                        "type": "string",
                        "description": f"Unique identifier for the DAG. Use '{example_id}'",
                        "enum": [example_id],
                    },
                    "nodes": schema,
                },
            },
            "strict": True,
        },
    }


def main(args):
    """
    Convert linear trajectories to DAGs using an LLM and save per-item JSON files.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        None
    """
    openai_client = OpenAIClient()

    # Load excluded and included IDs if provided
    exclude_ids = set()
    include_ids = set()
    if args.exclude_ids:
        with open(args.exclude_ids) as f:
            exclude_ids = set(json.load(f)[args.dataset])

    if args.include_ids:
        with open(args.include_ids) as f:
            include_ids = set(json.load(f)[args.dataset])

    # Load trajectories from dataset
    data = load_agentbank_dataset(
        args.dataset, exclude_ids=exclude_ids, include_ids=include_ids
    )

    # Load the prompt file and render the system message
    base_prompt, examples_text = load_prompt_yaml(args.path_prompt, args.dataset)
    system_message = build_system_message(base_prompt, examples_text)

    # Ensure output directory exists
    out_dir: Path = args.dir_output
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build request batches (system+user per trajectory)
    model_inputs = build_model_inputs(data, system_message)

    # Call the API in parallel
    print(
        f"Submitting {len(model_inputs)} requests to model {args.model} with {args.n_workers} workers..."
    )
    # Common model parameters
    model_args = {
        "model": args.model,
        "temperature": 0,
        "seed": 42,
    }
    if "gpt-5" in args.model:
        model_args["temperature"] = 1.0
        model_args["reasoning_effort"] = "low"
    responses = batch_completion(
        client=openai_client,
        batch=model_inputs,
        n_workers=args.n_workers,
        show_progress=True,
        **model_args,
    )

    # Map responses back to items and save one file per item
    num_valid = 0
    num_tokens = {"prompt": 0, "completion": 0, "reasoning": 0}
    ## Convert responses -> dict[id, nodes]
    dags, token_usages, errors = {}, {}, {}
    for resp in responses:
        rid = None
        try:
            result = json.loads(resp.choices[0].message.content)  # type: ignore
            rid = result["id"]
            dags[rid] = result["nodes"]
        except Exception as e:
            if rid is not None:
                errors[rid] = str(e)

        # Record token usage
        usage = getattr(resp, "usage", None)
        usage_dict = None
        if usage is not None:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "reasoning_tokens": 0,
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
            if getattr(usage, "completion_tokens_details", None):
                usage_dict["reasoning_tokens"] = getattr(
                    usage.completion_tokens_details, "reasoning_tokens", 0
                )
            num_tokens["prompt"] += usage_dict["prompt_tokens"]
            num_tokens["completion"] += usage_dict["completion_tokens"]
            num_tokens["reasoning"] += usage_dict["reasoning_tokens"]
        token_usages[rid] = usage_dict

    for item in data:
        rid = item["id"]

        item.setdefault("metadata", {})
        item["metadata"].update(
            {
                "model": args.model,
                "timestamp": datetime.now().isoformat() + "Z",
                "usage": token_usages.get(rid, {}),
            }
        )
        # Initialize children and parents fields with linear dependencies
        for idx in range(len(item["dag"])):
            item["dag"][idx]["parents"] = [idx - 1]
            item["dag"][idx]["children"] = []

        is_valid = True
        if rid in errors:
            item["metadata"]["dag_generation_error"] = errors[rid]
        else:
            nodes = dags.get(rid, None)
            if nodes is not None:
                if len(nodes) != len(item["dag"]):
                    is_valid = False
                    item["metadata"]["dag_generation_error"] = "Length mismatch"
                else:
                    for idx, node in enumerate(nodes):
                        item["dag"][idx]["parents"] = node["parents"]

        # Update children based on parents
        for idx in range(len(item["dag"])):
            for parent in item["dag"][idx]["parents"]:
                if parent >= 0:
                    item["dag"][parent]["children"].append(idx)

        # Validate only if nodes (= LLM generation) are present
        if nodes is not None:
            is_valid, _ = validate_dag(item["dag"])
            item["metadata"]["dag_valid"] = is_valid
            if is_valid:
                num_valid += 1

        # Write per-id JSON
        per_path = out_dir / f"{rid}.json"
        with open(per_path, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=2)

    print(f"Completed. Valid DAGs: {num_valid}/{len(dags)}. Outputs in: {out_dir}")
    print(f"Total tokens used: {num_tokens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "mathqa", "hotpotqa", "strategyqa"],
        required=True,
        help="The dataset to use",
    )
    parser.add_argument(
        "--output_directory",
        dest="dir_output",
        type=Path,
        required=True,
        help="The directory to save the output files",
    )
    parser.add_argument(
        "--prompt",
        dest="path_prompt",
        type=Path,
        required=True,
        help="Path to the prompt file (.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="The model to use for conversion",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=100,
        help="Number of parallel workers for API calls",
    )
    parser.add_argument(
        "--exclude_ids", type=Path, help="Path to a JSON file containing IDs to exclude"
    )
    parser.add_argument(
        "--include_ids", type=Path, help="Path to a JSON file containing IDs to include"
    )
    args = parser.parse_args()
    main(args)
