"""
Utility functions for OpenAI Batch API operations.

This module provides helper functions for batch processing of LLM requests,
including parameter validation, output transformation, and metadata management.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json


def validate_model_parameters(model: str, temperature: Optional[float]) -> dict[str, Any]:
    """
    Validate model parameters and return valid parameters dict.

    Args:
        model: Model name (e.g., "gpt-4o-mini-2024-07-18")
        temperature: Optional temperature parameter

    Returns:
        dict: Valid parameters to include in batch request body

    Raises:
        ValueError: If temperature is provided for gpt-5 models
    """
    parameters = {}

    # Check if temperature is compatible with the model
    if temperature is not None:
        if model.startswith("gpt-5"):
            raise ValueError(
                f"temperature parameter is not supported for {model}. "
                "Please remove --temperature flag for gpt-5 models."
            )
        parameters["temperature"] = temperature

    return parameters


def build_output_directory(
    dataset_id: str,
    subset: Optional[str],
    split: str,
    model: str,
    timestamp: Optional[str] = None,
) -> Path:
    """
    Build output directory path following the project structure.

    Directory structure:
    - With subset and split: data/<dataset_name>/llm_baseline_batches/<subset>/<split>/<model>.<timestamp>/
    - With subset only: data/<dataset_name>/llm_baseline_batches/<subset>/<model>.<timestamp>/
    - With split only: data/<dataset_name>/llm_baseline_batches/<split>/<model>.<timestamp>/
    - Neither: data/<dataset_name>/llm_baseline_batches/<model>.<timestamp>/

    Args:
        dataset_id: Dataset identifier (e.g., "drt/kqa_pro", "Solaris99/AgentBank")
        subset: Optional subset name (e.g., "gsm8k" for AgentBank)
        split: Dataset split (e.g., "train", "val")
        model: Model name (e.g., "gpt-4o-mini-2024-07-18")
        timestamp: Optional timestamp string (defaults to current time)

    Returns:
        Path: Output directory path
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Extract dataset name from dataset_id
    if "/" in dataset_id:
        dataset_name = dataset_id.split("/")[1].lower()
    else:
        dataset_name = dataset_id.lower()

    # Build base path
    base_path = Path("data") / dataset_name / "llm_baseline_batches"

    # Add subset if provided
    if subset:
        base_path = base_path / subset

    # Add split if provided
    if split:
        base_path = base_path / split

    # Add model and timestamp
    output_dir = base_path / f"{model}.{timestamp}"

    return output_dir


def load_batch_metadata(batch_dir: Path) -> dict[str, Any]:
    """
    Load batch metadata from batch_info.json.

    Args:
        batch_dir: Path to batch directory

    Returns:
        dict: Batch metadata

    Raises:
        FileNotFoundError: If batch_info.json does not exist
    """
    metadata_path = batch_dir / "batch_info.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"batch_info.json not found in {batch_dir}. "
            "Make sure you're pointing to the correct batch directory."
        )

    with open(metadata_path, "r") as f:
        return json.load(f)


def save_batch_metadata(batch_dir: Path, metadata: dict[str, Any]) -> None:
    """
    Save batch metadata to batch_info.json.

    Args:
        batch_dir: Path to batch directory
        metadata: Batch metadata dictionary
    """
    metadata_path = batch_dir / "batch_info.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def transform_batch_output(
    batch_result: dict[str, Any],
    record: dict[str, Any],
    episode_index: int,
    model: str,
) -> dict[str, Any]:
    """
    Transform batch API output to match run_llm.py result format.

    Args:
        batch_result: Single line from batch output file
        record: Original dataset record with 'id', 'query', 'answer'
        episode_index: Episode index in the dataset
        model: Model name used

    Returns:
        dict: Transformed result matching run_llm.py format
    """
    custom_id = batch_result["custom_id"]
    response = batch_result.get("response")
    error = batch_result.get("error")

    # Base result structure
    result = {
        "episode_index": episode_index,
        "id": custom_id,
        "query": record["query"],
        "expected_answer": record["answer"],
        "output": None,
        "metadata": {
            "success": False,
            "runtime": 0,  # Not available for batch API
            "model": model,
        },
    }

    if error:
        # Request failed
        result["metadata"]["error"] = error.get("message", str(error))
        result["metadata"]["error_type"] = error.get("code", "unknown")
    elif response and response.get("status_code") == 200:
        # Request succeeded
        body = response.get("body", {})

        # Extract output text from response
        output_items = body.get("output", [])
        output_texts = []
        for item in output_items:
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        output_texts.append(content.get("text", ""))

        result["output"] = "\n".join(output_texts) if output_texts else None
        result["metadata"]["success"] = True
        result["metadata"]["request_id"] = response.get("request_id")

        # Extract token usage if available
        usage = body.get("usage")
        if usage:
            result["metadata"]["token_usage"] = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            }
    else:
        # Unexpected response format
        result["metadata"]["error"] = f"Unexpected response: {response}"
        result["metadata"]["error_type"] = "unexpected_response"

    return result
