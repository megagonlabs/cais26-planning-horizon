"""
Download and process batch validation results from OpenAI Batch API.

This script checks the status of a validation batch job, downloads completed results,
transforms them to include validation results in the original examples, and calculates
statistics.

Output Files:
- batch_output.jsonl: Raw batch output from OpenAI
- batch_errors.jsonl: Raw error output (if any errors)
- validated.jsonl: Original examples with validation results added
- validation_stats.json: Statistics about validation results

Usage:
    uv run python data/multiobj_hotpotqa/scripts/batch_validate_download.py \
        --batch-dir data/multiobj_hotpotqa/batch_validation/gpt-4.1-mini-2025-04-14/2025-12-06-14-30-00
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.multiobj_hotpotqa.scripts.utils import (  # noqa: E402
    INPUT_COST,
    OUTPUT_COST
)

def load_batch_metadata(batch_dir: Path) -> dict:
    """Load batch metadata from batch_info.json."""
    metadata_path = batch_dir / "batch_info.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"batch_info.json not found in {batch_dir}")

    with open(metadata_path, "r") as f:
        return json.load(f)


def load_jsonl(file_path: Path) -> list[dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_json(file_path: Path) -> list[dict]:
    """Load data from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: list[dict], file_path: Path):
    """Save data to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def parse_validation_result(response_text: str) -> tuple[bool, str]:
    """
    Parse validation result from structured output JSON.

    Args:
        response_text: Response text from API (JSON string)

    Returns:
        tuple: (is_valid: bool, reasoning: str)
    """
    try:
        result = json.loads(response_text)
        is_valid = result.get("is_valid", False)
        reasoning = result.get("reasoning", "")
        return is_valid, reasoning
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse validation response as JSON: {e}")
        print(f"Response text: {response_text[:200]}")
        return False, ""


def calculate_validation_stats(validated_examples: list[dict]) -> dict:
    """
    Calculate validation statistics.

    Args:
        validated_examples: List of validated examples

    Returns:
        dict: Validation statistics
    """
    total = len(validated_examples)
    valid_count = sum(
        1 for ex in validated_examples if ex.get("valid_reasoning_structure", False)
    )
    invalid_count = total - valid_count

    # Statistics by type
    type_stats = {}
    for example in validated_examples:
        qtype = example["type"]
        if qtype not in type_stats:
            type_stats[qtype] = {"total": 0, "valid": 0, "invalid": 0}

        type_stats[qtype]["total"] += 1
        if example.get("valid_reasoning_structure", False):
            type_stats[qtype]["valid"] += 1
        else:
            type_stats[qtype]["invalid"] += 1

    return {
        "total_examples": total,
        "valid_examples": valid_count,
        "invalid_examples": invalid_count,
        "valid_rate": valid_count / total if total > 0 else 0.0,
        "invalid_rate": invalid_count / total if total > 0 else 0.0,
        "by_type": type_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download and process batch validation results from OpenAI Batch API"
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        required=True,
        help="Path to batch directory containing batch_info.json",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(override=True)

    # Load batch metadata
    print(f"Loading batch metadata from: {args.batch_dir}")
    try:
        metadata = load_batch_metadata(args.batch_dir)
    except FileNotFoundError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

    model = metadata["model"]
    input_file = Path(metadata["input_file"])

    # Handle both old (single batch) and new (multiple batches) format
    if "batches" in metadata:
        batch_infos = metadata["batches"]
        num_chunks = len(batch_infos)
        print(f"Found {num_chunks} batch chunk(s)")
    else:
        # Legacy format with single batch
        batch_infos = [{
            "batch_id": metadata["batch_id"],
            "input_file_id": metadata.get("input_file_id"),
            "chunk_index": 0,
            "num_requests": metadata.get("num_requests", 0),
        }]
        num_chunks = 1

    print(f"Model: {model}")
    print(f"Input file: {input_file}")

    # Check status of all batches
    print("\nChecking batch status...")
    client = OpenAI()
    batches = []
    all_completed = True

    for batch_info in batch_infos:
        batch_id = batch_info["batch_id"]
        chunk_idx = batch_info["chunk_index"]

        try:
            batch = client.batches.retrieve(batch_id)
            batches.append(batch)

            if num_chunks > 1:
                print(f"\nChunk {chunk_idx}:")
            print(f"  Batch ID: {batch_id}")
            print(f"  Status: {batch.status}")
            print(f"  Total: {batch.request_counts.total}")
            print(f"  Completed: {batch.request_counts.completed}")
            print(f"  Failed: {batch.request_counts.failed}")

            if batch.status != "completed":
                all_completed = False
        except Exception as e:
            sys.stderr.write(f"Error retrieving batch {batch_id}: {e}\n")
            sys.exit(1)

    # Check if all batches are completed
    if not all_completed:
        print("\nNot all batches are completed yet.")
        print("Please run this script again after all batches complete.")
        sys.exit(0)

    # Download output files from all batches
    print("\nDownloading batch outputs...")
    all_batch_outputs = []
    all_error_results = []

    for batch_info, batch in zip(batch_infos, batches):
        chunk_idx = batch_info["chunk_index"]

        if not batch.output_file_id:
            sys.stderr.write(f"Error: Batch chunk {chunk_idx} has no output file.\n")
            continue

        try:
            output_content = client.files.content(batch.output_file_id)
            if num_chunks > 1:
                output_path = args.batch_dir / f"batch_output_chunk_{chunk_idx}.jsonl"
            else:
                output_path = args.batch_dir / "batch_output.jsonl"

            with open(output_path, "wb") as f:
                f.write(output_content.content)
            print(f"Saved batch output to: {output_path}")

            # Load and collect outputs
            chunk_outputs = load_jsonl(output_path)
            all_batch_outputs.extend(chunk_outputs)
        except Exception as e:
            sys.stderr.write(f"Error downloading output file for chunk {chunk_idx}: {e}\n")
            continue

        # Download error file if exists
        if batch.error_file_id:
            try:
                error_content = client.files.content(batch.error_file_id)
                if num_chunks > 1:
                    error_path = args.batch_dir / f"batch_errors_chunk_{chunk_idx}.jsonl"
                else:
                    error_path = args.batch_dir / "batch_errors.jsonl"

                with open(error_path, "wb") as f:
                    f.write(error_content.content)
                print(f"Saved batch errors to: {error_path}")

                # Load and collect errors
                chunk_errors = load_jsonl(error_path)
                all_error_results.extend(chunk_errors)
            except Exception as e:
                sys.stderr.write(f"Error downloading error file for chunk {chunk_idx}: {e}\n")

    # Calculate total token usage and cost
    total_input_tokens, total_output_tokens = 0, 0
    total_cost = 0
    for output in all_batch_outputs:
        usage = output["response"]["body"]["usage"]
        model = output["response"]["body"]["model"]
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        if model in INPUT_COST and model in OUTPUT_COST:
            total_cost += input_tokens * INPUT_COST[model] / 2.0  # 50% discount for batch
            total_cost += output_tokens * OUTPUT_COST[model] / 2.0  # 50% discount for batch

    # Load original examples (only bridge questions)
    print("\nLoading original bridge questions...")
    try:
        all_original_examples = load_json(input_file)
        original_examples = [ex for ex in all_original_examples if ex["type"] == "bridge"]
        print(f"Loaded {len(original_examples)} bridge questions (total: {len(all_original_examples)})")
    except Exception as e:
        sys.stderr.write(f"Error loading original examples: {e}\n")
        sys.exit(1)

    # Create mapping from custom_id to example
    example_map = {ex["_id"]: ex for ex in original_examples}

    # Process merged batch outputs
    print("\nProcessing validation results...")
    validated_examples = []

    for result in all_batch_outputs:
        custom_id = result["custom_id"]

        if custom_id not in example_map:
            sys.stderr.write(f"Warning: custom_id {custom_id} not found in original bridge questions\n")
            continue

        # Get original example
        example = example_map[custom_id].copy()

        # Parse validation result
        if result.get("response") and result["response"].get("status_code") == 200:
            response_body = result["response"]["body"]
            # Extract content from Chat Completions API response
            response_text = response_body["choices"][0]["message"]["content"]
            is_valid, reasoning = parse_validation_result(response_text)
            example["valid_reasoning_structure"] = is_valid
            example["validation_reasoning"] = reasoning
        else:
            # Mark as invalid if validation failed
            example["valid_reasoning_structure"] = False
            example["validation_reasoning"] = ""
            error_msg = result.get("error", {}).get("message", "Unknown error")
            print(f"Warning: Validation failed for {custom_id}: {error_msg}")

        validated_examples.append(example)

    # Handle errors from all chunks
    for error_result in all_error_results:
        custom_id = error_result["custom_id"]
        if custom_id in example_map and custom_id not in {ex["_id"] for ex in validated_examples}:
            example = example_map[custom_id].copy()
            example["valid_reasoning_structure"] = False
            example["validation_reasoning"] = ""
            validated_examples.append(example)
            print(f"Warning: Validation error for {custom_id}")

    # Sort by original order (assuming IDs maintain order or have indices)
    validated_examples.sort(key=lambda x: x["_id"])

    # Save validated examples
    validated_path = args.batch_dir / "validated.jsonl"
    save_jsonl(validated_examples, validated_path)
    print(f"\nSaved {len(validated_examples)} validated examples to: {validated_path}")

    # Calculate and save statistics
    print("\nCalculating validation statistics...")
    stats = calculate_validation_stats(validated_examples)

    stats_path = args.batch_dir / "validation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved validation statistics to: {stats_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH VALIDATION SUMMARY")
    print("=" * 60)
    if num_chunks > 1:
        print(f"Number of chunks: {num_chunks}")
        for batch_info in batch_infos:
            print(f"  Chunk {batch_info['chunk_index']}: {batch_info['batch_id']}")
    else:
        print(f"Batch ID: {batch_infos[0]['batch_id']}")
    print(f"Model: {model}")
    print(f"\nTotal bridge questions: {stats['total_examples']}")
    print(f"Valid: {stats['valid_examples']} ({stats['valid_rate']:.1%})")
    print(f"Invalid: {stats['invalid_examples']} ({stats['invalid_rate']:.1%})")
    print()
    # Show token usage and cost
    print(f"\nTotal input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    if total_cost > 0:
        print(f"Estimated total cost (with 50% batch discount): ${total_cost:.6f}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
