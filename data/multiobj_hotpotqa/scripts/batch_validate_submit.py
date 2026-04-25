"""
Submit batch validation requests to OpenAI Batch API for HotpotQA structure validation.

This script creates and submits batch requests to validate whether HotpotQA examples
follow expected reasoning structures (bridge or comparison). It handles dataset loading,
request generation, file upload, and batch creation.

Output Structure:
- Creates timestamped batch directory
- requests.jsonl: Batch request file in OpenAI Batch API format
- batch_info.json: Batch metadata including batch ID and parameters

Usage:
    # Validate all training examples
    uv run python data/multiobj_hotpotqa/scripts/batch_validate_submit.py \
        --input data/multiobj_hotpotqa/hotpot_train_v1.1.json \
        --output data/multiobj_hotpotqa/batch_validation \
        --model gpt-4.1-mini-2025-04-14

    # Dry run to estimate costs
    uv run python data/multiobj_hotpotqa/scripts/batch_validate_submit.py \
        --input data/multiobj_hotpotqa/hotpot_train_v1.1.json \
        --output data/multiobj_hotpotqa/batch_validation \
        --dry-run
"""

from datetime import datetime
from pathlib import Path
import argparse
import json
import sys

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import tiktoken
import orjson

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.multiobj_hotpotqa.scripts.utils import (  # noqa: E402
    INPUT_COST,
    load_json,
    filter_by_type,
    create_validation_prompt,
    get_validation_response_format,
    get_api_params,
)

# Batch API limits
MAX_BATCH_SIZE = 50000
CHUNK_SIZE = 40000  # Split into chunks of 40K to be safe


def create_batch_requests(
    records: list[dict],
    model: str,
    show_cost: bool = True,
) -> tuple[list[dict], int]:
    """
    Create batch request records in OpenAI Batch API format.

    Args:
        records: HotpotQA examples with '_id', 'answer', 'question', 'supporting_facts', 'context', 'type', 'level'
        model: Model name
        show_cost: Whether to print token count and cost info

    Returns:
        tuple: (batch_requests, total_input_tokens)
    """
    encoding = tiktoken.encoding_for_model(model)
    total_input_tokens = 0
    batch_requests = []

    for record in tqdm(records, desc="Creating batch requests"):
        # Create validation prompt
        sys_msg, usr_msg = create_validation_prompt(record)
        total_input_tokens += len(encoding.encode(sys_msg))
        total_input_tokens += len(encoding.encode(usr_msg))

        # Build request body for Chat Completions API
        params = get_api_params(
            model,
            max_completion_tokens=512,
            response_format=get_validation_response_format(),
        )
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg},
            ],
            **params,
        }

        # Create batch request
        batch_request = {
            "custom_id": record["_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
        batch_requests.append(batch_request)

    # Print token usage and estimated cost (50% discount for batch)
    if show_cost:
        print(f"\nTotal input tokens: {total_input_tokens}")
        if model in INPUT_COST:
            estimated_cost = total_input_tokens * INPUT_COST[model] / 2.0
            print(
                f"Estimated input cost (@{INPUT_COST[model] * 1_000_000 / 2.0:.3f} per million tokens): ${estimated_cost:.6f}"
            )

    return batch_requests, total_input_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Submit batch validation requests to OpenAI Batch API for HotpotQA"
    )
    parser.add_argument(
        "--input",
        type=Path,
        choices=[
            Path("data/multiobj_hotpotqa/hotpot_train_v1.1.json"),
            Path("data/multiobj_hotpotqa/hotpot_dev_distractor_v1.json"),
        ],
        required=True,
        help="Path to input JSON file with HotpotQA examples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for batch validation results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini-2025-04-14",
        help="Model name (default: gpt-4.1-mini-2025-04-14)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="If set, do not submit the batch"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(override=True)

    # Load dataset
    print(f"Loading dataset from: {args.input}")
    try:
        all_records = load_json(args.input)
    except Exception as e:
        sys.stderr.write(f"Error loading dataset: {e}\n")
        sys.exit(1)

    print(f"Loaded {len(all_records)} records")

    # Filter only bridge questions (comparison questions don't need validation)
    records = filter_by_type(all_records, "bridge")
    print(
        f"Filtered to {len(records)} bridge questions (skipped {len(all_records) - len(records)} comparison questions)"
    )

    if len(records) == 0:
        print("No bridge questions found. Nothing to validate.")
        sys.exit(0)

    # Determine if we need to chunk the dataset
    need_chunking = len(records) > CHUNK_SIZE
    if need_chunking:
        chunks = [
            records[i : i + CHUNK_SIZE] for i in range(0, len(records), CHUNK_SIZE)
        ]
        print(
            f"\nDataset has {len(records)} bridge questions, splitting into {len(chunks)} chunks of up to {CHUNK_SIZE}..."
        )
    else:
        chunks = [records]

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = args.output / args.model / timestamp
    print(f"\nOutput directory: {output_dir}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Created output directory")

    # Process each chunk
    batch_infos = []
    total_input_tokens_all_chunks = 0
    client = OpenAI() if not args.dry_run else None

    for chunk_idx, chunk in enumerate(chunks):
        if need_chunking:
            print(f"\n{'=' * 60}")
            print(
                f"Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} records)"
            )
            print(f"{'=' * 60}")

        # Create batch requests
        print(f"\nGenerating batch requests for chunk {chunk_idx + 1}...")
        batch_requests, chunk_input_tokens = create_batch_requests(
            chunk, args.model, show_cost=need_chunking
        )
        total_input_tokens_all_chunks += chunk_input_tokens

        if args.dry_run:
            print(
                f"Dry run: Generated {len(batch_requests)} requests for chunk {chunk_idx + 1}"
            )
            continue

        assert client is not None  # For type checker

        # Write batch requests to file
        if need_chunking:
            requests_path = output_dir / f"requests_chunk_{chunk_idx}.jsonl"
        else:
            requests_path = output_dir / "requests.jsonl"

        with open(requests_path, "w") as f:
            for request in batch_requests:
                f.write(json.dumps(request) + "\n")
        print(f"Saved {len(batch_requests)} requests to: {requests_path}")

        # Upload batch file to OpenAI
        print("\nUploading batch file to OpenAI...")
        try:
            with open(requests_path, "rb") as f:
                batch_file = client.files.create(file=f, purpose="batch")
            print(f"Uploaded file: {batch_file.id}")
        except Exception as e:
            sys.stderr.write(f"Error uploading file for chunk {chunk_idx + 1}: {e}\n")
            sys.exit(1)

        # Create batch
        print("\nCreating batch...")
        try:
            batch = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            print(f"Created batch: {batch.id}")
            print(f"Status: {batch.status}")
        except Exception as e:
            sys.stderr.write(f"Error creating batch for chunk {chunk_idx + 1}: {e}\n")
            sys.exit(1)

        # Store batch info
        batch_info = {
            "batch_id": batch.id,
            "input_file_id": batch_file.id,
            "chunk_index": chunk_idx,
            "num_requests": len(batch_requests),
        }
        batch_infos.append(batch_info)

    if args.dry_run:
        print(
            f"\nDry run complete. Generated {len(records)} requests in {len(chunks)} chunk(s)."
        )
        print(f"\nTotal input tokens: {total_input_tokens_all_chunks}")
        if args.model in INPUT_COST:
            total_estimated_cost = (
                total_input_tokens_all_chunks * INPUT_COST[args.model]
            )
            print(
                f"Total estimated input cost (@{INPUT_COST[args.model] * 1_000_000:.3f} per million tokens): ${total_estimated_cost:.6f}"
            )
        return

    # Save batch metadata
    if need_chunking:
        metadata = {
            "batches": batch_infos,
            "model": args.model,
            "timestamp": timestamp,
            "input_file": str(args.input),
            "total_requests": len(records),
            "num_chunks": len(chunks),
            "total_input_tokens": total_input_tokens_all_chunks,
        }
    else:
        # Single batch - use old format for compatibility
        metadata = {
            "batch_id": batch_infos[0]["batch_id"],
            "input_file_id": batch_infos[0]["input_file_id"],
            "model": args.model,
            "timestamp": timestamp,
            "input_file": str(args.input),
            "num_requests": len(batch_requests),
            "total_input_tokens": total_input_tokens_all_chunks,
        }

    metadata_path = output_dir / "batch_info.json"
    with open(metadata_path, "w") as f:
        f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2).decode("utf-8"))
    print(f"\nSaved batch metadata to: {metadata_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH SUBMISSION SUMMARY")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Input file: {args.input}")
    print(f"Total bridge questions: {len(records)}")
    if need_chunking:
        print(f"Number of chunks: {len(chunks)}")
        print("\nBatch IDs:")
        for info in batch_infos:
            print(
                f"  Chunk {info['chunk_index']}: {info['batch_id']} ({info['num_requests']} requests)"
            )
    else:
        print(f"Batch ID: {batch_infos[0]['batch_id']}")
        print(f"Input file ID: {batch_infos[0]['input_file_id']}")
    print(f"Total input tokens: {total_input_tokens_all_chunks}")
    if args.model in INPUT_COST:
        total_estimated_cost = total_input_tokens_all_chunks * INPUT_COST[args.model]
        print(
            f"Total estimated input cost (@{INPUT_COST[args.model] * 1_000_000:.3f} per million tokens): ${total_estimated_cost:.6f}"
        )
    print(f"\nOutput directory: {output_dir}")
    print("=" * 60)
    print("\nTo download results after completion, run:")
    print(
        f"  uv run python data/multiobj_hotpotqa/scripts/batch_validate_download.py --batch-dir {output_dir}"
    )


if __name__ == "__main__":
    main()
