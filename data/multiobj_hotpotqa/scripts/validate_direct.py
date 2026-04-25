"""
Direct validation of HotpotQA bridge questions using ThreadPoolExecutor.

This script validates a specified number of random bridge questions directly via
OpenAI API (not using Batch API). Useful for prompt tuning and quick validation.

Usage:
    # Validate 10 random bridge questions from train split
    uv run python data/multiobj_hotpotqa/scripts/validate_direct.py \
        --input data/multiobj_hotpotqa/hotpot_train_v1.1.json \
        --num-examples 10 \
        --seed 42

    # Validate 5 random bridge questions from dev split
    uv run python data/multiobj_hotpotqa/scripts/validate_direct.py \
        --input data/multiobj_hotpotqa/hotpot_dev_distractor_v1.json \
        --num-examples 5 \
        --seed 123
"""

from pathlib import Path
import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from planning.services.openai import OpenAIClient  # noqa: E402
from data.multiobj_hotpotqa.scripts.utils import (  # noqa: E402
    INPUT_COST,
    OUTPUT_COST,
    load_json,
    filter_by_type,
    create_validation_prompt,
    get_validation_response_format,
    get_api_params,
)


def validate_example(
    client: OpenAIClient, example: dict, model: str = "gpt-4.1-mini-2025-04-14"
) -> dict[str, Any]:
    """
    Validate a single HotpotQA bridge question.

    Args:
        client: OpenAI client
        example: HotpotQA example
        model: Model name

    Returns:
        dict: Result with example ID, question, validation result, and token counts
    """
    sys_msg, usr_msg = create_validation_prompt(example)

    try:
        params = get_api_params(model, response_format=get_validation_response_format())
        response = client.call(
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg},
            ],
            **params,
        )

        # Parse structured output
        output = response.choices[0].message.content
        if isinstance(output, str):
            import json
            output = json.loads(output)
        reasoning = output.get("reasoning", "")
        is_valid = output.get("is_valid", None)

        # Extract token usage
        token_usage = response.usage.to_dict()
        input_tokens = token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))
        output_tokens = token_usage.get("completion_tokens", token_usage.get("output_tokens", 0))

        return {
            "example_id": example["_id"],
            "question": example["question"],
            "result": "valid" if is_valid else "invalid",
            "reasoning": reasoning,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "error": None,
        }
    except Exception as e:
        # Defensive: try to get input_tokens if available
        input_tokens = 0
        try:
            token_usage = response.usage.to_dict()
            input_tokens = token_usage.get("prompt_tokens", token_usage.get("input_tokens", 0))
        except Exception:
            pass
        return {
            "example_id": example["_id"],
            "question": example["question"],
            "result": None,
            "reasoning": None,
            "input_tokens": input_tokens,
            "output_tokens": 0,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Directly validate HotpotQA bridge questions using ThreadPoolExecutor"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSON file with HotpotQA examples",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of random bridge questions to validate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini-2025-04-14",
        help="Model name (default: gpt-4.1-mini-2025-04-14)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file to save validation results (JSON)",
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv(override=True)

    # Load dataset
    print(f"Loading dataset from: {args.input}")
    try:
        all_records = load_json(args.input)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(all_records)} records")

    # Filter only bridge questions
    bridge_examples = filter_by_type(all_records, "bridge")
    print(f"Found {len(bridge_examples)} bridge questions")

    if len(bridge_examples) < args.num_examples:
        print(
            f"Warning: Requested {args.num_examples} examples but only {len(bridge_examples)} bridge questions available",
            file=sys.stderr,
        )
        num_examples = len(bridge_examples)
    else:
        num_examples = args.num_examples

    # Sample random examples with seed
    random.seed(args.seed)
    sampled_examples = random.sample(bridge_examples, num_examples)
    print(f"\nSampled {num_examples} random bridge questions (seed={args.seed})")

    # Validate using ThreadPoolExecutor or sequential processing
    print(f"\nValidating with {args.num_workers} workers...")
    client = OpenAIClient()
    results = []

    if args.num_workers == 1:
        # Sequential processing for single worker
        for idx, example in enumerate(sampled_examples, 1):
            result = validate_example(client, example, args.model)
            results.append(result)

            # Print progress
            if result["error"]:
                status = f"ERROR: {result['error']}"
            else:
                status = result["result"]

            print(f"[{idx}/{num_examples}] {result['example_id']}: {status}")
    else:
        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(validate_example, client, ex, args.model)
                for ex in sampled_examples
            ]

            for idx, future in enumerate(futures, 1):
                result = future.result()
                results.append(result)

                # Print progress
                if result["error"]:
                    status = f"ERROR: {result['error']}"
                else:
                    status = result["result"]

                print(f"[{idx}/{num_examples}] {result['example_id']}: {status}")

    # Print summary
    print(f"\n{'=' * 80}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 80}")

    valid_count = sum(1 for r in results if r["result"] == "valid")
    invalid_count = sum(1 for r in results if r["result"] == "invalid")
    error_count = sum(1 for r in results if r["error"])

    print(f"Total validated: {len(results)}")
    print(f"  Valid:   {valid_count} ({100 * valid_count / len(results):.1f}%)")
    print(f"  Invalid: {invalid_count} ({100 * invalid_count / len(results):.1f}%)")
    print(f"  Errors:  {error_count}")

    # Calculate and print token usage and costs
    total_input_tokens = sum(r["input_tokens"] for r in results)
    total_output_tokens = sum(r["output_tokens"] for r in results)
    total_tokens = total_input_tokens + total_output_tokens

    if args.model in INPUT_COST and args.model in OUTPUT_COST:
        input_cost = total_input_tokens * INPUT_COST[args.model]
        output_cost = total_output_tokens * OUTPUT_COST[args.model]
        total_cost = input_cost + output_cost

        print(f"\n{'=' * 80}")
        print("TOKEN USAGE AND COST")
        print(f"{'=' * 80}")
        print(f"Model: {args.model}")
        print(f"Input tokens:  {total_input_tokens:,} (@${INPUT_COST[args.model] * 1_000_000:.3f}/1M tokens)")
        print(f"Output tokens: {total_output_tokens:,} (@${OUTPUT_COST[args.model] * 1_000_000:.3f}/1M tokens)")
        print(f"Total tokens:  {total_tokens:,}")
        print(f"Input cost:    ${input_cost:.6f}")
        print(f"Output cost:   ${output_cost:.6f}")
        print(f"Total cost:    ${total_cost:.6f}")
    else:
        print(f"\n{'=' * 80}")
        print("TOKEN USAGE")
        print(f"{'=' * 80}")
        print(f"Model: {args.model}")
        print(f"Input tokens:  {total_input_tokens:,}")
        print(f"Output tokens: {total_output_tokens:,}")
        print(f"Total tokens:  {total_tokens:,}")
        print("Note: Cost calculation not available for this model")

    # Save results to file if specified
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n{'=' * 80}")
        print(f"Results saved to: {args.output}")

    client.close()


if __name__ == "__main__":
    main()
