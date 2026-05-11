"""
Script for evaluating system outputs against groundtruth answers using OpenAI's GPT models.

This script reads a JSONL file containing query, expected answer, and system output,
evaluates each entry for correctness using a specified GPT model, and updates the file
with evaluation results. It also computes and updates metrics on correctness.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import argparse
import ast
import json
import openai
import sys

from dotenv import load_dotenv
from tqdm import tqdm
import orjsonl

# Add project root and src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from planning.agents.kopl_utils import (  # noqa: E402
    is_kopl_entity_facts,
    is_kopl_entity,
    is_kopl_value_class,
    kopl_entity_facts_to_str,
    kopl_entity_to_str,
    kopl_value_class_to_str,
)
from scripts.utils.evaluation_utils import evaluate_answer  # noqa: E402
from scripts.utils.openai_utils import TOKEN_COSTS  # noqa: E402

load_dotenv(override=True)

# Version of the evaluation prompt used for tracking changes
PROMPT_VERSION = "2.0"

# System prompt instructing the GPT model to act as an impartial judge of correctness
SYSTEM_PROMPT_KBQA = """
You are a strict, impartial grader of answer correctness.

**Goal:** Compare System Output to Correct Answer only. Do not use outside knowledge; treat Correct Answer as ground truth.

**Output:** Reply with exactly one label (no quotes, no punctuation, no extra text):
- correct
- partially_correct
- incorrect
- refusal/unsure

**Scoring rules:**
1) Non-answer → refusal/unsure:
    - System Output refuses, asks for clarification, says "unknown"/"no information", or otherwise does not contain an answer value.

2) Exact match → correct:
    - System Output conveys the same answer value(s) as in Correct Answer, and no additional answer values.
    - Treat lists as sets: ignore order, casing, surrounding filler, and duplicate repetitions of the same correct value(s).
    - Ignore purely explanatory filler (e.g., "the answer is ...").

3) Partial overlap → partially_correct:
    - At least one expected answer value appears in the System Output, but it is missing any other required values and/or includes extra answer values not in the Correct Answer.

4) Mismatch → incorrect:
    - None of the expected answer values appear in the System Output, or any provided value contradicts the Correct Answer.

**Definitions and matching guidance:**
- Answer value: an entity (ID or name), number, date, or other atomic item. Treat comma/semicolon/newline/bulleted lists and clear conjunctions as multiple values.
- Entity matching: "m.xxxxx (Name)"/"Qxxxxx (Name)" matches if either the ID or the Name appears in System Output (case-insensitive).
- Numeric/date matching: require exact equality unless Correct Answer explicitly lists multiple acceptable values. Rounded/truncated numbers are incorrect when Correct Answer provides a more precise value.
- Type/Hierarchy: Do not give credit for broader/narrower/related categories, roles, or specializations (e.g., hypernym/hyponym). Only the exact expected value counts.
- Strict comparison: Do not use the question text or outside knowledge to infer equivalence. Compare the System Output strictly to the Correct Answer text.
- Extra information:
    - Extra explanatory text may be ignored.
    - Extra answer values (additional entities/values) make the result partially_correct.
""".strip()
SYSTEM_PROMPT_HOTPOTQA = """
You are a strict, impartial grader of answer correctness.

**Goal:** Compare System Output to Correct Answer only. Do not use outside knowledge; treat Correct Answer as ground truth.

**Output:** Reply with exactly one label (no quotes, no punctuation, no extra text):
- correct
- partially_correct
- incorrect
- refusal/unsure

**Scoring rules:**
1) Non-answer → refusal/unsure:
    - System Output refuses, asks for clarification, says "unknown"/"no information", or otherwise does not contain an answer value.

2) Exact match → correct:
    - The System Output conveys exactly the same answer value(s) as the Correct Answer and no additional distinct answer values, in the same order for multi-part answers.
    - Ignore casing, articles, punctuation/spacing, and duplicate repetitions.
    - Ignore explanatory or descriptive context that does not add distinct answer values (e.g., prose around the answer, unit qualifiers, city+state after a city, manufacturer prefixes, honorifics/titles, adjectives like "World Famous").
    - Accept unambiguous aliases or minor name variants for the same entity (e.g., nicknames vs full names, with/without middle names, common alternate forms) when they clearly refer to the same entity and do not introduce a different one.
    - Numbers/dates: treat words vs digits and digit-grouping (e.g., 1,840 vs 1840) as equivalent. If the expected value is a year, a full date containing that year is acceptable. If the expected value includes month+year, month alone is incomplete.
    - Approximation: If the question or Correct Answer signals approximation (about/approximately/around/over/≈), allow consistent approximate or inequality phrasing near the value.

3) Partial overlap → partially_correct:
    - At least one expected answer value appears, but any required value is missing and/or at least one value is incorrect/contradictory.
    - The output adds extra distinct answer values for a slot (e.g., listing multiple roles like "actor and director", multiple candidate entities with and/or/slashes/lists) beyond what the Correct Answer expects.

4) Mismatch → incorrect:
    - None of the expected answer values appear in the System Output, or any provided value contradicts the Correct Answer.

**Definitions and matching guidance:**
- Answer value: an atomic item such as an entity (ID or name), number, date/time, or yes/no.
- Entity matching: give credit only when the expected entity (name/ID or a clear alias/variant) is explicitly present in the System Output. Do not award credit for merely implying the answer without naming it. Do not give credit for broader/narrower/related categories.
- Order: When multiple questions are asked, the correct answers are in that order; the System Output must align to be correct.
""".strip()

# Template for the user prompt containing task, correct answer, and given answer
USER_PROMPT = """
Question: {task}
Correct Answer: {correct_answer}
System Output: {given_answer}
"""


def evaluate_single(
    model: str,
    messages: list[dict],
    response_format: dict,
    index: int,
    base_url: str | None = None,
    max_tokens: int | None = None,
) -> tuple[int, dict, float]:
    """
    Evaluate a single entry using OpenAI's chat completion API.

    This function sends a prompt to the specified GPT model to determine if the given answer
    matches the correct answer. It returns the evaluation result along with the API cost.

    Args:
        model (str): The name of the OpenAI model to use for evaluation (e.g., "gpt-4").
        messages (list[dict]): A list of message dictionaries for the chat completion,
            including system and user prompts.
        response_format (dict): The expected response format schema for the API call.
        index (int): The index of the entry being evaluated, used for tracking.
        base_url (str | None): Optional OpenAI-compatible API base URL. When set,
            uses the provided endpoint (e.g., a local vLLM server) instead of
            the default OpenAI API.
        max_tokens (int | None): Maximum number of tokens to generate. Useful for
            capping output from thinking models (e.g., Qwen3) that otherwise
            generate very long chains of thought. Defaults to None (unlimited).

    Returns:
        tuple[int, dict, float]: A tuple containing:
            - index (int): The original index of the entry.
            - evaluation (dict): A dictionary with keys "is_correct" (str: "correct", "partially_correct", "incorrect", or "refusal/unsure"),
              "model" (str), and "prompt_version" (str).
            - cost (float): The estimated cost of the API call in USD.
    """
    # Create a local OpenAI client instance, optionally pointing to a custom endpoint
    if base_url is not None:
        local_client = openai.OpenAI(base_url=base_url, api_key="EMPTY")
    else:
        local_client = openai.OpenAI()

    # Make the API call to evaluate the answer
    usage = None
    try:
        extra_kwargs: dict = {}
        if max_tokens is not None:
            extra_kwargs["max_tokens"] = max_tokens
        response = local_client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore
            temperature=0,  # Deterministic responses
            seed=42,  # For reproducibility
            response_format=response_format,  # type: ignore
            **extra_kwargs,
        )
        usage = response.usage

        # Parse the JSON response from the model
        parsed = json.loads(response.choices[0].message.content)  # type: ignore
        reasoning = parsed["reasoning"]
        value = parsed["is_correct"]

        # Prepare the evaluation dictionary
        evaluation = {
            "is_correct": value.lower(),
            "reasoning": reasoning,
            "model": model,
            "prompt_version": PROMPT_VERSION,
        }
    except Exception as e:
        evaluation = {
            "is_correct": "refusal/unsure",
            "reasoning": f"Error during evaluation: {e}",
            "model": model,
            "prompt_version": PROMPT_VERSION,
        }

    # Calculate the cost based on token usage (skipped for unknown/local models)
    cost = 0.0
    if usage and model in TOKEN_COSTS:
        input_cost_per_token = TOKEN_COSTS[model]["prompt_tokens"]
        output_cost_per_token = TOKEN_COSTS[model]["completion_tokens"]
        cost = usage.prompt_tokens * input_cost_per_token + usage.completion_tokens * output_cost_per_token

    return index, evaluation, cost


def format_output(output: Any) -> str | None:
    """
    Convert system output (potentially KoPL structured data) into a readable string.

    Args:
        output: The raw output from the system (can be str, list, or dict)

    Returns:
        A formatted string representation of the output
    """
    if isinstance(output, str):
        try:
            # Try to parse as literal if it looks like a Python structure
            if output.strip().startswith(("[", "{")):
                parsed_output = ast.literal_eval(output)
                return format_output(parsed_output)
            return output
        except (ValueError, SyntaxError):
            return output

    if isinstance(output, list):
        if not output:
            return ""
        if isinstance(output[0], dict) and is_kopl_value_class(output[0]):
            return ", ".join(kopl_value_class_to_str(v) for v in output)
        return ", ".join(str(v) for v in output)

    if isinstance(output, dict):
        if is_kopl_value_class(output):
            return kopl_value_class_to_str(output)
        elif is_kopl_entity(output):
            return kopl_entity_to_str(output)
        elif is_kopl_entity_facts(output):
            return kopl_entity_facts_to_str(output)
        return str(output)

    return str(output)


def format_expected(expected: Any, expected_label: Any) -> str:
    """
    Format the expected answer and its labels into a readable string.

    Args:
        expected: The raw expected answer
        expected_label: The labels associated with the expected answer

    Returns:
        A formatted string representation of the expected answer
    """
    if expected_label is None:
        if isinstance(expected, list):
            return ", ".join(str(e) for e in expected)
        return str(expected)

    # expected_label is not None
    if isinstance(expected_label, list) and isinstance(expected, list):
        if len(expected) == len(expected_label):
            return ", ".join(
                f"{e} ({label})" if str(e) != label else str(e) for e, label in zip(expected, expected_label)
            )

    return str(expected)


def main(args: argparse.Namespace) -> None:
    """
    Main function to evaluate system outputs against groundtruth using GPT.

    Reads entries from a JSONL file, evaluates each using the specified model,
    updates the entries with evaluation results, overwrites the file,
    and updates a metrics file with correctness counts.

    Args:
        args (argparse.Namespace): Parsed command-line arguments, including:
            - path_input: Path to the input JSONL file.
            - model: The GPT model to use for evaluation.
            - num_workers: Number of concurrent workers for processing.

    The function prints the total evaluation cost and correctness totals.
    """
    # Load the input JSONL file containing entries to evaluate
    entries: list[dict[str, Any]] = orjsonl.load(args.path_input)  # type: ignore

    # Define the expected JSON schema for the model's response
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "evaluation_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "brief justification (one sentence)"},
                    "is_correct": {
                        "type": "string",
                        "description": "Indicates if the given answer is correct.",
                        "enum": ["correct", "partially_correct", "incorrect", "refusal/unsure"],
                    },
                },
                "required": ["reasoning", "is_correct"],
                "additionalProperties": False,
            },
        },
    }

    # Prepare a list of tasks for evaluation, each containing model, messages, format, and index
    is_hotpotqa = "hotpotqa" in str(args.path_input).lower()
    SYSTEM_PROMPT = SYSTEM_PROMPT_HOTPOTQA if is_hotpotqa else SYSTEM_PROMPT_KBQA

    tasks = []
    skipped_count = 0
    for i, entry in enumerate(entries):
        assert isinstance(entry, dict), "Entry should be a dict"
        expected = entry.get("expected_answer", "")
        actual = entry.get("output", "")
        expected_label = entry.get("expected_answer_label", None)

        # Skip entries that failed execution (no successful system output)
        if not entry["metadata"].get("success", False) or (isinstance(actual, (str, list, dict)) and len(actual) == 0):
            # Update evaluation and skip
            entry["evaluation"] = {"is_correct": "refusal/unsure"}
            skipped_count += 1
            continue

        # Check if the answer is already correct using fast local evaluation
        is_correct = evaluate_answer(expected, actual, expected_label=expected_label)
        if not is_correct:
            if format_expected(expected, expected_label) == format_output(actual):
                is_correct = True
        if is_correct:
            # Skip LLM evaluation for entries already judged correct
            entry["evaluation"] = {"is_correct": "correct" if is_correct else "incorrect"}
            skipped_count += 1
            continue

        if isinstance(expected, list) and isinstance(actual, list):
            expected_is_entity_id = all(str(e).startswith("m.") for e in expected)
            actual_is_entity_id = all(str(a).startswith("m.") for a in actual)
            if expected_is_entity_id and actual_is_entity_id:
                # Both expected and actual are lists of entity IDs, use hard matching
                if set(expected) == set(actual):
                    entry["evaluation"] = {"is_correct": "correct"}
                else:
                    entry["evaluation"] = {"is_correct": "incorrect"}
                skipped_count += 1
                continue

        # Only add to evaluation tasks if the answer failed local evaluation
        # Format the user prompt with task details and preprocessed answer
        user_content = USER_PROMPT.format(
            task=entry["query"],  # type: ignore
            correct_answer=format_expected(entry["expected_answer"], entry.get("expected_answer_label")),  # type: ignore
            given_answer=format_output(entry["output"]),  # type: ignore
        )
        # Create the message list for the API call
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        tasks.append((args.model, messages, response_format, i, args.base_url, args.max_tokens))

    total_cost = 0.0
    if args.num_workers == 1:
        # Single-threaded mode for easier debugging
        for task in tqdm(tasks, desc="Evaluating"):
            index, evaluation, cost = evaluate_single(*task)
            entries[index]["evaluation"] = evaluation  # type: ignore
            # DEBUG
            print(f"Entry {index} evaluation: {evaluation}")
            print(
                f"- Expected: {entries[index]['expected_answer']} (label: {entries[index].get('expected_answer_label', 'N/A')})"
            )
            print(f"- Actual: {entries[index]['output']}")
            # breakpoint()
            total_cost += cost
    else:
        # Multi-threaded execution for performance
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(evaluate_single, *task) for task in tasks]

            with tqdm(total=len(tasks), desc="Evaluating") as pbar:
                for future in as_completed(futures):
                    try:
                        index, evaluation, cost = future.result()
                        entries[index]["evaluation"] = evaluation  # type: ignore
                        total_cost += cost
                        pbar.update(1)
                    except Exception as e:
                        # On error, close progress bar, print error, and stop execution
                        pbar.close()
                        print(f"\nError during evaluation: {e}", file=sys.stderr)
                        executor.shutdown(wait=False)
                        raise

    # Print the total cost incurred for evaluations
    print(f"Total cost for evaluation: ${total_cost:.6f}")
    print(f"Entries evaluated with LLM: {len(tasks)}")
    print(f"Entries skipped (failed or already correct): {skipped_count}")

    # Overwrite the input file with updated entries including evaluations
    with open(args.path_input, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    # Update or create the metrics file in the same directory
    metrics_path = args.path_input.parent / "metrics.jsonl"
    if metrics_path.exists():
        metrics = orjsonl.load(metrics_path)[0]  # type: ignore
    else:
        metrics = {}

    # Calculate the number of correct evaluations
    original_total = metrics.get("correct_answers", 0)  # type: ignore
    total_correct = sum(1 for e in entries if e["evaluation"]["is_correct"] == "correct")  # type: ignore
    # Store the correct count with model and prompt version
    metrics[f"correct_answers_{args.model}_{PROMPT_VERSION}"] = total_correct  # type: ignore

    # Write the updated metrics back to file
    with open(metrics_path, "w") as f:
        f.write(json.dumps(metrics) + "\n")

    # Print the final correctness totals
    print(f"Total correct: {total_correct}, Original total: {original_total}")


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Evaluate system outputs against groundtruth using GPT models.")
    parser.add_argument(
        "-i",
        "--input",
        dest="path_input",
        type=Path,
        required=True,
        help="Path to the input JSONL file containing entries to evaluate.",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        type=str,
        default="gpt-4.1-mini-2025-04-14",
        help="The OpenAI model to use for evaluation (e.g., 'gpt-4').",
    )
    parser.add_argument(
        "-w",
        "--workers",
        dest="num_workers",
        type=int,
        default=50,
        help="Number of concurrent workers for parallel evaluation.",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        type=str,
        default=None,
        help=(
            "Base URL for an OpenAI-compatible API endpoint "
            "(e.g., 'http://localhost:8000/v1' for a local vLLM server). "
            "Defaults to the official OpenAI API."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=None,
        help=(
            "Maximum number of tokens to generate per evaluation call."
        ),
    )
    # Parse arguments and run the main function
    args = parser.parse_args()
    main(args)
