"""
Utility functions for evaluating experiment results.

This module provides functions for calculating evaluation metrics
across different datasets (KQA Pro, HotpotQA, etc.).
"""

import ast
import re
from typing import Any


def _extract_numeric(value: str) -> str | None:
    """
    Extract numeric value from a string if it contains digits.

    Removes all non-digit characters (commas, spaces, text, etc.)
    to enable comparison of numbers in different formats.

    Args:
        value: String that may contain a numeric value

    Returns:
        str: Digits only, or None if no digits found
    """
    if any(c.isdigit() for c in value):
        return re.sub(r"\D", "", value)
    return None


def _compare_values(expected_str: str, actual_str: str) -> bool:
    """
    Compare two string values with normalization.

    Handles:
    - Boolean normalization (yes/no/true/false)
    - Numeric extraction (digits only, ignoring commas/text)
    - Case-insensitive string matching

    Args:
        expected_str: Expected value as string (already stripped)
        actual_str: Actual value as string (already stripped)

    Returns:
        bool: True if values match after normalization
    """
    if not expected_str or not actual_str:
        return False

    # Try boolean normalization first
    expected_bool = _normalize_boolean(expected_str.lower())
    actual_bool = _normalize_boolean(actual_str.lower())
    if expected_bool and actual_bool:
        return expected_bool == actual_bool

    # Try numeric extraction
    expected_num = _extract_numeric(expected_str)
    actual_num = _extract_numeric(actual_str)
    if expected_num and actual_num:
        return expected_num == actual_num

    # Fall back to case-insensitive string comparison
    return expected_str.lower() == actual_str.lower()


def evaluate_answer(
    expected: Any,
    actual: Any,
    expected_label: list[str] | None = None,
) -> bool:
    """
    Evaluate if the actual answer matches the expected answer.

    Handles multiple formats across datasets:
    - KBQA: list[str] (entity IDs) or list[int] (counts, size 1)
    - HotpotQA: str, int, or bool
    - Entity Labels: When provided, fallback to label matching if ID matching fails
    - Normalization: numeric extraction, case-insensitive matching

    Args:
        expected: Ground truth answer (list or scalar)
        actual: System's output (list, str, int, bool, or None)
        expected_label: Optional list of entity labels for fallback matching. Only used when
                       expected is list[str]. Must have same length as expected if provided.

    Returns:
        bool: True if answers match, False otherwise

    Raises:
        ValueError: If expected_label is provided but data format is invalid
                   (length mismatch or not a list when expected is list)
    """
    # Handle None actual output
    if actual is None:
        return False

    if isinstance(actual, str) and actual.strip().startswith(("[", "{")):
        # Try to parse as literal if it looks like a Python structure
        try:
            parsed_output = ast.literal_eval(actual)
            return evaluate_answer(expected, parsed_output, expected_label=expected_label)
        except (ValueError, SyntaxError):
            pass  # Fall back to string comparison

    # Case 1: Expected is a list (typically KBQA)
    if isinstance(expected, list):
        # Validate expected_label if provided
        if expected_label is not None:
            if not isinstance(expected_label, list):
                raise ValueError(
                    f"expected_label must be a list, got {type(expected_label).__name__}"
                )
            if len(expected_label) != len(expected):
                raise ValueError(
                    f"expected and expected_label must have the same length, "
                    f"got {len(expected)} and {len(expected_label)}"
                )

        if isinstance(actual, list):
            # Both are lists: compare as sets (order-independent)
            # Convert all items to strings for comparison
            expected_set = set(str(item).strip().lower() for item in expected)
            actual_set = set(str(item).strip().lower() for item in actual)
            match = expected_set == actual_set

            # If ID matching fails and labels provided, try label matching
            if not match and expected_label is not None:
                expected_label_set = set(
                    str(label).strip().lower() for label in expected_label
                )
                actual_set_lower = set(str(item).strip().lower() for item in actual)
                match = expected_label_set == actual_set_lower

            return match
        else:
            # Expected is list, actual is scalar
            # Only match if expected has single item
            if len(expected) == 1:
                expected_str = str(expected[0]).strip()
                actual_str = str(actual).strip()

                # Try ID matching first
                if _compare_values(expected_str, actual_str):
                    return True

                # If ID matching fails and label provided, try label matching
                if expected_label is not None:
                    expected_label_str = str(expected_label[0]).strip()
                    return _compare_values(expected_label_str, actual_str)

                return False
            return False

    # Case 2: Expected is scalar (typically HotpotQA or single-value KBQA)
    # Validate expected_label is not provided for scalar expected
    if expected_label is not None:
        raise ValueError(
            "expected_label should only be provided when expected is a list"
        )

    # Convert boolean to string representation for consistency
    if isinstance(expected, bool):
        expected_str = "true" if expected else "false"
    else:
        expected_str = str(expected).strip()

    if isinstance(actual, bool):
        actual_str = "true" if actual else "false"
    else:
        actual_str = str(actual).strip()

    return _compare_values(expected_str, actual_str)


def _normalize_boolean(value: str) -> str | None:
    """
    Normalize yes/no/true/false variants to a standard representation.

    Args:
        value: String value (already lowercase)

    Returns:
        str: Normalized value ('yes', 'no', or None if not a boolean)
    """
    yes_variants = {"yes", "true", "y", "t"}
    no_variants = {"no", "false", "n", "f"}

    if value in yes_variants:
        return "yes"
    elif value in no_variants:
        return "no"
    return None


def calculate_evaluation_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics from episode results.

    Args:
        results: List of episode results

    Returns:
        dict: Evaluation metrics including accuracy, success rate, and token usage
    """
    if not results:
        return {
            "total_episodes": 0,
            "successful_episodes": 0,
            "success_rate": 0.0,
            "accuracy": 0.0,
            "avg_runtime": 0.0,
            "avg_steps": 0.0,
            "total_token_usage": {},
            "avg_token_usage": {},
        }

    total_episodes = len(results)
    successful_episodes = sum(1 for r in results if r["metadata"].get("success", False))
    success_rate = successful_episodes / total_episodes if total_episodes > 0 else 0

    # Calculate accuracy by comparing outputs with expected answers
    correct_answers = 0
    for result in results:
        if result["metadata"].get("success", False):
            expected = result.get("expected_answer", "")
            actual = result.get("output", "")
            expected_label = result.get("expected_answer_label", None)
            if evaluate_answer(expected, actual, expected_label=expected_label):
                correct_answers += 1

    accuracy = correct_answers / total_episodes if total_episodes > 0 else 0

    total_runtime_all = sum(r["metadata"].get("runtime", 0) for r in results)
    avg_runtime_all = total_runtime_all / total_episodes if total_episodes > 0 else 0

    total_steps_successful = sum(
        r["metadata"].get("step_count", 0)
        for r in results
        if r["metadata"].get("success", False) and r["metadata"].get("step_count") is not None
    )
    total_steps_all = sum(
        r["metadata"].get("step_count", 0)
        for r in results
        if r["metadata"].get("step_count") is not None
    )
    avg_steps_successful = total_steps_successful / successful_episodes if successful_episodes > 0 else 0
    avg_steps_all = total_steps_all / total_episodes if total_episodes > 0 else 0


    # Aggregate token usage
    total_token_usage = {}
    token_count = 0
    TOKEN_KEY_MAPPINGS = {
        "prompt_tokens": ["prompt_tokens", "input_tokens"],
        "completion_tokens": ["completion_tokens", "output_tokens"],
        "total_tokens": ["total_tokens"],
    }
    for result in results:
        token_usage = result["metadata"].get("token_usage", {})
        if token_usage:
            token_count += 1
            for model, usage in token_usage.items():
                if model not in total_token_usage:
                    total_token_usage[model] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    }
                for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    for alt_key in TOKEN_KEY_MAPPINGS[key]:
                        if alt_key in usage:
                            total_token_usage[model][key] += usage[alt_key]
                            break

    # Calculate average token usage
    avg_token_usage = {}
    if token_count > 0:
        for model, usage in total_token_usage.items():
            avg_token_usage[model] = {
                key: value / token_count for key, value in usage.items()
            }

    return {
        "total_episodes": total_episodes,
        "successful_episodes": successful_episodes,
        "success_rate": success_rate,
        "accuracy": accuracy,
        "correct_answers": correct_answers,
        "avg_steps_successful": avg_steps_successful,
        "avg_steps": avg_steps_all,
        "total_runtime": total_runtime_all,
        "avg_runtime": avg_runtime_all,
        "total_token_usage": total_token_usage,
        "avg_token_usage": avg_token_usage,
    }
