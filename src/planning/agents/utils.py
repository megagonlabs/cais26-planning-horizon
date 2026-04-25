"""
Utility functions for agent operations.

This module provides reusable functions for parameter processing,
substitution tracking, and other common agent operations.
"""

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING
import re

from .exceptions import AgentException

if TYPE_CHECKING:
    from ..environment import Environment
    from .memory import ContextMemory


@dataclass
class FinishResult:
    """Result of processing a finish action's answer argument.

    Attributes:
        success: True if processing succeeded, False if error occurred
        processed_answer: The final processed answer string (empty on failure)
        error_message: Error description if failed, None otherwise
    """

    success: bool
    processed_answer: str
    error_message: str | None


def update_params_with_processed_values(
    action_args: list[str] | dict[str, Any],
    processed_params: dict[str, Any],
) -> tuple[list[str] | dict[str, Any], list[str] | dict[str, Any]]:
    """
    Update action arguments with processed parameter values.

    This function applies soft-match substitutions from processed_params to action_args,
    while preserving step references (e.g., $0, $1) which are intentional references
    to previous step outputs.

    Args:
        action_args: Original action arguments (dict or list format)
        processed_params: Dictionary of processed parameter values from tool execution

    Returns:
        Tuple of (raw_params, updated_args):
            - raw_params: Copy of original arguments before substitution
            - updated_args: Arguments with processed values applied
    """
    raw_params = action_args.copy()

    for i, (key, val) in enumerate(processed_params.items()):
        _val = action_args[i] if isinstance(action_args, list) else action_args.get(key, "")

        # Skip substitution for step references (e.g., $0, $1)
        if isinstance(_val, str) and re.match(r"^\$(\d+)$", _val):
            continue

        # Apply substitution
        if isinstance(action_args, list):
            action_args[i] = val
        elif isinstance(action_args, dict):
            action_args[key] = val

    return raw_params, action_args


def format_param_substitutions(
    raw_params: list[str] | dict[str, Any],
    updated_params: list[str] | dict[str, Any],
) -> str:
    """
    Generate human-readable multi-line text describing parameter substitutions.

    Compares original parameters with updated parameters to identify what was
    substituted during soft-matching. Returns a formatted string showing each
    substitution that occurred.

    Args:
        raw_params: Original parameters before substitution
        updated_params: Parameters after substitution

    Returns:
        Multi-line string with substitution details, or empty string if no changes.
        Format: "# <param_name> '<original_value>' was not found. Substituted to '<new_value>'"
    """
    substitutions = []

    if isinstance(raw_params, dict) and isinstance(updated_params, dict):
        # Handle dict-based parameters
        for key in raw_params.keys():
            raw_val = raw_params.get(key)
            updated_val = updated_params.get(key)

            # Only track string substitutions (soft-matching applies to strings)
            if not isinstance(raw_val, str) or not isinstance(updated_val, str):
                continue

            # Skip step references (intentional references, not substitutions)
            if re.match(r"^\$(\d+)$", raw_val):
                continue

            # Check if value changed
            if raw_val != updated_val:
                substitutions.append(f"# {key} '{raw_val}' was not found. Substituted to '{updated_val}'")

    elif isinstance(raw_params, list) and isinstance(updated_params, list):
        # Handle list-based parameters
        for i, (raw_val, updated_val) in enumerate(zip(raw_params, updated_params)):
            # Only track string substitutions
            if not isinstance(raw_val, str) or not isinstance(updated_val, str):
                continue

            # Skip step references
            if re.match(r"^\$(\d+)$", raw_val):
                continue

            # Check if value changed
            if raw_val != updated_val:
                substitutions.append(f"# arg[{i}] '{raw_val}' was not found. Substituted to '{updated_val}'")

    # Return formatted string with all substitutions
    return "\n".join(substitutions) if substitutions else ""


def validate_and_normalize_step_references(
    action_args: dict[str, Any],
    current_step_index: int,
    memory: Optional["ContextMemory"] = None,
    check_has_data: bool = False,
    normalize: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """
    Validate and normalize step references ($i) in action arguments.

    This function:
    1. Detects step references anywhere in values (e.g., "$1", "text $2 more", "$1.something")
    2. Validates against negative references
    3. Validates against self-references
    4. Validates against forward references
    5. Optionally checks if referenced steps have data (for SH)
    6. Optionally normalizes single step references (e.g., "$1.something" → "$1")

    Args:
        action_args: Dictionary of action arguments that may contain step references
        current_step_index: Index of the current step (for self-reference validation)
        memory: Optional context memory for validating step existence and data
        check_has_data: Whether to validate that referenced steps have data (default: False)
        normalize: Whether to normalize single step references to $<num> format (default: True)

    Returns:
        Tuple of (normalized_args, errors):
            - normalized_args: Arguments with step references normalized to $<num> (if normalize=True and single reference)
            - errors: List of error messages (empty if validation passed)
    """
    normalized_args = action_args.copy()
    errors = []

    for key, val in normalized_args.items():
        # Convert value to list for uniform processing
        value_list = []
        if isinstance(val, str):
            value_list = [val]
        elif isinstance(val, list):
            value_list = val

        # Validate and normalize each string value
        for idx, value_str in enumerate(value_list):
            if not isinstance(value_str, str):
                continue

            # Find all step references in the value (e.g., "$1", "$2" in "combine $1 and $2")
            step_refs = re.findall(r"\$(\d+)", value_str)

            if not step_refs:
                continue  # No step references in this value

            # Validate each step reference found
            for step_idx_str in step_refs:
                step_idx = int(step_idx_str)

                # Validate step reference
                if step_idx < 0:
                    errors.append(f"Invalid negative step reference: {key}={val}")
                    continue

                if step_idx == current_step_index:
                    errors.append(f"Step cannot reference itself: {key}={val}")
                    continue

                if step_idx > current_step_index:
                    errors.append(f"Invalid forward step reference: {key}={val}")
                    continue

                # Check if referenced step has data (only if memory provided and check enabled)
                if check_has_data and memory is not None:
                    if 0 <= step_idx < len(memory.step_history):
                        step = memory.step_history[step_idx]
                        if step.data.get("full") is None:
                            errors.append(
                                f"Step {step_idx} does not contain data to reference. "
                                f"Please double check the step number."
                            )

            # Normalize if requested and value contains exactly one reference
            if normalize and len(step_refs) == 1:
                value_list[idx] = f"${step_refs[0]}"

        # Convert back to original type
        if isinstance(val, str):
            normalized_args[key] = value_list[0] if value_list else val
        elif isinstance(val, list):
            normalized_args[key] = value_list

    return normalized_args, errors


def process_finish_answer(
    answer: str | list[str],
    environment: "Environment",
    memory: "ContextMemory",
) -> FinishResult:
    """
    Process the answer argument from a finish action.

    This function handles:
    1. Memory reference resolution via environment.resolve_action_params
    2. List inputs with placeholder resolution for each item
    3. KoPL-specific answer postprocessing for exact single references
    4. Error handling for invalid references

    Args:
        answer: The answer value from finish action (string, list, or with memory references)
        environment: Environment instance for memory reference resolution
        memory: Current execution memory

    Returns:
        FinishResult with success status, processed answer, and optional error message

    Note:
        Due to pre-validation and normalization, embedded references in KoPL answers
        are not expected. Only exact references like "$0" may trigger kopl postprocessing.
    """
    try:
        # Check if answer is an exact single step reference for kopl postprocessing
        kopl_postprocess = False
        if isinstance(answer, str) and (m := re.match(r"^\$(\d+)$", answer)):
            # Exact single reference - check if it's from kopl agent
            ref_idx = int(m.group(1))
            if 0 <= ref_idx < len(memory.step_history):
                ref_step = memory.step_history[ref_idx]
                agent_type = ref_step.get_metadata("agent_type")
                if agent_type == "kopl":
                    from .kopl_utils import SCALAR_RESULT_OPERATORS

                    kopl_postprocess = True
                    # Validate the last step. It must be a scalar result step.
                    action_name = ref_step.get("action", {})["name"]
                    if action_name not in SCALAR_RESULT_OPERATORS:
                        SCALAR_RESULT_OPERATORS_list = sorted(list(SCALAR_RESULT_OPERATORS))
                        ops_str = (
                            ", ".join(SCALAR_RESULT_OPERATORS_list[:-1]) + ", or " + SCALAR_RESULT_OPERATORS_list[-1]
                        )  # type: ignore
                        raise AgentException(
                            f"The input of finish must be the output of: {ops_str}\nGot: ${ref_idx}={action_name}\n",
                            "Please try a different input/tool.",
                        )

        # Delegate resolution to environment
        resolved = environment.resolve_action_params(
            {"answer": answer}, memory.session_id, force_substitute_memory_reference=True
        )["answer"]

        # Apply kopl postprocessing if needed
        if kopl_postprocess:
            from .kopl_utils import postprocess_kopl_answer

            processed_answer = postprocess_kopl_answer(str(resolved))
            if processed_answer is None:
                raise AgentException("No valid answer found. Please try a different input/tool.")
        else:
            processed_answer = resolved

        return FinishResult(success=True, processed_answer=processed_answer, error_message=None)

    except AgentException as e:
        # Memory reference error or other agent-level error
        return FinishResult(success=False, processed_answer="", error_message=str(e))
