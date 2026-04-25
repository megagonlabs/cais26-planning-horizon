from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
import json
import random
import re
import sys

from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from planning.task_characterization.atomic_kbqa_dag_converter import (  # noqa: E402
    atomic_kbqa_dag_conversion,
)


def extract_entities_from_function_list(
    function_list: list[str],
) -> list[tuple[str | None, str]]:
    """
    Extract entities and values from function list.

    Based on the logic from vendor/KBQA-o1/reasoners/kbqa/prompt_function.py
    Extracts entities from START operations and time values from TC operations.

    Args:
        function_list: List of function strings

    Returns:
        List of tuples (label, id) for entities/values found
    """
    entities = []
    entity_ids = set()

    for func in function_list:
        if "START" in func:
            # Extract entity from START operation: expressionX = START('entity_id')
            match = re.search(r"START\('([^']+)'\)", func)
            if match:
                entity_id = match.group(1)
                entity_ids.add(entity_id)
        elif "TC" in func:
            # Extract time value from TC operation: expressionX = TC(expressionY, 'relation', 'time')
            match = re.search(r"TC\([^,]+,\s*'[^']+'\s*,\s*'([^']+)'\)", func)
            if match:
                time_value = match.group(1)
                if time_value != "NOW":
                    entities.append((time_value, time_value))
    # Get actual entity label from Freebase
    # Import here to avoid circular imports at module level
    from planning.tools.freebase.sparql_executor import (
        get_label_with_odbc,
    )

    for eid in entity_ids:
        try:
            entity_label = get_label_with_odbc(eid)
        except KeyboardInterrupt:
            raise
        except Exception:
            # Fallback to entity ID if database query fails
            entity_label = None
        entities.append((entity_label, eid))

    return sorted(list(set(entities)))


def get_answer_label(answer: list[str]) -> list[str]:
    """Get answer labels from Freebase for a list of answer IDs.

    Args:
        answer: List of answer entity IDs or literals

    Returns:
        List of answer labels (or original IDs if label not found)
    """
    from planning.tools.freebase.sparql_executor import (
        get_label_with_odbc,
    )

    answer_labels = []
    for ans in answer:
        if ans[:2] in ["m.", "g."]:
            label = get_label_with_odbc(ans)
            answer_labels.append(label if label else ans)
        else:
            answer_labels.append(ans)
    return answer_labels


def has_nested_relation_expression(function_list: list[str]) -> bool:
    """
    Check if function_list contains nested expressions in relational arguments.

    This validates that relational arguments in JOIN, ARG, CMP, and TC are
    string literals (e.g., 'relation.name') and not variable references
    (e.g., expression1).

    Args:
        function_list: List of function strings

    Returns:
        True if nested expressions found in relational arguments, False otherwise
    """
    for func in function_list:
        func = func.strip()

        # Check JOIN: JOIN('relation' OR expression, target)
        # Relation argument (first arg) should not be a variable reference
        if "JOIN" in func:
            match = re.search(r"JOIN\(([^,]+),", func)
            if match:
                relation_arg = match.group(1).strip()
                # Check if it's a variable reference (starts with 'expression')
                if relation_arg.startswith("expression"):
                    return True

        # Check ARG: ARG('mode', input_ref, 'property_relation' OR expression)
        # Property relation (third arg) should not be a variable reference
        elif "ARG" in func:
            match = re.search(r"ARG\([^,]+,\s*[^,]+,\s*([^)]+)\)", func)
            if match:
                property_arg = match.group(1).strip()
                # Check if it's a variable reference
                if property_arg.startswith("expression"):
                    return True

        # Check CMP: CMP('operator', 'property_relation' OR expression, literal)
        # Property relation (second arg) should not be a variable reference
        elif "CMP" in func:
            match = re.search(r"CMP\([^,]+,\s*([^,]+),", func)
            if match:
                property_arg = match.group(1).strip()
                # Check if it's a variable reference
                if property_arg.startswith("expression"):
                    return True

        # Check TC: TC(input_ref, 'temporal_relation' OR expression, temporal_literal)
        # Temporal relation (second arg) should not be a variable reference
        elif "TC" in func:
            match = re.search(r"TC\([^,]+,\s*([^,]+),", func)
            if match:
                temporal_arg = match.group(1).strip()
                # Check if it's a variable reference
                if temporal_arg.startswith("expression"):
                    return True

    return False


def run_preprocessing(
    input_file: Path,
    output_file: Path,
    split_name: str,
    assign_to_bin: Callable[[int], int],
    num_bins: int = 4,
    target_per_bin: int = 125,
    seed: int = 42,
    heldout_pool_file: Path | None = None,
) -> dict[str, Any]:
    """
    Preprocess a single dataset split (train or test).

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        split_name: Name of the split ("train" or "test")
        assign_to_bin: Function that maps workflow_length (int) to bin index (int)
        num_bins: Number of bins to use for balanced sampling
        target_per_bin: Target number of examples per bin
        seed: Random seed for reproducibility
        heldout_pool_file: Optional path to save non-selected examples (for train split only)

    Returns:
        Dict with statistics about preprocessing
    """
    random.seed(seed)
    # Load data
    print(f"Loading {input_file}...")
    with open(input_file, "r") as f:
        data = json.load(f)

    print(f"Total examples in {split_name}: {len(data)}")

    # Step 1: Deduplicate examples based on sexpr
    # For reproducibility, random.seed(seed) is set at the top of this function.
    # Group all examples by sexpr and pick one example at random from each group.
    unique_data: list[dict[str, Any]] = []
    duplicates_removed = 0

    # Collect examples by sexpr; treat missing sexpr as unique entries
    sexpr_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, example in enumerate(data):
        sexpr: str = example["sexpr"]
        example["original_idx"] = idx  # Track original index
        sexpr_groups[sexpr].append(example)

    # For each group of duplicates, choose one example at random and count duplicates
    for sexpr, group in sexpr_groups.items():
        if len(group) == 1:
            example = group[0]
            unique_data.append(example)
        else:
            chosen_example = random.choice(group)
            unique_data.append(chosen_example)
            duplicates_removed += len(group) - 1

    print(
        f"Deduplicated examples: removed {duplicates_removed} duplicates; {len(unique_data)} unique remain"
    )

    # Step 1.5: Filter out examples with nested expressions in relational arguments
    filtered_data = []
    nested_relation_removed = 0

    for example in tqdm(
        unique_data, desc="Filtering nested relation expressions", total=len(unique_data)
    ):
        if has_nested_relation_expression(example["function_list"]):
            nested_relation_removed += 1
        else:
            filtered_data.append(example)

    print(
        f"Filtered nested relation expressions: removed {nested_relation_removed} examples; {len(filtered_data)} remain"
    )

    # Step 2: Organize examples by bin
    bins: dict[int, list[dict[str, Any]]] = {i: [] for i in range(num_bins)}

    for idx, example in tqdm(
        enumerate(filtered_data), total=len(filtered_data), desc="Organizing by bin"
    ):
        workflow_length = len(example["function_list"])
        bin_idx = assign_to_bin(workflow_length)
        example["workflow_length"] = workflow_length

        # Store example
        bins[bin_idx].append(example)

    print("Bin distribution:")
    for bin_idx, examples in bins.items():
        print(f"  Bin {bin_idx}: {len(examples)} examples")

    # Step 3: Sample target_per_bin examples from each bin
    selected_examples = []
    selected_indices = set()  # Track selected original indices
    for bin_idx, examples in bins.items():
        if len(examples) < target_per_bin:
            print(
                f"Warning: Bin {bin_idx} has only {len(examples)} examples (need {target_per_bin})"
            )
            sampled = examples  # Take all available
        else:
            sampled = random.sample(examples, target_per_bin)

        # Extract entities for the sampled examples (do this after selection to save time)
        for example in tqdm(sampled, desc=f"Extracting entities for bin {bin_idx}"):
            func_list = example["function_list"]
            example["entities"] = extract_entities_from_function_list(func_list)
            dag = atomic_kbqa_dag_conversion(func_list, example["entities"])
            example["dag"] = dag
            example["answer_label"] = get_answer_label(example["answer"])
        selected_examples.extend(sampled)
        selected_indices.update(example["original_idx"] for example in sampled)

    print(f"\nSelected {len(selected_examples)} examples for {split_name}")

    # Step 3.5: Save heldout pool (non-selected examples) for train split
    if heldout_pool_file is not None and split_name == "train":
        # Collect all validated examples that were not selected
        heldout_pool = []
        for bin_idx, examples in bins.items():
            for example in tqdm(examples, desc=f"Collecting heldout for bin {bin_idx}"):
                if example["original_idx"] not in selected_indices:
                    func_list = example["function_list"]
                    entities = extract_entities_from_function_list(func_list)
                    dag = atomic_kbqa_dag_conversion(func_list, entities)
                    pool_entry = {
                        "ID": example["ID"],
                        "question": example["question"],
                        "answer": example["answer"],
                        "entities": entities,
                        "function_list": func_list,
                        "dag": dag,
                    }
                    heldout_pool.append(pool_entry)

        # Save heldout pool
        heldout_pool_file.parent.mkdir(parents=True, exist_ok=True)
        with open(heldout_pool_file, "w") as f:
            json.dump(heldout_pool, f, indent=2)

        print(f"Saved {len(heldout_pool)} heldout examples to {heldout_pool_file}")

    # Step 4: Generate unique IDs (DAGs already computed in Step 2)
    processed_examples = []
    for i, example in enumerate(selected_examples):
        # Generate unique ID: split_name + index
        unique_id = f"{split_name}_{i:04d}"

        # Create processed example
        processed = {
            "id": unique_id,
            "original_id": example["ID"],
            "question": example["question"],
            "answer": example["answer"],
            "answer_label": example["answer_label"],
            "sparql": example["sparql"],
            "sexpr": example["sexpr"],
            "function_list": example["function_list"],
            "entities": example["entities"],
            "dag": example["dag"],
            "metadata": {
                "split": split_name,
                "original_idx": example["original_idx"],
                "workflow_length": example["workflow_length"],
                "bin": assign_to_bin(example["workflow_length"]),
                "level": example.get("level", "unknown"),
            },
        }
        processed_examples.append(processed)

    # Step 5: Save preprocessed data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(processed_examples, f, indent=2)

    print(f"\nSaved {len(processed_examples)} examples to {output_file}")

    # Return statistics
    return {
        "total_input": len(data),
        "deduped_input": len(unique_data),
        "duplicates_removed": duplicates_removed,
        "filtered_input": len(filtered_data),
        "nested_relation_removed": nested_relation_removed,
        "converted": sum(len(b) for b in bins.values()),
        "selected": len(processed_examples),
        "bin_distribution": {
            bin_idx: sum(
                1 for ex in processed_examples if ex["metadata"]["bin"] == bin_idx
            )
            for bin_idx in range(num_bins)
        },
    }
