"""
Preprocess KQA Pro dataset for agent benchmarking.

This script validates KoPL programs and creates balanced benchmark splits with:
- Validated execution (programs that return correct answers)
- Deduplicated DAG representations
- Balanced workflow length distribution (100 examples per bin)
- Total of 500 examples per split (train/val)

Usage:
    uv run python data/kopl_kbqa/kqa_pro/scripts/preprocess_kqa_pro.py
"""

from pathlib import Path
import argparse
import json
import random
import sys

from kopl.kopl import KoPLEngine
from kopl import ValueClass
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from planning.task_characterization.kopl_dag_converter import (  # noqa: E402
    kopl_dag_conversion,
)


def execute_program(program, engine):
    """
    Execute a KoPL program and return the final result.

    Args:
        program: List of program steps with function, dependencies, and inputs
        engine: KoPLEngine instance for execution

    Returns:
        The final result of program execution

    Raises:
        Exception: If execution fails at any step
    """
    intermediate_results = []
    for step in program:
        func_name = step["function"]
        # Handle "What" function alias
        if func_name == "What":
            func_name = "QueryName"
        func = getattr(engine, func_name)
        deps = step["dependencies"]

        if len(deps) == 0:
            result = func(*step.get("inputs", []))
            intermediate_results.append(result)
        elif len(deps) == 1:
            result = func(intermediate_results[deps[0]], *step.get("inputs", []))
            intermediate_results.append(result)
        elif len(deps) == 2:
            result = func(
                intermediate_results[deps[0]],
                intermediate_results[deps[1]],
                *step.get("inputs", []),
            )
            intermediate_results.append(result)
        else:
            raise ValueError(
                f"Unsupported number of dependencies: {len(deps)} for step {step}"
            )

    # Return the last result (final answer)
    return intermediate_results[-1] if intermediate_results else None


def validate_example(example, engine):
    """
    Validate that a KoPL program executes successfully and returns the correct answer.

    Args:
        example: Dict with "program" and "answer" fields
        engine: KoPLEngine instance

    Returns:
        Tuple of (is_valid, executed_answer)
        - is_valid: True if execution succeeded and answer matches
        - executed_answer: The result from executing the program (or None if failed)
    """
    try:
        executed_answer = execute_program(example["program"], engine)

        # Normalize answers for comparison (strip whitespace, lowercase)
        expected = str(example["answer"]).strip().lower()
        if isinstance(executed_answer, list):
            if len(executed_answer) == 1:
                executed_answer = str(executed_answer[0])
            else:
                executed_answer = list(set([str(ans) for ans in executed_answer]))
                if len(executed_answer) == 1:
                    executed_answer = executed_answer.pop()
        elif isinstance(executed_answer, (ValueClass, str, int, float)):
            executed_answer = str(executed_answer)
        else:
            raise ValueError("Unsupported answer type")

        if isinstance(executed_answer, str):
            actual = executed_answer.strip().lower()
        else:
            actual = executed_answer

        is_valid = expected == actual
        return is_valid, executed_answer
    except Exception:
        # Execution failed
        return False, None


def assign_to_bin(workflow_length):
    """
    Assign workflow length to one of 5 bins for balanced sampling.

    Bins:
        0: [1-3]
        1: [4-5]
        2: [6-7]
        3: [8-9]
        4: [10+]

    Args:
        workflow_length: Integer workflow length

    Returns:
        Integer bin index (0-4)
    """
    if workflow_length <= 3:
        return 0
    elif workflow_length <= 5:
        return 1
    elif workflow_length <= 7:
        return 2
    elif workflow_length <= 9:
        return 3
    else:
        return 4


def preprocess_split(
    input_file, output_file, split_name, engine, seed=42, heldout_pool_file=None
):
    """
    Preprocess a single dataset split (train or val).

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        split_name: Name of the split ("train" or "val")
        engine: KoPLEngine instance
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

    # Step 1: Validate all examples, convert to DAG, and organize by bin
    bins = {0: [], 1: [], 2: [], 3: [], 4: []}
    discarded_count = 0
    discarded_reasons = {"execution_failed": 0, "answer_mismatch": 0}

    for idx, example in tqdm(
        enumerate(data), total=len(data), desc="Validating examples"
    ):
        is_valid, executed_answer = validate_example(example, engine)

        if not is_valid:
            discarded_count += 1
            if executed_answer is None:
                discarded_reasons["execution_failed"] += 1
            else:
                discarded_reasons["answer_mismatch"] += 1
            continue

        # Convert program to DAG and compute workflow length
        dag, mapping = kopl_dag_conversion(example["program"])
        ## Rewrite What -> QueryName
        for step in dag:
            if step["function"] == "What":
                step["function"] = "QueryName"
        workflow_length = len(dag)  # Use DAG length, not original program length
        bin_idx = assign_to_bin(workflow_length)

        # Store example with metadata (including pre-computed DAG)
        bins[bin_idx].append(
            {
                "example": example,
                "original_idx": idx,
                "workflow_length": workflow_length,
                "dag": dag,
                "dag_mapping": mapping,
            }
        )

    print(f"Validated examples: {sum(len(b) for b in bins.values())}")
    print(f"Discarded examples: {discarded_count}")
    print(f"  - Execution failed: {discarded_reasons['execution_failed']}")
    print(f"  - Answer mismatch: {discarded_reasons['answer_mismatch']}")
    print("\nBin distribution:")
    for bin_idx, examples in bins.items():
        print(f"  Bin {bin_idx}: {len(examples)} examples")

    # Step 2: Sample (target_per_bin) examples from each bin
    target_per_bin = 200

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

        selected_examples.extend(sampled)
        selected_indices.update(item["original_idx"] for item in sampled)

    print(f"\nSelected {len(selected_examples)} examples for {split_name}")

    # Step 2.5: Save heldout pool (non-selected examples) for train split
    if heldout_pool_file is not None and split_name == "train":
        # Collect all validated examples that were not selected
        heldout_pool = []
        for bin_idx, examples in bins.items():
            for item in examples:
                if item["original_idx"] not in selected_indices:
                    pool_entry = {
                        "index": item["original_idx"],
                        "question": item["example"]["question"],
                        "answer": item["example"]["answer"],
                        "dag": item["dag"],
                    }
                    heldout_pool.append(pool_entry)

        # Save heldout pool
        heldout_pool_file.parent.mkdir(parents=True, exist_ok=True)
        with open(heldout_pool_file, "w") as f:
            json.dump(heldout_pool, f, indent=2)

        print(f"Saved {len(heldout_pool)} heldout examples to {heldout_pool_file}")

    # Step 3: Generate unique IDs (DAGs already computed in Step 1)
    processed_examples = []
    for i, item in enumerate(selected_examples):
        example = item["example"]

        # Generate unique ID: split_name + index
        unique_id = f"{split_name}_{i:04d}"

        # Use pre-computed DAG from Step 1
        dag = item["dag"]

        # Create processed example
        processed = {
            "id": unique_id,
            "question": example["question"],
            "choices": example["choices"],
            "program": example["program"],
            "dag": dag,
            "sparql": example["sparql"],
            "answer": example["answer"],
            "metadata": {
                "split": split_name,
                "original_idx": item["original_idx"],
                "workflow_length": item["workflow_length"],
                "bin": assign_to_bin(item["workflow_length"]),
            },
        }
        processed_examples.append(processed)

    # Step 4: Save preprocessed data
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(processed_examples, f, indent=2)

    print(f"Saved {len(processed_examples)} examples to {output_file}")

    # Return statistics
    return {
        "total_input": len(data),
        "validated": sum(len(b) for b in bins.values()),
        "discarded": discarded_count,
        "selected": len(processed_examples),
        "bin_distribution": {
            bin_idx: sum(
                1 for ex in processed_examples if ex["metadata"]["bin"] == bin_idx
            )
            for bin_idx in range(5)
        },
    }


def main(args):
    """Main preprocessing workflow."""
    # Load knowledge base
    kb_file = Path(args.kb_file)
    print(f"Loading knowledge base from {kb_file}...")
    engine = KoPLEngine.from_json(str(kb_file))
    print("Knowledge base loaded successfully")

    # Preprocess train split
    train_input = Path(args.train_input)
    train_output = Path(args.train_output)
    train_heldout_pool = (
        Path(args.train_heldout_pool) if args.train_heldout_pool else None
    )
    print(f"\n{'=' * 60}")
    print("Preprocessing TRAIN split")
    print(f"{'=' * 60}")
    train_stats = preprocess_split(
        train_input,
        train_output,
        "train",
        engine,
        seed=args.seed,
        heldout_pool_file=train_heldout_pool,
    )

    # Preprocess val split
    val_input = Path(args.val_input)
    val_output = Path(args.val_output)
    print(f"\n{'=' * 60}")
    print("Preprocessing VAL split")
    print(f"{'=' * 60}")
    val_stats = preprocess_split(
        val_input, val_output, "val", engine, seed=args.seed + 1
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print("\nTrain split:")
    print(f"  Input: {train_stats['total_input']} examples")
    print(f"  Validated: {train_stats['validated']} examples")
    print(f"  Discarded: {train_stats['discarded']} examples")
    print(f"  Selected: {train_stats['selected']} examples")
    print(f"  Bin distribution: {train_stats['bin_distribution']}")

    print("\nVal split:")
    print(f"  Input: {val_stats['total_input']} examples")
    print(f"  Validated: {val_stats['validated']} examples")
    print(f"  Discarded: {val_stats['discarded']} examples")
    print(f"  Selected: {val_stats['selected']} examples")
    print(f"  Bin distribution: {val_stats['bin_distribution']}")

    print("\nOutput files:")
    print(f"  Train: {train_output}")
    print(f"  Val: {val_output}")
    if train_heldout_pool:
        print(f"  Train heldout pool: {train_heldout_pool}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess KQA Pro dataset for agent benchmarking"
    )
    parser.add_argument(
        "--kb-file",
        dest="kb_file",
        type=str,
        default="data/kopl_kbqa/kqa_pro/kb.json",
        help="Path to knowledge base file",
    )
    parser.add_argument(
        "--train-input",
        dest="train_input",
        type=str,
        default="data/kopl_kbqa/kqa_pro/train.json",
        help="Path to train input file",
    )
    parser.add_argument(
        "--train-output",
        dest="train_output",
        type=str,
        default="data/kopl_kbqa/kqa_pro/processed/train.v1.json",
        help="Path to train output file",
    )
    parser.add_argument(
        "--train-heldout-pool",
        dest="train_heldout_pool",
        type=str,
        default="data/kopl_kbqa/kqa_pro/processed/train_heldout_pool.v1.json",
        help="Path to save heldout train examples for in-context learning",
    )
    parser.add_argument(
        "--val-input",
        dest="val_input",
        type=str,
        default="data/kopl_kbqa/kqa_pro/val.json",
        help="Path to val input file",
    )
    parser.add_argument(
        "--val-output",
        dest="val_output",
        type=str,
        default="data/kopl_kbqa/kqa_pro/processed/val.v1.json",
        help="Path to val output file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    main(args)
