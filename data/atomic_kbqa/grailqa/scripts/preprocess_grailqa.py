"""
Preprocess GrailQA dataset for agent benchmarking.

This script validates function_list programs and creates balanced benchmark splits with:
- Validated DAG conversion (programs that can be converted to DAG)
- Balanced workflow length distribution (125 examples per bin)
- Total of 500 examples per split (train/test)
- Heldout pool for in-context learning retrieval

Usage:
    python data/atomic_kbqa/grailqa/scripts/preprocess_grailqa.py
"""

from pathlib import Path
import argparse
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import shared preprocessing utilities
sys.path.insert(0, str(project_root / "data" / "atomic_kbqa" / "scripts" / "utils"))
from preprocessing_utils import run_preprocessing  # noqa: E402


def assign_to_bin(workflow_length: int) -> int:
    """
    Assign workflow length to one of 4 bins for balanced sampling.

    Dataset-specific binning logic for GrailQA.

    Bins:
        0: [2-3] # note: there is no 1-length workflow in GrailQA
        1: [4-5]
        2: [6-7]
        3: [8+]
    Args:
        workflow_length: Integer workflow length

    Returns:
        Integer bin index (0-3)
    """
    if workflow_length <= 3:
        return 0
    elif workflow_length <= 5:
        return 1
    elif workflow_length <= 7:
        return 2
    else:
        return 3


def preprocess_split(
    input_file, output_file, split_name, seed=42, heldout_pool_file=None
):
    """
    Preprocess a single dataset split (train or test) for GrailQA.

    Wrapper around run_preprocessing from utils with GrailQA-specific binning logic.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        split_name: Name of the split ("train" or "test")
        seed: Random seed for reproducibility
        heldout_pool_file: Optional path to save non-selected examples (for train split only)

    Returns:
        Dict with statistics about preprocessing
    """
    return run_preprocessing(
        input_file=Path(input_file),
        output_file=Path(output_file),
        split_name=split_name,
        assign_to_bin=assign_to_bin,
        num_bins=4,
        target_per_bin=125,
        seed=seed,
        heldout_pool_file=Path(heldout_pool_file) if heldout_pool_file else None,
    )


def main(args):
    """Main preprocessing workflow for GrailQA."""
    # Preprocess train split
    if args.train_input:
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
            seed=args.seed,
            heldout_pool_file=train_heldout_pool,
        )

        print("\nTrain split:")
        print(f"  Input: {train_stats['total_input']} examples")
        print(f"  Converted: {train_stats['converted']} examples")
        print(f"  Selected: {train_stats['selected']} examples")
        print(f"  Bin distribution: {train_stats['bin_distribution']}")
        print(f"  Output: {train_output}")
        if train_heldout_pool:
            print(f"  Heldout pool: {train_heldout_pool}")

    # Preprocess test split
    if args.test_input:
        test_input = Path(args.test_input)
        test_output = Path(args.test_output)
        print(f"\n{'=' * 60}")
        print("Preprocessing TEST split")
        print(f"{'=' * 60}")
        test_stats = preprocess_split(
            test_input, test_output, "test", seed=args.seed + 1
        )

        print("\nTest split:")
        print(f"  Input: {test_stats['total_input']} examples")
        print(f"  Converted: {test_stats['converted']} examples")
        print(f"  Selected: {test_stats['selected']} examples")
        print(f"  Bin distribution: {test_stats['bin_distribution']}")
        print(f"  Output: {test_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess GrailQA dataset for agent benchmarking"
    )
    parser.add_argument(
        "--train-input",
        dest="train_input",
        type=str,
        default="data/atomic_kbqa/grailqa/GrailQA_train.json",
        help="Path to train input file",
    )
    parser.add_argument(
        "--train-output",
        dest="train_output",
        type=str,
        default="data/atomic_kbqa/grailqa/processed/grailqa_train.v1.json",
        help="Path to train output file",
    )
    parser.add_argument(
        "--train-heldout-pool",
        dest="train_heldout_pool",
        type=str,
        default="data/atomic_kbqa/grailqa/processed/grailqa_train_heldout_pool.v1.json",
        help="Path to save heldout train examples for in-context learning",
    )
    parser.add_argument(
        "--test-input",
        dest="test_input",
        type=str,
        default="data/atomic_kbqa/grailqa/GrailQA_test.json",
        help="Path to test input file",
    )
    parser.add_argument(
        "--test-output",
        dest="test_output",
        type=str,
        default="data/atomic_kbqa/grailqa/processed/grailqa_test.v1.json",
        help="Path to test output file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    main(args)
