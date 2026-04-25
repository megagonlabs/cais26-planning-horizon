"""
Retrieve similar in-context examples for GrailQA train/test samples.

Dataset-specific configuration for GrailQA retrieval using shared utilities.

Usage:
    python data/atomic_kbqa/grailqa/scripts/retrieve_grailqa_examples.py --num-candidates 50
    python data/atomic_kbqa/grailqa/scripts/retrieve_grailqa_examples.py --num-candidates 10 --batch-size 128
"""

from pathlib import Path
import argparse
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import shared retrieval utilities
sys.path.insert(0, str(project_root / "data" / "atomic_kbqa" / "scripts" / "utils"))
from retrieval_utils import run_retrieval  # noqa: E402


def main(args):
    """Main retrieval workflow for GrailQA."""

    # Auto-generate output paths if not provided
    if args.train_output is None:
        stem = args.train_input.stem  # e.g., "grailqa_train.v1"
        # Insert .{num_candidates}nn before the extension
        new_name = f"{stem}.{args.num_candidates}nn.json"
        args.train_output = args.train_input.parent / new_name

    if args.test_output is None:
        stem = args.test_input.stem
        new_name = f"{stem}.{args.num_candidates}nn.json"
        args.test_output = args.test_input.parent / new_name

    # Run shared retrieval workflow
    run_retrieval(
        heldout_pool_file=args.heldout_pool,
        train_input_file=args.train_input,
        train_output_file=args.train_output,
        test_input_file=args.test_input,
        test_output_file=args.test_output,
        model_name=args.model_name,
        num_candidates=args.num_candidates,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve similar in-context examples for GrailQA samples"
    )
    parser.add_argument(
        "--heldout-pool",
        dest="heldout_pool",
        type=Path,
        default=Path("data/atomic_kbqa/grailqa/processed/grailqa_train_heldout_pool.v1.json"),
        help="Path to heldout pool JSON file",
    )
    parser.add_argument(
        "--train-input",
        dest="train_input",
        type=Path,
        default=Path("data/atomic_kbqa/grailqa/processed/grailqa_train.v1.json"),
        help="Path to train input file",
    )
    parser.add_argument(
        "--train-output",
        dest="train_output",
        type=Path,
        default=None,
        help="Path to train output file with candidates (default: auto-generated from input path)",
    )
    parser.add_argument(
        "--test-input",
        dest="test_input",
        type=Path,
        default=Path("data/atomic_kbqa/grailqa/processed/grailqa_test.v1.json"),
        help="Path to test input file",
    )
    parser.add_argument(
        "--test-output",
        dest="test_output",
        type=Path,
        default=None,
        help="Path to test output file with candidates (default: auto-generated from input path)",
    )
    parser.add_argument(
        "--model-name",
        dest="model_name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="SentenceTransformer model name",
    )
    parser.add_argument(
        "--num-candidates",
        dest="num_candidates",
        type=int,
        default=50,
        help="Number of top similar candidates to retrieve per sample",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=512,
        help="Batch size for embedding",
    )
    args = parser.parse_args()

    main(args)
