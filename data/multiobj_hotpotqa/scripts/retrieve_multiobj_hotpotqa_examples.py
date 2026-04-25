"""
Retrieve similar in-context examples for Multi-objective HotpotQA train/test samples.

This script:
1. Loads the heldout pool of training examples
2. Embeds all questions using BAAI/bge-base-en-v1.5
3. For each train/test sample, finds top-k most similar heldout examples
4. Attaches candidates to each sample as 'demonstration_candidates' field
5. Saves updated train/test files with attached candidates

Usage:
    python data/multiobj_hotpotqa/scripts/retrieve_multiobj_hotpotqa_examples.py --num-candidates 50
    python data/multiobj_hotpotqa/scripts/retrieve_multiobj_hotpotqa_examples.py --num-candidates 10 --batch-size 128
"""

from pathlib import Path
from typing import Any
import argparse
import json
import sys

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from data.multiobj_hotpotqa.scripts.utils import (  # noqa: E402
    load_json,
    save_json,
)


DEMONSTRATION_TEMPLATE = """
Question: {question}
Solution:
{dag}
""".strip()


def compute_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 256
) -> np.ndarray:
    """
    Compute embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        model: SentenceTransformer model instance
        batch_size: Batch size for encoding

    Returns:
        np.ndarray: Embeddings array of shape [len(texts), embedding_dim]
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


def retrieve_top_k_similar(
    query_embeddings: np.ndarray,
    pool_embeddings: np.ndarray,
    k: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve top-k most similar examples for each query.

    Args:
        query_embeddings: np.ndarray of shape [num_queries, embedding_dim]
        pool_embeddings: np.ndarray of shape [num_pool, embedding_dim]
        k: Number of top similar examples to retrieve

    Returns:
        tuple[np.ndarray, np.ndarray]: (top_k_indices, top_k_scores)
            - top_k_indices: np.ndarray of shape [num_queries, k] containing indices
            - top_k_scores: np.ndarray of shape [num_queries, k] containing cosine similarity scores
    """
    # Compute cosine similarity (embeddings are already normalized)
    # similarity[i, j] = cosine_similarity(query_i, pool_j)
    similarity = np.dot(query_embeddings, pool_embeddings.T)  # [num_queries, num_pool]

    # Get indices of top-k most similar examples for each query
    # argsort returns indices in ascending order, so we take the last k and reverse
    top_k_indices = np.argsort(similarity, axis=1)[:, -k:][:, ::-1]  # [num_queries, k]

    # Get the corresponding similarity scores
    top_k_scores = np.take_along_axis(
        similarity, top_k_indices, axis=1
    )  # [num_queries, k]

    return top_k_indices, top_k_scores


def format_dag(dag: list[dict[str, Any]]) -> str:
    """
    Format a DAG structure into a human-readable string.

    Args:
        dag: List of DAG steps with function, dependencies, and inputs

    Returns:
        Formatted DAG string with numbered steps
    """
    lines = []
    for i, step in enumerate(dag):
        func = step["function"]
        inputs = step.get("inputs", [])

        # Format inputs
        if not inputs:
            raise ValueError(f"DAG step {i} has no inputs.")
        else:
            if func == "finish":
                input_str = json.dumps(inputs)
            else:
                assert len(inputs) == 1, f"Expected single input for step {i}, got {inputs}"
                input_str = f'"{inputs[0]}"'

        # Combine arguments
        args = []
        if input_str:
            args.append(input_str)

        args_str = ", ".join(args)
        lines.append(f"${i} = {func}({args_str})")

    return "\n".join(lines)


def process_split(
    split_file: Path,
    output_file: Path,
    heldout_pool: list[dict[str, Any]],
    pool_embeddings: np.ndarray,
    model: SentenceTransformer,
    num_candidates: int,
    batch_size: int,
    split_name: str,
) -> None:
    """
    Process a single split (train or test) and attach in-context candidates.

    Args:
        split_file: Path to input split JSON file
        output_file: Path to output split JSON file with candidates
        heldout_pool: List of heldout pool examples
        pool_embeddings: numpy array of embeddings for heldout pool questions
        model: SentenceTransformer model
        num_candidates: Number of top similar candidates to retrieve
        batch_size: Batch size for embedding
        split_name: Name of the split (train or test)
    """
    print(f"\n{'=' * 60}")
    print(f"Processing {split_name} split")
    print(f"{'=' * 60}")

    # Load split data
    print(f"Loading {split_file}...")
    split_data = load_json(split_file)
    print(f"Loaded {len(split_data)} examples")

    # Extract questions
    questions = [example["question"] for example in split_data]
    # Extract number of components
    num_components = [example["metadata"]["k"] for example in split_data]

    # Embed questions
    print(f"Embedding {len(questions)} questions...")
    query_embeddings = compute_embeddings(
        questions, model, batch_size=batch_size,
    )

    # Extract number of components for each pool example
    pool_num_components = [example["metadata"]["k"] for example in heldout_pool]

    # Attach candidates to each example
    print("Retrieving and attaching candidates to examples...")
    for i, example in enumerate(tqdm(split_data, desc="Attaching candidates")):
        query_k = num_components[i]
        query_emb = query_embeddings[i : i + 1]  # Keep 2D shape [1, dim]

        # Filter pool to only include examples with same num_components
        filtered_indices = [
            idx for idx, k in enumerate(pool_num_components) if k == query_k
        ]

        if len(filtered_indices) == 0:
            # No matching examples in pool - skip this query
            example["demonstration_candidates"] = []
            continue

        # Get embeddings for filtered pool
        filtered_embeddings = pool_embeddings[filtered_indices]  # [num_filtered, dim]

        # Retrieve top-k from filtered pool
        k = min(
            num_candidates, len(filtered_indices)
        )  # Handle case where pool is smaller than k
        _, top_k_scores = retrieve_top_k_similar(query_emb, filtered_embeddings, k=k)

        # Get indices in filtered pool, then map back to original pool indices
        filtered_top_k_indices = np.argsort(
            np.dot(query_emb, filtered_embeddings.T)[0]
        )[-k:][::-1]
        original_pool_indices = [
            filtered_indices[idx] for idx in filtered_top_k_indices
        ]

        # Attach candidates
        candidates = []
        for j, original_idx in enumerate(original_pool_indices):
            pool_example = heldout_pool[original_idx]
            candidates.append(
                {
                    "id": pool_example["id"],
                    "question": pool_example["question"],
                    "answers": pool_example["answers"],
                    "dag": pool_example["dag"],
                    "text": DEMONSTRATION_TEMPLATE.format(
                        question=pool_example["question"],
                        dag=format_dag(pool_example["dag"]),
                    ),
                    "similarity": float(top_k_scores[0, j]),
                }
            )
        example["demonstration_candidates"] = candidates

    # Save updated split
    print(f"Saving to {output_file}...")
    save_json(split_data, output_file)
    print(f"Saved {len(split_data)} examples with {num_candidates} candidates each")


def main(args: argparse.Namespace) -> None:
    """
    Main retrieval workflow.

    Args:
        args: Parsed command-line arguments
    """

    # Auto-generate output paths if not provided
    if args.train_output is None:
        stem = args.train_input.stem  # e.g., "train.v1"
        # Insert .{num_candidates}nn before the last extension
        if "." in stem:
            base_name = stem.rsplit(".", 1)[0]  # e.g., "train"
            version = stem.rsplit(".", 1)[1]  # e.g., "v1"
            new_name = f"{base_name}.{version}.{args.num_candidates}nn.json"
        else:
            new_name = f"{stem}.{args.num_candidates}nn.json"
        args.train_output = args.train_input.parent / new_name

    if args.test_output is None:
        stem = args.test_input.stem
        if "." in stem:
            base_name = stem.rsplit(".", 1)[0]
            version = stem.rsplit(".", 1)[1]
            new_name = f"{base_name}.{version}.{args.num_candidates}nn.json"
        else:
            new_name = f"{stem}.{args.num_candidates}nn.json"
        args.test_output = args.test_input.parent / new_name

    # Load model
    print(f"Loading model: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    print(
        f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}"
    )

    # Load heldout pool
    print(f"\nLoading heldout pool from {args.heldout_pool}...")
    heldout_pool = load_json(args.heldout_pool)
    print(f"Loaded {len(heldout_pool)} heldout examples")

    # Embed heldout pool questions
    print("\nEmbedding heldout pool questions...")
    pool_questions = [example["question"] for example in heldout_pool]
    pool_embeddings = compute_embeddings(
        pool_questions, model, batch_size=args.batch_size,
    )
    print(f"Pool embeddings shape: {pool_embeddings.shape}")

    # Process train split
    process_split(
        args.train_input,
        args.train_output,
        heldout_pool,
        pool_embeddings,
        model,
        args.num_candidates,
        args.batch_size,
        "train",
    )

    # Process test split
    process_split(
        args.test_input,
        args.test_output,
        heldout_pool,
        pool_embeddings,
        model,
        args.num_candidates,
        args.batch_size,
        "test",
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model: {args.model_name}")
    print(f"Heldout pool size: {len(heldout_pool)}")
    print(f"Candidates per sample: {args.num_candidates}")
    print("\nOutput files:")
    print(f"  Train: {args.train_output}")
    print(f"  Test: {args.test_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve similar in-context examples for Multi-objective HotpotQA samples"
    )
    parser.add_argument(
        "--heldout-pool",
        dest="heldout_pool",
        type=Path,
        default=Path(
            "data/multiobj_hotpotqa/processed/train_heldout_pool.v1.annotated.json"
        ),
        help="Path to heldout pool JSON file",
    )
    parser.add_argument(
        "--train-input",
        dest="train_input",
        type=Path,
        default=Path("data/multiobj_hotpotqa/processed/train.v1.annotated.json"),
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
        default=Path("data/multiobj_hotpotqa/processed/test.v1.annotated.json"),
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
