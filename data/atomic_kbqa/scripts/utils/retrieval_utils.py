"""
Shared utilities for retrieving similar in-context examples for Atomic KBQA datasets.

This module provides functions to:
1. Load and embed question pools using sentence transformers
2. Retrieve top-k similar examples based on cosine similarity
3. Format DAG representations for demonstrations
4. Process dataset splits with attached demonstration candidates
"""

from pathlib import Path
from typing import Any
import json

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np


DEMONSTRATION_TEMPLATE = """
Question: {question}
Solution:
{dag}
""".strip()


def load_json(file_path: Path) -> Any:
    """Load JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data: Any, file_path: Path) -> None:
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def compute_embeddings(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = 256,
    desc: str = "Embedding",
) -> np.ndarray:
    """
    Compute embeddings for a list of texts.

    Args:
        texts: List of text strings
        model: SentenceTransformer model
        batch_size: Batch size for encoding
        desc: Description for progress bar

    Returns:
        numpy array of embeddings (shape: [len(texts), embedding_dim])
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
        convert_to_tensor=False,
        device=None,  # Use default device
    )
    # Ensure embeddings are numpy arrays
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    return embeddings


def retrieve_top_k_similar(
    query_embeddings: np.ndarray, pool_embeddings: np.ndarray, k: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve top-k most similar examples for each query.

    Args:
        query_embeddings: numpy array of shape [num_queries, embedding_dim]
        pool_embeddings: numpy array of shape [num_pool, embedding_dim]
        k: Number of top similar examples to retrieve

    Returns:
        Tuple of (top_k_indices, top_k_scores):
        - top_k_indices: numpy array of shape [num_queries, k] containing indices
        - top_k_scores: numpy array of shape [num_queries, k] containing cosine similarity scores
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
    """Format DAG as executable function calls."""
    lines = []
    for i, step in enumerate(dag):
        func = step["function"]
        inputs = step.get("inputs", {})
        # Combine
        args = [f'{var}="{val}"' for var, val in inputs.items()]

        args_str = ", ".join(args)
        if func == "finish":
            # no assignment for finish
            lines.append(f"{func}({args_str})")
        else:
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

    if len(heldout_pool) == 0:
        print("Warning: Heldout pool is empty. No candidates will be retrieved.")
        return

    # Load split data
    print(f"Loading {split_file}...")
    split_data = load_json(split_file)
    print(f"Loaded {len(split_data)} examples")

    # Extract questions
    questions = [example["question"] for example in split_data]

    # Embed questions
    print(f"Embedding {len(questions)} questions...")
    query_embeddings = compute_embeddings(
        questions, model, batch_size=batch_size, desc=f"Embedding {split_name}"
    )

    # Retrieve top-k similar examples
    print(f"Retrieving top-{num_candidates} similar examples for each question...")
    top_k_indices, top_k_scores = retrieve_top_k_similar(
        query_embeddings, pool_embeddings, k=num_candidates
    )

    # Attach candidates to each example
    print("Attaching candidates to examples...")
    for i, example in enumerate(tqdm(split_data, desc="Attaching candidates")):
        candidates = []
        for j, idx in enumerate(top_k_indices[i]):
            pool_example = heldout_pool[idx]
            candidates.append(
                {
                    "ID": pool_example.get("ID", pool_example.get("id", "")),
                    "question": pool_example["question"],
                    "answer": pool_example["answer"],
                    "dag": pool_example["dag"],
                    "text": DEMONSTRATION_TEMPLATE.format(
                        question=pool_example["question"],
                        dag=format_dag(pool_example["dag"]),
                    ),
                    "function_list": pool_example["function_list"],
                    "similarity": float(top_k_scores[i, j]),
                }
            )
        example["demonstration_candidates"] = candidates

    # Save updated split
    print(f"Saving to {output_file}...")
    save_json(split_data, output_file)
    print(f"Saved {len(split_data)} examples with {num_candidates} candidates each")


def run_retrieval(
    heldout_pool_file: Path,
    train_input_file: Path,
    train_output_file: Path,
    test_input_file: Path,
    test_output_file: Path,
    model_name: str = "BAAI/bge-base-en-v1.5",
    num_candidates: int = 50,
    batch_size: int = 512,
) -> None:
    """
    Main retrieval workflow for Atomic KBQA datasets.

    Args:
        heldout_pool_file: Path to heldout pool JSON file
        train_input_file: Path to train input file
        train_output_file: Path to train output file with candidates
        test_input_file: Path to test input file
        test_output_file: Path to test output file with candidates
        model_name: SentenceTransformer model name
        num_candidates: Number of top similar candidates to retrieve per sample
        batch_size: Batch size for embedding
    """
    # Load model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(
        f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}"
    )

    # Load heldout pool
    print(f"\nLoading heldout pool from {heldout_pool_file}...")
    heldout_pool = load_json(heldout_pool_file)
    print(f"Loaded {len(heldout_pool)} heldout examples")

    # Embed heldout pool questions
    print("\nEmbedding heldout pool questions...")
    pool_questions = [example["question"] for example in heldout_pool]
    pool_embeddings = compute_embeddings(
        pool_questions, model, batch_size=batch_size, desc="Embedding heldout pool"
    )
    print(f"Pool embeddings shape: {pool_embeddings.shape}")

    # Process train split (uses only heldout pool)
    process_split(
        train_input_file,
        train_output_file,
        heldout_pool,
        pool_embeddings,
        model,
        num_candidates,
        batch_size,
        "train",
    )

    # Load training data to combine with heldout pool for test split
    print(f"\nLoading training data from {train_input_file}...")
    train_data = load_json(train_input_file)
    print(f"Loaded {len(train_data)} training examples")

    # Combine heldout pool with training data for test split
    combined_pool = heldout_pool + train_data
    print(f"Combined pool size: {len(combined_pool)} (heldout: {len(heldout_pool)}, train: {len(train_data)})")

    # Embed training questions and combine with heldout embeddings
    print("\nEmbedding training questions...")
    train_questions = [example["question"] for example in train_data]
    train_embeddings = compute_embeddings(
        train_questions, model, batch_size=batch_size, desc="Embedding training"
    )
    if len(pool_embeddings) == 0:
        combined_embeddings = train_embeddings
    else:
        combined_embeddings = np.vstack([pool_embeddings, train_embeddings])
    print(f"Combined embeddings shape: {combined_embeddings.shape}")

    # Process test split (uses heldout pool + training data)
    process_split(
        test_input_file,
        test_output_file,
        combined_pool,
        combined_embeddings,
        model,
        num_candidates,
        batch_size,
        "test",
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model: {model_name}")
    print(f"Heldout pool size: {len(heldout_pool)}")
    print(f"Training data size: {len(train_data)}")
    print(f"Combined pool for test: {len(combined_pool)}")
    print(f"Candidates per sample: {num_candidates}")
    print("\nOutput files:")
    print(f"  Train: {train_output_file} (candidates from heldout pool only)")
    print(f"  Test: {test_output_file} (candidates from heldout pool + train)")
