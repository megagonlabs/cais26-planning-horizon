"""
Retrieve similar in-context examples for KQA Pro train/val samples.

This script:
1. Loads the heldout pool of training examples
2. Embeds all questions using BAAI/bge-base-en-v1.5
3. For each train/val sample, finds top-k most similar heldout examples
4. Attaches candidates to each sample as 'demonstration_candidates' field
5. Saves updated train/val files with attached candidates

Usage:
    python scripts/retrieve_kqa_pro_examples.py --num-candidates 50
    python scripts/retrieve_kqa_pro_examples.py --num-candidates 10 --batch-size 128
"""

from pathlib import Path
from typing import Any
import argparse
import json

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np


DEMONSTRATION_TEMPLATE = """
Question: {question}
Solution:
{dag}
""".strip()

FUNCTION_NAME_ALIASES = {
    "FindAll": "find_all",
    "Find": "find",
    "FilterConcept": "filter_concept",
    "FilterStr": "filter_str",
    "FilterNum": "filter_num",
    "FilterYear": "filter_year",
    "FilterDate": "filter_date",
    "QFilterStr": "qfilter_str",
    "QFilterNum": "qfilter_num",
    "QFilterYear": "qfilter_year",
    "QFilterDate": "qfilter_date",
    "Relate": "relate",
    "And": "and",
    "Or": "or",
    "QueryName": "query_name",
    "Count": "count",
    "QueryAttr": "query_attr",
    "QueryAttrUnderCondition": "query_attr_under_condition",
    "QueryRelation": "query_relation",
    "SelectBetween": "select_between",
    "SelectAmong": "select_among",
    "VerifyStr": "verify_str",
    "VerifyNum": "verify_num",
    "VerifyYear": "verify_year",
    "VerifyDate": "verify_date",
    "QueryAttrQualifier": "query_attr_qualifier",
    "QueryRelationQualifier": "query_relation_qualifier",
}


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path):
    """Save data to JSON file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def compute_embeddings(texts, model, batch_size=256, desc="Embedding"):
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
    )
    return embeddings


def retrieve_top_k_similar(query_embeddings, pool_embeddings, k=50):
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
    lines = []
    for i, step in enumerate(dag):
        func = FUNCTION_NAME_ALIASES[step["function"]]
        deps = step["dependencies"]
        inputs = step.get("inputs", [])

        # Format dependencies
        dep_str = ", ".join([f"${d}" for d in deps]) if deps else ""

        # Format inputs
        input_str = ", ".join([f'"{inp}"' for inp in inputs]) if inputs else ""

        # Combine
        args = []
        if dep_str:
            args.append(dep_str)
        if input_str:
            args.append(input_str)

        args_str = ", ".join(args)
        lines.append(f"${i} = {func}({args_str})")

    # Add finish
    lines.append(f"finish(${len(dag) - 1})")

    return "\n".join(lines)


def process_split(
    split_file,
    output_file,
    heldout_pool,
    pool_embeddings,
    model,
    num_candidates,
    batch_size,
    split_name,
):
    """
    Process a single split (train or val) and attach in-context candidates.

    Args:
        split_file: Path to input split JSON file
        output_file: Path to output split JSON file with candidates
        heldout_pool: List of heldout pool examples
        pool_embeddings: numpy array of embeddings for heldout pool questions
        model: SentenceTransformer model
        num_candidates: Number of top similar candidates to retrieve
        batch_size: Batch size for embedding
        split_name: Name of the split (train or val)
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
                    "index": pool_example["index"],
                    "question": pool_example["question"],
                    "answer": pool_example["answer"],
                    "dag": pool_example["dag"],
                    "text": DEMONSTRATION_TEMPLATE.format(
                        question=pool_example["question"],
                        dag=format_dag(pool_example["dag"]),
                    ),
                    "similarity": float(top_k_scores[i, j]),
                }
            )
        example["demonstration_candidates"] = candidates

    # Save updated split
    print(f"Saving to {output_file}...")
    save_json(split_data, output_file)
    print(f"Saved {len(split_data)} examples with {num_candidates} candidates each")


def main(args):
    """Main retrieval workflow."""

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

    if args.val_output is None:
        stem = args.val_input.stem
        if "." in stem:
            base_name = stem.rsplit(".", 1)[0]
            version = stem.rsplit(".", 1)[1]
            new_name = f"{base_name}.{version}.{args.num_candidates}nn.json"
        else:
            new_name = f"{stem}.{args.num_candidates}nn.json"
        args.val_output = args.val_input.parent / new_name

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
        pool_questions, model, batch_size=args.batch_size, desc="Embedding heldout pool"
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

    # Process val split
    process_split(
        args.val_input,
        args.val_output,
        heldout_pool,
        pool_embeddings,
        model,
        args.num_candidates,
        args.batch_size,
        "val",
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
    print(f"  Val: {args.val_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve similar in-context examples for KQA Pro samples"
    )
    parser.add_argument(
        "--heldout-pool",
        dest="heldout_pool",
        type=Path,
        default=Path("data/kopl_kbqa/kqa_pro/processed/train_heldout_pool.v1.json"),
        help="Path to heldout pool JSON file",
    )
    parser.add_argument(
        "--train-input",
        dest="train_input",
        type=Path,
        default=Path("data/kopl_kbqa/kqa_pro/processed/train.v1.json"),
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
        "--val-input",
        dest="val_input",
        type=Path,
        default=Path("data/kopl_kbqa/kqa_pro/processed/val.v1.json"),
        help="Path to val input file",
    )
    parser.add_argument(
        "--val-output",
        dest="val_output",
        type=Path,
        default=None,
        help="Path to val output file with candidates (default: auto-generated from input path)",
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
