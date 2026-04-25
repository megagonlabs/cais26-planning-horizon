"""
Preprocess HotpotQA dataset for multi-objective question answering benchmarking.

This script creates multi-objective variants of HotpotQA by combining k questions:
- k=1: Original single-objective questions (100 bridge + 100 comparison)
- k=2,3,4,5: Multi-objective questions with random mix of bridge/comparison
- DAG field is left empty (placeholder) for later annotation by annotate_dag.py
- Balanced sampling across question types

Input files:
- Validated bridge questions (JSONL): Output from batch_validate_download.py
    Contains bridge questions with `valid_reasoning_structure` field
- Original HotpotQA file (JSON): For comparison questions (not validated)

Usage:
    # Preprocess dev split (test)
    uv run python data/multiobj_hotpotqa/scripts/preprocess_multiobj_hotpotqa.py \
        --validated-bridge data/multiobj_hotpotqa/batch_validation/gpt-4.1-2025-04-14/2025-12-15-23-02-19/validated.jsonl \
        --original-hotpotqa data/multiobj_hotpotqa/hotpot_dev_distractor_v1.json \
        --output data/multiobj_hotpotqa/processed/test.v1.json \
        --split test

    # Preprocess train split
    uv run python data/multiobj_hotpotqa/scripts/preprocess_multiobj_hotpotqa.py \
        --validated-bridge data/multiobj_hotpotqa/batch_validation/.../validated.jsonl \
        --original-hotpotqa data/multiobj_hotpotqa/hotpot_train_v1.1.json \
        --output data/multiobj_hotpotqa/processed/train.v1.json \
        --split train \
        --heldout-pool data/multiobj_hotpotqa/processed/train_heldout_pool.v1.json
"""

from pathlib import Path
from typing import Any
import argparse
import json
import random
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from planning.services.openai import OpenAIClient  # noqa: E402
from data.multiobj_hotpotqa.scripts.utils import (  # noqa: E402
    load_json,
    load_jsonl,
)



def normalize_hotpotqa_example(raw_example: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a raw HotpotQA example to the internal format used by this script.

    Raw HotpotQA format:
        {
            "_id": "...",
            "question": "...",
            "answer": "...",
            "supporting_facts": [["Title", sent_id], ...],
            "context": [["Title", ["sent0", "sent1", ...]], ...],
            "type": "bridge" | "comparison",
            "level": "easy" | "medium" | "hard"
        }

    Internal format:
        {
            "id": "...",
            "question": "...",
            "golden_answers": ["..."],
            "metadata": {
                "type": "bridge" | "comparison",
                "level": "...",
                "supporting_facts": {"title": [...], "sent_id": [...]},
                "context": {"title": [...], "sentences": [[...], ...]}
            }
        }
    """
    # Parse supporting_facts from list of [title, sent_id] pairs
    sf_titles = []
    sf_sent_ids = []
    for item in raw_example.get("supporting_facts", []):
        sf_titles.append(item[0])
        sf_sent_ids.append(item[1])

    # Parse context from list of [title, sentences] pairs
    ctx_titles = []
    ctx_sentences = []
    for item in raw_example.get("context", []):
        ctx_titles.append(item[0])
        ctx_sentences.append(item[1])

    return {
        "id": raw_example["_id"],
        "question": raw_example["question"],
        "golden_answers": [raw_example["answer"]],
        "metadata": {
            "type": raw_example["type"],
            "level": raw_example["level"],
            "supporting_facts": {
                "title": sf_titles,
                "sent_id": sf_sent_ids,
            },
            "context": {
                "title": ctx_titles,
                "sentences": ctx_sentences,
            },
        },
    }


def load_validated_bridge_examples(validated_file: Path) -> list[dict[str, Any]]:
    """
    Load validated bridge examples from JSONL file.

    Only returns examples with valid_reasoning_structure=True.
    """
    raw_data = load_jsonl(validated_file)
    valid_examples = []
    for raw in raw_data:
        if raw.get("valid_reasoning_structure", False):
            valid_examples.append(normalize_hotpotqa_example(raw))
    return valid_examples


def load_comparison_examples(original_file: Path) -> list[dict[str, Any]]:
    """
    Load comparison examples from original HotpotQA JSON file.

    Comparison questions are not validated (assumed to follow standard template).
    """
    raw_data = load_json(original_file)
    comparison_examples = []
    for raw in raw_data:
        if raw.get("type") == "comparison":
            comparison_examples.append(normalize_hotpotqa_example(raw))
    return comparison_examples


def extract_supporting_sentences(example: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract only the supporting sentences from context.

    Returns:
        List of dicts with {title, sent_id, text}
    """
    supporting_facts = example["metadata"]["supporting_facts"]
    context = example["metadata"]["context"]

    supporting_sentences = []
    for title, sent_id in zip(supporting_facts["title"], supporting_facts["sent_id"]):
        # Find the article in context
        try:
            article_idx = context["title"].index(title)
            sentence_text = context["sentences"][article_idx][sent_id]
            supporting_sentences.append(
                {"title": title, "sent_id": sent_id, "text": sentence_text}
            )
        except (ValueError, IndexError):
            # Skip if not found (shouldn't happen with valid data)
            continue

    return supporting_sentences


def sample_k1(
    bridge_examples: list[dict[str, Any]],
    comparison_examples: list[dict[str, Any]],
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Sample balanced examples for k=1 (50 bridge + 50 comparison).

    Returns:
        List of sampled individual examples
    """
    random.seed(seed)

    half = num_samples // 2
    sampled_bridge = random.sample(bridge_examples, half)
    sampled_comparison = random.sample(comparison_examples, half)

    combined = sampled_bridge + sampled_comparison
    random.shuffle(combined)

    return combined


def sample_multi_objective(
    bridge_examples: list[dict[str, Any]],
    comparison_examples: list[dict[str, Any]],
    k: int,
    num_samples: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """
    Sample multi-objective examples for k>=2.

    For each sample, randomly choose i ~ Uniform[0, k] and sample
    i bridge + (k-i) comparison questions. Examples can be reused across
    different samples, but not within the same sample.

    Args:
        bridge_examples: Pool of bridge examples
        comparison_examples: Pool of comparison examples
        k: Number of questions per sample
        num_samples: Number of multi-objective samples to create
        seed: Random seed

    Returns:
        List of samples, each containing k questions
    """
    random.seed(seed)

    samples = []
    for _ in range(num_samples):
        # Randomly choose how many bridge questions (0 to k)
        num_bridge = random.randint(0, k)
        num_comparison = k - num_bridge

        # Check if we have enough examples in the pool
        if (
            len(bridge_examples) < num_bridge
            or len(comparison_examples) < num_comparison
        ):
            raise ValueError(
                f"Not enough examples for k={k}: "
                f"need {num_bridge} bridge and {num_comparison} comparison, "
                f"have {len(bridge_examples)} bridge and {len(comparison_examples)} comparison"
            )

        # Sample questions (can reuse across samples, but not within the same sample)
        sample_components = []
        if num_bridge > 0:
            sampled_bridge = random.sample(bridge_examples, num_bridge)
            sample_components.extend(sampled_bridge)

        if num_comparison > 0:
            sampled_comparison = random.sample(comparison_examples, num_comparison)
            sample_components.extend(sampled_comparison)

        # Shuffle the order
        random.shuffle(sample_components)
        samples.append(sample_components)

    return samples


def combine_questions(components: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """
    Combine multiple questions into multi-objective format.

    Args:
        components: List of component questions
        k: Number of questions (should equal len(components))

    Returns:
        Combined multi-objective example (without DAG)
    """
    assert len(components) == k

    # Create numbered question text
    if k == 1:
        combined_question = components[0]["question"]
    else:
        question_parts = [
            f"{i + 1}. {comp['question']}" for i, comp in enumerate(components)
        ]
        combined_question = "\n".join(question_parts)
    # Combine answers
    combined_answers = [comp["golden_answers"][0] for comp in components]

    # Extract metadata from components
    component_metadata = []
    for comp in components:
        comp_meta = {
            "id": comp["id"],
            "type": comp["metadata"]["type"],
            "level": comp["metadata"]["level"],
            "supporting_sentences": extract_supporting_sentences(comp),
        }
        component_metadata.append(comp_meta)

    return {
        "question": combined_question,
        "answers": combined_answers,
        "component_metadata": component_metadata,
    }


def create_processed_example(
    combined_data: dict[str, Any],
    dag: list[dict[str, Any]],
    example_id: str,
    k: int,
    split_name: str,
) -> dict[str, Any]:
    """
    Create final processed example with all fields.

    Args:
        combined_data: Combined question data from combine_questions()
        dag: DAG structure
        example_id: Unique ID for this example
        k: Number of sub-questions
        split_name: "train" or "test"

    Returns:
        Complete processed example
    """
    return {
        "id": example_id,
        "question": combined_data["question"],
        "answers": combined_data["answers"],
        "dag": dag,
        "metadata": {
            "k": k,
            "split": split_name,
            "components": combined_data["component_metadata"],
        },
    }


def process_split(
    validated_bridge_file: Path,
    original_hotpotqa_file: Path,
    output_file: Path,
    split_name: str,
    samples_per_k: int = 200,
    heldout_pool_file: Path = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Process a single split (train or test). DAG field is left empty for later annotation.

    Args:
        validated_bridge_file: Path to validated bridge questions JSONL file
        original_hotpotqa_file: Path to original HotpotQA JSON file (for comparison questions)
        output_file: Path to output JSON file
        split_name: "train" or "test"
        samples_per_k: Number of samples per k value (default: 200)
        heldout_pool_file: Optional path to save heldout pool (train only)
        seed: Random seed

    Returns:
        Statistics dict
    """
    print(f"\n{'=' * 60}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'=' * 60}")

    # Load validated bridge examples
    print(f"Loading validated bridge examples from {validated_bridge_file}...")
    bridge_examples = load_validated_bridge_examples(validated_bridge_file)
    print(f"Valid bridge examples: {len(bridge_examples)}")

    # Load comparison examples from original file
    print(f"Loading comparison examples from {original_hotpotqa_file}...")
    comparison_examples = load_comparison_examples(original_hotpotqa_file)
    print(f"Comparison examples: {len(comparison_examples)}")

    # Process each k value
    all_processed = []
    k_stats = {}
    used_bridge_ids = set()  # Track unique bridge example IDs used
    used_comparison_ids = set()  # Track unique comparison example IDs used

    for k in [1, 2, 3, 4, 5]:
        print(f"\nProcessing k={k}...")
        k_seed = seed + k

        if k == 1:
            # Sample balanced k=1 (half bridge + half comparison)
            sampled = sample_k1(bridge_examples, comparison_examples, samples_per_k, k_seed)

            # Track used IDs by type
            for ex in sampled:
                if ex["metadata"]["type"] == "bridge":
                    used_bridge_ids.add(ex["id"])
                else:
                    used_comparison_ids.add(ex["id"])

            # Process each example
            for idx, example in enumerate(sampled):
                # Create combined data
                combined_data = combine_questions([example], k=1)

                # Create processed example (DAG left empty for later annotation)
                example_id = f"{split_name}_k{k}_{idx:04d}"
                processed = create_processed_example(
                    combined_data, [], example_id, k, split_name
                )
                all_processed.append(processed)

            k_stats[k] = {
                "total": len(sampled),
                "bridge": sum(
                    1 for ex in sampled if ex["metadata"]["type"] == "bridge"
                ),
                "comparison": sum(
                    1 for ex in sampled if ex["metadata"]["type"] == "comparison"
                ),
            }

        else:
            # Sample multi-objective examples
            samples = sample_multi_objective(
                bridge_examples,
                comparison_examples,
                k=k,
                num_samples=samples_per_k,
                seed=k_seed,
            )

            # Track used IDs by type
            for sample in samples:
                for ex in sample:
                    if ex["metadata"]["type"] == "bridge":
                        used_bridge_ids.add(ex["id"])
                    else:
                        used_comparison_ids.add(ex["id"])

            # Process each multi-objective sample
            for idx, components in enumerate(samples):
                # Combine questions
                combined_data = combine_questions(components, k)

                # Create processed example (DAG left empty for later annotation)
                example_id = f"{split_name}_k{k}_{idx:04d}"
                processed = create_processed_example(
                    combined_data, [], example_id, k, split_name
                )
                all_processed.append(processed)

            # Compute stats
            type_counts = {}
            for sample in samples:
                types = tuple(sorted([ex["metadata"]["type"] for ex in sample]))
                type_counts[types] = type_counts.get(types, 0) + 1

            k_stats[k] = {"total": len(samples), "type_combinations": type_counts}

    # Save processed data
    print(f"\nSaving {len(all_processed)} examples to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_processed, f, indent=2)

    # Save heldout pool (train only)
    if heldout_pool_file and split_name == "train":
        print("\nCreating heldout pool...")
        heldout_pool = []

        # For each k, sample 400 examples (can overlap with main split)
        for k in [1, 2, 3, 4, 5]:
            k_seed = seed + 1000 + k  # Different seed for heldout

            if k == 1:
                # Sample 400 individual examples (200 bridge + 200 comparison)
                if len(bridge_examples) < 200 or len(comparison_examples) < 200:
                    print(f"Warning: Not enough examples for k={k} heldout pool")
                    print(
                        f"  Available: {len(bridge_examples)} bridge, {len(comparison_examples)} comparison"
                    )
                    continue

                random.seed(k_seed)
                sampled_bridge = random.sample(bridge_examples, 200)
                sampled_comparison = random.sample(comparison_examples, 200)
                sampled = sampled_bridge + sampled_comparison
                random.shuffle(sampled)

                # Track used IDs by type
                used_bridge_ids.update(ex["id"] for ex in sampled_bridge)
                used_comparison_ids.update(ex["id"] for ex in sampled_comparison)

                for idx, example in enumerate(sampled):
                    # Create heldout entry (DAG left empty for later annotation)
                    combined_data = combine_questions([example], k=1)
                    heldout_id = f"heldout_k{k}_{idx:04d}"
                    heldout_entry = {
                        "id": heldout_id,
                        "question": combined_data["question"],
                        "answers": combined_data["answers"],
                        "dag": [],
                        "metadata": {
                            "k": k,
                            "components": combined_data["component_metadata"],
                        },
                    }
                    heldout_pool.append(heldout_entry)

            else:
                # Sample 400 multi-objective examples
                heldout_samples = []

                random.seed(k_seed)
                for _ in range(400):
                    num_bridge = random.randint(0, k)
                    num_comparison = k - num_bridge

                    if (
                        len(bridge_examples) < num_bridge
                        or len(comparison_examples) < num_comparison
                    ):
                        print(
                            f"Warning: Insufficient examples for k={k} heldout sample"
                        )
                        break

                    # Sample components (can reuse across samples)
                    sample_components = []
                    if num_bridge > 0:
                        sampled_bridge = random.sample(bridge_examples, num_bridge)
                        sample_components.extend(sampled_bridge)
                    if num_comparison > 0:
                        sampled_comparison = random.sample(
                            comparison_examples, num_comparison
                        )
                        sample_components.extend(sampled_comparison)

                    random.shuffle(sample_components)
                    heldout_samples.append(sample_components)

                # Track used IDs by type
                for sample in heldout_samples:
                    for ex in sample:
                        if ex["metadata"]["type"] == "bridge":
                            used_bridge_ids.add(ex["id"])
                        else:
                            used_comparison_ids.add(ex["id"])

                # Process heldout samples
                for idx, components in enumerate(heldout_samples):
                    # Combine questions (DAG left empty for later annotation)
                    combined_data = combine_questions(components, k)

                    # Create heldout entry
                    heldout_id = f"heldout_k{k}_{idx:04d}"
                    heldout_entry = {
                        "id": heldout_id,
                        "question": combined_data["question"],
                        "answers": combined_data["answers"],
                        "dag": [],
                        "metadata": {
                            "k": k,
                            "components": combined_data["component_metadata"],
                        },
                    }
                    heldout_pool.append(heldout_entry)

        # Save heldout pool
        print(f"Saving {len(heldout_pool)} heldout examples to {heldout_pool_file}...")
        with open(heldout_pool_file, "w") as f:
            json.dump(heldout_pool, f, indent=2)

    # Print statistics
    print(f"\n{split_name.upper()} Statistics:")
    for k, stats in k_stats.items():
        print(f"\nk={k}:")
        print(f"  Total: {stats['total']}")
        if k == 1:
            print(f"  Bridge: {stats['bridge']}")
            print(f"  Comparison: {stats['comparison']}")
        else:
            print("  Type combinations:")
            for types, count in sorted(stats["type_combinations"].items()):
                print(f"    {types}: {count}")

    return {"total_examples": len(all_processed), "k_stats": k_stats, "unique_bridge_used": len(used_bridge_ids), "unique_comparison_used": len(used_comparison_ids)}


def main(args):
    """Main preprocessing workflow."""
    validated_bridge = Path(args.validated_bridge)
    original_hotpotqa = Path(args.original_hotpotqa)
    output = Path(args.output)
    heldout_pool = Path(args.heldout_pool) if args.heldout_pool else None

    stats = process_split(
        validated_bridge_file=validated_bridge,
        original_hotpotqa_file=original_hotpotqa,
        output_file=output,
        split_name=args.split,
        samples_per_k=args.samples_per_k,
        heldout_pool_file=heldout_pool,
        seed=args.seed,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n{args.split.upper()}: {stats['total_examples']} examples")
    print(f"Unique bridge examples used: {stats['unique_bridge_used']}")
    print(f"Unique comparison examples used: {stats['unique_comparison_used']}")
    print(f"Total unique examples used: {stats['unique_bridge_used'] + stats['unique_comparison_used']}")
    print("\nOutput files:")
    print(f"  Output: {output}")
    if heldout_pool:
        print(f"  Heldout pool: {heldout_pool}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess HotpotQA for multi-objective question answering"
    )
    parser.add_argument(
        "--validated-bridge",
        dest="validated_bridge",
        type=str,
        required=True,
        help="Path to validated bridge questions JSONL file (from batch_validate_download.py)",
    )
    parser.add_argument(
        "--original-hotpotqa",
        dest="original_hotpotqa",
        type=str,
        required=True,
        help="Path to original HotpotQA JSON file (for comparison questions)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test"],
        required=True,
        help="Split name (train or test)",
    )
    parser.add_argument(
        "--heldout-pool",
        dest="heldout_pool",
        type=str,
        default=None,
        help="Path to save heldout examples for demonstrations (train split only)",
    )
    parser.add_argument(
        "--samples-per-k",
        dest="samples_per_k",
        type=int,
        default=200,
        help="Number of samples per k value (default: 200). Reduce for smaller datasets.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    main(args)
