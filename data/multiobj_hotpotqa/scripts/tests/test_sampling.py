"""
Unit tests for sampling strategy and data loading functions.

Tests cover:
1. Loading JSONL data
2. Filtering by question type
3. Sampling with balanced type distribution
4. Ensuring no overlap between sets
"""

import json
import random
import unittest
from pathlib import Path
from typing import List, Dict, Any, Tuple


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def filter_by_type(examples: List[Dict[str, Any]], question_type: str) -> List[Dict[str, Any]]:
    """
    Filter examples by question type.

    Args:
        examples: List of examples
        question_type: "bridge" or "comparison"

    Returns:
        Filtered list
    """
    return [ex for ex in examples if ex["metadata"]["type"] == question_type]


def sample_balanced_k1(
    bridge_examples: List[Dict[str, Any]],
    comparison_examples: List[Dict[str, Any]],
    num_samples: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Sample balanced examples for k=1.

    For k=1, we want equal numbers of bridge and comparison questions.

    Args:
        bridge_examples: Pool of bridge examples
        comparison_examples: Pool of comparison examples
        num_samples: Total number of samples to draw (should be even)
        seed: Random seed

    Returns:
        Balanced sample with num_samples/2 of each type
    """
    random.seed(seed)

    half = num_samples // 2

    # Sample without replacement
    sampled_bridge = random.sample(bridge_examples, half)
    sampled_comparison = random.sample(comparison_examples, half)

    # Combine and shuffle
    combined = sampled_bridge + sampled_comparison
    random.shuffle(combined)

    return combined


def sample_balanced_multi(
    bridge_examples: List[Dict[str, Any]],
    comparison_examples: List[Dict[str, Any]],
    k: int,
    num_samples: int,
    seed: int = 42
) -> List[List[Dict[str, Any]]]:
    """
    Sample balanced multi-objective examples for k>=2.

    For each sample, randomly choose i ~ Uniform[0, k] and sample
    i bridge + (k-i) comparison questions.

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

        # Sample questions
        sample_components = []
        if num_bridge > 0:
            sample_components.extend(random.sample(bridge_examples, num_bridge))
        if num_comparison > 0:
            sample_components.extend(random.sample(comparison_examples, num_comparison))

        # Shuffle the order
        random.shuffle(sample_components)
        samples.append(sample_components)

    return samples


def split_train_heldout(
    examples: List[Dict[str, Any]],
    num_train: int,
    num_heldout: int,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split examples into train and heldout sets with no overlap.

    Args:
        examples: Pool of examples
        num_train: Number of training examples
        num_heldout: Number of heldout examples
        seed: Random seed

    Returns:
        Tuple of (train_examples, heldout_examples)
    """
    random.seed(seed)

    if len(examples) < num_train + num_heldout:
        raise ValueError(f"Not enough examples: {len(examples)} < {num_train + num_heldout}")

    # Shuffle and split
    shuffled = examples.copy()
    random.shuffle(shuffled)

    train = shuffled[:num_train]
    heldout = shuffled[num_train:num_train + num_heldout]

    return train, heldout


def combine_questions(components: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine multiple questions into a multi-objective format.

    Args:
        components: List of component questions

    Returns:
        Combined multi-objective example
    """
    k = len(components)

    # Create numbered question text
    question_parts = [f"{i+1}. {comp['question']}" for i, comp in enumerate(components)]
    combined_question = "\n".join(question_parts)

    # Combine answers
    combined_answers = [comp["golden_answers"][0] for comp in components]

    # Combine metadata
    component_ids = [comp["id"] for comp in components]
    component_types = [comp["metadata"]["type"] for comp in components]

    # Create new ID
    new_id = "_".join(component_ids)

    return {
        "id": new_id,
        "question": combined_question,
        "golden_answers": combined_answers,
        "metadata": {
            "k": k,
            "component_ids": component_ids,
            "component_types": component_types,
            "components": components
        }
    }


class TestDataLoading(unittest.TestCase):
    """Test data loading functions."""

    def test_filter_by_type(self):
        """Test filtering examples by type."""
        examples = [
            {"id": "1", "metadata": {"type": "bridge"}},
            {"id": "2", "metadata": {"type": "comparison"}},
            {"id": "3", "metadata": {"type": "bridge"}},
            {"id": "4", "metadata": {"type": "comparison"}},
        ]

        bridge = filter_by_type(examples, "bridge")
        comparison = filter_by_type(examples, "comparison")

        self.assertEqual(len(bridge), 2)
        self.assertEqual(len(comparison), 2)
        self.assertEqual([ex["id"] for ex in bridge], ["1", "3"])
        self.assertEqual([ex["id"] for ex in comparison], ["2", "4"])


class TestSamplingK1(unittest.TestCase):
    """Test k=1 sampling strategy."""

    def test_balanced_sampling(self):
        """Test that k=1 sampling produces equal bridge/comparison."""
        bridge = [{"id": f"b{i}", "metadata": {"type": "bridge"}} for i in range(200)]
        comparison = [{"id": f"c{i}", "metadata": {"type": "comparison"}} for i in range(200)]

        sample = sample_balanced_k1(bridge, comparison, num_samples=100, seed=42)

        # Should have 100 total samples
        self.assertEqual(len(sample), 100)

        # Should have 50 of each type
        bridge_count = sum(1 for ex in sample if ex["metadata"]["type"] == "bridge")
        comparison_count = sum(1 for ex in sample if ex["metadata"]["type"] == "comparison")

        self.assertEqual(bridge_count, 50)
        self.assertEqual(comparison_count, 50)

    def test_no_duplicates(self):
        """Test that sampled examples are unique."""
        bridge = [{"id": f"b{i}", "metadata": {"type": "bridge"}} for i in range(200)]
        comparison = [{"id": f"c{i}", "metadata": {"type": "comparison"}} for i in range(200)]

        sample = sample_balanced_k1(bridge, comparison, num_samples=100, seed=42)

        ids = [ex["id"] for ex in sample]
        self.assertEqual(len(ids), len(set(ids)), "Sample should not contain duplicates")


class TestSamplingMulti(unittest.TestCase):
    """Test k>=2 sampling strategy."""

    def test_k2_samples_have_two_questions(self):
        """Test that k=2 samples contain exactly 2 questions."""
        bridge = [{"id": f"b{i}", "metadata": {"type": "bridge"}} for i in range(200)]
        comparison = [{"id": f"c{i}", "metadata": {"type": "comparison"}} for i in range(200)]

        samples = sample_balanced_multi(bridge, comparison, k=2, num_samples=50, seed=42)

        self.assertEqual(len(samples), 50)
        for sample in samples:
            self.assertEqual(len(sample), 2, "Each k=2 sample should have 2 questions")

    def test_k3_samples_have_three_questions(self):
        """Test that k=3 samples contain exactly 3 questions."""
        bridge = [{"id": f"b{i}", "metadata": {"type": "bridge"}} for i in range(200)]
        comparison = [{"id": f"c{i}", "metadata": {"type": "comparison"}} for i in range(200)]

        samples = sample_balanced_multi(bridge, comparison, k=3, num_samples=50, seed=42)

        self.assertEqual(len(samples), 50)
        for sample in samples:
            self.assertEqual(len(sample), 3, "Each k=3 sample should have 3 questions")

    def test_random_type_distribution(self):
        """Test that type distribution varies across samples."""
        bridge = [{"id": f"b{i}", "metadata": {"type": "bridge"}} for i in range(200)]
        comparison = [{"id": f"c{i}", "metadata": {"type": "comparison"}} for i in range(200)]

        samples = sample_balanced_multi(bridge, comparison, k=2, num_samples=100, seed=42)

        # Count different type combinations
        type_combinations = []
        for sample in samples:
            types = tuple(sorted([ex["metadata"]["type"] for ex in sample]))
            type_combinations.append(types)

        unique_combinations = set(type_combinations)

        # For k=2, we should see multiple combinations:
        # (bridge, bridge), (bridge, comparison), (comparison, comparison)
        self.assertGreater(len(unique_combinations), 1, "Should have variety in type combinations")


class TestTrainHeldoutSplit(unittest.TestCase):
    """Test splitting into train and heldout sets."""

    def test_no_overlap(self):
        """Test that train and heldout sets don't overlap."""
        examples = [{"id": f"ex{i}"} for i in range(1000)]

        train, heldout = split_train_heldout(examples, num_train=200, num_heldout=400, seed=42)

        self.assertEqual(len(train), 200)
        self.assertEqual(len(heldout), 400)

        train_ids = {ex["id"] for ex in train}
        heldout_ids = {ex["id"] for ex in heldout}

        self.assertEqual(len(train_ids & heldout_ids), 0, "Train and heldout must not overlap")

    def test_insufficient_examples(self):
        """Test error handling when not enough examples."""
        examples = [{"id": f"ex{i}"} for i in range(100)]

        with self.assertRaises(ValueError):
            split_train_heldout(examples, num_train=200, num_heldout=400)


class TestQuestionCombination(unittest.TestCase):
    """Test combining questions into multi-objective format."""

    def test_combine_two_questions(self):
        """Test combining two questions."""
        components = [
            {"id": "q1", "question": "What is X?", "golden_answers": ["A"], "metadata": {"type": "bridge"}},
            {"id": "q2", "question": "What is Y?", "golden_answers": ["B"], "metadata": {"type": "comparison"}},
        ]

        combined = combine_questions(components)

        expected_question = "1. What is X?\n2. What is Y?"
        self.assertEqual(combined["question"], expected_question)
        self.assertEqual(combined["golden_answers"], ["A", "B"])
        self.assertEqual(combined["metadata"]["k"], 2)
        self.assertEqual(combined["metadata"]["component_ids"], ["q1", "q2"])

    def test_combine_preserves_metadata(self):
        """Test that component metadata is preserved."""
        components = [
            {"id": "q1", "question": "Q1?", "golden_answers": ["A1"], "metadata": {"type": "bridge"}},
            {"id": "q2", "question": "Q2?", "golden_answers": ["A2"], "metadata": {"type": "comparison"}},
        ]

        combined = combine_questions(components)

        self.assertEqual(combined["metadata"]["component_types"], ["bridge", "comparison"])
        self.assertIn("components", combined["metadata"])
        self.assertEqual(len(combined["metadata"]["components"]), 2)


if __name__ == "__main__":
    unittest.main()
