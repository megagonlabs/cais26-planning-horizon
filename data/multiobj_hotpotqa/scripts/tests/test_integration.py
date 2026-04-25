"""
Integration tests using actual HotpotQA data files.

These tests validate that our functions work correctly with real data.
"""

import json
import unittest
from pathlib import Path
from typing import Dict, List, Any

# Import functions from test_sampling
import sys
sys.path.insert(0, str(Path(__file__).parent))
from test_sampling import load_jsonl, filter_by_type, sample_balanced_k1
from test_dag_generation import (
    generate_bridge_dag,
    generate_comparison_dag,
    merge_dags,
    compute_dag_depth,
    compute_dag_width
)


# Path to data directory
DATA_DIR = Path(__file__).parent.parent.parent
TRAIN_FILE = DATA_DIR / "train.jsonl"
DEV_FILE = DATA_DIR / "dev.jsonl"


class TestRealDataLoading(unittest.TestCase):
    """Test loading actual HotpotQA data files."""

    @classmethod
    def setUpClass(cls):
        """Load data once for all tests."""
        if TRAIN_FILE.exists():
            # Load first 1000 examples for testing (faster)
            cls.train_data = []
            with open(TRAIN_FILE, "r") as f:
                for i, line in enumerate(f):
                    if i >= 1000:
                        break
                    cls.train_data.append(json.loads(line))
        else:
            cls.train_data = []

        if DEV_FILE.exists():
            cls.dev_data = load_jsonl(DEV_FILE)
        else:
            cls.dev_data = []

    def test_train_file_exists(self):
        """Test that train file exists and can be loaded."""
        self.assertTrue(TRAIN_FILE.exists(), f"Train file not found: {TRAIN_FILE}")
        self.assertGreater(len(self.train_data), 0, "Train data should not be empty")

    def test_dev_file_exists(self):
        """Test that dev file exists and can be loaded."""
        self.assertTrue(DEV_FILE.exists(), f"Dev file not found: {DEV_FILE}")
        self.assertGreater(len(self.dev_data), 0, "Dev data should not be empty")

    def test_example_has_required_fields(self):
        """Test that examples have all required fields."""
        if not self.train_data:
            self.skipTest("No train data available")

        example = self.train_data[0]

        # Check top-level fields
        self.assertIn("id", example)
        self.assertIn("question", example)
        self.assertIn("golden_answers", example)
        self.assertIn("metadata", example)

        # Check metadata fields
        metadata = example["metadata"]
        self.assertIn("type", metadata)
        self.assertIn("level", metadata)
        self.assertIn("supporting_facts", metadata)
        self.assertIn("context", metadata)

    def test_question_types_present(self):
        """Test that both bridge and comparison questions are present."""
        if not self.train_data:
            self.skipTest("No train data available")

        types = [ex["metadata"]["type"] for ex in self.train_data]
        type_set = set(types)

        self.assertIn("bridge", type_set, "Bridge questions should be present")
        self.assertIn("comparison", type_set, "Comparison questions should be present")

    def test_filter_real_data_by_type(self):
        """Test filtering real data by type."""
        if not self.train_data:
            self.skipTest("No train data available")

        bridge = filter_by_type(self.train_data, "bridge")
        comparison = filter_by_type(self.train_data, "comparison")

        self.assertGreater(len(bridge), 0, "Should find bridge questions")
        self.assertGreater(len(comparison), 0, "Should find comparison questions")

        # Verify all filtered examples have correct type
        for ex in bridge:
            self.assertEqual(ex["metadata"]["type"], "bridge")
        for ex in comparison:
            self.assertEqual(ex["metadata"]["type"], "comparison")

    def test_sample_real_data(self):
        """Test sampling from real data."""
        if not self.train_data:
            self.skipTest("No train data available")

        bridge = filter_by_type(self.train_data, "bridge")
        comparison = filter_by_type(self.train_data, "comparison")

        # Skip if not enough examples
        if len(bridge) < 50 or len(comparison) < 50:
            self.skipTest("Not enough examples for sampling")

        sample = sample_balanced_k1(bridge, comparison, num_samples=100, seed=42)

        self.assertEqual(len(sample), 100)

        # Check balance
        bridge_count = sum(1 for ex in sample if ex["metadata"]["type"] == "bridge")
        comparison_count = sum(1 for ex in sample if ex["metadata"]["type"] == "comparison")

        self.assertEqual(bridge_count, 50)
        self.assertEqual(comparison_count, 50)


class TestDAGAnnotation(unittest.TestCase):
    """Test DAG annotation with real question types."""

    def test_annotate_bridge_question(self):
        """Test annotating a bridge question with DAG."""
        # Simulate a bridge question
        bridge_example = {
            "id": "test_bridge",
            "question": "Test bridge question?",
            "golden_answers": ["Answer"],
            "metadata": {"type": "bridge"}
        }

        # Generate DAG
        dag = generate_bridge_dag()

        # Add to example
        bridge_example["dag"] = dag

        # Validate
        self.assertIn("dag", bridge_example)
        self.assertEqual(len(bridge_example["dag"]), 3)
        self.assertEqual(bridge_example["dag"][-1]["function"], "finish")

    def test_annotate_comparison_question(self):
        """Test annotating a comparison question with DAG."""
        comparison_example = {
            "id": "test_comparison",
            "question": "Test comparison question?",
            "golden_answers": ["Answer"],
            "metadata": {"type": "comparison"}
        }

        # Generate DAG
        dag = generate_comparison_dag()

        # Add to example
        comparison_example["dag"] = dag

        # Validate
        self.assertIn("dag", comparison_example)
        self.assertEqual(len(comparison_example["dag"]), 4)
        self.assertEqual(comparison_example["dag"][-1]["function"], "finish")

    def test_annotate_multi_objective(self):
        """Test annotating a multi-objective question (k=2)."""
        bridge = {"metadata": {"type": "bridge"}}
        comparison = {"metadata": {"type": "comparison"}}

        # Generate individual DAGs
        bridge_dag = generate_bridge_dag()
        comparison_dag = generate_comparison_dag()

        # Merge
        merged_dag = merge_dags([bridge_dag, comparison_dag])

        # Validate merged structure
        self.assertEqual(len(merged_dag), 6)
        self.assertEqual(merged_dag[-1]["function"], "finish")
        self.assertEqual(set(merged_dag[-1]["dependencies"]), {1, 4})

        # Compute features
        depth = compute_dag_depth(merged_dag)
        width = compute_dag_width(merged_dag)

        self.assertGreaterEqual(depth, 3)
        self.assertGreaterEqual(width, 2)


class TestEndToEndWorkflow(unittest.TestCase):
    """Test end-to-end preprocessing workflow."""

    @classmethod
    def setUpClass(cls):
        """Load sample data."""
        if TRAIN_FILE.exists():
            cls.train_data = []
            with open(TRAIN_FILE, "r") as f:
                for i, line in enumerate(f):
                    if i >= 500:
                        break
                    cls.train_data.append(json.loads(line))
        else:
            cls.train_data = []

    def test_k1_preprocessing(self):
        """Test complete k=1 preprocessing."""
        if not self.train_data or len(self.train_data) < 100:
            self.skipTest("Not enough data")

        # Filter by type
        bridge = filter_by_type(self.train_data, "bridge")
        comparison = filter_by_type(self.train_data, "comparison")

        if len(bridge) < 50 or len(comparison) < 50:
            self.skipTest("Not enough examples of each type")

        # Sample
        sample = sample_balanced_k1(bridge, comparison, num_samples=100, seed=42)

        # Annotate with DAGs
        for ex in sample:
            if ex["metadata"]["type"] == "bridge":
                ex["dag"] = generate_bridge_dag()
            else:
                ex["dag"] = generate_comparison_dag()

        # Verify all have DAGs
        for ex in sample:
            self.assertIn("dag", ex)
            self.assertGreater(len(ex["dag"]), 0)
            self.assertEqual(ex["dag"][-1]["function"], "finish")

    def test_k2_preprocessing(self):
        """Test complete k=2 preprocessing."""
        if not self.train_data or len(self.train_data) < 100:
            self.skipTest("Not enough data")

        bridge = filter_by_type(self.train_data, "bridge")
        comparison = filter_by_type(self.train_data, "comparison")

        if len(bridge) < 2 or len(comparison) < 2:
            self.skipTest("Not enough examples")

        # Create a k=2 sample manually
        components = [bridge[0], comparison[0]]

        # Generate DAGs
        dags = []
        for comp in components:
            if comp["metadata"]["type"] == "bridge":
                dags.append(generate_bridge_dag())
            else:
                dags.append(generate_comparison_dag())

        # Merge
        merged_dag = merge_dags(dags)

        # Validate
        self.assertGreater(len(merged_dag), 4)
        self.assertEqual(merged_dag[-1]["function"], "finish")
        self.assertGreater(len(merged_dag[-1]["dependencies"]), 0)


if __name__ == "__main__":
    unittest.main()
