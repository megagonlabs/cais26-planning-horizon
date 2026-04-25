"""
Test preprocessing script with small sample of data.

This validates the preprocessing logic before running on full dataset.
"""

from pathlib import Path
import json
import sys

# Add scripts directory to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from preprocess_multiobj_hotpotqa import (
    load_jsonl,
    filter_by_type,
    sample_k1,
    sample_multi_objective,
    combine_questions,
    generate_bridge_dag,
    generate_comparison_dag,
    merge_dags,
)


def test_with_real_data():
    """Test preprocessing functions with real HotpotQA data."""

    data_dir = Path(__file__).parent.parent
    train_file = data_dir / "train.jsonl"

    if not train_file.exists():
        print(f"Error: {train_file} not found")
        return False

    print("Loading sample data...")
    # Load first 1000 examples for testing
    data = []
    with open(train_file, "r") as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            data.append(json.loads(line))

    print(f"Loaded {len(data)} examples")

    # Test filtering
    print("\nTesting filtering...")
    bridge = filter_by_type(data, "bridge")
    comparison = filter_by_type(data, "comparison")
    print(f"  Bridge: {len(bridge)}")
    print(f"  Comparison: {len(comparison)}")

    if len(bridge) < 100 or len(comparison) < 100:
        print("Error: Not enough examples of each type")
        return False

    # Test k=1 sampling
    print("\nTesting k=1 sampling...")
    k1_sample = sample_k1(bridge, comparison, 100, seed=42)
    print(f"  Sampled: {len(k1_sample)} examples")

    bridge_count = sum(1 for ex in k1_sample if ex["metadata"]["type"] == "bridge")
    comparison_count = sum(1 for ex in k1_sample if ex["metadata"]["type"] == "comparison")
    print(f"  Bridge: {bridge_count}, Comparison: {comparison_count}")

    if bridge_count != 50 or comparison_count != 50:
        print("Error: k=1 sampling not balanced")
        return False

    # Test k=2 sampling
    print("\nTesting k=2 sampling...")
    used_ids = {ex["id"] for ex in k1_sample}
    k2_samples = sample_multi_objective(bridge, comparison, k=2, num_samples=50, used_ids=used_ids, seed=43)
    print(f"  Sampled: {len(k2_samples)} multi-objective examples")

    # Verify no reuse
    k2_ids = set()
    for sample in k2_samples:
        for ex in sample:
            if ex["id"] in k2_ids:
                print(f"Error: Duplicate ID found: {ex['id']}")
                return False
            k2_ids.add(ex["id"])
    print(f"  All {len(k2_ids)} component IDs are unique")

    # Test question combination
    print("\nTesting question combination...")
    test_sample = k2_samples[0]
    combined = combine_questions(test_sample, k=2)
    print(f"  Combined question preview: {combined['question'][:100]}...")
    print(f"  Answers: {combined['answers']}")
    print(f"  Components: {len(combined['component_metadata'])}")

    # Test DAG generation
    print("\nTesting DAG generation...")
    bridge_dag = generate_bridge_dag()
    comparison_dag = generate_comparison_dag()
    print(f"  Bridge DAG nodes: {len(bridge_dag)}")
    print(f"  Comparison DAG nodes: {len(comparison_dag)}")

    # Test DAG merging
    print("\nTesting DAG merging...")
    merged = merge_dags([bridge_dag, comparison_dag])
    print(f"  Merged DAG nodes: {len(merged)}")
    print(f"  Last node (finish) dependencies: {merged[-1]['dependencies']}")

    if merged[-1]["function"] != "finish":
        print("Error: Last node should be finish")
        return False

    if len(merged[-1]["dependencies"]) != 2:
        print("Error: Finish should depend on 2 terminals")
        return False

    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    success = test_with_real_data()
    sys.exit(0 if success else 1)
