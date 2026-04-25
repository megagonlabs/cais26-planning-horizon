"""
Streamlit app to view Multi-objective HotpotQA retrieval results.

This app allows users to explore the in-context example retrieval results,
viewing examples and their nearest neighbors with filtering capabilities.
"""

from collections import defaultdict
from pathlib import Path
import json

import streamlit as st


@st.cache_data
def load_data(train_file: Path, test_file: Path) -> tuple[list[dict], list[dict]]:
    """Load and cache train and test data."""
    data_map = {}

    if train_file.exists():
        with open(train_file, "r") as f:
            train_data = json.load(f)
        for ex in train_data:
            ex["_split"] = "Train"
        data_map["Train"] = train_data
    else:
        st.warning(f"Train file not found: {train_file}")
        data_map["Train"] = []

    if test_file.exists():
        with open(test_file, "r") as f:
            test_data = json.load(f)
        for ex in test_data:
            ex["_split"] = "Test"
        data_map["Test"] = test_data
    else:
        st.warning(f"Test file not found: {test_file}")
        data_map["Test"] = []

    return data_map["Train"], data_map["Test"]


def compute_dag_features(dag: list[dict]) -> tuple[int, int, int]:
    """Compute DAG features: nodes, branches, merges."""
    if not dag:
        return 0, 0, 0

    num_nodes = len(dag)

    # Count dependencies per step
    dep_counts = defaultdict(int)
    for step in dag:
        for dep in step.get("dependencies", []):
            dep_counts[dep] += 1

    # Branches: steps that are used by multiple other steps
    num_branches = sum(1 for count in dep_counts.values() if count > 1)

    # Merges: steps that depend on multiple other steps
    num_merges = sum(1 for step in dag if len(step.get("dependencies", [])) > 1)

    return num_nodes, num_branches, num_merges


def main():
    st.set_page_config(layout="wide")
    st.title("Multi-objective HotpotQA Retrieval Results Viewer")

    # File paths - assume running from repo root
    base_dir = Path("data/multiobj_hotpotqa/processed")
    train_file = base_dir / "train.v1.annotated.50nn.json"
    test_file = base_dir / "test.v1.annotated.50nn.json"

    # Load data
    train_data, test_data = load_data(train_file, test_file)
    all_data = train_data + test_data

    if not all_data:
        st.error("No data loaded. Please check file paths.")
        st.stop()

    # Compute features for all examples
    if "features" not in st.session_state:
        st.session_state.features = []
        for example in all_data:
            dag = example.get("dag", [])
            nodes, branches, merges = compute_dag_features(dag)
            # Also get k from metadata if available
            k = example.get("metadata", {}).get("k", "?")
            st.session_state.features.append(
                {"nodes": nodes, "branches": branches, "merges": merges, "k": k}
            )

    st.write(
        f"Total examples: {len(all_data)} (Train: {len(train_data)}, Test: {len(test_data)})"
    )

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        split_filter = st.multiselect(
            "Filter by Split", ["Train", "Test"], default=["Train", "Test"]
        )

    # Filter data
    filtered_indices = [
        i for i, ex in enumerate(all_data) if ex["_split"] in split_filter
    ]

    if not filtered_indices:
        st.warning("No examples match filters.")
        st.stop()

    # Example selection with DAG features in display
    example_options = []
    # Map option index back to original index
    option_to_idx = {}

    for i in filtered_indices:
        example = all_data[i]
        features = st.session_state.features[i]
        split = example["_split"]
        # Truncate question for display
        q_text = (
            example["question"][:80] + "..."
            if len(example["question"]) > 80
            else example["question"]
        )
        option_text = f"[{split}] [k={features['k']}] {q_text} (ID: {example['id']})"
        example_options.append(option_text)
        option_to_idx[option_text] = i

    selected_option = st.selectbox("Select example", example_options)
    selected_idx = option_to_idx[selected_option]

    # Display selected example
    example = all_data[selected_idx]
    features = st.session_state.features[selected_idx]

    st.header("Example Details")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Question")
        st.write(example["question"])

        st.subheader("Expected Answers")
        st.write(example.get("answers", []))

        st.subheader("Metadata")
        st.json(example.get("metadata", {}))

    with col2:
        st.subheader("DAG Features")
        st.write(f"**Nodes:** {features['nodes']}")
        st.write(f"**Branches:** {features['branches']}")
        st.write(f"**Merges:** {features['merges']}")
        st.write(f"**k (components):** {features['k']}")

        st.subheader("DAG")
        dag_dict = example.get("dag", [])
        st.json(dag_dict, expanded=False)

    # Nearest neighbors
    st.header("Top Nearest Neighbors")

    num_neighbors = st.slider("Number of neighbors to show", 1, 50, 5)
    candidates = example.get("demonstration_candidates", [])[:num_neighbors]

    if not candidates:
        st.warning("No candidates found for this example.")
    else:
        for i, candidate in enumerate(candidates, 1):
            with st.expander(
                f"{i}. Similarity: {candidate['similarity']:.4f} - {candidate['question'][:60]}..."
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Question:** {candidate['question']}")
                    st.write(f"**Answers:** {candidate.get('answers', [])}")
                    st.write(f"**ID:** {candidate.get('id', 'N/A')}")

                with col2:
                    st.write("**Demonstration Text:**")
                    dag_text = candidate.get("text", "")
                    st.code(dag_text, language="text")


if __name__ == "__main__":
    main()
