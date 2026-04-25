"""
Streamlit app to view KQA Pro retrieval results.

This app allows users to explore the in-context example retrieval results,
viewing examples and their nearest neighbors with filtering capabilities.
"""

from collections import defaultdict
from pathlib import Path
import json

import streamlit as st


@st.cache_data
def load_data(train_file, val_file):
    """Load and cache train and val data."""
    with open(train_file, "r") as f:
        train_data = json.load(f)
    with open(val_file, "r") as f:
        val_data = json.load(f)
    return train_data, val_data


def compute_dag_features(dag):
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
    st.title("KQA Pro Retrieval Results Viewer")

    # File paths
    data_dir = Path(__file__).resolve().parent.parent / "processed"
    train_file = data_dir / "train.v1.50nn.json"
    val_file = data_dir / "val.v1.50nn.json"

    # Load data
    try:
        train_data, val_data = load_data(train_file, val_file)
        all_data = train_data + val_data
    except FileNotFoundError as e:
        st.error(f"Could not load data files: {e}")
        st.stop()

    # Compute features for all examples
    if "features" not in st.session_state:
        st.session_state.features = []
        for example in all_data:
            dag = example.get("dag", [])
            nodes, branches, merges = compute_dag_features(dag)
            st.session_state.features.append(
                {"nodes": nodes, "branches": branches, "merges": merges}
            )

    st.write(f"Total examples: {len(all_data)}")

    # Example selection with DAG features in display
    example_options = []
    for i, example in enumerate(all_data):
        features = st.session_state.features[i]
        split = "Train" if example["id"].startswith("train_") else "Val"
        option_text = f"{example['id']} [{split}] (N:{features['nodes']}, B:{features['branches']}, M:{features['merges']}): {example['question'][:60]}..."
        example_options.append(option_text)

    selected_idx = st.selectbox(
        "Select example", range(len(all_data)), format_func=lambda x: example_options[x]
    )

    # Display selected example
    example = all_data[selected_idx]
    features = st.session_state.features[selected_idx]

    st.header("Example Details")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Question")
        st.write(example["question"])

        st.subheader("Expected Answer")
        st.write(example["answer"])

    with col2:
        st.subheader("DAG Features")
        st.write(f"**Nodes:** {features['nodes']}")
        st.write(f"**Branches:** {features['branches']}")
        st.write(f"**Merges:** {features['merges']}")

        st.subheader("DAG")
        dag_dict = example.get("dag", "")
        st.json(dag_dict, expanded=False)

    # Nearest neighbors
    st.header("Top 10 Nearest Neighbors")
    candidates = example.get("demonstration_candidates", [])[:10]

    if not candidates:
        st.warning("No candidates found for this example.")
    else:
        for i, candidate in enumerate(candidates, 1):
            with st.expander(f"{i}. Similarity: {candidate['similarity']:.3f}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Question:** {candidate['question']}")
                    st.write(f"**Answer:** {candidate['answer']}")

                with col2:
                    st.write("**Demonstration Text:**")
                    dag_text = candidate.get("text", "")
                    st.code(dag_text, language="text")


if __name__ == "__main__":
    main()
