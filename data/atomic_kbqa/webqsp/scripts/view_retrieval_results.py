"""
Streamlit app to view WebQSP retrieval results.

This app allows users to explore the in-context example retrieval results,
viewing examples and their nearest neighbors with filtering capabilities.

Usage:
    uv run streamlit run data/atomic_kbqa/webqsp/scripts/view_retrieval_results.py
"""

from collections import defaultdict
from pathlib import Path
import json

import streamlit as st


@st.cache_data
def load_data(train_file, test_file):
    """Load and cache train and test data."""
    with open(train_file, "r") as f:
        train_data = json.load(f)
    with open(test_file, "r") as f:
        test_data = json.load(f)
    return train_data, test_data


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
    st.title("WebQSP Retrieval Results Viewer")

    # File paths
    data_dir = Path(__file__).parent.parent / "processed"
    train_file = data_dir / "webqsp_train.v1.50nn.json"
    test_file = data_dir / "webqsp_test.v1.50nn.json"

    # Load data
    try:
        train_data, test_data = load_data(train_file, test_file)
        all_data = train_data + test_data
    except FileNotFoundError as e:
        st.error(f"Could not load data files: {e}")
        st.info("Make sure to run the retrieval script first:")
        st.code(
            "uv run python data/atomic_kbqa/webqsp/scripts/retrieve_webqsp_examples.py --num-candidates 50",
            language="bash",
        )
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

    st.write(
        f"Total examples: {len(all_data)} ({len(train_data)} train, {len(test_data)} test)"
    )

    # Sidebar filters
    st.sidebar.header("Filters")

    # Split filter
    split_filter = st.sidebar.radio("Split", ["All", "Train", "Test"], index=0)

    # Workflow length filter
    workflow_lengths = [f["nodes"] for f in st.session_state.features]
    min_length, max_length = min(workflow_lengths), max(workflow_lengths)
    length_range = st.sidebar.slider(
        "Workflow Length (nodes)",
        min_value=min_length,
        max_value=max_length,
        value=(min_length, max_length),
    )

    # Operator filter: exclude START and STOP from criteria since they appear in all examples
    # Available operators (example-level): JOIN, AND, ARG, CMP, TC, COUNT, R
    AVAILABLE_OPERATORS = ["JOIN", "AND", "ARGMIN", "ARGMAX", "ge", "le", "gt", "lt", "TC", "COUNT", "R"]
    selected_operators = st.sidebar.multiselect(
        "Operators (in S-expression)", AVAILABLE_OPERATORS, default=AVAILABLE_OPERATORS
    )

    # Apply filters
    filtered_indices = []
    for i, example in enumerate(all_data):
        features = st.session_state.features[i]

        # Split filter
        if split_filter == "Train" and not example["id"].startswith("train_"):
            continue
        if split_filter == "Test" and not example["id"].startswith("test_"):
            continue

        # Length filter
        if not (length_range[0] <= features["nodes"] <= length_range[1]):
            continue

        # Operator filter - check operators called in example's S-expression
        # If user selected no operators, treat as no-op (include all)
        if selected_operators:
            sexpr = example.get("sexpr", "")
            if not all([f"({selected_op} " in sexpr
                        for selected_op in selected_operators]):
                continue

        filtered_indices.append(i)

    st.sidebar.write(f"**Filtered:** {len(filtered_indices)} examples")

    if not filtered_indices:
        st.warning("No examples match the selected filters.")
        st.stop()

    # Example selection with DAG features in display
    example_options = []
    for idx in filtered_indices:
        example = all_data[idx]
        features = st.session_state.features[idx]
        split = "Train" if example["id"].startswith("train_") else "Test"
        option_text = (
            f"{example['id']} [{split}] "
            f"(N:{features['nodes']}, B:{features['branches']}, M:{features['merges']}): "
            f"{example['question'][:60]}..."
        )
        example_options.append(option_text)

    selected_filtered_idx = st.selectbox(
        "Select example",
        range(len(filtered_indices)),
        format_func=lambda x: example_options[x],
    )
    selected_idx = filtered_indices[selected_filtered_idx]

    # Display selected example
    example = all_data[selected_idx]
    features = st.session_state.features[selected_idx]

    st.header("Example Details")

    # Basic info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ID", example["id"])
    with col2:
        st.metric("Original ID", example.get("original_id", "N/A"))
    with col3:
        st.metric("Level", example["metadata"].get("level", "unknown"))

    # Question and answer
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Question")
        st.write(example["question"])

        st.subheader("Expected Answer")
        answer = example.get("answer", [])
        answer_label = example.get("answer_label", [])
        if isinstance(answer, list):
            for i, ans in enumerate(answer):
                if answer_label and i < len(answer_label):
                    st.write(f"- {ans}  \t*(Label: {answer_label[i]})*")
                else:
                    st.write(f"- {ans}")
        else:
            st.write(answer)

    with col2:
        st.subheader("DAG Features")
        st.write(f"**Nodes:** {features['nodes']}")
        st.write(f"**Branches:** {features['branches']}")
        st.write(f"**Merges:** {features['merges']}")

        st.subheader("Metadata")
        metadata = example.get("metadata", {})
        st.write(f"**Workflow Length:** {metadata.get('workflow_length', 'N/A')}")
        st.write(f"**Bin:** {metadata.get('bin', 'N/A')}")
        st.write(f"**Original Index:** {metadata.get('original_idx', 'N/A')}")

    # Technical details in expanders
    with st.expander("View S-Expression"):
        st.code(example.get("sexpr", "N/A"), language="lisp")

    with st.expander("View SPARQL"):
        st.code(example.get("sparql", "N/A"), language="sparql")

    with st.expander("View Function List"):
        function_list = example.get("function_list", [])
        if function_list:
            st.code("\n".join(function_list), language="python")
        else:
            st.write("No function list available")

    with st.expander("View DAG (JSON)"):
        dag_dict = example.get("dag", [])
        st.json(dag_dict, expanded=False)

    # Nearest neighbors
    st.header("Top 10 Nearest Neighbors")
    candidates = example.get("demonstration_candidates", [])[:10]

    if not candidates:
        st.warning("No candidates found for this example.")
    else:
        # Show distribution of similarity scores
        similarities = [c["similarity"] for c in candidates]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Similarity", f"{min(similarities):.3f}")
        with col2:
            st.metric("Max Similarity", f"{max(similarities):.3f}")
        with col3:
            st.metric("Avg Similarity", f"{sum(similarities) / len(similarities):.3f}")

        st.divider()

        for i, candidate in enumerate(candidates, 1):
            with st.expander(
                f"{i}. Similarity: {candidate['similarity']:.3f} - {candidate['question'][:80]}..."
            ):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**ID:** {candidate.get('ID', 'N/A')}")
                    st.write(f"**Question:** {candidate['question']}")
                    st.write("**Answer:**")
                    answer = candidate.get("answer", [])
                    if isinstance(answer, list):
                        for ans in answer:
                            st.write(f"- {ans}")
                    else:
                        st.write(answer)

                with col2:
                    st.write("**Demonstration Text:**")
                    dag_text = candidate.get("text", "")
                    st.code(dag_text, language="text")

                # Show candidate DAG
                with st.expander("View Candidate DAG"):
                    st.json(candidate.get("dag", []), expanded=False)


if __name__ == "__main__":
    main()
