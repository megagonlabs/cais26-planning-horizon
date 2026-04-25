#!/usr/bin/env python3
"""
Streamlit app for visualizing atomic KBQA reasoning DAGs (GrailQA, WebQSP, GraphQ).

Usage:
    streamlit run dag_visualizer_atomic_kbqa.py
    streamlit run dag_visualizer_atomic_kbqa.py -- --data data/atomic_kbqa/grailqa
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import json
import os

import streamlit as st
import graphviz
import orjson


@dataclass
class DAGStats:
    """Statistics about a DAG structure."""

    num_nodes: int
    num_branches: int  # nodes with multiple children
    num_merges: int  # nodes with multiple parents
    max_parents: int  # maximum number of parents for any node
    max_children: int  # maximum number of children for any node


def load_verification_file(data_dir: str) -> dict[str, list[str]]:
    """Load the verification file and return verified example IDs by dataset."""
    verification_path = os.path.join(data_dir, "verified.v1.json")
    try:
        with open(verification_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Create empty verification file if it doesn't exist
        empty_verification = {}
        save_verification_file(data_dir, empty_verification)
        return empty_verification
    except json.JSONDecodeError:
        # Handle corrupted file
        return {}


def save_verification_file(
    data_dir: str, verification_data: dict[str, list[str]]
) -> None:
    """Save the verification data to the verification file."""
    verification_path = os.path.join(data_dir, "verified.v1.json")
    with open(verification_path, "w", encoding="utf-8") as f:
        json.dump(verification_data, f, indent=2, sort_keys=True)


def analyze_dag_structure(dag_data: list[dict[str, Any]]) -> DAGStats:
    """Analyze the structure of a DAG and return statistics."""
    if not dag_data:
        return DAGStats(0, 0, 0, 0, 0)

    num_nodes = len(dag_data)
    num_branches = 0  # nodes with multiple children
    num_merges = 0  # nodes with multiple parents
    max_parents = 0
    max_children = 0

    # Build children mapping
    children_map = {i: [] for i in range(num_nodes)}
    for i, node in enumerate(dag_data):
        parents = node.get("dependencies", [])
        for parent_idx in parents:
            if 0 <= parent_idx < num_nodes:
                children_map[parent_idx].append(i)

    for i, node in enumerate(dag_data):
        parents = node.get("dependencies", [])
        children = children_map[i]

        num_parents = len(parents)
        num_children = len(children)

        # Update maximums
        max_parents = max(max_parents, num_parents)
        max_children = max(max_children, num_children)

        # Count branches (nodes with 2+ children)
        if num_children >= 2:
            num_branches += 1

        # Count merges (nodes with 2+ parents)
        if num_parents >= 2:
            num_merges += 1

    return DAGStats(num_nodes, num_branches, num_merges, max_parents, max_children)


def detect_dataset_name(data_dir: Path) -> str | None:
    """Detect dataset name from directory structure."""
    # Check parent directories for known dataset names
    known_datasets = ["grailqa", "webqsp", "graphq"]
    parts = data_dir.parts
    for dataset in known_datasets:
        if dataset in parts:
            return dataset
    return None


@st.cache_data
def get_available_datasets_and_examples_with_stats(
    data_dir: str,
) -> dict[str, list[dict[str, Any]]]:
    """Get atomic KBQA examples with DAG statistics."""
    datasets = {}
    data_path = Path(data_dir)

    if not data_path.exists():
        return {}

    # Detect dataset name
    dataset_name = detect_dataset_name(data_path)
    if dataset_name is None:
        # Try to infer from directory name
        dataset_name = data_path.name

    # Load verification data
    verification_data = load_verification_file(data_dir)

    # Load preprocessed data (with DAGs)
    processed_dir = data_path / "processed"

    # Try different file patterns
    possible_files = [
        processed_dir / f"{dataset_name}_train.v1.json",
        processed_dir / f"{dataset_name}_test.v1.json",
        processed_dir / "train.v1.json",
        processed_dir / "test.v1.json",
    ]

    data = []
    for file_path in possible_files:
        if file_path.exists():
            print(f"Loading {file_path}...")
            with open(file_path, "r", encoding="utf-8") as f:
                data_ = orjson.loads(f.read())
                data.extend(data_)

    if not data:
        return {}

    examples = []
    verified_ids = set(verification_data.get(dataset_name, []))

    for i, item in enumerate(data):
        # Process DAG
        dag = item.get("dag", [])

        if not dag:
            continue

        # Compute statistics
        stats = analyze_dag_structure(dag)

        # Use ID from item, or fallback to index
        example_id = str(item.get("ID", i))

        # Truncate question for display
        question = item.get("question", "")
        question_display = (
            question[:100] + "..." if len(question) > 100 else question
        )

        examples.append(
            {
                "id": example_id,
                "data": {
                    "id": example_id,
                    "question": item.get("question", ""),
                    "answer": item.get("answer", []),
                    "sexpr": item.get("sexpr", ""),
                    "function_list": item.get("function_list", []),
                    "dag": dag,
                    "level": item.get("level", "unknown"),
                    "metadata": item.get("metadata", {}),
                },
                "stats": stats,
                "verified": example_id in verified_ids,
                "question_display": question_display,
            }
        )

    datasets[dataset_name] = examples
    return datasets


def create_dag_graph(dag_data: list[dict[str, Any]]) -> graphviz.Digraph:
    """Create a graphviz representation of the atomic KBQA DAG."""
    dot = graphviz.Digraph(comment="Atomic KBQA Program")
    dot.attr(rankdir="TB")  # Top to bottom layout
    dot.attr("node", shape="box", style="filled", fillcolor="lightblue")

    # Build children mapping for edge creation
    children_map = {i: [] for i in range(len(dag_data))}
    for i, node in enumerate(dag_data):
        parents = node.get("dependencies", [])
        for parent_idx in parents:
            if 0 <= parent_idx < len(dag_data):
                children_map[parent_idx].append(i)

    # Add nodes
    for i, node in enumerate(dag_data):
        function = node.get("function", "").strip()
        inputs = node.get("inputs", [])

        # Create label with multiple lines
        label_parts = [f"[{i}] {function}"]

        if inputs:
            # Format inputs
            inputs_str = ", ".join(str(inp) for inp in inputs)
            # Truncate long inputs
            if len(inputs_str) > 60:
                inputs_str = inputs_str[:57] + "..."
            label_parts.append(f"Args: {inputs_str}")

        label = "\\n".join(label_parts)

        # Color code based on function type
        if function == "START":
            fillcolor = "lightgreen"
        elif function == "STOP":
            fillcolor = "lightcoral"
        elif function in ["JOIN", "AND", "OR"]:
            fillcolor = "lightyellow"
        elif function.startswith("ARG"):
            fillcolor = "lightpink"
        else:
            fillcolor = "lightblue"

        dot.node(str(i), label, fillcolor=fillcolor)

    # Add edges based on dependencies
    for i, node in enumerate(dag_data):
        for parent_idx in node.get("dependencies", []):
            if 0 <= parent_idx < len(dag_data):
                dot.edge(str(parent_idx), str(i))

    return dot


def main(args):
    st.set_page_config(page_title="Atomic KBQA DAG Visualizer", layout="wide")

    st.title("🔍 Atomic KBQA DAG Visualizer")
    st.markdown("Interactive visualization of atomic KBQA reasoning programs (GrailQA, WebQSP, GraphQ)")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Data directory input
        default_data_dir = str(args.data_dir)
        data_dir = st.text_input(
            "Data Directory Path",
            value=default_data_dir,
            help="Path to the atomic KBQA dataset directory (e.g., data/atomic_kbqa/grailqa)",
        )

        # Check if directory exists
        if not os.path.exists(data_dir):
            st.error(f"Directory '{data_dir}' does not exist!")
            st.stop()

        # Show loading message and load data
        with st.spinner("Loading atomic KBQA data..."):
            datasets_examples = get_available_datasets_and_examples_with_stats(data_dir)

        if datasets_examples:
            # Show summary of loaded data
            dataset_name = list(datasets_examples.keys())[0]
            total_examples = len(datasets_examples[dataset_name])
            st.success(f"Loaded {total_examples} examples from {dataset_name}")

    # Get available examples
    if not datasets_examples:
        st.error(f"No atomic KBQA data found in '{data_dir}'")
        st.info("Make sure you have run the preprocessing script to generate processed/*.v1.json files")
        st.stop()

    dataset_name = list(datasets_examples.keys())[0]
    examples = datasets_examples[dataset_name]

    # Search and filter functionality
    st.header("🔍 Search & Filter")

    search_col1, search_col2, search_col3 = st.columns(3)

    with search_col1:
        node_range = st.slider(
            "Number of Nodes",
            min_value=1,
            max_value=max(ex["stats"].num_nodes for ex in examples) if examples else 50,
            value=(1, max(ex["stats"].num_nodes for ex in examples) if examples else 50),
            help="Filter examples by number of nodes in the DAG",
        )

    with search_col2:
        max_branches = max((ex["stats"].num_branches for ex in examples), default=10)
        max_branches += 1
        branch_range = st.slider(
            "Number of Branches",
            min_value=0,
            max_value=max_branches,
            value=(0, max_branches),
            help="Filter examples by number of nodes with multiple children",
        )

    with search_col3:
        max_merges = max((ex["stats"].num_merges for ex in examples), default=10)
        merge_range = st.slider(
            "Number of Merges",
            min_value=0,
            max_value=max_merges,
            value=(0, max_merges),
            help="Filter examples by number of nodes with multiple parents",
        )

    # Filter examples based on search criteria
    filtered_examples = []
    for example in examples:
        stats = example["stats"]
        if (
            node_range[0] <= stats.num_nodes <= node_range[1]
            and branch_range[0] <= stats.num_branches <= branch_range[1]
            and merge_range[0] <= stats.num_merges <= merge_range[1]
        ):
            filtered_examples.append(example)

    if filtered_examples:
        # Create display options with statistics
        display_options = []
        for example in filtered_examples:
            stats = example["stats"]
            display_text = f"{example['id']} (N:{stats.num_nodes}, B:{stats.num_branches}, M:{stats.num_merges})"
            display_options.append(display_text)

        selected_display = st.selectbox(
            f"Select Example ({len(filtered_examples)} matches)",
            options=display_options,
            help="Choose a specific example to visualize. N=Nodes, B=Branches, M=Merges",
        )

        # Extract the example ID from the selected display text
        if selected_display:
            example_id = selected_display.split(" ")[0]
            selected_example = next(
                ex for ex in filtered_examples if ex["id"] == example_id
            )
        else:
            selected_example = None
    else:
        st.warning("No examples match the current search criteria")
        selected_example = None

    # Display search results summary
    if selected_example:
        st.info(f"Found {len(filtered_examples)} examples matching your criteria")

    if selected_example:
        data = selected_example["data"]

        # Display question
        st.header("Question")
        st.text(data["question"])

        # Display answer
        with st.expander("Answer (Gold)"):
            st.json(data["answer"])

        # Display S-expression
        with st.expander("S-Expression (Gold)"):
            st.code(data["sexpr"], language="lisp")

        # Display function list
        with st.expander("Function List (Gold)"):
            for i, func in enumerate(data["function_list"]):
                st.code(f"{i}: {func}", language="python")

        # Display metadata
        st.subheader("Metadata")
        metadata_cols = st.columns(3)
        with metadata_cols[0]:
            st.metric("Level", data.get("level", "unknown"))
        with metadata_cols[1]:
            st.metric("Split", data.get("metadata", {}).get("split", "unknown"))
        with metadata_cols[2]:
            st.metric("Workflow Length", data.get("metadata", {}).get("workflow_length", len(data["dag"])))

        # Display DAG statistics
        st.header("📊 DAG Statistics")
        dag_nodes = data["dag"]
        dag_stats = analyze_dag_structure(dag_nodes)

        metrics_cols = st.columns(5)
        with metrics_cols[0]:
            st.metric("Nodes", dag_stats.num_nodes)
        with metrics_cols[1]:
            st.metric("Branches", dag_stats.num_branches)
        with metrics_cols[2]:
            st.metric("Merges", dag_stats.num_merges)
        with metrics_cols[3]:
            st.metric("Max Parents", dag_stats.max_parents)
        with metrics_cols[4]:
            st.metric("Max Children", dag_stats.max_children)

        # Display DAG visualization
        st.header("🔗 Program Graph")

        if dag_nodes:
            # Create and display graph
            graph = create_dag_graph(dag_nodes)
            st.graphviz_chart(graph.source)

            # Display detailed steps in expandable sections
            st.header("📋 Detailed Steps")

            # Build children mapping
            children_map = {i: [] for i in range(len(dag_nodes))}
            for i, node in enumerate(dag_nodes):
                parents = node.get("dependencies", [])
                for parent_idx in parents:
                    if 0 <= parent_idx < len(dag_nodes):
                        children_map[parent_idx].append(i)

            for i, node in enumerate(dag_nodes):
                function = node.get("function", "").strip()
                inputs = node.get("inputs", [])
                parents = node.get("dependencies", [])
                children = children_map[i]

                with st.expander(
                    f"Step {i}: {function if function else 'Unknown'}"
                ):
                    st.markdown(f"**Function:** `{function}`")

                    if inputs:
                        st.markdown(f"**Inputs:** `{inputs}`")

                    # Show relationships
                    rel_col1, rel_col2 = st.columns(2)
                    with rel_col1:
                        if parents:
                            st.markdown(f"**Parents:** {', '.join(map(str, parents))}")
                        else:
                            st.markdown("**Parents:** None (Root node)")

                    with rel_col2:
                        if children:
                            st.markdown(
                                f"**Children:** {', '.join(map(str, children))}"
                            )
                        else:
                            st.markdown("**Children:** None (Leaf node)")
        else:
            st.warning("No DAG data found.")

    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        ### Instructions

        1. **Data Directory**: Enter the path to your atomic KBQA dataset directory (e.g., `data/atomic_kbqa/grailqa`)
        2. **Search & Filter**: Use the sliders to filter examples by:
           - **Nodes**: Total number of program steps
           - **Branches**: Number of steps with multiple children (parallel execution)
           - **Merges**: Number of steps with multiple parents (convergence points)
        3. **Select Example**: Pick from filtered examples showing (N:nodes, B:branches, M:merges)
        4. **View Results**:
           - See the original question and gold answer
           - S-expression and function list
           - DAG structure statistics
           - Interactive DAG visualization
           - Detailed step-by-step breakdown

        ### DAG Visualization Legend
        - 🟢 **Green nodes**: START operations (entity/type initialization)
        - 🔴 **Red nodes**: STOP operations (final result)
        - 🟡 **Yellow nodes**: JOIN, AND, OR operations (relational/logical)
        - 🩷 **Pink nodes**: ARG* operations (aggregation, superlatives)
        - 🔵 **Blue nodes**: Other operations
        - **Arrows**: Dependencies between steps (parent → child)

        ### DAG Structure Metrics
        - **Nodes**: Total program steps
        - **Branches**: Steps that spawn multiple parallel paths (children ≥ 2)
        - **Merges**: Steps that combine multiple paths (parents ≥ 2)
        - **Max Parents/Children**: Highest degree of convergence/divergence

        ### Atomic KBQA Operations
        Common operations in GrailQA/WebQSP/GraphQ:
        - **START**: Initialize with an entity or type
        - **JOIN**: Navigate a relation from current expression
        - **AND**: Intersect two expressions (set intersection)
        - **OR**: Union two expressions (set union)
        - **ARGMAX/ARGMIN**: Find entities with max/min values
        - **COUNT**: Count number of entities
        - **TC**: Type constraint (filter by type)
        - **STOP**: Mark final result
        """)


if __name__ == "__main__":
    # Entry point of `streamlit run dag_visualizer_atomic_kbqa.py`
    parser = argparse.ArgumentParser("Atomic KBQA DAG Visualizer")
    parser.add_argument(
        "--data",
        dest="data_dir",
        type=Path,
        default="data/atomic_kbqa/grailqa",
        help="Path to the atomic KBQA dataset directory (will load from processed/ subdirectory)",
    )
    args = parser.parse_args()

    main(args)
