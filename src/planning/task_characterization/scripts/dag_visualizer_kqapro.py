#!/usr/bin/env python3
"""
Streamlit app for visualizing reasoning DAGs.

Usage:
    streamlit run dag_visualizer.py
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


def load_dag_file(file_path: str) -> dict[str, Any]:
    """Load a DAG file and return its contents."""
    print(f"Loading DAG file: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_verification_file(data_dir: str) -> dict[str, list[str]]:
    """Load the verification file and return verified example IDs by dataset."""
    verification_path = os.path.join(data_dir, "verified.v1.json")
    try:
        with open(verification_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Create empty verification file if it doesn't exist
        empty_verification = {
            "gsm8k": [],
            "math": [],
            "mathqa": [],
            "hotpotqa": [],
            "strategyqa": [],
        }
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


def toggle_verification(
    data_dir: str, dataset_name: str, example_id: str, is_verified: bool
) -> None:
    """Toggle verification status of an example."""
    verification_data = load_verification_file(data_dir)

    # Ensure dataset exists in verification data
    if dataset_name not in verification_data:
        verification_data[dataset_name] = []

    if is_verified:
        # Add to verified list if not already there
        if example_id not in verification_data[dataset_name]:
            verification_data[dataset_name].append(example_id)
            verification_data[dataset_name].sort()  # Keep sorted
    else:
        # Remove from verified list if present
        if example_id in verification_data[dataset_name]:
            verification_data[dataset_name].remove(example_id)

    save_verification_file(data_dir, verification_data)


def load_discard_file(data_dir: str) -> dict[str, list[str]]:
    """Load the discard file and return discarded example IDs by dataset."""
    discard_path = os.path.join(data_dir, "discarded.v1.json")
    try:
        with open(discard_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Create empty discard file if it doesn't exist
        empty_discard = {
            "gsm8k": [],
            "math": [],
            "mathqa": [],
            "hotpotqa": [],
            "strategyqa": [],
        }
        save_discard_file(data_dir, empty_discard)
        return empty_discard
    except json.JSONDecodeError:
        # Handle corrupted file
        return {}


def save_discard_file(data_dir: str, discard_data: dict[str, list[str]]) -> None:
    """Save the discard data to the discard file."""
    discard_path = os.path.join(data_dir, "discarded.v1.json")
    with open(discard_path, "w", encoding="utf-8") as f:
        json.dump(discard_data, f, indent=2, sort_keys=True)


def toggle_discard(
    data_dir: str, dataset_name: str, example_id: str, is_discarded: bool
) -> None:
    """Toggle discard status of an example."""
    discard_data = load_discard_file(data_dir)

    # Ensure dataset exists in discard data
    if dataset_name not in discard_data:
        discard_data[dataset_name] = []

    if is_discarded:
        # Add to discarded list if not already there
        if example_id not in discard_data[dataset_name]:
            discard_data[dataset_name].append(example_id)
            discard_data[dataset_name].sort()  # Keep sorted
    else:
        # Remove from discarded list if present
        if example_id in discard_data[dataset_name]:
            discard_data[dataset_name].remove(example_id)

    save_discard_file(data_dir, discard_data)


def analyze_dag_structure(dag_data: list[dict[str, Any]]) -> DAGStats:
    """Analyze the structure of a DAG and return statistics."""
    if not dag_data:
        return DAGStats(0, 0, 0, 0, 0)

    num_nodes = len(dag_data)
    num_branches = 0  # nodes with multiple children
    num_merges = 0  # nodes with multiple parents
    max_parents = 0
    max_children = 0

    for node in dag_data:
        parents = node.get("parents", [])
        children = node.get("children", [])

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


@st.cache_data
def get_available_datasets_and_examples_with_stats(
    data_dir: str,
) -> dict[str, list[dict[str, Any]]]:
    """Get KQA Pro examples with DAG statistics."""
    datasets = {}
    data_path = Path(data_dir)

    if not data_path.exists():
        return {}

    # Load verification and discard data
    verification_data = load_verification_file(data_dir)
    discard_data = load_discard_file(data_dir)

    # Load KQA Pro preprocessed data (with DAGs)
    processed_dir = data_path / "processed"
    train_path = processed_dir / "train.v1.json"
    val_path = processed_dir / "val.v1.json"

    data = []
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            data_ = orjson.loads(f.read())
            data.extend(data_)
    if val_path.exists():
        with open(val_path, "r", encoding="utf-8") as f:
            data_ = orjson.loads(f.read())
            data.extend(data_)

    if not data:
        return {}

    dataset_name = "kqapro"
    examples = []
    verified_ids = set(verification_data.get(dataset_name, []))
    discarded_ids = set(discard_data.get(dataset_name, []))

    for i, item in enumerate(data):
        # Process original program
        program = item["program"]
        program_nodes = []
        for j, step in enumerate(program):
            node = {
                "index": j,
                "parents": step["dependencies"],
                "children": [],
                "action": step["function"],
                "args": str(step["inputs"]),
            }
            program_nodes.append(node)

        # Compute children for program
        for node in program_nodes:
            for p in node["parents"]:
                if p >= 0:
                    program_nodes[p]["children"].append(node["index"])

        # Process deduplicated DAG if available
        dag_nodes = None
        if "dag" in item:
            dag = item["dag"]
            dag_nodes = []
            for j, step in enumerate(dag):
                node = {
                    "index": j,
                    "parents": step["dependencies"],
                    "children": [],
                    "action": step["function"],
                    "args": str(step["inputs"]),
                }
                dag_nodes.append(node)

            # Compute children for DAG
            for node in dag_nodes:
                for p in node["parents"]:
                    if p >= 0:
                        dag_nodes[p]["children"].append(node["index"])

        # Always use DAG stats if available (for fair comparison), otherwise program stats
        if dag_nodes:
            stats = analyze_dag_structure(dag_nodes)
        else:
            stats = analyze_dag_structure(program_nodes)
        example_id = item.get("id", str(i))

        examples.append(
            {
                "id": example_id,
                "program_data": {
                    "id": example_id,
                    "query": item["question"],
                    "dag": program_nodes,
                },
                "dag_data": {
                    "id": example_id,
                    "query": item["question"],
                    "dag": dag_nodes,
                }
                if dag_nodes
                else None,
                "stats": stats,
                "verified": example_id in verified_ids,
                "discarded": example_id in discarded_ids,
                "query": item["question"][:100] + "..."
                if len(item["question"]) > 100
                else item["question"],
            }
        )

    datasets[dataset_name] = examples
    return datasets


def create_dag_graph(dag_data: list[dict[str, Any]]) -> graphviz.Digraph:
    """Create a graphviz representation of the KQA Pro program DAG."""
    dot = graphviz.Digraph(comment="KQA Pro Program")
    dot.attr(rankdir="TB")  # Top to bottom layout
    dot.attr("node", shape="box", style="filled", fillcolor="lightblue")

    # Add program nodes
    for node in dag_data:
        index = node["index"]
        function = node.get("action", "").strip()  # function is stored in action
        inputs = node.get("args", "").strip()  # inputs are stored in args

        # Create label with multiple lines
        label_parts = [f"[{index}]"]

        if function:
            label_parts.append(f"Function: {function}")

        if inputs:
            # Truncate long inputs
            if len(inputs) > 60:
                inputs = inputs[:57] + "..."
            label_parts.append(f"Inputs: {inputs}")

        label = "\\n".join(label_parts) if label_parts else f"Step {index}"

        dot.node(str(index), label)

    # Add edges based on parent-child relationships
    for node in dag_data:
        index = node["index"]
        children = node.get("children", [])

        for child_index in children:
            dot.edge(str(index), str(child_index))

    return dot


def main(args):
    st.set_page_config(page_title="KQA Pro Program Visualizer", layout="wide")

    st.title("🔍 KQA Pro Program Visualizer")
    st.markdown("Interactive visualization of KQA Pro reasoning programs")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Data directory input
        default_data_dir = str(args.data_dir)
        data_dir = st.text_input(
            "Data Directory Path",
            value=default_data_dir,
            help="Path to the KQA Pro data directory",
        )

        # Check if directory exists
        if not os.path.exists(data_dir):
            st.error(f"Directory '{data_dir}' does not exist!")
            st.stop()

        # Show loading message and load data
        with st.spinner("Loading KQA Pro data..."):
            datasets_examples = get_available_datasets_and_examples_with_stats(data_dir)

        if datasets_examples:
            # Show summary of loaded data
            total_examples = len(datasets_examples.get("kqapro", []))
            st.success(f"Loaded {total_examples} KQA Pro examples")

    # Get available examples
    if not datasets_examples or "kqapro" not in datasets_examples:
        st.error(f"No KQA Pro data found in '{data_dir}'")
        st.stop()

    dataset_name = "kqapro"
    examples = datasets_examples[dataset_name]

    # Search and filter functionality
    st.header("🔍 Search & Filter")

    search_col1, search_col2, search_col3 = st.columns(3)

    with search_col1:
        node_range = st.slider(
            "Number of Nodes",
            min_value=1,
            max_value=50,
            value=(1, 50),
            help="Filter examples by number of nodes in the DAG",
        )

    with search_col2:
        branch_range = st.slider(
            "Number of Branches",
            min_value=0,
            max_value=10,
            value=(0, 10),
            help="Filter examples by number of nodes with multiple children",
        )

    with search_col3:
        merge_range = st.slider(
            "Number of Merges",
            min_value=0,
            max_value=10,
            value=(0, 10),
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
        for example in sorted(
            filtered_examples, key=lambda x: int(x["id"].split("_")[-1])
        ):
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
        # View selector: Original Program vs Deduplicated DAG
        has_dag = selected_example["dag_data"] is not None

        if has_dag:
            view_options = [
                "Original Program",
                "Deduplicated DAG",
                "Side-by-Side Comparison",
            ]
            selected_view = st.radio(
                "Select View",
                options=view_options,
                horizontal=True,
                help="Choose between original program, deduplicated DAG, or compare both",
            )
        else:
            selected_view = "Original Program"
            st.info("Only original program available (preprocessed data not found)")

        # Display query
        st.header("Question")
        question = selected_example["program_data"].get("query", "No question found")
        st.text(question)

        # Determine which data to display
        if selected_view == "Side-by-Side Comparison":
            # Show both program and DAG side-by-side
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Program")
                program_nodes = selected_example["program_data"].get("dag", [])
                program_stats = analyze_dag_structure(program_nodes)

                # Display program statistics
                metrics_cols = st.columns(5)
                with metrics_cols[0]:
                    st.metric("Nodes", program_stats.num_nodes)
                with metrics_cols[1]:
                    st.metric("Branches", program_stats.num_branches)
                with metrics_cols[2]:
                    st.metric("Merges", program_stats.num_merges)
                with metrics_cols[3]:
                    st.metric("Max Parents", program_stats.max_parents)
                with metrics_cols[4]:
                    st.metric("Max Children", program_stats.max_children)

                if program_nodes:
                    graph = create_dag_graph(program_nodes)
                    st.graphviz_chart(graph.source)

            with col2:
                st.subheader("Deduplicated DAG")
                dag_nodes = selected_example["dag_data"].get("dag", [])
                dag_stats = analyze_dag_structure(dag_nodes)

                # Display DAG statistics
                metrics_cols = st.columns(5)
                with metrics_cols[0]:
                    st.metric(
                        "Nodes",
                        dag_stats.num_nodes,
                        delta=dag_stats.num_nodes - program_stats.num_nodes,
                    )
                with metrics_cols[1]:
                    st.metric(
                        "Branches",
                        dag_stats.num_branches,
                        delta=dag_stats.num_branches - program_stats.num_branches,
                    )
                with metrics_cols[2]:
                    st.metric(
                        "Merges",
                        dag_stats.num_merges,
                        delta=dag_stats.num_merges - program_stats.num_merges,
                    )
                with metrics_cols[3]:
                    st.metric("Max Parents", dag_stats.max_parents)
                with metrics_cols[4]:
                    st.metric("Max Children", dag_stats.max_children)

                if dag_nodes:
                    graph = create_dag_graph(dag_nodes)
                    st.graphviz_chart(graph.source)

                # Show compression ratio
                if program_stats.num_nodes > 0:
                    compression = program_stats.num_nodes / dag_stats.num_nodes
                    st.success(
                        f"Compression Ratio: {compression:.2f}x ({program_stats.num_nodes} → {dag_stats.num_nodes} nodes)"
                    )

            # Skip the rest of the single-view display
            dag_nodes = None
        else:
            # Single view: show either program or DAG
            if selected_view == "Deduplicated DAG" and has_dag:
                current_data = selected_example["dag_data"]
                st.info("Viewing deduplicated DAG (merged equivalent subgraphs)")
            else:
                current_data = selected_example["program_data"]
                if has_dag:
                    st.info("Viewing original program (with potential redundancy)")

            dag_nodes = current_data.get("dag", [])
            dag_stats = analyze_dag_structure(dag_nodes)

        # Single-view display (skip if in comparison mode)
        if dag_nodes is not None:
            # Create metrics columns
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

            for node in sorted(dag_nodes, key=lambda x: x["index"]):
                index = node["index"]
                function = node.get("action", "").strip()
                inputs = node.get("args", "").strip()
                parents = node.get("parents", [])
                children = node.get("children", [])

                with st.expander(
                    f"Step {index}: {function.title() if function else 'Unknown'}"
                ):
                    if function:
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

        1. **Data Directory**: Enter the path to your KQA Pro data directory (e.g., `data/kqa_pro`)
        2. **Search & Filter**: Use the sliders to filter examples by:
           - **Nodes**: Total number of program steps
           - **Branches**: Number of steps with multiple children (parallel execution)
           - **Merges**: Number of steps with multiple parents (convergence points)
        3. **Select Example**: Pick from filtered examples showing (N:nodes, B:branches, M:merges)
        4. **View Results**:
           - See the original question
           - DAG structure statistics
           - Interactive DAG visualization of the program
           - Detailed step-by-step breakdown

        ### DAG Visualization Legend
        - 🔵 **Blue nodes**: Program steps
        - **Arrows**: Dependencies between steps (parent → child)

        ### DAG Structure Metrics
        - **Nodes**: Total program steps
        - **Branches**: Steps that spawn multiple parallel paths (children ≥ 2)
        - **Merges**: Steps that combine multiple paths (parents ≥ 2)
        - **Max Parents/Children**: Highest degree of convergence/divergence

        ### KQA Pro Program Format
        Each program consists of steps with:
        - **function**: KoPL operator (e.g., Find, FilterStr, And)
        - **dependencies**: List of parent step indices
        - **inputs**: Arguments for the function
        """)
        st.markdown("""
        ### Instructions

        1. **Data Directory**: Enter the path to your KQA Pro data directory (e.g., `data/kqa_pro`)
        2. **Search & Filter**: Use the sliders to filter examples by:
           - **Nodes**: Total number of program steps
           - **Branches**: Number of steps with multiple children (parallel execution)
           - **Merges**: Number of steps with multiple parents (convergence points)
        3. **Select Dataset**: KQA Pro dataset is loaded
        4. **Select Example**: Pick from filtered examples showing (N:nodes, B:branches, M:merges)
        5. **View Results**:
           - See the original question
           - DAG structure statistics
           - Interactive DAG visualization of the program
           - Detailed step-by-step breakdown

        ### DAG Visualization Legend
        - 🔵 **Blue nodes**: Regular program steps
        - **Arrows**: Dependencies between steps (parent → child)

        ### DAG Structure Metrics
        - **Nodes**: Total program steps
        - **Branches**: Steps that spawn multiple parallel paths (children ≥ 2)
        - **Merges**: Steps that combine multiple paths (parents ≥ 2)
        - **Max Parents/Children**: Highest degree of convergence/divergence

        ### KQA Pro Program Format
        Each program consists of steps with:
        - **function**: KoPL operator (e.g., Find, FilterStr, And)
        - **dependencies**: List of parent step indices
        - **inputs**: Arguments for the function
        """)


if __name__ == "__main__":
    # Entry point of `streamlit run dag_visualizer_kqapro.py`
    parser = argparse.ArgumentParser("KQA Pro DAG Visualizer")
    parser.add_argument(
        "--data",
        dest="data_dir",
        type=Path,
        default="data/kqa_pro",
        help="Path to the KQA Pro data directory (will load from processed/ subdirectory)",
    )
    args = parser.parse_args()

    main(args)
