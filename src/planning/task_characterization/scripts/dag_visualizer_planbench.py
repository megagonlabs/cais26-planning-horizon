#!/usr/bin/env python3
"""
Streamlit app for visualizing PlanBench reasoning DAGs.

Usage:
    streamlit run dag_visualizer_planbench.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
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


def load_planbench_data(data_dir: str) -> dict[str, list[dict[str, Any]]]:
    """Load PlanBench processed data and return examples by domain."""
    datasets = {}
    data_path = Path(data_dir) / "processed"

    if not data_path.exists():
        return {}

    # Load all PlanBench domains
    domain_files = {
        "blocksworld_basic": "blocksworld_basic.v1.json",
        "blocksworld_randomized": "blocksworld_randomized.v1.json",
        "logistics_basic": "logistics_basic.v1.json",
    }

    for domain_name, filename in domain_files.items():
        file_path = data_path / filename
        if not file_path.exists():
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = orjson.loads(f.read())

            examples = []
            for item in data:
                if "dag" not in item or not item["dag"]:
                    continue

                # Convert PlanBench DAG format to visualizer format
                dag_nodes = []
                for j, step in enumerate(item["dag"]):
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

                stats = analyze_dag_structure(dag_nodes)

                examples.append({
                    "id": item["id"],
                    "domain": domain_name,
                    "query": item["question"],
                    "goal": item["goal"],
                    "dag": dag_nodes,
                    "stats": stats,
                    "objects": item.get("objects", []),
                    "initial_state": item.get("initial_state", {}),
                    "goal_state": item.get("goal_state", {}),
                })

            if examples:
                datasets[domain_name] = examples

        except Exception as e:
            st.warning(f"Error loading {filename}: {e}")
            continue

    return datasets


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


def create_dag_graph(dag_data: list[dict[str, Any]], domain: str = "unknown") -> graphviz.Digraph:
    """Create a graphviz representation of the PlanBench DAG."""
    dot = graphviz.Digraph(comment=f"PlanBench {domain} DAG")
    dot.attr(rankdir="TB")  # Top to bottom layout

    # Color scheme based on domain
    if "blocksworld" in domain:
        node_color = "lightblue"
    elif "logistics" in domain:
        node_color = "lightgreen"
    else:
        node_color = "lightgray"

    dot.attr("node", shape="box", style="filled", fillcolor=node_color)

    # Add DAG nodes
    for node in dag_data:
        index = node["index"]
        action = node.get("action", "").strip()
        inputs = node.get("args", "").strip()

        # Create label with multiple lines
        label_parts = [f"[{index}]"]

        if action:
            label_parts.append(f"Action: {action}")

        if inputs:
            # Truncate long inputs for readability
            if len(inputs) > 40:
                inputs = inputs[:37] + "..."
            label_parts.append(f"Args: {inputs}")

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
    st.set_page_config(page_title="PlanBench DAG Visualizer", layout="wide")

    st.title("🔍 PlanBench DAG Visualizer")
    st.markdown("Interactive visualization of PlanBench planning problems")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Data directory input
        default_data_dir = str(args.data_dir)
        data_dir = st.text_input(
            "Data Directory Path",
            value=default_data_dir,
            help="Path to the PlanBench data directory",
        )

        # Check if directory exists
        if not os.path.exists(data_dir):
            st.error(f"Directory '{data_dir}' does not exist!")
            st.stop()

        # Show loading message and load data
        with st.spinner("Loading PlanBench data..."):
            datasets = load_planbench_data(data_dir)

        if datasets:
            total_domains = len(datasets)
            total_examples = sum(len(examples) for examples in datasets.values())
            st.success(f"Loaded {total_examples} examples across {total_domains} domains")

            # Domain selector
            available_domains = list(datasets.keys())
            selected_domain = st.selectbox(
                "Select Domain",
                options=available_domains,
                help="Choose which PlanBench domain to explore"
            )
        else:
            st.error(f"No PlanBench data found in '{data_dir}/processed/'")
            st.stop()

    # Get selected domain data
    if not datasets or selected_domain not in datasets:
        st.error(f"Selected domain '{selected_domain}' not found")
        st.stop()

    examples = datasets[selected_domain]

    # Search and filter functionality
    st.header("🔍 Search & Filter")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        node_range = st.slider(
            "Number of Nodes",
            min_value=1,
            max_value=50,
            value=(1, 50),
            help="Filter examples by number of actions in the plan",
        )

    with filter_col2:
        branch_range = st.slider(
            "Number of Branches",
            min_value=0,
            max_value=10,
            value=(0, 10),
            help="Filter examples by number of nodes with multiple children",
        )

    with filter_col3:
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
        for example in sorted(filtered_examples, key=lambda x: x["id"]):
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
        # Display problem information
        st.header("📋 Problem Information")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Initial State")
            initial_state = selected_example.get("initial_state", {})
            predicates = initial_state.get("predicates", [])
            if predicates:
                st.code("\n".join(predicates[:10]))  # Show first 10 predicates
                if len(predicates) > 10:
                    st.text(f"... and {len(predicates) - 10} more predicates")
            else:
                st.text("No initial state predicates found")

        with col2:
            st.subheader("Goal State")
            goal_state = selected_example.get("goal_state", {})
            predicates = goal_state.get("predicates", [])
            if predicates:
                st.code("\n".join(predicates))
            else:
                st.text("No goal state predicates found")

        # Display question/goal
        st.subheader("Problem Description")
        question = selected_example.get("query", "No description found")
        st.text(question)

        goal = selected_example.get("goal", "")
        if goal:
            st.text(f"**Goal:** {goal}")

        # Display objects
        objects = selected_example.get("objects", [])
        if objects:
            st.text(f"**Objects:** {', '.join(objects)}")

        # DAG statistics and visualization
        st.header("🔗 Plan DAG")

        dag_nodes = selected_example.get("dag", [])
        dag_stats = selected_example["stats"]

        # Create metrics columns
        metrics_cols = st.columns(5)

        with metrics_cols[0]:
            st.metric("Actions", dag_stats.num_nodes)

        with metrics_cols[1]:
            st.metric("Branches", dag_stats.num_branches)

        with metrics_cols[2]:
            st.metric("Merges", dag_stats.num_merges)

        with metrics_cols[3]:
            st.metric("Max Parents", dag_stats.max_parents)

        with metrics_cols[4]:
            st.metric("Max Children", dag_stats.max_children)

        # Display DAG visualization
        if dag_nodes:
            graph = create_dag_graph(dag_nodes, selected_domain)
            st.graphviz_chart(graph.source)

            # Display detailed steps in expandable sections
            st.header("📋 Detailed Action Sequence")

            for node in sorted(dag_nodes, key=lambda x: x["index"]):
                index = node["index"]
                action = node.get("action", "").strip()
                inputs = node.get("args", "").strip()
                parents = node.get("parents", [])
                children = node.get("children", [])

                with st.expander(
                    f"Action {index}: {action.title() if action else 'Unknown'}"
                ):
                    if action:
                        st.markdown(f"**Action:** `{action}`")

                    if inputs:
                        st.markdown(f"**Parameters:** `{inputs}`")

                    # Show relationships
                    rel_col1, rel_col2 = st.columns(2)
                    with rel_col1:
                        if parents:
                            st.markdown(f"**Dependencies:** {', '.join(map(str, parents))}")
                        else:
                            st.markdown("**Dependencies:** None (Independent action)")

                    with rel_col2:
                        if children:
                            st.markdown(
                                f"**Dependent Actions:** {', '.join(map(str, children))}"
                            )
                        else:
                            st.markdown("**Dependent Actions:** None (Final action)")
        else:
            st.warning("No DAG data found for this example.")

    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        ### Instructions

        1. **Data Directory**: Enter the path to your PlanBench data directory (e.g., `data/planbench`)
        2. **Select Domain**: Choose between Blocksworld and Logistics domains
        3. **Search & Filter**: Use the sliders to filter examples by:
           - **Nodes**: Total number of actions in the plan
           - **Branches**: Number of actions that enable parallel execution
           - **Merges**: Number of actions that depend on multiple previous actions
        4. **Select Example**: Pick from filtered examples showing (N:nodes, B:branches, M:merges)
        5. **View Results**:
           - Problem description and goal
           - Initial and goal states
           - DAG structure statistics
           - Interactive DAG visualization
           - Detailed action-by-action breakdown

        ### DAG Visualization Legend
        - 🔵 **Blue nodes (Blocksworld)**: Block manipulation actions
        - 🟢 **Green nodes (Logistics)**: Transportation actions
        - **Arrows**: Dependencies between actions (parent → child)

        ### DAG Structure Metrics
        - **Actions**: Total number of PDDL actions in the plan
        - **Branches**: Actions that spawn multiple parallel paths
        - **Merges**: Actions that combine multiple dependency paths
        - **Max Parents/Children**: Highest degree of convergence/divergence

        ### PlanBench Action Format
        Each action consists of:
        - **Action**: PDDL action name (e.g., unstack, pickup, load-truck)
        - **Parameters**: Objects the action operates on
        - **Dependencies**: Previous actions that must complete first
        """)


if __name__ == "__main__":
    # Entry point of `streamlit run dag_visualizer_planbench.py`
    parser = argparse.ArgumentParser("PlanBench DAG Visualizer")
    parser.add_argument(
        "--data",
        dest="data_dir",
        type=Path,
        default="data/planbench",
        help="Path to the PlanBench data directory (will load from processed/ subdirectory)",
    )
    args = parser.parse_args()

    main(args)
