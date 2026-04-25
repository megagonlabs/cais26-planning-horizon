#!/usr/bin/env python3
"""
Streamlit app for visualizing multi-objective HotpotQA reasoning DAGs.

Usage:
    streamlit run data/multiobj_hotpotqa/scripts/multiobj_dag_visualizer.py
"""

import json
import argparse
from pathlib import Path
from typing import Any

import streamlit as st
import graphviz


def load_data(file_path: str) -> list[dict[str, Any]]:
    """Load the annotated dataset."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_dag_graph(dag_data: list[dict[str, Any]]) -> graphviz.Digraph:
    """Create a graphviz representation of the DAG."""
    dot = graphviz.Digraph(comment="Reasoning DAG")
    dot.attr(rankdir="TB")  # Top to bottom layout
    dot.attr("node", shape="box", style="filled", fillcolor="lightblue")

    # Add nodes
    for i, node in enumerate(dag_data):
        func = node.get("function", "unknown")
        inputs = node.get("inputs", [])

        # Create label
        label_parts = [f"[{i}] {func.upper()}"]
        if inputs:
            # Join inputs and truncate if too long
            input_text = " | ".join(str(x) for x in inputs)
            if len(input_text) > 100:
                input_text = input_text[:97] + "..."
            label_parts.append(input_text)

        label = "\n".join(label_parts)

        # Color coding
        fillcolor = "lightblue"
        if func == "finish":
            fillcolor = "lightgreen"
        elif func == "aggregation":
            fillcolor = "lightyellow"

        dot.node(str(i), label, fillcolor=fillcolor)

    # Add edges based on dependencies
    for i, node in enumerate(dag_data):
        deps = node.get("dependencies", [])
        for dep in deps:
            dot.edge(str(dep), str(i))

    return dot


def main():
    st.set_page_config(page_title="Multi-objective DAG Visualizer", layout="wide")

    st.title("🔍 Multi-objective HotpotQA DAG Visualizer")
    st.markdown("Interactive visualization of reasoning DAGs for multi-objective questions.")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        default_path = "data/multiobj_hotpotqa/processed/test.v1.annotated.json"
        file_path = st.text_input("Input File Path", value=default_path)

        if not Path(file_path).exists():
            st.error(f"File not found: {file_path}")
            st.stop()

        @st.cache_data
        def get_data(path):
            return load_data(path)

        data = get_data(file_path)
        st.success(f"Loaded {len(data)} examples")

        st.header("Filters")

        # Filter by k (number of components)
        k_values = sorted(list(set(ex["metadata"]["k"] for ex in data)))
        selected_ks = st.multiselect("Number of Components (k)", options=k_values, default=k_values)

        # Filter by number of nodes
        node_counts = [len(ex["dag"]) for ex in data]
        min_nodes, max_nodes = min(node_counts), max(node_counts)
        selected_node_range = st.slider("Number of Nodes", min_nodes, max_nodes, (min_nodes, max_nodes))

    # Apply filters
    filtered_data = [
        ex for ex in data
        if ex["metadata"]["k"] in selected_ks
        and selected_node_range[0] <= len(ex["dag"]) <= selected_node_range[1]
    ]

    if not filtered_data:
        st.warning("No examples match the selected filters.")
        st.stop()

    # Navigation
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0

    # Reset index if it's out of bounds after filtering
    if st.session_state.current_idx >= len(filtered_data):
        st.session_state.current_idx = 0

    st.sidebar.header("Navigation")
    col_prev, col_next, col_rand = st.sidebar.columns(3)

    if col_prev.button("Prev"):
        st.session_state.current_idx = (st.session_state.current_idx - 1) % len(filtered_data)

    if col_next.button("Next"):
        st.session_state.current_idx = (st.session_state.current_idx + 1) % len(filtered_data)

    if col_rand.button("Random"):
        import random
        st.session_state.current_idx = random.randint(0, len(filtered_data) - 1)

    # Example selection
    example_options = [f"{ex['id']} (k={ex['metadata']['k']}, nodes={len(ex['dag'])})" for ex in filtered_data]
    selected_idx = st.selectbox(
        f"Select Example ({len(filtered_data)} matches)",
        range(len(filtered_data)),
        index=st.session_state.current_idx,
        format_func=lambda i: example_options[i],
        key="example_selector"
    )

    # Sync session state with selectbox
    st.session_state.current_idx = selected_idx

    example = filtered_data[st.session_state.current_idx]

    # Display Example
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Question")
        st.write(example["question"])

        st.subheader("Answers")
        for i, ans in enumerate(example["answers"]):
            st.write(f"{i+1}. {ans}")

        with st.expander("Metadata"):
            st.json(example["metadata"])

    with col2:
        st.subheader("Reasoning DAG")
        if example["dag"]:
            graph = create_dag_graph(example["dag"])
            st.graphviz_chart(graph)
        else:
            st.warning("No DAG data available for this example.")

    # Detailed Steps
    st.subheader("Detailed Steps")
    for i, node in enumerate(example["dag"]):
        with st.expander(f"Step {i}: {node.get('function', 'unknown').upper()}"):
            st.write(f"**Dependencies:** {node.get('dependencies', [])}")
            st.write(f"**Inputs:** {node.get('inputs', [])}")


if __name__ == "__main__":
    main()
