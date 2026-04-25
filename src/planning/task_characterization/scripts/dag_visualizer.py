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
import datetime
import json
import os
import random

import streamlit as st
import graphviz


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


def get_random_unverified_example(
    data_dir: str, datasets_examples: dict[str, list[dict[str, Any]]]
) -> dict[str, Any] | None:
    """Get a random unverified and undiscarded example from all datasets."""
    verification_data = load_verification_file(data_dir)
    discard_data = load_discard_file(data_dir)
    unverified_examples = []

    for dataset_name, examples in datasets_examples.items():
        verified_ids = set(verification_data.get(dataset_name, []))
        discarded_ids = set(discard_data.get(dataset_name, []))
        for example in examples:
            if example["id"] not in verified_ids and example["id"] not in discarded_ids:
                example_with_dataset = example.copy()
                example_with_dataset["dataset"] = dataset_name
                unverified_examples.append(example_with_dataset)

    if unverified_examples:
        return random.choice(unverified_examples)
    else:
        return None


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
    """Get all available datasets and their examples with DAG statistics."""
    datasets = {}
    data_path = Path(data_dir)

    if not data_path.exists():
        return {}

    # Load verification and discard data
    verification_data = load_verification_file(data_dir)
    discard_data = load_discard_file(data_dir)

    # Get all dataset directories first
    dataset_dirs = [d for d in data_path.iterdir() if d.is_dir()]

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        examples = []
        verified_ids = set(verification_data.get(dataset_name, []))
        discarded_ids = set(discard_data.get(dataset_name, []))

        # Get all JSON files in this dataset
        json_files = list(dataset_dir.glob("*.json"))

        for file_path in json_files:
            try:
                # Load file to get statistics
                dag_data = load_dag_file(str(file_path))
                dag_nodes = dag_data.get("dag", [])
                stats = analyze_dag_structure(dag_nodes)

                # Extract example ID from filename (e.g., gsm8k_1.json -> 1)
                example_id = dag_data["id"]

                examples.append(
                    {
                        "id": example_id,
                        "file_path": str(file_path),
                        "stats": stats,
                        "verified": example_id in verified_ids,
                        "discarded": example_id in discarded_ids,
                        "query": dag_data.get("query", "")[:100] + "..."
                        if len(dag_data.get("query", "")) > 100
                        else dag_data.get("query", ""),
                    }
                )
            except Exception:
                # Skip files that can't be loaded
                continue

        if examples:
            # Sort numerically if possible, otherwise alphabetically
            try:
                examples.sort(key=lambda x: int(x["id"]))
            except ValueError:
                examples.sort(key=lambda x: x["id"])
            datasets[dataset_name] = examples

    return datasets


def create_dag_graph(dag_data: list[dict[str, Any]]) -> graphviz.Digraph:
    """Create a graphviz representation of the DAG."""
    dot = graphviz.Digraph(comment="Reasoning DAG")
    dot.attr(rankdir="TB")  # Top to bottom layout
    dot.attr("node", shape="box", style="filled", fillcolor="lightblue")

    # Add nodes
    ## Add user input node
    dot.node("input", "Input Text", shape="ellipse", fillcolor="lightyellow")

    ## Add reasoning nodes
    for node in dag_data:
        index = node["index"]
        thought = node.get("thought", "").strip()
        action = node.get("action", "").strip()
        args = node.get("args", "").strip()
        observation = node.get("observation", "").strip()

        # Create label with multiple lines
        label_parts = [f"[{index}]"]

        if thought:
            # Truncate long thoughts
            if len(thought) > 80:
                thought = thought[:77] + "..."
            label_parts.append(f"Thought: {thought}")

        if action:
            label_parts.append(f"Action: {action}")

        if args:
            # Truncate long args
            if len(args) > 60:
                args = args[:57] + "..."
            label_parts.append(f"Args: {args}")

        if observation:
            # Truncate long observations
            if len(observation) > 60:
                observation = observation[:57] + "..."
            label_parts.append(f"Observation: {observation}")

        label = "\\n".join(label_parts) if label_parts else f"Step {index}"

        # Color final answer nodes differently
        if action == "finish":
            dot.node(str(index), label, fillcolor="lightgreen")
        else:
            dot.node(str(index), label)

    # Add edges based on parent-child relationships
    ## Add edges between the input node and reasoning nodes
    children = [node["index"] for node in dag_data if -1 in node["parents"]]
    for child_index in children:
        dot.edge("input", str(child_index))

    ## Add edges between reasoning nodes
    for node in dag_data:
        index = node["index"]
        children = node.get("children", [])

        for child_index in children:
            dot.edge(str(index), str(child_index))

    return dot


def main(args):
    st.set_page_config(page_title="DAG Visualizer", layout="wide")

    st.title("🔍 Reasoning DAG Visualizer")
    st.markdown(
        "Interactive visualization of reasoning trajectories as Directed Acyclic Graphs"
    )

    # Initialize session state for random selection
    if "random_selection" not in st.session_state:
        st.session_state.random_selection = None

    # Initialize session state for current selection
    if "current_dataset" not in st.session_state:
        st.session_state.current_dataset = None
    if "current_example_id" not in st.session_state:
        st.session_state.current_example_id = None

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Data directory input
        default_data_dir = str(args.data_dir)
        data_dir = st.text_input(
            "Data Directory Path",
            value=default_data_dir,
            help="Path to the directory containing DAG files",
        )

        # Check if directory exists
        if not os.path.exists(data_dir):
            st.error(f"Directory '{data_dir}' does not exist!")
            st.stop()

        # Show loading message and load data
        with st.spinner("Loading datasets and examples..."):
            datasets_examples = get_available_datasets_and_examples_with_stats(data_dir)

        if datasets_examples:
            # Show summary of loaded data
            total_examples = sum(len(examples) for examples in datasets_examples.values())
            st.success(f"Loaded {len(datasets_examples)} datasets with {total_examples} total examples")

        # Random unverified example button
        if st.button(
            "🎲 Random Unverified",
            help="Show a random unverified and undiscarded example from any dataset",
        ):
            random_example = get_random_unverified_example(data_dir, datasets_examples)
            if random_example:
                st.session_state.random_selection = {
                    "dataset": random_example["dataset"],
                    "example_id": random_example["id"],
                }
                st.rerun()
            else:
                st.info("All examples have been verified!")

    # Get available datasets and examples with statistics
    if not datasets_examples:
        st.error(f"No datasets found in '{data_dir}'")
        st.stop()

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

    # Dataset selection
    col1, col2 = st.columns(2)

    with col1:
        # Check if we have a random selection or current selection
        if st.session_state.random_selection:
            default_dataset = st.session_state.random_selection["dataset"]
        elif st.session_state.current_dataset:
            default_dataset = st.session_state.current_dataset
        else:
            default_dataset = list(datasets_examples.keys())[0] if datasets_examples else None

        # Find index of the default dataset
        dataset_options = list(datasets_examples.keys())
        if default_dataset and default_dataset in dataset_options:
            default_index = dataset_options.index(default_dataset)
        else:
            default_index = 0

        dataset_name = st.selectbox(
            "Select Dataset",
            options=dataset_options,
            index=default_index,
            help="Choose a dataset to visualize",
        )

        # Update current dataset in session state
        if dataset_name != st.session_state.current_dataset:
            st.session_state.current_dataset = dataset_name
            st.session_state.current_example_id = None  # Reset example when dataset changes

    with col2:
        filtered_examples = []
        if dataset_name:
            examples = datasets_examples[dataset_name]

            # Filter examples based on search criteria
            for example in examples:
                stats = example["stats"]
                if (
                    node_range[0] <= stats.num_nodes <= node_range[1]
                    and branch_range[0] <= stats.num_branches <= branch_range[1]
                    and merge_range[0] <= stats.num_merges <= merge_range[1]
                ):
                    filtered_examples.append(example)

            if filtered_examples:
                # Create display options with statistics and verification/discard status
                display_options = []
                default_selection_index = 0

                for i, example in enumerate(sorted(filtered_examples, key=lambda x: int(x["id"].rsplit("_", 1)[-1]))):
                    stats = example["stats"]
                    verified_mark = "✓ " if example["verified"] else ""
                    discarded_mark = "✗ " if example["discarded"] else ""
                    status_mark = f"{verified_mark}{discarded_mark}"
                    display_text = f"{status_mark}{example['id']} (N:{stats.num_nodes}, B:{stats.num_branches}, M:{stats.num_merges})"
                    display_options.append(display_text)

                    # Set default selection based on priority: random selection > current selection > first item
                    if (
                        st.session_state.random_selection
                        and st.session_state.random_selection["dataset"] == dataset_name
                        and st.session_state.random_selection["example_id"] == example["id"]
                    ):
                        default_selection_index = i
                    elif (
                        st.session_state.current_example_id == example["id"]
                        and not st.session_state.random_selection
                    ):
                        default_selection_index = i

                selected_display = st.selectbox(
                    f"Select Example ({len(filtered_examples)} matches)",
                    options=display_options,
                    index=default_selection_index,
                    help="Choose a specific example to visualize. N=Nodes, B=Branches, M=Merges. ✓ indicates verified, ✗ indicates discarded examples",
                )

                # Extract the example ID from the selected display text
                if selected_display:
                    # Remove status marks if present
                    clean_display = selected_display.lstrip("✓ ✗ ")
                    example_id = clean_display.split(" ")[0]
                    selected_example = next(
                        ex for ex in filtered_examples if ex["id"] == example_id
                    )

                    # Update current example in session state
                    st.session_state.current_example_id = example_id

                    # Clear random selection after use
                    if st.session_state.random_selection:
                        st.session_state.random_selection = None
                else:
                    selected_example = None
            else:
                st.warning("No examples match the current search criteria")
                selected_example = None
        else:
            selected_example = None

    # Display search results summary
    if dataset_name and selected_example:
        st.info(
            f"Found {len(filtered_examples)} examples matching your criteria in dataset '{dataset_name}'"
        )

    if selected_example:
        dag_file_path = selected_example["file_path"]

        # Add reload, verification, and discard buttons for the current example
        col_file_info, col_reload, col_verify, col_discard = st.columns([3, 1, 1, 1])

        with col_file_info:
            # Display file path and last modified time
            file_stat = os.stat(dag_file_path)

            last_modified = datetime.datetime.fromtimestamp(file_stat.st_mtime)
            st.caption(
                f"📁 File: {dag_file_path} | Last modified: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}"
            )

        with col_reload:
            # Initialize reload trigger for this specific file if not exists
            file_key = f"reload_{hash(dag_file_path)}"
            if file_key not in st.session_state:
                st.session_state[file_key] = 0

            if st.button(
                "🔄 Reload",
                help=f"Reload this specific file: {os.path.basename(dag_file_path)}",
                key=f"reload_btn_{hash(dag_file_path)}",
            ):
                st.session_state[file_key] += 1
                st.rerun()

        with col_verify:
            # Verification toggle
            current_verified = selected_example["verified"]
            verify_label = "✅ Verified" if current_verified else "❌ Verify"
            verify_help = "Click to toggle verification status"

            if st.button(
                verify_label, help=verify_help, key=f"verify_btn_{hash(dag_file_path)}"
            ):
                # Toggle verification
                toggle_verification(
                    data_dir, dataset_name, selected_example["id"], not current_verified
                )
                # Force cache refresh for datasets
                get_available_datasets_and_examples_with_stats.clear()
                st.rerun()

        with col_discard:
            # Discard toggle
            current_discarded = selected_example["discarded"]
            discard_label = "🗑️ Discarded" if current_discarded else "🗑️ Discard"
            discard_help = "Click to toggle discard status"

            if st.button(
                discard_label, help=discard_help, key=f"discard_btn_{hash(dag_file_path)}"
            ):
                # Toggle discard
                toggle_discard(
                    data_dir, dataset_name, selected_example["id"], not current_discarded
                )
                # Force cache refresh for datasets
                get_available_datasets_and_examples_with_stats.clear()
                st.rerun()

        try:
            # Load the selected DAG file (reload trigger will force cache invalidation)
            dag_data = load_dag_file(dag_file_path)

            # Display query
            st.header("📝 Query")
            query = dag_data.get("query", "No query found")
            st.text(query)

            # Display metadata and DAG statistics
            dag_nodes = dag_data.get("dag", [])
            dag_stats = analyze_dag_structure(dag_nodes)

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

            # Display additional metadata if available
            if "metadata" in dag_data:
                metadata = dag_data["metadata"]

                with st.expander("📊 Additional Metadata"):
                    meta_col1, meta_col2, meta_col3 = st.columns(3)

                    with meta_col1:
                        if "num_steps" in metadata:
                            st.metric("Total Steps", metadata["num_steps"])
                        if "num_actions" in metadata:
                            st.metric("Total Actions", metadata["num_actions"])

                    with meta_col2:
                        if "model" in metadata:
                            model_name = str(metadata["model"])
                            if len(model_name) > 25:
                                model_display = model_name[:22] + "..."
                            else:
                                model_display = model_name
                            st.metric("Model", model_display)

                        if "timestamp" in metadata:
                            st.text(f"Timestamp: {metadata['timestamp']}")

                    with meta_col3:
                        if "usage" in metadata:
                            usage = metadata["usage"]
                            if "total_tokens" in usage:
                                st.metric("Total Tokens", usage["total_tokens"])

            # Display DAG visualization
            st.header("🔗 Reasoning DAG")

            if dag_nodes:
                # Create and display graph
                graph = create_dag_graph(dag_nodes)
                st.graphviz_chart(graph.source)

                # Display detailed steps in expandable sections
                st.header("📋 Detailed Steps")

                for i, node in enumerate(sorted(dag_nodes, key=lambda x: x["index"])):
                    index = node["index"]
                    thought = node.get("thought", "").strip()
                    action = node.get("action", "").strip()
                    args = node.get("args", "").strip()
                    observation = node.get("observation", "").strip()
                    parents = node.get("parents", [])
                    children = node.get("children", [])

                    with st.expander(
                        f"Step {index}: {action.title() if action else 'Unknown'}"
                    ):
                        if thought:
                            st.markdown(f"**Thought:** {thought}")

                        if action:
                            st.markdown(f"**Action:** `{action}`")

                        if args:
                            st.markdown(f"**Arguments:** `{args}`")

                        if observation:
                            st.markdown(f"**Observation:** `{observation}`")

                        # Show relationships
                        rel_col1, rel_col2 = st.columns(2)
                        with rel_col1:
                            if parents:
                                st.markdown(
                                    f"**Parents:** {', '.join(map(str, parents))}"
                                )
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
                st.warning("No DAG data found in the selected file.")

        except FileNotFoundError:
            st.error(f"File not found: {dag_file_path}")
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON file: {e}")
        except Exception as e:
            st.error(f"Error loading DAG data: {e}")

    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        ### Instructions

        1. **Data Directory**: Enter the path to your DAG data directory (e.g., `data/agentbank/processed/dags/auto.v1`)
        2. **Random Unverified**: Click "🎲 Random Unverified" to jump to a random example that hasn't been verified or discarded
        3. **Search & Filter**: Use the sliders to filter examples by:
           - **Nodes**: Total number of reasoning steps
           - **Branches**: Number of nodes with multiple children (parallel execution)
           - **Merges**: Number of nodes with multiple parents (convergence points)
        4. **Select Dataset**: Choose from available datasets (e.g., gsm8k, hotpotqa, etc.)
        5. **Select Example**: Pick from filtered examples showing (N:nodes, B:branches, M:merges)
           - Examples marked with ✓ are verified
           - Examples marked with ✗ are discarded
        6. **Actions**:
           - **🔄 Reload**: Refresh the current example after editing its DAG file
           - **✅/❌ Verify**: Toggle verification status (saves to verified.v1.json)
           - **🗑️ Discard**: Toggle discard status (saves to discarded.v1.json)
        7. **View Results**:
           - See the original query and file info
           - DAG structure statistics and metadata
           - Interactive DAG visualization
           - Detailed step-by-step breakdown

        ### Verification and Discard Workflow
        - Use "🎲 Random Unverified" to find examples needing review (excludes both verified and discarded)
        - Review the DAG visualization and reasoning steps
        - Click "❌ Verify" to mark as correct (becomes "✅ Verified")
        - Click "🗑️ Discard" to mark as problematic (becomes "🗑️ Discarded")
        - Status changes are saved to `verified.v1.json` and `discarded.v1.json`
        - Verified examples are marked with ✓ in the dropdown
        - Discarded examples are marked with ✗ in the dropdown

        ### DAG Visualization Legend
        - 🔵 **Blue nodes**: Regular reasoning steps
        - 🟢 **Green nodes**: Final answer steps
        - **Arrows**: Dependencies between steps (parent → child)

        ### DAG Structure Metrics
        - **Nodes**: Total reasoning steps in the trajectory
        - **Branches**: Steps that spawn multiple parallel paths (children ≥ 2)
        - **Merges**: Steps that combine multiple paths (parents ≥ 2)
        - **Max Parents/Children**: Highest degree of convergence/divergence

        ### Search Examples
        - Simple linear: Nodes=1-10, Branches=0, Merges=0
        - Parallel computation: Nodes=3-15, Branches=1-5, Merges=0-2
        - Complex reasoning: Nodes=5+, Branches=2+, Merges=1+

        ### File Format
        Expected JSON structure:
        ```json
        {
          "id": "example_id",
          "query": "The original question",
          "dag": [
            {
              "index": 0,
              "thought": "reasoning text",
              "action": "action_type",
              "args": "action_arguments",
              "observation": "result",
              "parents": [],
              "children": [1]
            }
          ],
          "metadata": {
            "num_steps": 3,
            "dag_valid": true
          }
        }
        ```

        ### Status Files
        - **verified.v1.json**: Contains IDs of verified examples by dataset
        - **discarded.v1.json**: Contains IDs of discarded examples by dataset
        """)


if __name__ == "__main__":
    # Entry point of `stream run dag_visualizer.py`
    parser = argparse.ArgumentParser("DAG Visualizer")
    parser.add_argument("--data", dest="data_dir", type=Path, default="data/agentbank/processed/dags/auto.v1", help="Path to the DAG data directory")
    args = parser.parse_args()

    main(args)
