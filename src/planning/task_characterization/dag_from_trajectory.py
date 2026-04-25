"""
dag_from_trajectory.py

Utilities for converting linear reasoning trajectories into DAG representations.
"""

from typing import Optional
import json
import re


def parse_gpt_turn(
    gpt_value: str,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse a GPT turn to extract thought, action, and args.

    Args:
        gpt_value (str): The GPT turn content

    Returns:
        tuple[Optional[str], Optional[str], Optional[str]]: (thought, action, args)
    """
    thought = None
    action = None
    args = None

    # Handle Final Answer case
    if "Final Answer:" in gpt_value:
        thought = "Final Answer"
        action = "finish"
        # Extract the answer after "Final Answer:"
        final_answer_match = re.search(r"Final Answer:\s*(.+)", gpt_value, re.DOTALL)
        if final_answer_match:
            args = final_answer_match.group(1).strip()
        return thought, action, args

    # Extract thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|$)", gpt_value, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extract action
    action_match = re.search(r"Action:\s*(.+)", gpt_value, re.DOTALL)
    if action_match:
        action_content = action_match.group(1).strip()
        args = action_content

        # Determine action type based on content
        if "```python" in action_content:
            action = "code"
            args = re.sub(r"```python\n|\n```", "", args).strip()
        elif any(op in action_content for op in ["+", "-", "*", "/", "=", "%"]):
            action = "math"
        elif "search" in action_content.lower():
            action = "search"
        else:
            action = "math"  # Default to math for most AgentBank cases

        # Remove the action name from args (e.g., search[X] -> X)
        _args = re.sub(rf"^{action}\[", "", args, count=1)
        if _args != args:
            args = _args.strip("]")

    return thought, action, args


def convert_trajectory_to_dag(conversations: list[dict], include_dependencies: bool = True) -> list[dict]:
    """
    Convert original trajectory conversations to a list of DAG nodes.

    Each node is a dictionary with the following fields:
        - index (int): Unique step index, starting from 0
        - thought (str): Reasoning or explanation before taking action
        - action (str): Type of action taken (e.g., "math", "search", "finish")
        - args (str): Arguments or input for the action
        - observation (str): Result or output from executing the action
        - parents (list[int]): Indices of steps this step depends on. Index -1 represents the user query.
        - children (list[int]): Indices of steps that depend on this step

    Args:
        conversations (list[dict]): Original conversation turns
        include_dependencies (bool): Whether to include parent/child dependencies in the DAG (default: True)

    Returns:
        list[dict]: List of DAG nodes as described above
    """
    dag_nodes = []
    gpt_turns = []

    # Extract GPT turns and their corresponding observations
    for i, turn in enumerate(conversations):
        if turn["from"] == "gpt":
            observation = ""
            # Look for the next human turn to get observation
            if i + 1 < len(conversations) and conversations[i + 1]["from"] == "human":
                human_response = conversations[i + 1]["value"]
                # Extract observation content
                obs_match = re.search(r"Observation:\s*(.+)", human_response, re.DOTALL)
                if obs_match:
                    observation = obs_match.group(1).strip()

            gpt_turns.append({"content": turn["value"], "observation": observation})

    # Create DAG nodes
    for idx, gpt_turn in enumerate(gpt_turns):
        thought, action, args = parse_gpt_turn(gpt_turn["content"])

        # Determine parents and children based on trajectory structure

        # For most AgentBank trajectories, steps are sequential (linear dependency)
        parents = [idx - 1]  # Previous step is the parent. Node -1 represents the user query
        if idx < len(gpt_turns) - 1:
            children = [idx + 1]
        else:
            children = []

        observation = gpt_turn["observation"]
        if observation.startswith("Observation:"):
            observation = observation[len("Observation:"):].strip()

        node = {
            "index": idx,
            "thought": thought or "",
            "action": action or "",
            "args": args or "",
            "observation": observation
        }
        if include_dependencies:
            node["parents"] = parents
            node["children"] = children

        dag_nodes.append(node)

    return dag_nodes


def validate_dag(nodes: list[dict]) -> tuple[bool, list[str]]:
    """
    Validate that the DAG is well-formed according to trajectory_dag_schema.md.

    Args:
        nodes (list[dict]): List of DAG node dictionaries

    Returns:
        tuple[bool, list[str]]: (is_valid, list_of_errors)
    """
    errors = []

    if not nodes:
        return True, []

    # Check index consistency
    indices = [node["index"] for node in nodes]
    expected_indices = list(range(len(nodes)))
    if sorted(indices) != expected_indices:
        errors.append(f"Indices are not sequential: {indices}")

    # Check all nodes have parents
    if not all(node["parents"] for node in nodes):
        errors.append("All nodes must have parents")

    # Check root node
    if not any(node["parents"] == [-1] for node in nodes):
        errors.append("No root node found (at least one node must have -1 as parent)")

    # Check parent-child consistency
    for node in nodes:
        node_idx = node["index"]

        for child_idx in node["children"]:
            if child_idx >= len(nodes):
                errors.append(f"Node {node_idx} references invalid child {child_idx}")
            elif child_idx > 0 and node_idx not in nodes[child_idx]["parents"]:
                errors.append(
                    f"Node {node_idx} lists {child_idx} as child, but {child_idx} doesn't list {node_idx} as parent"
                )

        for parent_idx in node["parents"]:
            if parent_idx >= len(nodes):
                errors.append(f"Node {node_idx} references invalid parent {parent_idx}")
            elif parent_idx >= 0 and node_idx not in nodes[parent_idx]["children"]:
                errors.append(
                    f"Node {node_idx} lists {parent_idx} as parent, but {parent_idx} doesn't list {node_idx} as child"
                )

    # Check for cycles (simple DFS)
    def has_cycle(start_idx, visited, rec_stack):
        visited[start_idx] = True
        rec_stack[start_idx] = True

        for child_idx in nodes[start_idx]["children"]:
            if child_idx == -1:  # User query node
                continue
            if not visited[child_idx]:
                if has_cycle(child_idx, visited, rec_stack):
                    return True
            elif rec_stack[child_idx]:
                return True

        rec_stack[start_idx] = False
        return False

    visited = [False] * len(nodes)
    rec_stack = [False] * len(nodes)

    for i in range(len(nodes)):
        if not visited[i]:
            if has_cycle(i, visited, rec_stack):
                errors.append("DAG contains cycles")
                break

    return len(errors) == 0, errors


def save_dag_to_json(nodes: list[dict], filepath: str):
    """
    Save DAG nodes to JSON file.

    Args:
        nodes (list[dict]): List of DAG node dictionaries
        filepath (str): Output file path
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=2, ensure_ascii=False)


def load_dag_from_json(filepath: str) -> list[dict]:
    """
    Load DAG nodes from JSON file.

    Args:
        filepath (str): Input file path

    Returns:
        list[dict]: List of DAG node dictionaries
    """
    with open(filepath, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    return nodes
