"""
Focused unit tests for DAG generation and merging logic.

These tests validate the core DAG manipulation functions that will be used
in the preprocessing pipeline.
"""

import unittest
from typing import List, Dict, Any


# DAG node type
DagNode = Dict[str, Any]


def generate_bridge_dag() -> List[DagNode]:
    """
    Generate DAG structure for a bridge question.

    Bridge: Linear two-hop reasoning (sub-question 1 -> sub-question 2)

    Returns:
        DAG with 3 nodes: search -> search -> finish
    """
    return [
        {"function": "search", "dependencies": [], "inputs": ["<bridge_sub_q1>"]},
        {"function": "search", "dependencies": [0], "inputs": ["<bridge_sub_q2>"]},
        {"function": "finish", "dependencies": [1], "inputs": []}
    ]


def generate_comparison_dag() -> List[DagNode]:
    """
    Generate DAG structure for a comparison question.

    Comparison: Parallel reasoning with aggregation
    (sub-question 1, sub-question 2 -> aggregation)

    Returns:
        DAG with 4 nodes: search, search -> aggregation -> finish
    """
    return [
        {"function": "search", "dependencies": [], "inputs": ["<comp_sub_q1>"]},
        {"function": "search", "dependencies": [], "inputs": ["<comp_sub_q2>"]},
        {"function": "aggregation", "dependencies": [0, 1], "inputs": ["<aggregation>"]},
        {"function": "finish", "dependencies": [2], "inputs": []}
    ]


def merge_dags(dags: List[List[DagNode]]) -> List[DagNode]:
    """
    Merge multiple DAGs into a single DAG with one finish node.

    Algorithm:
    1. Remove the finish node from each input DAG
    2. Concatenate all remaining nodes, renumbering dependencies
    3. Find all terminal nodes (nodes with no dependents)
    4. Add a single finish node depending on all terminals

    Args:
        dags: List of DAG structures to merge

    Returns:
        Merged DAG with single finish node
    """
    if len(dags) == 1:
        return dags[0]

    merged = []
    offset = 0
    terminal_nodes = []

    for dag in dags:
        # Remove finish node (should be last)
        dag_without_finish = dag[:-1]

        # Find terminal node(s) in this DAG (nodes that finish depends on)
        finish_node = dag[-1]
        terminals_in_dag = finish_node["dependencies"]

        # Add nodes with renumbered dependencies
        for node in dag_without_finish:
            new_node = {
                "function": node["function"],
                "dependencies": [dep + offset for dep in node["dependencies"]],
                "inputs": node["inputs"].copy()
            }
            merged.append(new_node)

        # Track terminal nodes with offset applied
        terminal_nodes.extend([t + offset for t in terminals_in_dag])
        offset += len(dag_without_finish)

    # Add single finish node depending on all terminals
    merged.append({
        "function": "finish",
        "dependencies": terminal_nodes,
        "inputs": []
    })

    return merged


def compute_dag_depth(dag: List[DagNode]) -> int:
    """
    Compute the depth of a DAG (longest path from root to finish).

    Args:
        dag: DAG structure

    Returns:
        Maximum depth (number of nodes in longest path)
    """
    if not dag:
        return 0

    # Build adjacency list for forward traversal
    # For each node, track which nodes depend on it
    dependents = [[] for _ in range(len(dag))]
    for i, node in enumerate(dag):
        for dep in node["dependencies"]:
            dependents[dep].append(i)

    # BFS from root nodes (nodes with no dependencies)
    depths = [0] * len(dag)
    queue = []

    for i, node in enumerate(dag):
        if not node["dependencies"]:
            queue.append(i)
            depths[i] = 1

    while queue:
        current = queue.pop(0)
        current_depth = depths[current]

        for dependent in dependents[current]:
            depths[dependent] = max(depths[dependent], current_depth + 1)
            if dependent not in queue:
                queue.append(dependent)

    return max(depths)


def compute_dag_width(dag: List[DagNode]) -> int:
    """
    Compute the width of a DAG (maximum number of nodes at any level).

    Args:
        dag: DAG structure

    Returns:
        Maximum width
    """
    if not dag:
        return 0

    # Build adjacency list
    dependents = [[] for _ in range(len(dag))]
    for i, node in enumerate(dag):
        for dep in node["dependencies"]:
            dependents[dep].append(i)

    # Assign levels via BFS
    levels = [-1] * len(dag)
    queue = []

    for i, node in enumerate(dag):
        if not node["dependencies"]:
            queue.append(i)
            levels[i] = 0

    while queue:
        current = queue.pop(0)
        current_level = levels[current]

        for dependent in dependents[current]:
            levels[dependent] = max(levels[dependent], current_level + 1)
            if dependent not in queue:
                queue.append(dependent)

    # Count nodes at each level
    max_level = max(levels)
    level_counts = [0] * (max_level + 1)
    for level in levels:
        level_counts[level] += 1

    return max(level_counts)


class TestBridgeDAG(unittest.TestCase):
    """Test bridge DAG generation."""

    def test_bridge_structure(self):
        """Test that bridge DAG has correct structure."""
        dag = generate_bridge_dag()

        self.assertEqual(len(dag), 3, "Bridge DAG should have 3 nodes")
        self.assertEqual(dag[0]["function"], "search")
        self.assertEqual(dag[1]["function"], "search")
        self.assertEqual(dag[2]["function"], "finish")

        self.assertEqual(dag[0]["dependencies"], [])
        self.assertEqual(dag[1]["dependencies"], [0])
        self.assertEqual(dag[2]["dependencies"], [1])

    def test_bridge_depth(self):
        """Test bridge DAG depth."""
        dag = generate_bridge_dag()
        depth = compute_dag_depth(dag)
        self.assertEqual(depth, 3, "Bridge DAG depth should be 3")

    def test_bridge_width(self):
        """Test bridge DAG width."""
        dag = generate_bridge_dag()
        width = compute_dag_width(dag)
        self.assertEqual(width, 1, "Bridge DAG width should be 1 (sequential)")


class TestComparisonDAG(unittest.TestCase):
    """Test comparison DAG generation."""

    def test_comparison_structure(self):
        """Test that comparison DAG has correct structure."""
        dag = generate_comparison_dag()

        self.assertEqual(len(dag), 4, "Comparison DAG should have 4 nodes")
        self.assertEqual(dag[0]["function"], "search")
        self.assertEqual(dag[1]["function"], "search")
        self.assertEqual(dag[2]["function"], "aggregation")
        self.assertEqual(dag[3]["function"], "finish")

        self.assertEqual(dag[0]["dependencies"], [])
        self.assertEqual(dag[1]["dependencies"], [])
        self.assertEqual(dag[2]["dependencies"], [0, 1])
        self.assertEqual(dag[3]["dependencies"], [2])

    def test_comparison_depth(self):
        """Test comparison DAG depth."""
        dag = generate_comparison_dag()
        depth = compute_dag_depth(dag)
        self.assertEqual(depth, 3, "Comparison DAG depth should be 3")

    def test_comparison_width(self):
        """Test comparison DAG width."""
        dag = generate_comparison_dag()
        width = compute_dag_width(dag)
        self.assertEqual(width, 2, "Comparison DAG width should be 2 (parallel searches)")


class TestDAGMerging(unittest.TestCase):
    """Test DAG merging logic."""

    def test_merge_single_dag(self):
        """Test that merging single DAG returns it unchanged."""
        bridge = generate_bridge_dag()
        merged = merge_dags([bridge])
        self.assertEqual(merged, bridge)

    def test_merge_two_bridges(self):
        """Test merging two bridge DAGs."""
        bridge1 = generate_bridge_dag()
        bridge2 = generate_bridge_dag()

        merged = merge_dags([bridge1, bridge2])

        # Should have: 2 bridges * 2 search nodes + 1 finish = 5 nodes
        self.assertEqual(len(merged), 5)
        self.assertEqual(merged[-1]["function"], "finish")

        # Finish should depend on terminals from both bridges (nodes 1 and 3)
        self.assertEqual(set(merged[-1]["dependencies"]), {1, 3})

    def test_merge_bridge_and_comparison(self):
        """Test merging one bridge and one comparison DAG."""
        bridge = generate_bridge_dag()
        comparison = generate_comparison_dag()

        merged = merge_dags([bridge, comparison])

        # Should have:
        # - Bridge: 2 search nodes (indices 0, 1)
        # - Comparison: 2 search + 1 aggregation (indices 2, 3, 4)
        # - Merged finish (index 5)
        # Total: 6 nodes
        self.assertEqual(len(merged), 6)
        self.assertEqual(merged[-1]["function"], "finish")

        # Finish should depend on bridge terminal (1) and comparison terminal (4)
        self.assertEqual(set(merged[-1]["dependencies"]), {1, 4})

    def test_merge_dependencies_renumbered(self):
        """Test that dependencies are correctly renumbered after merge."""
        bridge = generate_bridge_dag()
        comparison = generate_comparison_dag()

        merged = merge_dags([bridge, comparison])

        # Bridge nodes (0, 1):
        # - Node 0: no dependencies
        # - Node 1: depends on 0
        self.assertEqual(merged[0]["dependencies"], [])
        self.assertEqual(merged[1]["dependencies"], [0])

        # Comparison nodes (2, 3, 4):
        # - Node 2: no dependencies
        # - Node 3: no dependencies
        # - Node 4: depends on 2, 3 (renumbered from 0, 1)
        self.assertEqual(merged[2]["dependencies"], [])
        self.assertEqual(merged[3]["dependencies"], [])
        self.assertEqual(set(merged[4]["dependencies"]), {2, 3})

    def test_merge_three_dags(self):
        """Test merging three DAGs."""
        bridge = generate_bridge_dag()
        comp1 = generate_comparison_dag()
        comp2 = generate_comparison_dag()

        merged = merge_dags([bridge, comp1, comp2])

        # Bridge: 2 nodes (0-1)
        # Comp1: 3 nodes (2-4)
        # Comp2: 3 nodes (5-7)
        # Finish: 1 node (8)
        self.assertEqual(len(merged), 9)
        self.assertEqual(merged[-1]["function"], "finish")

        # Finish depends on terminals: 1 (bridge), 4 (comp1), 7 (comp2)
        self.assertEqual(set(merged[-1]["dependencies"]), {1, 4, 7})

    def test_merged_dag_depth(self):
        """Test depth of merged DAG."""
        bridge = generate_bridge_dag()
        comparison = generate_comparison_dag()

        merged = merge_dags([bridge, comparison])
        depth = compute_dag_depth(merged)

        # Merged DAG has parallel execution so depth doesn't increase
        # Bridge: search[0] -> search[1] (depth 2)
        # Comparison: search[2], search[3] -> aggregation[4] (depth 2)
        # Finish[5] depends on max(1, 4) = depth 3
        self.assertEqual(depth, 3)

    def test_merged_dag_width(self):
        """Test width of merged DAG."""
        bridge = generate_bridge_dag()
        comparison = generate_comparison_dag()

        merged = merge_dags([bridge, comparison])
        width = compute_dag_width(merged)

        # At level 0: bridge_search[0], comp_search[2], comp_search[3] = 3 nodes
        # This is the widest level
        self.assertGreaterEqual(width, 2)


if __name__ == "__main__":
    unittest.main()
