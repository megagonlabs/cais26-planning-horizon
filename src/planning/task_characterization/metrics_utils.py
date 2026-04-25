"""
metrics_utils.py

Utility functions for DAG complexity metrics.
"""
from collections import defaultdict
from typing import Any, Optional

def compute_dag_complexity(nodes: list[dict[str, Any]], ignore_input_node: Optional[bool] = True) -> dict[str, Any]:
    """
    Compute a set of complexity metrics for a reasoning DAG as defined in docs/trajectory_dag_schema.md.

    Inputs:
        nodes: list of node dicts, each having at least:
            - index: int
            - parents: list[int]
            - children: list[int]
        ignore_input_node: if True, treat -1 in parents as an input node (root) and exclude it in graph edges.

    Returns:
        dict[str, Any]: Dictionary of computed DAG metrics:
            - num_nodes: (int) Number of nodes in the DAG.
            - num_edges: (int) Number of edges (excluding input node edges if ignore_input_node is True).
            - num_roots: (int) Number of root nodes (nodes with no parents).
            - num_leaves: (int) Number of leaf nodes (nodes with no children).
            - num_nonfinish_leaves: (int) Number of leaf nodes whose action is not "finish".
            - max_gap_including_input: (int) Maximum dependency gap (index difference to earliest parent, including input node).
            - max_gap_excluding_input: (int) Maximum dependency gap (excluding input node).
            - avg_gap_including_input: (float) Average dependency gap (including input node).
            - avg_gap_excluding_input: (float) Average dependency gap (excluding input node).
            - max_depth: (int) Maximum depth (longest path from a root to any node).
            - max_width: (int) Maximum width (largest number of nodes at any depth level).
            - merging_ratio: (float) Fraction of nodes with more than one parent (merge points).
            - branching_ratio: (float) Fraction of nodes with more than one child (branch points).
            - avg_headroom: (float) Average headroom (number of steps that could be executed before a node, but are not required).
            - max_headroom: (int) Maximum headroom among all nodes.
            - min_headroom: (int) Minimum headroom among all nodes.
            - max_best_join_span: (int) Maximum best-case join span among all nodes. (difference in ancestor set size between node and its largest parent).
            - max_worst_join_span: (int) Maximum worst-case join span among all nodes. (difference in ancestor set size between node and its smallest parent).
            - avg_best_join_span: (float) Average best-case join span
            - avg_worst_join_span: (float) Average worst-case join span
    """
    if not nodes:
        return {}

    n = len(nodes)
    indices = list(range(n))

    parents_orig: list[list[int]] = [[] for _ in range(n)]
    parents_excl: list[list[int]] = [[] for _ in range(n)]
    for i, node in enumerate(nodes):
        pars = node["parents"]
        parents_orig[i] = pars[:]
        if ignore_input_node:
            parents_excl[i] = [p for p in pars if p != -1]
        else:
            parents_excl[i] = pars

    children: list[list[int]] = [[] for _ in range(n)]
    for child_idx, pars in enumerate(parents_excl):
        for p in pars:
            children[p].append(child_idx)

    in_degree = [len(pars) for pars in parents_excl]
    out_degree = [len(children[i]) for i in indices]
    num_edges = sum(in_degree)

    roots = [i for i, value in enumerate(in_degree) if value == 0]
    leaves = [i for i, value in enumerate(out_degree) if value == 0]

    num_nonfinish_leaves = 0
    for i in leaves:
        act = (nodes[i].get("action") or "").lower()
        if act != "finish":
            num_nonfinish_leaves += 1

    ancestors: list[set[int]] = [set() for _ in range(n)]
    for i in indices:
        for p in parents_excl[i]:
            ancestors[i].add(p)
            ancestors[i] |= ancestors[p]

    def dep_gap(i: int, include_input: Optional[bool] = False) -> int:
        pars = parents_orig[i] if include_input else parents_excl[i]
        if not pars:
            return 0
        return i - min(pars)

    gaps_incl = [dep_gap(i, include_input=True) for i in indices if parents_orig[i]]
    gaps_excl = [dep_gap(i, include_input=False) for i in indices if parents_excl[i]]

    max_gap_incl = max(gaps_incl) if gaps_incl else 0
    max_gap_excl = max(gaps_excl) if gaps_excl else 0
    avg_gap_incl = (sum(gaps_incl) / len(gaps_incl)) if gaps_incl else 0.0
    avg_gap_excl = (sum(gaps_excl) / len(gaps_excl)) if gaps_excl else 0.0

    root_nodes = [i for i, pars in enumerate(parents_excl) if len(pars) == 0]
    idx2depth = [0 if idx in root_nodes else -1 for idx in range(n)]
    memory = root_nodes[:]
    visited = set()
    while memory:
        u = memory.pop(0)
        visited.add(u)
        for v in children[u]:
            idx2depth[v] = max(idx2depth[v], idx2depth[u] + 1)
            if v not in visited and v not in memory:
                memory.append(v)
    assert all(d != -1 for d in idx2depth)

    depth2indices = defaultdict(list)
    for idx, depth in enumerate(idx2depth):
        depth2indices[depth].append(idx)
    max_depth = max(depth2indices.keys()) if depth2indices else 0
    max_width = max(len(indices) for indices in depth2indices.values()) if depth2indices else 0

    merging_count = sum(1 for i in indices if in_degree[i] > 1)
    branching_count = sum(1 for i in indices if out_degree[i] > 1)
    merging_ratio = merging_count / n if n else 0.0
    branching_ratio = branching_count / n if n else 0.0

    headrooms = []
    for u in indices:
        anc = ancestors[u]
        best = len(anc)
        worst = sum(1 for v in indices if v != u and u not in ancestors[v])
        headrooms.append(max(0, worst - best))
    max_headroom = max(headrooms) if headrooms else 0
    min_headroom = min(headrooms) if headrooms else 0
    avg_headroom = sum(headrooms) / n if n else 0.0

    best_join_spans, worst_join_spans = [], []
    for u in indices:
        anc = ancestors[u]
        parent_ancestor_lens = [len(ancestors[p]) for p in parents_excl[u]] if parents_excl[u] else []
        if parent_ancestor_lens:
            best_join_span = len(anc) - max(parent_ancestor_lens)
            worst_join_span = len(anc) - min(parent_ancestor_lens)
        else:
            best_join_span = worst_join_span = 0
        best_join_spans.append(best_join_span)
        worst_join_spans.append(worst_join_span)
    max_best_join_span = max(best_join_spans) if best_join_spans else 0
    max_worst_join_span = max(worst_join_spans) if worst_join_spans else 0
    avg_best_join_span = sum(best_join_spans) / len(best_join_spans) if best_join_spans else 0.0
    avg_worst_join_span = sum(worst_join_spans) / len(worst_join_spans) if worst_join_spans else 0.0

    return {
        "num_nodes": n,
        "num_edges": num_edges,
        "num_roots": len(roots),
        "num_leaves": len(leaves),
        "num_nonfinish_leaves": num_nonfinish_leaves,
        "max_gap_including_input": max_gap_incl,
        "max_gap_excluding_input": max_gap_excl,
        "avg_gap_including_input": avg_gap_incl,
        "avg_gap_excluding_input": avg_gap_excl,
        "max_depth": max_depth,
        "max_width": max_width,
        "merging_ratio": merging_ratio,
        "branching_ratio": branching_ratio,
        "avg_headroom": avg_headroom,
        "max_headroom": max_headroom,
        "min_headroom": min_headroom,
        "max_best_join_span": max_best_join_span,
        "max_worst_join_span": max_worst_join_span,
        "avg_best_join_span": avg_best_join_span,
        "avg_worst_join_span": avg_worst_join_span
    }
