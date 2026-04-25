"""
Converts KOPL programs into DAGs by identifying commutable filtering operations and merging equivalent nodes.

This module provides functionality to compact KOPL (Knowledge-oriented Programming Language) programs
by applying two key optimizations:

1. **Canonical Reordering**: Reorder commutable filtering operations into a canonical order
   (FilterConcept first, QFilterDate last) to enable better merging across different orderings.

2. **Recursive Merging**: Identify and merge equivalent nodes (nodes with same function, inputs,
   and dependency contexts) to eliminate redundant computation.

The combination of these optimizations maximizes DAG compaction, which is critical for fair
comparison across different agent-generated programs and measuring task complexity.
"""

# Canonical order for filtering operators (lower number = higher priority, comes first)
# This ordering ensures consistent program structure regardless of original filter arrangement
FILTER_ORDER = {
    "FilterConcept": 0,  # Type filtering (e.g., "person", "book")
    "FilterStr": 1,  # String attribute filtering (e.g., country="France")
    "FilterNum": 2,  # Numeric attribute filtering (e.g., age > 18)
    "FilterYear": 3,  # Year filtering (e.g., year > 2000)
    "FilterDate": 4,  # Date filtering (e.g., date < 2020-01-01)
    "QFilterStr": 5,  # Qualified string filtering (with relation)
    "QFilterNum": 6,  # Qualified numeric filtering (with relation)
    "QFilterYear": 7,  # Qualified year filtering (with relation)
    "QFilterDate": 8,  # Qualified date filtering (with relation)
}


def identify_commutable_sequences(program, parse_tree=None):
    """
    Identify sequences of filtering operations that can be safely reordered.

    A commutable sequence is a chain of filtering operations where each filter depends
    only on the previous filter in the chain (linear dependency). These can be reordered
    without changing the program's semantics because filtering operations commute.

    Example:
        Original: FindAll() -> FilterNum(age>18) -> FilterConcept(person) -> Count()
        Identified chain: [1, 2] (the FilterNum and FilterConcept nodes)

    Args:
        program: List of program steps, where each step is a dict with keys:
                 'function', 'inputs', 'dependencies'
        parse_tree: Unused parameter (kept for backwards compatibility)

    Returns:
        List of chains, where each chain is a list of node indices that form
        a commutable sequence. Only chains with 2+ nodes are returned.

    Note:
        Chains are broken when:
        - A node has multiple dependents (branching)
        - A dependent is not a filtering operation
        - The chain reaches the end of the program
    """
    commutable_groups = []
    visited = set()

    # Scan through all nodes to find chain starting points
    for i, step in enumerate(program):
        if i in visited:
            continue

        # Check if this node is a filtering operation
        if step["function"] in FILTER_ORDER:
            chain = [i]
            visited.add(i)
            current = i

            # Follow the chain forward through single-dependent filtering operations
            while True:
                # Find all nodes that depend on the current node
                dependents = [
                    j for j, s in enumerate(program) if current in s["dependencies"]
                ]

                # Break if current node has no dependents or multiple dependents (branching)
                if len(dependents) != 1:
                    break

                dependent = dependents[0]
                dep_step = program[dependent]

                # Continue chain only if dependent is also a filtering operation
                if dep_step["function"] in FILTER_ORDER:
                    chain.append(dependent)
                    visited.add(dependent)
                    current = dependent
                else:
                    break

            # Only include chains with 2+ filterable operations
            # (single-node chains cannot be reordered)
            if len(chain) >= 2:
                commutable_groups.append(chain)

    return commutable_groups


def generate_reorderings(program, commutable_groups):
    """
    Generate reordered versions of the program by sorting filtering operations
    in each commutable chain according to canonical order.

    This function creates two program variants:
    1. The original program (unchanged)
    2. A reordered version with filters sorted by FILTER_ORDER

    Example:
        Original chain: FilterDate -> FilterNum -> FilterConcept
        Reordered chain: FilterConcept -> FilterNum -> FilterDate

    Args:
        program: List of program steps to reorder
        commutable_groups: List of chains (from identify_commutable_sequences)
                          where each chain is a list of node indices

    Returns:
        List containing [original_program, reordered_program]
        If no commutable groups exist, returns [original_program] only

    Note:
        The reordered version maintains the same node positions but replaces
        nodes within each chain with their canonically sorted versions.
        Dependencies are updated so each node in the chain depends on the
        previous node in canonical order.
    """
    if not commutable_groups:
        return [program]

    # Create a deep copy for the reordered version
    # We copy each field explicitly to avoid shared references
    reordered_program = []
    for step in program:
        reordered_program.append(
            {
                "function": step["function"],
                "inputs": step["inputs"].copy()
                if isinstance(step["inputs"], list)
                else step["inputs"],
                "dependencies": step["dependencies"].copy()
                if isinstance(step["dependencies"], list)
                else step["dependencies"],
            }
        )

    # Process each commutable chain
    for chain in commutable_groups:
        if len(chain) < 2:
            continue

        # Extract the nodes in this chain from the original program
        chain_nodes = [program[idx] for idx in chain]

        # Sort nodes by canonical filter order (FilterConcept first, QFilterDate last)
        # Filters not in FILTER_ORDER get priority 999 (stay at end)
        sorted_nodes = sorted(
            chain_nodes, key=lambda x: FILTER_ORDER.get(x["function"], 999)
        )

        # Get the base dependencies (what the first node in chain depends on)
        # This is preserved for the first node in the reordered chain
        base_deps = program[chain[0]]["dependencies"].copy()

        # Rewrite the chain in canonical sorted order at the same positions
        for i, node in enumerate(sorted_nodes):
            target_position = chain[i]  # Position in the program to update

            # Set dependencies: first in chain uses base_deps, rest depend on previous
            if i == 0:
                deps = base_deps
            else:
                # Each subsequent node depends on the previous node in the chain
                deps = [chain[i - 1]]

            # Replace the node at target_position with the canonically ordered node
            reordered_program[target_position] = {
                "function": node["function"],
                "inputs": node["inputs"].copy()
                if isinstance(node["inputs"], list)
                else node["inputs"],
                "dependencies": deps,
            }

    return [program, reordered_program]


def compute_node_context(program, node_idx, processed_merges=None):
    """
    Compute a recursive context signature for a node based on its function, inputs, and dependencies.

    The context is a tuple that uniquely identifies a node's semantics, including:
    - The node's function name
    - The node's inputs
    - The recursive contexts of all its dependencies (sorted for order-independence)

    Two nodes with identical contexts are semantically equivalent and can be merged.

    Args:
        program: List of program steps
        node_idx: Index of the node to compute context for
        processed_merges: Dict mapping old node indices to their merged representatives
                         (used to follow merge chains during recursive context computation)

    Returns:
        Tuple of (function, inputs_tuple, dependencies_context_tuple)
        where dependencies_context_tuple is a sorted tuple of dependency contexts

    Example:
        Node: FilterConcept(person) <- [FindAll()]
        Context: ('FilterConcept', ('person',), (('FindAll', (), ()),))
    """
    if processed_merges is None:
        processed_merges = {}

    node = program[node_idx]
    dependencies = node["dependencies"]

    # Base case: node has no dependencies
    if not dependencies:
        return (node["function"], tuple(node["inputs"]), tuple())

    # Recursive case: compute contexts for all dependencies
    dep_contexts = []
    for dep in dependencies:
        # Follow merge chain to get actual dependency
        actual_dep = processed_merges.get(dep, dep)
        # Recursively compute dependency context
        dep_context = compute_node_context(program, actual_dep, processed_merges)
        dep_contexts.append(dep_context)

    # Sort dependency contexts for order-independence
    # This ensures that nodes with same dependencies in different orders are considered equivalent
    return (node["function"], tuple(node["inputs"]), tuple(sorted(dep_contexts)))


def are_nodes_equivalent_with_context(program, idx1, idx2, processed_merges=None):
    """
    Check if two nodes are semantically equivalent based on their contexts.

    Two nodes are equivalent if they have:
    - Same function name
    - Same inputs
    - Same dependency contexts (recursively)

    Args:
        program: List of program steps
        idx1: Index of first node
        idx2: Index of second node
        processed_merges: Dict mapping merged node indices to representatives

    Returns:
        True if nodes are equivalent, False otherwise
    """
    context1 = compute_node_context(program, idx1, processed_merges)
    context2 = compute_node_context(program, idx2, processed_merges)
    return context1 == context2


def recursive_dag_conversion(program, max_iterations=10, verbose=True):
    """
    Iteratively merge equivalent nodes in a program to create a compact DAG.

    This function repeatedly identifies groups of equivalent nodes (nodes with identical
    contexts) and merges them by keeping one representative and redirecting all references
    to the merged nodes. The process continues until no more merges are possible or the
    maximum iteration limit is reached.

    Example:
        Before: FindAll() -> FilterConcept(person) -> Count()
                FindAll() -> FilterConcept(person) -> Sum()
        After:  FindAll() -> FilterConcept(person) -> Count()
                                                   -> Sum()

    Args:
        program: List of program steps to compact
        max_iterations: Maximum number of merge iterations (default: 10)
        verbose: Whether to print progress information (default: True)

    Returns:
        Tuple of (compacted_program, final_mapping) where:
        - compacted_program: List of nodes after merging
        - final_mapping: Dict mapping original node indices to final node indices

    Note:
        The merging process is iterative because merging one set of nodes may enable
        new merges to be discovered (e.g., when two previously different nodes become
        equivalent after their dependencies are merged).
    """
    current_program = [step.copy() for step in program]
    all_merges = {}  # Track all merges across iterations: old_idx -> representative_idx
    iteration = 0

    if verbose:
        print("Starting recursive DAG conversion...")

    while iteration < max_iterations:
        if verbose:
            print("\n--- Iteration {}/{} ---".format(iteration + 1, max_iterations))

        # Phase 1: Identify all groups of equivalent nodes
        mergeable_groups = []
        processed = set()

        for i in range(len(current_program)):
            if i in processed:
                continue

            # Start a new group with node i as representative
            group = [i]

            # Find all other nodes equivalent to node i
            for j in range(i + 1, len(current_program)):
                if j not in processed:
                    if are_nodes_equivalent_with_context(current_program, i, j):
                        group.append(j)

            # Only track groups with 2+ nodes (actual merges)
            if len(group) > 1:
                mergeable_groups.append(group)
                processed.update(group)

        # No more merges possible - exit loop
        if not mergeable_groups:
            if verbose:
                print("No more mergeable nodes found.")
            break

        if verbose:
            print("Found {} mergeable groups:".format(len(mergeable_groups)))

        # Phase 2: Record merges (keep first node in each group as representative)
        nodes_to_remove = set()
        for group in mergeable_groups:
            representative = group[0]
            for node_idx in group[1:]:
                nodes_to_remove.add(node_idx)
                all_merges[node_idx] = representative

        # Phase 3: Build new program by removing merged nodes
        old_to_new_idx = {}  # Map indices in current_program to new_program
        new_program = []
        new_idx = 0

        for old_idx in range(len(current_program)):
            if old_idx not in nodes_to_remove:
                old_to_new_idx[old_idx] = new_idx
                new_program.append(current_program[old_idx].copy())
                new_idx += 1

        # Phase 4: Update dependencies to point to new indices and representatives
        for step in new_program:
            new_deps = []

            for dep in step["dependencies"]:
                # If dependency was merged, use its representative
                if dep in nodes_to_remove:
                    representative = all_merges[dep]
                    if representative in old_to_new_idx:
                        new_dep = old_to_new_idx[representative]
                        # Avoid duplicate dependencies
                        if new_dep not in new_deps:
                            new_deps.append(new_dep)
                else:
                    # Dependency still exists, just update its index
                    if dep in old_to_new_idx:
                        new_dep = old_to_new_idx[dep]
                        if new_dep not in new_deps:
                            new_deps.append(new_dep)

            # Sort dependencies for consistency
            step["dependencies"] = sorted(new_deps)

        current_program = new_program
        iteration += 1

    # Build final mapping from original program indices to final program indices
    final_mapping = {}
    for orig_idx in range(len(program)):
        # Follow merge chain to find final representative
        current = orig_idx
        while current in all_merges:
            current = all_merges[current]

        # Find position of representative in final program
        final_pos = None
        program_idx = 0
        for i, orig_step in enumerate(program):
            if i not in all_merges:  # This node wasn't merged away
                if i == current:
                    final_pos = program_idx
                    break
                program_idx += 1

        final_mapping[orig_idx] = final_pos if final_pos is not None else 0

    return current_program, final_mapping


def kopl_dag_conversion(program):
    """
    Comprehensive DAG conversion that combines canonical reordering and recursive merging.

    This is the main entry point for converting KOPL programs to compact DAGs. It applies
    two optimizations in sequence:

    1. **Canonical Reordering**: Reorders filtering operations within commutable chains
       according to FILTER_ORDER (FilterConcept first, QFilterDate last)

    2. **Recursive Merging**: Iteratively merges equivalent nodes to eliminate redundancy

    The function tries both the original and reordered versions, preferring the reordered
    version when both produce equal-sized DAGs (to ensure consistent canonical ordering).

    Example:
        Input:  FindAll() -> FilterNum(age>18) -> FilterConcept(person) -> Count()
                FindAll() -> FilterConcept(person) -> FilterNum(age>18) -> Sum()

        Step 1: Reorder both branches to FilterConcept -> FilterNum
        Step 2: Merge the identical FindAll() and FilterConcept->FilterNum chains

        Output: FindAll() -> FilterConcept(person) -> FilterNum(age>18) -> Count()
                                                                        -> Sum()

    Args:
        program: List of program steps (nodes) to convert to DAG, where each step is
                a dict with keys 'function', 'inputs', 'dependencies'

    Returns:
        Tuple of (dag_program, final_mapping) where:
        - dag_program: Compacted DAG as list of nodes
        - final_mapping: Dict mapping original node indices to final node indices

    Note:
        If both original and reordered versions produce DAGs of the same size, the
        reordered version is preferred to ensure canonical filter ordering. This is
        achieved by trying the reordered version first and using strict inequality (<)
        for accepting better results.
    """
    # Step 1: Identify chains of filtering operations that can be reordered
    commutable = identify_commutable_sequences(program, None)

    # Step 2: Generate program variants (original + canonically reordered)
    reorderings = generate_reorderings(program, commutable)

    # Step 3: Try reordered version first to prefer canonical ordering
    # when compaction results are equal
    programs_to_try = reorderings[::-1]  # Reverse to try reordered first

    best_program = None
    best_mapping = None
    best_size = float("inf")

    # Step 4: Try each program variant and keep the most compact one
    for variant_program in programs_to_try:
        try:
            dag_program, mapping = recursive_dag_conversion(
                variant_program, max_iterations=5, verbose=False
            )
            # Use < (not <=) to only accept strictly better results
            # This ensures we keep the FIRST variant that achieves a given size
            # Since reordered is tried first, we prefer it when sizes are equal
            if len(dag_program) < best_size:
                best_program = dag_program
                best_mapping = mapping
                best_size = len(dag_program)
        except Exception:
            # If conversion fails for any reason, try next variant
            continue

    # Fallback if all attempts failed (should be rare)
    if best_program is None:
        best_program = program
        best_mapping = {i: i for i in range(len(program))}

    return best_program, best_mapping
