"""
Converts Atomic KBQA function lists into DAGs.

This module converts linearized function lists from Atomic KBQA dataset (atomic KBQA operations)
into directed acyclic graph (DAG) representations. The conversion enables:
1. Task complexity analysis based on DAG structure
2. Fair comparison across different agent-generated trajectories
3. Deduplication of semantically equivalent programs

The function list format consists of variable assignments like:
    expression = START('m.0example')
    expression = JOIN('relation.name', expression)
    expression1 = START('type.name')
    expression = AND(expression1, expression)
    expression = STOP(expression)

Each line is parsed to extract:
- Target variable (e.g., 'expression')
- Function name (e.g., 'JOIN')
- Input parameters (e.g., 'relation.name')
- Dependency variables (e.g., 'expression')
"""

from typing import Any
import re


def parse_function_string(func_str: str) -> dict[str, Any]:
    """
    Parse a single function string from Atomic KBQA function_list.

    Args:
        func_str: String like "expression = JOIN('relation', expression)"

    Returns:
        Dict with keys:
        - target_var: Variable being assigned (e.g., 'expression')
        - function: Function name (e.g., 'JOIN')
        - inputs: List of string inputs (e.g., ['relation'])
        - dep_vars: List of variable dependencies (e.g., ['expression'])

    Example:
        >>> parse_function_string("expression = JOIN('film.actor.film', expression)")
        {
            'target_var': 'expression',
            'function': 'JOIN',
            'inputs': ['film.actor.film', 'expression'],
            'dep_vars': ['expression']
        }
    """
    # Pattern: variable_name = FUNCTION(args...)
    match = re.match(r"(\w+)\s*=\s*(\w+)\((.*)\)", func_str.strip())
    if not match:
        raise ValueError(f"Invalid function string format: {func_str}")

    target_var = match.group(1)
    function = match.group(2)
    args_str = match.group(3)

    # Parse arguments (mix of string literals and variable references)
    inputs = []
    dep_vars = []

    if args_str.strip():
        # Split by comma, handling nested quotes
        args = []
        current_arg = ""
        in_quotes = False
        paren_depth = 0

        for char in args_str:
            if char == "'" and (not current_arg or current_arg[-1] != "\\"):
                in_quotes = not in_quotes
            elif char == "(" and not in_quotes:
                paren_depth += 1
            elif char == ")" and not in_quotes:
                paren_depth -= 1
            elif char == "," and not in_quotes and paren_depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
                continue
            current_arg += char

        if current_arg.strip():
            args.append(current_arg.strip())

        # Classify each argument as input (string literal) or dependency (variable)
        for arg in args:
            arg = arg.strip()
            if arg.startswith("'") and arg.endswith("'"):
                # String literal input (remove quotes)
                inputs.append(arg[1:-1])
            elif arg.startswith('"') and arg.endswith('"'):
                # String literal input (remove quotes)
                inputs.append(arg[1:-1])
            else:
                # Variable dependency
                inputs.append(arg)
                dep_vars.append(arg)

    return {
        "target_var": target_var,
        "function": function,
        "inputs": inputs,
        "dep_vars": dep_vars,
    }


def function_list_to_dag(
    function_list: list[str], entities: list[tuple[str, str]] = []
) -> list[dict[str, Any]]:
    """
    Convert Atomic KBQA function_list to DAG representation.

    Args:
        function_list: List of function strings like:
            [
                "expression = START('m.0example')",
                "expression = JOIN('relation', expression)",
                ...
            ]
        entities: List of (label, entity_id) tuples for entity resolution

    Returns:
        Compact DAG as list of nodes with format:
            {
                "function": "find_relation",
                "inputs": ["relation", "$0"],
                "dependencies": [0]  # Indices of dependency nodes
            }
    """

    def process_ref_string(val: str, var_to_idx: dict[str, int]) -> str:
        """Helper to process reference strings."""
        if val in var_to_idx:  # if it's a variable reference (e.g., 'expression')
            return f"${var_to_idx[val]}"  # convert to node index reference (e.g., '$0')
        return val  # otherwise, return as is

    # Track variable definitions: var_name -> node_index
    FUNCTION_MAPPING = {
        "START": "extract_entity",
        "JOIN": "find_relation",
        "AND": "merge",
        "ARG": "order",
        "CMP": "compare",
        "TC": "time_constraint",
        "COUNT": "count",
        "STOP": "finish",
    }
    var_to_idx = {}
    dag = []

    eid2label = {ent_id: label for label, ent_id in entities}

    for i, func_str in enumerate(function_list):
        parsed = parse_function_string(func_str)

        # Resolve dependencies to node indices
        dep_indices = []
        for dep_var in parsed["dep_vars"]:
            if dep_var not in var_to_idx:
                raise ValueError(
                    f"Variable '{dep_var}' referenced before definition at step {i}"
                )
            dep_indices.append(var_to_idx[dep_var])

        # Create DAG node
        function = FUNCTION_MAPPING[parsed["function"]]
        ## TODO: process inputs
        inputs = {}
        if function == "extract_entity":
            assert len(parsed["inputs"]) == 1, "START should have one input"
            inputs = {"input_value": eid2label[parsed["inputs"][0]]}
        elif function == "find_relation":
            assert len(parsed["inputs"]) == 2, (
                "JOIN should have two inputs (relation, target_ref)"
            )
            assert len(parsed["dep_vars"]) == 1, "JOIN should have one dependency"
            relation, target_ref = parsed["inputs"]
            direction = "forward"
            if relation.startswith("(R ") and relation.endswith(")"):
                relation = relation[3:-1].strip()
                direction = "backward"
            inputs = {
                "relation": relation,
                "direction": direction,
                "target_ref": process_ref_string(target_ref, var_to_idx),
            }
        elif function == "merge":
            assert len(parsed["inputs"]) == 2, "AND should have two inputs"
            assert len(parsed["dep_vars"]) == 2, "AND should have two dependencies"
            input_ref1, input_ref2 = parsed["inputs"]
            inputs = {
                "input_ref1": process_ref_string(input_ref1, var_to_idx),
                "input_ref2": process_ref_string(input_ref2, var_to_idx),
            }
        elif function == "order":
            assert len(parsed["inputs"]) == 3, (
                "ARG should have three inputs (mode, input_ref, property_relation)"
            )
            assert len(parsed["dep_vars"]) >= 1, (
                "ARG should have at least one dependency"
            )
            mode, input_ref, property_relation = parsed["inputs"]
            if property_relation.startswith("(R ") and property_relation.endswith(")"):
                # This should not happen
                raise ValueError("Property relation in CMP should not be reversed")
            inputs = {
                "mode": mode,
                "input_ref": process_ref_string(input_ref, var_to_idx),
                "property_relation": property_relation,
            }
        elif function == "compare":
            assert len(parsed["inputs"]) == 3, (
                "CMP should have three inputs (operator, property_relation, literal_ref)"
            )
            operator, property_relation, literal_ref = parsed["inputs"]
            if property_relation.startswith("(R ") and property_relation.endswith(")"):
                # This should not happen
                raise ValueError("Property relation in CMP should not be reversed")
            inputs = {
                "operator": operator,
                "property_relation": property_relation,
                "literal_ref": process_ref_string(literal_ref, var_to_idx),
            }
        elif function == "time_constraint":
            assert len(parsed["inputs"]) == 3, (
                "TC should have three inputs (input_ref, temporal_relation, temporal_literal)"
            )
            input_ref, temporal_relation, temporal_literal = parsed["inputs"]
            if temporal_relation.startswith("(R ") and temporal_relation.endswith(")"):
                # This should not happen
                raise ValueError("Temporal relation in TC should not be reversed")
            inputs = {
                "input_ref": process_ref_string(input_ref, var_to_idx),
                "temporal_relation": temporal_relation,
                "temporal_literal": temporal_literal,
            }
        elif function == "count":
            assert len(parsed["inputs"]) == 1, "COUNT should have one input"
            assert len(parsed["dep_vars"]) == 1, "COUNT should have one dependency"
            input_ref = parsed["inputs"][0]
            inputs = {"input_ref": process_ref_string(input_ref, var_to_idx)}
        elif function == "finish":
            assert len(parsed["inputs"]) == 1, "STOP should have one input"
            assert len(parsed["dep_vars"]) == 1, "STOP should have one dependency"
            input_ref = parsed["inputs"][0]
            inputs = {"answer": process_ref_string(input_ref, var_to_idx)}
        else:
            breakpoint()
        node = {
            "function": function,
            "inputs": inputs,
            "dependencies": dep_indices,
        }
        dag.append(node)

        # Register variable definition
        var_to_idx[parsed["target_var"]] = i

    return dag


def validate_dag(dag: list[dict[str, Any]]) -> None:
    """
    Validate DAG structure for correctness.

    Checks:
    1. All steps except the last are referenced by at least one subsequent step
    2. All reference inputs (e.g., '$0', '$1') exist
    3. All steps except 'extract_entity' have at least one dependency

    Args:
        dag: DAG representation as list of nodes

    Raises:
        ValueError: If validation fails with descriptive error message
    """
    if not dag:
        raise ValueError("DAG is empty")

    # Track which nodes are referenced
    referenced_nodes = set()

    for step_idx, node in enumerate(dag):
        function = node.get("function")
        inputs = node.get("inputs", {})
        dependencies = node.get("dependencies", [])

        # Check 3: All steps except 'extract_entity' should have dependencies
        if function != "extract_entity" and not dependencies:
            raise ValueError(
                f"Step {step_idx} (function: {function}) has no dependencies but is not 'extract_entity'"
            )

        # Collect all references from inputs
        for key, value in inputs.items():
            if isinstance(value, str) and value.startswith("$"):
                # Extract node index from reference (e.g., '$0' -> 0)
                try:
                    ref_idx = int(value[1:])
                except ValueError:
                    raise ValueError(
                        f"Step {step_idx}: Invalid reference format '{value}' in input '{key}'"
                    )

                # Check 2: Referenced node must exist
                if ref_idx < 0 or ref_idx >= step_idx:
                    raise ValueError(
                        f"Step {step_idx}: Reference '{value}' in input '{key}' points to non-existent node (valid range: 0-{len(dag)-1})"
                    )

                # Track referenced nodes for check 1
                referenced_nodes.add(ref_idx)

        # Also track dependencies
        for dep_idx in dependencies:
            if dep_idx < 0 or dep_idx >= len(dag):
                raise ValueError(
                    f"Step {step_idx}: Dependency index {dep_idx} is out of range (valid range: 0-{len(dag)-1})"
                )
            referenced_nodes.add(dep_idx)

    # Check 1: All steps except the last should be referenced
    for step_idx in range(len(dag) - 1):
        if step_idx not in referenced_nodes:
            raise ValueError(
                f"Step {step_idx} (function: {dag[step_idx]['function']}) is never referenced by subsequent steps"
            )


def atomic_kbqa_dag_conversion(
    function_list: list[str], entities: list[tuple[str, str]] = []
) -> list[dict[str, Any]]:
    """
    Convert Atomic KBQA function_list to compact DAG with optional node merging.

    This is the main entry point for DAG conversion.
    It converts function_list to DAG format.

    Args:
        function_list: List of function strings from Atomic KBQA dataset

    Returns:
        Compact DAG as list of nodes

    Example:
        >>> function_list = [
        ...     "expression = START('m.0example')",
        ...     "expression = JOIN('film.actor.film', expression)",
        ...     "expression = STOP(expression)"
        ... ]
        >>> entities = [('Example Entity', 'm.0example')]
        >>> dag = atomic_kbqa_dag_conversion(function_list, entities)
        >>> len(dag)
        3
    """
    # Convert to basic DAG
    # Unlike KoPL, there is no recursive merging here
    dag = function_list_to_dag(function_list, entities)

    # Validate DAG structure
    validate_dag(dag)

    return dag
