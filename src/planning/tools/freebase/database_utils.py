"""Utilities for converting function lists to s-expressions.

This module provides functions to convert a linearized list of function calls
(function_list) into nested s-expressions that can be converted to SPARQL.

Modified from KBQA-o1 codebase.
"""

# Error constants
BAD_EXPRESSION = "@BAD_EXPRESSION"
BAD_SPARQL = "@BAD_SPARQL"


# Function definitions for exec() evaluation
# These mirror the atomic KB query operations


def START(entity: str) -> str:
    """Initialize with an entity from the knowledge base.

    Args:
        entity: Freebase entity ID (e.g., "m.02mjmr")

    Returns:
        Entity ID (used as base for further operations)
    """
    # If entity is a class (not an ID and not a literal), return all instances
    if not (entity.startswith("m.") or entity.startswith("g.") or "^^" in entity):
        return f"(JOIN type.object.type {entity})"
    return entity


def JOIN(relation: str, expression: str) -> str:
    """Follow a relation to get connected entities.

    Args:
        relation: Freebase relation (e.g., "film.actor.film")
        expression: Previous expression to apply relation to

    Returns:
        S-expression: (JOIN relation expression)
    """
    return f"(JOIN {relation} {expression})"


def AND(expression: str, sub_expression: str) -> str:
    """Compute set intersection of two expressions.

    Args:
        expression: First expression
        sub_expression: Second expression

    Returns:
        S-expression: (AND expression sub_expression)
    """
    return f"(AND {expression} {sub_expression})"


def ARG(operator: str, expression: str, relation: str) -> str:
    """Find entity with maximum or minimum property value.

    Args:
        operator: "ARGMAX" or "ARGMIN"
        expression: Expression to apply operator to
        relation: Property to compare (e.g., "film.film.budget")

    Returns:
        S-expression: (ARGMAX/ARGMIN expression relation)
    """
    assert operator in ["ARGMAX", "ARGMIN"], f"Invalid ARG operator: {operator}"
    return f"({operator} {expression} {relation})"


def ARGMAX(expression: str, relation: str) -> str:
    """Find entity with maximum property value.

    Args:
        expression: Expression to apply operator to
        relation: Property to maximize

    Returns:
        S-expression: (ARGMAX expression relation)
    """
    return ARG("ARGMAX", expression, relation)


def ARGMIN(expression: str, relation: str) -> str:
    """Find entity with minimum property value.

    Args:
        expression: Expression to apply operator to
        relation: Property to minimize

    Returns:
        S-expression: (ARGMIN expression relation)
    """
    return ARG("ARGMIN", expression, relation)


def CMP(operator: str, relation: str, expression: str) -> str:
    """Compare property value with a threshold.

    Args:
        operator: Comparison operator ("<", "<=", ">", ">=")
        relation: Property to compare
        expression: Threshold value

    Returns:
        S-expression: (operator relation expression)
    """
    return f"({operator} {relation} {expression})"


def TC(expression: str, relation: str, entity: str) -> str:
    """Apply time constraint (transitive closure with time filtering).

    Args:
        expression: Expression to filter
        relation: Temporal relation
        entity: Time constraint value

    Returns:
        S-expression: (TC expression relation entity)
    """
    return f"(TC {expression} {relation} {entity})"


def COUNT(expression: str) -> str:
    """Count entities in an expression.

    Args:
        expression: Expression to count

    Returns:
        S-expression: (COUNT expression)
    """
    return f"(COUNT {expression})"


def STOP(expression: str) -> str:
    """Finalize the query and return results.

    Args:
        expression: Final expression to return

    Returns:
        The expression unchanged (marks completion)
    """
    return expression


# Global namespace for exec() evaluation
_exec_globals = {
    "START": START,
    "JOIN": JOIN,
    "AND": AND,
    "ARG": ARG,
    "ARGMAX": ARGMAX,
    "ARGMIN": ARGMIN,
    "CMP": CMP,
    "TC": TC,
    "COUNT": COUNT,
    "STOP": STOP,
}


def functions_to_expression(function_list: list[str], target_variable: str) -> str:
    """Convert a list of function calls to a nested s-expression.

    This function evaluates a sequence of Python function calls that build up
    an s-expression incrementally. Each function call assigns to a variable
    (e.g., "expression0 = START('m.02mjmr')"), and the final result is
    retrieved from the target variable.

    Args:
        function_list: List of function call strings, e.g.:
            [
                "expression0 = START('m.02mjmr')",
                "expression1 = JOIN('film.actor.film', expression0)",
                "expression2 = STOP(expression1)"
            ]
        target_variable: Name of variable containing final result (e.g., "expression2")

    Returns:
        Nested s-expression string, e.g.:
            "(JOIN film.actor.film m.02mjmr)"
        Or BAD_EXPRESSION if evaluation fails

    Example:
        >>> func_list = [
        ...     "expression0 = START('m.02mjmr')",
        ...     "expression1 = JOIN('film.actor.film', expression0)"
        ... ]
        >>> functions_to_expression(func_list, "expression1")
        "(JOIN film.actor.film m.02mjmr)"

    Note:
        - Function calls are executed using exec() in a controlled namespace
        - STOP() function is typically the last operation
        - Returns BAD_EXPRESSION if any function call fails
    """
    try:
        # Local namespace for variables (expression0, expression1, etc.)
        local_namespace = {}

        # Execute each function call in sequence
        # This builds up the s-expression incrementally
        function_code = "\n".join(function_list)
        exec(function_code, _exec_globals, local_namespace)

        # Return the final s-expression from target variable
        return local_namespace[target_variable]

    except Exception:
        # Evaluation failed (syntax error, undefined variable, etc.)
        return BAD_EXPRESSION
