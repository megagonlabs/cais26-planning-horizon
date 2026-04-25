"""Utilities for converting s-expressions to SPARQL queries.

This module converts nested s-expressions into executable SPARQL queries
following GrailQA conventions. It handles:
- Basic JOIN operations (forward and reverse relations)
- Set operations (AND)
- Aggregations (COUNT, ARGMAX, ARGMIN)
- Comparisons (lt, le, gt, ge)
- Time constraints (TC)
- String literal matching with SUBSTR for Freebase compatibility

**Supported Datasets:**
- GrailQA
- WebQSP
- GraphQ (same as GrailQA for SPARQL structure)

**Validation:** See data/atomic_kbqa/scripts/README.md for validation details

Modified from KBQA-o1 codebase (originally based on RNG-KBQA).
Original license: MIT License
"""

from pathlib import Path
from collections import defaultdict
from typing import List, Set

# Load Freebase ontology files
_FREEBASE_DATA_DIR = (
    Path(__file__).parent.parent.parent.parent / "data" / "atomic_kbqa" / "freebase"
)

# Load reverse properties mapping
_reverse_properties = {}
_reverse_props_file = _FREEBASE_DATA_DIR / "reverse_properties"
if _reverse_props_file.exists():
    with open(_reverse_props_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                _reverse_properties[parts[0]] = parts[1]

# Load relation domain/range info
_relation_dr = {}
_relations = set()
_fb_roles_file = _FREEBASE_DATA_DIR / "fb_roles"
if _fb_roles_file.exists():
    with open(_fb_roles_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                domain, relation, range_type = parts[0], parts[1], parts[2]
                _relation_dr[relation] = (domain, range_type)
                _relations.add(relation)

# Load type hierarchy
_upper_types = defaultdict(set)
_types = set()
_fb_types_file = _FREEBASE_DATA_DIR / "fb_types"
if _fb_types_file.exists():
    with open(_fb_types_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                child_type, _, parent_type = parts[0], parts[1], parts[2]
                _upper_types[child_type].add(parent_type)
                _types.add(child_type)
                _types.add(parent_type)


def lisp_to_nested_expression(lisp_string: str) -> List:
    """Parse a lisp-style s-expression string into nested list structure.

    Args:
        lisp_string: S-expression as string, e.g., "(JOIN film.actor.film (START m.02mjmr))"

    Returns:
        Nested list representation, e.g., ['JOIN', 'film.actor.film', ['START', 'm.02mjmr']]

    Example:
        >>> lisp_to_nested_expression("(COUNT (JOIN film.actor.film (START m.02mjmr)))")
        ['COUNT', ['JOIN', 'film.actor.film', ['START', 'm.02mjmr']]]
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()

    for token in tokens:
        # Handle opening parentheses
        while token[0] == "(":
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]

        # Add token (without closing parentheses)
        current_expression.append(token.replace(")", ""))

        # Handle closing parentheses
        while token[-1] == ")":
            current_expression = stack.pop()
            token = token[:-1]

    return current_expression[0]


def _linearize_lisp_expression(
    expression: list, sub_formula_id: List[int]
) -> List[list]:
    """Convert nested s-expression to linearized form with variable assignments.

    This flattens the nested structure by assigning intermediate results to
    variables (#0, #1, etc.).

    Args:
        expression: Nested list representation of s-expression
        sub_formula_id: Counter for variable numbering (modified in place)

    Returns:
        List of linearized sub-expressions with variables

    Example:
        >>> expr = ['JOIN', 'film.actor.film', ['START', 'm.02mjmr']]
        >>> _linearize_lisp_expression(expr, [0])
        [['START', 'm.02mjmr'], ['JOIN', 'film.actor.film', '#0']]
    """
    sub_formulas = []

    # Recursively linearize nested sub-expressions
    for i, element in enumerate(expression):
        if isinstance(element, list) and element[0] != "R":
            # Recursively process nested expression
            sub_formulas.extend(_linearize_lisp_expression(element, sub_formula_id))
            # Replace nested expression with variable reference
            expression[i] = "#" + str(sub_formula_id[0] - 1)

    # Add current expression to linearized list
    sub_formulas.append(expression)
    sub_formula_id[0] += 1

    return sub_formulas


def _get_root_variable(var: int, identical_vars_reverse: dict) -> int:
    """Follow variable aliasing chain to get root variable.

    When variables are merged (e.g., in AND operations), we track which
    variables are identical. This function resolves aliases.

    Args:
        var: Variable ID
        identical_vars_reverse: Mapping of aliased variables

    Returns:
        Root variable ID
    """
    while var in identical_vars_reverse:
        var = identical_vars_reverse[var]
    return var


def lisp_to_sparql(lisp_program: str, dataset_name: str = "grailqa") -> str:
    """Convert s-expression to SPARQL query

    This is the main conversion function that translates a nested s-expression
    into an executable SPARQL query.

    Args:
        lisp_program: S-expression string, e.g.:
            "(JOIN film.actor.film (START m.02mjmr))"
        dataset_name: Dataset name for format-specific handling.
            - "grailqa": Nested subquery for superlatives (returns all matches)
            - "graphq": Nested subquery for superlatives (returns all matches)
            - "webqsp": Flat query with LIMIT 1 (returns single match)
            Default: "grailqa"

    Returns:
        SPARQL query string with PREFIX, SELECT, WHERE clauses

    Example:
        >>> lisp_to_sparql("(JOIN film.actor.film (START m.02mjmr))")
        '''PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?x WHERE {
          ns:m.02mjmr ns:film.actor.film ?x .
          FILTER (?x != ns:m.02mjmr)
          FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))
        }'''

    Note:
        - Handles forward relations: (JOIN rel expr) → "?x ns:rel ?prev"
        - Handles reverse relations: (JOIN (R rel) expr) → "?prev ns:rel ?x"
        - Handles aggregations: COUNT, ARGMAX, ARGMIN
        - Handles set operations: AND
        - Comparisons: lt, le, gt, ge
        - Time constraints: TC
    """
    # Parse s-expression to nested structure
    expression = lisp_to_nested_expression(lisp_program)

    # Initialize tracking variables
    clauses = []  # WHERE clause patterns
    order_clauses = []  # ORDER BY and LIMIT clauses
    entities = set()  # Entities to filter from results
    identical_variables_r = {}  # Variable aliasing (key -> value mapping)
    count = False  # Whether query uses COUNT
    superlative = False  # Track if query has ARGMAX/ARGMIN

    # Determine variable name for superlative argument based on dataset
    arg_var = "sk0" if dataset_name in ["grailqa", "graphq"] else "arg0"

    # Handle superlative queries (ARGMAX/ARGMIN)
    if expression[0] in ["ARGMAX", "ARGMIN"]:
        superlative = True
        # Flatten nested JOIN operations in relation chain
        if isinstance(expression[2], list):
            relations = _retrieve_relations(expression[2])
            expression = expression[:2]
            expression.extend(relations)

    # Linearize nested expression into sequential operations
    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1

    # Process each sub-program
    for i, subp in enumerate(sub_programs):
        i_str = str(i)

        if subp[0] == "JOIN":
            _process_join(subp, i_str, clauses, entities)

        elif subp[0] == "AND":
            _process_and(subp, i, i_str, identical_variables_r, clauses)

        elif subp[0] in ["le", "lt", "ge", "gt"]:
            _process_comparison(subp, i_str, clauses)

        elif subp[0] == "TC":
            _process_time_constraint(subp, i, i_str, identical_variables_r, clauses)

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            _process_superlative(
                subp, i, i_str, identical_variables_r, clauses, order_clauses, arg_var
            )

        elif subp[0] == "COUNT":
            var = int(subp[1][1:])
            root_var = _get_root_variable(var, identical_variables_r)
            identical_variables_r[int(i_str)] = root_var
            count = True

    # Merge identical variables in clauses
    for i in range(len(clauses)):
        for k in identical_variables_r:
            root_var = _get_root_variable(k, identical_variables_r)
            clauses[i] = clauses[i].replace(f"?x{k} ", f"?x{root_var} ")

    # Rename question variable to ?x
    question_var = _get_root_variable(question_var, identical_variables_r)
    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f"?x{question_var} ", "?x ")

    # Collect all variables used in clauses (for filtering)
    import re

    filter_variables = []
    for clause in clauses:
        variables = re.findall(r"\?\w+", clause)
        for var in variables:
            var = var.strip()
            # Exclude answer variable ?x and internal variables ?sk* and ?arg*
            if (
                var not in filter_variables
                and var != "?x"
                and not var.startswith("?sk")
                and not var.startswith("?arg")
            ):
                filter_variables.append(var)

    # Add entity filters
    for entity in entities:
        clauses.append(f"FILTER (?x != ns:{entity})")

    # Add variable filters (prevent ?x from being equal to intermediate variables)
    for var in filter_variables:
        clauses.append(f"FILTER (?x != {var})")

    # Handle string literal comparisons using SUBSTR for partial matching
    # This is needed because Freebase string values may have language tags or variations
    # Only apply this for WebQSP dataset
    if dataset_name == "webqsp":
        sentences = list(clauses)
        num = 0
        for c, sentence in enumerate(sentences):
            parts = sentence.split(" ")
            if len(parts) == 4 and parts[-1] == ".":
                # Check if object is a string literal (quoted)
                if parts[-2].startswith('"') and parts[-2].endswith('"'):
                    name = parts[-2]
                    # Replace literal with variable
                    clauses[c] = clauses[c].replace(name, f"?st{num}")
                    # Add SUBSTR filter for partial string matching
                    clauses.append(
                        f"FILTER (SUBSTR(STR(?st{num}), 1, STRLEN({name})) = {name})"
                    )
                    num += 1

    # Dataset-specific query structure
    if dataset_name in ["grailqa", "graphq"] and superlative:
        # GrailQA/GraphQ: Use nested subquery to return all entities with extreme value
        arg_clauses = clauses[:]

        # Add language filter and WHERE to inner subquery
        arg_clauses.insert(
            0, "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))"
        )
        arg_clauses.insert(0, "WHERE {")
        arg_clauses.insert(0, f"{{SELECT ?{arg_var}")
        arg_clauses.append("}")
        arg_clauses.extend(order_clauses)
        arg_clauses.append("}")

        # Outer query with original clauses
        clauses.insert(0, "WHERE {")
        clauses.extend(arg_clauses)
        clauses.append("}")
        clauses.insert(0, "SELECT DISTINCT ?x")
    else:
        # WebQSP: Flat query structure
        # Add language filter
        clauses.insert(
            0, "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))"
        )
        clauses.insert(0, "WHERE {")

        # Add SELECT clause
        if count:
            clauses.insert(0, "SELECT COUNT DISTINCT ?x")
        else:
            clauses.insert(0, "SELECT DISTINCT ?x")

        # Close WHERE clause
        clauses.append("}")
        clauses.extend(order_clauses)

    # Add PREFIX
    clauses.insert(0, "PREFIX ns: <http://rdf.freebase.com/ns/>")

    return "\n".join(clauses)


def _retrieve_relations(exp: list) -> List:
    """Extract relations from nested JOIN expressions.

    Used for flattening ARGMAX/ARGMIN relation chains.

    Args:
        exp: Nested expression

    Returns:
        List of relations
    """
    rtn = []
    for element in exp:
        if element == "JOIN":
            continue
        elif isinstance(element, str):
            rtn.append(element)
        elif isinstance(element, list) and element[0] == "R":
            rtn.append(element)
        elif isinstance(element, list) and element[0] == "JOIN":
            rtn.extend(_retrieve_relations(element))
    return rtn


def _process_join(subp: list, i: str, clauses: List[str], entities: Set[str]) -> None:
    """Process JOIN operation and add SPARQL pattern to clauses.

    JOIN(relation, target) -> SELECT ?x WHERE { x? relation target }

    Args:
        subp: Sub-program [operator, arg1, arg2, ...]
        i: Variable index string
        clauses: List of SPARQL clause strings (modified in place)
        entities: Set of entity IDs (modified in place)
    """
    relation = subp[1]
    target = subp[2]

    # Handle reverse relations
    if isinstance(relation, list) and relation[0] == "R":
        # JOIN((R relation), target)
        # -> SELECT ?x WHERE { target relation ?x }
        relation_name = relation[1]
        # Reverse: ?x ns:relation source
        if target[0] == "#":  # Variable
            clauses.append(f"?x{target[1:]} ns:{relation_name} ?x{i} .")
        elif "^^" in target or target.startswith('"'):  # Typed or string literal
            literal = _format_literal(target)
            clauses.append(f"{literal} ns:{relation_name} ?x{i} .")
        else:  # Freebase resource (entity ID or class)
            clauses.append(f"ns:{target} ns:{relation_name} ?x{i} .")
            entities.add(target)
    else:
        # Forward
        # JOIN(relation, target)
        # -> SELECT ?x WHERE { ?x relation target }
        if target[0] == "#":  # Variable
            clauses.append(f"?x{i} ns:{relation} ?x{target[1:]} .")
        elif "^^" in target or target.startswith('"'):  # Typed or string literal
            literal = _format_literal(target)
            clauses.append(f"?x{i} ns:{relation} {literal} .")
        else:  # Freebase resource (entity ID or class)
            clauses.append(f"?x{i} ns:{relation} ns:{target} .")
            entities.add(target)


def _process_and(
    subp: list, i: int, i_str: str, identical_vars_r: dict, clauses: List[str]
) -> None:
    """Process AND operation (set intersection).

    Args:
        subp: Sub-program [operator, arg1, arg2]
        i: Variable index
        i_str: Variable index as string
        identical_vars_r: Variable aliasing mapping (modified in place)
        clauses: List of SPARQL clause strings (modified in place)
    """
    # Track which variable to use as the root
    rooti = _get_root_variable(int(i_str), identical_vars_r)
    final_root = rooti

    # Process both arguments - each can be either a variable or a class constraint
    for arg in [subp[1], subp[2]]:
        if arg[0] == "#":
            # Variable: merge with current variable
            var = int(arg[1:])
            root_var = _get_root_variable(var, identical_vars_r)

            # Merge variables - use the smaller root as the final root
            if final_root != root_var:
                if final_root > root_var:
                    identical_vars_r[final_root] = root_var
                    final_root = root_var
                else:
                    identical_vars_r[root_var] = final_root
        else:
            # Class constraint: add type constraint
            clauses.append(f"?x{i_str} ns:type.object.type ns:{arg} .")


def _process_comparison(subp: list, i: str, clauses: List[str]) -> None:
    """Process comparison operation (lt, le, gt, ge).

    Args:
        subp: Sub-program [operator, relation, value]
        i: Variable index string
        clauses: List of SPARQL clause strings (modified in place)
    """
    operator_map = {"le": "<=", "lt": "<", "ge": ">=", "gt": ">"}
    op = operator_map[subp[0]]

    clauses.append(f"?x{i} ns:{subp[1]} ?y{i} .")

    literal = _format_literal(subp[2])
    clauses.append(f"FILTER (?y{i} {op} {literal})")


def _process_time_constraint(
    subp: list, i: int, i_str: str, identical_vars_r: dict, clauses: List[str]
) -> None:
    """Process time constraint (TC operation).

    Args:
        subp: Sub-program [operator, expression, relation, time_value]
        i: Variable index
        i_str: Variable index as string
        identical_vars_r: Variable aliasing mapping (modified in place)
        clauses: List of SPARQL clause strings (modified in place)
    """
    var = int(subp[1][1:])
    rooti = _get_root_variable(int(i_str), identical_vars_r)
    root_var = _get_root_variable(var, identical_vars_r)

    if rooti > root_var:
        identical_vars_r[rooti] = root_var
    else:
        identical_vars_r[root_var] = rooti

    year = subp[3]
    if year == "NOW":
        from_para = '"2015-08-10"^^xsd:dateTime'
        to_para = '"2015-08-10"^^xsd:dateTime'
    else:
        from_para = f'"{year}-12-31"^^xsd:dateTime'
        to_para = f'"{year}-01-01"^^xsd:dateTime'

    # Add time constraint filters
    clauses.append(f"FILTER(NOT EXISTS {{?x{i_str} ns:{subp[2]} ?sk0}} || ")
    clauses.append(f"EXISTS {{?x{i_str} ns:{subp[2]} ?sk1 . ")
    clauses.append(f"FILTER(xsd:datetime(?sk1) <= {from_para}) }})")

    # Add "to" date constraint
    if subp[2].endswith("from"):
        to_relation = subp[2][:-4] + "to"
    else:
        to_relation = subp[2][:-9] + "to_date"

    clauses.append(f"FILTER(NOT EXISTS {{?x{i_str} ns:{to_relation} ?sk2}} || ")
    clauses.append(f"EXISTS {{?x{i_str} ns:{to_relation} ?sk3 . ")
    clauses.append(f"FILTER(xsd:datetime(?sk3) >= {to_para}) }})")


def _process_superlative(
    subp: list,
    i: int,
    i_str: str,
    identical_vars_r: dict,
    clauses: List[str],
    order_clauses: List[str],
    arg_var: str = "sk0",
) -> None:
    """Process ARGMAX/ARGMIN operation.

    Args:
        subp: Sub-program [operator, expression, relation1, relation2, ...]
        i: Variable index
        i_str: Variable index as string
        identical_vars_r: Variable aliasing mapping (modified in place)
        clauses: List of SPARQL clause strings (modified in place)
        order_clauses: List of ORDER BY clauses (modified in place)
        arg_var: Variable name for argument (default: "sk0" for GrailQA, "arg0" for WebQSP)
    """
    # Handle first argument (expression or class)
    if subp[1][0] == "#":
        var = int(subp[1][1:])
        rooti = _get_root_variable(int(i_str), identical_vars_r)
        root_var = _get_root_variable(var, identical_vars_r)
        if rooti > root_var:
            identical_vars_r[rooti] = root_var
        else:
            identical_vars_r[root_var] = rooti
    else:  # Class
        clauses.append(f"?x{i_str} ns:type.object.type ns:{subp[1]} .")

    # Handle relation chain
    if len(subp) == 3:
        # Single relation
        clauses.append(f"?x{i_str} ns:{subp[2]} ?{arg_var} .")
    elif len(subp) > 3:
        j = 0
        # Multi-hop relation chain
        for j, relation in enumerate(subp[2:-1]):
            var0 = f"x{i_str}" if j == 0 else f"c{j - 1}"
            var1 = f"c{j}"

            if isinstance(relation, list) and relation[0] == "R":
                clauses.append(f"?{var1} ns:{relation[1]} ?{var0} .")
            else:
                clauses.append(f"?{var0} ns:{relation} ?{var1} .")

        clauses.append(f"?c{j} ns:{subp[-1]} ?{arg_var} .")

    # Add ORDER BY clause
    if subp[0] == "ARGMIN":
        order_clauses.append(f"ORDER BY ?{arg_var}")
    elif subp[0] == "ARGMAX":
        order_clauses.append(f"ORDER BY DESC(?{arg_var})")

    order_clauses.append("LIMIT 1")


def _format_literal(value: str) -> str:
    """Format literal values for SPARQL.

    Args:
        value: Literal value, possibly with datatype suffix

    Returns:
        Formatted literal string
    """
    if "^^" in value:
        data_type = value.split("^^")[1].split("#")[1]
        base_value = value.split("^^")[0]

        if data_type not in ["integer", "float", "dateTime"]:
            # Add timezone offset for non-standard types
            return f'"{base_value}-08:00"^^<{value.split("^^")[1]}>'
        else:
            return f'"{base_value}"^^<{value.split("^^")[1]}>'
    else:
        return value
