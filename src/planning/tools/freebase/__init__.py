"""Freebase KB interaction utilities for KBQA tasks.

This module provides utilities for:
- Executing SPARQL queries against Virtuoso-hosted Freebase
- Converting function lists to s-expressions
- Converting s-expressions to SPARQL queries
"""

from planning.tools.freebase.sparql_executor import (
    execute_query_with_odbc,
    get_label_with_odbc,
    configure_odbc_connection,
)
from planning.tools.freebase.database_utils import (
    functions_to_expression,
    BAD_EXPRESSION,
)
from planning.tools.freebase.logic_form_utils import (
    lisp_to_sparql,
    lisp_to_nested_expression,
)

__all__ = [
    "execute_query_with_odbc",
    "get_label_with_odbc",
    "configure_odbc_connection",
    "functions_to_expression",
    "BAD_EXPRESSION",
    "lisp_to_sparql",
    "lisp_to_nested_expression",
]
