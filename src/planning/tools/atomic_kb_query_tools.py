"""
Tool wrapper for Atomic KB Query operations (KBQA-o1 style).

This module provides tools for compositional knowledge base querying
where each atomic operation builds upon previous results to construct
complex SPARQL queries against Freebase.

Architecture:
- Factory pattern with shared resources (SPARQL executor, converters)
- Tools are stateless functions wrapped in FunctionTool
- Worker agents perform schema grounding before calling tools
- Tools receive hidden context (function_list, dataset_config) via **kwargs
- Each tool appends to function_list and optionally executes for validation

Operations:
- ExtractEntity: Initialize with entity (START operation)
- FindRelation: Follow relation to connected entities (JOIN operation)
- Merge: Compute set intersection (AND operation)
- Order: Find entity with max/min property (ARGMAX/ARGMIN operation)
- Compare: Filter by comparison (lt, le, gt, ge operations)
- TimeConstraint: Apply temporal filtering (TC operation)
- Count: Count entities in result set (COUNT operation)
"""

from typing import Annotated, Any, Literal, Optional

from .base_tools import FunctionTool, ParamMetadata, Tool, ToolSetFactory
from .freebase.default_config import FREEBASE_ODBC_PORT, FREEBASE_SPARQL_ENDPOINT_URL

ENTITY_TYPE_QUERY_TEMPLATE = """
PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT DISTINCT ?x WHERE {{
    ?x ns:type.object.type ns:{class_id} .
    FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))
}}""".strip()


class AtomicKBQueryToolFactory(ToolSetFactory):
    """
    Factory class for creating atomic KB query tools.

    This class manages shared resources across all tools:
    - SPARQL executor connection (via ODBC to Virtuoso)
    - S-expression to SPARQL converter
    - Function list to s-expression converter

    Tools created by this factory:
    - extract_entity: Initialize with entity (START)
    - find_relation: Follow relation (JOIN)
    - merge: Set intersection (AND)
    - order: Find max/min (ARGMAX/ARGMIN)
    - compare: Comparison filter (lt, le, gt, ge)
    - time_constraint: Temporal filter (TC)
    - count: Count results (COUNT)
    """

    def __init__(self):
        """Initialize factory (resources loaded lazily on first tool creation)."""
        self.virtuoso_url = None
        self.virtuoso_port = None
        self._initialized = False

    def _initialize_resources(
        self,
        virtuoso_url: str = FREEBASE_SPARQL_ENDPOINT_URL,
        virtuoso_port: int = FREEBASE_ODBC_PORT,
    ):
        """
        Initialize shared resources (SPARQL executor connection).

        This is called lazily when create_all_tools() is invoked.

        Args:
            virtuoso_url: URL for Virtuoso SPARQL endpoint
            virtuoso_port: ODBC port for Virtuoso connection

        Note:
            SPARQL executor is initialized via module in sparql_executor.py
            using global connection that gets initialized on first use.
        """
        from .freebase.sparql_executor import configure_odbc_connection

        # Configure ODBC connection settings
        # Each thread will create its own connection lazily on first use
        configure_odbc_connection(port=str(virtuoso_port))

        self.virtuoso_url = virtuoso_url
        self.virtuoso_port = virtuoso_port
        self._initialized = True

    def _execute_function_list(self, function_list: list[str], dataset_name: str = "grailqa") -> dict[str, Any]:
        """
        Execute a function list by converting to s-expression and then SPARQL.

        Args:
            function_list: List of function calls (e.g., ["expression0 = START('m.02mjmr')", ...])
            dataset_name: Dataset name for s-expression conversion ("grailqa", "webqsp", or "graphq")

        Returns:
            Dict with:
                - success: bool
                - s_expression: str (nested s-expression or error)
                - sparql: str (SPARQL query or error)
                - results: set[str] (query results or empty set)
                - error: str (error message if failed)

        Note:
            This provides feedback to agents about whether their query is valid
            and can be executed against the KB.
        """
        from .freebase.database_utils import (
            functions_to_expression,
            BAD_EXPRESSION,
        )
        from .freebase.logic_form_utils import lisp_to_sparql
        from .freebase.sparql_executor import execute_query_with_odbc

        result = {
            "success": False,
            "s_expression": "",
            "sparql": "",
            "results": set(),
            "error": "",
        }

        # Step 1: Convert function list to s-expression
        if not function_list:
            result["error"] = "Empty function list"
            return result

        # Find target variable (last assignment)
        target_variable = None
        for func_call in reversed(function_list):
            if "=" in func_call:
                target_variable = func_call.split("=")[0].strip()
                break

        if target_variable is None:
            result["error"] = "No variable assignments in function list"
            return result

        s_expression = functions_to_expression(function_list, target_variable)

        if s_expression == BAD_EXPRESSION:
            result["error"] = "Failed to convert function list to s-expression"
            return result

        result["s_expression"] = s_expression

        # Step 2: Convert s-expression to SPARQL
        try:
            sparql_query = lisp_to_sparql(s_expression, dataset_name=dataset_name)
            result["sparql"] = sparql_query
        except Exception as e:
            result["error"] = f"Failed to convert s-expression to SPARQL: {str(e)}"
            return result

        # Step 3: Execute SPARQL query
        try:
            query_results = execute_query_with_odbc(sparql_query)
            result["results"] = query_results
            result["success"] = True
        except Exception as e:
            result["error"] = f"Failed to execute SPARQL query: {str(e)}"
            return result

        return result

    def create_all_tools(self, **shared_params: Any) -> dict[str, Tool]:
        """
        Create all atomic KB query tools.

        Args:
            **shared_params: Shared parameters:
                - virtuoso_url: str (default: "http://localhost:13002/sparql")
                - virtuoso_port: int (default: 13002)

        Returns:
            Dict mapping tool_id to Tool instance:
                - "atomic_kb_query/extract_entity"
                - "atomic_kb_query/find_relation"
                - "atomic_kb_query/merge"
                - "atomic_kb_query/order"
                - "atomic_kb_query/compare"
                - "atomic_kb_query/time_constraint"
                - "atomic_kb_query/count"
        """
        # Initialize shared resources if not already done
        if not self._initialized:
            virtuoso_url = shared_params.get("virtuoso_url", FREEBASE_SPARQL_ENDPOINT_URL)
            virtuoso_port = shared_params.get("virtuoso_port", FREEBASE_ODBC_PORT)
            self._initialize_resources(virtuoso_url, virtuoso_port)

        # Create all tools
        tools = {
            "atomic_kb_query/extract_entity": self._create_extract_entity_tool(),
            "atomic_kb_query/find_relation": self._create_find_relation_tool(),
            "atomic_kb_query/merge": self._create_merge_tool(),
            "atomic_kb_query/order": self._create_order_tool(),
            "atomic_kb_query/compare": self._create_compare_tool(),
            "atomic_kb_query/time_constraint": self._create_time_constraint_tool(),
            "atomic_kb_query/count": self._create_count_tool(),
        }

        return tools

    def _create_extract_entity_tool(self) -> FunctionTool:
        """
        Create tool for entity extraction (START operation).

        This tool initializes a query with a specific entity from the KB.
        The worker agent performs entity linking in preprocessing and passes
        the resolved entity_id here.

        Example:
            Input: entity_id="m.02mjmr", entity_name="Barack Obama"
            Function: expression0 = START('m.02mjmr')
            Output: {entity_id: "m.02mjmr", entity_name: "Barack Obama"}
        """

        def extract_entity(
            input_value: Annotated[
                str,
                ParamMetadata(
                    description="Entity mention, class name, or literal value",
                ),
            ],
            input_id: Annotated[
                str,
                ParamMetadata(
                    description="Freebase entity ID, entity class, or typed literal",
                    visible=False,
                ),
            ],
            step_index: Annotated[
                int,
                ParamMetadata(
                    description="Current step index in the overall plan (hidden from LLM)",
                    visible=False,
                ),
            ],
            function_list: Optional[
                Annotated[
                    list[str],
                    ParamMetadata(
                        description="Current function list (hidden from LLM)",
                        visible=False,
                    ),
                ]
            ] = None,
            dataset_config: Annotated[
                dict,
                ParamMetadata(
                    description="Dataset configuration (hidden from LLM)",
                    visible=False,
                ),
            ] = {},
        ) -> dict[str, Any]:
            """
            Initialize query with an entity from the knowledge base.

            Args:
                input_value: Entity mention, class name, or literal value
                input_id: Freebase entity ID, entity class, or typed literal
                function_list: Current accumulated function list (hidden)
                dataset_config: Dataset-specific configuration (hidden)

            Returns:
                Dict with entity information and updated function_list
            """
            if function_list is None:
                function_list = []

            # Determine next expression variable
            if step_index is None:
                # Programming error
                raise ValueError("step_index must be provided for extract_entity tool")
            expression_id = f"expression{step_index}"

            # Build function string
            function_str = f"{expression_id} = START('{input_id}')"
            function_list.append(function_str)

            results = []
            if input_id.startswith("m.") or input_id.startswith("g."):
                # an entity ID - return directly
                results = [input_id]
            elif "^^^" in input_id:
                # typed literal - return directly
                results = [input_value]
            else:
                # an entity class - retrieve all instances
                from .freebase.sparql_executor import execute_query_with_odbc

                sparql_query = ENTITY_TYPE_QUERY_TEMPLATE.format(class_id=input_id)
                try:
                    result_set = execute_query_with_odbc(sparql_query)
                    # Clean results (remove URI prefix)
                    results = [val.replace("http://rdf.freebase.com/ns/", "") for val in result_set]
                    results = sorted(list(set(results)))  # Deduplicate and sort
                except Exception:
                    # Query failed - return empty results
                    results = []

            return {
                "input_value": input_value,
                "input_id": input_id,
                "expression_id": expression_id,
                "results": results,
                "function_list": function_list,
            }

        from .freebase.sparql_executor import close_thread_connection

        return FunctionTool(
            extract_entity,
            name="extract_entity",
            description="Resolve an input (entity mention, entity class, or literal) to Freebase entity IDs or a typed literal.",
            cleanup_func=close_thread_connection,
        )

    def _create_find_relation_tool(self) -> FunctionTool:
        """
        Create tool for finding relations (JOIN operation).

        This tool follows a relation from the current expression to discover
        connected entities. The worker agent performs relation grounding in
        preprocessing and passes the resolved relation_id here.
        """

        def find_relation(
            relation: Annotated[
                str,
                ParamMetadata(
                    description="Freebase relation path (e.g., 'film.actor.film' to connect an actor (input) to their associated films)",
                ),
            ],
            direction: Annotated[
                Literal["forward", "backward"],
                ParamMetadata(
                    description="Direction to follow the relation: 'forward' (?x -R-> target) or 'backward' (target -R-> ?x)",
                ),
            ],
            target_ref: Annotated[
                str,
                ParamMetadata(
                    description="Reference to previous expression variable (e.g., '$0' for the output of step 0)",
                    pattern=r"^\$\d+$",
                ),
            ],
            step_index: Annotated[
                int,
                ParamMetadata(
                    description="Current step index in the overall plan (hidden from LLM)",
                    visible=False,
                ),
            ],
            function_list: Optional[
                Annotated[
                    list[str],
                    ParamMetadata(
                        description="Current function list (hidden from LLM)",
                        visible=False,
                    ),
                ]
            ] = None,
            dataset_config: Annotated[
                dict,
                ParamMetadata(
                    description="Dataset configuration (hidden from LLM)",
                    visible=False,
                ),
            ] = {},
        ) -> dict[str, Any]:
            """
            Find entities that point to the given input entities via a Freebase relation.

            Given a relation R and a reference to previous step output (target_ref),
            returns entities x such that x -R-> target. Results are deduplicated
            and sorted; may be an empty list.

            Args:
                relation: Freebase relation
                target: Reference to previous expression variable
                function_list: Current accumulated function list (hidden)
                dataset_config: Dataset-specific configuration (hidden)

            Returns:
                Dict with relation information and updated function_list
            """
            if function_list is None:
                function_list = []

            # Determine next expression variable
            if step_index is None:
                # Programming error
                raise ValueError("step_index must be provided for order tool")
            expression_id = f"expression{step_index}"

            # By default, JOIN(relation, target) is converted to
            # SELECT ?x WHERE { ?x relation target }
            if direction == "backward":
                # JOIN((R relation), target) will be converted to
                # SELECT ?x WHERE { target relation ?x }
                relation = f"(R {relation})"

            # Build function string
            target_expression = target_ref.replace("$", "expression")
            function_str = f"{expression_id} = JOIN('{relation}', {target_expression})"
            function_list.append(function_str)

            # Execute the current function list
            dataset_name = dataset_config.get("name", "grailqa") if dataset_config else "grailqa"
            execution_result = self._execute_function_list(function_list, dataset_name=dataset_name)

            results = [val.replace("http://rdf.freebase.com/ns/", "") for val in execution_result.get("results", [])]
            results = sorted(list(set(results)))  # Deduplicate and sort results

            return {
                "relation": relation,
                "direction": direction,
                "target": target_ref,
                "expression_id": expression_id,
                "results": results,
                "function_list": function_list,
            }

        from .freebase.sparql_executor import close_thread_connection

        return FunctionTool(
            find_relation,
            name="find_relation",
            description="Find entities that point to the given target entities (target_ref) via the specified Freebase relation (?x -relation-> target_ref).",
            cleanup_func=close_thread_connection,
        )

    def _create_merge_tool(self) -> FunctionTool:
        """
        Create tool for merging expressions (AND operation).

        This tool computes the set intersection of two expressions.

        Example:
            Input: input_ref1="$0", input_ref2="$1"
            Function: expression2 = AND(expression0, expression1)
            Output: {input_ref1: "$0", input_ref2: "$1", expression_id: "expression2"}
        """

        def merge(
            input_ref1: Annotated[
                str,
                ParamMetadata(
                    description="Reference to first input entity set (e.g., '$0' for the output of step 0)",
                    pattern=r"^\$\d+$",
                ),
            ],
            input_ref2: Annotated[
                str,
                ParamMetadata(
                    description="Reference to second input entity set (e.g., '$1' for the output of step 1)",
                    pattern=r"^\$\d+$",
                ),
            ],
            step_index: Annotated[
                int,
                ParamMetadata(
                    description="Current step index in the overall plan (hidden from LLM)",
                    visible=False,
                ),
            ],
            function_list: Optional[
                Annotated[
                    list[str],
                    ParamMetadata(
                        description="Current function list (hidden from LLM)",
                        visible=False,
                    ),
                ]
            ] = None,
            dataset_config: Annotated[
                dict,
                ParamMetadata(
                    description="Dataset configuration (hidden from LLM)",
                    visible=False,
                ),
            ] = {},
        ) -> dict[str, Any]:
            """
            Compute set intersection of two expressions.

            Args:
                input_ref1: Reference to first expression variable
                input_ref2: Reference to second expression variable
                function_list: Current accumulated function list (hidden)
                dataset_config: Dataset-specific configuration (hidden)

            Returns:
                Dict with merge information and updated function_list
            """
            if function_list is None:
                function_list = []

            # Determine next expression variable
            if step_index is None:
                # Programming error
                raise ValueError("step_index must be provided for order tool")
            expression_id = f"expression{step_index}"

            # Build function string
            expression1 = input_ref1.replace("$", "expression")
            expression2 = input_ref2.replace("$", "expression")
            function_str = f"{expression_id} = AND({expression1}, {expression2})"
            function_list.append(function_str)

            # Execute the current function list
            dataset_name = dataset_config.get("name", "grailqa") if dataset_config else "grailqa"
            execution_result = self._execute_function_list(function_list, dataset_name=dataset_name)

            results = [val.replace("http://rdf.freebase.com/ns/", "") for val in execution_result.get("results", [])]
            results = sorted(list(set(results)))  # Deduplicate and sort results

            return {
                "input_ref1": input_ref1,
                "input_ref2": input_ref2,
                "expression_id": expression_id,
                "results": results,
                "function_list": function_list,
            }

        from .freebase.sparql_executor import close_thread_connection

        return FunctionTool(
            merge,
            name="merge",
            description="Compute set intersection of two entity sets",
            cleanup_func=close_thread_connection,
        )

    def _create_order_tool(self) -> FunctionTool:
        """
        Create tool for ordering (ARGMAX/ARGMIN operation).

        This tool finds the entity with maximum or minimum value for a property.

        Example:
            Input: mode="ARGMAX", literal_relation="film.film.budget", expression_ref="expression0"
            Function: expression1 = ARG('ARGMAX', expression0, 'film.film.budget')
            Output: {mode: "ARGMAX", relation: "film.film.budget", expression_id: "expression1"}
        """

        def order(
            mode: Annotated[
                Literal["ARGMAX", "ARGMIN"],
                ParamMetadata(
                    description="Ordering mode: ARGMAX (maximum) or ARGMIN (minimum)",
                ),
            ],
            input_ref: Annotated[
                str,
                ParamMetadata(
                    description="Reference to input entity set (e.g., '$0' for the output of step 0)",
                    pattern=r"^\$\d+$",
                ),
            ],
            property_relation: Annotated[
                str,
                ParamMetadata(
                    description="Numeric/literal relation to order by (e.g., 'film.film.budget')",
                ),
            ],
            step_index: Annotated[
                int,
                ParamMetadata(
                    description="Current step index in the overall plan (hidden from LLM)",
                    visible=False,
                ),
            ],
            function_list: Optional[
                Annotated[
                    list[str],
                    ParamMetadata(
                        description="Current function list (hidden from LLM)",
                        visible=False,
                    ),
                ]
            ] = None,
            dataset_config: Annotated[
                dict,
                ParamMetadata(
                    description="Dataset configuration (hidden from LLM)",
                    visible=False,
                ),
            ] = {},
        ) -> dict[str, Any]:
            """
            Find entity with maximum or minimum property value.

            Args:
                mode: "ARGMAX" or "ARGMIN"
                literal_relation: Property to compare
                expression_ref: Reference to previous expression variable
                function_list: Current accumulated function list (hidden)
                dataset_config: Dataset-specific configuration (hidden)

            Returns:
                Dict with order information and updated function_list
            """
            if function_list is None:
                function_list = []

            # Determine next expression variable
            if step_index is None:
                # Programming error
                raise ValueError("step_index must be provided for order tool")
            expression_id = f"expression{step_index}"

            expression_ref = input_ref.replace("$", "expression")

            # Build function string
            function_str = f"{expression_id} = ARG('{mode}', {expression_ref}, '{property_relation}')"
            function_list.append(function_str)

            # Execute the current function list
            dataset_name = dataset_config.get("name", "grailqa")
            execution_result = self._execute_function_list(function_list, dataset_name=dataset_name)

            results = [val.replace("http://rdf.freebase.com/ns/", "") for val in execution_result.get("results", [])]
            results = sorted(list(set(results)))  # Deduplicate and sort results

            return {
                "mode": mode,
                "input_ref": input_ref,
                "property_relation": property_relation,
                "expression_id": expression_id,
                "results": results,
                "function_list": function_list,
            }

        from .freebase.sparql_executor import close_thread_connection

        return FunctionTool(
            order,
            name="order",
            description="Find a set of entities with maximum or minimum property value",
            cleanup_func=close_thread_connection,
        )

    def _create_compare_tool(self) -> FunctionTool:
        """
        Create tool for comparison (lt, le, gt, ge operations).

        This tool returns entity IDs whose numeric/literal property satisfies
        the comparison with a literal produced by a previous step.
        """

        def compare(
            operator: Annotated[
                Literal["<", "<=", ">", ">="],
                ParamMetadata(
                    description="Comparison operator: '<', '<=', '>', or '>='",
                ),
            ],
            property_relation: Annotated[
                str,
                ParamMetadata(
                    description="Numeric/literal property to compare (e.g., 'film.film.budget')",
                ),
            ],
            literal_ref: Annotated[
                str,
                ParamMetadata(
                    description="Reference to a previously produced literal (e.g., '$0' for the output of step 0)",
                    pattern=r"^\$\d+$",
                ),
            ],
            step_index: Annotated[
                int,
                ParamMetadata(
                    description="Current step index in the overall plan (hidden from LLM)",
                    visible=False,
                ),
            ],
            function_list: Optional[
                Annotated[
                    list[str],
                    ParamMetadata(
                        description="Current function list (hidden from LLM)",
                        visible=False,
                    ),
                ]
            ] = None,
            dataset_config: Annotated[
                dict,
                ParamMetadata(
                    description="Dataset configuration (hidden from LLM)",
                    visible=False,
                ),
            ] = {},
        ) -> dict[str, Any]:
            """
            Return entities whose property_relation compares (operator) to a
            previously produced literal (literal_ref). Returns deduplicated,
            sorted entity IDs (may be empty).

            Args:
                operator: Comparison operator ("<", "<=", ">", ">=")
                property_relation: Numeric/literal property to compare
                literal_ref: Reference to previous literal variable
                function_list: Current accumulated function list (hidden)
                dataset_config: Dataset-specific configuration (hidden)

            Returns:
                Dict with comparison information and updated function_list
            """
            OP_MAPPING = {
                "<": "lt",
                "<=": "le",
                ">": "gt",
                ">=": "ge",
            }
            if function_list is None:
                function_list = []

            # Determine next expression variable
            if step_index is None:
                # Programming error
                raise ValueError("step_index must be provided for order tool")
            expression_id = f"expression{step_index}"

            # Convert input_id from $n to expressionn ($0 -> expression0)
            _literal_ref = literal_ref.replace("$", "expression")

            # Build function string
            function_str = f"{expression_id} = CMP('{OP_MAPPING[operator]}', '{property_relation}', {_literal_ref})"
            function_list.append(function_str)

            # Execute the current function list
            dataset_name = dataset_config.get("name", "grailqa")
            execution_result = self._execute_function_list(function_list, dataset_name=dataset_name)

            results = [val.replace("http://rdf.freebase.com/ns/", "") for val in execution_result.get("results", [])]
            results = sorted(list(set(results)))  # Deduplicate and sort results

            return {
                "operator": operator,
                "property_relation": property_relation,
                "literal_ref": literal_ref,
                "expression_id": expression_id,
                "results": results,
                "function_list": function_list,
            }

        from .freebase.sparql_executor import close_thread_connection

        return FunctionTool(
            compare,
            name="compare",
            description="Return entities whose property compares to a previously produced literal",
            cleanup_func=close_thread_connection,
        )

    def _create_time_constraint_tool(self) -> FunctionTool:
        """
        Create tool for time constraints (TC operation).

        This tool filters an input entity set by a temporal relation compared
        to a date/time literal (or a reference to a previously produced
        literal).
        """

        def time_constraint(
            input_ref: Annotated[
                str,
                ParamMetadata(
                    description="Reference to input entity set (e.g., '$0' for the output of step 0)",
                    pattern=r"^\$\d+$",
                ),
            ],
            temporal_relation: Annotated[
                str,
                ParamMetadata(
                    description="Temporal relation (e.g., 'people.person.date_of_birth')",
                ),
            ],
            temporal_literal: Annotated[
                str,
                ParamMetadata(
                    description="Temporal value use for filtering: Year (e.g., '1990') or 'NOW' to indicate the current time",
                    pattern=r"^(\d+|NOW)$",
                ),
            ],
            step_index: Annotated[
                int,
                ParamMetadata(
                    description="Current step index in the overall plan (hidden from LLM)",
                    visible=False,
                ),
            ],
            function_list: Optional[
                Annotated[
                    list[str],
                    ParamMetadata(
                        description="Current function list (hidden from LLM)",
                        visible=False,
                    ),
                ]
            ] = None,
            dataset_config: Annotated[
                dict,
                ParamMetadata(
                    description="Dataset configuration (hidden from LLM)",
                    visible=False,
                ),
            ] = {},
        ) -> dict[str, Any]:
            """
            Apply temporal filtering to entities.

            Args:
                input_ref: Reference to previous expression variable
                temporal_relation: Temporal relation
                temporal_literal: year or "NOW" literal
                function_list: Current accumulated function list (hidden)
                dataset_config: Dataset-specific configuration (hidden)

            Returns:
                Dict with time constraint information and updated function_list
            """
            if function_list is None:
                function_list = []

            # Determine next expression variable
            if step_index is None:
                # Programming error
                raise ValueError("step_index must be provided for order tool")
            expression_id = f"expression{step_index}"

            # Build function string
            _input_ref = input_ref.replace("$", "expression")
            function_str = f"{expression_id} = TC({_input_ref}, '{temporal_relation}', '{temporal_literal}')"
            function_list.append(function_str)

            # Execute the current function list
            dataset_name = dataset_config.get("name", "grailqa")
            execution_result = self._execute_function_list(function_list, dataset_name=dataset_name)

            results = [val.replace("http://rdf.freebase.com/ns/", "") for val in execution_result.get("results", [])]
            results = sorted(list(set(results)))  # Deduplicate and sort results

            return {
                "input_ref": input_ref,
                "temporal_relation": temporal_relation,
                "temporal_literal": temporal_literal,
                "expression_id": expression_id,
                "results": results,
                "function_list": function_list,
            }

        from .freebase.sparql_executor import close_thread_connection

        return FunctionTool(
            time_constraint,
            name="time_constraint",
            description="Filter an input entity set by equality of a temporal property to a year, or 'NOW'.",
            cleanup_func=close_thread_connection,
        )

    def _create_count_tool(self) -> FunctionTool:
        """
        Create tool for counting (COUNT operation).

        This tool counts the number of entities in an input entity set.
        """

        def count(
            input_ref: Annotated[
                str,
                ParamMetadata(
                    description="Reference to input entity set (e.g., '$0' for the output of step 0)",
                    pattern=r"^\$\d+$",
                ),
            ],
            step_index: Annotated[
                int,
                ParamMetadata(
                    description="Current step index in the overall plan (hidden from LLM)",
                    visible=False,
                ),
            ],
            function_list: Optional[
                Annotated[
                    list[str],
                    ParamMetadata(
                        description="Current function list (hidden from LLM)",
                        visible=False,
                    ),
                ]
            ] = None,
            dataset_config: Annotated[
                dict,
                ParamMetadata(
                    description="Dataset configuration (hidden from LLM)",
                    visible=False,
                ),
            ] = {},
        ) -> dict[str, Any]:
            """
            Count entities in an expression.

            Args:
                input_ref: Reference to input entity set (e.g., '$0' for the output of step 0)
                function_list: Current accumulated function list (hidden)
                dataset_config: Dataset-specific configuration (hidden)

            Returns:
                Dict with count information and updated function_list
            """
            if function_list is None:
                function_list = []

            # Determine next expression variable
            if step_index is None:
                # Programming error
                raise ValueError("step_index must be provided for order tool")
            expression_id = f"expression{step_index}"

            # Build function string
            _input_ref = input_ref.replace("$", "expression")
            function_str = f"{expression_id} = COUNT({_input_ref})"
            function_list.append(function_str)

            # Execute the current function list
            dataset_name = dataset_config.get("name", "grailqa")
            execution_result = self._execute_function_list(function_list, dataset_name=dataset_name)

            results = list(execution_result.get("results", []))
            if len(results) != 1:
                results = None
            else:
                results = results[0]

            result = {
                "input_ref": input_ref,
                "expression_id": expression_id,
                "results": results,
                "function_list": function_list,
            }

            return result

        from .freebase.sparql_executor import close_thread_connection

        return FunctionTool(
            count,
            name="count",
            description="Count the number of entities in an expression",
            cleanup_func=close_thread_connection,
        )
