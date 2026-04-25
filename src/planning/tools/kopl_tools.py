"""
Tool wrapper for KoPL-related tools.

This module provides tools specifically designed for solving
KBQA problems using the KoPL engine. It implements KoPL operators
as FunctionTool instances that can be used by LLM agents.

KoPL (Knowledge-oriented Programming Language) is designed for complex
reasoning and question answering over knowledge bases.
"""

from pathlib import Path
from typing import Annotated, Any, Literal, TYPE_CHECKING
import threading

# Lazy imports - only import heavy dependencies when actually creating tools
if TYPE_CHECKING:
    from kopl import ValueClass
    from kopl.kopl import EntityTuple
    from kopl.kopl import KoPLEngine

import orjson

from planning.agents.exceptions import AgentException

from .base_tools import FunctionTool, ParamMetadata, Tool, ToolSetFactory

# Module-level cache for KoPL engine instances (kb_path -> KoPLEngine)
_KOPL_ENGINE_CACHE: dict[str, "KoPLEngine"] = {}
_KOPL_ENGINE_CACHE_LOCK = threading.Lock()


class KoPLToolFactory(ToolSetFactory):
    """
    Factory class for creating KoPL tools.

    This class manages a shared KoPL engine instance across all operators
    to optimize memory usage.
    """

    def __init__(self):
        self.kb_path = None
        self.engine = None  # Will be KoPLEngine after initialization

    def _initialize_engine(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ):
        """Initialize the KoPL engine with the knowledge base (using module-level cache)."""
        # Lazy import - only load when actually needed
        from kopl.kopl import KoPLEngine

        kb_path = Path(kb_path)
        kb_key = str(kb_path.resolve())

        # Fast path: check without lock (safe for reads)
        cached_engine = _KOPL_ENGINE_CACHE.get(kb_key)
        if cached_engine is not None:
            self.kb_path = kb_path
            self.engine = cached_engine
            return

        # Lock for thread-safe write
        with _KOPL_ENGINE_CACHE_LOCK:
            # Double-check in case another thread loaded it
            cached_engine = _KOPL_ENGINE_CACHE.get(kb_key)
            if cached_engine is not None:
                self.kb_path = kb_path
                self.engine = cached_engine
                return

            # Not cached -> load and cache
            engine = KoPLEngine.from_json(str(kb_path))

            # Cache and set on instance
            _KOPL_ENGINE_CACHE[kb_key] = engine
            self.kb_path = kb_path
            self.engine = engine

    def _convert_from_value_class(self, obj: Any) -> Any:
        """Recursively convert ValueClass objects to JSON-serializable data."""
        # Lazy import
        from kopl import ValueClass

        if isinstance(obj, ValueClass):
            return {"type": obj.type, "value": obj.value, "unit": obj.unit}
        elif isinstance(obj, list):
            return [self._convert_from_value_class(item) for item in obj]
        elif isinstance(obj, dict):
            return {
                key: self._convert_from_value_class(value) for key, value in obj.items()
            }
        else:
            return obj

    def _convert_to_value_class(self, obj: Any) -> Any:
        """Recursively convert data back to ValueClass objects where applicable."""
        if isinstance(obj, dict):
            # Check if this looks like a serialized ValueClass
            if all(key in obj for key in ["type", "value", "unit"]) and len(obj) == 3:
                return self.engine.kb._parse_value(obj)  # type: ignore
            if all(key in obj for key in ["type", "value"]) and len(obj) == 2:
                _obj = {"type": obj["type"], "value": obj["value"], "unit": None}
                return self.engine.kb._parse_value(_obj)  # type: ignore
            else:
                # Recursively process other dictionaries
                return {
                    key: self._convert_to_value_class(value)
                    for key, value in obj.items()
                }
        elif isinstance(obj, list):
            return [self._convert_to_value_class(item) for item in obj]
        else:
            return obj

    def _serialize_entity_tuple(self, entity_tuple: "EntityTuple") -> str:
        """Serialize an EntityTuple to a JSON string."""
        entities, facts = entity_tuple
        # Convert ValueClass objects recursively to JSON-serializable format
        serializable_facts = self._convert_from_value_class(facts)
        data = {"entities": entities, "facts": serializable_facts}
        return orjson.dumps(data).decode("utf-8")

    def _deserialize_entity_tuple(self, entity_tuple_str: str) -> "EntityTuple":
        """Deserialize a JSON string of entities into a Python list."""
        data = orjson.loads(entity_tuple_str)
        entities = data.get("entities", [])
        facts = data.get("facts", None)
        if not isinstance(entities, list):
            raise ValueError("Entities should be a JSON array.")
        # Convert serialized ValueClass dictionaries back to ValueClass objects
        if facts is not None:
            facts = self._convert_to_value_class(facts)
        entity_tuple: "EntityTuple" = (entities, facts)
        return entity_tuple

    def is_valid_entity_tuple(self, entity_tuple_str: str) -> bool:
        """Check if the input string is a valid serialized EntityTuple."""
        try:
            self._deserialize_entity_tuple(entity_tuple_str)
            return True
        except Exception:
            return False

    def create_all_tools(
        self, kb_path: str = "data/kopl_kbqa/kqa_pro/kb.json", **kwargs
    ) -> dict[str, Tool]:
        """
        Create all KoPL tools with shared KB engine.

        Args:
            kb_path: Path to the knowledge base JSON file
            **kwargs: Additional parameters (ignored)

        Returns:
            Dictionary mapping tool_id to Tool instance
        """
        # Initialize engine with KB
        self._initialize_engine(Path(kb_path))

        # Create all 28 KoPL operator tools
        tools = {}
        operators = [
            "find_all",
            "find",
            "filter_concept",
            "filter_str",
            "filter_num",
            "filter_year",
            "filter_date",
            "qfilter_str",
            "qfilter_num",
            "qfilter_year",
            "qfilter_date",
            "relate",
            "and",
            "or",
            "query_name",
            "count",
            "query_attr",
            "query_attr_under_condition",
            "query_relation",
            "query_attr_qualifier",
            "query_relation_qualifier",
            "select_between",
            "select_among",
            "verify_str",
            "verify_num",
            "verify_year",
            "verify_date",
        ]

        for operator in operators:
            tool_id = f"kopl/{operator}"
            method_name = f"create_{operator}_tool"
            creator_method = getattr(self, method_name)
            tools[tool_id] = creator_method()

        return tools

    # Search and Find Operations
    def create_find_all_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL FindAll tool that returns all entities in the knowledge base."""
        self._initialize_engine(kb_path)

        def kopl_find_all() -> str:
            """Return all entities in the knowledge base."""
            assert self.engine is not None
            result = self.engine.FindAll()
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_find_all,
            name="find_all",
            description="Return all entities in the knowledge base.",
        )

    def create_find_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL Find tool that finds entities by name."""
        self._initialize_engine(kb_path)

        def kopl_find(
            name: Annotated[
                str, ParamMetadata(description="Name of the entity to find")
            ],
        ) -> str:
            """Return all entities with the given name."""
            assert self.engine is not None
            result = self.engine.Find(name)
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_find,
            name="find",
            description="Return all entities with the given name. (Example: find(name='LeBron James'))",
        )

    def create_filter_concept_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL FilterConcept tool for filtering entities by concept."""
        self._initialize_engine(kb_path)

        def kopl_filter_concept(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            concept_name: Annotated[
                str, ParamMetadata(description="Concept to filter by")
            ],
        ) -> str:
            """Find those belonging to the given concept."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                result = self.engine.FilterConcept(entity_tuple, concept_name)
            except KeyError as e:
                raise AgentException(f"Concept not found: {e}") from e
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_filter_concept,
            name="filter_concept",
            description="Find those belonging to the given concept. (Example: filter_concept(entities_and_facts=$1, concept_name='athlete'))",
        )

    # Attribute Filtering Operations
    def create_filter_str_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL FilterStr tool for filtering entities by string attributes."""
        self._initialize_engine(kb_path)

        def kopl_filter_str(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            key: Annotated[
                str, ParamMetadata(description="Attribute key to filter by")
            ],
            value: Annotated[str, ParamMetadata(description="String value to match")],
        ) -> str:
            """Filter entities with an attribute condition of string type, return entities and corresponding facts=(entity, key, value)."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                result = self.engine.FilterStr(entity_tuple, key, value)
            except KeyError as e:
                raise AgentException(f"Attribute key not found: {e}") from e
            except ValueError as e:
                raise AgentException(f"Invalid string value: {e}") from e
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_filter_str,
            name="filter_str",
            description="Filter entities with an attribute condition of string type, return entities and corresponding facts=(entity, key, value). (Example: filter_str(entities_and_facts=$1, key='gender', value='male'))",
        )

    def create_filter_num_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL FilterNum tool for filtering entities by numeric attributes."""
        self._initialize_engine(kb_path)

        def kopl_filter_num(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            key: Annotated[
                str, ParamMetadata(description="Attribute key to filter by")
            ],
            value: Annotated[
                str,
                ParamMetadata(
                    description="Numeric value to compare. Can contain units after a whitespace, e.g., '200 centimetres'"
                ),
            ],
            op: Annotated[
                Literal["=", "!=", "<", ">"],
                ParamMetadata(description="Comparison operator: '=', '!=', '<', '>'"),
            ],
        ) -> str:
            """Filter entities with an attribute condition of numeric type, return entities and corresponding facts=(entity, key, value)."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                result = self.engine.FilterNum(entity_tuple, key, value, op)
            except KeyError as e:
                raise AgentException(f"Attribute key not found: {e}") from e
            except ValueError as e:
                raise AgentException(f"Invalid numeric value: {e}") from e
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_filter_num,
            name="filter_num",
            description="Filter entities with an attribute condition of numeric type, return entities and corresponding facts=(entity, key, value). (Example: filter_num(entities_and_facts=$1, key='height', value='200 centimetres', op='>'))",
        )

    def create_filter_year_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL FilterYear tool for filtering entities by year attributes."""
        self._initialize_engine(kb_path)

        def kopl_filter_year(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            key: Annotated[
                str, ParamMetadata(description="Attribute key to filter by")
            ],
            value: Annotated[
                str,
                ParamMetadata(description="Year value to compare", pattern=r"^\d{4}$"),
            ],
            op: Annotated[
                Literal["=", "!=", "<", ">"],
                ParamMetadata(description="Comparison operator: '=', '!=', '<', '>'"),
            ],
        ) -> str:
            """Filter entities with an attribute condition of year type, return entities and corresponding facts=(entity, key, value)."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                result = self.engine.FilterYear(entity_tuple, key, value, op)
            except KeyError as e:
                raise AgentException(f"Attribute key not found: {e}") from e
            except ValueError as e:
                raise AgentException(f"Invalid year value: {e}") from e
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_filter_year,
            name="filter_year",
            description="Filter entities with an attribute condition of year type, return entities and corresponding facts=(entity, key, value). (Example: filter_year(entities_and_facts=$1, key='birthday', value='1980', op='='))",
        )

    def create_filter_date_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL FilterDate tool for filtering entities by date attributes."""
        self._initialize_engine(kb_path)

        def kopl_filter_date(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            key: Annotated[
                str, ParamMetadata(description="Attribute key to filter by")
            ],
            value: Annotated[
                str,
                ParamMetadata(
                    description="Date value to compare (YYYY-MM-DD)",
                    pattern=r"^\d{4}-\d{2}-\d{2}$",
                ),
            ],
            op: Annotated[
                Literal["=", "!=", "<", ">"],
                ParamMetadata(description="Comparison operator: '=', '!=', '<', '>'"),
            ],
        ) -> str:
            """Filter entities with an attribute condition of date type, return entities and corresponding facts=(entity, key, value)."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                result = self.engine.FilterDate(entity_tuple, key, value, op)
            except KeyError as e:
                raise AgentException(f"Attribute key not found: {e}") from e
            except ValueError as e:
                raise AgentException(f"Invalid date value: {e}") from e
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_filter_date,
            name="filter_date",
            description="Filter entities with an attribute condition of date type, return entities and corresponding facts=(entity, key, value). (Example: filter_date(entities_and_facts=$1, key='birthday', value='1980-06-01', op='<'))",
        )

    # Qualifier Filtering Operations
    def create_qfilter_str_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL QFilterStr tool for filtering by string qualifiers."""
        self._initialize_engine(kb_path)

        def kopl_qfilter_str(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts (the output of filter_str, filter_num, filter_year, filter_date, or relate)",
                    pattern=r"^\$\d+$",
                ),
            ],
            qkey: Annotated[
                str, ParamMetadata(description="Qualifier key to filter by")
            ],
            qvalue: Annotated[
                str, ParamMetadata(description="String qualifier value to match")
            ],
        ) -> str:
            """Filter entities and corresponding facts (=entity, key, value, qualifier_key, qualifier_value) with a qualifier condition of string type."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                result = self.engine.QFilterStr(entity_tuple, qkey, qvalue)
            except KeyError as e:
                raise AgentException(f"Qualifier key not found: {e}") from e
            except ValueError as e:
                raise AgentException(f"Invalid string value: {e}") from e
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_qfilter_str,
            name="qfilter_str",
            description="Filter entities and corresponding facts (=entity, key, value, qualifier_key, qualifier_value) with a qualifier condition of string type. 'entities_and_facts' must be the output of filter_str, filter_num, filter_year, filter_date, or relate. (Example: qfilter_str(entities_and_facts=$1, qkey='language', qvalue='English'))",
        )

    def create_qfilter_num_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL QFilterNum tool for filtering by numeric qualifiers."""
        self._initialize_engine(kb_path)

        def kopl_qfilter_num(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts (the output of filter_str, filter_num, filter_year, filter_date, or relate)",
                    pattern=r"^\$\d+$",
                ),
            ],
            qkey: Annotated[
                str, ParamMetadata(description="Qualifier key to filter by")
            ],
            qvalue: Annotated[
                str,
                ParamMetadata(
                    description="Numeric qualifier value to compare. Can contain units after a whitespace, e.g., '2000 dollars'",
                    pattern=r"^\$\d+$",
                ),
            ],
            op: Annotated[
                Literal["=", "!=", "<", ">"],
                ParamMetadata(description="Comparison operator: '=', '!=', '<', '>'"),
            ],
        ) -> str:
            """Filter entities and corresponding facts (=entity, key, value, qualifier_key, qualifier_value) with a qualifier condition of numeric type."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                result = self.engine.QFilterNum(entity_tuple, qkey, qvalue, op)
            except KeyError as e:
                raise AgentException(f"Qualifier key not found: {e}") from e
            except ValueError as e:
                raise AgentException(f"Invalid numeric qualifier value: {e}") from e
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_qfilter_num,
            name="qfilter_num",
            description="Filter entities and corresponding facts (=entity, key, value, qualifier_key, qualifier_value) with a qualifier condition of numeric type. 'entities_and_facts' must be the output of filter_str, filter_num, filter_year, filter_date, or relate. (Example: qfilter_num(entities_and_facts=$1, qkey='bonus', qvalue='2000 dollars', op='>'))",
        )

    def create_qfilter_year_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL QFilterYear tool for filtering by year qualifiers."""
        self._initialize_engine(kb_path)

        def kopl_qfilter_year(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts (the output of filter_str, filter_num, filter_year, filter_date, or relate)",
                    pattern=r"^\$\d+$",
                ),
            ],
            qkey: Annotated[
                str, ParamMetadata(description="Qualifier key to filter by")
            ],
            qvalue: Annotated[
                str,
                ParamMetadata(
                    description="Year qualifier value to compare", pattern=r"^\d{4}$"
                ),
            ],
            op: Annotated[
                Literal["=", "!=", "<", ">"],
                ParamMetadata(description="Comparison operator: '=', '!=', '<', '>'"),
            ],
        ) -> str:
            """Filter entities and corresponding facts (=entity, key, value, qualifier_key, qualifier_value) with a qualifier condition of year type."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                result = self.engine.QFilterYear(entity_tuple, qkey, qvalue, op)
            except KeyError as e:
                raise AgentException(f"Qualifier key not found: {e}") from e
            except ValueError as e:
                raise AgentException(f"Invalid year qualifier value: {e}") from e
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_qfilter_year,
            name="qfilter_year",
            description="Filter entities and corresponding facts (=entity, key, value, qualifier_key, qualifier_value) with a qualifier condition of year type. 'entities_and_facts' must be the output of filter_str, filter_num, filter_year, filter_date, or relate. (Example: qfilter_year(entities_and_facts=$1, qkey='award_year', qvalue='2020', op='>'))",
        )

    def create_qfilter_date_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL QFilterDate tool for filtering by date qualifiers."""
        self._initialize_engine(kb_path)

        def kopl_qfilter_date(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts (the output of filter_str, filter_num, filter_year, filter_date, or relate)",
                    pattern=r"^\$\d+$",
                ),
            ],
            qkey: Annotated[
                str, ParamMetadata(description="Qualifier key to filter by")
            ],
            qvalue: Annotated[
                str,
                ParamMetadata(
                    description="Date qualifier value to compare (YYYY-MM-DD)",
                    pattern=r"^\d{4}-\d{2}-\d{2}$",
                ),
            ],
            op: Annotated[
                Literal["=", "!=", "<", ">"],
                ParamMetadata(description="Comparison operator: '=', '!=', '<', '>'"),
            ],
        ) -> str:
            """Filter entities and corresponding facts (=entity, key, value, qualifier_key, qualifier_value) with a qualifier condition of date type."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                result = self.engine.QFilterDate(entity_tuple, qkey, qvalue, op)
            except KeyError as e:
                raise AgentException(f"Qualifier key not found: {e}") from e
            except ValueError as e:
                raise AgentException(f"Invalid date qualifier value: {e}") from e
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_qfilter_date,
            name="qfilter_date",
            description="Filter entities and corresponding facts (=entity, key, value, qualifier_key, qualifier_value) with a qualifier condition of date type. 'entities_and_facts' must be the output of filter_str, filter_num, filter_year, filter_date, or relate. (Example: qfilter_date(entities_and_facts=$1, qkey='start_time', qvalue='1980-06-01', op='<'))",
        )

    # Relation and Logic Operations
    def create_relate_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL Relate tool for following relations between entities."""
        self._initialize_engine(kb_path)

        def kopl_relate(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            relation: Annotated[str, ParamMetadata(description="Relation to follow")],
            direction: Annotated[
                Literal["forward", "backward"],
                ParamMetadata(description="Direction: 'forward' or 'backward'"),
            ],
        ) -> str:
            """Find entities that have a specific relation with the given entity"""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                result = self.engine.Relate(entity_tuple, relation, direction)
            except KeyError as e:
                raise AgentException(f"Relation not found: {e}") from e
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_relate,
            name="relate",
            description="Find entities that have a specific relation with the given entity. (Example: relate(entities_and_facts=$1, relation='capital', direction='forward'))",
        )

    def create_and_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL And tool for intersection of entity sets."""
        self._initialize_engine(kb_path)

        def kopl_and(
            l_entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Left entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            r_entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Right entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
        ) -> str:
            """Return the intersection of two entity sets."""
            left_entity_tuple = self._deserialize_entity_tuple(l_entities_and_facts)
            right_entity_tuple = self._deserialize_entity_tuple(r_entities_and_facts)
            assert self.engine is not None
            result = self.engine.And(left_entity_tuple, right_entity_tuple)
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_and,
            name="and",
            description="Return the intersection of two entity sets. (Example: and(l_entities_and_facts=$1, r_entities_and_facts=$2))",
        )

    def create_or_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL Or tool for union of entity sets."""
        self._initialize_engine(kb_path)

        def kopl_or(
            l_entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Left entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            r_entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Right entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
        ) -> str:
            """Return the union of two entity sets."""
            left_entity_tuple = self._deserialize_entity_tuple(l_entities_and_facts)
            right_entity_tuple = self._deserialize_entity_tuple(r_entities_and_facts)
            assert self.engine is not None
            result = self.engine.Or(left_entity_tuple, right_entity_tuple)
            return self._serialize_entity_tuple(result)

        return FunctionTool(
            kopl_or,
            name="or",
            description="Return the union of two entity sets. (Example: or(l_entities_and_facts=$1, r_entities_and_facts=$2))",
        )

    # Query Operations
    def create_query_name_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL QueryName tool for getting entity names."""
        self._initialize_engine(kb_path)

        def kopl_query_name(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
        ) -> str:
            """Return the entity names as list[ValueClass]."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            result = self.engine.QueryName(entity_tuple)
            # Wrap each name result in ValueClass
            from kopl import ValueClass
            result_list = [ValueClass(type="string", value=name, unit=None) for name in result]
            serializable = self._convert_from_value_class(result_list)
            return orjson.dumps(serializable).decode("utf-8")

        return FunctionTool(
            kopl_query_name,
            name="query_name",
            description="Return the entity name. (Example: query_name(entities_and_facts=$1))",
        )

    def create_count_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL Count tool for counting entities."""
        self._initialize_engine(kb_path)

        def kopl_count(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
        ) -> str:
            """Return the number of entities as list[ValueClass]."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            result = self.engine.Count(entity_tuple)
            # Wrap count result in ValueClass
            from kopl import ValueClass
            value_class = ValueClass(type="int", value=result, unit=None)
            result_list = [value_class]
            serializable = self._convert_from_value_class(result_list)
            return orjson.dumps(serializable).decode("utf-8")

        return FunctionTool(
            kopl_count,
            name="count",
            description="Return the number of entities. (Example: count(entities_and_facts=$1))",
        )

    def create_query_attr_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL QueryAttr tool for querying entity attributes."""
        self._initialize_engine(kb_path)

        def kopl_query_attr(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            key: Annotated[str, ParamMetadata(description="Attribute key to query")],
        ) -> str:
            """Return the attribute value of the entity."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                _result: list[ValueClass] = self.engine.QueryAttr(entity_tuple, key)
            except KeyError as e:
                raise AgentException(f"Attribute key not found: {e}") from e
            result: list[dict[str, Any]] = self._convert_from_value_class(_result)
            return orjson.dumps(result).decode("utf-8")

        return FunctionTool(
            kopl_query_attr,
            name="query_attr",
            description="Return the attribute values of the entities. (Example: query_attr(entities_and_facts=$1, key='height'))",
        )

    def create_query_attr_under_condition_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL QueryAttrUnderCondition tool for conditional attribute queries."""
        self._initialize_engine(kb_path)

        def kopl_query_attr_under_condition(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            key: Annotated[str, ParamMetadata(description="Attribute key to query")],
            qkey: Annotated[
                str, ParamMetadata(description="Qualifier key for condition")
            ],
            qvalue: Annotated[
                str, ParamMetadata(description="Qualifier value for condition")
            ],
        ) -> str:
            """Return the attribute value, whose corresponding fact should satisfy the qualifier condition."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                _result: list[ValueClass] = self.engine.QueryAttrUnderCondition(
                    entity_tuple, key, qkey, qvalue
                )
            except KeyError as e:
                raise AgentException(
                    f"Attribute or qualifier key not found: {e}"
                ) from e
            except ValueError as e:
                raise AgentException(f"Invalid qualifier value: {e}") from e
            result: list[dict[str, Any]] = self._convert_from_value_class(_result)
            return orjson.dumps(result).decode("utf-8")

        return FunctionTool(
            kopl_query_attr_under_condition,
            name="query_attr_under_condition",
            description="Return the attribute value, whose corresponding fact should satisfy the qualifier condition. (Example: query_attr_under_condition(entities_and_facts=$1, key='population', qkey='point in time', qvalue='2019'))",
        )

    def create_query_relation_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL QueryRelation tool for querying relations between entities."""
        self._initialize_engine(kb_path)

        def kopl_query_relation(
            s_entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Source entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            t_entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Target entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
        ) -> str:
            """Return the relations between two entities as list[ValueClass]."""
            source_entity_tuple = self._deserialize_entity_tuple(s_entities_and_facts)
            target_entity_tuple = self._deserialize_entity_tuple(t_entities_and_facts)
            assert self.engine is not None
            result = self.engine.QueryRelation(source_entity_tuple, target_entity_tuple)
            # Wrap each relation result in ValueClass
            from kopl import ValueClass
            result_list = [ValueClass(type="string", value=relation, unit=None) for relation in result]
            serializable = self._convert_from_value_class(result_list)
            return orjson.dumps(serializable).decode("utf-8")

        return FunctionTool(
            kopl_query_relation,
            name="query_relation",
            description="Return the relation between two entities. (Example: query_relation(s_entities_and_facts=$1, t_entities_and_facts=$2))",
        )

    # Selection and Comparison Operations
    def create_select_between_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL SelectBetween tool for comparing two entity sets."""
        self._initialize_engine(kb_path)

        def kopl_select_between(
            l_entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Left entities and facts. Only the entities part is used, and the list is assumed to contain exactly one entity.",
                    pattern=r"^\$\d+$",
                ),
            ],
            r_entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Right entities and facts. Only the entities part is used, and the list is assumed to contain exactly one entity.",
                    pattern=r"^\$\d+$",
                ),
            ],
            key: Annotated[str, ParamMetadata(description="Attribute key to compare")],
            op: Annotated[
                Literal["less", "greater"],
                ParamMetadata(description="Comparison: 'less' or 'greater'"),
            ],
        ) -> str:
            """From the two entities, find the one whose attribute value is greater or less and return its name as list[ValueClass]."""
            left_entity_tuple = self._deserialize_entity_tuple(l_entities_and_facts)
            right_entity_tuple = self._deserialize_entity_tuple(r_entities_and_facts)
            assert self.engine is not None
            try:
                result = self.engine.SelectBetween(
                    left_entity_tuple, right_entity_tuple, key, op
                )
            except KeyError as e:
                raise AgentException(f"Attribute key not found: {e}") from e
            # Wrap entity name result in ValueClass
            from kopl import ValueClass
            # Ensure result is a string (handle both single string and other types)
            result_str = result if isinstance(result, str) else str(result)
            value_class = ValueClass(type="string", value=result_str, unit=None)
            result_list = [value_class]
            serializable = self._convert_from_value_class(result_list)
            return orjson.dumps(serializable).decode("utf-8")

        return FunctionTool(
            kopl_select_between,
            name="select_between",
            description="From the two entities, find the one whose attribute value is greater or less and return its name. (Example: select_between(l_entities_and_facts=$1, r_entities_and_facts=$2, key='height', op='greater')",
        )

    def create_select_among_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL SelectAmong tool for finding min/max in entity set."""
        self._initialize_engine(kb_path)

        def kopl_select_among(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            key: Annotated[str, ParamMetadata(description="Attribute key to compare")],
            op: Annotated[
                Literal["smallest", "largest"],
                ParamMetadata(description="Selection: 'smallest' or 'largest'"),
            ],
        ) -> str:
            """From the entity set, find the one whose attribute value is the largest or smallest, returned as list[ValueClass]."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                result = self.engine.SelectAmong(entity_tuple, key, op)
            except KeyError as e:
                raise AgentException(f"Attribute key not found: {e}") from e
            except IndexError as e:
                raise AgentException("Invalid 'entities_and_facts'") from e
            # Wrap entity name result in ValueClass
            from kopl import ValueClass
            # Ensure result is a string (handle both single string and list cases)
            result_str = result if isinstance(result, str) else str(result)
            value_class = ValueClass(type="string", value=result_str, unit=None)
            result_list = [value_class]
            serializable = self._convert_from_value_class(result_list)
            return orjson.dumps(serializable).decode("utf-8")

        return FunctionTool(
            kopl_select_among,
            name="select_among",
            description="From the entity set, find the one whose attribute value is the largest or smallest. (Example: select_among(entities_and_facts=$1, key='height', op='largest'))",
        )

    # Verification Operations
    def create_verify_str_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL VerifyStr tool for string verification."""
        self._initialize_engine(kb_path)

        def kopl_verify_str(
            s_value: Annotated[
                str,
                ParamMetadata(
                    description="Source string value (the output of query_attr, query_attr_under_condition, query_attr_qualifier or query_relation_qualifier)",
                    pattern=r"^\$\d+$",
                ),
            ],
            t_value: Annotated[str, ParamMetadata(description="Target string value")],
        ) -> str:
            """Return whether the output of Query* and the given value are equal as string."""
            assert self.engine is not None
            try:
                s_value_list: list[dict[str, Any]] = orjson.loads(s_value)
                s_value_parsed: list[ValueClass] = [
                    self.engine.kb._parse_value(value) for value in s_value_list
                ]
            except Exception as e:
                raise AgentException(
                    "Invalid 's_value'. It must be the output of query_attr, query_attr_under_condition, query_attr_qualifier or query_relation_qualifier"
                ) from e
            try:
                result = self.engine.VerifyStr(s_value_parsed, t_value)
            except ValueError as e:
                raise AgentException(f"Invalid target string value: {e}") from e
            return str(result)

        return FunctionTool(
            kopl_verify_str,
            name="verify_str",
            description="Return whether the output of Query* and the given value are equal as string. (Example: verify_str(s_value=$2, t_value='male')",
        )

    def create_verify_num_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL VerifyNum tool for numeric verification."""
        self._initialize_engine(kb_path)

        def kopl_verify_num(
            s_value: Annotated[
                str,
                ParamMetadata(
                    description="Source numeric value. (the output of query_attr, query_attr_under_condition, query_attr_qualifier or query_relation_qualifier)",
                    pattern=r"^\$\d+$",
                ),
            ],
            t_value: Annotated[
                float | str,
                ParamMetadata(description="Target numeric value"),
            ],
            op: Annotated[
                Literal["=", "!=", "<", ">"],
                ParamMetadata(description="Comparison operator: '=', '!=', '<', '>'"),
            ],
        ) -> str:
            """Return whether the output of Query* and the given number satisfy the condition."""
            assert self.engine is not None

            if isinstance(t_value, str):
                try:
                    t_value = float(t_value)
                except ValueError as e:
                    raise AgentException(
                        f"t_value should be a number, got: {t_value}"
                    ) from e

            try:
                s_value_list: list[dict[str, Any]] = orjson.loads(s_value)
                s_value_parsed: list[ValueClass] = [
                    self.engine.kb._parse_value(value) for value in s_value_list
                ]
            except Exception as e:
                raise AgentException(
                    "Invalid 's_value'. It must be the output of query_attr, query_attr_under_condition, query_attr_qualifier or query_relation_qualifier"
                ) from e
            try:
                result = self.engine.VerifyNum(s_value_parsed, str(t_value), op)
            except ValueError as e:
                raise AgentException(f"Invalid target numeric value: {e}") from e

            return str(result)

        return FunctionTool(
            kopl_verify_num,
            name="verify_num",
            description="Return whether the output of Query* and the given number satisfy the condition. (Example: verify_num(s_value=$2, t_value=2000, op='>'))",
        )

    def create_verify_year_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL VerifyYear tool for year verification."""
        self._initialize_engine(kb_path)

        def kopl_verify_year(
            s_value: Annotated[
                str,
                ParamMetadata(
                    description="Source year value (the output of query_attr, query_attr_under_condition, query_attr_qualifier or query_relation_qualifier)",
                    pattern=r"^\$\d+$",
                ),
            ],
            t_value: Annotated[
                str,
                ParamMetadata(
                    description="Target year value (yyyy-mm-dd)", pattern=r"^\d{4}$"
                ),
            ],
            op: Annotated[
                Literal["=", "!=", "<", ">"],
                ParamMetadata(description="Comparison operator: '=', '!=', '<', '>'"),
            ],
        ) -> str:
            """Return whether the output of Query* and the given year satisfy the condition."""
            assert self.engine is not None
            try:
                s_value_list: list[dict[str, Any]] = orjson.loads(s_value)
                s_value_parsed: list[ValueClass] = [
                    self.engine.kb._parse_value(value) for value in s_value_list
                ]
            except Exception as e:
                raise AgentException(
                    "Invalid 's_value'. It must be the output of query_attr, query_attr_under_condition, query_attr_qualifier or query_relation_qualifier"
                ) from e
            try:
                result = self.engine.VerifyYear(s_value_parsed, t_value, op)
            except ValueError as e:
                raise AgentException(f"Invalid target year value: {e}") from e
            return str(result)

        return FunctionTool(
            kopl_verify_year,
            name="verify_year",
            description="Return whether the output of Query* and the given year satisfy the condition. (Example: verify_year(s_value=$2, t_value='1980', op='>'))",
        )

    def create_verify_date_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL VerifyDate tool for date verification."""
        self._initialize_engine(kb_path)

        def kopl_verify_date(
            s_value: Annotated[
                str,
                ParamMetadata(
                    description="Source date value (the output of query_attr, query_attr_under_condition, query_attr_qualifier or query_relation_qualifier)",
                    pattern=r"^\$\d+$",
                ),
            ],
            t_value: Annotated[
                str,
                ParamMetadata(
                    description="Target date value (YYYY-MM-DD)",
                    pattern=r"^\d{4}-\d{2}-\d{2}$",
                ),
            ],
            op: Annotated[
                Literal["=", "!=", "<", ">"],
                ParamMetadata(description="Comparison operator: '=', '!=', '<', '>'"),
            ],
        ) -> str:
            """Return whether the output of Query* and the given date satisfy the condition."""
            assert self.engine is not None
            try:
                s_value_list: list[dict[str, Any]] = orjson.loads(s_value)
                s_value_parsed: list[ValueClass] = [
                    self.engine.kb._parse_value(value) for value in s_value_list
                ]
            except Exception as e:
                raise AgentException(
                    "Invalid 's_value'. It must be the output of query_attr, query_attr_under_condition, query_attr_qualifier or query_relation_qualifier"
                ) from e
            try:
                result = self.engine.VerifyDate(s_value_parsed, t_value, op)
            except ValueError as e:
                raise AgentException(f"Invalid target date value: {e}") from e
            return str(result)

        return FunctionTool(
            kopl_verify_date,
            name="verify_date",
            description="Return whether the output of Query* and the given date satisfy the condition. (Example: verify_date(s_value=$2, t_value='1980-06-01', op='>'))",
        )

    def create_query_attr_qualifier_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL QueryAttrQualifier tool for querying attribute qualifiers."""
        self._initialize_engine(kb_path)

        def kopl_query_attr_qualifier(
            entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Input entities and facts", pattern=r"^\$\d+$"
                ),
            ],
            key: Annotated[str, ParamMetadata(description="Attribute key")],
            value: Annotated[str, ParamMetadata(description="Attribute value")],
            qkey: Annotated[str, ParamMetadata(description="Qualifier key to query")],
        ) -> str:
            """Return the qualifier values of the fact (Entity, Key, Value) as list[ValueClass]."""
            entity_tuple = self._deserialize_entity_tuple(entities_and_facts)
            assert self.engine is not None
            try:
                values: list[ValueClass] = self.engine.QueryAttrQualifier(
                    entity_tuple, key, value, qkey
                )
            except KeyError as e:
                raise AgentException(f"Attribute key not found: {e}") from e
            except ValueError as e:
                raise AgentException(f"Invalid attribute value: {e}") from e
            # Return full ValueClass objects, not just values
            result: list[dict[str, Any]] = self._convert_from_value_class(values)
            return orjson.dumps(result).decode("utf-8")

        return FunctionTool(
            kopl_query_attr_qualifier,
            name="query_attr_qualifier",
            description="Return the qualifier value of the fact (Entity, Key, Value). (Example: query_attr_qualifier(entities_and_facts=$2, key='population', value='199110', qkey='point in time'))",
        )

    def create_query_relation_qualifier_tool(
        self, kb_path: Path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    ) -> FunctionTool:
        """Create a KoPL QueryRelationQualifier tool for querying relation qualifiers."""
        self._initialize_engine(kb_path)

        def kopl_query_relation_qualifier(
            s_entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Source entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            t_entities_and_facts: Annotated[
                str,
                ParamMetadata(
                    description="Target entities and facts. Only the entities part is used.",
                    pattern=r"^\$\d+$",
                ),
            ],
            relation: Annotated[str, ParamMetadata(description="Relation to query")],
            qkey: Annotated[str, ParamMetadata(description="Qualifier key to query")],
        ) -> str:
            """Return the qualifier value of the fact (Entity, Pred, Entity)."""
            source_entity_tuple = self._deserialize_entity_tuple(s_entities_and_facts)
            target_entity_tuple = self._deserialize_entity_tuple(t_entities_and_facts)
            assert self.engine is not None
            try:
                _result: list[ValueClass] = self.engine.QueryRelationQualifier(
                    source_entity_tuple, target_entity_tuple, relation, qkey
                )
            except KeyError as e:
                raise AgentException(f"Relation not found: {e}") from e
            result: list[dict[str, Any]] = self._convert_from_value_class(_result)
            return orjson.dumps(result).decode("utf-8")

        return FunctionTool(
            kopl_query_relation_qualifier,
            name="query_relation_qualifier",
            description="Return the qualifier value of the fact (Entity, Pred, Entity). (Example: query_relation_qualifier(s_entities_and_facts=$2, t_entities_and_facts=$3, relation='drafted by', qkey='point in time'))",
        )


class KoPLSchemaFreeToolFactory(KoPLToolFactory):
    """Factory for schema-free KoPL tools subset."""

    def create_all_tools(
        self, kb_path: str = "data/kopl_kbqa/kqa_pro/kb.json", **kwargs
    ) -> dict[str, Tool]:
        """Create only schema-free KoPL tools."""
        # Call parent to create all tools
        all_tools = super().create_all_tools(kb_path=kb_path, **kwargs)

        # Filter to schema-free operators only
        schema_free_operators = [
            "find_all",
            "and",
            "or",
            "query_name",
            "count",
            "query_relation",
            "verify_str",
            "verify_num",
            "verify_year",
            "verify_date",
        ]

        return {
            tool_id: tool
            for tool_id, tool in all_tools.items()
            if tool_id.split("/")[-1] in schema_free_operators
        }


class KoPLFindFilterConceptToolFactory(KoPLToolFactory):
    """Factory for find and filter_concept tools."""

    def create_all_tools(
        self, kb_path: str = "data/kopl_kbqa/kqa_pro/kb.json", **kwargs
    ) -> dict[str, Tool]:
        """Create only find and filter_concept tools."""
        all_tools = super().create_all_tools(kb_path=kb_path, **kwargs)
        return {
            tool_id: tool
            for tool_id, tool in all_tools.items()
            if tool_id in ["kopl/find", "kopl/filter_concept"]
        }


class KoPLKeyOnlyToolFactory(KoPLToolFactory):
    """Factory for key-only KoPL tools."""

    def create_all_tools(
        self, kb_path: str = "data/kopl_kbqa/kqa_pro/kb.json", **kwargs
    ) -> dict[str, Tool]:
        """Create only key-only KoPL tools."""
        all_tools = super().create_all_tools(kb_path=kb_path, **kwargs)
        key_only_operators = ["relate", "query_attr", "select_between", "select_among"]
        return {
            tool_id: tool
            for tool_id, tool in all_tools.items()
            if tool_id.split("/")[-1] in key_only_operators
        }


class KoPLKeyValueToolFactory(KoPLToolFactory):
    """Factory for key-value KoPL tools."""

    def create_all_tools(
        self, kb_path: str = "data/kopl_kbqa/kqa_pro/kb.json", **kwargs
    ) -> dict[str, Tool]:
        """Create only key-value KoPL tools."""
        all_tools = super().create_all_tools(kb_path=kb_path, **kwargs)
        key_value_operators = [
            "filter_str",
            "filter_num",
            "filter_year",
            "filter_date",
            "qfilter_str",
            "qfilter_num",
            "qfilter_year",
            "qfilter_date",
            "query_attr_under_condition",
            "query_attr_qualifier",
            "query_relation_qualifier",
        ]
        return {
            tool_id: tool
            for tool_id, tool in all_tools.items()
            if tool_id.split("/")[-1] in key_value_operators
        }


if __name__ == "__main__":
    # Test basic functionality
    kb_path = Path("data/kopl_kbqa/kqa_pro/kb.json")
    factory = KoPLToolFactory()

    # Test basic tools
    find_tool = factory.create_find_tool(kb_path=kb_path)
    filter_num_tool = factory.create_filter_num_tool(kb_path=kb_path)
    select_between_tool = factory.create_select_between_tool(kb_path=kb_path)

    # result1 = engine.Find("Jerusalem")
    # print(result1)
    # result2 = engine.FilterNum(result1, "population", "75200", "=")
    # print(result2)
    # result3 = engine.Find("Baghdad")
    # print(result3)
    # result4 = engine.SelectBetween(result2, result3, "elevation above sea level", "greater")
    # print(result4)

    result1 = find_tool.execute("Jerusalem").result_data
    print(f"Find Tool Result: {result1}")
    result2 = filter_num_tool.execute(result1, "population", "75200", "=").result_data
    print(f"FilterNum Tool Result: {result2}")
    result3 = find_tool.execute(name="Baghdad").result_data
    print(f"Find Tool Result: {result3}")
    result4 = select_between_tool.execute(
        result2, result3, "elevation above sea level", "greater"
    ).result_data
    print(f"SelectBetween Tool Result: {result4}")
