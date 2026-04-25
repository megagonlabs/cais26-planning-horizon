"""
KoPL Worker Agents for Knowledge Base Question Answering.

This module provides specialized Worker Agents for KBQA tasks using KoPL operators.
Each agent performs a single KoPL operator with a 3-step workflow:
preprocess → execute → postprocess.
"""

from pathlib import Path
from textwrap import indent
from time import time
from typing import Any, Callable, Optional, TYPE_CHECKING
import copy
import json
import logging
import pickle
import threading

# Lazy imports - only load heavy ML libraries when actually needed
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from kopl import ValueClass
import numpy as np
import orjson

from ...environment.environment import Environment
from ..base_agent import BaseWorkerAgent
from ..exceptions import AgentException
from ..executable import ExecutionResult, ExecutableType
from ..kopl_utils import OPERATOR_TO_VALUE_TYPE, SCALAR_RESULT_OPERATORS
from ..memory import ContextMemory
from ..step import Step, StepStatus

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from ...environment.environment import Environment

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_EMBEDDINGS_DIR = Path("data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5")
DEFAULT_ENCODER_MODEL = "BAAI/bge-base-en-v1.5"
ENTITY_EMBEDDINGS_FILE = "entity_embeddings.pkl"
KEY_EMBEDDINGS_FILE = "key_embeddings.pkl"
VALUE_EMBEDDINGS_FILE = "value_embeddings.pkl"
MAX_ENTITY_LABELS_IN_OBSERVATION = 10  # Maximum number of entity labels to show in observation text

# Thread-safe cache for embedding files (filepath -> loaded object)
_EMBEDDING_CACHE: dict[str, Any] = {}
_EMBEDDING_CACHE_LOCK = threading.Lock()

# Thread-safe cache for encoder models (model_name -> SentenceTransformer)
_ENCODER_MODEL_CACHE: dict[str, "SentenceTransformer"] = {}
_ENCODER_MODEL_CACHE_LOCK = threading.Lock()

# Prompt for KBQA schema grounding
SYS_PROMPT_KBQA_SCHEMA_GROUNDING = """
You are a KB schema grounding assistant.

## Task

Given:
- an input question (natural language; use only for light context),
- a query relation (Freebase-style),
- a ranked list of candidate KB relations/attributes,

select the single candidate that is an exact semantic match to the provided query relation. Match the relation's semantics, direction, and argument types (domain → range). If no candidate **clearly** satisfies the query relation, output "none". Optimize for precision over recall.

## Strict Matching Policy (apply exactly)

- The query relation is the source of truth; do not reinterpret it to fit the question text.
- Direction must match. If a candidate is the inverse (B → A when the query is A → B), choose "none".
- Domain and range types must match. Reject candidates whose subject/object types or scope differ (e.g., game vs. version; episode vs. segment).
- Granularity must match. Do not select container/super-properties or aggregates in place of a specific attribute (e.g., overall statistics vs. a specific stat field).
- Representation/units/aggregation must match. Do not equate a generic value with a specific unit-converted value, or total vs. per-item, category vs. members.
- Synonym: Accept only trivial differences (pluralization or obvious name synonyms) when direction, granularity, and types are identical.
- Prefer "none" when uncertain or when candidates are only loosely related.

Constraints:
- The answer must be exactly one candidate string from the list or "none".
- Keep reasoning concise and focused on why the chosen candidate exactly matches (or why none do).
""".strip()
USER_PROMPT_KBQA_SCHEMA_GROUNDING = """
Ground the query relation to one candidate (or `"none"`).

- **Question:** {question}
- **Query relation:** {query}
- **Candidates (pick one or none):**
{candidates}
""".strip()

RESPONSE_FORMAT_KBQA_SCHEMA_GROUNDING = {
    "name": "schema_matching_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string", "description": "brief justification (1 sentence)"},
            "answer": {"type": "string", "description": "one of the candidate strings or 'none'"},
        },
        "required": ["reasoning", "answer"],
        "additionalProperties": False,
    },
}


def load_embeddings(path: Path) -> Any:
    """
    Load embeddings from a pickle file with thread-safe caching.

    Args:
        path: Path to the embeddings file

    Returns:
        Any: Loaded embeddings object

    Raises:
        ValueError: If the file does not exist
        pickle.UnpicklingError: If the file is not a valid pickle
    """
    if not path.exists():
        raise ValueError(f"Embeddings file not found at {path}")
    path_str = str(path.resolve())
    # Fast path: check without lock (safe for reads)
    obj = _EMBEDDING_CACHE.get(path_str)
    if obj is not None:
        return obj
    # Lock for thread-safe write
    with _EMBEDDING_CACHE_LOCK:
        # Double-check in case another thread loaded it
        obj = _EMBEDDING_CACHE.get(path_str)
        if obj is not None:
            return obj
        with open(path, "rb") as f:
            obj = pickle.load(f)
        _EMBEDDING_CACHE[path_str] = obj
        logger.info(f"Loaded {len(obj)} embeddings from {path}")
        return obj


def get_encoder_model(model_name: str) -> "SentenceTransformer":
    """
    Get or create a shared encoder model with thread-safe caching.

    Args:
        model_name: Name of the SentenceTransformer model (e.g., "BAAI/bge-base-en-v1.5")

    Returns:
        SentenceTransformer: Shared model instance

    Raises:
        ImportError: If sentence-transformers is not installed
    """
    # Fast path: check without lock (safe for reads)
    model = _ENCODER_MODEL_CACHE.get(model_name)
    if model is not None:
        return model

    # Lock for thread-safe write
    with _ENCODER_MODEL_CACHE_LOCK:
        # Double-check in case another thread loaded it
        model = _ENCODER_MODEL_CACHE.get(model_name)
        if model is not None:
            return model

        # Lazy import and load model
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        _ENCODER_MODEL_CACHE[model_name] = model
        logger.info(f"Loaded shared encoder model: {model_name}")
        return model


def parse_entities_and_facts(
    params: dict[str, Any],
    field_names: list[str],
    agent_name: str,
) -> tuple[list[str], Optional[list[dict[str, Any]]]]:
    """
    Parse entities and optionally facts from JSON parameters.

    Handles multiple parameter names (entities_and_facts, l_entities_and_facts,
    r_entities_and_facts, s_entities_and_facts).

    Args:
        params: Input parameters dict
        field_names: List of field names to check (e.g., ["entities_and_facts"])
        agent_name: Name of agent for error messages

    Returns:
        tuple[list[str], Optional[list[dict[str, Any]]]]: Tuple of (entities, facts)

    Raises:
        AgentException: If JSON parsing fails or required fields are missing
    """
    entities = []
    facts = None

    for field_name in field_names:
        if field_name not in params:
            continue

        try:
            data = json.loads(params[field_name])
            if "entities" in data:
                entities.extend(data["entities"])
            if "facts" in data and facts is None:
                facts = data["facts"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise AgentException(
                f"Invalid JSON format for '{field_name}'\n> {str(params[field_name])[:100]}...\nError: {e}",
                agent_name=agent_name,
            )

    if not entities and any(field_name in params for field_name in field_names):
        raise AgentException(
            f"No entities found in {field_names}",
            agent_name=agent_name,
        )

    return entities, facts


def extract_entity_name_candidates(kb: Any, entities: Optional[list[str]] = None) -> set[str]:
    """
    Extract entity/concept names from KB.

    Args:
        kb: Knowledge base object
        entities: Optional list of entity IDs to extract concepts from.
                 If None, returns all entity names in KB.

    Returns:
        set[str]: Set of entity/concept names
    """
    if entities is None:
        # All entity/concept names in KB
        return set(kb.name_to_id.keys())

    # Get all candidate names linked to specific entities
    candidates = [kb.entities[cid]["name"] for eid in entities for cid in kb.get_all_concepts(eid)]
    return set(candidates)


def extract_attribute_key_candidates(
    kb: Any,
    entities: list[str],
    value_type: Optional[set[str]] = None,
) -> set[str]:
    """
    Extract attribute keys from entities, optionally filtered by value type.

    Args:
        kb: Knowledge base object
        entities: List of entity IDs
        value_type: Optional set of value types to filter by (e.g., {"string", "quantity"})

    Returns:
        set[str]: Set of attribute keys
    """
    candidates = []
    for ent_id in entities:
        entity = kb.entities[ent_id]
        for attr in entity["attributes"]:
            if value_type is None or attr["value"].type in value_type:
                candidates.append(attr["key"])
    return set(candidates)


def extract_relation_key_candidates(kb: Any, entities: list[str]) -> set[str]:
    """
    Extract relation keys from entities.

    Args:
        kb: Knowledge base object
        entities: List of entity IDs

    Returns:
        set[str]: Set of relation keys
    """
    candidates = []
    for ent_id in entities:
        entity = kb.entities[ent_id]
        for rel in entity["relations"]:
            candidates.append(rel["relation"])
    return set(candidates)


def _matches_value_type(qval: Any, value_type: set[str]) -> bool:
    """
    Check if a qualifier value matches one of the expected value types.

    Handles both ValueClass instances and dict representations.

    Args:
        qval: Qualifier value (ValueClass or dict)
        value_type: Set of expected value types to match against

    Returns:
        bool: True if the value's type is in value_type
    """
    # Lazy import
    from kopl import ValueClass

    if isinstance(qval, ValueClass):
        return qval.type in value_type
    elif isinstance(qval, dict):
        return qval.get("type") in value_type
    return False


def extract_qualifier_key_candidates(
    kb: Any,
    entities: list[str],
    is_attribute: bool,
    facts: Optional[list[dict[str, Any]]] = None,
    value_type: Optional[set[str]] = None,
) -> set[str]:
    """
    Extract qualifier keys from entities or facts.

    Args:
        kb: Knowledge base object
        entities: List of entity IDs
        is_attribute: Whether to extract from attributes or relations
        facts: Optional list of facts to extract qualifiers from
        value_type: Optional set of value types to filter by

    Returns:
        set[str]: Set of qualifier keys
    """
    candidates = []

    # Extract from facts if provided
    if facts:
        for fact in facts:
            qualifiers = fact.get("qualifiers", {})
            for qkey, qvals in qualifiers.items():
                if value_type is None:
                    candidates.append(qkey)
                else:
                    # Check if any qualifier value matches the expected type
                    for qval in qvals:
                        if _matches_value_type(qval, value_type):
                            candidates.append(qkey)
                            break

    # Extract from entity attributes if no facts provided
    if not candidates and entities:
        for ent_id in entities:
            entity = kb.entities[ent_id]
            field_name = "attributes" if is_attribute else "relations"
            for attr in entity[field_name]:
                qualifiers = attr.get("qualifiers", {})
                for qkey in qualifiers.keys():
                    candidates.append(qkey)

    return set(candidates)


class KoPLAgent(BaseWorkerAgent):
    """
    Unified KoPL Worker Agent for single operator execution.

    Implements a 3-step workflow:
    1. preprocess_input: Validate parameters and perform schema grounding unless strict_mode
    2. execute_tool: Call KoPL operator with resolved parameters
    3. postprocess_output: Generate short observation for Meta Agent

    Stores full intermediate results in Step.data["full"] while returning
    concise observations for Meta Agent coordination.

    Tools are looked up at runtime from the environment's tool registry,
    allowing proper separation of tool and agent management.
    """

    def __init__(
        self,
        operator_name: str,
        llm_call: Optional[Callable] = None,
        llm_options: dict[str, Any] = {},
        tool_ids: Optional[list[str]] = None,
        update_schema_from_tool: bool = True,
        strict_mode: bool = False,
        retrieval_topk: int = 10,
        **kwargs,
    ):
        """
        Initialize the KoPL Agent.

        Args:
            operator_name: Name of the KoPL operator (e.g., "find_all", "filter_concept")
            llm_call: Callable function to get LLM responses (optional for preprocessing)
            llm_options: Options to configure LLM behavior
            tool_ids: List of tool IDs this agent can use (e.g., ["kopl/find_all"])
            update_schema_from_tool: Whether to update agent schema from tool definition
            strict_mode: If True, require exact matches for schema grounding
            retrieval_topk: Number of top candidates to consider for LLM schema matching
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            llm_call,
            llm_options,
            name=operator_name,
            max_steps=1,
            tool_ids=tool_ids,
            update_schema_from_tool=update_schema_from_tool,
        )
        self.operator_name = operator_name
        self.tool_name = f"kopl/{operator_name}"  # Tool to look up at runtime
        self.strict_mode = strict_mode
        self.kopl_factory: Optional[Any] = None  # Injected by AgentRegistry for preprocessing
        self.retrieval_topk = retrieval_topk  # Threshold for embedding pre-filtering

    def _llm_schema_matching(
        self,
        question: str,
        query_item: str,
        candidates: list[str],
        **kwargs,
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Use LLM to match query item to KB schema candidates.

        Args:
            question: The original question for context
            query_item: The item description to ground (entity/key/relation)
            candidates: List of candidate KB items (≤10 recommended)
            **kwargs: Additional context

        Returns:
            Tuple of (matched_item_or_none, metadata_dict)
            - matched_item: The selected candidate, or None if LLM returns "none"
            - metadata: Dict with llm_reasoning and llm_token_usage

        Raises:
            AgentException: If LLM call or parsing fails
            ValueError: If LLM returns invalid answer
        """

        # Format candidates for prompt
        candidates_text = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(candidates)])

        # Format prompts
        system_prompt = SYS_PROMPT_KBQA_SCHEMA_GROUNDING
        user_prompt = USER_PROMPT_KBQA_SCHEMA_GROUNDING.format(
            question=question,
            query=query_item,
            candidates=candidates_text,
        )

        llm_input = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Create dynamic response format with enum constraint
        response_format = {
            "type": "json_schema",
            "json_schema": copy.deepcopy(RESPONSE_FORMAT_KBQA_SCHEMA_GROUNDING),
        }
        # Add dynamic enum constraint
        response_format["json_schema"]["schema"]["properties"]["answer"]["enum"] = candidates + ["none"]  # type: ignore

        # Call LLM with structured output
        try:
            response = self.get_llm_response(llm_input, response_format=response_format)
            parsed_result = json.loads(response.content)
            token_usage = response.token_usage
        except KeyboardInterrupt:
            logger.info("LLM schema matching interrupted by user")
            raise
        except Exception as e:
            logger.error(f"LLM schema matching failed: {str(e)}")
            raise ValueError(f"Schema matching failed: {str(e)}")

        # Extract result
        reasoning = parsed_result["reasoning"]
        answer = parsed_result["answer"]

        # Validate answer
        if answer not in candidates and answer != "none":
            ## This should not happen due to enum constraint
            raise ValueError(f"LLM returned invalid answer: {answer}, expected one of {candidates} or 'none'")

        # Return None if LLM says "none"
        matched_item = None if answer == "none" else answer

        metadata = {
            "llm_reasoning": reasoning,
            "llm_token_usage": token_usage,
        }

        return matched_item, metadata

    def _schema_matching(
        self,
        query: str,
        candidates: set[str],
        precomputed_embeddings: dict[str, Any],
        encoder_model: "SentenceTransformer",
        question: str,
        param_name: str = "",
        op_name: str = "",
        use_llm_matching: bool = True,
        min_similarity: float = 0.9,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Unified schema matching: embedding-based or hybrid (embedding + LLM) matching.

        Args:
            query: Query string to match
            candidates: Set of candidate strings
            precomputed_embeddings: Dictionary mapping strings to embeddings
            encoder_model: Model for encoding the query
            question: The original user question for LLM context
            param_name: Parameter name for error messages
            op_name: Operation name for error messages
            use_llm_matching: If True, use LLM for final selection; if False, use embedding similarity only
            min_similarity: Minimum similarity threshold (only used when use_llm_matching=False)

        Returns:
            str: Best matching candidate

        Raises:
            ValueError: If no candidates are provided
            AgentException: If no match is found
        """
        if not candidates:
            raise ValueError("No candidates provided for similarity matching")

        # Return if exact match exists
        if query in candidates:
            return query

        # Single candidate - return it directly
        candidates_list = list(candidates)
        if len(candidates_list) == 1:
            return candidates_list[0]

        # Compute embedding similarities
        candidate_embs = np.vstack([precomputed_embeddings[name] for name in candidates_list])
        query_emb = precomputed_embeddings.get(
            query,
            encoder_model.encode(query, normalize_embeddings=True),
        )
        similarities = candidate_embs.dot(query_emb)
        sorted_indices = np.argsort(-similarities)
        sorted_candidates = [candidates_list[idx] for idx in sorted_indices]
        sorted_similarities = similarities[sorted_indices]

        # Pre-filter to top-k candidates if needed
        candidates_for_llm = candidates_list
        if len(candidates_list) > self.retrieval_topk:
            candidates_for_llm = sorted_candidates[: self.retrieval_topk]
            logger.info(f"Pre-filtered {len(candidates_list)} candidates to top-{self.retrieval_topk} using embeddings")

        # Strict mode - raise error with candidate list
        if self.strict_mode:
            candidates_str = "\n".join(f"- {candidate}" for candidate in candidates_for_llm)
            error_msg = f"'{query}' not found in the KB for the given input."
            if len(candidates_list) > self.retrieval_topk:
                error_msg += f" Top candidates among {len(candidates_list)} relations:\n"
            else:
                error_msg += " All candidates:\n"
            error_msg += f"{candidates_str}\nRetry with a valid candidate, or try a different input/tool."
            raise AgentException(error_msg)

        # Branch based on whether LLM matching is enabled
        if not use_llm_matching:  # Embedding-only matching path
            # Check minimum similarity threshold
            if sorted_similarities[0] < min_similarity:
                log_candidates = ", ".join(
                    f"{name} ({sim:.4f})" for name, sim in zip(sorted_candidates, sorted_similarities)
                )
                logger.info(
                    f"KB grounding failed. '{query}' does not match any candidate closely enough (min={min_similarity}). Top candidates: {log_candidates}"
                )
                candidate_str = "\n".join(f"- {candidate}" for candidate in sorted_candidates)
                error_msg = f"'{query}' not found in the KB for the given input."
                if len(candidates_list) > self.retrieval_topk:
                    error_msg += f" Top candidates among {len(candidates_list)} relations:\n"
                else:
                    error_msg += " Available candidates (all):\n"
                error_msg += f"{candidate_str}\nRetry with a valid candidate, or try a different input/tool."
                raise AgentException(error_msg, agent_name=param_name)

            # Return top-1 match
            logger.info(f"Schema matching: '{query}' → '{sorted_candidates[0]}' (embedding matched)")
            return sorted_candidates[0]

        # LLM-based matching path

        # LLM-based schema matching
        best_match, llm_metadata = self._llm_schema_matching(
            question=question,
            query_item=query,
            candidates=candidates_for_llm,
        )

        # Store LLM metadata if a metadata dict was provided
        if metadata is not None:
            metadata.update(llm_metadata)

        # Handle "none" result
        if best_match is None:
            candidate_str = "\n".join(f"- {candidate}" for candidate in candidates_for_llm)
            error_msg = f"'{query}' not found in the KB for the given input."
            if len(candidates_list) > self.retrieval_topk:
                error_msg += f" Top candidates among {len(candidates_list)} items:\n"
            else:
                error_msg += " Available candidates (all):\n"
            error_msg += f"{candidate_str}\nRetry with a valid candidate, or try a different input/tool."

            logger.info(error_msg)
            raise AgentException(error_msg, agent_name=param_name)

        logger.info(f"Schema matching: '{query}' → '{best_match}' (LLM matched)")
        return best_match

    def run_episode(
        self,
        query_or_params: str | dict[str, Any],
        environment: "Environment",
        task_config: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        memory: Optional[ContextMemory] = None,
        step_index: Optional[int] = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Execute the 3-step KoPL workflow.

        Note: KoPL agents expect dict parameters via query_or_params.

        Args:
            query_or_params: Operator parameters from Meta Agent (already with $i references resolved)
            environment: Environment providing state management and memory access
            task_config: Optional task-specific configuration
            session_id: Optional session ID (unused)
            parent_session_id: Optional parent session ID for memory resolution
            step_index: Current step index in the overall plan
            memory: Context memory for storing execution steps
            **kwargs: Additional parameters (unused)

        Returns:
            ExecutionResult: Result with short observation and full data in metadata
        """
        metadata = {}
        error_message = None

        start_time = time()

        if step_index is None:
            raise ValueError("step_index must be provided for KoPLAgent")

        if not isinstance(query_or_params, dict):
            error_message = (
                f"{self.name} requires structured parameters as a dict via query_or_params, got: {query_or_params}"
            )
            logger.error(error_message)
            return ExecutionResult(
                executable_name=self.name,
                executable_type=ExecutableType.AGENT,
                success=False,
                result_data=None,
                error_message=error_message,
                execution_time=0.0,
                token_usage={},
                metadata={"agent_type": "kopl"},
            )

        params = query_or_params  # Expecting dict parameters

        # Validate parameters
        try:
            self.validate_input(params)
        except AgentException as e:
            execution_time = time() - start_time
            error_message = "Parameter validation failed.\n" + indent(str(e), "> ")
            logger.debug(error_message)
            return ExecutionResult(
                executable_name=self.name,
                executable_type=ExecutableType.AGENT,
                success=False,
                error_message=error_message,
                execution_time=execution_time,
                metadata={
                    "agent_type": "kopl",
                },
            )

        # Initialize memory if not provided
        if memory is None:
            memory = ContextMemory(
                query="",  # KoPL agents don't have a user query
                session_id=session_id or f"kopl_{self.operator_name}_{start_time}",
                agent_name=self.name,
                parent_session_id=parent_session_id,
            )
            environment.register_memory(memory)

        # Step 1: Preprocess input - resolve $i references to actual data
        step_num = len(memory.step_history)
        preprocess_step = Step(
            step_num=step_num,
            step_type=f"{self.name.lower()}_preprocess",
            status=StepStatus.PLANNED,
            data={"input": query_or_params},
        )
        memory.add_step(preprocess_step)

        # Extract parent question once for preprocessing
        parent_memory = environment.get_memory(parent_session_id) if parent_session_id else None
        question = parent_memory.query if parent_memory else ""

        try:
            processed_params, preprocessing_metadata = self.preprocess_input(params, parent_question=question)
        except AgentException as e:
            preprocess_step.status = StepStatus.FAILED
            # LLM-recoverable error
            error_message = f"Error: {str(e)}"
            logger.info(error_message)
            execution_time = time() - start_time
            return ExecutionResult(
                executable_name=self.name,
                executable_type=ExecutableType.AGENT,
                success=False,
                result_data=None,
                error_message=error_message,
                execution_time=execution_time,
                token_usage={},
                metadata={"agent_type": "kopl"},
            )

        # Update preprocessing step with results
        preprocess_step.status = StepStatus.COMPLETED
        preprocess_step.data["output"] = processed_params
        preprocess_step.metadata = preprocessing_metadata

        metadata["processed_params"] = processed_params
        metadata.update(preprocessing_metadata)

        # Step 2: Execute tool - call KoPL operator with resolved parameters
        step_num = len(memory.step_history)
        execute_step = Step(
            step_num=step_num,
            step_type=f"{self.name.lower()}_execute",
            status=StepStatus.PLANNED,
            data={"input": processed_params},
        )
        memory.add_step(execute_step)

        tool_result = self.execute_tool(processed_params, environment)

        # Update execution step with results
        execute_step.status = StepStatus.COMPLETED
        execute_step.data["output"] = tool_result.result_data
        execute_step.metadata = tool_result.metadata

        # Step 3: Postprocess output - generate short observation
        observation = None
        postprocess_success = False
        if tool_result.success:
            step_num = len(memory.step_history)
            postprocess_step = Step(
                step_num=step_num,
                step_type=f"{self.name.lower()}_postprocess",
                status=StepStatus.PLANNED,
                data={"input": tool_result.result_data},
            )
            memory.add_step(postprocess_step)

            observation, postprocess_success = self.postprocess_output(tool_result, environment)

            # Update postprocessing step with results
            if postprocess_success:
                postprocess_step.status = StepStatus.COMPLETED
            else:
                postprocess_step.status = StepStatus.FAILED
                error_message = observation  # observation contains error message when success=False
            postprocess_step.data["output"] = observation
        else:
            error_message = tool_result.error_message

        execution_time = time() - start_time

        # Overall success requires both tool execution and postprocessing to succeed
        overall_success = tool_result.success and postprocess_success

        # Extract token usage from preprocessing metadata
        token_usage = metadata.pop("llm_token_usage", {})

        # Merge remaining metadata with result metadata
        result_metadata = {
            "agent_type": "kopl",
            "full_data": tool_result.result_data,
            "processed_params": processed_params,
        }
        result_metadata.update(metadata)

        return ExecutionResult(
            executable_name=self.name,
            executable_type=ExecutableType.AGENT,
            success=overall_success,
            result_data=observation,  # Short text for Meta Agent
            error_message=error_message,
            execution_time=execution_time,
            token_usage=token_usage,
            metadata=result_metadata,
        )

    def validate_input(self, params: dict[str, Any]) -> None:
        """
        Validate input parameters.

        Args:
            params: Input parameters to validate

        Raises:
            AgentException: If validation fails
        """

        error_messages = []
        # Additional type checks could be added here
        for key, value in params.items():
            if key == "direction":
                if value not in {"forward", "backward"}:
                    error_messages.append(f"Invalid value for 'direction': {value}. Must be 'forward' or 'backward'.")
            if "entities" in key or "entity" in key:
                if self.kopl_factory and not self.kopl_factory.is_valid_entity_tuple(value):
                    error_messages.append(f"Invalid format for '{key}': {value}.")
            if "value" in key:
                if not isinstance(value, (str, int, float)):
                    error_messages.append(f"Invalid type for '{key}': {type(value)}. Must be str, int, or float.")
        if error_messages:
            raise AgentException(
                "\n".join(error_messages),
                agent_name=self.name,
            )

    def preprocess_input(
        self,
        params: dict[str, Any],
        parent_question: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Preprocess input parameters.

        Args:
            params: Input parameters from Meta Agent
            parent_question: Original user question for context (used for LLM-based grounding)

        Returns:
            Tuple of (processed_params, metadata)
        """
        return params, {}

    def execute_tool(
        self,
        processed_params: dict[str, Any],
        environment: "Environment",
        **kwargs,
    ) -> ExecutionResult:
        """
        Execute the KoPL tool with resolved parameters.

        Args:
            processed_params: Parameters with $i references resolved
            environment: Environment providing tool execution context
            **kwargs: Additional execution parameters

        Returns:
            ExecutionResult: Tool execution result
        """
        result = environment.execute_tool(self.tool_name, params=processed_params)
        return result

    def postprocess_output(
        self,
        tool_result: ExecutionResult,
        environment: "Environment",
        **kwargs,
    ) -> tuple[str, bool]:
        """
        Postprocess tool result to generate concise observation.

        Generates a short summary suitable for Meta Agent. The full data
        is preserved in ExecutionResult.metadata["full_data"] for Step storage.

        Args:
            tool_result: Raw result from KoPL tool (JSON string)
            environment: Environment providing context (unused)
            **kwargs: Additional postprocessing parameters (unused)

        Returns:
            tuple[str, bool]: Tuple of (observation_message, success_flag)
        """
        # Dispatch to operator-specific postprocessing
        if tool_result.executable_name in SCALAR_RESULT_OPERATORS:
            return self._postprocess_scalar_result(tool_result)
        return self._postprocess_entity_result(tool_result)

    def _postprocess_scalar_result(self, tool_result: ExecutionResult) -> tuple[str, bool]:
        """
        Handle operators that return scalar values.

        Args:
            tool_result: Tool execution result

        Returns:
            tuple[str, bool]: Tuple of (observation, success)
        """
        from ..kopl_utils import postprocess_kopl_answer

        success = True
        error_message_for_not_found = "No results found for the given input. Please try a different input/tool."
        try:
            parsed_data = postprocess_kopl_answer(tool_result.result_data)  # type: ignore
            if parsed_data is None:
                success = False
                return error_message_for_not_found, success
            return parsed_data, success
        except Exception as e:
            logger.error(f"Error in postprocessing scalar result: {str(e)}")
            success = False
            return error_message_for_not_found, success

    def _postprocess_entity_result(self, tool_result: ExecutionResult) -> tuple[str, bool]:
        """
        Handle operators that return entities/facts.

        Args:
            tool_result: Tool execution result

        Returns:
            tuple[str, bool]: Tuple of (observation, success)
        """
        observation_lines = []
        success = True

        # Parse the result to get entity count
        data = orjson.loads(tool_result.result_data)  # type: ignore
        entities, facts = [], None

        error_message_for_not_found = "No results found for the given input. Please try a different input/tool."

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict result for entity/fact operators, got: {type(data)}")
        # data = {"entities": [list of entity IDs], "facts": [list of tuples] | None}
        entities = data.get("entities", [])
        facts = data.get("facts", None)

        num_entities = len(entities)
        num_facts = len(facts) if facts is not None else 0

        if facts is None or len(facts) == 0:
            if len(entities) == 0:
                success = False
                return error_message_for_not_found, success
            else:
                observation_lines.append(f"Found {num_entities} entities.")
        else:
            if len(entities) == 0:
                observation_lines.append(f"Found {num_facts} facts.")
            else:
                observation_lines.append(f"Found {num_entities} entities and {num_facts} facts.")

        truncated_entities = entities[:MAX_ENTITY_LABELS_IN_OBSERVATION]
        if len(truncated_entities) > 0:
            entity_labels = self.kopl_factory.engine.QueryName((truncated_entities, None))  # type: ignore
            for entity_id, entity_label in zip(truncated_entities, entity_labels):
                observation_lines.append(f"- {entity_id} ({entity_label})")
            if num_entities > MAX_ENTITY_LABELS_IN_OBSERVATION:
                observation_lines.append("... [truncated]")

        return "\n".join(observation_lines), success


class KoPLFindFilterConceptAgent(KoPLAgent):
    """
    Specialized KoPL Agent for the "find" and "filter_concept" operators.

    Uses precomputed embeddings to disambiguate names (concept/entity) by finding similar candidates
    linked to input entities. Falls back to exact match if available.
    """

    def __init__(
        self,
        operator_name: str,
        llm_call: Optional[Callable] = None,
        llm_options: dict[str, Any] = {},
        tool_ids: Optional[list[str]] = None,
        update_schema_from_tool: bool = True,
        path_to_embeddings: str = "data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5/entity_embeddings.pkl",
        encoder_model_name: str = "BAAI/bge-base-en-v1.5",
        **kwargs,
    ):
        """
        Initialize the KoPL Find/Filter Concept Agent.

        Args:
            operator_name: Name of the KoPL operator ("find" or "filter_concept")
            llm_call: Callable function to get LLM responses (optional for preprocessing)
            llm_options: Options to configure LLM behavior
            tool_ids: List of tool IDs this agent can use (e.g., ["kopl/find"])
            update_schema_from_tool: Whether to update agent schema from tool definition
            path_to_embeddings: Path to precomputed embeddings
            encoder_model_name: Name of the SentenceTransformer model
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            operator_name=operator_name,
            llm_call=llm_call,
            llm_options=llm_options,
            tool_ids=tool_ids,
            update_schema_from_tool=update_schema_from_tool,
            **kwargs,
        )

        # Load precomputed embeddings (entity/concept names to vectors)
        self.path_to_embeddings = Path(path_to_embeddings)
        self.precomputed_embeddings = load_embeddings(self.path_to_embeddings)

        # Get shared encoder model (cached across all instances)
        self.encoder_model_name = encoder_model_name
        self.encoder_model = get_encoder_model(self.encoder_model_name)

    def preprocess_input(
        self,
        params: dict[str, Any],
        parent_question: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Preprocess input by disambiguating names using embeddings.

        For "find", uses the "name" field; for "filter_concept", uses "concept_name".
        Extracts candidate names from the KB, checks for exact match, and uses similarity search
        to select the top matching candidate if needed.

        Args:
            params: Input parameters including entities and name field
            parent_question: Original user question for context (used for LLM-based grounding)

        Returns:
            Tuple of (processed_params, metadata)

        Raises:
            AgentException: If JSON parsing fails or no candidates are found
        """
        metadata = {}
        key = "name" if self.operator_name == "find" else "concept_name"

        if not self.kopl_factory or not self.kopl_factory.engine:
            raise ValueError("KoPL factory not initialized. Factory must be injected by AgentRegistry.")

        kb = self.kopl_factory.engine.kb  # type: ignore
        if kb is None:
            raise ValueError("KoPL knowledge base not initialized in the tool factory engine")

        if self.operator_name == "find":
            # All entity/concept names in KB
            candidates = extract_entity_name_candidates(kb)
        else:  # filter_concept
            # Parse entities from JSON
            entities, _ = parse_entities_and_facts(params, ["entities_and_facts"], self.name)
            # Get all candidate names linked to entities
            candidates = extract_entity_name_candidates(kb, entities)

        # Hybrid matching (embedding + LLM)
        if len(candidates) == 0:
            if self.operator_name == "find":
                error_message = (
                    f"KB grounding failed. The given name '{params[key]}' does not match any entity in the KB."
                )
            else:
                error_message = f"KB grounding failed for input entities. The given concept name '{params[key]}' does not match any concept linked to the input entities."
            raise AgentException(error_message, agent_name=self.name)
        params[key] = self._schema_matching(
            query=params[key],
            candidates=candidates,
            precomputed_embeddings=self.precomputed_embeddings,
            encoder_model=self.encoder_model,
            question=parent_question,
            param_name=key,
            op_name=self.operator_name,
            metadata=metadata,
        )
        return params, metadata


class KoPLKeyOnlyAgent(KoPLAgent):
    """
    Specialized KoPL Agent for operators that use a single key parameter.

    Handles operators like "relate" (relation key) and "query_attr" (attribute key).
    Uses precomputed embeddings to disambiguate key names by finding similar keys
    from the attributes or relations of input entities.
    """

    def __init__(
        self,
        operator_name: str,
        llm_call: Optional[Callable] = None,
        llm_options: dict[str, Any] = {},
        tool_ids: Optional[list[str]] = None,
        update_schema_from_tool: bool = True,
        path_to_embeddings: str = "data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5/key_embeddings.pkl",
        encoder_model_name: str = "BAAI/bge-base-en-v1.5",
        **kwargs,
    ):
        """
        Initialize the KoPL Key-Only Agent.

        Args:
            operator_name: Name of the KoPL operator (e.g., "relate", "query_attr")
            llm_call: Callable function to get LLM responses (optional for preprocessing)
            llm_options: Options to configure LLM behavior
            tool_ids: List of tool IDs this agent can use
            update_schema_from_tool: Whether to update agent schema from tool definition
            path_to_embeddings: Path to precomputed key embeddings (Pickle file)
            encoder_model_name: Name of the SentenceTransformer model
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            operator_name=operator_name,
            llm_call=llm_call,
            llm_options=llm_options,
            tool_ids=tool_ids,
            update_schema_from_tool=update_schema_from_tool,
            **kwargs,
        )

        # Load precomputed key embeddings
        self.path_to_embeddings = Path(path_to_embeddings)
        self.precomputed_embeddings = load_embeddings(self.path_to_embeddings)

        # Get shared encoder model (cached across all instances)
        self.encoder_model_name = encoder_model_name
        self.encoder_model = get_encoder_model(self.encoder_model_name)

    def preprocess_input(
        self,
        params: dict[str, Any],
        parent_question: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Preprocess input by disambiguating key names using embeddings.

        For "relate" operator, disambiguates the "relation" parameter by extracting
        relation keys from input entities. For other operators (e.g., "query_attr"),
        disambiguates the "key" parameter by extracting attribute keys.

        Args:
            params: Input parameters including entities and key parameter
            parent_question: Original user question for context

        Returns:
            Tuple of (processed_params, metadata)

        Raises:
            AgentException: If JSON parsing fails or no candidates are found
        """
        metadata = {}
        # Parse entities from JSON (handles multiple entity fields for operators like "and"/"or")
        entities, _ = parse_entities_and_facts(
            params,
            ["entities_and_facts", "l_entities_and_facts", "r_entities_and_facts"],
            self.name,
        )

        # Retrieve KB and extract candidate keys from entities
        if not self.kopl_factory or not self.kopl_factory.engine:
            raise ValueError("KoPL factory not initialized. Factory must be injected by AgentRegistry.")

        kb = self.kopl_factory.engine.kb  # type: ignore
        if kb is None:
            raise ValueError("KoPL knowledge base not initialized in the tool factory engine")

        key = "relation" if self.operator_name == "relate" else "key"
        if self.operator_name == "relate":
            candidates = extract_relation_key_candidates(kb, entities)
        else:
            candidates = extract_attribute_key_candidates(kb, entities)

        if len(candidates) == 0:
            if self.operator_name == "relate":
                error_message = f"KB grounding failed for input entities. The given relation '{params[key]}' does not match any relation linked to the input entities."
            else:
                error_message = f"KB grounding failed for input entities. The given key '{params[key]}' does not match any attribute available for the input entities with '{self.operator_name}'."
            raise AgentException(error_message, agent_name=self.name)

        # Hybrid matching (embedding + LLM)
        params[key] = self._schema_matching(
            query=params[key],
            candidates=candidates,
            precomputed_embeddings=self.precomputed_embeddings,
            encoder_model=self.encoder_model,
            question=parent_question,
            param_name=key,
            op_name=self.operator_name,
            metadata=metadata,
        )
        return params, metadata


class KoPLKeyValueAgent(KoPLAgent):
    """
    Specialized KoPL Agent for operators with key and/or value parameters.

    Handles operators like "filter_str" (key + value), "query_relation" (relation key),
    "qfilter_str" (qualifier key + value), and "query_relation_qualifier" (relation + qkey).
    Uses precomputed embeddings to disambiguate keys and values by finding similar ones
    from the attributes, relations, and qualifiers of input entities.
    """

    def __init__(
        self,
        operator_name: str,
        llm_call: Optional[Callable] = None,
        llm_options: dict[str, Any] = {},
        tool_ids: Optional[list[str]] = None,
        update_schema_from_tool: bool = True,
        path_to_key_embeddings: str = "data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5/key_embeddings.pkl",
        path_to_value_embeddings: str = "data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5/value_embeddings.pkl",
        encoder_model_name: str = "BAAI/bge-base-en-v1.5",
        **kwargs,
    ):
        """
        Initialize the KoPL Key-Value Agent.

        Args:
            operator_name: Name of the KoPL operator (e.g., "filter_str", "query_relation", "qfilter_str")
            llm_call: Callable function to get LLM responses (optional for preprocessing)
            llm_options: Options to configure LLM behavior
            tool_ids: List of tool IDs this agent can use
            update_schema_from_tool: Whether to update agent schema from tool definition
            path_to_key_embeddings: Path to precomputed key embeddings (Pickle file)
            path_to_value_embeddings: Path to precomputed value embeddings (Pickle file)
            encoder_model_name: Name of the SentenceTransformer model
            **kwargs: Additional configuration parameters
        """
        super().__init__(
            operator_name=operator_name,
            llm_call=llm_call,
            llm_options=llm_options,
            tool_ids=tool_ids,
            update_schema_from_tool=update_schema_from_tool,
            **kwargs,
        )

        # Load precomputed key and value embeddings
        self.path_to_key_embeddings = Path(path_to_key_embeddings)
        self.precomputed_key_embeddings = load_embeddings(self.path_to_key_embeddings)

        self.path_to_value_embeddings = Path(path_to_value_embeddings)
        self.precomputed_value_embeddings = load_embeddings(self.path_to_value_embeddings)

        # Get shared encoder model (cached across all instances)
        self.encoder_model_name = encoder_model_name
        self.encoder_model = get_encoder_model(self.encoder_model_name)

    def preprocess_input(
        self,
        params: dict[str, Any],
        parent_question: str = "",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Preprocess input by disambiguating keys and values using embeddings.

        Handles multiple parameter combinations:
        - "key" + optional "value" for attribute-based operators (filter_*, query_attr_under_condition)
        - "relation" for relation-based operators (query_relation, query_relation_qualifier)
        - "qkey" + optional "qvalue" for qualifier-based operators (qfilter_*, query_attr_qualifier)

        Args:
            params: Input parameters including entities_and_facts and key/value parameters
            parent_question: Original user question for context

        Returns:
            Tuple of (processed_params, metadata)

        Raises:
            AgentException: If JSON parsing fails or no candidates are found
        """
        metadata = {}
        # Parse entities and facts from JSON input
        field_names = ["entities_and_facts"]
        # Special handling for query_relation_qualifier operator
        if "s_entities_and_facts" in params:
            field_names = ["s_entities_and_facts"]

        entities, facts = parse_entities_and_facts(params, field_names, self.name)

        # Load KB
        if not self.kopl_factory or not self.kopl_factory.engine:
            raise ValueError("KoPL factory not initialized. Factory must be injected by AgentRegistry.")

        kb = self.kopl_factory.engine.kb  # type: ignore
        if kb is None:
            raise ValueError("KoPL knowledge base not initialized in the tool factory engine")

        # Determine expected value types for filtering operators
        value_type = (
            None if self.operator_name.startswith("qfilter") else OPERATOR_TO_VALUE_TYPE.get(self.operator_name)
        )
        qvalue_type = (
            OPERATOR_TO_VALUE_TYPE.get(self.operator_name) if self.operator_name.startswith("qfilter") else None
        )

        # Disambiguate attribute key if present
        if "key" in params:
            # Extract candidate keys from entity attributes
            key_candidates = extract_attribute_key_candidates(kb, entities, value_type)

            if len(key_candidates) == 0:
                error_msg = f"The input entities do not have any attributes of the expected type for '{self.operator_name}'. Please try a different input/tool."
                raise AgentException(
                    error_msg,
                )

            # Hybrid matching for key (embedding + LLM)
            params["key"] = self._schema_matching(
                query=params["key"],
                candidates=key_candidates,
                precomputed_embeddings=self.precomputed_key_embeddings,
                encoder_model=self.encoder_model,
                question=parent_question,
                param_name="key",
                op_name=self.operator_name,
                metadata=metadata,
            )

            # For filter_str operator, also disambiguate the value parameter
            if self.operator_name == "filter_str" and "value" in params:
                # Get all values for the disambiguated key
                value_candidates = set()
                if params["key"] in kb.key_values:
                    value_candidates = set([v.value for v in kb.key_values[params["key"]]])

                if len(value_candidates) == 0:
                    error_msg = f"The attribute key '{params['key']}' does not have any associated values in the KB for '{self.operator_name}'. Please try a different input/tool."
                    raise AgentException(error_msg)

                # Disambiguate the value parameter (embedding-only matching)
                params["value"] = self._schema_matching(
                    query=params["value"],
                    candidates=value_candidates,
                    precomputed_embeddings=self.precomputed_value_embeddings,
                    encoder_model=self.encoder_model,
                    question=parent_question,
                    param_name="value",
                    op_name=self.operator_name,
                    use_llm_matching=False,
                )

        # Disambiguate relation key if present
        if "relation" in params:
            # Extract candidate relations from entity relations
            relation_candidates = extract_relation_key_candidates(kb, entities)

            if len(relation_candidates) == 0:
                error_msg = f"The input entities do not have any relations for '{self.operator_name}'. Please try a different input/tool."
                raise AgentException(error_msg)

            # Hybrid matching for relation (embedding + LLM)
            params["relation"] = self._schema_matching(
                query=params["relation"],
                candidates=relation_candidates,
                precomputed_embeddings=self.precomputed_key_embeddings,
                encoder_model=self.encoder_model,
                question=parent_question,
                param_name="relation",
                metadata=metadata,
                op_name=self.operator_name,
            )

        # Disambiguate qualifier key if present
        if "qkey" in params:
            # Extract candidate qualifier keys from facts or entity attributes
            qkey_candidates = extract_qualifier_key_candidates(
                kb,
                entities,
                is_attribute=("key" in params),
                facts=facts,
                value_type=qvalue_type,
            )

            if len(qkey_candidates) == 0:
                error_msg = f"The input entities do not have any qualifiers of the expected type for '{self.operator_name}'. Please try a different input/tool."
                raise AgentException(error_msg)

            # Hybrid matching for qkey (embedding + LLM)
            params["qkey"] = self._schema_matching(
                query=params["qkey"],
                candidates=qkey_candidates,
                precomputed_embeddings=self.precomputed_key_embeddings,
                encoder_model=self.encoder_model,
                question=parent_question,
                param_name="qkey",
                op_name=self.operator_name,
                metadata=metadata,
            )

            # For qfilter_str operator, also disambiguate the qvalue parameter
            if self.operator_name == "qfilter_str" and "qvalue" in params:
                # Extract candidate qualifier values for the disambiguated qkey
                qvalue_candidates = []
                if facts:
                    for fact in facts:
                        qualifiers = fact.get("qualifiers", {})
                        for qval in qualifiers.get(params["qkey"], []):
                            if _matches_value_type(qval, {"string"}):
                                if isinstance(qval, ValueClass):
                                    qvalue_candidates.append(qval.value)
                                else:
                                    qvalue_candidates.append(qval["value"])
                else:
                    # Extract from entity attributes and relations
                    for ent_id in entities:
                        entity = kb.entities[ent_id]
                        for attr in entity["attributes"]:
                            qualifiers = attr.get("qualifiers", {})
                            for qval in qualifiers.get(params["qkey"], []):
                                if _matches_value_type(qval, {"string"}):
                                    if isinstance(qval, ValueClass):
                                        qvalue_candidates.append(qval.value)
                                    else:
                                        qvalue_candidates.append(qval["value"])
                        for rel in entity["relations"]:
                            qualifiers = rel.get("qualifiers", {})
                            for qval in qualifiers.get(params["qkey"], []):
                                if _matches_value_type(qval, {"string"}):
                                    if isinstance(qval, ValueClass):
                                        qvalue_candidates.append(qval.value)
                                    else:
                                        qvalue_candidates.append(qval["value"])

                qvalue_candidates = set(qvalue_candidates)

                if len(qvalue_candidates) == 0:
                    error_msg = f"The qualifier key '{params['qkey']}' does not have any associated qualifier values of the expected type in the KB for '{self.operator_name}'. Please try a different input/tool."
                    raise AgentException(error_msg)

                # Disambiguate the qvalue parameter (embedding-only matching)
                params["qvalue"] = self._schema_matching(
                    query=params["qvalue"],
                    candidates=qvalue_candidates,
                    precomputed_embeddings=self.precomputed_value_embeddings,
                    encoder_model=self.encoder_model,
                    question=parent_question,
                    param_name="qvalue",
                    op_name=self.operator_name,
                    use_llm_matching=False,
                )
        return params, metadata


class KoPLAgentFactory:
    """
    Factory for creating KoPL operator-specific Worker Agents.

    Simplified factory that creates agents without tool coupling.
    Tools are managed separately by the ToolRegistry.
    """

    KOPL_AGENT_CLASSES = {
        "kopl_agent": KoPLAgent,
        "kopl_find_and_filter_concept_agent": KoPLFindFilterConceptAgent,
        "kopl_key_only_agent": KoPLKeyOnlyAgent,
        "kopl_key_and_value_agent": KoPLKeyValueAgent,
    }

    # Default paths for embeddings
    DEFAULT_EMBEDDINGS_DIR = "data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5"
    DEFAULT_ENCODER_MODEL = "BAAI/bge-base-en-v1.5"

    @staticmethod
    def create_agent(
        agent_type: str,
        operator_name: str,
        llm_call: Optional[Callable] = None,
        llm_options: dict[str, Any] = {},
        tool_ids: Optional[list[str]] = None,
        embeddings_dir: Optional[str] = None,
        encoder_model_name: Optional[str] = None,
        strict_mode: bool = False,
        **kwargs,
    ) -> KoPLAgent:
        """
        Create a KoPL agent for specific operator.

        Args:
            agent_type: Type of agent to create
            operator_name: Name of the KoPL operator
            llm_call: Callable function to get LLM responses
            llm_options: Options to configure LLM behavior
            tool_ids: List of tool IDs this agent can use (e.g., ["kopl/find_all"])
            embeddings_dir: Directory containing embedding files (for specialized agents)
            encoder_model_name: Name of encoder model for similarity matching
            strict_mode: If True, require exact matches for schema grounding
            **kwargs: Additional configuration parameters

        Returns:
            KoPLAgent: Configured agent instance for the specified operator

        Raises:
            ValueError: If operator_name or agent_type is not supported
        """
        all_operators = KoPLAgentFactory.get_all_operators()

        if operator_name not in all_operators:
            raise ValueError(f"Unknown KoPL operator: {operator_name}. Supported operators: {', '.join(all_operators)}")
        if agent_type not in KoPLAgentFactory.KOPL_AGENT_CLASSES:
            raise ValueError(
                f"Unknown KoPL agent type: {agent_type}. "
                f"Supported types: {', '.join(KoPLAgentFactory.KOPL_AGENT_CLASSES.keys())}"
            )
        agent_class = KoPLAgentFactory.KOPL_AGENT_CLASSES[agent_type]

        # Set default values
        if embeddings_dir is None:
            embeddings_dir = KoPLAgentFactory.DEFAULT_EMBEDDINGS_DIR
        if encoder_model_name is None:
            encoder_model_name = KoPLAgentFactory.DEFAULT_ENCODER_MODEL

        # Build agent-specific kwargs based on agent type
        agent_kwargs = {
            "operator_name": operator_name,
            "llm_call": llm_call,
            "llm_options": llm_options,
            "tool_ids": tool_ids,
            "update_schema_from_tool": True,
            "strict_mode": strict_mode,
        }

        # Add embedding paths for specialized agents
        if agent_type == "kopl_find_and_filter_concept_agent":
            agent_kwargs["path_to_embeddings"] = f"{embeddings_dir}/entity_embeddings.pkl"
            agent_kwargs["encoder_model_name"] = encoder_model_name
        elif agent_type == "kopl_key_only_agent":
            agent_kwargs["path_to_embeddings"] = f"{embeddings_dir}/key_embeddings.pkl"
            agent_kwargs["encoder_model_name"] = encoder_model_name
        elif agent_type == "kopl_key_and_value_agent":
            agent_kwargs["path_to_key_embeddings"] = f"{embeddings_dir}/key_embeddings.pkl"
            agent_kwargs["path_to_value_embeddings"] = f"{embeddings_dir}/value_embeddings.pkl"
            agent_kwargs["encoder_model_name"] = encoder_model_name

        # Merge additional kwargs
        agent_kwargs.update(kwargs)

        return agent_class(**agent_kwargs)

    @staticmethod
    def get_all_operators() -> list[str]:
        """
        Return list of all supported KoPL operators.

        Returns:
            list[str]: List of operator names
        """
        return [
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
