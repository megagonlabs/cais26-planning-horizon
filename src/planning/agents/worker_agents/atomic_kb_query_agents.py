"""
Atomic KB Query worker agents for compositional KBQA operations.

This module implements worker agents that perform single atomic operations used
by higher-level meta agents (e.g., START, JOIN, AND, ARGMAX/ARGMIN, CMP, TC, COUNT).
Each worker:
- Implements preprocess(...) to ground LLM parameters (e.g., link entities/relations).
- Calls a tool via the Environment to execute the operation.
- Implements postprocess(...) to convert tool results into observation strings and
  structured output used by meta agents.

Shared resources (schema lists, embedding model) are cached in a thread-safe way.
These    agents are dataset-agnostic: dataset configuration is supplied at runtime
via task_config passed to run_episode.
"""

from pathlib import Path
from time import time
from typing import Any, Callable, Optional, TYPE_CHECKING
import copy
import json
import logging
import re
import threading

import numpy.random as np_random

# Lazy imports for heavy ML libraries
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

import numpy as np

from ...environment.environment import Environment
from ..base_agent import BaseWorkerAgent
from ..exceptions import AgentException
from ..executable import ExecutionResult, ExecutableType
from ..memory import ContextMemory
from ..step import Step, StepStatus

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_EMBEDDINGS_DIR = Path("data/atomic_kbqa/embeddings/BAAI___bge-base-en-v1.5")
DEFAULT_ENCODER_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_SCHEMA_RESOURCES_PATH = Path("data/atomic_kbqa/freebase")
MAX_ENTITY_LABELS_IN_OBSERVATION = 10  # Maximum number of entity labels to show in observation text
MAX_ENTITIES_TO_QUERY = 100000

# Thread-safe cache for embedding model (model_name -> SentenceTransformer)
_ENCODER_MODEL_CACHE: dict[str, "SentenceTransformer"] = {}
_ENCODER_MODEL_CACHE_LOCK = threading.Lock()

# Thread-safe cache for schema resources (path -> dict)
_SCHEMA_RESOURCES_CACHE: dict[str, dict] = {}
_SCHEMA_RESOURCES_CACHE_LOCK = threading.Lock()

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


def format_entity_observation(entity_ids: list[str]) -> str:
    """Format entity observation with labels.

    Args:
        entity_ids: List of entity IDs (non-empty).

    Returns:
        Formatted observation string with entity count and labels.
        Format: "Found {n} results\n- {id1} ({label1})\n- {id2} ({label2})\n..."
    """
    from ...tools.freebase.sparql_executor import get_label_with_odbc

    # Truncate entity IDs before label fetching for efficiency
    num_entities = len(entity_ids)
    truncated_ids = entity_ids[:MAX_ENTITY_LABELS_IN_OBSERVATION]

    # Fetch labels for truncated entities
    observation_lines = [f"Found {num_entities} results"]
    for entity_id in truncated_ids:
        label = get_label_with_odbc(entity_id)
        observation_lines.append(f"- {entity_id} ({label})")
    if num_entities > MAX_ENTITY_LABELS_IN_OBSERVATION:
        observation_lines.append("... [truncated]")

    return "\n".join(observation_lines)


def get_encoder_model(model_name: str) -> "SentenceTransformer":
    """
    Get or create a shared encoder model with thread-safe caching.

    Args:
        model_name: Name of the SentenceTransformer model (e.g., "BAAI/bge-base-en-v1.5")

    Returns:
        Shared SentenceTransformer model instance

    Raises:
        ImportError: If sentence-transformers is not installed
    """
    # Fast path: check without lock
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


def load_schema_resources(schema_path: Path, embedding_model: "SentenceTransformer") -> dict:
    """
    Load schema resources (relation lists, etc.) with thread-safe caching and embedding precomputation.

    Args:
        schema_path: Path to schema resources directory
        embedding_model: SentenceTransformer model for precomputing relation embeddings

    Returns:
        Dict with schema resources:
            - relation_list: All valid Freebase relations
            - join_ban_relation_list: Relations to exclude from JOIN
            - literal_relation_list: Relations for numeric operations
            - literal_relation_list_sorted: Sorted list matching literal_relation_embeddings order
            - literal_relation_embeddings: Precomputed embeddings for literal relations (numpy array)
            - name_relation_list: Relations for name entities
            - tc_time_list: Valid time constraint values

    Raises:
        FileNotFoundError: If schema files are missing
    """
    path_str = str(schema_path.resolve())
    cache_key = f"{path_str}:{id(embedding_model)}"

    # Fast path: check without lock
    resources = _SCHEMA_RESOURCES_CACHE.get(cache_key)
    if resources is not None:
        return resources

    # Lock for thread-safe write
    with _SCHEMA_RESOURCES_CACHE_LOCK:
        # Double-check in case another thread loaded it
        resources = _SCHEMA_RESOURCES_CACHE.get(cache_key)
        if resources is not None:
            return resources

        # Load relation lists from vendor code structure
        # These lists are imported from reasoners/kbqa/limit.py in the vendor code
        relation_list_file = schema_path / "relation_list.txt"
        join_ban_file = schema_path / "join_ban_relation_list.txt"
        literal_relation_file = schema_path / "literal_relation_list.txt"
        name_relation_file = schema_path / "name_relation_list.txt"
        tc_time_file = schema_path / "tc_time_list.txt"

        resources = {}

        # Load relation_list (all valid Freebase relations)
        if relation_list_file.exists():
            with open(relation_list_file) as f:
                resources["relation_list"] = set(line.strip() for line in f if line.strip())
        else:
            logger.warning(f"relation_list.txt not found at {relation_list_file}")
            resources["relation_list"] = set()

        # Load join_ban_relation_list (relations to exclude from JOIN)
        if join_ban_file.exists():
            with open(join_ban_file) as f:
                resources["join_ban_relation_list"] = set(line.strip() for line in f if line.strip())
        else:
            logger.warning(f"join_ban_relation_list.txt not found at {join_ban_file}")
            resources["join_ban_relation_list"] = set()

        # Load literal_relation_list (relations for numeric operations)
        if literal_relation_file.exists():
            with open(literal_relation_file) as f:
                literal_relations_set = set(line.strip() for line in f if line.strip())
                literal_relations_sorted = sorted(list(literal_relations_set))
                resources["literal_relation_list"] = literal_relations_set
                resources["literal_relation_list_sorted"] = literal_relations_sorted

                # Precompute embeddings for all literal relations
                start_embed_time = time()
                literal_embeddings = embedding_model.encode(literal_relations_sorted, normalize_embeddings=True)
                embed_time = time() - start_embed_time
                resources["literal_relation_embeddings"] = literal_embeddings
                logger.info(
                    f"Precomputed embeddings for {len(literal_relations_sorted)} literal relations "
                    f"(dim={literal_embeddings.shape[1]}, time={embed_time:.2f}s)"
                )
        else:
            logger.warning(f"literal_relation_list.txt not found at {literal_relation_file}")
            resources["literal_relation_list"] = set()
            resources["literal_relation_list_sorted"] = []
            resources["literal_relation_embeddings"] = np.array([])

        # Load name_relation_list (relations for name entities)
        if name_relation_file.exists():
            with open(name_relation_file) as f:
                resources["name_relation_list"] = set(line.strip() for line in f if line.strip())
        else:
            logger.warning(f"name_relation_list.txt not found at {name_relation_file}")
            resources["name_relation_list"] = set()

        # Load tc_time_list (valid time constraint values)
        if tc_time_file.exists():
            with open(tc_time_file) as f:
                resources["tc_time_list"] = set(line.strip() for line in f if line.strip())
        else:
            logger.warning(f"tc_time_list.txt not found at {tc_time_file}")
            resources["tc_time_list"] = set()

        _SCHEMA_RESOURCES_CACHE[cache_key] = resources
        logger.info(f"Loaded schema resources from {schema_path}")
        return resources


class AtomicKBQueryWorkerAgent(BaseWorkerAgent):
    """
    Base class for atomic KB query worker agents.

    Responsibilities:
    - Provide common utilities for grounding parameters (schema_resources,
      embedding_model) and managing function lists.
    - Define the expected lifecycle: preprocess -> execute tool -> postprocess.
    - Implement run_episode(...) that executes the above sequence and returns an
      ExecutionResult. The ExecutionResult.metadata includes an entry
      'full_data' (result), 'function_list' (current functions),
      and 'processed_params' (grounded parameters).

    Subclasses must implement:
    - preprocess(raw_params, task_config, memory, **kwargs) -> (grounded_params, metadata)
      (ground parameters and produce worker metadata)
    - postprocess(tool_result, grounded_params, task_config=None, metadata=None, **kwargs)
      -> (output_data, observation_str)

    Notes:
    - These agents always delegate actual KB operations to tools via Environment.
    - They are stateless with respect to dataset-specific state; dataset info
      is provided through task_config at runtime.
    """

    def __init__(
        self,
        llm_call: Callable,
        llm_options: Optional[dict[str, Any]] = None,
        name: str = "AtomicKBQueryWorkerAgent",
        tool_id: str = "",
        schema_resources: Optional[dict] = None,
        embedding_model: Optional["SentenceTransformer"] = None,
        retrieval_topk: int = 10,
        strict_mode: bool = False,
    ):
        """
        Initialize Atomic KB Query worker agent.

        Args:
            llm_call: Callable function to get LLM responses (not used for these agents)
            llm_options: Options to configure LLM behavior (not used for these agents)
            name: Name identifier for the agent
            tool_id: ID of the tool this agent wraps
            schema_resources: Shared schema resources (relation lists, etc.)
            embedding_model: Shared embedding model for schema grounding
            strict_mode: If True, require exact matches for schema grounding
        """
        from ...tools.freebase.sparql_executor import close_thread_connection

        if llm_options is None:
            llm_options = {}

        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            max_steps=1,  # Worker agents execute exactly one operation
            tool_ids=[tool_id] if tool_id else [],
            update_schema_from_tool=True,
            cleanup_func=close_thread_connection,
        )

        self.tool_id = tool_id
        self.operator_name = tool_id.split("/")[-1] if tool_id else "unknown_operator"
        self.schema_resources = schema_resources or {}
        self.embedding_model = embedding_model
        self.retrieval_topk = retrieval_topk
        self.strict_mode = strict_mode
        self.rng = np_random.default_rng(42)  # Deterministic RNG for sampling

    def supports_pure_reasoning(self) -> bool:
        """
        Check if this agent can work without tools.

        Returns:
            False - Atomic KB Query agents always require tools
        """
        return False

    def _get_function_list_from_context(self, params: dict[str, Any], memory: ContextMemory) -> list[str]:
        """
        Extract function_list from execution context.

        Args:
            params: Parameters from LLM or MetaAgent
            memory: ContextMemory instance for retrieving step metadata

        Returns:
            List of function strings (may be empty for first operation)
        """
        function_list: list[str] = []
        for key, val in params.items():
            if not isinstance(val, str):
                continue
            if m := re.match(r"\$(\d+)", val):
                step_index = int(m.group(1))
                step = memory.get_step(step_index)
                function_list += step.get_metadata("function_list")

        if len(function_list) > 0:
            function_list = sorted(list(set(function_list)))  # Deduplicate and sort

        return function_list

    def _llm_schema_matching(
        self,
        question: str,
        query_relation: str,
        candidates: list[str],
        **kwargs,
    ) -> tuple[str | None, dict[str, Any]]:
        """
        Use LLM to match query relation to KB schema candidates.

        Args:
            question: The original question for context
            query_relation: The relation description to ground
            candidates: List of candidate KB relations
            **kwargs: Additional context

        Returns:
            Tuple of (matched_relation_or_none, metadata_dict)
            - matched_relation: The selected candidate, or None if LLM returns "none"
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
            query=query_relation,
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
        matched_relation = None if answer == "none" else answer

        metadata = {
            "llm_reasoning": reasoning,
            "llm_token_usage": token_usage,
        }

        return matched_relation, metadata

    def preprocess(
        self,
        raw_params: dict,
        task_config: dict[str, Any],
        parent_memory: ContextMemory,
        **kwargs,
    ) -> tuple[dict, dict]:
        """
        Preprocess parameters from LLM.

        By default, this method extracts the function_list from context.

        Args:
            raw_params: Parameters from LLM or MetaAgent
            task_config: Task-specific configuration (contains dataset name, entity candidates, etc.)
            parent_memory: ContextMemory instance for retrieving step metadata
            **kwargs: Hidden context (not used)

        Returns:
            tuple of (grounded parameters, metadata)
        """
        processed_params = copy.deepcopy(raw_params)
        metadata = {}

        # Extract function_list from the parent memory using step references ($0, $1, etc.)
        function_list = self._get_function_list_from_context(params=raw_params, memory=parent_memory)
        processed_params["function_list"] = function_list

        return processed_params, metadata

    def postprocess(
        self,
        tool_result: dict,
        grounded_params: dict,
        task_config: dict[str, Any],
        metadata: dict[str, Any],
        **kwargs,
    ) -> tuple[Any, str]:
        """
        Postprocess tool results into output data and observation string.

        Subclasses should override for operation-specific formatting.

        Args:
            tool_result: Results from tool execution
            grounded_params: Grounded parameters used for tool execution
            task_config: Task-specific configuration
            metadata: Metadata from the worker
            **kwargs: Hidden context

        Returns:
            Tuple of (output data, observation string for LLM)
        """
        raise NotImplementedError("Subclasses must implement postprocess method")

    def run_episode(
        self,
        query_or_params: str | dict[str, Any],
        environment: "Environment",
        task_config: dict[str, Any],
        session_id: str,
        parent_session_id: str,
        memory: Optional["ContextMemory"] = None,
        step_index: Optional[int] = None,
        **kwargs,
    ) -> "ExecutionResult":
        """
        Execute the atomic KB query operation.

        This method implements the BaseAgent.run_episode interface and performs
        the complete agent lifecycle for atomic KB query operations:
        1) schema grounding/preprocessing
        2) tool execution via the Environment
        3) postprocessing to format the observation
        The agent returns an ExecutionResult containing the observation and
        a 'full_data' metadata field with grounded parameters and tool results.

        Args:
            query_or_params: Parameters from MetaAgent (expected to be dict)
            environment: Environment providing tools and state
            task_config: Task-specific configuration
            session_id: Optional session ID (unused)
            parent_session_id: Optional parent session ID for memory resolution
            memory: Context memory for storing execution steps and intermediate results
            **kwargs: Additional parameters (previous_full, function_list, etc.)

        Returns:
            ExecutionResult from the execute method
        """
        if not isinstance(query_or_params, dict):
            raise ValueError("AtomicKBQueryWorkerAgent expects dict parameters from MetaAgent")
        if step_index is None:
            raise ValueError("step_index must be provided for AtomicKBQueryWorkerAgent")

        start_time = time()
        episode_metadata: dict[str, Any] = {}

        parent_memory = environment.get_memory(parent_session_id)
        assert parent_memory is not None, "Parent memory must be available"

        query_or_params["step_index"] = step_index

        if memory is None:
            memory = ContextMemory(
                session_id=session_id,
                agent_name=self.name,
                query="",
                parent_session_id=parent_session_id,
            )
            environment.register_memory(memory)

        dataset_name = task_config.get("dataset") if task_config else None

        try:
            # Step 1: Preprocess
            step_num = len(memory.step_history)
            preprocess_step = Step(
                step_num=step_num,
                step_type=f"{self.name.lower()}_preprocess",
                status=StepStatus.PLANNED,
                data={"input": query_or_params},
            )
            memory.add_step(preprocess_step)

            try:
                processed_params, preprocessing_metadata = self.preprocess(
                    query_or_params,
                    task_config=task_config,
                    parent_memory=parent_memory,
                    dataset_name=dataset_name,
                    **kwargs,
                )
            except AgentException as e:
                preprocess_step.status = StepStatus.FAILED
                raise e

            preprocess_step.status = StepStatus.COMPLETED
            preprocess_step.data["output"] = processed_params
            preprocess_step.metadata = preprocessing_metadata
            episode_metadata.update(preprocessing_metadata)

            processed_params["dataset_config"] = {"dataset": dataset_name}

            # Step 2: Execute tool
            step_num = len(memory.step_history)
            execute_step = Step(
                step_num=step_num,
                step_type=f"{self.name.lower()}_execute",
                status=StepStatus.PLANNED,
                data={"input": processed_params},
            )
            memory.add_step(execute_step)

            tool_execution_result = environment.execute_tool(self.tool_id, processed_params)
            execute_step.metadata = tool_execution_result.metadata
            if not tool_execution_result.success:
                execute_step.status = StepStatus.FAILED
                raise AgentException(f"Tool '{self.tool_id}' execution failed: {tool_execution_result.error_message}")

            execute_step.status = StepStatus.COMPLETED
            execute_step.data["output"] = tool_execution_result.result_data

            tool_result = tool_execution_result.result_data
            assert tool_result is not None

            # Step 3: Postprocess
            step_num = len(memory.step_history)
            postprocess_step = Step(
                step_num=step_num,
                step_type=f"{self.name.lower()}_postprocess",
                status=StepStatus.PLANNED,
                data={"input": tool_result},
            )
            memory.add_step(postprocess_step)

            try:
                output, observation = self.postprocess(
                    tool_result,
                    processed_params,
                    task_config=task_config,
                    metadata=episode_metadata,
                    **kwargs,
                )
            except AgentException as e:
                postprocess_step.status = StepStatus.FAILED
                raise e

            postprocess_step.status = StepStatus.COMPLETED
            postprocess_step.data["output"] = output
            postprocess_step.metadata = {"observation": observation}

        except AgentException as e:
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
                metadata={
                    "agent_type": "atomic_kb_query",
                },
            )

        except Exception:
            # Unexpected error (developer must fix)
            logger.exception(f"Unexpected error in {self.name}")
            raise

        execution_time = time() - start_time
        return ExecutionResult(
            executable_name=self.name,
            executable_type=ExecutableType.AGENT,
            success=True,
            result_data=observation,
            execution_time=execution_time,
            token_usage={},
            metadata={
                "agent_type": "atomic_kb_query",
                "full_data": output,
                "function_list": tool_result["function_list"],
                "processed_params": processed_params,
            },
        )


class ExtractEntityAgent(AtomicKBQueryWorkerAgent):
    """
    Worker agent for entity linking / extraction (START operation).

    Key points:
    - Expects input parameter 'input_value' (string mention from the question).
    - Uses task_config['entities'] (list of (name, id) tuples) as candidate pool.
    - If multiple candidates are present, uses the shared embedding_model to rank
      candidates and selects the top match. Top candidates are added to metadata
      under 'top_candidates'.
    - postprocess returns (tool_results, observation_str) and the ExecutionResult
      metadata includes 'full_data' as the structured tool output.

    Error handling:
    - Raises AgentException for missing 'input_value' or failed grounding.
    """

    def __init__(
        self,
        llm_call: Callable,
        llm_options: dict[str, Any],
        name: str = "extract-entity-agent",
        tool_id: str = "atomic_kb_query/extract_entity",
        schema_resources: Optional[dict] = None,
        embedding_model: Optional["SentenceTransformer"] = None,
        retrieval_topk: int = 10,
        strict_mode: bool = False,
        **kwargs,
    ):
        """
        Initialize ExtractEntity agent.

        Args:
            llm_call: LLM call function (not used)
            llm_options: LLM options (not used)
            name: Agent name
            tool_id: Tool ID for extract_entity tool
            schema_resources: Shared schema resources
            embedding_model: Shared embedding model
            retrieval_topk: Maximum number of entity candidates to consider
            strict_mode: If True, require exact matches for schema grounding
        """
        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            tool_id=tool_id,
            schema_resources=schema_resources,
            embedding_model=embedding_model,
            retrieval_topk=retrieval_topk,
            strict_mode=strict_mode,
        )

    def preprocess(
        self,
        raw_params: dict,
        task_config: dict[str, Any],
        parent_memory: ContextMemory,
        min_similarity: float = 0.8,
        **unused_kwargs,
    ) -> tuple[dict, dict]:
        """
        Preprocess entity name by linking to KB entity.

        Args:
            raw_params: {"input_value": "Barack Obama"}
            task_config: Task configuration with entity_candidates
            parent_memory: ContextMemory (not used)
            min_similarity: Minimum similarity threshold for linking
            **unused_kwargs: Hidden context (not used)

        Returns:
            tuple of (grounded parameters, metadata)

        Raises:
            AgentException: If no entity candidates available or linking fails
        """
        processed_params = copy.deepcopy(raw_params)
        processed_params["function_list"] = []  # Initialize function_list
        metadata = {}

        input_value = raw_params.get("input_value", "")

        if not input_value:
            raise AgentException("'input_value' is required")

        # Get entity candidates from task_config
        candidate_entities = task_config.get("entities", [])
        if not candidate_entities:
            raise ValueError("No entity candidates ('entities') provided in task_config")

        # If exact match exists, use it directly
        if input_value in [name for name, _ in candidate_entities]:
            for name, eid in candidate_entities:
                if name == input_value:
                    logger.info(f"Exact match found for '{input_value}' → '{name}'")
                    processed_params["input_value"] = name
                    processed_params["input_id"] = eid
                    return processed_params, metadata

        # Use embedding-based similarity for linking
        ## Embed entity mention
        query_embedding = self.embedding_model.encode(  # type: ignore
            input_value,
            normalize_embeddings=True,
        )

        # Embed candidates and compute similarity
        candidate_names = [entity[0] for entity in candidate_entities]
        candidate_embeddings = self.embedding_model.encode(candidate_names, normalize_embeddings=True)  # type: ignore
        similarity_scores = np.dot(candidate_embeddings, query_embedding)

        # Sort by similarity and take top-k
        sorted_indices = np.argsort(-similarity_scores)  # Descending order
        top_entities = [
            (candidate_entities[idx], similarity_scores[idx]) for idx in sorted_indices[: self.retrieval_topk]
        ]

        # Select best entity (highest similarity)
        best_entity, best_score = top_entities[0], top_entities[0][1]
        if best_score < min_similarity:
            candidate_str = "\n".join([f"- {ent[0][0]}" for ent in top_entities])
            error_msg = f"'{input_value}' not found."
            if len(candidate_entities) > self.retrieval_topk:
                error_msg += f" Top candidates among {len(candidate_entities)} entities:\n"
            else:
                error_msg += " Available candidates (all):\n"
            error_msg += f"{candidate_str}\nRetry with a different input value or try a different tool."
            raise AgentException(error_msg)

        best_entity_name, best_entity_id = best_entity[0]
        logger.info(f"Input linking: '{input_value}' → '{best_entity_name}' (score: {best_score:.3f})")
        processed_params["input_value"] = best_entity_name
        processed_params["input_id"] = best_entity_id
        metadata["top_candidates"] = top_entities

        return processed_params, metadata

    def postprocess(
        self,
        tool_result: dict,
        grounded_params: dict,
        task_config: dict[str, Any],
        metadata: dict[str, Any],
        **kwargs,
    ) -> tuple[Any, str]:
        """
        Format entity extraction observation.

        Args:
            tool_result: Tool execution result
            grounded_params: Grounded parameters
            task_config: Task configuration
            metadata: Metadata from the worker
            **kwargs: Hidden context

        Returns:
            Tuple of (output data, observation string for LLM)
        """
        input_value = grounded_params["input_value"]
        input_id = grounded_params["input_id"]
        output_data = tool_result["results"]
        if len(output_data) > 1:  # entity class:
            observation_text = format_entity_observation(output_data)
        else:
            observation_text = f"Found {input_value} (ID: {input_id})"

        return output_data, observation_text


class FindRelationAgent(AtomicKBQueryWorkerAgent):
    """
    Worker agent for relation discovery / JOIN.

    Key points:
    - Expects parameters:
        - 'relation' (freebase relation)
        - 'target' (a)
      (a reference like '$0' pointing to a previous expression).
    - Retrieves entity ids from the referenced step via ContextMemory.
    - Discovers candidate relations via KB lookups (1-hop relations) and filters
      using schema_resources; uses embedding_model to rank candidates.
    - Adds 'top_candidates' to metadata and sets processed_params['relation'] to
      the grounded relation id used by the tool.

    Error handling:
    - Raises AgentException when referenced entities or candidate relations are missing.
    """

    def __init__(
        self,
        llm_call: Callable,
        llm_options: dict[str, Any],
        name: str = "find-relation-agent",
        tool_id: str = "atomic_kb_query/find_relation",
        schema_resources: Optional[dict] = None,
        embedding_model: Optional["SentenceTransformer"] = None,
        retrieval_topk: int = 10,
        strict_mode: bool = False,
        **kwargs,
    ):
        """
        Initialize FindRelation agent.

        Args:
            llm_call: LLM call function (not used)
            llm_options: LLM options (not used)
            name: Agent name
            tool_id: Tool ID for find_relation tool
            schema_resources: Shared schema resources
            embedding_model: Shared embedding model
            retrieval_topk: Maximum number of relations to consider
            strict_mode: If True, require exact matches for schema grounding
        """
        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            tool_id=tool_id,
            schema_resources=schema_resources,
            embedding_model=embedding_model,
            retrieval_topk=retrieval_topk,
            strict_mode=strict_mode,
        )

    def _discover_candidate_relations(self, entity_ids: list[str], direction: str, dataset_name: str) -> list[str]:
        """
        Discover candidate relations by querying KB for 1-hop relations.

        Args:
            entity_ids: List of entity IDs
            direction: "forward" or "backward" relation direction
            dataset_name: Dataset name

        Returns:
            List of candidate relation IDs
        """
        from ...tools.freebase.sparql_executor import (
            get_in_relations_with_odbc,
            get_out_relations_with_odbc,
        )

        # Limit the number of entities to query to prevent timeouts
        # If there are too many entities, we sample a subset
        if len(entity_ids) > MAX_ENTITIES_TO_QUERY:
            logger.info(
                f"Too many entities ({len(entity_ids)}) for relation discovery. Sampling {MAX_ENTITIES_TO_QUERY}."
            )
            sampled_entity_ids = self.rng.choice(entity_ids, MAX_ENTITIES_TO_QUERY, replace=False).tolist()
        else:
            sampled_entity_ids = entity_ids

        # Get 1-hop relations
        relations = set()
        for entity_id in sampled_entity_ids:
            if direction == "forward":  # ?x -> entity_id (target)
                entity_relations = get_in_relations_with_odbc(entity_id)
            else:
                entity_relations = get_out_relations_with_odbc(entity_id)
            relations.update(entity_relations)

        return sorted(list(relations))

    def preprocess(
        self,
        raw_params: dict,
        task_config: dict[str, Any],
        parent_memory: ContextMemory,
        **kwargs,
    ) -> tuple[dict, dict]:
        """
        Preprocess relation description by linking to KB relation.

        Args:
            raw_params: {"relation": "films acted in", "target": "$0"}
            task_config: Task configuration
            parent_memory: ContextMemory for retrieving previous step data
            **kwargs: Hidden context (function_list, etc.)

        Returns:
            tuple of (grounded parameters, metadata)

        Raises:
            AgentException: If relation discovery or linking fails
        """
        processed_params = copy.deepcopy(raw_params)
        metadata = {}

        # Extract function_list from the parent memory using step references ($0, $1, etc.)
        function_list = self._get_function_list_from_context(params=raw_params, memory=parent_memory)
        processed_params["function_list"] = function_list

        relation = raw_params.get("relation")
        if relation is None:
            raise AgentException("'relation' is required")

        direction = raw_params.get("direction", "forward")

        target_ref = raw_params.get("target_ref")
        if target_ref is None:
            raise AgentException("'target_ref' ($0, $1, etc.) is required")

        # Get entities from referenced expression
        entity_ids = parent_memory.get_step(int(target_ref.replace("$", ""))).get("full")
        if not isinstance(entity_ids, list):
            # if the previous step is extract_entity, it can return a single entity_id
            entity_ids = [entity_ids]
        entity_ids = [
            eid for eid in entity_ids if eid.startswith("m.") or eid.startswith("g.") or eid.startswith("base.")
        ]
        if len(entity_ids) == 0:
            raise AgentException(f"No valid entity IDs found in '{target_ref}'")

        # Discover candidate relations from KB
        dataset_name = task_config.get("dataset_name", "grailqa")
        candidate_relations = self._discover_candidate_relations(entity_ids, direction, dataset_name)

        if not candidate_relations:
            raise AgentException(f"'{target_ref}' does not have any relations")

        # # Check for direct match
        # if relation in candidate_relations:
        #     # Direct match found
        #     best_relation = relation
        #     logger.info(
        #         f"Relation linking: '{relation}' → '{best_relation}' (direct match)"
        #     )
        #     processed_params["relation"] = best_relation
        #     return processed_params

        # Note: We do not apply filtering here to avoid removing valid relations
        # # Apply filtering as the original KBQA-o1 experiments do
        # relation_list = self.schema_resources.get("relation_list", set())
        # join_ban_list = self.schema_resources.get("join_ban_relation_list", set())
        # filtered_relations = [
        #     rel
        #     for rel in candidate_relations
        #     if (not relation_list or rel in relation_list) and rel not in join_ban_list
        # ]
        # if not filtered_relations:
        #     raise AgentException(f"'{target_ref}' does not have any relations")

        filtered_relations = candidate_relations

        # If direct match exists after filtering, use it
        if relation in filtered_relations:
            processed_params["relation"] = relation
            return processed_params, metadata

        # Use embedding-based similarity for ranking
        if self.embedding_model is None:
            # Programming error
            raise ValueError("No embedding model available for relation linking")

        # Embedding-based pre-filtering if > self.retrieval_topk candidates
        candidates_for_llm = filtered_relations
        if len(filtered_relations) > self.retrieval_topk:
            query_embedding = self.embedding_model.encode(relation, normalize_embeddings=True)
            candidate_embeddings = self.embedding_model.encode(filtered_relations, normalize_embeddings=True)
            similarity_scores = candidate_embeddings.dot(query_embedding)
            sorted_indices = np.argsort(-similarity_scores)
            candidates_for_llm = [filtered_relations[idx] for idx in sorted_indices[: self.retrieval_topk]]
            logger.info(
                f"Pre-filtered {len(filtered_relations)} candidates to top-{self.retrieval_topk} using embeddings"
            )

        if self.strict_mode:
            candidates_str = "\n".join(f"- {candidate}" for candidate in candidates_for_llm)
            error_msg = f"'{relation}' not found in the KB for the given input."
            if len(filtered_relations) > self.retrieval_topk:
                error_msg += f" Top candidates among {len(filtered_relations)} relations:\n"
            else:
                error_msg += " Available candidates (all):\n"
            error_msg += f"{candidates_str}\nRetry with a valid candidate, or try a different input/tool."
            raise AgentException(error_msg)

        # LLM-based schema matching
        best_relation, llm_metadata = self._llm_schema_matching(
            question=parent_memory.query,
            query_relation=relation,
            candidates=candidates_for_llm,
        )

        # Handle "none" result
        if best_relation is None:
            candidate_str = "\n".join(f"- {candidate}" for candidate in candidates_for_llm)
            error_msg = f"'{relation}' not found in the KB for the given input."
            if len(filtered_relations) > self.retrieval_topk:
                error_msg += f" Top candidates among {len(filtered_relations)} relations:\n"
            else:
                error_msg += " Available candidates (all):\n"
            error_msg += f"{candidate_str}\nRetry with a valid candidate, or try a different input/direction/tool."

            logger.info(error_msg)
            raise AgentException(error_msg)

        logger.info(f"Relation linking: '{relation}' → '{best_relation}' (LLM matched)")

        processed_params["relation"] = best_relation
        metadata.update(llm_metadata)
        metadata["candidates_considered"] = candidates_for_llm

        return processed_params, metadata

    def postprocess(
        self,
        tool_result: dict,
        grounded_params: dict,
        task_config: dict[str, Any],
        metadata: dict[str, Any],
        **kwargs,
    ) -> tuple[Any, str]:
        """
        Format relation discovery observation.

        Args:
            tool_result: Tool execution result
            grounded_params: Grounded parameters
            task_config: Task configuration
            metadata: Metadata from the worker
            **kwargs: Hidden context

        Returns:
            Tuple of (observation string for LLM, additional info)
        """
        output_data = tool_result["results"]

        if len(output_data) == 0:
            error_msg = "No results found for the given input. The input entity may not have the specified relation, or the input may be empty. Please try a different input/tool."
            raise AgentException(error_msg)

        observation_text = format_entity_observation(output_data)

        return output_data, observation_text


class MergeAgent(AtomicKBQueryWorkerAgent):
    """
    Worker agent for merging two expressions (AND operation).

    Key points:
    - Expects references to two previous expressions via 'input_ref1'and 'input_ref2'.
    - Calls the merge tool and returns tool results and a concise observation
      like "Found N results".
    - Raises AgentException if the merge yields no results.
    """

    def __init__(
        self,
        llm_call: Callable,
        llm_options: dict[str, Any],
        name: str = "merge-agent",
        tool_id: str = "atomic_kb_query/merge",
        schema_resources: Optional[dict] = None,
        embedding_model: Optional["SentenceTransformer"] = None,
        strict_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            tool_id=tool_id,
            schema_resources=schema_resources,
            embedding_model=embedding_model,
            strict_mode=strict_mode,
        )

    def postprocess(
        self,
        tool_result: dict,
        grounded_params: dict,
        task_config: dict[str, Any],
        metadata: dict[str, Any],
        **kwargs,
    ) -> tuple[Any, str]:
        """Format merge observation.

        Args:
            tool_result: Tool execution result
            grounded_params: Grounded parameters
            task_config: Task configuration
            metadata: Metadata from the worker
            **kwargs: Hidden context

        Returns:
            Tuple of (output data, observation string for LLM)
        """
        output_data = tool_result["results"]

        if len(output_data) == 0:
            error_msg = "No results found for the given input. The two inputs have no common results, or one input is empty. Please try a different input/tool."
            raise AgentException(error_msg)

        observation_text = format_entity_observation(output_data)
        return output_data, observation_text


class OrderAgent(AtomicKBQueryWorkerAgent):
    """
    Worker agent for ordering / ARGMAX/ARGMIN.

    Key points:
    - Expects parameters such as 'mode' ('ARGMAX'/'ARGMIN'), 'property_relation'
      (description), and 'input_ref' (reference to an expression).
    - Discovers or ranks candidate property relations (uses get_out_relations_with_odbc
      for discovery) and grounds 'property_relation' to a KB relation id.
    - Adds 'top_candidates' to metadata and returns ordered results via postprocess.

    Error handling:
    - Raises AgentException when no valid relations or results are found.
    """

    def __init__(
        self,
        llm_call: Callable,
        llm_options: dict[str, Any],
        name: str = "order-agent",
        tool_id: str = "atomic_kb_query/order",
        schema_resources: Optional[dict] = None,
        embedding_model: Optional["SentenceTransformer"] = None,
        retrieval_topk: int = 10,
        strict_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            tool_id=tool_id,
            schema_resources=schema_resources,
            embedding_model=embedding_model,
            retrieval_topk=retrieval_topk,
            strict_mode=strict_mode,
        )

    def _discover_candidate_relations(self, entity_ids: list[str], dataset_name: str) -> dict[str, list[str]]:
        """
        Discover candidate relations by querying KB for 1-hop relations.

        Args:
            entity_ids: List of entity IDs
            dataset_name: Dataset name

        Returns:
            Dict with 'forward' and 'backward' lists of candidate relation IDs
        """
        from ...tools.freebase.sparql_executor import (
            get_in_relations_with_odbc,
            get_out_relations_with_odbc,
        )

        # Limit the number of entities to query to prevent timeouts
        # If there are too many entities, we sample a subset
        if len(entity_ids) > MAX_ENTITIES_TO_QUERY:
            logger.info(
                f"Too many entities ({len(entity_ids)}) for relation discovery. Sampling {MAX_ENTITIES_TO_QUERY}."
            )
            sampled_entity_ids = self.rng.choice(entity_ids, MAX_ENTITIES_TO_QUERY, replace=False).tolist()
        else:
            sampled_entity_ids = entity_ids

        # Get 1-hop relations
        relations = {"forward": set(), "backward": set()}
        for entity_id in sampled_entity_ids:
            entity_relations = get_out_relations_with_odbc(entity_id)
            relations["forward"].update(entity_relations)  # entity_id -> ?x
            entity_relations = get_in_relations_with_odbc(entity_id)
            relations["backward"].update(entity_relations)  # ?x -> entity_id

        return {k: sorted(list(v)) for k, v in relations.items()}

    def preprocess(
        self,
        raw_params: dict,
        task_config: dict[str, Any],
        parent_memory: ContextMemory,
        **kwargs,
    ) -> tuple[dict, dict]:
        """
        Ground literal relation by linking to KB relation.

        Args:
            raw_params: {"relation_description": "films acted in", "expression_ref": "expression0"}
            task_config: Task configuration
            parent_memory: ContextMemory for retrieving previous step data
            **kwargs: Hidden context (function_list, etc.)

        Returns:
            tuple of (grounded parameters, metadata)

        Raises:
            AgentException: If relation discovery or linking fails
        """
        processed_params = copy.deepcopy(raw_params)
        metadata = {}

        # Extract function_list from the parent memory using step references ($0, $1, etc.)
        function_list = self._get_function_list_from_context(params=raw_params, memory=parent_memory)
        processed_params["function_list"] = function_list

        property_relation = raw_params.get("property_relation")
        if property_relation is None:
            raise AgentException("'property_relation' is required")

        input_ref = raw_params.get("input_ref")
        if input_ref is None:
            raise AgentException("'input_ref' ($0, $1, etc.) is required")
        # Get entities from referenced expression
        entity_ids = parent_memory.get_step(int(input_ref.replace("$", ""))).get("full")
        if not isinstance(entity_ids, list):
            # if the previous step is extract_entity, it can return a single entity_id
            entity_ids = [entity_ids]
        entity_ids = [
            eid for eid in entity_ids if eid.startswith("m.") or eid.startswith("g.") or eid.startswith("base.")
        ]
        if len(entity_ids) == 0:
            raise AgentException(f"No valid entity IDs found in '{input_ref}'")

        # Discover candidate relations from KB
        dataset_name = task_config.get("dataset_name", "grailqa")
        candidate_relations: dict[str, list[str]] = self._discover_candidate_relations(entity_ids, dataset_name)

        if not any(candidate_relations.values()):
            raise AgentException(f"No candidate relations discovered for '{input_ref}'")

        # Note: We do not apply filtering here to avoid removing valid relations
        # # Filter by predefined lists
        # literal_relation_list = self.schema_resources.get("literal_relation_list", set())
        #
        # # Apply filtering
        # if not literal_relation_list:
        #     filtered_relations = candidate_relations
        # else:
        #     filtered_relations: dict[str, list[str]] = {
        #         direction: [rel for rel in rels if rel in literal_relation_list]
        #         for direction, rels in candidate_relations.items()
        #     }
        #     if not any(filtered_relations.values()):
        #         raise AgentException(f"No candidate relations discovered for '{input_ref}'")

        filtered_relations = candidate_relations
        filtered_relations_combined = filtered_relations["forward"] + filtered_relations["backward"]

        # If direct match exists after filtering, use it
        if property_relation in filtered_relations_combined:
            if property_relation in candidate_relations["backward"]:
                best_relation = f"(R {property_relation})"  # reverse relation
            else:
                best_relation = property_relation
            processed_params["property_relation"] = best_relation
            return processed_params, metadata

        # Use embedding-based similarity for ranking
        if self.embedding_model is None:
            # Programming error
            raise ValueError("No embedding model available for relation linking")

        # Embedding-based pre-filtering if > self.retrieval_topk candidates
        candidates_for_llm = filtered_relations_combined
        if len(filtered_relations_combined) > self.retrieval_topk:
            query_embedding = self.embedding_model.encode(property_relation, normalize_embeddings=True)
            candidate_embeddings = self.embedding_model.encode(filtered_relations_combined, normalize_embeddings=True)
            similarity_scores = candidate_embeddings.dot(query_embedding)
            sorted_indices = np.argsort(-similarity_scores)
            candidates_for_llm = [filtered_relations_combined[idx] for idx in sorted_indices[: self.retrieval_topk]]
            logger.info(
                f"Pre-filtered {len(filtered_relations_combined)} candidates to top-{self.retrieval_topk} using embeddings"
            )

        if self.strict_mode:
            candidates_str = "\n".join(f"- {candidate}" for candidate in candidates_for_llm)
            error_msg = f"'{property_relation}' not found in the KB for the given input."
            if len(filtered_relations_combined) > self.retrieval_topk:
                error_msg += f" Top candidates among {len(filtered_relations_combined)} relations:\n"
            else:
                error_msg += " Available candidates (all):\n"
            error_msg += f"{candidates_str}\nRetry with a valid candidate, or try a different input/tool."
            raise AgentException(error_msg)

        # LLM-based schema matching (present candidates without direction markers)
        best_relation, llm_metadata = self._llm_schema_matching(
            question=parent_memory.query,
            query_relation=property_relation,
            candidates=candidates_for_llm,
        )

        # Handle "none" result
        if best_relation is None:
            candidate_str = "\n".join(f"- {candidate}" for candidate in candidates_for_llm)
            error_msg = f"'{property_relation}' not found in the KB for the given input."
            if len(filtered_relations_combined) > self.retrieval_topk:
                error_msg += f" Top candidates among {len(filtered_relations_combined)} relations:\n"
            else:
                error_msg += " Available candidates (all):\n"
            error_msg += f"{candidate_str}\nRetry with a valid candidate, or try a different input/tool."

            logger.info(error_msg)
            raise AgentException(error_msg)

        # Add direction marker if backward relation
        if best_relation in candidate_relations["backward"]:
            best_relation = f"(R {best_relation})"  # reverse relation

        logger.info(f"Relation linking: '{property_relation}' → '{best_relation}' (LLM matched)")

        processed_params["property_relation"] = best_relation
        metadata.update(llm_metadata)
        metadata["candidates_considered"] = candidates_for_llm

        return processed_params, metadata

    def postprocess(
        self,
        tool_result: dict,
        grounded_params: dict,
        task_config: dict[str, Any],
        metadata: dict[str, Any],
        **kwargs,
    ) -> tuple[Any, str]:
        """Format order observation.

        Args:
            tool_result: Tool execution result
            grounded_params: Grounded parameters
            task_config: Task configuration
            metadata: Metadata dictionary
            **kwargs: Hidden context

        Returns:
            Tuple of (output data, observation string for LLM)
        """
        output_data = tool_result["results"]

        if len(output_data) == 0:
            error_msg = "No results found for the given input. Please try a different input/tool."
            raise AgentException(error_msg)

        observation_text = format_entity_observation(output_data)

        return output_data, observation_text


class CompareAgent(AtomicKBQueryWorkerAgent):
    """
    Worker agent for numeric comparison filtering (lt, le, gt, ge operations).

    Key points:
    - Expects 'property_relation' (description) and comparison parameters (e.g.,
      'operator', 'threshold', and an expression reference).
    - Uses schema_resources['literal_relation_list'] as candidate relations and
      embedding_model to rank them; sets processed_params['property_relation'] to
      the grounded relation id.
    - Returns filtered results and includes 'top_candidates' in metadata.

    Error handling:
    - Raises AgentException when grounding fails or no results are returned.
    """

    def __init__(
        self,
        llm_call: Callable,
        llm_options: dict[str, Any],
        name: str = "compare-agent",
        tool_id: str = "atomic_kb_query/compare",
        schema_resources: Optional[dict] = None,
        embedding_model: Optional["SentenceTransformer"] = None,
        retrieval_topk: int = 10,
        strict_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            tool_id=tool_id,
            schema_resources=schema_resources,
            embedding_model=embedding_model,
            retrieval_topk=retrieval_topk,
            strict_mode=strict_mode,
        )

    def preprocess(
        self,
        raw_params: dict,
        task_config: dict[str, Any],
        parent_memory: ContextMemory,
        **kwargs,
    ) -> tuple[dict, dict]:
        """
        Ground literal relation by linking to KB relation.

        Args:
            raw_params: {"relation_description": "films acted in", "expression_ref": "expression0"}
            task_config: Task configuration
            parent_memory: ContextMemory for retrieving previous step data
            **kwargs: Hidden context (function_list, etc.)

        Returns:
            tuple of (grounded parameters, metadata)

        Raises:
            AgentException: If relation discovery or linking fails
        """
        processed_params = copy.deepcopy(raw_params)
        metadata = {}

        # Extract function_list from the parent memory using step references ($0, $1, etc.)
        function_list = self._get_function_list_from_context(params=raw_params, memory=parent_memory)
        processed_params["function_list"] = function_list

        property_relation = raw_params.get("property_relation")
        if property_relation is None:
            raise AgentException("'property_relation' is required")

        # Get candidate relations from schema resources
        # Note: Use complete literal_relation_list generated from KB
        # Run data/atomic_kbqa/scripts/generate_literal_relation_list.py to update
        candidate_relations_sorted = self.schema_resources["literal_relation_list_sorted"]
        candidate_relations_set = self.schema_resources["literal_relation_list"]

        # If direct match exists, use it
        if property_relation in candidate_relations_set:
            processed_params["property_relation"] = property_relation
            return processed_params, metadata

        # Use embedding-based similarity for ranking
        if self.embedding_model is None:
            # Programming error
            raise ValueError("No embedding model available for relation linking")

        # Embedding-based pre-filtering if > self.retrieval_topk candidates
        candidates_for_llm = candidate_relations_sorted
        if len(candidate_relations_sorted) > self.retrieval_topk:
            # Encode only the query relation (fast)
            query_embedding = self.embedding_model.encode(property_relation, normalize_embeddings=True)
            # Use precomputed embeddings (no re-encoding needed)
            candidate_embeddings = self.schema_resources["literal_relation_embeddings"]
            similarity_scores = candidate_embeddings.dot(query_embedding)
            sorted_indices = np.argsort(-similarity_scores)
            candidates_for_llm = [candidate_relations_sorted[idx] for idx in sorted_indices[: self.retrieval_topk]]
            logger.info(
                f"Pre-filtered {len(candidate_relations_sorted)} candidates to top-{self.retrieval_topk} using cached embeddings"
            )

        if self.strict_mode:
            candidates_str = "\n".join(f"- {candidate}" for candidate in candidates_for_llm)
            error_msg = f"'{property_relation}' not found in the KB for the given input."
            if len(candidate_relations_sorted) > self.retrieval_topk:
                error_msg += f" Top candidates among {len(candidate_relations_sorted)} relations:\n"
            else:
                error_msg += " Available candidates (all):\n"
            error_msg += f"{candidates_str}\nRetry with a valid candidate, or try a different input/tool."
            raise AgentException(error_msg)

        # LLM-based schema matching
        best_relation, llm_metadata = self._llm_schema_matching(
            question=parent_memory.query,
            query_relation=property_relation,
            candidates=candidates_for_llm,
        )

        # Handle "none" result
        if best_relation is None:
            candidate_str = "\n".join(f"- {candidate}" for candidate in candidates_for_llm)
            error_msg = f"'{property_relation}' not found in the KB for the given input."
            if len(candidate_relations_sorted) > self.retrieval_topk:
                error_msg += f" Top candidates among {len(candidate_relations_sorted)} relations:\n"
            else:
                error_msg += " Available candidates (all):\n"
            error_msg += f"{candidate_str}\nRetry with a valid candidate, or try a different input/tool."

            logger.info(error_msg)
            raise AgentException(error_msg)

        logger.info(f"Relation linking: '{property_relation}' → '{best_relation}' (LLM matched)")

        processed_params["property_relation"] = best_relation
        metadata.update(llm_metadata)
        metadata["candidates_considered"] = candidates_for_llm

        return processed_params, metadata

    def postprocess(
        self,
        tool_result: dict,
        grounded_params: dict,
        task_config: dict[str, Any],
        metadata: dict[str, Any],
        **kwargs,
    ) -> tuple[Any, str]:
        """Format compare observation.

        Args:
            tool_result: Tool execution result
            grounded_params: Grounded parameters
            task_config: Task configuration
            metadata: Metadata dictionary
            **kwargs: Hidden context

        Returns:
            Tuple of (output data, observation string for LLM)
        """
        output_data = tool_result["results"]

        if len(output_data) == 0:
            error_msg = "No results found for the given input. Please try a different input/tool."
            raise AgentException(error_msg)

        observation_text = format_entity_observation(output_data)

        return output_data, observation_text


class TimeConstraintAgent(AtomicKBQueryWorkerAgent):
    """
    Worker agent for temporal filtering (TC operation).

    Key points:
    - Expects 'input_ref' (expression reference), 'temporal_relation' (description),
      and a temporal literal reference (e.g., '$1').
    - Discovers candidate temporal relations (using out-relations), filters by
      literal relations list, and grounds the temporal_relation via the
      embedding_model. Adds 'top_candidates' to metadata.
    - Returns time-filtered results and raises AgentException when no results exist.
    """

    def __init__(
        self,
        llm_call: Callable,
        llm_options: Optional[dict[str, Any]] = None,
        name: str = "time-constraint-agent",
        tool_id: str = "atomic_kb_query/time_constraint",
        schema_resources: Optional[dict] = None,
        embedding_model: Optional["SentenceTransformer"] = None,
        retrieval_topk: int = 10,
        strict_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            tool_id=tool_id,
            schema_resources=schema_resources,
            embedding_model=embedding_model,
            retrieval_topk=retrieval_topk,
            strict_mode=strict_mode,
        )

    def _discover_candidate_relations(self, entity_ids: list[str], dataset_name: str) -> list[str]:
        """
        Discover candidate relations by querying KB for 1-hop relations.

        Args:
            entity_ids: List of entity IDs
            dataset_name: Dataset name

        Returns:
            List of candidate relation IDs

        Raises:
            AgentException: If discovery fails
        """
        from ...tools.freebase.sparql_executor import get_out_relations_with_odbc

        # Limit the number of entities to query to prevent timeouts
        # If there are too many entities, we sample a subset
        if len(entity_ids) > MAX_ENTITIES_TO_QUERY:
            logger.info(
                f"Too many entities ({len(entity_ids)}) for relation discovery. Sampling {MAX_ENTITIES_TO_QUERY}."
            )
            sampled_entity_ids = self.rng.choice(entity_ids, MAX_ENTITIES_TO_QUERY, replace=False).tolist()
        else:
            sampled_entity_ids = entity_ids

        # Get 1-hop relations
        relations = set()
        for entity_id in sampled_entity_ids:
            entity_relations = get_out_relations_with_odbc(entity_id)
            relations.update(entity_relations)

        return sorted(list(relations))

        # Get 1-hop relations
        relations = set()
        for entity_id in entity_ids:
            entity_relations = get_out_relations_with_odbc(entity_id)
            relations.update(entity_relations)

        return sorted(list(relations))

    def preprocess(
        self,
        raw_params: dict,
        task_config: dict[str, Any],
        parent_memory: ContextMemory,
        **kwargs,
    ) -> tuple[dict, dict]:
        """
        Ground temporal relation by linking to KB relation.

        Args:
            raw_params: {"input_ref": "$0", "temporal_relation": "people.person.date_of_birth", "temporal_literal": "NOW"}
            task_config: Task configuration
            parent_memory: ContextMemory for retrieving previous step data
            **kwargs: Hidden context (function_list, etc.)

        Returns:
            tuple of (grounded parameters, metadata)

        Raises:
            AgentException: If relation discovery or linking fails
        """
        processed_params = copy.deepcopy(raw_params)
        metadata = {}

        # Extract function_list from the parent memory using step references ($0, $1, etc.)
        function_list = self._get_function_list_from_context(params=raw_params, memory=parent_memory)
        processed_params["function_list"] = function_list

        temporal_relation = raw_params.get("temporal_relation")
        if temporal_relation is None:
            raise AgentException("'temporal_relation' is required")
        input_ref = raw_params.get("input_ref")
        if input_ref is None:
            raise AgentException("'input_ref' ($0, $1, etc.) is required")
        # Get entities from referenced expression
        entity_ids = parent_memory.get_step(int(input_ref.replace("$", ""))).get("full")
        if not isinstance(entity_ids, list):
            # if the previous step is extract_entity, it can return a single entity_id
            entity_ids = [entity_ids]
        entity_ids = [
            eid for eid in entity_ids if eid.startswith("m.") or eid.startswith("g.") or eid.startswith("base.")
        ]
        if len(entity_ids) == 0:
            raise AgentException(f"No valid entity IDs found in '{input_ref}'")

        # Discover candidate relations from KB
        dataset_name = task_config.get("dataset_name", "grailqa")
        candidate_relations = self._discover_candidate_relations(entity_ids, dataset_name)

        if not candidate_relations:
            error_message = f"'{input_ref}' does not have any outgoing relations. Cannot apply time constraint."
            raise AgentException(error_message)

        # Note: We do not apply filtering here to avoid removing valid relations
        # # Filter by predefined lists
        # literal_relation_list = self.schema_resources.get("literal_relation_list", set())
        #
        # # Apply filtering
        # filtered_relations = [
        #     rel for rel in candidate_relations if not literal_relation_list or rel in literal_relation_list
        # ]
        #
        # if not filtered_relations:
        #     error_message = f"'{input_ref}' does not have any outgoing literal relations. Cannot apply time constraint."
        #     raise AgentException(error_message)

        filtered_relations = candidate_relations

        # If direct match exists after filtering, use it
        if temporal_relation in filtered_relations:
            processed_params["temporal_relation"] = temporal_relation
            return processed_params, metadata

        # Use embedding-based similarity for ranking
        if self.embedding_model is None:
            # Programming error
            raise ValueError("No embedding model available for relation linking")

        # Embedding-based pre-filtering if > self.retrieval_topk candidates
        candidates_for_llm = filtered_relations
        if len(filtered_relations) > self.retrieval_topk:
            query_embedding = self.embedding_model.encode(temporal_relation, normalize_embeddings=True)
            candidate_embeddings = self.embedding_model.encode(filtered_relations, normalize_embeddings=True)
            similarity_scores = candidate_embeddings.dot(query_embedding)
            sorted_indices = np.argsort(-similarity_scores)
            candidates_for_llm = [filtered_relations[idx] for idx in sorted_indices[: self.retrieval_topk]]
            logger.info(
                f"Pre-filtered {len(filtered_relations)} candidates to top-{self.retrieval_topk} using embeddings"
            )

        if self.strict_mode:
            candidates_str = "\n".join(f"- {candidate}" for candidate in candidates_for_llm)
            error_msg = f"'{temporal_relation}' not found in the KB for the given input."
            if len(filtered_relations) > self.retrieval_topk:
                error_msg += f" Top candidates among {len(filtered_relations)} relations:\n"
            else:
                error_msg += " Available candidates (all):\n"
            error_msg += f"{candidates_str}\nRetry with a valid candidate, or try a different input/tool."
            raise AgentException(error_msg)

        # LLM-based schema matching
        best_relation, llm_metadata = self._llm_schema_matching(
            question=parent_memory.query,
            query_relation=temporal_relation,
            candidates=candidates_for_llm,
        )

        # Handle "none" result
        if best_relation is None:
            candidate_str = "\n".join(f"- {candidate}" for candidate in candidates_for_llm)
            error_msg = f"'{temporal_relation}' not found in the KB for the given input."
            if len(filtered_relations) > self.retrieval_topk:
                error_msg += f" Top candidates among {len(filtered_relations)} relations:\n"
            else:
                error_msg += " Available candidates (all):\n"
            error_msg += f"{candidate_str}\nRetry with a valid candidate, or try a different input/tool."

            logger.info(error_msg)
            raise AgentException(error_msg)

        logger.info(f"Relation linking: '{temporal_relation}' → '{best_relation}' (LLM matched)")

        processed_params["temporal_relation"] = best_relation
        metadata.update(llm_metadata)
        metadata["candidates_considered"] = candidates_for_llm

        return processed_params, metadata

    def postprocess(
        self,
        tool_result: dict,
        grounded_params: dict,
        task_config: dict[str, Any],
        metadata: dict[str, Any],
        **kwargs,
    ) -> tuple[Any, str]:
        """Format time_constraint observation.

        Args:
            tool_result: Tool execution result
            grounded_params: Grounded parameters
            task_config: Task configuration
            metadata: Metadata dictionary
            **kwargs: Hidden context

        Returns:
            Tuple of (output data, observation string for LLM)
        """
        output_data = tool_result["results"]

        if len(output_data) == 0:
            error_msg = "No results found for the given input. Temporal constraint may not exist for the specified entity and relation. Please try a different input/tool."
            raise AgentException(error_msg)

        observation_text = format_entity_observation(output_data)

        return output_data, observation_text


class CountAgent(AtomicKBQueryWorkerAgent):
    """
    Worker agent for counting elements in an expression (COUNT operation).

    Key points:
    - Expects 'expression_ref' (reference to an expression).
    - Calls the count tool and returns the numeric count as output_data.
    - postprocess returns ("Count: <n>") as the observation text.
    - Raises AgentException if the count tool returns None or is invalid.
    """

    def __init__(
        self,
        llm_call: Callable,
        llm_options: dict[str, Any],
        name: str = "count-agent",
        tool_id: str = "atomic_kb_query/count",
        schema_resources: Optional[dict] = None,
        embedding_model: Optional["SentenceTransformer"] = None,
        strict_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            tool_id=tool_id,
            schema_resources=schema_resources,
            embedding_model=embedding_model,
            strict_mode=strict_mode,
        )

    def postprocess(
        self,
        tool_result: dict,
        grounded_params: dict,
        task_config: dict[str, Any],
        metadata: dict[str, Any],
        **kwargs,
    ) -> tuple[Any, str]:
        """Format count observation.

        Args:
            tool_result: Tool execution result
            grounded_params: Grounded parameters
            task_config: Task configuration
            metadata: Metadata from the worker
            **kwargs: Hidden context

        Returns:
            Tuple of (output data, observation string for LLM)
        """
        output_data = tool_result["results"]

        if output_data is None:
            raise AgentException("No count result returned. Please try a different input/tool.")

        observation_text = f"Count: {output_data}"
        return output_data, observation_text


class AtomicKBQueryWorkerAgentFactory:
    """
    Factory for creating atomic KB query worker agents.

    This factory loads shared resources (schema, embeddings) once and creates
    all worker agents with these shared resources. This ensures memory efficiency
    and consistent configuration across all agents.

    Dataset-agnostic design: dataset name/config is NOT embedded in worker
    constructors. Instead, it's provided at execution time via task_config.
    """

    def __init__(
        self,
        schema_resources_path: Optional[str] = None,
        embedding_model_name: str = DEFAULT_ENCODER_MODEL,
    ):
        """
        Initialize factory.

        Args:
            schema_resources_path: Path to schema resources directory
            embedding_model_name: Name of embedding model for schema grounding
        """
        self.schema_resources_path = (
            Path(schema_resources_path) if schema_resources_path else DEFAULT_SCHEMA_RESOURCES_PATH
        )
        self.embedding_model_name = embedding_model_name

        # Load embedding model first
        self.embedding_model = get_encoder_model(self.embedding_model_name)

        # Load shared resources with embedding precomputation
        self.schema_resources = load_schema_resources(self.schema_resources_path, self.embedding_model)

    def create_agents(
        self,
        llm_call: Callable,
        llm_options: Optional[dict[str, Any]] = None,
        agent_configs: Optional[dict[str, Any]] = None,
    ) -> dict[str, BaseWorkerAgent]:
        """
        Create all atomic KB query worker agents.

        Args:
            llm_call: LLM call function (not used by these agents)
            llm_options: LLM options (not used by these agents)
            agent_configs: Agent-specific configurations (optional)

        Returns:
            Dict mapping agent_name to agent instance:
                - "extract-entity-agent"
                - "find-relation-agent"
                - "merge-agent"
                - "order-agent"
                - "compare-agent"
                - "time-constraint-agent"
                - "count-agent"
        """
        if llm_options is None:
            llm_options = {}

        if agent_configs is None:
            agent_configs = {}

        agents = {}

        # Extract Entity Agent
        agents["extract-entity-agent"] = ExtractEntityAgent(
            llm_call=llm_call,
            llm_options=llm_options,
            schema_resources=self.schema_resources,
            embedding_model=self.embedding_model,
            **agent_configs.get("extract_entity_agent", {}),
        )

        # Find Relation Agent
        agents["find-relation-agent"] = FindRelationAgent(
            llm_call=llm_call,
            llm_options=llm_options,
            schema_resources=self.schema_resources,
            embedding_model=self.embedding_model,
            **agent_configs.get("find_relation_agent", {}),
        )

        # Merge Agent
        agents["merge-agent"] = MergeAgent(
            llm_call=llm_call,
            llm_options=llm_options,
            schema_resources=self.schema_resources,
            embedding_model=self.embedding_model,
        )

        # Order Agent
        agents["order-agent"] = OrderAgent(
            llm_call=llm_call,
            llm_options=llm_options,
            schema_resources=self.schema_resources,
            embedding_model=self.embedding_model,
        )

        # Compare Agent
        agents["compare-agent"] = CompareAgent(
            llm_call=llm_call,
            llm_options=llm_options,
            schema_resources=self.schema_resources,
            embedding_model=self.embedding_model,
        )

        # Time Constraint Agent
        agents["time-constraint-agent"] = TimeConstraintAgent(
            llm_call=llm_call,
            llm_options=llm_options,
            schema_resources=self.schema_resources,
            embedding_model=self.embedding_model,
        )

        # Count Agent
        agents["count-agent"] = CountAgent(
            llm_call=llm_call,
            llm_options=llm_options,
            schema_resources=self.schema_resources,
            embedding_model=self.embedding_model,
        )

        return agents

    @staticmethod
    def create_agent(
        operator_name: str,
        llm_call: Optional[Callable] = None,
        llm_options: Optional[dict[str, Any]] = None,
        tool_id: Optional[str] = None,
        schema_resources_path: Optional[str] = None,
        embedding_model_name: str = DEFAULT_ENCODER_MODEL,
        retrieval_topk: int = 10,
        strict_mode: bool = False,
        operator_config: Optional[dict[str, Any]] = None,
    ) -> AtomicKBQueryWorkerAgent:
        """
        Create a single atomic KB query worker agent for a specific operator.

        Args:
            operator_name: Name of the atomic operator (e.g., "extract_entity", "find_relation")
            llm_call: Callable function to get LLM responses (not used by these agents)
            llm_options: Options to configure LLM behavior
            tool_id: Tool ID this agent will use (e.g., "atomic_kb_query/extract_entity")
            schema_resources_path: Path to schema resources directory
            embedding_model_name: Name of embedding model for schema grounding
            retrieval_topk: Number of top candidates to consider during retrieval
            strict_mode: If True, require exact matches for schema grounding
            operator_config: Operator-specific configuration parameters

        Returns:
            BaseWorkerAgent instance configured for the specified operator

        Raises:
            ValueError: If operator_name is not supported
        """
        if llm_options is None:
            llm_options = {}
        if operator_config is None:
            operator_config = {}

        # Load shared resources
        resources_path = Path(schema_resources_path) if schema_resources_path else DEFAULT_SCHEMA_RESOURCES_PATH
        embedding_model = get_encoder_model(embedding_model_name)
        schema_resources = load_schema_resources(resources_path, embedding_model)

        # Map operator names to agent classes
        AGENT_CLASSES = {
            "extract_entity": ExtractEntityAgent,
            "find_relation": FindRelationAgent,
            "merge": MergeAgent,
            "order": OrderAgent,
            "compare": CompareAgent,
            "time_constraint": TimeConstraintAgent,
            "count": CountAgent,
        }

        if operator_name not in AGENT_CLASSES:
            raise ValueError(
                f"Unknown atomic KB query operator: {operator_name}. "
                f"Supported operators: {', '.join(AGENT_CLASSES.keys())}"
            )

        agent_class = AGENT_CLASSES[operator_name]

        # Create agent with shared resources and operator-specific config
        return agent_class(
            llm_call=llm_call,
            llm_options=llm_options,
            schema_resources=schema_resources,
            embedding_model=embedding_model,
            retrieval_topk=retrieval_topk,
            strict_mode=strict_mode,
            **operator_config,
        )

    @staticmethod
    def get_all_operators() -> list[str]:
        """
        Return list of all supported atomic KB query operators.

        Returns:
            List of operator names
        """
        return [
            "extract_entity",
            "find_relation",
            "merge",
            "order",
            "compare",
            "time_constraint",
            "count",
        ]
