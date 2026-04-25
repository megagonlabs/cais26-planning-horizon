"""
Agent registry for configuration-based agent management.

This module provides the AgentRegistry class for managing Worker Agents
through configuration loading with OpenAI function schema generation
for Meta Agent selection.
"""

from typing import Any, Callable, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from ..agents.base_agent import BaseWorkerAgent
    from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry for managing Worker Agents through configuration-based loading.

    The AgentRegistry provides configuration-driven management of Worker Agent
    components for the multi-agent system. It loads and instantiates agents
    from configuration dictionaries, maintains collections of agents, and
    generates OpenAI function calling schemas for Meta Agent selection.
    """

    def __init__(self):
        """Initialize an empty agent registry."""
        self.agents: dict[str, "BaseWorkerAgent"] = {}
        self.llm_providers: dict[str, str] = {}  # Map agent_id to llm_provider_id

    def close_all(self) -> None:
        """
        Close all registered agents and clean up resources.

        Calls the close() method on each registered agent to release any
        resources (e.g., ODBC connections, file handles).
        """
        for agent_id, agent in self.agents.items():
            try:
                agent.close()
                logger.debug(f"Closed agent: {agent_id}")
            except Exception as e:
                logger.warning(f"Error closing agent {agent_id}: {e}")

    def load_from_config(self, config_dict: dict[str, Any], tool_registry: "ToolRegistry") -> None:
        """
        Load and instantiate Worker Agents from configuration dictionary.

        Args:
            config_dict: Dictionary containing agent configurations with
                        agent IDs as keys and settings as values
            tool_registry: ToolRegistry instance for querying tool information

        Raises:
            ValueError: If agent configuration is invalid
            ImportError: If agent class cannot be imported
            RuntimeError: If agent instantiation fails
        """
        if not isinstance(config_dict, dict):
            raise ValueError("Configuration must be a dictionary")

        logger.info(f"Loading {len(config_dict)} agents from configuration")

        for agent_id, agent_config in config_dict.items():
            self._load_agent(agent_id, agent_config, tool_registry)

        logger.info(f"Successfully loaded {len(self.agents)} agents")

    def _load_agent(self, agent_id: str, agent_config: dict[str, Any], tool_registry: "ToolRegistry") -> None:
        """
        Load and instantiate a single Worker Agent from configuration.

        Args:
            agent_id: Unique identifier for the agent
            agent_config: Configuration dictionary for the agent
            tool_registry: ToolRegistry instance for querying tool information

        Raises:
            ValueError: If agent configuration is missing required fields
            ImportError: If agent class cannot be imported
        """
        if not isinstance(agent_config, dict):
            raise ValueError(f"Agent configuration for '{agent_id}' must be a dictionary")

        # Get agent type and parameters
        agent_type = agent_config.get("type")
        if not agent_type:
            raise ValueError(f"Agent '{agent_id}' missing required 'type' field")

        # Load agent based on type
        elif agent_type == "husky_commonsense_agent":
            self._load_husky_commonsense(agent_id, agent_config, tool_registry)
        elif agent_type == "husky_math_agent":
            self._load_husky_math(agent_id, agent_config, tool_registry)
        elif agent_type == "husky_code_agent":
            self._load_husky_code(agent_id, agent_config, tool_registry)
        elif agent_type == "husky_search_agent":
            self._load_husky_search(agent_id, agent_config, tool_registry)
        elif agent_type.startswith("kopl_"):  # bulk load all KoPL agents
            self._load_kopl_agents(agent_id, agent_config, tool_registry)
        elif agent_type == "atomic_kb_query_agents":  # bulk load atomic KB query agents
            self._load_atomic_kb_query_agents(agent_id, agent_config, tool_registry)
        else:
            raise NotImplementedError(f"Agent type '{agent_type}' is not supported.")

    def _load_husky_commonsense(
        self, agent_id: str, agent_config: dict[str, Any], tool_registry: "ToolRegistry"
    ) -> None:
        """Load a Husky Commonsense Agent."""
        from ..agents.worker_agents.husky_agents import HuskyCommonsenseAgent

        # Extract agent configuration
        llm_options = agent_config.get("model", {})

        # Extract prompt template from config
        prompt_config = agent_config.get("prompt", {})
        generation_prompt = prompt_config.get("generation", "")
        synthesis_prompt = prompt_config.get("synthesis", "")

        if not generation_prompt:
            logger.warning(f"No prompt template found for agent '{agent_id}'")

        # Create agent instance
        agent = HuskyCommonsenseAgent(
            llm_call=None,  # Will be injected later
            llm_options=llm_options,
            name=agent_id,
            generation_prompt=generation_prompt,
            synthesis_prompt=synthesis_prompt,
        )

        llm_provider_id = agent_config.get("llm_provider", "default")
        self.register(agent_id, agent, llm_provider_id=llm_provider_id)

    def _load_husky_math(self, agent_id: str, agent_config: dict[str, Any], tool_registry: "ToolRegistry") -> None:
        """Load a Husky Math Agent."""
        from ..agents.worker_agents.husky_agents import HuskyMathAgent

        # Extract agent configuration
        llm_options = agent_config.get("model", {})

        # Extract prompt template from config
        prompt_config = agent_config.get("prompt", {})
        generation_prompt = prompt_config.get("generation", "")
        synthesis_prompt = prompt_config.get("synthesis", "")

        if not generation_prompt:
            logger.warning(f"No prompt template found for agent '{agent_id}'")

        # Create agent instance
        agent = HuskyMathAgent(
            llm_call=None,  # Will be injected later
            llm_options=llm_options,
            name=agent_id,
            generation_prompt=generation_prompt,
            synthesis_prompt=synthesis_prompt,
        )

        llm_provider_id = agent_config.get("llm_provider", "default")
        self.register(agent_id, agent, llm_provider_id=llm_provider_id)

    def _load_husky_code(self, agent_id: str, agent_config: dict[str, Any], tool_registry: "ToolRegistry") -> None:
        """Load a Husky Code Agent."""
        from ..agents.worker_agents.husky_agents import HuskyCodeAgent

        # Extract agent configuration
        llm_options = agent_config.get("model", {})

        # Extract code_header if provided
        code_header = agent_config.get("code_header", "").strip()

        # Extract prompt template from config
        prompt_config = agent_config.get("prompt", {})
        generation_prompt = prompt_config.get("generation", "")
        synthesis_prompt = prompt_config.get("synthesis", "")

        if not generation_prompt or not synthesis_prompt:
            logger.warning(f"No prompt template found for agent '{agent_id}'")

        # Create agent instance
        agent = HuskyCodeAgent(
            llm_call=None,  # Will be injected later
            llm_options=llm_options,
            name=agent_id,
            generation_prompt=generation_prompt,
            synthesis_prompt=synthesis_prompt,
            code_header=code_header,
        )

        llm_provider_id = agent_config.get("llm_provider", "default")
        self.register(agent_id, agent, llm_provider_id=llm_provider_id)

    def _load_husky_search(self, agent_id: str, agent_config: dict[str, Any], tool_registry: "ToolRegistry") -> None:
        """Load a Husky Search Agent."""
        from ..agents.worker_agents.husky_agents import HuskySearchAgent

        # Extract agent configuration
        llm_options = agent_config.get("model", {})

        # Extract prompt template from config
        prompt_config = agent_config.get("prompt", {})
        generation_prompt = prompt_config.get("generation", "")
        synthesis_prompt = prompt_config.get("synthesis", "")

        if not generation_prompt or not synthesis_prompt:
            logger.warning(f"No prompt template found for agent '{agent_id}'")

        # Create agent instance
        agent = HuskySearchAgent(
            llm_call=None,  # Will be injected later
            llm_options=llm_options,
            name=agent_id,
            generation_prompt=generation_prompt,
            synthesis_prompt=synthesis_prompt,
        )

        llm_provider_id = agent_config.get("llm_provider", "default")
        self.register(agent_id, agent, llm_provider_id=llm_provider_id)

    def _resolve_tool_set_ids_to_tool_ids(self, tool_set_ids: list[str], tool_registry: "ToolRegistry") -> list[str]:
        """
        Resolve tool set IDs to concrete tool IDs.

        Queries ToolRegistry to get all tool IDs created by each tool set.

        Args:
            tool_set_ids: List of tool set IDs (e.g., ["kopl_schema_free_tools"])
            tool_registry: ToolRegistry instance for querying tool information

        Returns:
            List of concrete tool IDs (e.g., ["kopl/find_all", "kopl/and", ...])

        Raises:
            KeyError: If tool set ID not found in registry
        """
        tool_ids = []
        for tool_set_id in tool_set_ids:
            # Query ToolRegistry for tools created by this tool set
            set_tool_ids = tool_registry.get_tool_ids_by_tool_set(tool_set_id)
            tool_ids.extend(set_tool_ids)

        return tool_ids

    def _load_kopl_agents(self, agent_id: str, agent_config: dict[str, Any], tool_registry: "ToolRegistry") -> None:
        """
        Load KoPL agents: one agent per tool (bulk instantiation only).

        This creates multiple agents (one per KoPL operator) from a single
        configuration entry. Each agent performs one KoPL operator
        with KB grounding and validation.

        Args:
            agent_id: Unique identifier for the agent group (e.g., "kopl_agents")
            agent_config: Configuration dictionary containing:
                - type: KoPL agent type (e.g., "kopl_agent", "kopl_find_and_filter_concept_agent")
                - available_tools: List of tool set IDs (e.g., ["kopl_schema_free_tools"])
                - agent_per_tool: bool (must be true) - only bulk instantiation supported
                - llm_provider: LLM provider ID (optional, default: "default")
                - model: LLM configuration options
                - enable_schema_update_from_tool: bool (default: true)
                - common_parameters: Shared parameters for all agents
                    - embeddings_dir: Path to embeddings directory
                    - encoder_model_name: Name of embedding model
                    - strict_mode: does not perform fuzzy KB grounding if true (default: false)
                - preprocessing_prompts: Optional dict mapping operator names to prompts
                - postprocessing_prompts: Optional dict mapping operator names to prompts
        """
        from ..agents.worker_agents.kopl_agents import KoPLAgentFactory

        # Get available_tools configuration
        available_tools = agent_config.get("available_tools", [])
        if not available_tools:
            raise ValueError(f"Agent '{agent_id}' missing required 'available_tools' field")

        # Check instantiation mode - only agent_per_tool=true is supported
        agent_per_tool = agent_config.get("agent_per_tool", True)
        if not agent_per_tool:
            raise ValueError(
                f"KoPL agents only support agent_per_tool=true (bulk instantiation). "
                f"Got agent_per_tool={agent_per_tool} for agent '{agent_id}'"
            )

        # Resolve tool set IDs to concrete tool IDs by querying ToolRegistry
        tool_ids = self._resolve_tool_set_ids_to_tool_ids(available_tools, tool_registry)

        if not tool_ids:
            raise ValueError(f"No tools found for agent '{agent_id}' with available_tools={available_tools}")

        # Get the KoPL tool factory instance from ToolRegistry (reuse existing instance)
        # Agents need access to the factory's KB engine for input validation/preprocessing
        first_tool_set_id = available_tools[0]
        try:
            # Get factory from the first tool set (all should use the same KB)
            kopl_factory = tool_registry.get_factory_instance(first_tool_set_id)
            logger.debug(f"Reusing KoPL factory from tool set '{first_tool_set_id}' for agent preprocessing")
        except KeyError as e:
            raise ValueError(f"Could not retrieve KoPL factory for tool set '{first_tool_set_id}': {e}")

        # Get common parameters
        common_params = agent_config.get("common_parameters", {})
        embeddings_dir = common_params.get(
            "embeddings_dir", "data/kopl_kbqa/kqa_pro/embeddings/BAAI___bge-base-en-v1.5"
        )
        encoder_model_name = common_params.get("encoder_model_name", "BAAI/bge-base-en-v1.5")
        strict_mode = common_params.get("strict_mode", False)
        retrieval_topk = common_params.get("retrieval_topk", 10)

        # Get KoPL agent type (default: "kopl_agent")
        agent_type = agent_config.get("type", "kopl_agent")

        # Get LLM configuration
        llm_options = agent_config.get("model", {})
        llm_provider_id = agent_config.get("llm_provider", "default")

        # Get schema update setting
        enable_schema_update = agent_config.get("enable_schema_update_from_tool", True)

        # BULK INSTANTIATION: Create one agent per tool
        logger.info(f"Loading {len(tool_ids)} KoPL agents (bulk mode) for '{agent_id}'")

        for tool_id in tool_ids:
            if not tool_id.startswith("kopl/"):
                logger.warning(f"Invalid KoPL tool ID '{tool_id}', skipping")
                continue

            # Extract operator name from tool ID (e.g., "kopl/find" -> "find")
            operator_name = tool_id.split("/", 1)[1]

            # Create agent using factory
            agent = KoPLAgentFactory.create_agent(
                agent_type=agent_type,
                operator_name=operator_name,
                llm_call=None,  # Will be injected later
                llm_options=llm_options,
                tool_ids=[tool_id],  # Single tool per agent
                embeddings_dir=embeddings_dir,
                encoder_model_name=encoder_model_name,
                strict_mode=strict_mode,
                retrieval_topk=retrieval_topk
            )

            # Inject factory for preprocessing/validation
            agent.kopl_factory = kopl_factory

            # Perform inline schema update if enabled
            if enable_schema_update:
                tool = tool_registry.get_tool(tool_id)
                agent.update_schema_from_tool(tool)
                logger.debug(f"Updated schema for agent '{operator_name}' from tool '{tool_id}'")

            # Register agent with operator-specific ID (e.g., "find_agent")
            agent_id = operator_name
            self.register(agent_id, agent, llm_provider_id=llm_provider_id)
            logger.debug(f"Registered KoPL agent '{agent_id}'")

        logger.info(f"Successfully loaded {len(tool_ids)} KoPL agents (bulk mode)")

    def _load_atomic_kb_query_agents(
        self, agent_id: str, agent_config: dict[str, Any], tool_registry: "ToolRegistry"
    ) -> None:
        """
        Load Atomic KB Query agents: one agent per tool (bulk instantiation only).

        This creates multiple agents (one per atomic operator) from a single
        configuration entry. Each agent performs one atomic KB query operator
        with schema grounding and validation.

        Args:
            agent_id: Unique identifier for the agent group (e.g., "atomic_kb_query_agents")
            agent_config: Configuration dictionary containing:
                - type: "atomic_kb_query_agents"
                - available_tools: List of tool set IDs (e.g., ["atomic_kb_query_tools"])
                - agent_per_tool: bool (must be true) - only bulk instantiation supported
                - llm_provider: LLM provider ID (optional, default: "default")
                - model: LLM configuration options
                - enable_schema_update_from_tool: bool (default: true)
                - common_parameters: Shared parameters for all agents
                    - schema_resources_path: Path to schema resources
                    - embedding_model_name: Name of embedding model
                - agent-specific parameters (e.g., extract_entity_agent, find_relation_agent)
        """
        from ..agents.worker_agents.atomic_kb_query_agents import (
            AtomicKBQueryWorkerAgentFactory,
            AtomicKBQueryWorkerAgent,
        )

        # Get available_tools configuration
        available_tools = agent_config.get("available_tools", [])
        if not available_tools:
            raise ValueError(f"Agent '{agent_id}' missing required 'available_tools' field")

        # Check instantiation mode - only agent_per_tool=true is supported
        agent_per_tool = agent_config.get("agent_per_tool", True)
        if not agent_per_tool:
            raise ValueError(
                f"Atomic KB Query agents only support agent_per_tool=true (bulk instantiation). "
                f"Got agent_per_tool={agent_per_tool} for agent '{agent_id}'"
            )

        # Resolve tool set IDs to concrete tool IDs by querying ToolRegistry
        tool_ids = self._resolve_tool_set_ids_to_tool_ids(available_tools, tool_registry)

        if not tool_ids:
            raise ValueError(f"No tools found for agent '{agent_id}' with available_tools={available_tools}")

        # Get the Atomic KB Query tool factory instance from ToolRegistry
        # Agents need access to the factory's shared resources (schema, embeddings)
        first_tool_set_id = available_tools[0]
        try:
            # Get factory from the first tool set
            atomic_kb_factory = tool_registry.get_factory_instance(first_tool_set_id)
            logger.debug(f"Reusing Atomic KB Query factory from tool set '{first_tool_set_id}' for agent preprocessing")
        except KeyError as e:
            raise ValueError(f"Could not retrieve Atomic KB Query factory for tool set '{first_tool_set_id}': {e}")

        # Get common parameters
        common_params = agent_config.get("common_parameters", {})
        schema_resources_path = common_params.get("schema_resources_path", "data/atomic_kbqa/freebase/")
        embedding_model_name = common_params.get("embedding_model_name", "BAAI/bge-base-en-v1.5")
        retrieval_topk = common_params.get("retrieval_topk", 10)
        strict_mode = common_params.get("strict_mode", False)

        # Get LLM configuration
        llm_options = agent_config.get("model", {})
        llm_provider_id = agent_config.get("llm_provider", "default")

        # Get schema update setting
        enable_schema_update = agent_config.get("enable_schema_update_from_tool", True)

        # Get agent-specific configurations
        agent_specific_configs = {
            "extract_entity_agent": agent_config.get("extract_entity_agent", {}),
            "find_relation_agent": agent_config.get("find_relation_agent", {}),
        }

        # BULK INSTANTIATION: Create one agent per tool
        logger.info(f"Loading {len(tool_ids)} Atomic KB Query agents (bulk mode) for '{agent_id}'")

        for tool_id in tool_ids:
            if not tool_id.startswith("atomic_kb_query/"):
                logger.warning(f"Invalid Atomic KB Query tool ID '{tool_id}', skipping")
                continue

            # Extract operator name from tool ID (e.g., "atomic_kb_query/extract_entity" -> "extract_entity")
            operator_name = tool_id.split("/", 1)[1]

            # Get operator-specific configuration
            operator_config = agent_specific_configs.get(f"{operator_name}_agent", {})

            # Create agent using factory
            agent: AtomicKBQueryWorkerAgent = AtomicKBQueryWorkerAgentFactory.create_agent(
                operator_name=operator_name,
                llm_call=None,  # Will be injected later
                llm_options=llm_options,
                tool_id=tool_id,
                schema_resources_path=schema_resources_path,
                embedding_model_name=embedding_model_name,
                retrieval_topk=retrieval_topk,
                strict_mode=strict_mode,
                operator_config=operator_config,
            )

            # Inject factory for preprocessing/validation
            agent.atomic_kb_factory = atomic_kb_factory  # type: ignore

            # Perform inline schema update if enabled
            if enable_schema_update:
                tool = tool_registry.get_tool(tool_id)
                agent.update_schema_from_tool(tool)
                logger.debug(f"Updated schema for agent '{operator_name}' from tool '{tool_id}'")

            # Register agent with operator-specific ID (e.g., "extract_entity_agent")
            agent_id = operator_name
            self.register(agent_id, agent, llm_provider_id=llm_provider_id)
            logger.debug(f"Registered Atomic KB Query agent '{agent_id}'")

        logger.info(f"Successfully loaded {len(tool_ids)} Atomic KB Query agents (bulk mode)")

    def register(self, agent_id: str, agent: "BaseWorkerAgent", llm_provider_id: str = "default") -> None:
        """
        Register a Worker Agent instance with the registry.

        Args:
            agent_id: Unique identifier for the agent
            agent: BaseWorkerAgent instance to register
            llm_provider_id: Identifier for the LLM provider to use with this agent

        Raises:
            ValueError: If agent_id is already registered or agent is invalid
        """
        from ..agents.base_agent import BaseWorkerAgent

        if not isinstance(agent, BaseWorkerAgent):
            raise ValueError(f"Agent must be an instance of BaseWorkerAgent, got {type(agent)}")

        if agent_id in self.agents:
            raise ValueError(f"Agent ID '{agent_id}' is already registered")

        agent.set_name(agent_id)  # Ensure agent has correct name
        self.agents[agent_id] = agent
        self.llm_providers[agent_id] = llm_provider_id
        logger.debug(f"Registered agent '{agent_id}': {agent.name}")

    def get_agent(self, agent_id: str) -> "BaseWorkerAgent":
        """
        Get an agent by its ID.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            BaseWorkerAgent: The agent instance

        Raises:
            KeyError: If the agent is not found
        """
        if agent_id not in self.agents:
            error_message = f"Unknown agent '{agent_id}'. Available agents: {list(self.get_all_agents().keys())}"
            raise KeyError(error_message)
        return self.agents[agent_id]

    def get_llm_provider_id(self, agent_id: str) -> str:
        """
        Get the LLM provider ID associated with a given agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            str: The LLM provider ID

        Raises:
            KeyError: If the agent is not found
        """
        if agent_id not in self.llm_providers:
            error_message = f"Unknown agent '{agent_id}'. Available agents: {list(self.get_all_agents().keys())}"
            raise KeyError(error_message)
        return self.llm_providers[agent_id]

    def get_all_agents(self) -> dict[str, "BaseWorkerAgent"]:
        """
        Get all registered agents.

        Returns:
            dict[str, BaseWorkerAgent]: Dictionary mapping agent IDs to agent instances
        """
        return self.agents.copy()

    def get_agent_names(self) -> list[str]:
        """
        Get the names of all registered agents.

        Returns:
            list[str]: List of agent IDs
        """
        return list(self.agents.keys())

    def get_openai_schema(self) -> list[dict[str, Any]]:
        """
        Generate OpenAI function calling schemas for all registered agents.

        This allows Meta Agents to treat Worker Agents as callable executables.

        Returns:
            list[dict[str, Any]]: List of OpenAI function calling format schemas
        """
        schemas = []
        for agent_id, agent in self.agents.items():
            # Get the agent schema using the Executable interface
            schema = agent.get_schema(agent_id=agent_id)

            # Ensure parameters field exists
            if "parameters" not in schema["function"]:
                schema["function"]["parameters"] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                }

            schemas.append(schema)

        return schemas

    def set_llm_call(self, llm_call: Callable) -> None:
        """
        Set the LLM call function for all registered agents.

        This is typically called by the Environment after agent registration
        to inject the LLM service dependency.

        Args:
            llm_call: Callable function to get LLM responses
        """
        for agent_id, agent in self.agents.items():
            agent.llm_call = llm_call
            logger.debug(f"Set LLM call for agent '{agent_id}'")

    def clear(self) -> None:
        """Clear all registered agents."""
        self.agents.clear()
        logger.debug("Cleared all agents from registry")

    def __len__(self) -> int:
        """Get the number of registered agents."""
        return len(self.agents)

    def __contains__(self, agent_id: str) -> bool:
        """Check if an agent is registered."""
        return agent_id in self.agents

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"AgentRegistry(agents={list(self.agents.keys())})"
