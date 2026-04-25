"""
Environment for routing tool calls and managing agent state.

The Environment class serves as the central orchestration layer for the
multi-agent system, managing both tools and agents through registries,
and providing unified access to executables for Meta Agents and Worker Agents.
"""

from datetime import datetime, timezone
from typing import Any, Callable, Optional, TYPE_CHECKING
import copy
import json
import logging
import re

if TYPE_CHECKING:
    from ..agents.executable import ExecutionResult, ExecutableType
    from ..agents.memory import ContextMemory
    from ..services.llm_registry import LLMProviderRegistry
else:
    from ..agents.executable import ExecutionResult, ExecutableType

from ..agents.exceptions import AgentException
from ..tools.base_tools import create_finish_tool
from .agent_registry import AgentRegistry
from .tool_registry import ToolRegistry


logger = logging.getLogger(__name__)


class Environment:
    """
    Environment that manages tools and agents through registries and executes actions.

    The Environment serves as the central orchestration layer for the multi-agent
    system. It integrates ToolRegistry and AgentRegistry instances, provides
    configuration loading, and offers unified access to executables for both
    Meta Agents and Worker Agents.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the environment with registries.

        Args:
            debug: Enable debug mode logging
        """
        self.memory: dict[str, "ContextMemory"] = {}

        # Multi-agent system registries
        self.tool_registry = ToolRegistry()
        self.agent_registry = AgentRegistry()

        # Configuration tracking
        self.config: dict[str, Any] = {}
        self.substitute_memory_reference: bool = True
        self.list_finish: bool = False  # If False (default), use finish tool that accepts string

        # Finish tool instance
        self.finish_tool = create_finish_tool(use_list=self.list_finish)  # default

        # Debug mode
        self.debug = debug
        if self.debug:
            logger.setLevel(logging.DEBUG)
        logger.debug("Environment initialized with registries")

    def load_from_config(self, config: dict[str, Any]) -> None:
        """
        Load configuration from dictionary and delegate to registries.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration format is invalid
            RuntimeError: If registry loading fails
        """
        self.config = config
        logger.info("Loaded configuration from dictionary")

        self.substitute_memory_reference = config.get("substitute_memory_reference", True)

        # Update list_finish from config if specified
        list_finish = config.get("list_finish", self.list_finish)
        if list_finish != self.list_finish:
            self.list_finish = list_finish
            self.finish_tool = create_finish_tool(use_list=self.list_finish)

        # Load tools FIRST so agents can query tool information
        tools_config = config.get("tools", {})
        if tools_config:
            logger.debug(f"Loading {len(tools_config)} tools from configuration")
            self.tool_registry.load_from_config(tools_config)

        # Load agents SECOND (pass tool_registry so agents can query tool information)
        agents_config = config.get("agents", {})
        if agents_config:
            logger.debug(f"Loading {len(agents_config)} agents from configuration")
            self.agent_registry.load_from_config(agents_config, self.tool_registry)

    def get_available_agents(self, include_finish: bool = True) -> list[dict[str, Any]]:
        """
        Get OpenAI function schemas for all available agents.

        Args:
            include_finish: Whether to include the `finish` tool in the list

        Returns:
            List of OpenAI function calling format schemas for all registered agents
        """
        schemas = self.agent_registry.get_openai_schema()
        if include_finish:
            schemas.append(self.finish_tool.spec())
        return schemas

    def set_llm_call(self, llm_call: Callable) -> None:
        """
        Set the LLM call function for all registered agents.

        This should be called after configuration loading to inject the LLM
        service dependency into all Worker Agents.

        Args:
            llm_call: Callable function to get LLM responses
        """
        self.agent_registry.set_llm_call(llm_call)
        logger.debug("Set LLM call for all registered agents")

    def assign_llm_calls(self, llm_registry: "LLMProviderRegistry") -> None:
        """
        Assign LLM providers to each agent

        Args:
            llm_registry: LLMProviderRegistry instance with configured providers
        """
        for agent_id in self.agent_registry.get_all_agents().keys():
            llm_provider_id = self.agent_registry.get_llm_provider_id(agent_id)
            if llm_provider_id:
                llm_call = llm_registry.get_call(llm_provider_id)
                agent = self.agent_registry.get_agent(agent_id)
                agent.llm_call = llm_call
                logger.debug(f"Assigned LLM provider '{llm_provider_id}' to agent '{agent_id}'")
            else:
                logger.warning(f"No LLM provider specified for agent '{agent_id}'")
                raise ValueError(f"No LLM provider specified for agent '{agent_id}'")

    def execute_tool(self, tool_id: str, params: dict[str, Any] | list[str]) -> "ExecutionResult":
        """
        Execute a tool action with the given name and arguments.

        This function does not throw exceptions on tool execution failure.
        Instead, it captures errors and returns them within the ExecutionResult.

        Args:
            tool_id: ID of the tool to execute
            params: Arguments for the tool, either as a dict or list

        Returns:
            str: The observation/result from executing the tool
        """
        start_ts = datetime.now(timezone.utc).isoformat() + "Z"

        # Check registry tools
        try:
            tool = self.tool_registry.get_tool(tool_id)
        except KeyError as e:
            error_message = str(e)
            logger.error(error_message)
            return ExecutionResult(
                executable_name=tool_id,
                executable_type=ExecutableType.TOOL,
                success=False,
                error_message=error_message,
            )

        # Process parameters
        spec = tool.spec(include_hidden_params=True)["function"]
        properties = spec.get("parameters", {}).get("properties", {})
        required = spec.get("parameters", {}).get("required", [])

        ## If action_args is a list, convert to dict with positional arguments
        if isinstance(params, list):
            if len(params) != len(properties):
                error_message = f"Expected {len(properties)} arguments for '{tool_id}', got {len(params)}."
                return ExecutionResult(
                    executable_name=tool_id,
                    executable_type=ExecutableType.TOOL,
                    success=False,
                    error_message=error_message,
                )
            params = {param: arg for param, arg in zip(properties.keys(), params)}

        # Validate tool parameters
        ## Check if all required parameters are provided
        # Allow internal/hidden parameters to be passed through (not part of schema)
        INTERNAL_PARAMS = {"task_config", "current_world_state"}

        for param in required:
            if param not in params:
                error_message = f"Missing required parameter '{param}' for '{tool_id}'"
                return ExecutionResult(
                    executable_name=tool_id,
                    executable_type=ExecutableType.TOOL,
                    success=False,
                    error_message=error_message,
                )

        ## Check if any provided arguments are not defined in the spec
        for param in params.keys():
            # Skip validation for internal parameters intended for tool internals
            if param in INTERNAL_PARAMS:
                continue
            if param not in properties:
                error_message = f"Unexpected parameter '{param}' for tool '{tool_id}'"
                return ExecutionResult(
                    executable_name=tool_id,
                    executable_type=ExecutableType.TOOL,
                    success=False,
                    error_message=error_message,
                )

        # Additional validation could be added here:
        # - Type checking based on parameter specifications
        # - Value range validation for numeric parameters
        # - String length validation
        # - Pattern matching for string parameters

        # Execute and capture result
        params_str = str(params)
        if len(params_str) > 200:
            params_str = params_str[:200] + "... [truncated]"
        logger.info(f"Executing tool '{tool_id}' with params: {params_str}")

        try:
            result = tool.execute(**params)
        except Exception as e:
            logger.error(f"Tool '{tool_id}' execution failed with error: {e}")
            # If debug, reraise the original exception. This stops execution.
            if not self.debug:
                raise
            # Otherwise, wrap the exception in an ExecutionResult
            result = ExecutionResult(
                executable_name=tool_id,
                executable_type=ExecutableType.TOOL,
                success=False,
                error_message=str(e),
            )

        end_ts = datetime.now(timezone.utc).isoformat() + "Z"

        # Add timing metadata to the execution result
        result.metadata["started_at"] = start_ts
        result.metadata["ended_at"] = end_ts

        return result

    def execute_agent(
        self,
        agent_id: str,
        params: dict[str, Any] | list[str],
        parent_session_id: str,
        step_index: Optional[int] = None,
        task_config: Optional[dict[str, Any]] = None,
    ) -> "ExecutionResult":
        """
        Execute a worker agent with the given name and arguments.

        Args:
            agent_id: ID of the agent to execute
            params: Arguments for the agent, either as a dict or list
            parent_session_id: Optional ID of the parent session
            step_index: Optional step index within the parent session
            task_config: Optional task configuration (e.g., for PlanBench: objects, initial_state, goal_state)

        Returns:
            ExecutionResult: The result from executing the agent
        """
        start_ts = datetime.now(timezone.utc).isoformat() + "Z"

        # Check registry agents
        try:
            agent = self.agent_registry.get_agent(agent_id)
        except KeyError as e:
            error_message = str(e)
            logger.error(error_message)
            return ExecutionResult(
                executable_name=agent_id,
                executable_type=ExecutableType.AGENT,
                success=False,
                error_message=error_message,
            )

        # Process parameters
        spec = agent.get_schema()["function"]
        properties = spec.get("parameters", {}).get("properties", {})
        required = spec.get("parameters", {}).get("required", [])

        ## If action_args is a list, convert to dict with positional arguments
        if isinstance(params, list):
            if len(params) != len(properties):
                error_message = f"Expected {len(properties)} arguments for '{agent_id}', got {len(params)}."
                return ExecutionResult(
                    executable_name=agent_id,
                    executable_type=ExecutableType.AGENT,
                    success=False,
                    error_message=error_message,
                )
            params = {param: arg for param, arg in zip(properties.keys(), params)}

        try:
            # Resolve memory references ($1, $2, etc.) before passing to agent
            if parent_session_id is not None:
                params = self.resolve_action_params(params, parent_session_id)
        except AgentException as e:
            return ExecutionResult(
                executable_name=agent_id,
                executable_type=ExecutableType.AGENT,
                success=False,
                error_message=str(e),
            )

        # Validate agent parameters
        ## Check if all required parameters are provided
        for param in required:
            if param not in params:
                error_message = f"Missing required parameter '{param}' for '{agent_id}'"
                return ExecutionResult(
                    executable_name=agent_id,
                    executable_type=ExecutableType.AGENT,
                    success=False,
                    error_message=error_message,
                )

        ## Check if any provided arguments are not defined in the spec
        for param in params.keys():
            if param not in properties:
                error_message = f"Unexpected parameter '{param}' for '{agent_id}'"
                return ExecutionResult(
                    executable_name=agent_id,
                    executable_type=ExecutableType.AGENT,
                    success=False,
                    error_message=error_message,
                )

        # Execute and capture result
        _task_config = copy.deepcopy(task_config) if task_config is not None else {}
        params_str = str(params)
        if len(params_str) > 200:
            params_str = params_str[:200] + "... [truncated]"
        logger.info(f"Executing agent '{agent_id}' with params: {params_str}")
        session_id = f"{agent_id}_{start_ts}"
        if "query" in params and len(params) == 1:
            query: str = params["query"]
            result = agent.run_episode(
                query,
                environment=self,
                task_config=_task_config,
                session_id=session_id,
                parent_session_id=parent_session_id,
            )
        elif "question" in params and len(params) == 1:
            question: str = params["question"]
            result = agent.run_episode(
                question,
                environment=self,
                task_config=_task_config,
                session_id=session_id,
                parent_session_id=parent_session_id,
            )
        else:
            result = agent.run_episode(
                params,
                environment=self,
                task_config=_task_config,
                session_id=session_id,
                parent_session_id=parent_session_id,
                step_index=step_index,
            )

        end_ts = datetime.now(timezone.utc).isoformat() + "Z"

        # Add execution timestamps to the existing result's metadata
        result.metadata["started_at"] = start_ts
        result.metadata["ended_at"] = end_ts

        return result

    def register_memory(self, memory: "ContextMemory") -> None:
        """Register a ContextMemory instance for state management."""
        self.memory[memory.session_id] = memory

    def get_memory(self, session_id: str) -> Optional["ContextMemory"]:
        """Retrieve a ContextMemory instance by session ID."""
        return self.memory.get(session_id)

    def reset_memory(self) -> None:
        """Reset all registered ContextMemory instances."""
        self.memory.clear()

    def resolve_action_params(
        self, params: dict[str, Any], parent_session_id: str, force_substitute_memory_reference: bool = False
    ) -> dict[str, Any]:
        """
        Resolve memory references ($0, $1, etc.) in action parameter values.

        Handles multiple cases:
        - Exact match ("$0") → resolves to data object from step
        - Embedded refs ("use $0 here") → text substitution with json.dumps
        - Lists (["$0", "$1"]) → resolves each item recursively

        Args:
            params: Parameter dictionary potentially containing memory references
            parent_session_id: Session ID of the parent Meta Agent
            force_substitute_memory_reference: substitute memory reference

        Returns:
            Parameter dictionary with memory references resolved

        Raises:
            ValueError: If a memory reference cannot be resolved
            AgentException: If step index is out of range or data missing
        """

        def substitute_references_in_text(text: str) -> str:
            """Replace all $i placeholders in text with resolved values."""

            def replace_func(match):
                ref = match.group(0)
                try:
                    data = self.resolve_reference(ref, parent_session_id)
                    # Use json.dumps for complex objects, str() for simple types
                    if isinstance(data, (dict, list)):
                        return json.dumps(data, ensure_ascii=False)
                    else:
                        return str(data) if data is not None else ref
                except Exception:
                    # If resolution fails, keep placeholder as-is
                    return ref

            return re.sub(r"\$(\d+)", replace_func, text)

        def resolve_value(value: Any) -> Any:
            """Recursively resolve references in a value."""
            if isinstance(value, str):
                # Check if it's an exact reference match
                if re.match(r"^\$(\d+)$", value):
                    # Exact match - resolve to data object
                    resolved = self.resolve_reference(value, parent_session_id)
                    if resolved is None:
                        raise ValueError(f"Failed to resolve memory reference '{value}'")
                    if force_substitute_memory_reference or self.substitute_memory_reference:
                        return resolved
                    else:
                        return value
                # Check if it contains embedded references
                elif re.search(r"\$(\d+)", value):
                    # Embedded references - respect substitute_memory_reference flag
                    if force_substitute_memory_reference or self.substitute_memory_reference:
                        # Do text substitution
                        return substitute_references_in_text(value)
                    else:
                        # Keep original value with placeholders
                        return value
                else:
                    # No references - pass through
                    return value
            elif isinstance(value, list):
                # Recursively resolve each item in list
                return [resolve_value(item) for item in value]
            else:
                # Pass through other types (int, bool, dict, etc.)
                return value

        resolved_params = {}
        for param_name, param_value in params.items():
            resolved_params[param_name] = resolve_value(param_value)

        return resolved_params

    def resolve_reference(self, reference: str, parent_session_id: Optional[str]) -> Optional[Any]:
        """
        Resolve a memory reference (e.g., "$0", "$1") to actual data.

        Memory references map to Step indices within the Meta Agent's session.
        The full data is retrieved from Step.data["full"] in the parent session's
        ContextMemory.

        Args:
            reference: Memory reference string (e.g., "$0", "$1")
            parent_session_id: Session ID of the parent Meta Agent

        Returns:
            The full data from the referenced step, or None if reference cannot be resolved

        Raises:
            ValueError: If reference format is invalid or step not found
            AgentException: If step index is out of range or data missing
        """
        # Validate reference format
        match = re.match(r"^\$(\d+)$", reference)
        if not match:
            raise ValueError(f"Invalid memory reference format: '{reference}'")

        step_index = int(match.group(1))

        # Get parent session memory
        if parent_session_id is None:
            raise ValueError(f"Cannot resolve memory reference '{reference}' without parent_session_id")

        parent_memory = self.get_memory(parent_session_id)
        if parent_memory is None:
            raise ValueError(f"Parent session '{parent_session_id}' not found in environment memory")

        # Retrieve step from history (using 1-based indexing for $0, $1, etc.)
        if step_index < 0 or step_index >= len(parent_memory.step_history):
            raise AgentException(
                f"Memory reference '{reference}' is out of range. "
                f"Valid range: $0 to ${len(parent_memory.step_history) - 1}"
            )

        # Get the step (convert to 0-based indexing)
        step = parent_memory.step_history[step_index]

        # Retrieve full data from step
        if step.data.get("full") is None:
            logger.error(f"Step {step_index} in session '{parent_session_id}' does not contain 'full' data")
            raise AgentException(
                f"Step {step_index} does not contain data to reference. Please double check the step number."
            )
        else:
            return step.data["full"]

    def get_environment_info(self) -> dict[str, Any]:
        """
        Get comprehensive information about the environment configuration.

        Returns:
            Dictionary containing environment configuration and status
        """
        return {
            "config": self.config,
            "registry_tools": list(self.tool_registry.tools.keys()),
            "registry_agents": list(self.agent_registry.agents.keys()),
            "memory_sessions": list(self.memory.keys()),
            "total_tools": len(self.tool_registry.tools),
            "total_agents": len(self.agent_registry.agents),
        }

    def close(self) -> None:
        """
        Clean up environment resources.

        This method closes all registered tools and agents, releasing any resources.
        """
        self.tool_registry.close_all()
        self.agent_registry.close_all()
