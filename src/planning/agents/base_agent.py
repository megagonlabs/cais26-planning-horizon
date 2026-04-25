"""
Base agent classes for LLM planning agents.

This module provides the abstract base classes for different types of agents
in the multi-agent system: BaseAgent, BaseMetaAgent, and BaseWorkerAgent.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING
import copy

from openai.types.chat import ChatCompletion
from openai.types.responses import Response

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from ..environment import Environment
    from ..tools.base_tools import Tool
    from .executable import Executable, ExecutionResult, ExecutableType
    from .memory import ContextMemory
else:
    from .executable import Executable, ExecutableType, ExecutionResult
    from .memory import ContextMemory


@dataclass
class LLMResponse:
    """
    Response object from LLM calls containing content, token usage, and tool calls.

    This dataclass provides a structured way to return LLM responses that is
    extensible and type-safe compared to tuple unpacking.
    """

    content: str = ""
    token_usage: dict[str, dict[str, int]] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_summary: list[str] = field(default_factory=list)
    item_list: list[dict[str, Any]] = field(default_factory=list)
    extra_content: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "LLMResponse":
        """
        Create a deep copy of this LLMResponse instance.

        Returns:
            LLMResponse: A new instance with deep-copied mutable fields
        """
        return LLMResponse(
            content=self.content,
            token_usage=copy.deepcopy(self.token_usage),
            tool_calls=copy.deepcopy(self.tool_calls),
            reasoning_summary=copy.deepcopy(self.reasoning_summary),
            item_list=copy.deepcopy(self.item_list),
            extra_content=copy.deepcopy(self.extra_content),
        )

    def get_reasoning_summary(self) -> list[str]:
        """
        Extract reasoning summary texts from the item list.

        Returns:
            list[str]: List of reasoning summary texts
        """
        if self.reasoning_summary:
            return self.reasoning_summary

        summaries = []
        for item in self.item_list:
            if item.get("type") == "reasoning":
                for summary in item.get("summary", []):
                    if summary.get("type") == "summary_text":
                        summaries.append(summary.get("text", ""))
        return summaries


class BaseAgent(ABC):
    """
    Abstract base class for all planning agents.

    This class defines the common interface that all agents should implement,
    including methods for running episodes.
    """

    def __init__(
        self,
        llm_call: Callable,
        llm_options: dict[str, Any],
        name: str = "BaseAgent",
        max_steps: int = 10,
        use_builtin_tool_input: bool = False,
        use_builtin_tool_output: bool = False,
        cleanup_func: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize the base agent.

        Args:
            llm_call: Callable function to get LLM responses
            llm_options: Options to configure the LLM behavior
            name: Name identifier for the agent
            max_steps: Maximum number of steps before stopping execution
            use_builtin_tool_input: Whether to use built-in tool input handling
            use_builtin_tool_output: Whether to use built-in tool output handling
            cleanup_func: Optional function to call when the agent is closed
        """
        self.name = name
        self.max_steps = max_steps

        self.current_step = 0
        self.user_query = None

        self.llm_call = llm_call
        self.llm_options = llm_options
        self.use_builtin_tool_input = use_builtin_tool_input
        self.use_builtin_tool_output = use_builtin_tool_output
        self.cleanup_func = cleanup_func

    @abstractmethod
    def run_episode(
        self,
        query_or_params: str | dict[str, Any],
        environment: "Environment",
        task_config: dict[str, Any],
        session_id: str,
        parent_session_id: str,
        memory: "ContextMemory",
        **kwargs,
    ) -> "ExecutionResult":
        """
        Run a complete episode to solve the given task.

        Args:
            query_or_params: The query (str) or parameters (dict) for the task.
                             Most agents take a string query, but some agents like
                             KoPL agents take structured parameters as a dict.
            environment: The environment providing tools and state management
            task_config: Optional dictionary containing task-specific config
                         (e.g., objects, initial_state, goal_state). This
                         parameter is intended for worker agents/tools that
                         require access to the problem definition and is
                         passed as a hidden context parameter (not exposed
                         to LLM tool schemas).
            session_id: Optional session ID for tracking
            parent_session_id: Optional parent session ID for tracking
            step_index: Optional step index within the parent session
            memory: Optional context memory as initial state
            **kwargs: Additional configuration parameters

        Returns:
            ExecutionResult: The result of the episode execution
        """
        pass

    def get_llm_response(self, messages: list[dict[str, str]] | str, **kwargs) -> LLMResponse:
        """
        Get the response from the LLM based on the provided messages.

        Handles responses from both Chat Completions and Responses API.

        Args:
            messages: List of message dictionaries or a single prompt string

        Returns:
            LLMResponse: Response object containing content, token usage, and tool calls
        """
        response: ChatCompletion | Response = self.llm_call(messages, **self.llm_options, **kwargs)

        content = ""
        token_usage = {self.llm_options["model"]: {}}
        tool_calls: list[dict[str, Any]] = []
        reasoning_summary: list[str] = []
        item_list: list[dict[str, Any]] = []
        extra_content: dict[str, Any] = {}

        # Extract usage if available
        if hasattr(response, "usage") and response.usage:
            token_usage[self.llm_options["model"]] = response.usage.to_dict()

        if isinstance(response, ChatCompletion):  # ChatCompletion API response
            if response.choices:
                message = response.choices[0].message  # type: ignore
                if hasattr(message, "tool_calls") and message.tool_calls:
                    # Tool call response
                    tool_call_obj = message.tool_calls[0]
                    tool_calls = [
                        {
                            "arguments": tool_call_obj.function.arguments,  # type: ignore
                            "call_id": tool_call_obj.id,
                            "name": tool_call_obj.function.name,  # type: ignore
                            "type": tool_call_obj.type,
                        }
                    ]

                # Regular chat response
                if hasattr(message, "content"):
                    content = message.content or ""

                # If content starts with <think>, extract reasoning summary
                if content.startswith("<think>"):
                    # The reasoning summary should end with </think>
                    end_think_idx = content.find("</think>")
                    if end_think_idx != -1:
                        reasoning_text = content[len("<think>") : end_think_idx].strip()
                        reasoning_summary.append(reasoning_text)
                        # Remove the reasoning part from content
                        content = content[end_think_idx + len("</think>") :].strip()

                # Extract extra content if available
                ## VertexAI API returns thought signature in extra_content
                if hasattr(message, "extra_content"):
                    extra_content = message.extra_content

        elif isinstance(response, Response):  # Responses API response
            # Responses API returns output as a list of items
            for item in response.output:
                if item.type == "function_call":
                    tool_call = {
                        "arguments": item.arguments,
                        "call_id": item.call_id,
                        "name": item.name,
                        "type": item.type,
                    }
                    item_list.append(tool_call)
                    tool_calls.append(tool_call)
                elif item.type == "message":
                    if item.role != "assistant":
                        raise ValueError(f"Unexpected role in message item: {item.role}")
                    content_list = []
                    for content in item.content:
                        if content.type == "output_text":
                            content_list.append(
                                {
                                    "annotations": content.annotations,
                                    "text": content.text,
                                    "type": "output_text",
                                }
                            )
                        elif content.type == "refusal":
                            content_list.append({"refusal": content.refusal, "type": "refusal"})
                        else:
                            raise NotImplementedError(f"Unknown content type in message item: {content.type}")
                    item_list.append(
                        {
                            "id": item.id,
                            "content": content_list,
                            "role": "assistant",
                            "type": item.type,
                        }
                    )
                elif item.type == "reasoning":
                    item_list.append(
                        {
                            "id": item.id,
                            "summary": [{"type": "summary_text", "text": summary.text} for summary in item.summary],
                            "type": item.type,
                            "encrypted_content": item.encrypted_content,
                        }
                    )
                else:
                    raise NotImplementedError(f"Unknown item type in Responses output: {item.type}")

            # Extract text content from output
            content = response.output_text
        else:
            raise ValueError("Unknown response type from LLM call")

        return LLMResponse(
            content=content or "",
            token_usage=token_usage,
            tool_calls=tool_calls,
            reasoning_summary=reasoning_summary,
            item_list=item_list,
            extra_content=extra_content,
        )

    def close(self) -> None:
        """
        Clean up any resources used by the agent.

        This method is called when the environment is shutting down or
        when the agent is no longer needed.
        """
        if self.cleanup_func:
            self.cleanup_func()


class BaseMetaAgent(BaseAgent, ABC):
    """
    Minimal abstract base class for Meta Agents.

    Meta Agents are responsible for high-level planning, task decomposition,
    and coordination of Worker Agents. This base class provides only the
    essential interface without prescriptive method definitions or
    coordination-specific attributes, allowing concrete implementations
    to define their own planning approaches and coordination strategies.

    Concrete Meta Agent classes should define their own attributes and
    methods based on their specific planning paradigm (e.g., ReAct-based
    iterative planning vs Plan-and-Execute upfront planning).
    """

    def __init__(
        self,
        llm_call,
        llm_options: dict[str, Any] = {},
        name: str = "BaseMetaAgent",
        max_steps: int = 10,
        use_builtin_tool_input: bool = False,
        use_builtin_tool_output: bool = False,
        cleanup_func: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        """
        Initialize the Meta Agent with minimal interface.

        Args:
            llm_call: Callable function to get LLM responses
            llm_options: Options to configure the LLM behavior
            name: Name identifier for the agent
            max_steps: Maximum number of steps before stopping execution
            use_builtin_tool_input: Whether to use built-in tool input handling
            use_builtin_tool_output: Whether to use built-in tool output handling
            cleanup_func: Optional function to call when the agent is closed
            **kwargs: Additional configuration parameters for concrete implementations
        """
        super().__init__(
            llm_call,
            llm_options,
            name,
            max_steps,
            use_builtin_tool_input,
            use_builtin_tool_output,
            cleanup_func,
        )


class BaseWorkerAgent(BaseAgent, Executable, ABC):
    """
    Abstract base class for Worker Agents.

    Worker Agents execute specific sub-tasks assigned by Meta Agents.
    They can use tools and LLM reasoning to complete their assigned work.
    They use specialized WorkerAgentStep objects to represent their execution.
    Implements the Executable interface to allow Meta Agents to treat
    Worker Agents as callable executables.
    """

    def __init__(
        self,
        llm_call,
        llm_options: dict[str, Any] = {},
        name: str = "BaseWorkerAgent",
        max_steps: int = 10,
        use_builtin_tool_input: bool = False,
        use_builtin_tool_output: bool = False,
        tool_ids: Optional[list[str]] = None,
        update_schema_from_tool: bool = False,
        cleanup_func: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize the Worker Agent.

        Args:
            llm_call: Callable function to get LLM responses
            llm_options: Options to configure the LLM behavior
            name: Name identifier for the agent
            max_steps: Maximum number of steps for sub-task execution
            use_builtin_tool_input: Whether to use built-in tool input handling
            use_builtin_tool_output: Whether to use built-in tool output handling
            tool_ids: Optional list of tool IDs this agent can use
            update_schema_from_tool: Whether to update schema from associated tool
            cleanup_func: Optional function to call when the agent is closed
        """
        super().__init__(
            llm_call,
            llm_options,
            name,
            max_steps,
            use_builtin_tool_input,
            use_builtin_tool_output,
            cleanup_func,
        )
        self.tool_ids = tool_ids if tool_ids is not None else []
        self.enable_schema_update_from_tool = update_schema_from_tool
        self._schema: Optional[dict[str, Any]] = None

    @abstractmethod
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
        Run a complete episode to solve the given task.

        Args:
            query_or_params: The query (str) or parameters (dict) for the task.
                             Most agents take a string query, but some agents like
                             KoPL agents take structured parameters as a dict.
            environment: The environment providing tools and state management
            task_config: Optional dictionary containing task-specific config
                         (e.g., objects, initial_state, goal_state). This
                         parameter is intended for worker agents/tools that
                         require access to the problem definition and is
                         passed as a hidden context parameter (not exposed
                         to LLM tool schemas).
            session_id: Optional session ID for tracking
            parent_session_id: Optional parent session ID for tracking
            memory: Optional context memory as initial state
            step_index: Optional step index within the parent session
            **kwargs: Additional configuration parameters

        Returns:
            ExecutionResult: The result of the episode execution
        """
        pass

    def get_schema(self, agent_id: Optional[str] = None) -> dict[str, Any]:
        """
        Get the schema for this Worker Agent as an executable.

        Returns cached schema if available (from tool), otherwise returns default schema.

        Args:
            agent_id: Optional ID of the agent to customize schema name

        Returns:
            dict: OpenAI-style schema for this agent
        """
        if self._schema is not None:
            schema = self._schema.copy()
            if agent_id:
                schema["function"]["name"] = agent_id
            return schema

        # Default schema
        return {
            "type": "function",
            "function": {
                "name": agent_id if agent_id else self.name,
                "description": "Worker agent that can execute sub-tasks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subtask_description": {
                            "type": "string",
                            "description": "Description of the sub-task to execute",
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context for sub-task execution",
                        },
                    },
                    "required": ["subtask_description"],
                },
            },
        }

    def update_schema_from_tool(self, tool: "Tool") -> None:
        """
        Update agent schema using the provided tool's schema.

        Args:
            tool: The tool whose schema should be used to update this agent's schema
        """
        tool_schema = tool.get_schema()["function"]

        # Copy tool schema and adapt for agent use
        self._schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": tool_schema.get("description", f"Agent for {tool.get_name()}"),
                "parameters": tool_schema.get(
                    "parameters",
                    {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query to be processed by the agent",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                ),
                "strict": True,
            },
        }

    def get_tool_schemas(self, environment: "Environment") -> list[dict[str, Any]]:
        """
        Get OpenAI schemas for this agent's tools.

        Args:
            environment: The environment providing the tool registry

        Returns:
            list[dict[str, Any]]: List of OpenAI-style schemas for agent's tools
        """
        schemas = []
        for tool_id in self.tool_ids:
            tool = environment.tool_registry.get_tool(tool_id)
            schemas.append(tool.spec(tool_id=tool_id))
        return schemas

    # Executable interface methods
    def get_name(self) -> str:
        """
        Get the name identifier for this agent.

        Returns:
            str: The unique name of this agent
        """
        return self.name

    def set_name(self, name: str) -> None:
        """
        Set the name identifier for this agent.

        Args:
            name: The new name to set for this agent
        """
        self.name = name

    def get_description(self) -> str:
        """
        Get a human-readable description of this agent.

        Returns:
            str: Description of what this agent does
        """
        return "Worker agent that can execute sub-tasks"

    def get_type(self) -> "ExecutableType":
        """
        Get the type of this executable (agent).

        Returns:
            ExecutableType: Always returns ExecutableType.AGENT
        """
        return ExecutableType.AGENT
