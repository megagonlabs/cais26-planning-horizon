"""
Tool wrapper and helper utilities for LLM planning agents.

This module provides the base Tool class that wraps functions with
multi-argument support and OpenAI-style JSON schema generation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import time
from types import UnionType
from typing import Any, Callable, Literal, Optional, Union
from typing import get_origin, get_args, Annotated
import inspect

from ..agents.exceptions import AgentException
from ..agents.executable import Executable, ExecutableType, ExecutionResult


@dataclass
class ParamMetadata:
    """
    Metadata for function parameters in tool definitions.

    This replaces Pydantic Field for simpler metadata storage
    without validation constraints. Designed for JSON schema generation
    in OpenAI function calling format.

    Attributes:
        description: Human-readable description of the parameter
        pattern: Regex pattern for string validation (JSON Schema)
        gt: Greater than (exclusive) constraint for numbers
        ge: Greater than or equal constraint for numbers
        lt: Less than (exclusive) constraint for numbers
        le: Less than or equal constraint for numbers
        multiple_of: Multiple of constraint for numbers
        min_length: Minimum length for strings
        max_length: Maximum length for strings
        extra: Dictionary for custom metadata not covered by standard fields
    """

    description: Optional[str] = None
    pattern: Optional[str] = None
    # JSON Schema numeric constraints
    gt: Optional[float] = None  # Greater than (exclusive)
    ge: Optional[float] = None  # Greater than or equal
    lt: Optional[float] = None  # Less than (exclusive)
    le: Optional[float] = None  # Less than or equal
    multiple_of: Optional[float] = None
    # JSON Schema string constraints
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    # Custom metadata - store anything you want!
    extra: dict[str, Any] = field(default_factory=dict)
    # If False the parameter will be omitted from generated LLM/tool schemas
    # (used for passing hidden context like task_config or current_world_state)
    visible: bool = True


class Tool(Executable, ABC):
    """
    Abstract base class for tools that can be used by Worker Agents.

    Each tool wraps a function and provides metadata about its usage,
    including OpenAI-style JSON schema for function calling.
    Implements the Executable interface for unified handling with agents.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize a tool with basic metadata.

        Args:
            name: The name of the tool (should be unique)
            description: A human-readable description of what the tool does
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, *args, **kwargs) -> "ExecutionResult":
        """
        Execute the tool with the given arguments.

        Args:
            *args: Positional arguments for the tool
            **kwargs: Keyword arguments for the tool

        Returns:
            ExecutionResult: The result of the tool execution
        """
        pass

    @abstractmethod
    def spec(self, tool_id: str = "tool", include_hidden_params: bool = False) -> dict[str, Any]:
        """
        Return an OpenAI-style JSON schema for this tool.

        Args:
            tool_id: Optional ID of the tool to customize schema name
            include_hidden_params: If True, includes parameters marked as not visible
                                   in the generated schema (default: False)

        Returns:
            dict: OpenAI-style schema for this tool
        """
        pass

    def close(self) -> None:
        """
        Clean up any resources used by the tool.

        This method is called when the environment is shutting down or
        when the tool is no longer needed.
        """
        pass

    # Executable interface methods
    def get_name(self) -> str:
        """
        Get the name identifier for this tool.

        Returns:
            str: The unique name of this tool
        """
        return self.name

    def get_description(self) -> str:
        """
        Get a human-readable description of this tool.

        Returns:
            str: Description of what this tool does
        """
        return self.description

    def get_schema(self) -> dict[str, Any]:
        """
        Get the schema for this Tool as an executable.

        Returns:
            dict: OpenAI-style schema for this tool
        """
        return self.spec()

    def get_type(self):
        """
        Get the type of this executable (tool).

        Returns:
            ExecutableType: Always returns ExecutableType.TOOL
        """
        return ExecutableType.TOOL


class FunctionTool(Tool):
    """
    A tool that wraps a Python function with automatic schema generation.

    This class automatically introspects the function signature to generate
    the appropriate JSON schema for function calling.
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameter_descriptions: Optional[dict[str, str]] = None,
        cleanup_func: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize a function tool.

        Args:
            func: The Python function to wrap
            name: The name of the tool. If None, will be derived from the function name.
            description: Description of what the tool does. If None, will be
                        extracted from the function's docstring
            parameter_descriptions: Dict mapping parameter names to human-readable descriptions. If None,
                                    will be extracted from ParamMetadata annotations (if available.)
            cleanup_func: Optional function to call when the tool is closed.
        """
        # Extract name from function if not provided
        if name is None:
            name = func.__name__ if func else "function_tool"  # type: ignore

        # Extract description from docstring if not provided
        if description is None:
            description = self._extract_description_from_docstring(func)

        super().__init__(name, description or "No description available")

        self.func = func
        self.parameter_descriptions = parameter_descriptions or {}
        self.cleanup_func = cleanup_func
        self._signature = inspect.signature(func) if func else None

    def close(self) -> None:
        """
        Clean up any resources used by the tool.
        """
        if self.cleanup_func:
            self.cleanup_func()

    def _extract_description_from_docstring(self, func: Callable) -> Optional[str]:
        """
        Extract description from function docstring.

        Args:
            func: The function to extract docstring from

        Returns:
            The docstring, or None if no docstring
        """
        if func.__doc__:
            # Strip whitespace from the docstring
            return func.__doc__.strip()
        return None

    def execute(self, *args, **kwargs) -> "ExecutionResult":
        """
        Call the wrapped function and return result

        Args:
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            ExecutionResult: The function result wrapped in ExecutionResult

        Raises:
            AgentException: If the function raises an exception, it is caught and returned as error
        """

        start = time()
        try:
            result = self.func(*args, **kwargs)
            execution_time = time() - start
            return ExecutionResult.from_tool_execution(
                tool_name=self.name,
                success=True,
                result_data=result,
                execution_time=execution_time,
            )
        except AgentException as e:
            execution_time = time() - start
            return ExecutionResult(
                executable_name=self.name,
                executable_type=ExecutableType.TOOL,
                success=False,
                error_message=str(e),
                execution_time=execution_time,
            )

    def spec(self, tool_id: Optional[str] = None, include_hidden_params: bool = False) -> dict[str, Any]:
        """
        Generate OpenAI-style JSON schema from function signature.

        Supports ParamMetadata annotations for parameter descriptions
        and constraints.

        Args:
            tool_id: Optional ID of the tool to customize schema name
            include_hidden_params: If True, includes parameters marked as not visible
                                   in the generated schema (default: False)

        Returns:
            Dict containing the tool specification with name, description,
            and parameters in OpenAI function calling format
        """
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

        for param_name, param in self._signature.parameters.items():  # type: ignore
            # Skip *args and **kwargs
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            param_spec = {"type": "string"}  # Default to string
            param_metadata = None
            actual_type = param.annotation

            # Extract ParamMetadata if available
            if param.annotation != inspect.Parameter.empty:
                param_metadata, actual_type = self._extract_param_metadata(param.annotation)

            # Skip parameters that are marked as not visible (hidden context)
            if not include_hidden_params and param_metadata and not getattr(param_metadata, "visible", True):
                # Hidden parameter - do not include in the generated schema
                continue

            # Add description from ParamMetadata or fallback to manual descriptions
            description = None
            if param_metadata and param_metadata.description:
                description = param_metadata.description
            elif param_name in self.parameter_descriptions:
                description = self.parameter_descriptions[param_name]

            if description:
                param_spec["description"] = description

            # Infer type from annotation
            ## Handle UnionType (|)
            ## the first type denotes the main type --> use that for type
            if get_origin(actual_type) is UnionType:
                actual_type = actual_type.__args__[0]

            if actual_type != inspect.Parameter.empty:
                # Handle list types with items
                origin = get_origin(actual_type)
                if origin is list:
                    param_spec["type"] = "array"
                    # Extract item type
                    args = get_args(actual_type)
                    if args:
                        item_type = args[0]
                        item_type_name = self._python_type_to_json_type(item_type)
                        param_spec["items"] = {"type": item_type_name}
                    else:
                        # No item type specified, default to string
                        param_spec["items"] = {"type": "string"}
                else:
                    type_name = self._python_type_to_json_type(actual_type)
                    param_spec["type"] = type_name

            # Extract regex pattern from ParamMetadata if available
            if param_metadata and param_metadata.pattern:
                param_spec["pattern"] = param_metadata.pattern

            # Set enum values if the type is Literal
            if get_origin(actual_type) is Literal:
                param_spec["enum"] = list(get_args(actual_type))  # type: ignore

            # Add ParamMetadata constraints to JSON schema if available
            if param_metadata:
                self._add_param_constraints_to_spec(param_metadata, param_spec)

            parameters["properties"][param_name] = param_spec  # type: ignore

            # Mark as required if no default value
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)  # type: ignore

        return {
            "type": "function",
            "function": {
                "name": tool_id if tool_id else self.name,
                "description": self.description,
                "parameters": parameters,
                "strict": True,
            },
        }

    def _python_type_to_json_type(self, python_type) -> str:
        """
        Convert Python type annotation to JSON schema type.

        Args:
            python_type: The Python type annotation

        Returns:
            str: Corresponding JSON schema type
        """
        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        # Handle Union types, Optional, etc.
        if hasattr(python_type, "__origin__"):
            return "string"  # Default for complex types

        return type_mapping.get(python_type, "string")

    def _extract_param_metadata(self, annotation) -> tuple[Optional[ParamMetadata], Any]:
        """
        Extract ParamMetadata and actual type from annotation.

        Handles Annotated types with ParamMetadata, and unwraps Optional/Union.

        Args:
            annotation: The type annotation to examine

        Returns:
            Tuple of (ParamMetadata or None, actual_type)
        """
        origin = get_origin(annotation)

        # Handle Optional/Union
        if origin is Union or (UnionType and origin is UnionType):
            args = get_args(annotation)
            # Filter out NoneType to find the actual type
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                # Recurse on the first non-None type
                return self._extract_param_metadata(non_none_args[0])

        # Handle Annotated
        if origin is Annotated:
            args = get_args(annotation)
            if args:
                actual_type = args[0]
                metadata_obj = None
                for metadata in args[1:]:
                    if isinstance(metadata, ParamMetadata):
                        metadata_obj = metadata
                        break

                # Recurse to handle nested Annotated or Union/Optional
                inner_metadata, inner_actual_type = self._extract_param_metadata(actual_type)
                return metadata_obj or inner_metadata, inner_actual_type

        return None, annotation

    def _add_param_constraints_to_spec(self, param_metadata: ParamMetadata, param_spec: dict) -> None:
        """
        Add ParamMetadata constraints to the JSON schema parameter spec.

        Args:
            param_metadata: The ParamMetadata object
            param_spec: The parameter specification to modify
        """
        # Add string constraints
        if param_metadata.min_length is not None:
            param_spec["minLength"] = param_metadata.min_length
        if param_metadata.max_length is not None:
            param_spec["maxLength"] = param_metadata.max_length

        # Add numeric constraints
        if param_metadata.gt is not None:
            param_spec["exclusiveMinimum"] = param_metadata.gt
        if param_metadata.ge is not None:
            param_spec["minimum"] = param_metadata.ge
        if param_metadata.lt is not None:
            param_spec["exclusiveMaximum"] = param_metadata.lt
        if param_metadata.le is not None:
            param_spec["maximum"] = param_metadata.le
        if param_metadata.multiple_of is not None:
            param_spec["multipleOf"] = param_metadata.multiple_of


# Common tools
def create_finish_tool(use_list: bool = False) -> Tool:
    """
    Create a tool for finishing the task with a final answer.

    Args:
        use_list: If True, create a finish tool that accepts list[str].
                  If False (default), create one that accepts a single string.

    Returns:
        Tool: A FunctionTool for finishing tasks
    """

    def finish(
        answer: Annotated[
            str,
            ParamMetadata(
                description="Final answer to the problem. Only provide the answer value without any additional text. Use $i to reference intermediate results as the answer.",
            ),
        ],
    ) -> str:
        """
        Finish the task with the final answer.

        Args:
            answer: The final answer to the problem

        Returns:
            str: String representation of the final answer
        """
        return str(answer)

    def finish_list(
        answer: Annotated[
            list[str],
            ParamMetadata(
                description="Final answer(s) to the problem. Only provide the answer value without any additional text. Use $i to reference intermediate results as the answer.",
            ),
        ],
    ) -> str | list[str]:
        """
        Finish the task with the final answer.

        Args:
            answer: The final answers (list of strings)

        Returns:
            list[str]: The answers as provided
        """
        return answer

    func = finish_list if use_list else finish
    return FunctionTool(func, name="finish", description="Finish the task with the final answer")


class ToolSetFactory(ABC):
    """
    Abstract base class for tool set factories.

    All tool set factories must inherit from this class and implement
    the create_all_tools() method to generate their tool sets.

    Tool set factories are responsible for creating multiple related tools
    that share common initialization parameters (e.g., a KB engine or
    PDDL domain file). They enable lazy instantiation and cleaner tool
    registration in the ToolRegistry.
    """

    @abstractmethod
    def create_all_tools(self, **shared_params: Any) -> dict[str, "Tool"]:
        """
        Create all tools in this set.

        Args:
            **shared_params: Shared initialization parameters (e.g., kb_path, domain_path).
                           Subclasses should document their required/optional parameters.

        Returns:
            Dictionary mapping tool_id to Tool instance.
            Tool IDs should follow the convention: "{namespace}/{tool_name}"
            (e.g., "kopl/find", "pddl/blocksworld/pickup")

        Raises:
            ValueError: If required parameters are missing or invalid.
            RuntimeError: If tool creation fails.
        """
        pass
