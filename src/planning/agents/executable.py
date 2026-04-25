"""
Executable interface and coordination dataclasses for multi-agent system.

This module provides the unified interface that both tools and agents implement,
along with the ExecutionResult dataclass for unified result handling.
The planning functionality has been moved to the unified Step class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class ExecutableType(Enum):
    """Enum for distinguishing between different types of executables."""

    TOOL = "tool"
    AGENT = "agent"


class Executable(ABC):
    """
    Abstract interface for unified schema representation across tools and agents.

    This interface allows the Environment to treat tools and agents uniformly
    for discovery, schema generation, and execution coordination.
    """

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name identifier for this executable.

        Returns:
            str: The unique name of this executable
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get a human-readable description of this executable.

        Returns:
            str: Description of what this executable does
        """
        pass

    @abstractmethod
    def get_schema(self) -> dict[str, Any]:
        """
        Get the OpenAI-style JSON schema for this executable.

        Returns:
            dict: Schema containing name, description, and parameters
                  following OpenAI function calling format
        """
        pass

    @abstractmethod
    def get_type(self) -> ExecutableType:
        """
        Get the type of this executable (tool vs agent).

        Returns:
            ExecutableType: The type of this executable
        """
        pass


@dataclass
class ExecutionResult:
    """
    Unified result format for both tool and agent execution.

    This class provides a standardized way to return execution results
    that can be consumed by Step objects and integrated into agent workflows.

    Attributes:
        executable_name: Name of the tool or agent that was executed
        executable_type: Type of executable (tool or agent)
        success: Whether the execution was successful
        result_data: The main result data from execution
        error_message: Error message if execution failed
        execution_time: Time taken for execution in seconds
        token_usage: LLM token usage information (for agents)
        metadata: Additional metadata about the execution
        step_count: Number of steps taken (for agents)
        timestamp: When the execution completed
    """

    executable_name: str
    executable_type: ExecutableType
    success: bool
    result_data: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    token_usage: Optional[dict[str, dict[str, int]]] = None
    step_count: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tool_execution(
        cls,
        tool_name: str,
        success: bool,
        result_data: Any = None,
        error_message: Optional[str] = None,
        execution_time: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ExecutionResult":
        """Create an ExecutionResult from tool execution."""
        return cls(
            executable_name=tool_name,
            executable_type=ExecutableType.TOOL,
            success=success,
            result_data=result_data,
            error_message=error_message,
            execution_time=execution_time,
            metadata=metadata or {},
        )

    @classmethod
    def from_agent_execution(
        cls,
        agent_name: str,
        success: bool,
        result_data: Any = None,
        error_message: Optional[str] = None,
        execution_time: Optional[float] = None,
        token_usage: Optional[dict[str, dict[str, int]]] = None,
        step_count: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ExecutionResult":
        """Create an ExecutionResult from agent execution."""
        return cls(
            executable_name=agent_name,
            executable_type=ExecutableType.AGENT,
            success=success,
            result_data=result_data,
            error_message=error_message,
            execution_time=execution_time,
            token_usage=token_usage or {},
            step_count=step_count,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the execution result to a dictionary."""
        return {
            "executable_name": self.executable_name,
            "executable_type": self.executable_type.value,
            "success": self.success,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "execution_time": self.execution_time,
            "token_usage": self.token_usage,
            "metadata": self.metadata,
            "step_count": self.step_count,
            "timestamp": self.timestamp.isoformat(),
        }
