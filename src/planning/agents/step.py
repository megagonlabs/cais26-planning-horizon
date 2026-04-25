"""
Unified step class for both planned and executed steps across all agent types.

This module provides a single, flexible Step class that can handle planning
and execution phases while storing rich metadata from ExecutionResult.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING
import json

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .executable import ExecutionResult


class StepStatus(Enum):
    """Status of a step in execution."""

    PLANNED = "planned"  # Step is planned but not yet executed
    COMPLETED = "completed"  # Step has been executed successfully
    FAILED = "failed"  # Step execution failed


@dataclass
class Step:
    """
    Unified step class for both planned and executed steps across all agent types.

    Supports flexible attribute storage, automatic key field display,
    and metadata integration from ExecutionResult. Can represent both
    planned steps (execution plans) and executed steps (execution history).
    """

    step_num: int
    step_type: str  # e.g., "react", "meta_agent", "husky_commonsense"
    status: StepStatus = StepStatus.PLANNED
    timestamp: datetime = field(default_factory=datetime.now)

    # Core flexible storage for step data
    data: dict[str, Any] = field(default_factory=dict)

    # Planning and execution coordination
    dependencies: list[int] = field(
        default_factory=list
    )  # step_nums this step depends on

    # Metadata from ExecutionResult (populated after execution)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Key fields for display (set automatically based on step_type)
    key_fields: list[str] = field(init=False)

    def __post_init__(self):
        """Set key_fields based on step_type after initialization."""
        self.key_fields = self._get_key_fields_for_type(self.step_type)

    def _get_key_fields_for_type(self, step_type: str) -> list[str]:
        """Get the key fields to display for each step type."""
        field_mapping = {
            "sh": ["action", "observation"],
            "fh": ["action", "observation"],
            "husky_commonsense": ["input", "output"],
            "husky_math": ["input", "output"],
            "husky_code": ["input", "code", "code_output", "output"],
            "husky_search": ["input", "search_query", "search_result", "output"],
        }
        return field_mapping.get(step_type, ["input", "output"])

    def __str__(self) -> str:
        """Format step showing key_fields and status."""
        status_indicator = {
            StepStatus.COMPLETED: "[completed]",
            StepStatus.PLANNED: "[planned]",
            StepStatus.FAILED: "[failed]",
        }[self.status]

        parts = [f"Step {self.step_num} {status_indicator}"]

        for field_name in self.key_fields:
            if field_name in self.data and self.data[field_name] is not None:
                if isinstance(self.data[field_name], dict):
                    value = json.dumps(self.data[field_name])
                else:
                    value = str(self.data[field_name])
                parts.append(f"{field_name.title()}: {value}")

        return "\n".join(parts)

    # Data access methods
    def get(self, key: str, default: Any = None) -> Any:
        """Get data attribute with default."""
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set data attribute."""
        self.data[key] = value

    def get_metadata(self, key: str) -> Any:
        """Get metadata attribute."""
        return self.metadata.get(key)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata attribute."""
        self.metadata[key] = value

    def update_from_execution_result(self, result: "ExecutionResult") -> None:
        """Update step with metadata from ExecutionResult after execution."""
        self.status = StepStatus.COMPLETED if result.success else StepStatus.FAILED

        # Store ExecutionResult metadata in step metadata
        if result.execution_time is not None:
            self.metadata["execution_time"] = result.execution_time
        if result.token_usage:
            self.metadata["token_usage"] = result.token_usage
        if result.error_message:
            self.metadata["error_message"] = result.error_message
        if result.metadata:
            self.metadata.update(result.metadata)

    def can_execute(self, completed_steps: list[int]) -> bool:
        """Check if this step can be executed based on dependencies."""
        return self.status == StepStatus.PLANNED and all(
            dep_step_num in completed_steps for dep_step_num in self.dependencies
        )

    def to_dict(self, exclude_large_data: bool = False) -> dict[str, Any]:
        """
        Convert step to dictionary for serialization.

        Args:
            exclude_large_data: If True, excludes large data fields (e.g., "full")
                                from serialization, keeping only lightweight fields
                                for logging. Default: False for backward compatibility.

        Returns:
            Dictionary representation of the step
        """
        data = (
            self.data.copy()
            if not exclude_large_data
            else self._filter_data_for_logging()
        )

        return {
            "step_num": self.step_num,
            "step_type": self.step_type,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "data": data,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }

    def _filter_data_for_logging(self) -> dict[str, Any]:
        """
        Filter step data to exclude large datasets for logging.

        Excludes the "full" field which contains complete intermediate results
        while keeping lightweight fields like "observation", "action", "thought", etc.

        Returns:
            Filtered data dictionary suitable for logging
        """
        filtered_data = {}
        # Fields to exclude from logging (large datasets)
        exclude_keys = {"full"}

        for key, value in self.data.items():
            if key not in exclude_keys:
                filtered_data[key] = value

        return filtered_data

    def to_dict_for_logging(self) -> dict[str, Any]:
        """
        Convert step to dictionary for logging with large data excluded.

        This is a convenience method equivalent to to_dict(exclude_large_data=True).
        Excludes large data fields (e.g., "full") from serialization while keeping
        lightweight fields for visibility in logs.

        Returns:
            Dictionary representation suitable for logging
        """
        return self.to_dict(exclude_large_data=True)

    @classmethod
    def from_dict(cls, step_dict: dict[str, Any]) -> "Step":
        """Create step from dictionary."""
        step = cls(
            step_num=step_dict["step_num"],
            step_type=step_dict["step_type"],
            status=StepStatus(step_dict["status"]),
            timestamp=datetime.fromisoformat(step_dict["timestamp"]),
            data=step_dict.get("data", {}),
            dependencies=step_dict.get("dependencies", []),
            metadata=step_dict.get("metadata", {}),
        )
        return step
