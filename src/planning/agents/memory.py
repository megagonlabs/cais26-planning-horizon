from typing import Any, Callable, Optional
from dataclasses import dataclass, field

from .step import Step, StepStatus


@dataclass
class ContextMemory:
    session_id: str
    agent_name: str
    query: str
    parent_session_id: Optional[str] = None
    step_history: list[Step] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: Step):
        """Add a step to the execution history."""
        self.step_history.append(step)

    def get_step(self, index: int) -> Step:
        """Get a specific step from the execution history."""
        if 0 <= index < len(self.step_history):
            return self.step_history[index]
        raise IndexError("Step not found")

    def get_last_step(self) -> Optional[Step]:
        """Get the last step from the execution history."""
        if len(self.step_history) == 0:
            raise IndexError("No steps in history")
        return self.step_history[-1]

    def get_history(self, copy: bool = True) -> list[Step]:
        """Get the full execution history."""
        if copy:
            return self.step_history.copy()
        else:
            return self.step_history

    def get_formatted_history(self, include_planned: bool = True, step_formatter: Callable = str) -> str:
        """Get history formatted for LLM consumption."""
        if not self.step_history:
            return ""

        formatted_steps = []
        for step in self.step_history:
            if not include_planned and step.status == StepStatus.PLANNED:
                continue
            formatted_steps.append(step_formatter(step))
        return "\n\n".join(formatted_steps)

    def get_metadata(self, key: str) -> Any:
        """Get metadata for the context."""
        return self.metadata.get(key)

    def set_metadata(self, key: str, value: Any):
        """Set metadata for the context."""
        self.metadata[key] = value
