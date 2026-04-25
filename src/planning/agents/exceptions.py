from typing import Any, Optional


class AgentException(Exception):
    """
    Exception for errors that can be addressed by LLM agents themselves.

    Used for recoverable issues like:
    - LLM parsing/generation failures
    - Invalid action or response formats
    - Plan validation errors
    - Content generation issues that can be retried

    Non-recoverable errors (bugs, config issues, system failures) should use
    standard Python exceptions and will bubble up for debugging.
    """

    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the AgentException.

        Args:
            message: Human-readable error message
            agent_name: Name of the agent that raised the exception
            details: Additional context or metadata about the error
        """
        self.agent_name = agent_name
        self.details = details or {}

        super().__init__(message)
