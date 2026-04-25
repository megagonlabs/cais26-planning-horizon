"""Wrapper for VertexAI API calls via the OpenAI client

# Setup

1. Install the `google-auth` Python SDK and complete authentication
2. Set environment variables for `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` (the fields are configurable by conf/model/*.yaml)
"""

from typing import Optional
from typing import Any

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
import google.auth
import google.auth.transport.requests
import openai

OPENAI_MESSAGE_CLASS = {
    "user": ChatCompletionUserMessageParam,
    "assistant": ChatCompletionAssistantMessageParam,
    "system": ChatCompletionSystemMessageParam,
    "tool": ChatCompletionToolMessageParam,
}


class VertexAIOpenAIClient:
    """Wrapper for VertexAI API calls via the OpenAI client"""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "global",
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs,
    ):
        if not project_id:
            raise ValueError("project_id must be provided for Vertex AI OpenAI client")
        base_url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi"
        # Programmatically get an access token
        self.creds, self.project = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
            quota_project_id=project_id,
        )

        client_kwargs: dict[str, Any] = {
            "api_key": "<PLACEHOLDER>",  # The API key will be set later
            "base_url": base_url,
        }
        if timeout is not None:
            client_kwargs["timeout"] = float(timeout)
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries

        self.client = openai.OpenAI(**client_kwargs)

    def reflesh_api_key(self) -> None:
        """Check and refresh auth token if needed.

        Notes: The access token expires after one hour.

        Based on https://docs.cloud.google.com/vertex-ai/generative-ai/docs/migrate/openai/auth-and-credentials#refresh_your_credentials
        """
        if not self.creds.valid:
            self.creds.refresh(google.auth.transport.requests.Request())

            if not self.creds.valid:
                raise RuntimeError("Unable to refresh auth")

            self.client.api_key = self.creds.token

    def _call_chat_completion(self, messages: list[dict[str, str]], model: str, **kwargs) -> ChatCompletion:
        """
        Call the Chat Completions API with the given messages.

        Args:
            messages: List of message dictionaries
            model: Model name
            **kwargs: Additional parameters for the API call

        Returns:
            ChatCompletion: Response from the Chat Completions API
        """
        self.reflesh_api_key()  # Refresh auth token if needed

        _messages = []

        for message in messages:
            _messages.append(OPENAI_MESSAGE_CLASS[message["role"]](**message))

        return self.client.chat.completions.create(messages=_messages, model=model, **kwargs)

    def call(self, messages: list[dict[str, str]], model: str, api_type: str = "chat", **kwargs) -> ChatCompletion:
        """
        Call the Vertex AI API.

        Args:
            messages: List of message dictionaries
            model: Model name
            api_type: Type of API to call ("chat" for Chat Completions)
            **kwargs: Additional parameters for the API call

        Returns:
            ChatCompletion: Response from the appropriate API
        """
        # Validate that we have a model
        if model is None:
            raise ValueError("'model' parameter is required")

        # Validate api_type: Vertex AI OpenAI endpoint only supports Chat Completions API
        if api_type != "chat":
            raise ValueError(f"Unsupported api_type: {api_type}")

        # Chat Completions API - validate messages is list
        if not isinstance(messages, list):
            raise ValueError("Chat Completions API requires messages to be a list of dicts")

        return self._call_chat_completion(messages, model, **kwargs)

    def close(self) -> None:
        """Close the underlying OpenAI client connection."""
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
