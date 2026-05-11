from __future__ import annotations

from typing import Any, Optional

from openai.types.chat import ChatCompletion

from .openai import OpenAIClient


def _normalize_vllm_base_url(base_url: str) -> str:
    """Normalize a vLLM base URL to the OpenAI client base format.

    The OpenAI Python client expects a base URL like
    ``http://localhost:8000/v1``. Users often provide the full chat
    completions endpoint instead, so we strip that suffix here.

    Args:
        base_url: Configured base URL or full endpoint URL

    Returns:
        str: Normalized base URL suitable for ``openai.OpenAI``
    """
    normalized = base_url.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


class vLLMClient(OpenAIClient):
    """OpenAI-compatible chat client wrapper for local vLLM servers."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = "EMPTY",
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key or "EMPTY",
            base_url=_normalize_vllm_base_url(base_url),
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

    def call(
        self,
        messages: list[dict[str, str]] | str,
        model: str,
        api_type: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        """Call a local vLLM server through the Chat Completions API.

        Args:
            messages: Chat messages or a raw prompt string
            model: Model identifier exposed by the vLLM server
            api_type: Requested API type; only chat is supported here
            **kwargs: Additional OpenAI-compatible chat parameters

        Returns:
            ChatCompletion: Response returned by the vLLM server
        """
        requested_api_type = api_type or "chat"
        if requested_api_type != "chat":
            raise ValueError(
                "vLLMClient only supports OpenAI-compatible chat completions"
            )

        if isinstance(messages, str):
            chat_messages: list[dict[str, str]] = [
                {"role": "user", "content": messages}
            ]
        elif isinstance(messages, list):
            chat_messages = messages
        else:
            raise ValueError("messages must be a list of dicts or a string")

        if "max_completion_tokens" in kwargs and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")

        response = super().call(
            messages=chat_messages,
            model=model,
            api_type="chat",
            **kwargs,
        )

        if not isinstance(response, ChatCompletion):
            raise ValueError(f"Unexpected response type from vLLM client: {type(response)}")
        return response
