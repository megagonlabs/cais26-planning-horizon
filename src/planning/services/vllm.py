from typing import Optional
from urllib import response
from openai.types import Completion
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
import openai

OPENAI_MESSAGE_CLASS = {
    "user": ChatCompletionUserMessageParam,
    "assistant": ChatCompletionAssistantMessageParam,
    "system": ChatCompletionSystemMessageParam,
}


class vLLMClient:
    """Wrapper for vLLM API calls (OpenAI API-compatible)."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = "EMPTY",
        **kwargs
    ):
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def call(
        self, messages: list[dict[str, str]] | str, model: str, **kwargs
    ) -> ChatCompletion | Completion:
        """
        Call the vLLM API with the given messages.

        Args:
            messages: List of message dictionaries or a string prompt

        Returns:
            ChatCompletion | Completion: Response from the LLM
        """
        if isinstance(messages, str):
            prompt = messages
            if "max_completion_tokens" in kwargs:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
            text_response: Completion = self.client.completions.create(
                prompt=prompt, model=model, **kwargs
            )
            return text_response

        if not isinstance(messages, list):
            raise ValueError("messages must be a list of dicts or a string")

        _messages = []
        for message in messages:
            _messages.append(OPENAI_MESSAGE_CLASS[message["role"]](**message))

        chat_response: ChatCompletion = self.client.chat.completions.create(
            messages=_messages, model=model, **kwargs
        )
        return chat_response
