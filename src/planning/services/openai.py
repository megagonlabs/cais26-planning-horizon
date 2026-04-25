from typing import Optional, Union
from typing import Any
import os

from dotenv import load_dotenv
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam
)
from openai.types.responses import Response
import openai

OPENAI_MESSAGE_CLASS = {
    "user": ChatCompletionUserMessageParam,
    "assistant": ChatCompletionAssistantMessageParam,
    "system": ChatCompletionSystemMessageParam,
    "tool": ChatCompletionToolMessageParam
}


class OpenAIClient:
    """Wrapper for OpenAI API calls."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        **kwargs,
    ):
        if api_key:
            self.api_key = api_key
        else:
            load_dotenv(override=True)
            self.api_key = os.getenv("OPENAI_API_KEY")

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if organization:
            client_kwargs["organization"] = organization
        if timeout is not None:
            client_kwargs["timeout"] = float(timeout)
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries

        self.client = openai.OpenAI(**client_kwargs)

    def _call_chat_completion(
        self, messages: list[dict[str, str]], model: str, **kwargs
    ) -> ChatCompletion:
        """
        Call the Chat Completions API with the given messages.

        Args:
            messages: List of message dictionaries
            model: Model name
            **kwargs: Additional parameters for the API call

        Returns:
            ChatCompletion: Response from the Chat Completions API
        """
        _messages = []
        for message in messages:
            _messages.append(OPENAI_MESSAGE_CLASS[message["role"]](**message))

        return self.client.chat.completions.create(
            messages=_messages, model=model, **kwargs
        )

    def _call_responses(
        self, input_data: str | list[dict[str, str]], model: str, **kwargs
    ) -> Response:
        """
        Call the Responses API with the given input.

        Args:
            input_data: String input or list of message dictionaries
            model: Model name
            **kwargs: Additional parameters for the API call

        Returns:
            Response: Response from the Responses API
        """
        # Convert messages to input format if needed
        if isinstance(input_data, list):
            # Extract system message as instructions if present
            instructions = None
            input_messages = []
            for msg in input_data:
                if msg.get("role") == "system":
                    instructions = msg["content"]
                else:
                    input_messages.append(msg)

            # If we have a single user message, use it as a string
            if len(input_messages) == 1 and input_messages[0].get("role") == "user":
                input_item_list = input_messages[0]["content"]
            else:
                input_item_list = input_messages

            if instructions:
                kwargs["instructions"] = instructions
        else:
            input_item_list = input_data

        return self.client.responses.create(
            input=input_item_list, model=model, **kwargs
        )

    def _translate_to_responses_params(self, model_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Translate Chat Completions API parameters to Responses API parameters.

        Args:
            model_name: Name of the model being used
            kwargs: Parameters for Chat Completions API

        Returns:
            dict: Translated parameters for Responses API
        """
        translated = kwargs.copy()

        # Do not store the response by default
        if "store" not in translated:
            translated["store"] = False

        if model_name.startswith("gpt-5"):
            if "include" not in translated:
                translated["include"] = ["reasoning.encrypted_content"]
            elif "reasoning.encrypted_content" not in translated["include"]:
                translated["include"].append("reasoning.encrypted_content")

        # Translate max_completion_tokens to max_output_tokens
        if "max_completion_tokens" in translated:
            translated["max_output_tokens"] = translated.pop("max_completion_tokens")

        # Translate reasoning_effort to reasoning.effort
        if "reasoning_effort" in translated:
            reasoning_effort = translated.pop("reasoning_effort")
            if "reasoning" in translated:
                raise ValueError(
                    "Cannot specify both reasoning_effort and reasoning parameters"
                )
            translated["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}

        # Translate response_format to text.format for structured outputs
        if "response_format" in translated:
            response_format = translated.pop("response_format")
            if "text" not in translated:
                translated["text"] = {}
            translated["text"]["format"] = {
                "type": "json_schema",
                **response_format["json_schema"]
            }

        # Translate the format of tool definitions
        if "tools" in translated:
            tools = translated.pop("tools")
            functions = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"].copy()
                    functions.append({"type": "function", **func})
            if functions:
                translated["tools"] = functions

        # tool_choice remains the same
        # Other parameters like temperature, max_tokens, etc. remain the same

        return translated

    def call(
        self,
        messages: Optional[Union[list[dict[str, str]], str]] = None,
        model: Optional[str] = None,
        api_type: Optional[str] = None,
        **kwargs
    ) -> ChatCompletion | Response:
        """
        Call the OpenAI API (Chat Completions or Responses API).

        Automatically detects which API to use based on:
        1. Explicit api_type parameter
        2. Presence of 'input' in kwargs (indicates Responses API)
        3. Default to Chat Completions for backward compatibility

        Args:
            messages: List of message dictionaries or string input (for Responses API)
            model: Model name
            api_type: Explicitly specify 'chat' or 'responses'
            **kwargs: Additional parameters for the API call

        Returns:
            ChatCompletion | Response: Response from the appropriate API
                (Any is used for Response type until correct import is verified)
        """
        # Handle the case where 'input' is passed as a kwarg (Responses API style)
        if messages is None and "input" in kwargs:
            messages = kwargs.pop("input")
            if api_type is None:
                api_type = "responses"

        # Validate that we have messages at this point
        if messages is None:
            raise ValueError("Either 'messages' or 'input' must be provided")

        # Validate that we have a model
        if model is None:
            raise ValueError("'model' parameter is required")

        # Detect API type if not explicitly specified
        if api_type is None:
            # Check if any Responses-specific parameters are present
            if any(key in kwargs for key in ["instructions", "previous_response_id", "store"]):
                api_type = "responses"
            # Check if messages is a simple string (common for Responses API)
            elif isinstance(messages, str):
                api_type = "responses"
            else:
                # Default to Chat Completions for backward compatibility
                api_type = "chat"

        # Route to appropriate API
        if api_type == "responses":
            # Translate parameters if needed
            translated_kwargs = self._translate_to_responses_params(model, kwargs)
            return self._call_responses(messages, model, **translated_kwargs)
        else:
            # Chat Completions API - validate messages is list
            if not isinstance(messages, list):
                raise ValueError(
                    "Chat Completions API requires messages to be a list of dicts"
                )
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


# def batch_completion(
#     client: OpenAIClient,
#     batch: Sequence[dict[str, str]],
#     n_workers: Optional[int] = 100,
#     show_progress: Optional[bool] = False,
#     **kwargs,
# ) -> list[ChatCompletion]:
#     """
#     Generate completions for a batch of conversations.

#     Args:
#         client: OpenAI client wrapper.
#         batch: Iterable of message lists (one per request).
#         n_workers: Max parallel workers.
#         show_progress: Whether to show a tqdm progress bar.
#         **kwargs: Extra options forwarded to the OpenAI call (e.g., model).

#     Returns:
#         list: A list of OpenAI responses, in the same order as `batch`.
#     """
#     if n_workers == 1:
#         return [
#             client.call(**model_input, **kwargs)
#             for model_input in tqdm(batch, desc="Generating completions")
#         ]
#     with ThreadPoolExecutor(max_workers=n_workers) as executor:
#         futures = [
#             executor.submit(client.call, **model_input, **kwargs)
#             for model_input in batch
#         ]
#         if show_progress:
#             futures = tqdm(futures, total=len(batch), desc="Generating completions")
#         return [future.result() for future in futures]
