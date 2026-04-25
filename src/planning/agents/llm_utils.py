"""
Utility functions for LLM interaction in Meta Agents.

This module provides common functions for interacting with LLMs including:
- Retry logic with error feedback
- Message preparation
- Tool schema handling
- LLM option preparation
"""

from copy import deepcopy
from textwrap import indent
from typing import Any, Callable, Optional
import json
import logging

from .base_agent import LLMResponse
from .exceptions import AgentException
from .step import Step

logger = logging.getLogger(__name__)


def default_response_preprocessor(response: LLMResponse) -> LLMResponse:
    """Default response preprocessor that returns the response as-is."""
    if not response.content and response.tool_calls is None:
        raise AgentException("Empty response")
    return response


def default_json_response_preprocessor(response: LLMResponse) -> LLMResponse:
    """Preprocessor that ensures the response content is valid JSON.

    Args:
        response: LLMResponse object

    Returns:
        LLMResponse object with validated JSON content

    Raises:
        AgentException: If the response content is not valid JSON
    """
    if not response.content and response.tool_calls is None:
        raise AgentException("Empty response")

    new_response = response.copy()
    try:
        decoder = json.JSONDecoder()
        # Take only the first JSON object in case of extra text
        parsed_response, _ = decoder.raw_decode(response.content)  # type: ignore

        # Update response content if modified
        parsed_response_text = json.dumps(parsed_response)
        if parsed_response_text != response.content:
            new_response.content = parsed_response_text
            if new_response.item_list:
                new_response.item_list[-1]["content"] = [
                    {
                        "annotations": [],
                        "text": parsed_response_text,
                        "type": "output_text",
                    }
                ]
    except json.JSONDecodeError as e:
        error_message = indent(str(e), "> ")
        raise AgentException(f"Found invalid JSON\n{error_message}")

    return new_response


def default_response_parser(response: LLMResponse) -> Step | list[Step]:
    """Default response parser that returns the response content as a Step object."""
    return Step(-1, step_type="default", data={"content": response.content})


def default_result_logger(
    response: LLMResponse, parsed_result: Step | list[Step]
) -> None:
    """Default result logger that logs the response and parsed result."""
    logger.info(f"LLM Response: {response.content}")
    logger.info(f"Parsed Result: {parsed_result}")


def append_error_feedback(
    messages: list,
    error_message: str,
    use_builtin_tool_output: bool = False,
) -> None:
    """
    Append error feedback to message history.

    Args:
        messages: Message list to append to
        error_message: Error message to include in feedback
        use_builtin_tool_output: Whether using builtin tool output mode
    """
    error_message = indent(error_message, "> ")
    feedback = (
        f"Your response is in an invalid format. Please read the instructions "
        f"carefully and try again.\n\nError:\n{error_message}"
    )

    if use_builtin_tool_output:
        tool_call_id = messages[-1]["tool_calls"][0]["id"]
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": feedback,
            }
        )
        return

    messages.append({"role": "user", "content": feedback})


def append_assistant_message(
    messages: list[dict[str, Any]],
    response: LLMResponse,
    use_builtin_tool_output: bool,
) -> None:
    """
    Append assistant message to message history based on response content.

    Args:
        messages: Message list to append to
        response: LLMResponse object
        use_builtin_tool_output: Whether using builtin tool output mode
    """
    # If response has item_list (Responses API), append them directly
    if response.item_list:
        messages += response.item_list
        return

    # If using builtin tool output, append tool calls
    if use_builtin_tool_output:
        assert response.tool_calls is not None
        tool_call = response.tool_calls[0]
        tool_call_id = tool_call["call_id"]
        message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": tool_call["arguments"],
                        },
                    }
                ],
            }
    else:  # Otherwise, append the raw content
        message = {"role": "assistant", "content": response.content}

    # Include extra content if available
    ## VertexAI API returns thought signature in extra_content
    if response.extra_content:
        message["extra_content"] = response.extra_content

    messages.append(message)


def generate_plan_with_retries(
    messages: list[dict[str, Any]],
    llm_call: Callable[[list], LLMResponse],
    llm_options: dict[str, Any],
    response_preprocessor: Callable[
        [LLMResponse], LLMResponse
    ] = default_response_preprocessor,
    response_parser: Callable[
        [LLMResponse], Step | list[Step]
    ] = default_response_parser,
    result_logger: Callable[
        [LLMResponse, Step | list[Step]], None
    ] = default_result_logger,
    max_retries: int = 3,
    agent_name: Optional[str] = None,
    error_context: str = "response",
    use_builtin_tool_output: bool = False,
) -> tuple[Step | list[Step], dict[str, dict[str, int]]]:
    """
    Call LLM with retry logic and feedback on parsing errors.

    Args:
        messages: Mutable message list for LLM. Output messages will be appended to this list.
        llm_call: Function to call LLM
        llm_options: Options for LLM call
        parser: Function to parse LLM response (should raise AgentException on error)
        max_retries: Maximum number of retries
        agent_name: Name of the agent for error messages
        error_context: Context description for error messages
        use_builtin_tool_output: Whether using builtin tool output mode

    Returns:
        Tuple of (parsed_result, token_usage)

    Raises:
        AgentException: If max retries exceeded or parsing fails
    """
    parsed_result: Step | list[Step] | None = None
    llm_token_usage = {}
    success = False

    while max_retries > 0:
        response = llm_call(messages, **llm_options)
        step_token_usage = response.token_usage

        # Update token usage
        update_token_usage(llm_token_usage, step_token_usage)

        response_processed = None
        try:
            response_processed = response_preprocessor(response)
            parsed_result = response_parser(response_processed)
            result_logger(response_processed, parsed_result)
            append_assistant_message(
                messages=messages,
                response=response_processed,
                use_builtin_tool_output=use_builtin_tool_output,
            )
            success = True
            break

        except KeyboardInterrupt:
            logger.info(f"{error_context.capitalize()} generation interrupted")
            raise
        except AgentException as e:
            logger.info(f"{error_context.capitalize()} generation failed: {e}")

            # Update messages with assistant message and error feedback
            _response = response_processed if response_processed else response
            append_assistant_message(
                messages=messages,
                response=_response,
                use_builtin_tool_output=use_builtin_tool_output,
            )
            append_error_feedback(
                messages=messages,
                error_message=str(e),
                use_builtin_tool_output=use_builtin_tool_output,
            )

            max_retries -= 1

    if not success:
        raise AgentException(
            f"Max retries exceeded while generating {error_context}",
            agent_name=agent_name,
        )

    assert parsed_result is not None

    return parsed_result, llm_token_usage


def prepare_llm_input(
    system_message: Optional[str],
    user_message: str,
    tools: list[dict[str, Any]],
    use_builtin_tool_input: bool,
    tool_formatter_id: str,
    demonstrations_text: str = "",
) -> list[dict[str, Any]]:
    """
    Prepare LLM input messages with optional tool embedding and demonstrations.

    Args:
        system_message: Optional system message
        user_message: User message
        tools: List of tool definitions
        use_builtin_tool_input: Whether to use built-in tool input
        tool_formatter_id: ID of formatter to use for embedding tools
        demonstrations_text: Optional pre-formatted demonstration text to embed in system message

    Returns:
        List of message dictionaries

    Raises:
        ValueError: If tool_definitions placeholder is missing when needed
    """
    from .tool_formatters import get_formatter

    # Handle tool definitions based on builtin_tool_input setting
    if not use_builtin_tool_input:
        formatter = get_formatter(tool_formatter_id)
        formatted_tools = formatter.format(tools)

        if system_message and "{tool_definitions}" in system_message:
            system_message = system_message.replace("{tool_definitions}", formatted_tools)
        elif user_message and "{tool_definitions}" in user_message:
            user_message = user_message.replace("{tool_definitions}", formatted_tools)
        else:
            raise ValueError(
                "When use_builtin_tool_input=False, either system or user message "
                "must contain '{tool_definitions}' placeholder"
            )

    # Handle demonstrations embedding in system message
    if system_message and "{demonstrations}" in system_message:
        system_message = system_message.replace("{demonstrations}", demonstrations_text)

    llm_input = []
    if system_message:
        llm_input.append({"role": "system", "content": system_message})
    llm_input.append({"role": "user", "content": user_message})

    return llm_input


def prepare_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Transform tool definitions to JSON Schema format for structured output.

    Args:
        tools: List of tool definitions in function calling format

    Returns:
        List of JSON Schema objects for tools
    """
    return [
        {
            "type": "object",
            "description": tool["function"]["description"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Tool name",
                    "enum": [tool["function"]["name"]],
                },
                "arguments": {
                    "type": "object",
                    "description": f"The arguments to pass to the {tool['function']['name']} action",
                    "properties": tool["function"]["parameters"]["properties"],
                    "additionalProperties": False,
                    "required": tool["function"]["parameters"]["required"],
                },
            },
            "additionalProperties": False,
            "required": ["name", "arguments"],
        }
        for tool in tools
    ]


def inject_tool_schemas_into_response_format(
    response_format: dict[str, Any],
    tool_schema_path: list[str],
    tool_schemas: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Inject tool schemas into response format at the specified path.

    Args:
        response_format: Response format schema
        tool_schema_path: Path to inject tool schemas
        tool_schemas: Tool schemas to inject

    Returns:
        Modified response format

    Raises:
        ValueError: If the tool_schema_path is invalid
    """
    current_node = response_format
    for path_key in tool_schema_path[:-1]:
        if path_key not in current_node:
            raise ValueError(
                f"invalid tool_schema_path: '{path_key}' not found in {response_format}"
            )
        current_node = current_node[path_key]

    current_node[tool_schema_path[-1]] = {"anyOf": tool_schemas}
    return response_format


def prepare_llm_options(
    tools: list[dict[str, Any]],
    use_builtin_tool_input: bool,
    use_builtin_tool_output: bool,
    tool_choice: Optional[str] = None,
    response_format: Optional[dict[str, Any]] = None,
    tool_schema_path: Optional[list[str]] = None,
    step_index_path: Optional[list[str]] = None,
    start_index: int = 0,
) -> dict[str, Any]:
    """
    Prepare LLM options including tool handling and response format.

    Args:
        tools: List of tool definitions
        use_builtin_tool_input: Whether to use built-in tool input
        use_builtin_tool_output: Whether to use built-in tool output
        tool_choice: Tool choice setting ("none", "required", etc.)
        response_format: Optional response format schema
        tool_schema_path: Optional path for injecting tool schemas

    Returns:
        Dictionary of LLM options
    """
    llm_options = {}

    # Tool input handling
    if use_builtin_tool_input:
        llm_options["tools"] = tools
        if tool_choice is not None:
            llm_options["tool_choice"] = tool_choice

    # Structured output handling
    if response_format is not None:
        # Create a deep copy to avoid mutating the original config
        # This prevents race conditions when multiple threads use the same config
        response_format = deepcopy(response_format)

        if tool_schema_path is not None:
            tool_schemas = prepare_tool_schemas(tools)
            response_format = inject_tool_schemas_into_response_format(
                response_format, tool_schema_path, tool_schemas
            )
        if step_index_path is not None:
            # Inject start_index into response format at step_index_path
            current_node = response_format
            for path_key in step_index_path[:-1]:
                if path_key not in current_node:
                    raise ValueError(
                        f"invalid step_index_path: '{path_key}' not found in {response_format}"
                    )
                current_node = current_node[path_key]
            current_node[step_index_path[-1]]["minimum"] = start_index
        llm_options["response_format"] = response_format

    return llm_options


def update_token_usage(
    cumulative_usage: dict[str, dict[str, int]],
    step_usage: dict[str, dict[str, int]],
) -> None:
    """
    Update cumulative token usage with step usage in-place.

    Args:
        cumulative_usage: Cumulative token usage dictionary to update
        step_usage: Token usage from current step to add
    """
    for model_name, token_info in step_usage.items():
        if model_name not in cumulative_usage:
            cumulative_usage[model_name] = token_info.copy()
        else:
            for key, val in token_info.items():
                if not isinstance(val, int):
                    # Skip non-integer values (e.g., nested dicts like reasoning_tokens)
                    continue
                cumulative_usage[model_name][key] = (
                    cumulative_usage[model_name].get(key, 0) + val
                )
