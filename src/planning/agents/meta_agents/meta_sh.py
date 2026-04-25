"""
SH (Single-step Horizon) Meta Agent implementation.

This module implements a step-by-step planning paradigm for Meta Agents that uses
built-in function calling from LLM APIs. Unlike ReAct, it does NOT generate
"thought" text - it only generates actions and processes observations.

The agent supports two output modes:
1. Native tool calls (use_builtin_tool_output=True): Extract from response.choices[0].message.tool_calls
2. Structured output (use_builtin_tool_output=False): Parse action from JSON schema response
"""

from functools import partial
from textwrap import indent
from time import time
from typing import Any, Callable, Optional
import json
import logging
import uuid

from ...environment import Environment
from ..base_agent import BaseMetaAgent, LLMResponse
from ..exceptions import AgentException
from ..executable import ExecutionResult
from ..memory import ContextMemory
from ..step import Step, StepStatus
from ..utils import (
    update_params_with_processed_values,
    format_param_substitutions,
    process_finish_answer,
    validate_and_normalize_step_references,
)
from ..llm_utils import (
    default_json_response_preprocessor,
    default_response_preprocessor,
    generate_plan_with_retries,
    prepare_llm_input,
    prepare_llm_options,
    update_token_usage,
)

logger = logging.getLogger(__name__)


class SHMetaAgent(BaseMetaAgent):
    """
    Single-step Horizon (SH) Meta Agent that coordinates Worker Agents through plan-execute-observe loops.

    The agent follows a plan-execute-observe loop where it:
    1. Plans the next sub-task and determines which agent to use (no thought generation)
    2. Executes by calling a Worker Agent through the environment
    3. Observes the result and incorporates it into the next iteration

    This continues until it reaches a final answer or hits the maximum number of steps.

    Key differences from ReActMetaAgent:
    - No "thought" text generation or parsing
    - Always uses built-in function calling for input (use_builtin_tool_input=True)
    - Supports two output modes: native tool calls or structured JSON
    - Method names: _plan_step (instead of _think) and _execute_step (instead of _act)
    """

    def __init__(
        self,
        llm_call: Callable,
        llm_options: dict[str, Any] = {},
        name: str = "SHMetaAgent",
        max_steps: int = 10,
        max_retries: int = 3,
        use_builtin_tool_input: bool = True,
        use_builtin_tool_output: bool = False,
        plan_step_params: dict[str, Any] = {},
        tool_formatter_id: str = "json",
        normalize_step_references: bool = True,
        **kwargs,
    ):
        """
        Initialize the SH Meta Agent.

        Args:
            llm_call: Callable function to get LLM responses
            llm_options: Options to configure the LLM behavior
            name: Name of the agent
            max_steps: Maximum number of steps before stopping execution
            max_retries: Maximum number of retries for LLM generation when parsing fails
            use_builtin_tool_input: Whether to use built-in tool input (True) or embed in messages (False)
            use_builtin_tool_output: Whether to use built-in tool output (native tool calls)
            plan_step_params: Configuration for plan step generation
            tool_formatter_id: ID of formatter to use when embedding tools (default: "json")
            normalize_step_references: Whether to normalize step references (default: True)
        """
        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            max_steps=max_steps,
            use_builtin_tool_input=use_builtin_tool_input,
            use_builtin_tool_output=use_builtin_tool_output,
            **kwargs,
        )
        self.plan_step_params = plan_step_params
        self.max_retries = max_retries
        self.tool_formatter_id = tool_formatter_id
        self.normalize_step_references = normalize_step_references
        self._messages = []  # Store messages for LLM calls

    def run_episode(
        self,
        query_or_params: str | dict[str, Any],
        environment: Environment,
        task_config: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        memory: Optional[ContextMemory] = None,
        demonstration_candidates: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Run a complete SH episode to coordinate Worker Agents to solve the given query.

        Args:
            query_or_params: The query (str) or question to solve.
                             SH Meta Agent expects a string query.
            environment: Environment providing agents and state
            session_id: Optional session identifier
            parent_session_id: Optional parent session identifier
            memory: Optional pre-existing memory context
            demonstration_candidates: Optional list of pre-retrieved similar examples for in-context learning.
                                    Each candidate should have a 'text' field with pre-formatted demonstration.
            **kwargs: Additional parameters

        Returns:
            ExecutionResult: Episode results including final_answer, success, steps, etc.
        """
        # SH Meta Agent expects string query
        if not isinstance(query_or_params, str):
            raise ValueError(f"SH Meta Agent expects string query, got {type(query_or_params)}")
        query = query_or_params

        start_time = time()

        # Initialize memory if not provided
        if memory is None:
            session_id = session_id or f"meta_sh_{uuid.uuid4().hex[:8]}"
            memory = ContextMemory(
                session_id=session_id,
                agent_name=self.name,
                query=query,
                parent_session_id=parent_session_id,
            )
            self._messages = []  # Reset messages for new episode

        # Register memory in environment
        environment.register_memory(memory)

        success = False
        retry_count = 0
        final_answer = ""
        llm_token_usage = {}

        try:
            while len(memory.step_history) < self.max_steps:
                # Check if maximum failed steps reached
                if retry_count > self.max_retries:
                    error_message = "Maximum failed steps reached without finding answer"
                    raise AgentException(error_message, agent_name=self.name)

                current_step = len(memory.step_history)
                logger.debug(f"SH Meta Agent step {current_step}")
                # Plan: Generate next step using LLM (no thought, only action)

                try:
                    step, step_token_usage = self._plan_step(
                        environment,
                        memory,
                        demonstration_candidates=demonstration_candidates,
                        max_retries=self.max_retries,
                    )

                    # Extract action details
                    action_data = step.get("action", {})
                    action_name = action_data.get("name", "")
                    action_args = action_data.get("arguments", {})
                    logger.info(f"Action: {action_name}[{action_args}]")

                    # Update token usage
                    update_token_usage(llm_token_usage, step_token_usage)

                except KeyboardInterrupt:
                    logger.info(f"Step generation interrupted at step {current_step}")
                    raise
                except AgentException as e:
                    logger.error(f"Agent error during step generation {current_step}: {e}")
                    final_answer = f"Agent coordination failed: {str(e)}"
                    success = False
                    break

                # Check if the action is to finish
                if action_name == "finish":
                    final_answer_raw = action_args.get("answer", "")

                    # Process finish answer (handles memory references and kopl postprocessing)
                    _final_answer = process_finish_answer(final_answer_raw, environment, memory)

                    if not _final_answer.success:
                        # Processing failed (e.g., invalid memory reference)
                        step.status = StepStatus.FAILED
                        observation_text = f"# {_final_answer.error_message}"
                        step.set("observation", observation_text)

                        logger.info(f"Step {current_step} finish action failed: {_final_answer.error_message}")

                        # Append error observation to message history for retry
                        if self.use_builtin_tool_output:
                            tool_call_id = step.get_metadata("tool_call_id")
                            self._messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": observation_text,
                                }
                            )
                        else:
                            # For structured output mode
                            self._messages.append({"role": "user", "content": observation_text})

                        # Continue loop to retry with corrected action
                        retry_count += 1
                        continue

                    # Success - set final answer
                    final_answer = _final_answer.processed_answer
                    success = True

                    step.status = StepStatus.COMPLETED
                    step.set("observation", final_answer)
                    break

                # Execute: Execute action through environment
                result = self._execute_step(step, environment, memory, task_config=task_config)
                if result.success:
                    observation = str(result.result_data) if result.result_data is not None else ""
                    step.status = StepStatus.COMPLETED
                else:
                    observation = result.error_message if result.error_message is not None else "Execution failed"
                    step.status = StepStatus.FAILED
                    retry_count += 1

                # Update parameters with processed values and track substitutions
                raw_params = action_args.copy()
                if "processed_params" in result.metadata:
                    raw_params, action_args = update_params_with_processed_values(
                        action_args, result.metadata["processed_params"]
                    )

                # Update step with action and updated arguments
                step.set(
                    "action",
                    {
                        "name": action_name,
                        "arguments": raw_params,
                        "updated_arguments": action_args,
                    },
                )

                # Store metadata from execution result
                for key, val in result.metadata.items():
                    if key == "full_data":
                        # Goes to the main data field
                        step.set("full", result.metadata["full_data"])
                        continue
                    step.set_metadata(key, val)

                # Update step with observation
                observation_text = indent(observation, "# ")

                # Add substitution information if parameters were changed
                substitution_info = format_param_substitutions(raw_params, action_args)
                if substitution_info:
                    observation_text += f"\n{substitution_info}"

                if result.success:
                    # Add step reference
                    observation_text += f"\n${current_step} = {action_name}[{action_args}]"

                step.set("observation", observation_text)

                logger.info(step)

                # Append observation to message history
                if self.use_builtin_tool_output:
                    tool_call_id = step.get_metadata("tool_call_id")
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": observation_text,
                        }
                    )
                else:
                    # For structured output mode
                    self._messages.append({"role": "user", "content": observation_text})

            if not success:
                num_executed_steps = len(memory.step_history)
                if num_executed_steps >= self.max_steps:
                    logger.warning(f"Reached max_steps ({self.max_steps})")
                    final_answer = f"Maximum steps ({self.max_steps}) reached"
                else:
                    final_answer = "Episode terminated without success"

        except KeyboardInterrupt:
            logger.info("SH Meta Agent episode interrupted")
            raise
        except AgentException as e:
            logger.error(f"Error during SH Meta Agent episode: {e}")
            final_answer = f"Episode failed with error: {str(e)}"
            success = False

        end_time = time()
        execution_time = end_time - start_time

        # Convert steps for serialization
        steps_list = [step.to_dict(exclude_large_data=True) for step in memory.step_history]

        return ExecutionResult.from_agent_execution(
            agent_name=self.name,
            success=success,
            result_data=final_answer,
            execution_time=execution_time,
            token_usage=llm_token_usage,
            step_count=len(memory.step_history),
            metadata={
                "max_steps_reached": len(memory.step_history) >= self.max_steps,
                "retry_count": retry_count,
                "max_retries_exceeded": retry_count > self.max_retries,
                "session_id": memory.session_id,
                "parent_session_id": memory.parent_session_id,
                "steps": steps_list,
            },
        )

    def _plan_step(
        self,
        environment: Environment,
        memory: ContextMemory,
        demonstration_candidates: Optional[list[dict[str, Any]]] = None,
        max_retries: int = 3,
    ) -> tuple[Step, dict[str, dict[str, int]]]:
        """
        Generate the next action for the given task based on history.

        Args:
            environment: The environment providing agents and state
            memory: Current execution memory
            demonstration_candidates: Optional list of pre-retrieved similar examples
            max_retries: Maximum number of retries for LLM generation when parsing fails

        Returns:
            Tuple of (step, token_usage)
        """
        # Initialize messages on first call
        tools = environment.get_available_agents(include_finish=True)
        if not self._messages:
            system_message, user_message = self._extract_plan_step_messages(memory)

            # Handle demonstrations
            demonstrations_text = ""
            num_demonstrations = self.plan_step_params.get("num_demonstrations", 0)
            if num_demonstrations > 0:
                # Validate that system message contains {demonstrations} placeholder
                if system_message is None or "{demonstrations}" not in system_message:
                    raise ValueError(
                        "meta_agent.plan_step_params.system must contain {demonstrations} placeholder "
                        f"when num_demonstrations={num_demonstrations} > 0"
                    )
                if demonstration_candidates is None or len(demonstration_candidates) < num_demonstrations:
                    raise ValueError(
                        f"Insufficient demonstration_candidates provided: "
                        f"required={num_demonstrations}, provided={len(demonstration_candidates) if demonstration_candidates is not None else 0}"
                    )

                # Select top k demonstrations (candidates are already sorted by similarity)
                selected = demonstration_candidates[:num_demonstrations]
                # Extract pre-formatted text and join with blank lines as separators
                demonstrations_text = "\n\n".join(candidate["text"] for candidate in selected)

            self._messages = prepare_llm_input(
                system_message,
                user_message,
                tools,
                self.use_builtin_tool_input,
                self.tool_formatter_id,
                demonstrations_text=demonstrations_text,
            )

        # Prepare LLM options
        tool_choice = "required" if self.use_builtin_tool_output else None
        response_format, tool_schema_path = None, None
        if not self.use_builtin_tool_output:
            response_format = self.plan_step_params["response_format"]
            tool_schema_path = self.plan_step_params["tool_schema_path"]
        llm_options = prepare_llm_options(
            tools=tools,
            use_builtin_tool_input=self.use_builtin_tool_input,
            use_builtin_tool_output=self.use_builtin_tool_output,
            tool_choice=tool_choice,
            response_format=response_format,
            tool_schema_path=tool_schema_path,
        )

        # Call LLM with retry and custom feedback logic
        response_preprocessor = default_json_response_preprocessor
        if self.use_builtin_tool_output:
            # default response processor does nothing
            response_preprocessor = default_response_preprocessor

        step, token_usage = generate_plan_with_retries(
            messages=self._messages,
            llm_call=self.get_llm_response,
            llm_options=llm_options,
            response_preprocessor=response_preprocessor,
            response_parser=partial(self._parse_step_response, memory=memory),
            result_logger=partial(self._store_step_in_memory, memory=memory),
            agent_name=self.name,
            error_context="plan step",
            max_retries=max_retries,
            use_builtin_tool_output=self.use_builtin_tool_output,
        )

        assert isinstance(step, Step)  # for type checker
        step.set_metadata("token_usage", token_usage)
        return step, token_usage

    def _extract_plan_step_messages(self, memory: ContextMemory) -> tuple[Optional[str], str]:
        """Extract and validate system and user messages.

        Returns:
            Tuple of (system_message, user_message)
        """
        if not isinstance(self.plan_step_params, dict):
            raise ValueError("plan_step_params must be a dict")

        system_message = self.plan_step_params.get("instructions", self.plan_step_params.get("system"))
        user_message = self.plan_step_params.get("input", self.plan_step_params.get("user"))

        if user_message is None:
            raise ValueError("meta_agent.plan_step_params must contain 'user' key")
        if "{query}" not in user_message:
            raise ValueError("meta_agent.plan_step_params.user must contain {query}")

        user_message = user_message.format(query=memory.query)
        return system_message, user_message

    def _execute_step(
        self,
        step: Step,
        environment: Environment,
        memory: ContextMemory,
        task_config: Optional[dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute an agent action through the environment.

        Args:
            step: The step to execute
            environment: The environment to execute in
            memory: Current execution memory
            task_config: Optional task configuration to pass to worker agents

        Returns:
            ExecutionResult: Result of the agent execution
        """
        # Execute agent through environment
        action = step.get("action", {})
        action_name = action.get("name", "")
        action_args = action.get("arguments", {})
        result = environment.execute_agent(
            action_name,
            action_args,
            parent_session_id=memory.session_id,
            step_index=step.step_num,
            task_config=task_config,
        )
        return result

    def _parse_step_response(
        self,
        response: LLMResponse,
        memory: Optional[ContextMemory] = None,
    ) -> Step:
        """
        Parse LLM response into action name and action arguments.

        Expected formats:
        1. Native tool calls mode: response object with tool_calls
        2. Structured JSON mode: {"action": {"name": <name>, "arguments": <args>}}

        Args:
            response: The response from the LLM (string or response object)
            memory: Current execution memory for validation

        Returns:
            Tuple of (action_name, action_args)

        Raises:
            AgentException: If parsing fails or format is invalid
        """
        action_name = ""
        action_args = {}
        tool_call_id = None

        # Check if this is a native tool call response
        if self.use_builtin_tool_output:
            if response.tool_calls is None or len(response.tool_calls) == 0:
                raise AgentException("No tool calls found in response", agent_name=self.name)
            # Take the first tool call
            tool_call = response.tool_calls[0]
            tool_call_id = tool_call["call_id"]
            action_name = tool_call["name"]
            try:
                action_args = json.loads(tool_call["arguments"])
            except json.JSONDecodeError as e:
                error_message = indent(str(e), "> ")
                raise AgentException(
                    f"Invalid JSON in tool call arguments: {tool_call['arguments']}\n{error_message}",
                    agent_name=self.name,
                )
        else:
            response_text = response.content.strip()
            # Assume that the response is already cleaned to be JSON
            try:
                parsed_response = json.loads(response_text)
            except json.JSONDecodeError as e:
                # If JSON parsing fails, this is a programming error
                raise ValueError(f"Failed to parse response as JSON: {str(e)}")

            # Extract action (no thought field)
            action = parsed_response.get("action", {})
            if not isinstance(action, dict):
                raise AgentException(
                    f"Action must be a dict, got {type(action)} in {response}",
                    agent_name=self.name,
                )

            action_name = action.get("name", "").strip()
            action_args = action.get("arguments", {})

            if not action_name:
                raise AgentException(f"Action name is missing.\n> {response}")

            # Parse arguments if they're a string
            if isinstance(action_args, str):
                try:
                    action_args = json.loads(action_args)
                except json.JSONDecodeError:
                    # If it's not JSON, treat as single argument
                    action_args = {"value": action_args}

            if not isinstance(action_args, dict):
                raise AgentException(
                    f'Action arguments must be a dictionary of key-value pairs ({{"arg1": "val1", "arg2": "val2"}}).\n> {response}'
                )

        # Validate and normalize step references in arguments
        if memory is not None:
            action_args, errors = validate_and_normalize_step_references(
                action_args=action_args,
                current_step_index=len(memory.step_history),
                memory=memory,
                check_has_data=True,
                normalize=self.normalize_step_references,
            )
            if errors:
                raise AgentException("\n".join(errors), agent_name=self.name)

        # Remove namespace used in OpenAI API tool calls
        if self.use_builtin_tool_input:
            action_name = action_name.split(".", 1)[-1]

        # Create a new Step object with the parsed action and arguments
        step = Step(
            step_num=len(memory.step_history) if memory else 0,
            step_type="sh",
            status=StepStatus.PLANNED,
            data={"action": {"name": action_name, "arguments": action_args}},
        )

        if tool_call_id:
            step.set_metadata("tool_call_id", tool_call_id)

        reasoning_summary = response.get_reasoning_summary()
        if reasoning_summary:
            step.set_metadata("reasoning", reasoning_summary)

        return step

    def _store_step_in_memory(
        self,
        response: LLMResponse,
        step: Step | list[Step],
        memory: ContextMemory,
    ) -> None:
        """Store the planned step in memory."""
        assert isinstance(step, Step)  # for type checker
        # Add the planned step to memory
        memory.add_step(step)
