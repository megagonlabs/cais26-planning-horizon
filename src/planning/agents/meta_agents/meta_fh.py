"""
FH (full-horizon) Meta Agent implementation.

This module implements an upfront planning paradigm for Meta Agents that generates
a complete upfront plan using structured output, then executes all steps sequentially.
Unlike SH (Single-step Horizon) Meta Agent, it generates the entire plan before execution begins.

The agent supports only structured JSON output for plan generation (no native function calling).
If execution fails, it revises the plan based on the execution log and retries.
"""

from functools import partial
from textwrap import indent
from time import time
from typing import Any, Callable, Optional, cast
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
    generate_plan_with_retries,
    default_json_response_preprocessor,
    prepare_llm_input,
    prepare_llm_options,
    update_token_usage,
)

logger = logging.getLogger(__name__)


class FHMetaAgent(BaseMetaAgent):
    """
    Full-Horizon (FH) Meta Agent that coordinates Worker Agents through upfront planning.

    The agent follows a full-horizon planning pattern where it:
    1. Decomposes the task into a complete upfront plan with agent assignments
    2. Executes the plan step by step, replacing placeholders with previous results
    3. Revises the plan if execution fails, up to a maximum retry limit

    This continues until it reaches a final answer or hits the maximum number of retries.

    Key differences from SHMetaAgent:
    - Generates complete plan upfront (not iterative)
    - Only supports structured JSON output (no native function calling)
    - Implements plan revision logic on execution failure
    """

    def __init__(
        self,
        llm_call: Callable,
        llm_options: dict[str, Any] = {},
        name: str = "FHMetaAgent",
        max_steps: int = 10,
        max_retries: int = 3,
        plan_params: dict[str, Any] = {},
        revision_params: dict[str, Any] = {},
        tool_formatter_id: str = "json",
        use_builtin_tool_input: bool = False,
        normalize_step_references: bool = True,
        **kwargs,
    ):
        """
        Initialize the FH Meta Agent.

        Args:
            llm_call: Callable function to get LLM responses
            llm_options: Options to configure the LLM behavior
            name: Name of the agent
            max_steps: Maximum number of steps in plan before stopping execution
            max_retries: Maximum number of retries for plan generation and execution
            plan_params: Configuration for initial plan generation
            revision_params: Configuration for plan revision
            tool_formatter_id: ID of formatter to use when embedding tools (default: "json")
        """
        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            max_steps=max_steps,
            use_builtin_tool_input=use_builtin_tool_input,
            use_builtin_tool_output=False,  # FH does not use native tool output
            **kwargs,
        )
        self.plan_params = plan_params
        self.revision_params = revision_params
        self.max_retries = max_retries
        self.tool_formatter_id = tool_formatter_id
        self.normalize_step_references = normalize_step_references
        self._messages = []  # Store messages for LLM calls
        self._plan_history = []  # Store all generated plans

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
        Run a complete FH episode to coordinate Worker Agents.

        Args:
            query_or_params: The query (str) or question to solve.
                           FH Meta Agent expects a string query.
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
        # FH Meta Agent expects string query
        if not isinstance(query_or_params, str):
            raise ValueError(f"FH Meta Agent expects string query, got {type(query_or_params)}")
        query = query_or_params

        start_time = time()

        # Initialize memory if not provided
        if memory is None:
            session_id = session_id or f"meta_fh_{uuid.uuid4().hex[:8]}"
            memory = ContextMemory(
                session_id=session_id,
                agent_name=self.name,
                query=query,
                parent_session_id=parent_session_id,
            )
            self._messages = []
            self._plan_history = []  # Reset plan history for new episode
        else:
            raise NotImplementedError

        # Register memory in environment
        environment.register_memory(memory)

        success = False
        final_answer = ""
        retry_count = 0
        llm_token_usage = {}

        try:
            # Generate initial plan
            planned_steps, step_token_usage = self._plan(
                environment, memory, demonstration_candidates=demonstration_candidates, max_retries=self.max_retries
            )

            # Update token usage
            update_token_usage(llm_token_usage, step_token_usage)

            # Execute plan with retry logic
            while retry_count <= self.max_retries:
                # Execute the plan
                execution_success, final_answer, exec_token_usage = self._execute_plan(
                    planned_steps, environment, memory, task_config=task_config
                )
                num_executed_steps = len(memory.step_history)

                # Update token usage
                update_token_usage(llm_token_usage, exec_token_usage)

                # Check if execution succeeded
                ## Note: if num_executed_steps > max_steps, execution_success will be False
                if execution_success:
                    success = True
                    break

                # Check if maximum steps reached during execution
                ## Note: if num_executed_steps == max_steps, we will also terminate without replanning
                if num_executed_steps >= self.max_steps:
                    logger.warning(f"Reached max_steps ({self.max_steps}) during execution")
                    final_answer = f"Maximum steps ({self.max_steps}) reached"
                    break

                # Execution failed, try replanning
                retry_count += 1

                if retry_count <= self.max_retries:
                    logger.info(f"Plan execution failed on attempt {retry_count}, replanning...")
                    # Revise plan based on execution failure
                    planned_steps, step_token_usage = self._replan(
                        environment=environment,
                        memory=memory,
                    )

                    # Update token usage
                    update_token_usage(llm_token_usage, step_token_usage)
                else:
                    logger.info(f"Max retries ({self.max_retries}) reached")
                    final_answer = f"Plan execution failed after {self.max_retries} retries"
                    break

        except KeyboardInterrupt:
            logger.info("FH Meta Agent episode interrupted")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during FH Meta Agent episode: {e}")
            final_answer = f"Episode failed with unexpected error: {str(e)}"
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
                "retry_count": retry_count,
                "max_retries_exceeded": retry_count > self.max_retries,
                "session_id": memory.session_id,
                "parent_session_id": memory.parent_session_id,
                "steps": steps_list,
                "plan_history": self._plan_history,
                "total_plans_generated": len(self._plan_history),
            },
        )

    def _plan(
        self,
        environment: Environment,
        memory: ContextMemory,
        demonstration_candidates: Optional[list[dict[str, Any]]] = None,
        max_retries: int = 3,
    ) -> tuple[list[Step], dict[str, dict[str, int]]]:
        """
        Generate a complete upfront plan with agent assignments.

        Args:
            environment: The environment providing agents and state
            memory: Current execution memory
            demonstration_candidates: Optional list of pre-retrieved similar examples
            max_retries: Maximum number of retries for plan generation

        Returns:
            Tuple of (planned_steps, token_usage)

        Raises:
            AgentException: If plan generation fails after max retries
        """
        # Validate and extract parameters
        system_message, user_message = self._extract_plan_messages(memory)
        tools = environment.get_available_agents(include_finish=True)

        # Handle demonstrations
        demonstrations_text = ""
        num_demonstrations = self.plan_params.get("num_demonstrations", 0)
        if num_demonstrations > 0:
            # Validate that system message contains {demonstrations} placeholder
            if system_message is None or "{demonstrations}" not in system_message:
                raise ValueError(
                    "meta_agent.plan_params.system must contain {demonstrations} placeholder "
                    f"when num_demonstrations={num_demonstrations} > 0"
                )

            # Select and format demonstrations
            if demonstration_candidates:
                # Select top k demonstrations (candidates are already sorted by similarity)
                selected = demonstration_candidates[:num_demonstrations]
                # Extract pre-formatted text and join with blank lines as separators
                demonstrations_text = "\n\n".join(candidate["text"] for candidate in selected)
            else:
                logger.warning(f"num_demonstrations={num_demonstrations} but no demonstration_candidates provided")

        # Prepare LLM input
        self._messages = prepare_llm_input(
            system_message,
            user_message,
            tools,
            self.use_builtin_tool_input,
            self.tool_formatter_id,
            demonstrations_text=demonstrations_text,
        )

        # Prepare LLM options
        llm_options = prepare_llm_options(
            tools=tools,
            use_builtin_tool_input=self.use_builtin_tool_input,
            use_builtin_tool_output=self.use_builtin_tool_output,
            tool_choice=None,
            response_format=self.plan_params["response_format"],
            tool_schema_path=self.plan_params["tool_schema_path"],
            step_index_path=self.plan_params.get("step_index_path"),
            start_index=0,
        )

        # Call LLM with retry logic
        planned_steps, token_usage = generate_plan_with_retries(
            messages=self._messages,
            llm_call=self.get_llm_response,
            llm_options=llm_options,
            response_preprocessor=default_json_response_preprocessor,
            response_parser=partial(self._parse_plan_response, memory=memory, start_index=0),
            result_logger=self._store_plan_in_history,
            max_retries=max_retries,
            agent_name=self.name,
            error_context="plan",
            use_builtin_tool_output=self.use_builtin_tool_output,
        )

        planned_steps = cast(list[Step], planned_steps)
        return planned_steps, token_usage

    def _extract_plan_messages(self, memory: ContextMemory) -> tuple[Optional[str], str]:
        """Extract and validate system and user messages from plan_params.

        Returns:
            Tuple of (system_message, user_message)
        """
        if not isinstance(self.plan_params, dict):
            raise ValueError("plan_params must be a dict")

        system_message = self.plan_params.get("instructions", self.plan_params.get("system"))
        user_message = self.plan_params.get("input", self.plan_params.get("user"))

        if user_message is None:
            raise ValueError("meta_agent.plan_params must contain 'user' key")
        if "{query}" not in user_message:
            raise ValueError("meta_agent.plan_params.user must contain {query}")

        user_message = user_message.format(query=memory.query)
        return system_message, user_message

    def _store_plan_in_history(self, response: LLMResponse, planned_steps: Step | list[Step]) -> None:
        """Store generated plan in plan history.

        This is used as the result_logger callback for generate_plan_with_retries.
        It archives each plan (initial and revisions) along with any reasoning
        provided by the LLM.

        Args:
            response: LLM response containing the plan
            planned_steps: Parsed list of step objects
        """
        assert isinstance(planned_steps, list)  # for type checker

        self._plan_history.append(
            {
                "steps": [step.to_dict(exclude_large_data=True) for step in planned_steps],  # type: ignore
                "reasoning": response.get_reasoning_summary(),
            }
        )

    def _execute_plan(
        self,
        planned_steps: list[Step],
        environment: Environment,
        memory: ContextMemory,
        task_config: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, str, dict[str, dict[str, int]]]:
        """
        Execute the planned steps sequentially with placeholder replacement.

        Args:
            planned_steps: List of planned steps to execute
            environment: The environment to execute in
            memory: Current execution memory
            task_config: Optional task configuration

        Returns:
            Tuple of (success, final_answer, token_usage)
        """
        token_usage = {}
        final_answer = ""

        # Assign the results of previous successful steps to step_results
        for step in memory.step_history:
            if step.status != StepStatus.COMPLETED:
                continue
            observation = step.get("observation", "")
            # Extract the actual result (removing formatting)
            if observation.startswith("# "):
                lines = observation.split("\n")
                observation = lines[0][2:]  # Remove "# " prefix

        for step_info in planned_steps:
            # Check if maximum steps reached
            num_executed_steps = len(memory.step_history)
            if num_executed_steps >= self.max_steps:
                logger.warning(f"Reached max_steps ({self.max_steps}) during execution")
                return False, f"Maximum steps ({self.max_steps}) reached", token_usage

            action = step_info.get("action", {})
            action_name = action.get("name", "")
            action_args = action.get("arguments", {})

            # Create step number (0-indexed)
            step_num = num_executed_steps

            if action_name == "finish":
                final_answer = action_args.get("answer", "")

                # Final answer step
                step = Step(
                    step_num=step_num,
                    step_type="fh",
                    status=StepStatus.PLANNED,
                )
                step.set(
                    "action",
                    {"name": "finish", "arguments": {"answer": final_answer}},
                )
                memory.add_step(step)

                # Process finish answer (handles memory references and kopl postprocessing)
                _final_answer = process_finish_answer(final_answer, environment, memory)

                if not _final_answer.success:
                    # Processing failed (e.g., invalid memory reference)
                    step.status = StepStatus.FAILED
                    step.set("observation", _final_answer.error_message)
                    logger.info(f"Step {step_num} failed: {_final_answer.error_message}")
                    return False, "", token_usage

                # Success - set final answer
                step.status = StepStatus.COMPLETED
                step.set("observation", _final_answer.processed_answer)
                return True, _final_answer.processed_answer, token_usage

            # Execute agent through environment
            logger.info(f"Executing step {step_num}: Agent '{action_name}' with args {action_args}")
            _final_answer = environment.execute_agent(
                action_name,
                action_args,
                parent_session_id=memory.session_id,
                step_index=step_num,
                task_config=task_config,
            )
            logger.info(f"Step {step_num} execution result: Success={_final_answer.success}")

            # Check if execution succeeded
            if not _final_answer.success:
                # Mark step as failed
                step = Step(
                    step_num=step_num,
                    step_type="fh",
                    status=StepStatus.FAILED,
                )
                step.set("action", {"name": action_name, "arguments": action_args})
                error_message = _final_answer.error_message if _final_answer.error_message else "Agent execution failed"
                step.set("observation", error_message)
                memory.add_step(step)

                # Return failure to trigger replanning
                logger.info(f"Step {step_num} failed: {error_message}")
                return False, "", token_usage

            # Execution succeeded
            observation = str(_final_answer.result_data)
            logger.info(f"Result data: {observation}")

            # Create step object
            step = Step(
                step_num=step_num,
                step_type="fh",
                status=StepStatus.COMPLETED,
            )

            # Update parameters with processed values
            raw_params = action_args.copy()
            if "processed_params" in _final_answer.metadata:
                raw_params, action_args = update_params_with_processed_values(
                    action_args, _final_answer.metadata["processed_params"]
                )

            step.set(
                "action",
                {
                    "name": action_name,
                    "arguments": raw_params,
                    "updated_arguments": action_args,
                },
            )

            # Store metadata from execution result
            for key, val in _final_answer.metadata.items():
                if key == "full_data":
                    # Goes to the main data field
                    step.set("full", _final_answer.metadata["full_data"])
                    continue
                step.set_metadata(key, val)

            # Update step with observation
            observation_text = indent(observation, "# ")

            # Add substitution information if parameters were changed
            substitution_info = format_param_substitutions(raw_params, action_args)
            if substitution_info:
                observation_text += f"\n{substitution_info}"

            # Add step reference
            observation_text += f"\n${step_num} = {action_name}[{action_args}]"

            step.set("observation", observation_text)

            memory.add_step(step)

            # Update token usage
            if _final_answer.token_usage:
                update_token_usage(token_usage, _final_answer.token_usage)

        # If we reach here without finding finish, that's a planning error
        logger.warning("Plan completed without finish action")
        return False, "", token_usage

    def _replan(
        self,
        environment: Environment,
        memory: ContextMemory,
        max_retries: int = 3,
    ) -> tuple[list[Step], dict[str, dict[str, int]]]:
        """
        Revise the plan based on execution failure.

        Reuses the initial plan's system and user messages, appends the previous plan
        as an assistant message, and adds a revision instruction with execution result.

        Args:
            environment: The environment providing agents and state
            memory: Current execution memory
            max_retries: Maximum number of retries for plan revision

        Returns:
            Tuple of (revised_steps, token_usage)
        """
        # Prepare revision prompt
        revision_user_message = self._prepare_revision_message(memory)
        self._messages.append({"role": "user", "content": revision_user_message})

        # Prepare LLM options (reuse from plan_params)
        tools = environment.get_available_agents(include_finish=True)
        llm_options = prepare_llm_options(
            tools=tools,
            use_builtin_tool_input=self.use_builtin_tool_input,
            use_builtin_tool_output=self.use_builtin_tool_output,
            tool_choice=None,
            response_format=self.plan_params["response_format"],
            tool_schema_path=self.plan_params["tool_schema_path"],
            step_index_path=self.plan_params.get("step_index_path"),
            start_index=len(memory.step_history),
        )

        # Call LLM with retry logic
        revised_steps, token_usage = generate_plan_with_retries(
            messages=self._messages,
            llm_call=self.get_llm_response,
            llm_options=llm_options,
            response_preprocessor=default_json_response_preprocessor,
            response_parser=partial(self._parse_plan_response, memory=memory, start_index=len(memory.step_history)),
            result_logger=self._store_plan_in_history,
            max_retries=max_retries,
            agent_name=self.name,
            error_context="plan revision",
            use_builtin_tool_output=self.use_builtin_tool_output,
        )

        revised_steps = cast(list[Step], revised_steps)
        return revised_steps, token_usage

    def _prepare_revision_message(self, memory: ContextMemory) -> str:
        """Prepare revision user message with execution results.

        Extracts the revision template from revision_params and formats it with
        the execution history and starting index for the revised plan.

        Args:
            memory: Current execution memory containing step history

        Returns:
            Formatted revision message string

        Raises:
            ValueError: If revision_params is invalid or missing required fields
        """
        if not isinstance(self.revision_params, dict):
            raise ValueError("revision_params must be a dict")

        revision_user_template = self.revision_params.get("input", self.revision_params.get("user"))
        if revision_user_template is None:
            raise ValueError("meta_agent.revision_params must contain 'user' key")

        execution_result = self._format_execution_result(memory)
        return revision_user_template.format(
            execution_result=execution_result,
            start_index=len(memory.step_history),
        )

    def _parse_plan_response(
        self,
        response: LLMResponse,
        memory: Optional[ContextMemory] = None,
        start_index: int = 0,
    ) -> list[Step]:
        """
        Parse LLM response into a list of planned steps.

        Args:
            response: LLM response containing the plan
            memory: Current execution memory for validation
            start_index: The starting index for step numbering (default 0)

        Returns:
            list[Step]: List of step objects with action information

        Raises:
            AgentException: If parsing fails
        """
        response_text = response.content.strip()
        # Assume that the response is already cleaned to be JSON
        try:
            parsed_response = json.loads(response_text)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, this is a programming error
            raise ValueError(f"Failed to parse response as JSON: {str(e)}")

        if not isinstance(parsed_response, dict):
            raise AgentException("Expected top-level response to be a JSON object", agent_name=self.name)

        steps_list = parsed_response.get("steps", [])
        if not isinstance(steps_list, list):
            raise AgentException("Expected 'steps' to be an array", agent_name=self.name)

        planned_steps = []
        errors = []
        for i, step_obj in enumerate(steps_list):
            if not isinstance(step_obj, dict):
                errors.append(f"Invalid step format at index {i}: {step_obj}")
                continue

            step_dict = cast(dict[str, Any], step_obj)

            index = step_dict.get("index")
            if not isinstance(index, int):
                errors.append(f"Step {i} has invalid or missing index: {step_obj}")
                continue

            if index != i + start_index:
                errors.append(f"Step {i} has incorrect index {index}, expected {i + start_index}")

            action = step_dict.get("action", {})
            if not isinstance(action, dict):
                errors.append(f"Invalid action format in step {i}: {step_obj}")
                continue

            action_name = action.get("name", "")
            action_args = action.get("arguments", {})

            if not action_name:
                errors.append(f"Action name is missing in step {i}: {step_obj}")
                continue

            if not isinstance(action_args, dict):
                errors.append(f"Action arguments must be a dictionary in step {i}: {step_obj}")
                continue

            # Validate and normalize step references in arguments
            if memory is not None:
                action_args, step_errors = validate_and_normalize_step_references(
                    action_args=action_args,
                    current_step_index=index,
                    memory=memory,
                    check_has_data=False,
                    normalize=self.normalize_step_references,
                )
                # Add step index context to error messages
                errors.extend([f"Step {i}: {err}" for err in step_errors])

            step = Step(
                step_num=index,
                step_type="fh",
                status=StepStatus.PLANNED,
            )
            step.set("action", {"name": action_name, "arguments": action_args})
            planned_steps.append(step)

        if not planned_steps:
            errors.append("No valid steps found in the response.")
        else:
            # Validate that plan ends with finish action
            last_step = planned_steps[-1]
            last_action = last_step.get("action", {})
            if last_action.get("name") != "finish":
                errors.append("The plan must end with a `finish` action.")

        if errors:
            raise AgentException("\n".join(errors), agent_name=self.name)

        return planned_steps

    def _format_execution_result(self, memory: ContextMemory) -> str:
        """
        Format execution history from memory for plan revision.

        Converts the step history into a readable format showing each step's
        action and observation. Used to provide execution context when requesting
        a revised plan from the LLM.

        Args:
            memory: Current execution memory containing step history

        Returns:
            Formatted string with step actions and observations
        """
        formatted_lines = []
        for step in memory.step_history:
            action = step.get("action", {})
            observation = step.get("observation", "")

            if isinstance(action, dict):
                action_name = action.get("name", "")
                action_args = action.get("arguments", {})
                formatted_lines.append(f"Step {step.step_num}:")
                formatted_lines.append(f"Action: {action_name}[{action_args}]")
            else:
                formatted_lines.append(f"Step {step.step_num}:")
                formatted_lines.append(f"Action: {action}")

            formatted_lines.append(f"Observation: {observation}")
            formatted_lines.append("")

        return "\n".join(formatted_lines).strip()
