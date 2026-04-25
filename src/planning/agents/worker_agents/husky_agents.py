"""
Husky-style fixed-workflow Worker Agents.

This module provides the HuskyAgent base class for implementing simple,
single-pass workflow agents that follow a generate → execute_tool → synthesize_answer
pattern. These agents provide a consistent baseline for comparison with more
advanced iterative agents.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from time import time
from typing import Any, Callable, Optional, TYPE_CHECKING
import json
import logging
import re

from ..base_agent import BaseWorkerAgent
from ..exceptions import AgentException
from ..executable import ExecutionResult, ExecutableType
from ..llm_utils import update_token_usage
from ..memory import ContextMemory
from ..step import Step, StepStatus

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from ...environment.environment import Environment

logger = logging.getLogger(__name__)


@dataclass
class HuskyStepResult:
    """Information for creating a Step after sub-function execution."""

    result: str  # Primary result for workflow chaining
    success: bool = True  # Operation success status
    input_data: Any = None  # For logging/debugging
    output_data: Any = None  # For logging/debugging
    metadata: dict[str, Any] = field(default_factory=dict)


STEP_STR_TEMPLATE = """
Step: {action_str}
Status: {status}
Output{note}:
{observation_str}
""".strip()


def step_formatter_for_husky(step: Step) -> str:
    """Format a Step for Husky agent history."""
    action = step.data["action"]
    action_name = action["name"]
    action_args = action["arguments"]
    action_str = f"{action_name}[{json.dumps(action_args)}]"

    observation_str = step.data.get("observation", "").strip()
    if len(observation_str) == 0:
        observation_str = "[no output]"

    if step.status == StepStatus.COMPLETED:
        status = "Completed"
        note = f"(The output can be referenced as ${step.step_num} in later steps)"
    else:
        status = "Failed"
        note = ""

    return STEP_STR_TEMPLATE.format(
        action_str=action_str,
        status=status,
        note=note,
        observation_str=observation_str,
    )


DEFAULT_SYNTHESIS_PROMPT = """
You are evaluating a response to determine if it successfully answers the question.

Question: {query}

Response to evaluate: {response}

Your task:
1. Determine if the response successfully provides a direct answer to the question
2. If successful, extract the concise answer value without verbose phrases
3. If not successful, provide a concise reason for failure based on the response content

Guidelines for failure reasons:
- "Insufficient information provided" - when the input lacks necessary details
- "Generation incomplete due to constraints" - when response was cut off
- "Search results didn't contain supporting information" - for search-based tasks
- "Code execution failed" - when code produced errors
- "Answer not found in response" - when no clear answer is present

Return your evaluation in the specified format.
""".strip()


def extract_boxed_answer(s: str) -> str:
    """Remove LaTeX boxed content from a string, handling nested braces. (obsolete)"""
    from pylatexenc.latex2text import LatexNodes2Text

    # Find all \boxed{ positions
    boxed_starts = []
    for match in re.finditer(r"\\boxed\{", s):
        boxed_starts.append(match.start())

    if not boxed_starts:
        raise ValueError("No boxed content (\\boxed{...}) found")

    values = []
    for start in boxed_starts:
        # Start after \boxed{
        pos = start + len("\\boxed{")
        brace_count = 1  # We've consumed the opening {

        while pos < len(s) and brace_count > 0:
            if s[pos] == "{":
                brace_count += 1
            elif s[pos] == "}":
                brace_count -= 1
            pos += 1

        if brace_count != 0:
            raise ValueError("Malformed boxed content: unmatched braces")

        # Extract the content inside the braces
        content = s[
            start + len("\\boxed{") : pos - 1
        ]  # pos - 1 to exclude the closing }
        ## Convert LaTeX to plain text
        content = LatexNodes2Text().latex_to_text(content)

        values.append(content.strip())

    if len(values) == 1:
        return values[0]
    return "[" + ", ".join(values) + "]"


class HuskyAgent(BaseWorkerAgent):
    """
    Base class for Husky-style fixed-workflow Worker Agents.

    HuskyAgent implements a simple, single-pass workflow:
    1. generate() - Generate initial response/plan using LLM
    2. execute_tool() - Execute tool if needed (default: pass-through)
    3. synthesize_answer() - Synthesize final answer from results

    This provides a consistent baseline workflow for all Husky agents.
    Concrete implementations override specific methods as needed.
    """

    def __init__(
        self,
        llm_call: Optional[Callable] = None,
        llm_options: dict[str, Any] = {},
        name: str = "HuskyAgent",
        max_steps: int = 1,
        generation_prompt: str = "",
        synthesis_prompt: str = "",
    ):
        """
        Initialize the Husky Agent with fixed single-pass workflow.

        Args:
            llm_call: Callable function to get LLM responses
            llm_options: Options to configure the LLM behavior
            name: Name identifier for the agent
            max_steps: Maximum number of steps (default: 1 for single-pass)
            generation_prompt: Prompt template for generation step
            synthesis_prompt: Prompt template for synthesis step
        """
        super().__init__(llm_call, llm_options, name, max_steps)
        self.generation_prompt = generation_prompt
        self.synthesis_prompt = (
            synthesis_prompt if synthesis_prompt else DEFAULT_SYNTHESIS_PROMPT
        )

    def _prepare_context(
        self, memory: ContextMemory, environment: "Environment"
    ) -> dict[str, Any]:
        """
        Extract all context needed for workflow execution.

        Args:
            memory: The memory context for the current session
            environment: The environment providing context

        Returns:
            dict with keys: query, parent_query, history
        """
        context = {
            "query": memory.query,
            "parent_query": "",
            "history": "",
        }

        if memory.parent_session_id:
            parent_memory = environment.get_memory(memory.parent_session_id)
            if parent_memory:
                context["parent_query"] = parent_memory.query
                context["history"] = parent_memory.get_formatted_history(
                    include_planned=False, step_formatter=step_formatter_for_husky
                )
        if len(context["history"]) == 0:
            context["history"] = "[none]"

        return context

    def _format_prompt(self, prompt: str | dict, **format_vars) -> str | list[dict]:
        """
        Format prompt template with variables.

        Args:
            prompt: Prompt template (string or dict with system/user keys)
            **format_vars: Variables to substitute in the template

        Returns:
            Formatted prompt (string for simple templates, list of messages for dict templates)

        Raises:
            ValueError: If prompt format is invalid or missing required placeholders
        """
        if isinstance(prompt, str):
            if not prompt or "{query}" not in prompt:
                raise ValueError("Invalid prompt: missing {query} in prompt")

            formatted = prompt
            for key, value in format_vars.items():
                formatted = formatted.replace(f"{{{key}}}", str(value))
            return formatted

        elif isinstance(prompt, dict):
            system_message = prompt.get("system", "").strip()
            user_message = prompt.get("user", "").strip()

            if not user_message or "{query}" not in user_message:
                raise ValueError("Invalid prompt: missing {query} in user message")

            # Format user message
            formatted_user = user_message
            for key, value in format_vars.items():
                formatted_user = formatted_user.replace(f"{{{key}}}", str(value))

            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": formatted_user})

            return messages
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

    def _create_step(
        self,
        memory: ContextMemory,
        step_label: str,
        initial_data: dict[str, Any],
    ) -> Step:
        """
        Create a PLANNED step and add it to memory.

        Args:
            memory: Memory to add the step to
            step_label: Label for the step (e.g., "generate", "execute", "synthesis")
            initial_data: Initial data for the step (typically just input)

        Returns:
            Step: The created step (can be updated later)
        """
        step = Step(
            step_num=len(memory.step_history),
            step_type=f"{self.name.lower()}_{step_label}",
            data=initial_data,
            status=StepStatus.PLANNED,
        )
        memory.add_step(step)
        return step

    def _update_step_with_result(
        self,
        step: Step,
        step_result: HuskyStepResult,
    ) -> None:
        """
        Update a step with execution results.

        Args:
            step: The step to update
            step_result: HuskyStepResult containing execution details
        """
        step.status = StepStatus.COMPLETED
        step.data = {
            "input": step_result.input_data,
            "output": step_result.output_data,
        }
        step.metadata = step_result.metadata

    def run_episode(
        self,
        query_or_params: str | dict[str, Any],
        environment: "Environment",
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        memory: Optional[ContextMemory] = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Run the fixed single-pass workflow: generate → execute_tool → synthesize_answer.

        This method enforces the standard Husky workflow pattern for all
        concrete implementations.

        Args:
            query_or_params: The input query (str) for the sub-task to execute.
                           Husky agents expect a string query.
            environment: The environment providing tools and state management
            session_id: Optional session ID for tracking
            parent_session_id: Optional parent session ID for tracking
            **kwargs: Additional execution parameters

        Returns:
            ExecutionResult: The result of the episode execution
        """
        # Store the query for internal use (Husky agents expect string query)
        if not isinstance(query_or_params, str):
            raise ValueError(
                f"Husky agents expect string query, got {type(query_or_params)}"
            )
        query = query_or_params
        self.user_query = query

        # Set up memory if not provided
        if memory is None:
            if session_id is None:
                session_id = f"{self.name}___{int(time())}"
            memory = ContextMemory(
                agent_name=self.name,
                session_id=session_id,
                query=query,
                parent_session_id=parent_session_id,
            )

        # Track LLM token usage
        if "llm_token_usage" not in memory.metadata:
            # model_name -> input/output -> token count
            memory.metadata["llm_token_usage"] = {}

        # Register the memory to Environment
        environment.register_memory(memory)

        # Extract context once for all workflow steps
        context = self._prepare_context(memory, environment)

        # Initialize workflow variables
        final_result = None

        start_time = time()
        try:
            # Step 1: Generate initial response/plan
            gen_step = self._create_step(
                memory, "generate", {"input": context["query"]}
            )
            gen_step_result = self.generate(
                query=context["query"],
                parent_query=context["parent_query"],
                history=context["history"],
            )
            self._update_step_with_result(gen_step, gen_step_result)
            if "token_usage" in gen_step.metadata:
                update_token_usage(
                    memory.metadata["llm_token_usage"], gen_step.metadata["token_usage"]
                )

            # Step 2: Execute tool (always called, but may pass through unchanged)
            execute_step = self._create_step(
                memory, "execute", {"input": gen_step_result.result}
            )
            # self.execute_tool raises AgentException on failure
            tool_result: ExecutionResult = self.execute_tool(
                gen_step_result.result, memory, environment, **kwargs
            )
            assert (
                tool_result.success
            )  # This should not fail; exceptions are raised instead
            execute_step.status = StepStatus.COMPLETED
            execute_step.data["output"] = tool_result.result_data
            execute_step.metadata = tool_result.metadata

            # Step 3: Synthesize final answer
            synth_step = self._create_step(
                memory, "synthesis", {"input": tool_result.result_data}
            )
            synth_step_result = self.synthesize_answer(
                input_data=tool_result.result_data,
                query=context["query"],
                parent_query=context["parent_query"],
            )
            self._update_step_with_result(synth_step, synth_step_result)
            if "token_usage" in synth_step.metadata:
                update_token_usage(
                    memory.metadata["llm_token_usage"],
                    synth_step.metadata["token_usage"],
                )

            if not synth_step_result.success:
                raise AgentException(synth_step_result.result)

            final_result = synth_step_result.result
        except KeyboardInterrupt:
            logger.info("Run episode interrupted by user")
            raise
        except AgentException as e:
            # LLM-recoverable error
            error_msg = f"Error in {self.name}: {str(e)}"
            logger.info(error_msg)
            execution_time = time() - start_time
            memory.get_last_step().status = StepStatus.FAILED  # Set last step as failed
            return ExecutionResult(
                executable_name=self.name,
                executable_type=ExecutableType.AGENT,
                success=False,
                result_data=None,
                error_message=error_msg,
                execution_time=execution_time,
                token_usage=memory.metadata.get("llm_token_usage", {}),
                metadata={
                    "agent_type": "husky",
                    "session_id": session_id,
                },
            )
        except Exception:
            # Fail-fast: re-raise for programming errors
            logger.error(f"Unexpected error in {self.name}")
            raise

        # Create ExecutionResult with all execution metadata
        execution_time = time() - start_time
        execution_result = ExecutionResult(
            executable_name=self.name,
            executable_type=ExecutableType.AGENT,
            success=True,
            result_data=f"Found the answer: {final_result}",
            execution_time=execution_time,
            token_usage=memory.metadata.get("llm_token_usage", {}),
            metadata={
                "agent_type": "husky",
                "full_data": final_result,
                "session_id": session_id,
            },
        )

        return execution_result

    @abstractmethod
    def generate(
        self, query: str, parent_query: str = "", history: str = ""
    ) -> HuskyStepResult:
        """
        Generate initial response or plan using LLM.

        This method should use the LLM to generate an initial response,
        plan, or analysis based on the input query.

        Args:
            query: The input query to process
            parent_query: Optional parent query for multi-step problems
            history: Optional conversation history from parent session

        Returns:
            HuskyStepResult: Step result with result (LLM output) and token usage
        """
        pass

    def execute_tool(
        self,
        input_data: str,
        memory: ContextMemory,
        environment: "Environment",
        **kwargs,
    ) -> ExecutionResult:
        """
        Execute tool if needed (default: pass-through).

        Default implementation returns input unchanged. Concrete agents
        that use tools (HuskyCodeAgent, HuskySearchAgent) should override
        this method to perform actual tool execution.

        Add input and output to memory.

        Args:
            input_data: Input data from generate() method
            memory: The memory context for the current session
            environment: The environment providing tools
            **kwargs: Additional execution parameters

        Returns:
            str: Result from tool execution (default: unchanged input)
        """
        # Default implementation: pass-through without tool execution
        return ExecutionResult(
            executable_name="no_tool_execution",
            executable_type=ExecutableType.TOOL,
            success=True,
            result_data=input_data,
            error_message=None,
            metadata={},
        )

    def synthesize_answer(
        self,
        input_data: str,
        query: str,
        parent_query: str = "",
    ) -> HuskyStepResult:
        """
        Synthesize final answer from previous steps using LLM with structured output.

        This method combines results from generate() and execute_tool()
        to produce a final, concise answer using LLM with structured output.
        Uses the configured synthesis_prompt or falls back to DEFAULT_SYNTHESIS_PROMPT.

        Args:
            input_data: Result from execute_tool() method
            query: The input query to answer
            parent_query: Optional parent query for multi-step problems

        Returns:
            HuskyStepResult: Step result with result (answer), success flag, and token usage
        """
        # Format prompt with context
        format_vars = {"query": query, "response": input_data}
        if parent_query:
            format_vars["parent_query"] = parent_query

        llm_input = self._format_prompt(self.synthesis_prompt, **format_vars)

        # Call the LLM with structured output using JSON schema
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "husky_synthesis_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "A short summary of the reasoning process used to arrive at the answer. Examine if the provided information is sufficient to fully and directly answer the question.",
                        },
                        "answer_value": {
                            "type": "string",
                            "description": "The concise answer value without verbose expressions like 'The answer to <question> is...'",
                        },
                        "is_success": {
                            "type": "boolean",
                            "description": "True only if the found answer fully and directly addresses the question. Otherwise False.",
                        },
                        "failure_reason": {
                            "type": ["string", "null"],
                            "description": "Reason for failure if is_success=False (e.g., 'No supporting information found', 'Generation incomplete due to token limit', 'No sufficient information provided'). Return null if is_success=True",
                        },
                    },
                    "required": [
                        "reasoning",
                        "is_success",
                        "answer_value",
                        "failure_reason",
                    ],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

        response = self.get_llm_response(llm_input, response_format=response_format)
        synthesis_result = json.loads(response.content)
        token_usage = response.token_usage

        # Extract values from dict response
        reasoning = synthesis_result["reasoning"]
        is_success = synthesis_result["is_success"]
        answer_value = synthesis_result["answer_value"]
        failure_reason = synthesis_result.get("failure_reason")

        if is_success:
            failure_reason = None
        else:
            answer_value = f"Failed to find the answer to \"{query}\"\n"
            answer_value += (failure_reason or "Unknown reason") + "\n"
            answer_value += "Retry with a different question or try a different tool."

        # Prepare step info (token tracking happens in run_episode)
        step_output_data = {
            "reasoning": reasoning,
            "is_success": is_success,
            "answer_value": answer_value,
            "failure_reason": failure_reason,
        }
        step_result = HuskyStepResult(
            result=answer_value,
            success=is_success,
            input_data=llm_input,
            output_data=step_output_data,
            metadata={"token_usage": token_usage},
        )

        return step_result

    def get_schema(self, agent_id: Optional[str] = None) -> dict[str, Any]:
        """
        Get the parameter schema for this Husky Agent.

        Args:
            agent_id: Optional ID (used for schema name customization)

        Returns:
            dict: OpenAI function calling schema format
        """
        return {
            "type": "function",
            "function": {
                "name": agent_id if agent_id else self.name,
                "description": "Return a concise answer to a question",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "A single-hop question (complete sentence) to answer",
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        }


class HuskyCommonsenseAgent(HuskyAgent):
    """
    Husky-style agent for pure LLM reasoning using commonsense knowledge.

    This agent uses only LLM capabilities without external tools to answer
    questions through logical reasoning and commonsense knowledge.
    Follows the fixed workflow: generate() → execute_tool() → synthesize_answer().
    """

    def __init__(
        self,
        llm_call: Optional[Callable] = None,
        llm_options: dict[str, Any] = {},
        name: str = "HuskyCommonsenseAgent",
        generation_prompt: str = "",
        synthesis_prompt: str = "",
    ):
        """
        Initialize the Husky Commonsense Agent.

        Args:
            llm_call: Callable function to get LLM responses
            llm_options: Options to configure the LLM behavior
            name: Name identifier for the agent
            generation_prompt: Prompt template for generation step
            synthesis_prompt: Prompt template for synthesis step
        """
        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            max_steps=1,
            generation_prompt=generation_prompt,
            synthesis_prompt=synthesis_prompt,
        )

    def generate(
        self, query: str, parent_query: str = "", history: str = ""
    ) -> HuskyStepResult:
        """
        Generate response using LLM with commonsense reasoning.

        Uses the configured prompt template to guide the LLM in applying
        logical reasoning and commonsense knowledge to solve the given query.

        Args:
            query: The input query to process
            parent_query: Optional parent query for multi-step problems
            history: Optional conversation history from parent session

        Returns:
            HuskyStepResult: Step result with result (LLM output) and token usage
        """
        # Prepare format variables based on context
        format_vars = {"query": query, "history": history, "sub_question": query}

        # Handle parent query case
        if parent_query:
            format_vars["query"] = parent_query

        # Format prompt using helper
        llm_input = self._format_prompt(self.generation_prompt, **format_vars)

        # Call LLM
        response = self.get_llm_response(llm_input)
        output = response.content
        token_usage = response.token_usage

        # Create step info (token tracking happens in run_episode)
        step_result = HuskyStepResult(
            result=output,
            success=True,
            input_data=llm_input,
            output_data=output,
            metadata={"token_usage": token_usage},
        )

        return step_result

    def get_schema(self, agent_id: Optional[str] = None) -> dict[str, Any]:
        """
        Get the parameter schema for this Husky Commonsense Agent.

        Args:
            agent_id: Optional ID (used for schema name customization)

        Returns:
            dict: OpenAI function calling schema format
        """
        return {
            "type": "function",
            "function": {
                "name": agent_id if agent_id else self.name,
                "description": "Return a concise answer to a question using logical reasoning and commonsense knowledge. The question should be answerable in one reasoning step.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "A complete natural-language single-hop question (one inference). Not keywords; answerable with one reasoning step",
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        }


class HuskyMathAgent(HuskyAgent):
    """
    Husky-style agent for pure LLM reasoning using mathematical problem solving.

    This agent uses only LLM capabilities without external tools to answer
    questions through logical reasoning and basic math skills.
    Follows the fixed workflow: generate() → execute_tool() → synthesize_answer().
    """

    def __init__(
        self,
        llm_call: Optional[Callable] = None,
        llm_options: dict[str, Any] = {},
        name: str = "HuskyMathAgent",
        generation_prompt: str = "",
        synthesis_prompt: str = "",
    ):
        """
        Initialize the Husky Math Agent.

        Args:
            llm_call: Callable function to get LLM responses
            llm_options: Options to configure the LLM behavior
            name: Name identifier for the agent
            generation_prompt: Prompt template for generation step
            synthesis_prompt: Prompt template for synthesis step
        """
        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            max_steps=1,
            generation_prompt=generation_prompt,
            synthesis_prompt=synthesis_prompt,
        )

    def generate(
        self, query: str, parent_query: str = "", history: str = ""
    ) -> HuskyStepResult:
        """
        Generate response using LLM with mathematical reasoning.

        Note: HuskyMathAgent is not currently supported after refactoring.
        """
        raise NotImplementedError("HuskyMathAgent is not currently supported")

    def get_schema(self, agent_id: Optional[str] = None) -> dict[str, Any]:
        """
        Get the parameter schema for this Husky Math Agent.

        Args:
            agent_id: Optional ID (used for schema name customization)

        Returns:
            dict: OpenAI function calling schema format
        """
        return {
            "type": "function",
            "function": {
                "name": agent_id if agent_id else self.name,
                "description": "Return a concise answer to a question using logical reasoning and mathematical problem solving skills. Suitable for 1) solving math questions, writing or re-organizing equations, performing abstract reasoning such as case-by-case analysis, or identifying the conditions given in the question.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "A question (complete sentence) to answer using logical reasoning and mathematical problem solving skills",
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        }


class HuskyCodeAgent(HuskyAgent):
    """
    Husky-style agent for code generation and execution

    This agent uses LLM capabilities along with code execution tools to generate and run code snippets.
    Follows the fixed workflow: generate() → execute_tool() → synthesize_answer().
    """

    def __init__(
        self,
        llm_call: Optional[Callable] = None,
        llm_options: dict[str, Any] = {},
        name: str = "HuskyCodeAgent",
        generation_prompt: str = "",
        synthesis_prompt: str = "",
        code_header: str = "",
    ):
        """
        Initialize the Husky Code Agent.

        Args:
            llm_call: Callable function to get LLM responses
            llm_options: Options to configure the LLM behavior
            name: Name identifier for the agent
        """
        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            max_steps=1,
            generation_prompt=generation_prompt,
            synthesis_prompt=synthesis_prompt,
        )
        self.code_header = code_header

    def generate(
        self, query: str, parent_query: str = "", history: str = ""
    ) -> HuskyStepResult:
        """
        Generate response using LLM with code generation reasoning.

        Note: HuskyCodeAgent is not currently supported after refactoring.
        """
        raise NotImplementedError("HuskyCodeAgent is not currently supported")

    def execute_tool(
        self,
        input_data: str,
        memory: ContextMemory,
        environment: "Environment",
        **kwargs,
    ) -> ExecutionResult:
        """
        Execute the generated code using the Python executor tool.

        Overrides the default pass-through behavior to actually execute
        the generated Python code using the environment's tool registry.

        Args:
            input_data: Generated Python code from generate() method
            memory: The memory context for the current session
            environment: The environment providing tools
            **kwargs: Additional execution parameters

        Returns:
            ExecutionResult: Tool execution result

        Raises:
            AgentException: If tool execution fails
        """
        code = input_data

        code = self.code_header + "\n\n" + code if self.code_header else code

        # Execute the code using the python_executor tool
        execution_result = environment.execute_tool("python_executor", {"code": code})

        if not execution_result.success:
            error_msg = f"Tool execution failed: {execution_result.error_message}"
            raise AgentException(error_msg)

        output = str(execution_result.result_data)

        return ExecutionResult(
            executable_name=execution_result.executable_name,
            executable_type=ExecutableType.TOOL,
            success=True,
            result_data=output,
            error_message=None,
            metadata=execution_result.metadata,
        )

    def get_schema(self, agent_id: Optional[str] = None) -> dict[str, Any]:
        """
        Get the schema for this Husky Code Agent as an executable.

        Args:
            agent_id: Optional ID to use for the agent in the schema

        Returns:
            dict: OpenAI-style schema for this agent
        """
        return {
            "type": "function",
            "function": {
                "name": agent_id if agent_id else self.name,
                "description": "Return a concise answer to a question by generating and executing Python code snippets to obtain the answer. Suitable for 1) computing large numbers (at least 100), fractions or decimals. 2) counting or averaging long lists of numbers. 3) performing date-related operations, such as counting the number of days between two dates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "A question (complete sentence) to answer by generating and executing Python code snippets",
                        }
                    },
                    "required": ["question"],
                },
            },
        }


class HuskySearchAgent(HuskyAgent):
    """
    Husky-style agent for information retrieval.

    This agent uses LLM capabilities along with search tools to retrieve relevant information.
    Follows the fixed workflow: generate() → execute_tool() → synthesize_answer().
    """

    def __init__(
        self,
        llm_call: Optional[Callable] = None,
        llm_options: dict[str, Any] = {},
        name: str = "HuskySearchAgent",
        generation_prompt: str = "",
        synthesis_prompt: str = "",
    ):
        """
        Initialize the Husky Search Agent.

        Args:
            llm_call: Callable function to get LLM responses
            llm_options: Options to configure the LLM behavior
            name: Name identifier for the agent
        """
        super().__init__(
            llm_call=llm_call,
            llm_options=llm_options,
            name=name,
            max_steps=1,
            generation_prompt=generation_prompt,
            synthesis_prompt=synthesis_prompt,
        )

    def generate(
        self, query: str, parent_query: str = "", history: str = ""
    ) -> HuskyStepResult:
        """
        Generate response using LLM with information retrieval.

        Uses the configured prompt template to guide the LLM in generating
        a search query for information retrieval.

        Args:
            query: The input query to process
            parent_query: Optional parent query for multi-step problems
            history: Optional conversation history from parent session

        Returns:
            HuskyStepResult: Step result with result (search query) and token usage
        """

        # Prepare format variables based on context
        format_vars = {"query": query, "history": history, "sub_question": query}

        # Handle parent query case
        if parent_query:
            format_vars["query"] = parent_query

        # Format prompt using helper
        llm_input = self._format_prompt(self.generation_prompt, **format_vars)

        # Call LLM
        try:
            response = self.get_llm_response(llm_input)
            output = response.content
            token_usage = response.token_usage
        except KeyboardInterrupt:
            logger.info("Generation interrupted by user")
            raise
        except Exception as e:
            raise AgentException(
                f"LLM generation failed: {str(e)}", agent_name=self.name
            )

        # Create step info (token tracking happens in run_episode)
        step_result = HuskyStepResult(
            result=output,
            success=True,
            input_data=llm_input,
            output_data=output,
            metadata={"token_usage": token_usage},
        )

        return step_result

    def execute_tool(
        self,
        input_data: str,
        memory: ContextMemory,
        environment: "Environment",
        **kwargs,
    ) -> ExecutionResult:
        """
        Execute the search tool with the generated search query.

        Overrides the default pass-through behavior to actually execute
        the generated search query using the environment's tool registry.

        Args:
            input_data: Generated search query from generate() method
            memory: The memory context for the current session
            environment: The environment providing tools
            **kwargs: Additional execution parameters

        Returns:
            ExecutionResult: Tool execution result

        Raises:
            AgentException: If tool execution fails
            ValueError: If the execution result format is invalid
        """
        # Retrieve information using the search tool
        query = input_data
        execution_result = environment.execute_tool("search", params={"query": query})

        if not execution_result.success:
            error_msg = f"Tool execution failed: {execution_result.error_message}"
            raise AgentException(error_msg)

        if not isinstance(execution_result.result_data, dict):
            raise ValueError("Expected result_data to be a dictionary")
        if "results" not in execution_result.result_data:
            raise ValueError(
                f"Expected 'results' field in result_data: {execution_result.result_data.keys()=}"
            )
        if not isinstance(execution_result.result_data["results"], str):
            raise ValueError(
                f"Expected 'results' field in result_data to be a string: {type(execution_result.result_data['results'])}"
            )
        output = execution_result.result_data["results"]

        return ExecutionResult(
            executable_name=execution_result.executable_name,
            executable_type=ExecutableType.TOOL,
            success=True,
            result_data=output,
            error_message=None,
            metadata=execution_result.metadata,
        )

    def get_schema(self, agent_id: Optional[str] = None) -> dict[str, Any]:
        """
        Get the parameter schema for this Husky Search Agent.

        Args:
            agent_id: Optional ID (used for schema name customization)

        Returns:
            dict: OpenAI function calling schema format
        """
        return {
            "type": "function",
            "function": {
                "name": agent_id if agent_id else self.name,
                "description": "Return a concise answer to a single-hop factual question about history, sports, culture, geography, medicine, science, and more using search over retrieved information. The question should be answerable with one retrieval step.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "A complete natural-language question (single sentence) (e.g., 'Who is the spouse of XX?')",
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        }
