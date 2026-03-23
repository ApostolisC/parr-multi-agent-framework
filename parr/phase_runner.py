"""
Phase Runner for the Agentic Framework.

The single-phase agentic loop. This is the core execution engine.
Each phase (Plan/Act/Review/Report) runs as a self-contained loop:

    system prompt + tools → LLM → tool calls → execute → repeat
    until: no tool calls (phase complete) OR max iterations reached

The phase runner does NOT manage cross-phase transitions — that's the
agent runtime's job. The phase runner handles ONE phase at a time.

Orchestrator-level tools (spawn_agent, wait_for_agents, get_agent_result)
are yielded back to the caller for handling, not executed here.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .budget_tracker import BudgetExceededException, BudgetTracker
from .context_manager import ContextManager
from .event_bus import EventBus
from .event_types import (
    FrameworkEvent,
    agent_token,
    budget_warning,
    context_compacted,
    llm_call_completed,
    phase_completed,
    phase_iteration_limit,
    phase_started,
    tool_executed,
)
from .stall_detector import StallDetector
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry
from .core_types import (
    AgentConfig,
    AgentInput,
    AgentNode,
    AgentStatus,
    LLMResponse,
    Message,
    MessageRole,
    Phase,
    StallDetectionConfig,
    ToolCall,
    ToolResult,
    TokenUsage,
    WorkflowExecution,
)

logger = logging.getLogger(__name__)


class CancelledException(Exception):
    """Raised when an agent is cancelled mid-phase."""
    pass


# Default iteration limits per phase.
# Each LLM call (with or without tool calls) counts as one iteration.
DEFAULT_PHASE_LIMITS: Dict[Phase, int] = {
    Phase.PLAN: 8,
    Phase.ACT: 25,
    Phase.REVIEW: 5,
    Phase.REPORT: 8,
}

# Stall detection defaults are defined in StallDetectionConfig (core_types.py).
# The PhaseRunner accepts an optional StallDetectionConfig at construction time,
# or a custom StallDetector instance for fully pluggable stall detection.


@dataclass
class PhaseResult:
    """Output from a single phase run."""
    phase: Phase
    content: Optional[str] = None
    iterations: int = 0
    hit_iteration_limit: bool = False
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls_made: List[Dict[str, Any]] = field(default_factory=list)
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class OrchestratorToolRequest:
    """
    A tool call that the orchestrator needs to handle.

    Yielded back to the caller (agent runtime / orchestrator) when
    the LLM requests an orchestrator-level tool like spawn_agent.
    """
    tool_call: ToolCall
    # The caller sets this after handling the tool call
    result: Optional[ToolResult] = None


class PhaseRunner:
    """
    Runs a single phase of the agent lifecycle.

    The phase runner iterates the LLM ↔ tool loop until:
    - The LLM responds without tool calls (natural completion)
    - Max iterations reached (forced completion with best-effort output)
    - Budget exceeded (BudgetExceededException)
    - Cancellation (CancelledException)
    """

    def __init__(
        self,
        llm: Any,  # ToolCallingLLM protocol
        tool_executor: ToolExecutor,
        tool_registry: ToolRegistry,
        budget_tracker: BudgetTracker,
        context_manager: ContextManager,
        event_bus: EventBus,
        phase_limits: Optional[Dict[Phase, int]] = None,
        stream: bool = False,
        stall_config: Optional[StallDetectionConfig] = None,
        stall_detector: Optional[StallDetector] = None,
        on_tool_persisted: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_llm_call_persisted: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._llm = llm
        self._tool_executor = tool_executor
        self._tool_registry = tool_registry
        self._budget_tracker = budget_tracker
        self._context_manager = context_manager
        self._event_bus = event_bus
        self._phase_limits = phase_limits or DEFAULT_PHASE_LIMITS
        self._stream = stream
        self._stall = stall_config or StallDetectionConfig()
        # Pluggable stall detector. If not provided, a default one is created.
        # If stall_config is provided but stall_detector is not, the config
        # is passed to the default detector.
        self._stall_detector = stall_detector or StallDetector(
            registry=tool_registry,
            config=self._stall,
        )
        # Tracks the in-progress phase result so it can be recovered
        # when BudgetExceededException interrupts a phase mid-execution.
        self._in_progress_result: Optional[PhaseResult] = None
        # Optional callback invoked after each tool call is tracked.
        # Used for incremental persistence so the debug UI can show
        # live updates during phase execution (not just at phase end).
        self._on_tool_persisted = on_tool_persisted
        # Optional callback invoked after each LLM call for persistence.
        self._on_llm_call_persisted = on_llm_call_persisted

    async def run_continuation(
        self,
        phase: Phase,
        node: AgentNode,
        workflow: WorkflowExecution,
        config: AgentConfig,
        input: AgentInput,
        initial_messages: List[Message],
        initial_tool_calls: Optional[List[Dict[str, Any]]] = None,
        initial_iteration: int = 1,
        on_orchestrator_tool: Optional[Callable] = None,
    ) -> PhaseResult:
        """Continue a phase from an existing message history.

        Used by the adaptive flow: the entry call produces messages and
        tool calls that become iteration 0 of the detected phase.  This
        method picks up from there.

        Args:
            phase: The detected phase.
            node: The agent's node in the tree.
            workflow: The parent workflow execution.
            config: Agent configuration.
            input: Agent input.
            initial_messages: Complete message history from the entry call.
            initial_tool_calls: Tool calls already made in the entry call.
            initial_iteration: Iteration counter starting point (default 1).
            on_orchestrator_tool: Async callback for orchestrator-level tools.

        Returns:
            PhaseResult with the phase's output.
        """
        return await self.run(
            phase=phase,
            node=node,
            workflow=workflow,
            config=config,
            input=input,
            on_orchestrator_tool=on_orchestrator_tool,
            _continuation_messages=initial_messages,
            _continuation_tool_calls=initial_tool_calls,
            _continuation_iteration=initial_iteration,
        )

    async def run(
        self,
        phase: Phase,
        node: AgentNode,
        workflow: WorkflowExecution,
        config: AgentConfig,
        input: AgentInput,
        working_memory_snapshot: Optional[str] = None,
        extra_context: Optional[str] = None,
        on_orchestrator_tool: Optional[Callable] = None,
        # Private continuation params (used by run_continuation)
        _continuation_messages: Optional[List[Message]] = None,
        _continuation_tool_calls: Optional[List[Dict[str, Any]]] = None,
        _continuation_iteration: int = 0,
    ) -> PhaseResult:
        """
        Execute a single phase.

        Args:
            phase: Which phase to run.
            node: The agent's node in the tree.
            workflow: The parent workflow execution.
            config: Agent configuration.
            input: Agent input.
            working_memory_snapshot: Current state of todo list, findings, etc.
            extra_context: Additional context (e.g., review feedback).
            on_orchestrator_tool: Async callback for orchestrator-level tools.
                Signature: async (ToolCall) -> ToolResult

        Returns:
            PhaseResult with the phase's output.
        """
        is_continuation = _continuation_messages is not None

        node.current_phase = phase
        max_iterations = self._phase_limits.get(phase, 10)

        # Set tool executor to current phase for filtering
        self._tool_executor.set_phase(phase)

        if is_continuation:
            # Continuation: use entry call tools (same set the agent already saw)
            entry_tools = self._tool_registry.get_for_entry()
            tool_schemas_by_name = {
                t.name: t.to_llm_schema() for t in entry_tools
            }
            messages = list(_continuation_messages)
        else:
            # Normal: get phase-specific tools
            phase_tools = self._tool_registry.get_for_phase(phase)
            tool_schemas_by_name = {
                t.name: t.to_llm_schema() for t in phase_tools
            }
            # Get tools visible (description-only) in this phase
            visible_tools = self._tool_registry.get_visible_for_phase(phase)
            # Build initial messages
            messages = self._context_manager.build_phase_messages(
                phase=phase,
                config=config,
                input=input,
                working_memory_snapshot=working_memory_snapshot,
                extra_context=extra_context,
                visible_tools=visible_tools,
            )

        tool_consecutive_failures: Dict[str, int] = {}

        # Emit phase started
        await self._event_bus.publish(phase_started(
            workflow_id=workflow.workflow_id,
            task_id=node.task_id,
            agent_id=node.agent_id,
            phase=phase.value,
        ))

        result = PhaseResult(phase=phase)
        if _continuation_tool_calls:
            result.tool_calls_made = list(_continuation_tool_calls)
        self._in_progress_result = result
        iteration = _continuation_iteration
        _cumulative_tokens = 0
        _budget_warning_injected = False
        _iteration_advisory_injected = False
        _iteration_warning_injected = False
        _progress_injected = False
        _mandatory_nudge_given = False
        _circuit_breaker_warnings: Set[str] = set()
        # Reset stall detector state for this phase
        self._stall_detector.reset()
        # Iteration limit awareness thresholds.
        # Advisory fires first (a few iterations before the end),
        # warning fires on the penultimate iteration.
        # For very small limits (≤3), thresholds are clamped so that
        # advisory fires at iteration 1 and warning fires at most at
        # max_iterations-1, ensuring they never overlap.
        _advisory_threshold = max(1, max_iterations - 3)
        _warn_threshold = max(_advisory_threshold + 1, max_iterations - 1)
        _mid_threshold = max(2, max_iterations // 2)
        _domain_tool_calls = 0
        _framework_tool_calls = 0

        while iteration < max_iterations:
            # Budget check before every LLM call
            self._budget_tracker.check_budget(node, workflow)

            # Check cancellation
            if node.status == AgentStatus.CANCELLED:
                raise CancelledException(f"Agent {node.agent_id} was cancelled")

            # Two-stage iteration limit awareness:
            # 1. Advisory at max-3: let the LLM know it's running low
            # 2. Firmer guidance at max-1: strongly recommend wrapping up
            # The LLM retains the choice to continue if it judges the
            # remaining work is critical.
            if not _iteration_advisory_injected and iteration >= _advisory_threshold:
                _iteration_advisory_injected = True
                remaining = max_iterations - iteration
                messages.append(Message(
                    role=MessageRole.USER,
                    content=(
                        f"[ITERATION ADVISORY] You have used {iteration} of "
                        f"{max_iterations} iterations for this phase ({remaining} "
                        f"remaining). Consider wrapping up: prioritize the most "
                        f"important remaining work, and prepare to produce your "
                        f"final text response soon."
                    ),
                ))
                logger.info(
                    f"Iteration advisory injected at iteration "
                    f"{iteration}/{max_iterations} in {phase.value} "
                    f"for agent {node.agent_id}"
                )
            if not _iteration_warning_injected and iteration >= _warn_threshold:
                _iteration_warning_injected = True
                messages.append(Message(
                    role=MessageRole.USER,
                    content=(
                        f"[ITERATION WARNING] This is your last iteration "
                        f"({iteration}/{max_iterations}). You should produce "
                        f"your final text response now to complete this phase. "
                        f"If you have critical unfinished work, you may make "
                        f"one more tool call, but be aware the phase will end "
                        f"after this iteration."
                    ),
                ))
                logger.info(
                    f"Iteration warning injected at iteration "
                    f"{iteration}/{max_iterations} in {phase.value} "
                    f"for agent {node.agent_id}"
                )

            # Build streaming callback if enabled
            on_token_cb = None
            if self._stream:
                async def _on_token(token: str) -> None:
                    await self._event_bus.publish(agent_token(
                        workflow_id=workflow.workflow_id,
                        task_id=node.task_id,
                        agent_id=node.agent_id,
                        phase=phase.value,
                        token=token,
                    ))
                on_token_cb = _on_token

            # Circuit breaker: filter out tools that have failed too many
            # consecutive times in this phase invocation.
            active_tool_schemas = [
                schema for name, schema in tool_schemas_by_name.items()
                if tool_consecutive_failures.get(name, 0) < self._stall.max_consecutive_tool_failures
            ]

            # Call LLM (with error capture for terminal failures)
            try:
                response = await self._llm.chat_with_tools(
                    messages=messages,
                    tools=active_tool_schemas,
                    model=config.model,
                    model_config=config.model_config,
                    stream=self._stream,
                    on_token=on_token_cb,
                )
            except (RuntimeError, Exception) as llm_error:
                # LLM call failed after all retries — record the error
                _llm_error_record = {
                    "phase": phase.value,
                    "iteration": iteration,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "response_content": None,
                    "tool_calls": None,
                    "error": str(llm_error),
                    "cumulative_tokens": _cumulative_tokens,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                result.llm_calls.append(_llm_error_record)
                if self._on_llm_call_persisted:
                    try:
                        self._on_llm_call_persisted(_llm_error_record)
                    except Exception:
                        pass
                raise

            # Track usage — record_usage returns the calculated cost and
            # also sets it on the usage object, making the flow explicit.
            usage = response.usage or TokenUsage()
            cost = self._budget_tracker.record_usage(node, workflow, usage, config.model)
            result.total_usage.input_tokens += usage.input_tokens
            result.total_usage.output_tokens += usage.output_tokens
            result.total_usage.total_cost += cost

            # Budget warning: nudge the LLM when approaching limits
            if not _budget_warning_injected:
                _warning_msg = self._budget_tracker.check_warning_threshold(node)
                if _warning_msg:
                    _budget_warning_injected = True
                    messages.append(Message(
                        role=MessageRole.USER,
                        content=(
                            f"[BUDGET WARNING] Approaching budget limits: "
                            f"{_warning_msg}. Wrap up your current work "
                            f"efficiently and move toward completing this phase."
                        ),
                    ))
                    await self._event_bus.publish(budget_warning(
                        workflow_id=workflow.workflow_id,
                        task_id=node.task_id,
                        agent_id=node.agent_id,
                        consumed_tokens=node.budget_consumed.tokens,
                        max_tokens=node.budget.max_tokens,
                        consumed_cost=node.budget_consumed.cost,
                        max_cost=node.budget.max_cost,
                    ))

            # Emit LLM call event (enriched with response content + tool calls)
            _cumulative_tokens += usage.input_tokens + usage.output_tokens
            _event_tool_calls = None
            if response.tool_calls:
                _event_tool_calls = [
                    {"name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ]
            await self._event_bus.publish(llm_call_completed(
                workflow_id=workflow.workflow_id,
                task_id=node.task_id,
                agent_id=node.agent_id,
                phase=phase.value,
                iteration=iteration,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                response_content=response.content,
                tool_calls=_event_tool_calls,
                cumulative_tokens=_cumulative_tokens,
            ))

            # Build and persist LLM call record
            _llm_call_record = {
                "phase": phase.value,
                "iteration": iteration,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "response_content": response.content,
                "tool_calls": _event_tool_calls,
                "error": None,
                "cumulative_tokens": _cumulative_tokens,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            result.llm_calls.append(_llm_call_record)
            if self._on_llm_call_persisted:
                try:
                    self._on_llm_call_persisted(_llm_call_record)
                except Exception:
                    logger.debug(
                        "LLM call persistence callback failed",
                        exc_info=True,
                    )

            # Process response
            if response.has_tool_calls():
                # Append assistant message with tool calls
                assistant_msg = response.raw_message or Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
                messages.append(assistant_msg)

                # Execute each tool call and check for non-framework work
                has_non_framework_calls = False
                # Collect circuit-breaker messages to append AFTER all tool
                # results — inserting USER messages between TOOL messages
                # violates the OpenAI API invariant.
                _deferred_cb_messages: List[Message] = []
                for tool_call in response.tool_calls:
                    # Check cancellation between individual tool calls
                    if node.status == AgentStatus.CANCELLED:
                        raise CancelledException(f"Agent {node.agent_id} was cancelled")

                    tool_result = await self._handle_tool_call(
                        tool_call=tool_call,
                        node=node,
                        workflow=workflow,
                        phase=phase,
                        on_orchestrator_tool=on_orchestrator_tool,
                    )

                    # Append tool result message
                    messages.append(Message(
                        role=MessageRole.TOOL,
                        content=tool_result.content,
                        tool_call_id=tool_result.tool_call_id,
                    ))

                    # Track tool call
                    tool_record = {
                        "name": tool_call.name,
                        "phase": phase.value,
                        "arguments": tool_call.arguments,
                        "result_content": tool_result.content,
                        "success": tool_result.success,
                        "error": tool_result.error,
                    }
                    result.tool_calls_made.append(tool_record)

                    # Incremental persistence: write to disk immediately so
                    # the debug UI can render live updates during the phase.
                    if self._on_tool_persisted:
                        try:
                            self._on_tool_persisted(tool_record)
                        except Exception:
                            logger.debug(
                                "Incremental persistence callback failed",
                                exc_info=True,
                            )

                    # Circuit breaker: track consecutive failures per tool.
                    # Reset on success; warn then suppress after threshold.
                    if tool_result.success:
                        tool_consecutive_failures.pop(tool_call.name, None)
                        _circuit_breaker_warnings.discard(tool_call.name)
                    else:
                        count = tool_consecutive_failures.get(tool_call.name, 0) + 1
                        tool_consecutive_failures[tool_call.name] = count
                        # At threshold-1: warn the LLM (advisory, not removal)
                        if (count == self._stall.max_consecutive_tool_failures - 1
                                and tool_call.name not in _circuit_breaker_warnings):
                            _circuit_breaker_warnings.add(tool_call.name)
                            _deferred_cb_messages.append(Message(
                                role=MessageRole.USER,
                                content=(
                                    f"[TOOL WARNING] Tool '{tool_call.name}' has "
                                    f"failed {count} consecutive times. It will "
                                    f"be temporarily disabled if it fails once "
                                    f"more. Consider an alternative approach or "
                                    f"different parameters."
                                ),
                            ))
                        if count == self._stall.max_consecutive_tool_failures:
                            logger.warning(
                                f"Circuit breaker: suppressing '{tool_call.name}' "
                                f"after {count} consecutive failures in "
                                f"{phase.value} for agent {node.agent_id}"
                            )
                            _deferred_cb_messages.append(Message(
                                role=MessageRole.USER,
                                content=(
                                    f"[TOOL DISABLED] Tool '{tool_call.name}' has "
                                    f"been temporarily disabled after "
                                    f"{count} consecutive failures. Use alternative "
                                    f"tools or approaches to continue your work."
                                ),
                            ))

                    # Check if this is a non-framework tool call
                    tool_def = self._tool_registry.get(tool_call.name)
                    if not tool_def or not tool_def.is_framework_tool:
                        has_non_framework_calls = True
                        _domain_tool_calls += 1
                    else:
                        _framework_tool_calls += 1

                # Append deferred circuit-breaker messages after all tool
                # results, so we don't break the assistant→tool pairing.
                messages.extend(_deferred_cb_messages)

                # Always count every LLM-call iteration toward the phase limit.
                # Phase limits are hard caps that must be respected regardless
                # of whether the tools called are framework or domain tools.
                iteration += 1

                # Periodic iteration awareness: inject once at ~50% of
                # the phase limit so the LLM knows its resource usage.
                if (not _progress_injected
                        and not _iteration_warning_injected
                        and iteration == _mid_threshold):
                    _progress_injected = True
                    messages.append(Message(
                        role=MessageRole.USER,
                        content=(
                            f"[PROGRESS] Iteration {iteration}/{max_iterations}. "
                            f"Tool calls so far: {_domain_tool_calls} domain, "
                            f"{_framework_tool_calls} framework. "
                            f"Use remaining iterations wisely — "
                            f"prefer domain tools for actual work."
                        ),
                    ))
                    logger.debug(
                        "Iteration awareness injected at %d/%d in %s "
                        "for agent %s",
                        iteration, max_iterations, phase.value,
                        node.agent_id,
                    )

                # ── Stall detection (delegated to StallDetector) ──────
                verdict = self._stall_detector.check_iteration(response.tool_calls)

                if verdict.should_warn and verdict.warning_message:
                    messages.append(Message(
                        role=MessageRole.USER,
                        content=verdict.warning_message,
                    ))
                    # Annotate the LLM call record so the UI can show
                    # stall warnings on per-iteration blocks.
                    if result.llm_calls:
                        result.llm_calls[-1]["stall_warning"] = verdict.reason

                if verdict.is_stalled:
                    logger.warning(
                        f"Stall detected ({verdict.reason}): "
                        f"forcing phase completion in "
                        f"{phase.value} for agent {node.agent_id}."
                    )
                    result.iterations = iteration
                    result.hit_iteration_limit = True
                    result.content = self._extract_best_effort(messages)

                    await self._event_bus.publish(phase_iteration_limit(
                        workflow_id=workflow.workflow_id,
                        task_id=node.task_id,
                        agent_id=node.agent_id,
                        phase=phase.value,
                        limit=max_iterations,
                    ))

                    return result

                # Check for context truncation
                messages = self._context_manager.truncate_if_needed(messages)

            else:
                # No tool calls = LLM wants to complete the phase.
                # But first, check if any mandatory tools were never called.
                mandatory_tools = self._tool_registry.get_mandatory_for_phase(phase)
                called_tool_names = {tc["name"] for tc in result.tool_calls_made}
                uncalled_mandatory = [
                    t.name for t in mandatory_tools
                    if t.name not in called_tool_names
                ]

                if uncalled_mandatory and iteration < max_iterations - 1:
                    if _mandatory_nudge_given:
                        # Already nudged once — LLM ignored it. Log warning
                        # and let the phase complete rather than looping.
                        logger.warning(
                            f"Mandatory tools {uncalled_mandatory} still uncalled "
                            f"after nudge in {phase.value} for agent "
                            f"{node.agent_id}. Completing phase anyway."
                        )
                    else:
                        # First nudge — give the LLM one more chance
                        _mandatory_nudge_given = True
                        messages.append(Message(
                            role=MessageRole.ASSISTANT,
                            content=response.content,
                        ))
                        messages.append(Message(
                            role=MessageRole.USER,
                            content=(
                                f"You have not called the following mandatory tools: "
                                f"{', '.join(uncalled_mandatory)}. You MUST call them "
                                f"before completing this phase."
                            ),
                        ))
                        iteration += 1
                        continue

                # Phase complete
                result.content = response.content
                result.iterations = iteration
                return result

        # Hit iteration limit
        result.iterations = iteration
        result.hit_iteration_limit = True
        result.content = self._extract_best_effort(messages)

        await self._event_bus.publish(phase_iteration_limit(
            workflow_id=workflow.workflow_id,
            task_id=node.task_id,
            agent_id=node.agent_id,
            phase=phase.value,
            limit=max_iterations,
        ))

        logger.warning(
            f"Phase {phase.value} hit iteration limit ({max_iterations}) "
            f"for agent {node.agent_id}"
        )

        return result

    async def _handle_tool_call(
        self,
        tool_call: ToolCall,
        node: AgentNode,
        workflow: WorkflowExecution,
        phase: Phase,
        on_orchestrator_tool: Optional[Callable],
    ) -> ToolResult:
        """Handle a single tool call — either locally or via orchestrator."""

        # Check if this is an orchestrator-level tool
        tool_def = self._tool_registry.get(tool_call.name)
        if tool_def and tool_def.is_orchestrator_tool:
            if on_orchestrator_tool:
                tool_result = await on_orchestrator_tool(tool_call)
            else:
                tool_result = ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    content="",
                    error=f"Orchestrator tool '{tool_call.name}' is not available "
                           f"(no orchestrator handler configured).",
                )
        else:
            # Execute via tool executor
            tool_result = await self._tool_executor.execute(tool_call)

        # Emit tool executed event (enriched with arguments + result content)
        await self._event_bus.publish(tool_executed(
            workflow_id=workflow.workflow_id,
            task_id=node.task_id,
            agent_id=node.agent_id,
            phase=phase.value,
            tool_name=tool_call.name,
            success=tool_result.success,
            error=tool_result.error,
            arguments=tool_call.arguments,
            result_content=tool_result.content if tool_result.content else None,
        ))

        return tool_result

    _TRIVIAL_TOOL_RESPONSES = frozenset({
        "No todo items.",
        "No findings recorded.",
        "No review checklist recorded.",
    })

    def _extract_best_effort(self, messages: List[Message]) -> str:
        """Extract the best content from messages when iteration limit is hit.

        Priority:
        1. Last assistant message with non-empty text content.
        2. Synthesized summary from recent tool results (when the LLM
           never produced text-only responses, tool results contain
           the actual work output).
        3. Generic fallback message.
        """
        # Priority 1: last assistant message with actual text
        for msg in reversed(messages):
            if msg.role == MessageRole.ASSISTANT and msg.content and msg.content.strip():
                return msg.content

        # Priority 2: synthesize from tool result messages
        tool_results: List[str] = []
        for msg in reversed(messages):
            if msg.role == MessageRole.TOOL and msg.content and msg.content.strip():
                content = msg.content.strip()
                if content in self._TRIVIAL_TOOL_RESPONSES:
                    continue
                if len(content) > 20:
                    tool_results.append(content)
                if len(tool_results) >= 5:
                    break

        if tool_results:
            tool_results.reverse()  # restore chronological order
            combined = "\n\n---\n\n".join(tool_results)
            if len(combined) > 3000:
                combined = combined[:3000] + "\n\n[... truncated]"
            return (
                "[Best-effort output synthesized from tool results — "
                "the agent did not produce a text summary before the "
                "iteration limit.]\n\n" + combined
            )

        # Priority 3: generic fallback
        return "Phase completed without producing text output (iteration limit reached)."
