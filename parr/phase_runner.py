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
# The PhaseRunner accepts an optional StallDetectionConfig at construction time.

# Framework tools that are considered read-only (no state change).
# Consecutive iterations containing ONLY these tools (and no others)
# are "stalled" iterations.
_READ_ONLY_FRAMEWORK_TOOLS = frozenset({
    "get_todo_list",
    "get_findings",
})

# Framework tools that produce actual work output (not just bookkeeping).
# When the LLM calls these, it IS doing real work — logging analysis results
# or marking progress.  These reset the framework-only stall counter so that
# tasks without domain tools (pure analysis / sub-agents with no external
# data sources) aren't force-stopped while making genuine progress.
_PROGRESS_FRAMEWORK_TOOLS = frozenset({
    "log_finding",
    "batch_log_findings",
    "mark_todo_complete",
    "batch_mark_todo_complete",
    "review_checklist",
    "submit_report",
})



def _hash_tool_call(name: str, arguments: dict) -> str:
    """Produce a stable hash for a (tool_name, arguments) pair."""
    key = name + ":" + json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.md5(key.encode()).hexdigest()


@dataclass
class PhaseResult:
    """Output from a single phase run."""
    phase: Phase
    content: Optional[str] = None
    iterations: int = 0
    hit_iteration_limit: bool = False
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls_made: List[Dict[str, Any]] = field(default_factory=list)


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
        node.current_phase = phase
        max_iterations = self._phase_limits.get(phase, 10)

        # Set tool executor to current phase for filtering
        self._tool_executor.set_phase(phase)

        # Get tools available in this phase (with circuit-breaker support)
        phase_tools = self._tool_registry.get_for_phase(phase)
        tool_schemas_by_name: Dict[str, dict] = {
            t.name: t.to_llm_schema() for t in phase_tools
        }
        tool_consecutive_failures: Dict[str, int] = {}

        # Build initial messages
        messages = self._context_manager.build_phase_messages(
            phase=phase,
            config=config,
            input=input,
            working_memory_snapshot=working_memory_snapshot,
            extra_context=extra_context,
        )

        # Emit phase started
        await self._event_bus.publish(phase_started(
            workflow_id=workflow.workflow_id,
            task_id=node.task_id,
            agent_id=node.agent_id,
            phase=phase.value,
        ))

        result = PhaseResult(phase=phase)
        iteration = 0
        _consecutive_stall_iterations = 0
        _consecutive_fw_only_iterations = 0
        _cumulative_tokens = 0
        # Duplicate tool call detection state
        _recent_iteration_signatures: List[Set[str]] = []
        _consecutive_duplicate_iterations = 0
        _budget_warning_injected = False
        _iteration_advisory_injected = False
        _iteration_warning_injected = False
        _progress_injected = False
        _mandatory_nudge_given = False
        _circuit_breaker_warnings: Set[str] = set()
        _duplicate_warning_injected = False
        _stall_warning_injected = False
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

            # Call LLM
            response = await self._llm.chat_with_tools(
                messages=messages,
                tools=active_tool_schemas,
                model=config.model,
                model_config=config.model_config,
                stream=self._stream,
                on_token=on_token_cb,
            )

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
                    result.tool_calls_made.append({
                        "name": tool_call.name,
                        "success": tool_result.success,
                        "error": tool_result.error,
                    })

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
                            messages.append(Message(
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
                            messages.append(Message(
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

                # ── Duplicate tool call detection ──────────────────────
                # Track (tool_name, args_hash) signatures per iteration.
                # If ALL calls in the last N iterations were already seen
                # in the recent window, the agent is stuck in a loop.
                _iter_sigs: Set[str] = set()
                for tc in response.tool_calls:
                    _iter_sigs.add(_hash_tool_call(tc.name, tc.arguments))

                _all_seen_before = False
                if _recent_iteration_signatures:
                    _historical_sigs: Set[str] = set()
                    for prev_sigs in _recent_iteration_signatures[-self._stall.duplicate_call_window:]:
                        _historical_sigs.update(prev_sigs)
                    _all_seen_before = _iter_sigs.issubset(_historical_sigs)

                _recent_iteration_signatures.append(_iter_sigs)

                if _all_seen_before:
                    _consecutive_duplicate_iterations += 1
                else:
                    _consecutive_duplicate_iterations = 0

                # Advisory at 2 consecutive duplicate iterations
                if (_consecutive_duplicate_iterations == 2
                        and not _duplicate_warning_injected):
                    _duplicate_warning_injected = True
                    messages.append(Message(
                        role=MessageRole.USER,
                        content=(
                            "[DUPLICATE CALL WARNING] You have been making the "
                            "same tool calls repeatedly for 2 iterations. This "
                            "pattern suggests a loop. Consider: (1) using "
                            "different parameters, (2) trying a different tool, "
                            "or (3) producing your final response if you have "
                            "enough data. The phase will be force-completed if "
                            "this pattern continues."
                        ),
                    ))

                if _consecutive_duplicate_iterations >= self._stall.max_duplicate_call_iterations:
                    logger.warning(
                        f"Duplicate tool call loop detected: "
                        f"{_consecutive_duplicate_iterations} consecutive "
                        f"iterations with only previously-seen (tool, args) "
                        f"pairs in {phase.value} for agent {node.agent_id}. "
                        f"Forcing phase completion."
                    )
                    result.iterations = iteration
                    result.hit_iteration_limit = True
                    result.content = self._extract_best_effort(messages)

                    await self._event_bus.publish(phase_iteration_limit(
                        workflow_id=workflow.workflow_id,
                        task_id=node.task_id,
                        agent_id=node.agent_id,
                        phase=phase.value,
                        limit=self._stall.max_duplicate_call_iterations,
                    ))

                    return result

                if has_non_framework_calls:
                    _consecutive_stall_iterations = 0
                    _consecutive_fw_only_iterations = 0
                else:
                    tools_this_iteration = {tc.name for tc in response.tool_calls}

                    # Progress-producing framework tools (log_finding,
                    # mark_todo_complete, review_checklist, submit_report)
                    # represent genuine work — they record analysis output or
                    # advance execution state.  When present, reset the
                    # framework-only counter so tasks without domain tools
                    # aren't force-stopped while making real progress.
                    has_progress_tools = bool(
                        tools_this_iteration & _PROGRESS_FRAMEWORK_TOOLS
                    )
                    if has_progress_tools:
                        _consecutive_fw_only_iterations = 0
                    else:
                        _consecutive_fw_only_iterations += 1

                    # Check if this framework-only iteration is purely
                    # read-only (no state change at all).
                    made_progress = bool(
                        tools_this_iteration - _READ_ONLY_FRAMEWORK_TOOLS
                    )

                    if made_progress:
                        _consecutive_stall_iterations = 0
                    else:
                        _consecutive_stall_iterations += 1

                    # Two stall limits:
                    # 1. Read-only stall: fast detection (5 iters of get_todo_list)
                    # 2. Framework-only cap: catches loops of state-changing
                    #    framework tools (update_todo_list, create_todo_list)
                    #    that evade the read-only detector.  Progress-producing
                    #    tools reset this counter, so only pure bookkeeping
                    #    loops are caught.

                    # Advisory warning before force-completion
                    _approaching_stall = (
                        _consecutive_stall_iterations == self._stall.max_framework_stall_iterations - 2
                        or _consecutive_fw_only_iterations == self._stall.max_fw_only_consecutive_iterations - 2
                    )
                    if _approaching_stall and not _stall_warning_injected:
                        _stall_warning_injected = True
                        messages.append(Message(
                            role=MessageRole.USER,
                            content=(
                                "[STALL WARNING] You have been calling only "
                                "framework tools (get_todo_list, get_findings, "
                                "etc.) without making progress with domain tools. "
                                "This pattern suggests you may be stuck. Consider: "
                                "(1) calling domain tools to fetch or process data, "
                                "(2) using log_finding or mark_todo_complete to "
                                "record progress, or (3) producing your final "
                                "response. The phase will be force-completed if "
                                "no domain tool progress is detected soon."
                            ),
                        ))

                    _hit_stall = (
                        _consecutive_stall_iterations >= self._stall.max_framework_stall_iterations
                        or _consecutive_fw_only_iterations >= self._stall.max_fw_only_consecutive_iterations
                    )
                    if _hit_stall:
                        _reason = (
                            "read-only stall" if _consecutive_stall_iterations >= self._stall.max_framework_stall_iterations
                            else "framework-only loop"
                        )
                        logger.warning(
                            f"Framework stall detected ({_reason}): "
                            f"{_consecutive_fw_only_iterations} consecutive "
                            f"framework-only iterations in "
                            f"{phase.value} for agent {node.agent_id}. "
                            f"Forcing phase completion."
                        )
                        result.iterations = iteration
                        result.hit_iteration_limit = True
                        result.content = self._extract_best_effort(messages)

                        await self._event_bus.publish(phase_iteration_limit(
                            workflow_id=workflow.workflow_id,
                            task_id=node.task_id,
                            agent_id=node.agent_id,
                            phase=phase.value,
                            limit=self._stall.max_fw_only_consecutive_iterations,
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
