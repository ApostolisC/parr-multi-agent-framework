"""
Agent Runtime for the Agentic Framework.

Runs the full 4-phase lifecycle for a single agent:
    Plan → Act → Review → Report

Each phase is delegated to the PhaseRunner. The runtime manages
phase transitions, context passing between phases, review retry loops,
and structured output extraction.

The runtime does NOT manage the agent tree — that's the orchestrator's job.
The runtime handles ONE agent at a time.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as jsonschema_validate

from .adapters.llm_adapter import ContentFilterError
from .budget_tracker import BudgetExceededException, BudgetTracker
from .context_manager import ContextManager
from .event_bus import EventBus
from .event_types import (
    agent_completed,
    agent_failed,
    agent_started,
    phase_completed,
)
from .framework_tools import (
    AgentWorkingMemory,
    build_act_tools,
    build_agent_management_tools,
    build_plan_tools,
    build_report_tools,
    build_review_tools,
)
from .persistence import AgentFileStore
from .phase_runner import CancelledException, PhaseResult, PhaseRunner
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry
from .core_types import (
    AgentConfig,
    AgentInput,
    AgentNode,
    AgentOutput,
    AgentStatus,
    BudgetConfig,
    ErrorEntry,
    ErrorSource,
    ExecutionMetadata,
    Phase,
    StallDetectionConfig,
    ToolCall,
    ToolDef,
    ToolResult,
    TokenUsage,
    WorkflowExecution,
    generate_id,
)
from .protocols import ToolCallingLLM

logger = logging.getLogger(__name__)

# Review pass/fail detection
REVIEW_PASSED_MARKER = "REVIEW_PASSED"
REVIEW_FAILED_MARKER = "REVIEW_FAILED"


class AgentRuntime:
    """
    Executes a single agent through its full phase lifecycle.

    Usage:
        runtime = AgentRuntime(llm, budget_tracker, event_bus, ...)
        output = await runtime.execute(config, input, node, workflow)
    """

    def __init__(
        self,
        llm: ToolCallingLLM,
        budget_tracker: BudgetTracker,
        event_bus: EventBus,
        max_review_cycles: int = 2,
        phase_limits: Optional[Dict[Phase, int]] = None,
        available_roles_description: str = "",
        report_template_handler: Optional[Callable] = None,
        stream: bool = False,
        stall_config: Optional[StallDetectionConfig] = None,
        budget_config: Optional[BudgetConfig] = None,
        agent_file_store: Optional[AgentFileStore] = None,
    ) -> None:
        self._llm = llm
        self._budget_tracker = budget_tracker
        self._event_bus = event_bus
        self._max_review_cycles = max_review_cycles
        self._phase_limits = phase_limits
        self._available_roles_description = available_roles_description
        self._report_template_handler = report_template_handler
        self._stream = stream
        self._stall_config = stall_config
        self._budget_config = budget_config
        self._file_store = agent_file_store

    async def execute(
        self,
        config: AgentConfig,
        input: AgentInput,
        node: AgentNode,
        workflow: WorkflowExecution,
        on_orchestrator_tool: Optional[Callable] = None,
    ) -> AgentOutput:
        """
        Execute the full agent lifecycle.

        Args:
            config: Agent configuration.
            input: Agent input (task, tools, etc.).
            node: The agent's node in the tree.
            workflow: The parent workflow.
            on_orchestrator_tool: Callback for spawn_agent, wait_for_agents, etc.

        Returns:
            AgentOutput (always produced, even on failure).
        """
        start_time = time.time()
        task_id = node.task_id

        # Set up working memory for this agent
        memory = AgentWorkingMemory()

        # Build tool registry with framework + domain tools
        registry = self._build_tool_registry(memory, input.tools, node, input.output_schema)

        # Set up components
        tool_executor = ToolExecutor(registry)
        # Estimate tool schema overhead: ~150 tokens per registered tool
        tool_schema_overhead = len(registry.get_all()) * 150
        context_manager = ContextManager(
            tool_schema_overhead=tool_schema_overhead,
            soft_compaction_pct=self._budget_config.context_soft_compaction_pct if self._budget_config else 0.40,
            hard_truncation_pct=self._budget_config.context_hard_truncation_pct if self._budget_config else 0.65,
            chars_per_token=self._budget_config.chars_per_token if self._budget_config else 4.0,
        )

        phase_runner = PhaseRunner(
            llm=self._llm,
            tool_executor=tool_executor,
            tool_registry=registry,
            budget_tracker=self._budget_tracker,
            context_manager=context_manager,
            event_bus=self._event_bus,
            phase_limits=self._phase_limits,
            stream=self._stream,
            stall_config=self._stall_config,
        )

        # Emit agent started
        await self._event_bus.publish(agent_started(
            workflow_id=workflow.workflow_id,
            task_id=task_id,
            agent_id=config.agent_id,
            role=config.role,
            sub_role=config.sub_role,
            depth=node.depth,
        ))

        execution_metadata = ExecutionMetadata()
        total_usage = TokenUsage()
        errors: List[ErrorEntry] = []
        phases_hitting_limit: List[str] = []

        try:
            # ── Phase 1: Plan ──
            plan_result = await self._run_phase(
                phase_runner=phase_runner,
                phase=Phase.PLAN,
                node=node,
                workflow=workflow,
                config=config,
                input=input,
                context_manager=context_manager,
                memory=memory,
                on_orchestrator_tool=on_orchestrator_tool,
            )
            self._accumulate_usage(total_usage, plan_result)
            self._accumulate_tool_calls(execution_metadata, plan_result)
            execution_metadata.phases_completed.append("plan")
            execution_metadata.iterations_per_phase["plan"] = plan_result.iterations
            execution_metadata.phase_outputs["plan"] = plan_result.content or ""
            if plan_result.hit_iteration_limit:
                phases_hitting_limit.append("plan")
            self._persist_phase(plan_result, memory)

            # ── Phase 2: Act ──
            act_result = await self._run_phase(
                phase_runner=phase_runner,
                phase=Phase.ACT,
                node=node,
                workflow=workflow,
                config=config,
                input=input,
                context_manager=context_manager,
                memory=memory,
                on_orchestrator_tool=on_orchestrator_tool,
            )
            self._accumulate_usage(total_usage, act_result)
            self._accumulate_tool_calls(execution_metadata, act_result)
            execution_metadata.phases_completed.append("act")
            execution_metadata.iterations_per_phase["act"] = act_result.iterations
            execution_metadata.phase_outputs["act"] = act_result.content or ""
            if act_result.hit_iteration_limit:
                phases_hitting_limit.append("act")
            self._persist_phase(act_result, memory)
            review_result = await self._run_phase(
                phase_runner=phase_runner,
                phase=Phase.REVIEW,
                node=node,
                workflow=workflow,
                config=config,
                input=input,
                context_manager=context_manager,
                memory=memory,
                on_orchestrator_tool=on_orchestrator_tool,
            )
            self._accumulate_usage(total_usage, review_result)
            self._accumulate_tool_calls(execution_metadata, review_result)

            review_pass = self._evaluate_review(review_result, memory)
            review_iterations = review_result.iterations
            retry_count = 0
            prev_fail_count = self._count_review_failures(memory)

            # If the LLM produced no checklist and no REVIEW_PASSED/FAILED
            # marker, re-prompt once to get a clear signal before defaulting.
            if review_pass is None:
                logger.info(
                    f"Review produced no structured signal for agent "
                    f"{config.agent_id}. Re-prompting for explicit evaluation."
                )
                memory.review_checklist = None
                review_result = await self._run_phase(
                    phase_runner=phase_runner,
                    phase=Phase.REVIEW,
                    node=node,
                    workflow=workflow,
                    config=config,
                    input=input,
                    context_manager=context_manager,
                    memory=memory,
                    on_orchestrator_tool=on_orchestrator_tool,
                    extra_context=(
                        "Your previous review did not produce a clear verdict. "
                        "You MUST use the review_checklist tool to evaluate each "
                        "criterion, then state REVIEW_PASSED or REVIEW_FAILED."
                    ),
                )
                self._accumulate_usage(total_usage, review_result)
                self._accumulate_tool_calls(execution_metadata, review_result)
                review_pass = self._evaluate_review(review_result, memory)
                review_iterations += review_result.iterations
                # If still ambiguous after re-prompt, assume pass
                if review_pass is None:
                    review_pass = True

            while not review_pass and retry_count < self._max_review_cycles:
                retry_count += 1
                logger.info(
                    f"Review failed for agent {config.agent_id}. "
                    f"Retry {retry_count}/{self._max_review_cycles}"
                )

                # Extract review feedback
                review_feedback = self._extract_review_feedback(review_result, memory)

                # Clear stale review checklist so the LLM generates a fresh one
                memory.review_checklist = None

                # Re-run Act with review feedback
                act_result = await self._run_phase(
                    phase_runner=phase_runner,
                    phase=Phase.ACT,
                    node=node,
                    workflow=workflow,
                    config=config,
                    input=input,
                    context_manager=context_manager,
                    memory=memory,
                    on_orchestrator_tool=on_orchestrator_tool,
                    extra_context=review_feedback,
                )
                self._accumulate_usage(total_usage, act_result)
                self._accumulate_tool_calls(execution_metadata, act_result)

                # Re-run Review
                review_result = await self._run_phase(
                    phase_runner=phase_runner,
                    phase=Phase.REVIEW,
                    node=node,
                    workflow=workflow,
                    config=config,
                    input=input,
                    context_manager=context_manager,
                    memory=memory,
                    on_orchestrator_tool=on_orchestrator_tool,
                )
                self._accumulate_usage(total_usage, review_result)
                self._accumulate_tool_calls(execution_metadata, review_result)

                review_pass = self._evaluate_review(review_result, memory)
                review_iterations += review_result.iterations
                # In retry loop, ambiguous means the LLM still isn't
                # producing a signal — treat as pass to avoid looping.
                if review_pass is None:
                    review_pass = True

                # Plateau detection: stop retrying if quality isn't improving
                current_fail_count = self._count_review_failures(memory)
                if not review_pass and current_fail_count >= prev_fail_count:
                    logger.warning(
                        f"Review quality not improving for agent {config.agent_id} "
                        f"(fail count: {current_fail_count} >= {prev_fail_count}). "
                        f"Stopping retries to avoid wasting tokens."
                    )
                    break
                prev_fail_count = current_fail_count

            if not review_pass:
                logger.warning(
                    f"Review still failing after {retry_count} retries "
                    f"for agent {config.agent_id}. Proceeding with current output."
                )

            execution_metadata.phases_completed.append("review")
            execution_metadata.iterations_per_phase["review"] = review_iterations
            execution_metadata.phase_outputs["review"] = review_result.content or ""
            if review_result.hit_iteration_limit:
                phases_hitting_limit.append("review")
            self._persist_phase(review_result, memory)

            # ── Phase 4: Report ──
            report_result = await self._run_phase(
                phase_runner=phase_runner,
                phase=Phase.REPORT,
                node=node,
                workflow=workflow,
                config=config,
                input=input,
                context_manager=context_manager,
                memory=memory,
                on_orchestrator_tool=on_orchestrator_tool,
            )
            self._accumulate_usage(total_usage, report_result)
            self._accumulate_tool_calls(execution_metadata, report_result)
            execution_metadata.phases_completed.append("report")
            execution_metadata.iterations_per_phase["report"] = report_result.iterations
            execution_metadata.phase_outputs["report"] = report_result.content or ""
            if report_result.hit_iteration_limit:
                phases_hitting_limit.append("report")
            self._persist_phase(report_result, memory)

            # Validate submitted report against output schema (graceful)
            if memory.submitted_report and input.output_schema:
                try:
                    jsonschema_validate(
                        instance=memory.submitted_report,
                        schema=input.output_schema,
                    )
                except JsonSchemaValidationError as e:
                    logger.warning(
                        "Report schema validation failed for agent %s: %s",
                        config.agent_id, e.message,
                    )
                    errors.append(ErrorEntry(
                        source=ErrorSource.SYSTEM,
                        name="agent_runtime",
                        error_type="output_schema_validation",
                        message=(
                            f"Submitted report does not match output_schema: "
                            f"{e.message}"
                        ),
                        recoverable=True,
                    ))

            # Build output from submitted report or phase content
            output = self._build_output(
                task_id=task_id,
                config=config,
                memory=memory,
                report_result=report_result,
                total_usage=total_usage,
                execution_metadata=execution_metadata,
                errors=errors,
                start_time=start_time,
                node=node,
                phases_hitting_limit=phases_hitting_limit,
            )

            node.status = AgentStatus.COMPLETED
            node.result = output

            # Persist final output and status
            self._persist_output(output)

            await self._event_bus.publish(agent_completed(
                workflow_id=workflow.workflow_id,
                task_id=task_id,
                agent_id=config.agent_id,
                summary=output.summary,
                token_usage={
                    "input_tokens": total_usage.input_tokens,
                    "output_tokens": total_usage.output_tokens,
                },
            ))

            return output

        except BudgetExceededException as e:
            logger.warning(f"Budget exceeded for agent {config.agent_id}: {e}")
            output = self._build_partial_output(
                task_id=task_id,
                config=config,
                memory=memory,
                reason="budget_exceeded",
                detail=str(e),
                total_usage=total_usage,
                execution_metadata=execution_metadata,
                errors=errors,
                start_time=start_time,
            )
            node.status = AgentStatus.FAILED
            node.result = output

            await self._event_bus.publish(agent_failed(
                workflow_id=workflow.workflow_id,
                task_id=task_id,
                agent_id=config.agent_id,
                reason=str(e),
            ))

            return output

        except ContentFilterError as e:
            logger.warning(
                f"Content filter blocked agent {config.agent_id}: {e}"
            )
            # Produce a degraded output preserving all work done so far,
            # rather than failing completely.
            output = self._build_partial_output(
                task_id=task_id,
                config=config,
                memory=memory,
                reason="content_filter",
                detail=str(e),
                total_usage=total_usage,
                execution_metadata=execution_metadata,
                errors=errors,
                start_time=start_time,
            )
            # Use COMPLETED status if we have meaningful results, FAILED otherwise
            has_findings = bool(memory.findings)
            has_report = bool(memory.submitted_report)
            if has_findings or has_report:
                output.status = "degraded"
                node.status = AgentStatus.COMPLETED
            else:
                node.status = AgentStatus.FAILED
            node.result = output

            await self._event_bus.publish(agent_failed(
                workflow_id=workflow.workflow_id,
                task_id=task_id,
                agent_id=config.agent_id,
                reason=f"content_filter: {e}",
            ))

            return output

        except CancelledException:
            logger.info(f"Agent {config.agent_id} cancelled")
            output = self._build_partial_output(
                task_id=task_id,
                config=config,
                memory=memory,
                reason="cancelled",
                detail="Agent was cancelled",
                total_usage=total_usage,
                execution_metadata=execution_metadata,
                errors=errors,
                start_time=start_time,
            )
            node.status = AgentStatus.CANCELLED
            node.result = output
            return output

        except Exception as e:
            logger.error(f"Agent {config.agent_id} failed: {e}", exc_info=True)
            # Check whether a framework-specific exception was wrapped
            # (e.g. by _with_retry's RuntimeError wrapper) so we can
            # preserve the correct error classification.
            cause = e.__cause__
            if isinstance(cause, BudgetExceededException):
                reason = "budget_exceeded"
            else:
                reason = "error"
            output = self._build_partial_output(
                task_id=task_id,
                config=config,
                memory=memory,
                reason=reason,
                detail=str(e),
                total_usage=total_usage,
                execution_metadata=execution_metadata,
                errors=errors,
                start_time=start_time,
            )
            node.status = AgentStatus.FAILED
            node.result = output

            await self._event_bus.publish(agent_failed(
                workflow_id=workflow.workflow_id,
                task_id=task_id,
                agent_id=config.agent_id,
                reason=str(e),
            ))

            return output

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _persist_phase(
        self, phase_result: PhaseResult, memory: AgentWorkingMemory,
    ) -> None:
        """Persist phase conversation, tool calls, and working memory."""
        if not self._file_store:
            return
        try:
            self._file_store.save_phase_conversation(
                phase_result.phase.value,
                content=phase_result.content,
                iterations=phase_result.iterations,
                hit_iteration_limit=phase_result.hit_iteration_limit,
                tool_calls_made=phase_result.tool_calls_made,
            )
            if phase_result.tool_calls_made:
                self._file_store.append_tool_calls(phase_result.tool_calls_made)
            self._file_store.save_memory(memory)
        except Exception as e:
            logger.error(f"Failed to persist phase data: {e}", exc_info=True)

    def _persist_output(self, output: AgentOutput) -> None:
        """Persist the agent's final output and update agent status."""
        if not self._file_store:
            return
        try:
            self._file_store.save_output(output)
            self._file_store.update_agent_status(output.status)
        except Exception as e:
            logger.error(f"Failed to persist agent output: {e}", exc_info=True)

    def _build_tool_registry(
        self,
        memory: AgentWorkingMemory,
        domain_tools: List[ToolDef],
        node: AgentNode,
        output_schema: Optional[Dict[str, Any]] = None,
    ) -> ToolRegistry:
        """Build the combined registry of framework + domain tools."""
        registry = ToolRegistry()

        # Framework tools
        for tool in build_plan_tools(memory):
            registry.register(tool)
        for tool in build_act_tools(memory):
            registry.register(tool)
        for tool in build_review_tools(memory):
            registry.register(tool)
        for tool in build_report_tools(memory, self._report_template_handler, output_schema):
            registry.register(tool)

        # Agent management tools — only register when:
        # 1. There are spawnable roles configured in the domain adapter
        # 2. The agent's depth allows spawning children (depth + 1 < max_agent_depth)
        # Without the depth check, the LLM sees spawn_agent, calls it, gets
        # "depth exceeded" errors, and wastes tokens retrying.
        can_spawn = (
            bool(self._available_roles_description and self._available_roles_description.strip())
            and node.depth + 1 < node.budget.max_agent_depth
        )
        if can_spawn:
            for tool in build_agent_management_tools(self._available_roles_description):
                registry.register(tool)

        # Domain tools from input — validate no shadowing of framework tools
        framework_names = set(registry.tool_names)
        for tool in domain_tools:
            if tool.name in framework_names:
                logger.warning(
                    f"Domain tool '{tool.name}' shadows a framework tool and "
                    f"will be skipped. Rename it to avoid conflicts."
                )
                continue
            registry.register(tool)

        return registry

    async def _run_phase(
        self,
        phase_runner: PhaseRunner,
        phase: Phase,
        node: AgentNode,
        workflow: WorkflowExecution,
        config: AgentConfig,
        input: AgentInput,
        context_manager: ContextManager,
        memory: AgentWorkingMemory,
        on_orchestrator_tool: Optional[Callable],
        extra_context: Optional[str] = None,
    ) -> PhaseResult:
        """Run a single phase and record the summary."""
        # Build working memory snapshot
        snapshot_parts = []
        todo = memory.get_todo_list()
        if todo != "No todo items.":
            snapshot_parts.append(f"Todo List:\n{todo}")
        findings = memory.get_findings()
        if findings != "No findings recorded.":
            snapshot_parts.append(f"Findings:\n{findings}")
        review = memory.get_review_summary()
        if review != "No review checklist recorded.":
            snapshot_parts.append(f"Review:\n{review}")
        working_snapshot = "\n\n".join(snapshot_parts) if snapshot_parts else None

        result = await phase_runner.run(
            phase=phase,
            node=node,
            workflow=workflow,
            config=config,
            input=input,
            working_memory_snapshot=working_snapshot,
            extra_context=extra_context,
            on_orchestrator_tool=on_orchestrator_tool,
        )

        # Record phase summary for cross-phase context
        if result.content:
            context_manager.record_phase_summary(phase, result.content)

        # Emit phase completed
        await self._event_bus.publish(phase_completed(
            workflow_id=workflow.workflow_id,
            task_id=node.task_id,
            agent_id=config.agent_id,
            phase=phase.value,
            iterations=result.iterations,
        ))

        return result

    def _evaluate_review(
        self, review_result: PhaseResult, memory: AgentWorkingMemory,
    ) -> Optional[bool]:
        """Determine if the review passed.

        The structured checklist (from the review_checklist tool) is the
        primary signal when available — it is more reliable than text
        markers because it provides per-criterion ratings.  Only "fail"
        ratings count as a true failure; "partial" ratings are treated as
        acceptable (the agent did *some* work, and a retry is unlikely to
        improve things when no additional data sources are available).

        Text markers (REVIEW_PASSED / REVIEW_FAILED) are used as a
        fallback when no checklist was recorded.

        Returns:
            True if review passed, False if it explicitly failed,
            None if no structured signal was found (ambiguous).
        """
        # Prefer structured checklist when available
        if memory.review_checklist:
            fail_count = sum(
                1 for item in memory.review_checklist if item.rating == "fail"
            )
            return fail_count == 0

        # Fall back to explicit text markers
        content = (review_result.content or "").strip()
        if REVIEW_PASSED_MARKER in content:
            return True
        if REVIEW_FAILED_MARKER in content:
            return False

        # No structured signal — caller should re-prompt once
        return None

    @staticmethod
    def _count_review_failures(memory: AgentWorkingMemory) -> int:
        """Count the number of 'fail' ratings in the current review checklist."""
        if not memory.review_checklist:
            return 0
        return sum(1 for item in memory.review_checklist if item.rating == "fail")

    def _extract_review_feedback(
        self, review_result: PhaseResult, memory: AgentWorkingMemory,
    ) -> str:
        """Extract actionable feedback from the review for act retry."""
        parts = []
        if review_result.content:
            parts.append(f"Review assessment:\n{review_result.content}")
        if memory.review_checklist:
            failed = [
                item for item in memory.review_checklist
                if item.rating in ("fail", "partial")
            ]
            if failed:
                lines = [
                    f"- [{item.rating}] {item.criterion}: {item.justification}"
                    for item in failed
                ]
                parts.append(
                    "Issues to address:\n" + "\n".join(lines)
                )
        else:
            # Review said FAILED but no checklist was used — inject stronger
            # guidance so the next review uses the review_checklist tool
            parts.append(
                "IMPORTANT: The previous review indicated failure but did not "
                "use the review_checklist tool to specify which criteria failed. "
                "During the next review phase, you MUST use the review_checklist "
                "tool to evaluate each criterion explicitly."
            )
        return "\n\n".join(parts) if parts else "Review indicated issues. Please re-examine your work."

    def _accumulate_usage(self, total: TokenUsage, result: PhaseResult) -> None:
        total.input_tokens += result.total_usage.input_tokens
        total.output_tokens += result.total_usage.output_tokens
        total.total_cost += result.total_usage.total_cost

    @staticmethod
    def _accumulate_tool_calls(
        metadata: ExecutionMetadata, result: PhaseResult,
    ) -> None:
        """Append tool call records from a phase result to execution metadata."""
        for tc_info in result.tool_calls_made:
            metadata.tools_called.append(tc_info)

    def _build_output(
        self,
        task_id: str,
        config: AgentConfig,
        memory: AgentWorkingMemory,
        report_result: PhaseResult,
        total_usage: TokenUsage,
        execution_metadata: ExecutionMetadata,
        errors: List[ErrorEntry],
        start_time: float,
        node: Optional[AgentNode] = None,
        phases_hitting_limit: Optional[List[str]] = None,
    ) -> AgentOutput:
        """Build the final AgentOutput from the agent's work."""
        execution_metadata.total_duration_ms = (time.time() - start_time) * 1000

        # Populate sub_agents_spawned from the node's children list
        if node and node.children:
            execution_metadata.sub_agents_spawned = list(node.children)

        # Use submitted report if available, else use report phase content
        findings = {}
        if memory.submitted_report:
            findings = memory.submitted_report
        elif report_result.content:
            findings = {"report_text": report_result.content}

        # Build summary from findings + report content
        summary = report_result.content or "Agent completed without producing a text summary."
        if len(summary) > 500:
            summary = summary[:500] + "..."

        # Determine status based on iteration limit hits.
        # "degraded" = all phases exhausted iteration budgets, output quality
        # is likely very low.  Partial hits are still "completed" with a
        # warning appended to errors.
        status = "completed"
        if phases_hitting_limit:
            if len(phases_hitting_limit) >= 4:
                status = "degraded"
                logger.warning(
                    f"All phases hit iteration limits for agent "
                    f"{config.agent_id}: {phases_hitting_limit}"
                )
            else:
                errors.append(ErrorEntry(
                    source=ErrorSource.SYSTEM,
                    name="agent_runtime",
                    error_type="iteration_limit_warning",
                    message=(
                        f"Phases hitting iteration limits: "
                        f"{', '.join(phases_hitting_limit)}. "
                        f"Output quality may be reduced for those phases."
                    ),
                    recoverable=True,
                ))

        return AgentOutput(
            task_id=task_id,
            agent_id=config.agent_id,
            role=config.role,
            sub_role=config.sub_role,
            status=status,
            summary=summary,
            findings=findings,
            errors=errors,
            token_usage=total_usage,
            execution_metadata=execution_metadata,
        )

    def _build_partial_output(
        self,
        task_id: str,
        config: AgentConfig,
        memory: AgentWorkingMemory,
        reason: str,
        detail: str,
        total_usage: TokenUsage,
        execution_metadata: ExecutionMetadata,
        errors: List[ErrorEntry],
        start_time: float,
    ) -> AgentOutput:
        """Build a partial AgentOutput on failure/cancellation."""
        execution_metadata.total_duration_ms = (time.time() - start_time) * 1000

        # Salvage what we can from memory
        findings = {}
        if memory.submitted_report:
            findings = memory.submitted_report
        elif memory.findings:
            findings = {
                "partial_findings": [
                    {"category": f.category, "content": f.content, "source": f.source}
                    for f in memory.findings
                ]
            }

        errors.append(ErrorEntry(
            source=ErrorSource.SYSTEM,
            name="agent_runtime",
            error_type=reason,
            message=detail,
            recoverable=False,
        ))

        return AgentOutput(
            task_id=task_id,
            agent_id=config.agent_id,
            role=config.role,
            sub_role=config.sub_role,
            status="failed" if reason != "cancelled" else "partial",
            summary=f"Agent terminated: {reason}. {detail}",
            findings=findings,
            errors=errors,
            token_usage=total_usage,
            execution_metadata=execution_metadata,
        )
