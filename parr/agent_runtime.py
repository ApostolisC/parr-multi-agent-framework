"""
Agent Runtime for the Agentic Framework.

Runs either:
    1) Direct-answer bypass for simple queries, or
    2) The full 4-phase lifecycle:
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
import re
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
    Message,
    MessageRole,
    ModelConfig,
    Phase,
    SimpleQueryBypassConfig,
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
EXECUTION_PATH_DIRECT = "direct_answer"
EXECUTION_PATH_FULL = "full_workflow"


class AgentRuntime:
    """
    Executes a single agent through direct-answer or phased lifecycle.

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
        simple_query_bypass: Optional[SimpleQueryBypassConfig] = None,
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
        self._simple_query_bypass = simple_query_bypass or SimpleQueryBypassConfig()
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
        execution_metadata.execution_path = EXECUTION_PATH_FULL

        try:
            routing_decision = await self._route_execution_path(
                config=config,
                input=input,
                node=node,
                workflow=workflow,
                total_usage=total_usage,
            )
            execution_metadata.routing_decision = routing_decision
            selected_path = routing_decision.get("selected_path", EXECUTION_PATH_FULL)
            execution_metadata.execution_path = selected_path

            if selected_path == EXECUTION_PATH_DIRECT:
                direct_result = await self._run_direct_answer(
                    config=config,
                    input=input,
                    node=node,
                    workflow=workflow,
                    total_usage=total_usage,
                )
                if direct_result.get("escalate_to_full_workflow"):
                    execution_metadata.execution_path = EXECUTION_PATH_FULL
                    routing_decision["initial_selected_path"] = EXECUTION_PATH_DIRECT
                    routing_decision["selected_path"] = EXECUTION_PATH_FULL
                    routing_decision["escalated_after_direct_answer"] = True
                    routing_decision["escalation_reason"] = direct_result.get("reason")
                else:
                    routing_decision["escalated_after_direct_answer"] = False
                    output = self._build_direct_output(
                        task_id=task_id,
                        config=config,
                        direct_result=direct_result,
                        total_usage=total_usage,
                        execution_metadata=execution_metadata,
                        errors=errors,
                        start_time=start_time,
                    )
                    node.status = AgentStatus.COMPLETED
                    node.result = output
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

            # ── Phase 3: Review (with retry loop) ──
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

            # Persist any in-progress phase data that was interrupted.
            # The phase_runner tracks the current PhaseResult which
            # accumulates tool_calls_made incrementally.
            partial_phase = getattr(phase_runner, '_in_progress_result', None)
            if partial_phase and partial_phase.tool_calls_made:
                partial_phase.content = None  # No LLM summary was produced
                partial_phase.iterations = len([
                    tc for tc in partial_phase.tool_calls_made
                ]) or partial_phase.iterations
                self._persist_phase(partial_phase, memory)
                self._accumulate_usage(total_usage, partial_phase)
                self._accumulate_tool_calls(execution_metadata, partial_phase)
                phase_name = partial_phase.phase.value
                if phase_name not in execution_metadata.phases_completed:
                    execution_metadata.phases_completed.append(phase_name)

            # Include limit type in the detail for specificity
            limit_type = getattr(e, 'limit_type', 'unknown')
            detail = str(e)
            if limit_type != 'unknown':
                detail = f"[{limit_type}] {detail}"

            output = self._build_partial_output(
                task_id=task_id,
                config=config,
                memory=memory,
                reason="budget_exceeded",
                detail=detail,
                total_usage=total_usage,
                execution_metadata=execution_metadata,
                errors=errors,
                start_time=start_time,
            )
            node.status = AgentStatus.FAILED
            node.result = output

            # Persist output + status even on failure
            self._persist_output(output)

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
                # Persist partial phase data for wrapped budget exceptions too
                partial_phase = getattr(phase_runner, '_in_progress_result', None)
                if partial_phase and partial_phase.tool_calls_made:
                    partial_phase.content = None
                    self._persist_phase(partial_phase, memory)
                    self._accumulate_usage(total_usage, partial_phase)
                    self._accumulate_tool_calls(execution_metadata, partial_phase)
                    phase_name = partial_phase.phase.value
                    if phase_name not in execution_metadata.phases_completed:
                        execution_metadata.phases_completed.append(phase_name)
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

            # Persist output + status even on failure
            self._persist_output(output)

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
        for tool in build_report_tools(
            memory,
            self._report_template_handler,
            output_schema,
            default_role=node.config.role,
            default_sub_role=node.config.sub_role,
        ):
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

    async def _route_execution_path(
        self,
        config: AgentConfig,
        input: AgentInput,
        node: AgentNode,
        workflow: WorkflowExecution,
        total_usage: TokenUsage,
    ) -> Dict[str, Any]:
        """Decide whether to run direct-answer mode or full workflow."""
        cfg = self._simple_query_bypass
        decision: Dict[str, Any] = {
            "mode": EXECUTION_PATH_FULL,
            "confidence": 0.0,
            "reason": "",
            "requires_external_data": False,
            "selected_path": EXECUTION_PATH_FULL,
            "policy_reason": "",
            "parser_status": "skipped",
        }

        if not cfg.enabled:
            decision.update({
                "reason": "Simple-query bypass is disabled.",
                "policy_reason": "bypass_disabled",
            })
            return decision

        if cfg.force_full_workflow_if_output_schema and input.output_schema:
            decision.update({
                "reason": "Output schema is required for this task.",
                "policy_reason": "output_schema_requires_full_workflow",
            })
            return decision

        routing_model_config = ModelConfig(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max(96, min(256, cfg.direct_answer_max_tokens)),
        )
        messages = self._build_routing_messages(input)
        response = await self._call_llm_without_tools(
            messages=messages,
            config=config,
            node=node,
            workflow=workflow,
            model_config=routing_model_config,
            total_usage=total_usage,
        )

        raw_content = (response.content or "").strip()
        parsed = self._parse_json_object(raw_content)
        if not isinstance(parsed, dict):
            decision.update({
                "reason": "Routing output was not valid JSON.",
                "policy_reason": "invalid_router_output",
                "parser_status": "invalid_json",
                "raw_response_preview": raw_content[:300],
            })
            return decision

        mode = str(parsed.get("mode", EXECUTION_PATH_FULL)).strip().lower()
        if mode not in {EXECUTION_PATH_DIRECT, EXECUTION_PATH_FULL}:
            mode = EXECUTION_PATH_FULL
        confidence = self._normalize_confidence(parsed.get("confidence", 0.0))
        reason = str(parsed.get("reason", "")).strip() or "No routing reason provided."
        requires_external_data = self._coerce_bool(
            parsed.get("requires_external_data"),
            default=False,
        )

        decision.update({
            "mode": mode,
            "confidence": confidence,
            "reason": reason,
            "requires_external_data": requires_external_data,
            "parser_status": "ok",
            "raw_response_preview": raw_content[:300],
        })

        if mode != EXECUTION_PATH_DIRECT:
            decision["selected_path"] = EXECUTION_PATH_FULL
            decision["policy_reason"] = "router_selected_full_workflow"
            return decision

        if confidence < cfg.route_confidence_threshold:
            decision["selected_path"] = EXECUTION_PATH_FULL
            decision["policy_reason"] = "route_confidence_below_threshold"
            return decision

        if requires_external_data:
            decision["selected_path"] = EXECUTION_PATH_FULL
            decision["policy_reason"] = "router_requires_external_data"
            return decision

        decision["selected_path"] = EXECUTION_PATH_DIRECT
        decision["policy_reason"] = "router_selected_direct_answer"
        return decision

    async def _run_direct_answer(
        self,
        config: AgentConfig,
        input: AgentInput,
        node: AgentNode,
        workflow: WorkflowExecution,
        total_usage: TokenUsage,
    ) -> Dict[str, Any]:
        """Attempt direct response without tools; may request escalation."""
        cfg = self._simple_query_bypass
        model_config = ModelConfig(
            temperature=min(config.model_config.temperature, 0.3),
            top_p=config.model_config.top_p,
            max_tokens=cfg.direct_answer_max_tokens,
        )
        messages = self._build_direct_answer_messages(input)
        response = await self._call_llm_without_tools(
            messages=messages,
            config=config,
            node=node,
            workflow=workflow,
            model_config=model_config,
            total_usage=total_usage,
        )

        raw_content = (response.content or "").strip()
        parsed = self._parse_json_object(raw_content)

        answer = ""
        reason = ""
        confidence = 0.0
        needs_full_workflow = True
        parser_status = "invalid_json"

        if isinstance(parsed, dict):
            parser_status = "ok"
            answer = str(parsed.get("answer", "")).strip()
            reason = str(parsed.get("reason", "")).strip()
            confidence = self._normalize_confidence(parsed.get("confidence", 0.0))
            needs_full_workflow = self._coerce_bool(
                parsed.get("needs_full_workflow"),
                default=False,
            )
            if not answer and not needs_full_workflow:
                needs_full_workflow = True
                reason = reason or "Direct response did not include an answer."
        else:
            answer = raw_content
            reason = "Direct-answer output was not valid JSON."
            confidence = 0.0
            needs_full_workflow = True

        if not reason:
            reason = "Direct-answer attempt completed."

        escalate = cfg.allow_escalation_to_full_workflow and (
            needs_full_workflow
            or confidence < cfg.route_confidence_threshold
        )

        return {
            "answer": answer,
            "confidence": confidence,
            "needs_full_workflow": needs_full_workflow,
            "reason": reason,
            "parser_status": parser_status,
            "raw_response_preview": raw_content[:300],
            "escalate_to_full_workflow": escalate,
        }

    async def _call_llm_without_tools(
        self,
        *,
        messages: List[Message],
        config: AgentConfig,
        node: AgentNode,
        workflow: WorkflowExecution,
        model_config: ModelConfig,
        total_usage: TokenUsage,
    ):
        """Run an LLM call with no tools and account for budget/usage."""
        self._budget_tracker.check_budget(node, workflow)
        response = await self._llm.chat_with_tools(
            messages=messages,
            tools=[],
            model=config.model,
            model_config=model_config,
            stream=False,
            on_token=None,
        )

        usage = response.usage or TokenUsage()
        cost = self._budget_tracker.record_usage(node, workflow, usage, config.model)
        total_usage.input_tokens += usage.input_tokens
        total_usage.output_tokens += usage.output_tokens
        total_usage.total_cost += cost
        return response

    def _build_routing_messages(self, input: AgentInput) -> List[Message]:
        """Build the routing prompt requesting strict JSON output."""
        user_payload = [
            f"Task: {input.task}",
            f"Has output schema: {bool(input.output_schema)}",
            f"Has raw_data: {bool(input.raw_data)}",
            f"RAG results count: {len(input.rag_results or [])}",
            f"Has additional context: {bool(input.additional_context)}",
            "",
            "Respond with JSON only using this schema:",
            '{"mode":"direct_answer|full_workflow","confidence":0.0,'
            '"reason":"...","requires_external_data":false}',
        ]
        return [
            Message(
                role=MessageRole.SYSTEM,
                content=(
                    "You are a routing controller. Decide if a task should be "
                    "answered directly without tools or run through a full multi-phase workflow. "
                    "Return JSON only."
                ),
            ),
            Message(role=MessageRole.USER, content="\n".join(user_payload)),
        ]

    def _build_direct_answer_messages(self, input: AgentInput) -> List[Message]:
        """Build prompt for direct-answer mode with optional escalation signal."""
        context_parts = [f"Task: {input.task}"]
        if input.additional_context:
            context_parts.append(
                f"Additional context:\n{str(input.additional_context)[:1200]}"
            )
        if input.raw_data:
            raw_data_str = json.dumps(input.raw_data, ensure_ascii=False, default=str)
            context_parts.append(f"raw_data (truncated):\n{raw_data_str[:2000]}")
        if input.rag_results:
            rag_preview = json.dumps((input.rag_results or [])[:3], ensure_ascii=False, default=str)
            context_parts.append(f"rag_results (first 3, truncated):\n{rag_preview[:2000]}")

        context_parts.append(
            "Respond with JSON only: "
            '{"answer":"...","confidence":0.0,"needs_full_workflow":false,"reason":"..."}'
        )
        return [
            Message(
                role=MessageRole.SYSTEM,
                content=(
                    "Answer the user directly without tools when possible. "
                    "If context is missing or uncertainty is high, set "
                    '"needs_full_workflow" to true.'
                ),
            ),
            Message(role=MessageRole.USER, content="\n\n".join(context_parts)),
        ]

    @staticmethod
    def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Parse a JSON object from plain text or fenced-code output."""
        raw = (text or "").strip()
        if not raw:
            return None

        candidates: List[str] = [raw]
        fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
        candidates.extend(m.strip() for m in fence_matches if m.strip())

        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidates.append(raw[first_brace:last_brace + 1].strip())

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _normalize_confidence(value: Any) -> float:
        """Normalize confidence into [0, 1]."""
        try:
            conf = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, conf))

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        """Coerce booleans from bool/string/number forms."""
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        return default

    def _build_direct_output(
        self,
        task_id: str,
        config: AgentConfig,
        direct_result: Dict[str, Any],
        total_usage: TokenUsage,
        execution_metadata: ExecutionMetadata,
        errors: List[ErrorEntry],
        start_time: float,
    ) -> AgentOutput:
        """Build final output for direct-answer execution path."""
        execution_metadata.total_duration_ms = (time.time() - start_time) * 1000
        answer = str(direct_result.get("answer", "")).strip()
        confidence = self._normalize_confidence(direct_result.get("confidence", 0.0))
        reason = str(direct_result.get("reason", "")).strip()
        parser_status = str(direct_result.get("parser_status", "invalid_json"))
        raw_preview = str(direct_result.get("raw_response_preview", ""))

        if answer:
            execution_metadata.phase_outputs["direct_answer"] = answer
        execution_metadata.tools_called = []

        status = "completed"
        if confidence < self._simple_query_bypass.route_confidence_threshold:
            status = "degraded"
            errors.append(ErrorEntry(
                source=ErrorSource.SYSTEM,
                name="agent_runtime",
                error_type="direct_answer_low_confidence",
                message=(
                    f"Direct answer confidence {confidence:.2f} is below threshold "
                    f"{self._simple_query_bypass.route_confidence_threshold:.2f}."
                ),
                recoverable=True,
            ))

        if parser_status != "ok":
            status = "degraded"
            errors.append(ErrorEntry(
                source=ErrorSource.SYSTEM,
                name="agent_runtime",
                error_type="direct_answer_parse_warning",
                message=(
                    "Direct-answer response was not valid JSON. "
                    f"Preview: {raw_preview[:200]}"
                ),
                recoverable=True,
            ))

        summary = answer or reason or "Direct-answer path completed."
        if len(summary) > 500:
            summary = summary[:500] + "..."

        findings: Dict[str, Any] = {
            "direct_answer": answer,
            "confidence": confidence,
            "reason": reason,
        }

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

        # Determine status based on quality signals.
        # "degraded" signals that the output exists but is likely incomplete:
        #   - All phases hit iteration limits  (original rule)
        #   - Act + report both hit limits  (primary data + synthesis degraded)
        #   - 2+ phases hit limits AND review explicitly failed
        # Partial hits with passing review are still "completed" with a
        # warning appended to errors.
        status = "completed"
        review_failed = (
            memory.review_checklist
            and any(item.rating == "fail" for item in memory.review_checklist)
        )

        if phases_hitting_limit:
            critical_phases_hit = {"act", "report"}.issubset(set(phases_hitting_limit))
            many_limits_and_review_failed = (
                len(phases_hitting_limit) >= 2 and review_failed
            )

            if (len(phases_hitting_limit) >= 4
                    or critical_phases_hit
                    or many_limits_and_review_failed):
                status = "degraded"
                logger.warning(
                    f"Agent {config.agent_id} output degraded: "
                    f"phases_hitting_limit={phases_hitting_limit}, "
                    f"review_failed={review_failed}"
                )
                errors.append(ErrorEntry(
                    source=ErrorSource.SYSTEM,
                    name="agent_runtime",
                    error_type="quality_degraded",
                    message=(
                        f"Output quality degraded. "
                        f"Phases hitting limits: {', '.join(phases_hitting_limit)}."
                        + (f" Review failed." if review_failed else "")
                    ),
                    recoverable=True,
                ))
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
