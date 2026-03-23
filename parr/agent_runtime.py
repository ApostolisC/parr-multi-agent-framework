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
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .adapters.llm_adapter import ContentFilterError
from .budget_tracker import BudgetExceededException, BudgetTracker
from .context_manager import ContextManager
from .event_bus import EventBus
from .output_validator import JsonSchemaValidator, OutputValidator
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
    build_coordination_tools,
    build_plan_tools,
    build_report_tools,
    build_review_tools,
    build_transition_tools,
)
from .persistence import AgentFileStore
from .phase_runner import CancelledException, PhaseResult, PhaseRunner
from .tool_executor import ToolExecutor
from .tool_registry import ToolRegistry
from .core_types import (
    AdaptiveFlowConfig,
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
    PhaseConfig,
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
EXECUTION_PATH_ADAPTIVE = "adaptive"


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
        phase_config: Optional[PhaseConfig] = None,
        output_validator: Optional[OutputValidator] = None,
        adaptive_config: Optional[AdaptiveFlowConfig] = None,
    ) -> None:
        self._llm = llm
        self._budget_tracker = budget_tracker
        self._event_bus = event_bus
        self._available_roles_description = available_roles_description
        self._report_template_handler = report_template_handler
        self._stream = stream
        self._stall_config = stall_config
        self._simple_query_bypass = simple_query_bypass or SimpleQueryBypassConfig()
        self._budget_config = budget_config
        self._file_store = agent_file_store
        self._output_validator = output_validator or JsonSchemaValidator()
        self._adaptive_config = adaptive_config
        # PhaseConfig: if provided, takes precedence; otherwise built from legacy params
        if phase_config is not None:
            self._phase_config = phase_config
        else:
            self._phase_config = PhaseConfig(
                max_review_cycles=max_review_cycles,
                phase_limits=phase_limits,
            )
        # Expose for backward compatibility
        self._max_review_cycles = self._phase_config.max_review_cycles
        self._phase_limits = self._phase_config.phase_limits

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
            phase_prompts=self._phase_config.phase_prompts,
            phase_sequence=self._phase_config.phases,
        )

        # Incremental persistence callback — writes each tool call and
        # memory snapshot to disk immediately so the debug UI can show
        # live updates during phase execution (not just at phase end).
        def _on_tool_persisted(tool_record: Dict[str, Any]) -> None:
            if not self._file_store:
                return
            try:
                self._file_store.append_tool_calls([tool_record])
                self._file_store.save_memory(memory)
            except Exception as e:
                logger.debug("Incremental persist failed: %s", e)

        # Incremental LLM call persistence — writes each LLM call record
        # to disk so the debug UI can show per-iteration details.
        def _on_llm_call_persisted(llm_record: Dict[str, Any]) -> None:
            if not self._file_store:
                return
            try:
                self._file_store.append_llm_calls([llm_record])
            except Exception as e:
                logger.debug("Incremental LLM call persist failed: %s", e)

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
            on_tool_persisted=_on_tool_persisted if self._file_store else None,
            on_llm_call_persisted=_on_llm_call_persisted if self._file_store else None,
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
            # ── Adaptive flow (agent-controlled path) ──
            if self._adaptive_config and self._adaptive_config.enabled:
                return await self._run_adaptive_flow(
                    config=config,
                    input=input,
                    node=node,
                    workflow=workflow,
                    memory=memory,
                    registry=registry,
                    tool_executor=tool_executor,
                    context_manager=context_manager,
                    phase_runner=phase_runner,
                    execution_metadata=execution_metadata,
                    total_usage=total_usage,
                    errors=errors,
                    start_time=start_time,
                    on_orchestrator_tool=on_orchestrator_tool,
                )

            # ── Legacy flow (router-controlled path) ──
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

            # ── Run configured phase sequence ──
            review_phase = self._phase_config.effective_review_phase
            review_retry_phase = self._phase_config.effective_review_retry_phase
            last_phase_result: Optional[PhaseResult] = None

            for phase in self._phase_config.phases:
                if phase == review_phase:
                    # Review phase with retry loop
                    last_phase_result = await self._run_review_cycle(
                        phase_runner=phase_runner,
                        review_phase=phase,
                        review_retry_phase=review_retry_phase,
                        node=node,
                        workflow=workflow,
                        config=config,
                        input=input,
                        context_manager=context_manager,
                        memory=memory,
                        on_orchestrator_tool=on_orchestrator_tool,
                        execution_metadata=execution_metadata,
                        total_usage=total_usage,
                        phases_hitting_limit=phases_hitting_limit,
                    )
                else:
                    last_phase_result = await self._execute_phase_with_bookkeeping(
                        phase_runner=phase_runner,
                        phase=phase,
                        node=node,
                        workflow=workflow,
                        config=config,
                        input=input,
                        context_manager=context_manager,
                        memory=memory,
                        on_orchestrator_tool=on_orchestrator_tool,
                        execution_metadata=execution_metadata,
                        total_usage=total_usage,
                        phases_hitting_limit=phases_hitting_limit,
                    )

            # Use the last phase result (typically Report) for output building
            report_result = last_phase_result

            # Validate submitted report via pluggable OutputValidator (graceful)
            if memory.submitted_report:
                validation = self._output_validator.validate(
                    output=memory.submitted_report,
                    schema=input.output_schema,
                    role=config.role,
                    sub_role=config.sub_role,
                )
                if not validation.is_valid:
                    for err_msg in validation.errors:
                        logger.warning(
                            "Output validation failed for agent %s: %s",
                            config.agent_id, err_msg,
                        )
                        errors.append(ErrorEntry(
                            source=ErrorSource.SYSTEM,
                            name="agent_runtime",
                            error_type="output_validation",
                            message=err_msg,
                            recoverable=True,
                        ))
                for warn_msg in validation.warnings:
                    logger.info(
                        "Output validation warning for agent %s: %s",
                        config.agent_id, warn_msg,
                    )
                    errors.append(ErrorEntry(
                        source=ErrorSource.SYSTEM,
                        name="agent_runtime",
                        error_type="output_validation_warning",
                        message=warn_msg,
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
        """Persist phase conversation and working memory.

        Tool calls are persisted incrementally during phase execution via
        the ``on_tool_persisted`` callback on PhaseRunner, so they are NOT
        re-appended here.  This method only writes the phase conversation
        summary and the final memory state.
        """
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

    async def _execute_phase_with_bookkeeping(
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
        execution_metadata: ExecutionMetadata,
        total_usage: TokenUsage,
        phases_hitting_limit: List[str],
        extra_context: Optional[str] = None,
    ) -> PhaseResult:
        """Run a phase and update all bookkeeping (metadata, usage, persistence)."""
        result = await self._run_phase(
            phase_runner=phase_runner,
            phase=phase,
            node=node,
            workflow=workflow,
            config=config,
            input=input,
            context_manager=context_manager,
            memory=memory,
            on_orchestrator_tool=on_orchestrator_tool,
            extra_context=extra_context,
        )
        self._accumulate_usage(total_usage, result)
        self._accumulate_tool_calls(execution_metadata, result)
        execution_metadata.phases_completed.append(phase.value)
        execution_metadata.iterations_per_phase[phase.value] = result.iterations
        execution_metadata.phase_outputs[phase.value] = result.content or ""
        if result.hit_iteration_limit:
            phases_hitting_limit.append(phase.value)
        self._persist_phase(result, memory)
        return result

    async def _run_review_cycle(
        self,
        phase_runner: PhaseRunner,
        review_phase: Phase,
        review_retry_phase: Optional[Phase],
        node: AgentNode,
        workflow: WorkflowExecution,
        config: AgentConfig,
        input: AgentInput,
        context_manager: ContextManager,
        memory: AgentWorkingMemory,
        on_orchestrator_tool: Optional[Callable],
        execution_metadata: ExecutionMetadata,
        total_usage: TokenUsage,
        phases_hitting_limit: List[str],
    ) -> PhaseResult:
        """Run the review phase with retry loop.

        If review fails and ``review_retry_phase`` is configured, re-runs the
        retry phase with review feedback, then re-runs the review phase, up to
        ``max_review_cycles`` times.
        """
        review_result = await self._run_phase(
            phase_runner=phase_runner,
            phase=review_phase,
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
                phase=review_phase,
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

        max_review_cycles = self._phase_config.max_review_cycles

        while not review_pass and retry_count < max_review_cycles:
            retry_count += 1
            logger.info(
                f"Review failed for agent {config.agent_id}. "
                f"Retry {retry_count}/{max_review_cycles}"
            )

            # Extract review feedback
            review_feedback = self._extract_review_feedback(review_result, memory)

            # Clear stale review checklist so the LLM generates a fresh one
            memory.review_checklist = None

            # Re-run retry phase with review feedback (if configured)
            if review_retry_phase is not None:
                retry_result = await self._run_phase(
                    phase_runner=phase_runner,
                    phase=review_retry_phase,
                    node=node,
                    workflow=workflow,
                    config=config,
                    input=input,
                    context_manager=context_manager,
                    memory=memory,
                    on_orchestrator_tool=on_orchestrator_tool,
                    extra_context=review_feedback,
                )
                self._accumulate_usage(total_usage, retry_result)
                self._accumulate_tool_calls(execution_metadata, retry_result)

            # Re-run review
            review_result = await self._run_phase(
                phase_runner=phase_runner,
                phase=review_phase,
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

        execution_metadata.phases_completed.append(review_phase.value)
        execution_metadata.iterations_per_phase[review_phase.value] = review_iterations
        execution_metadata.phase_outputs[review_phase.value] = review_result.content or ""
        if review_result.hit_iteration_limit:
            phases_hitting_limit.append(review_phase.value)
        self._persist_phase(review_result, memory)

        return review_result

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

        # Coordination tools (message passing, shared state)
        for tool in build_coordination_tools():
            registry.register(tool)

        # Transition tools (set_next_phase — adaptive flow)
        for tool in build_transition_tools(memory):
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

    # -------------------------------------------------------------------
    # Adaptive flow
    # -------------------------------------------------------------------

    # Maximum number of phase transitions to prevent infinite loops.
    # Each phase execution counts as one transition.
    _MAX_PHASE_TRANSITIONS = 12

    async def _run_adaptive_flow(
        self,
        config: AgentConfig,
        input: AgentInput,
        node: AgentNode,
        workflow: WorkflowExecution,
        memory: AgentWorkingMemory,
        registry: ToolRegistry,
        tool_executor: ToolExecutor,
        context_manager: ContextManager,
        phase_runner: PhaseRunner,
        execution_metadata: ExecutionMetadata,
        total_usage: TokenUsage,
        errors: List[ErrorEntry],
        start_time: float,
        on_orchestrator_tool: Optional[Callable],
    ) -> AgentOutput:
        """Agent-controlled adaptive flow with flexible phase transitions.

        Instead of a separate router LLM call, the agent gets ALL tools from
        the first call and its behavior determines the execution path:
        - Text-only response → direct answer (1 LLM call)
        - ``create_todo_list`` called → PLAN phase detected, then ACT → REPORT
        - Domain tools / ``log_finding`` called → ACT phase detected, then REPORT

        The agent controls phase flow via ``set_next_phase()``. It can:
        - Go forward: act → review → report (normal)
        - Go backward: review → act (fix issues), review → plan (revise plan)
        - Skip: act → report (skip review), review → report (override failed review)

        A phase transition counter prevents infinite loops.
        """
        task_id = node.task_id
        execution_metadata.execution_path = EXECUTION_PATH_ADAPTIVE

        # 1. Build entry messages with all tools visible
        entry_tools = registry.get_for_entry()
        entry_messages = context_manager.build_entry_messages(
            config=config, input=input, all_tools=entry_tools,
        )
        tool_schemas = [t.to_llm_schema() for t in entry_tools]

        # 2. Make entry LLM call
        self._budget_tracker.check_budget(node, workflow)
        entry_response = await self._llm.chat_with_tools(
            messages=entry_messages,
            tools=tool_schemas,
            model=config.model,
            model_config=config.model_config,
            stream=self._stream,
            on_token=None,
        )

        # 3. Record usage
        usage = entry_response.usage or TokenUsage()
        cost = self._budget_tracker.record_usage(
            node, workflow, usage, config.model,
        )
        total_usage.input_tokens += usage.input_tokens
        total_usage.output_tokens += usage.output_tokens
        total_usage.total_cost += cost

        # Persist entry LLM call record for debug UI visibility
        _entry_tool_calls = None
        if entry_response.tool_calls:
            _entry_tool_calls = [
                {"name": tc.name, "arguments": tc.arguments}
                for tc in entry_response.tool_calls
            ]
        _entry_llm_record = {
            "phase": "entry",
            "iteration": 0,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "response_content": entry_response.content,
            "tool_calls": _entry_tool_calls,
            "error": None,
            "cumulative_tokens": usage.input_tokens + usage.output_tokens,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self._file_store:
            try:
                self._file_store.append_llm_calls([_entry_llm_record])
            except Exception:
                logger.debug(
                    "Entry LLM call persist failed", exc_info=True,
                )

        # 4. If no tool calls → direct answer
        if not entry_response.has_tool_calls():
            execution_metadata.detected_mode = "direct_answer"
            output = self._build_adaptive_direct_output(
                task_id=task_id,
                config=config,
                content=entry_response.content or "",
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

        # 5. Execute entry tool calls and accumulate messages
        messages = list(entry_messages)
        if entry_response.raw_message:
            messages.append(entry_response.raw_message)

        entry_tool_records: List[Dict[str, Any]] = []
        for tc in entry_response.tool_calls:
            # Check if this is an orchestrator-level tool
            tool_def = registry.get(tc.name)
            if tool_def and tool_def.is_orchestrator_tool and on_orchestrator_tool:
                result = await on_orchestrator_tool(tc)
            else:
                result = await tool_executor.execute(tc)

            messages.append(Message(
                role=MessageRole.TOOL,
                content=result.content,
                tool_call_id=tc.id,
            ))
            tool_record = {
                "name": tc.name,
                "arguments": tc.arguments,
                "result_content": (result.content or "")[:200],
                "success": result.success,
                "phase": "entry",
            }
            entry_tool_records.append(tool_record)

            # Incremental persistence for entry tool calls
            if self._file_store:
                try:
                    self._file_store.append_tool_calls([tool_record])
                    self._file_store.save_memory(memory)
                except Exception:
                    logger.debug(
                        "Incremental persist of entry tool call failed",
                        exc_info=True,
                    )

        # 6. Detect phase from tool calls
        detected_phase = self._detect_entry_phase(entry_response.tool_calls)
        execution_metadata.detected_mode = (
            "deep_work" if detected_phase == Phase.PLAN else "light_work"
        )
        logger.info(
            "Adaptive flow: detected %s mode (entry phase: %s) for agent %s",
            execution_metadata.detected_mode, detected_phase.value,
            config.agent_id,
        )

        # 7. Continue detected phase via PhaseRunner
        phases_hitting_limit: List[str] = []

        phase_result = await phase_runner.run_continuation(
            phase=detected_phase,
            node=node,
            workflow=workflow,
            config=config,
            input=input,
            initial_messages=messages,
            initial_tool_calls=entry_tool_records,
            initial_iteration=1,
            on_orchestrator_tool=on_orchestrator_tool,
        )
        self._accumulate_usage(total_usage, phase_result)
        self._accumulate_tool_calls(execution_metadata, phase_result)
        execution_metadata.phases_completed.append(detected_phase.value)
        execution_metadata.iterations_per_phase[detected_phase.value] = (
            phase_result.iterations
        )
        execution_metadata.phase_outputs[detected_phase.value] = (
            phase_result.content or ""
        )
        if phase_result.hit_iteration_limit:
            phases_hitting_limit.append(detected_phase.value)
        if phase_result.content:
            context_manager.record_phase_summary(detected_phase, phase_result.content)
        self._persist_phase(phase_result, memory)

        # 8. Agent-controlled phase loop
        # Track phase visit counts for loop prevention
        phase_visit_counts: Dict[str, int] = {detected_phase.value: 1}
        transition_count = 1  # entry phase counts as 1
        current_phase = detected_phase

        # Determine next phase from default flow or agent request
        def _default_next(current: Phase) -> Phase:
            """Default forward transition when agent doesn't request a specific phase."""
            if current == Phase.PLAN:
                return Phase.ACT
            if current == Phase.ACT:
                return Phase.REPORT
            if current == Phase.REVIEW:
                return Phase.REPORT
            return Phase.REPORT  # fallback

        def _resolve_next(current: Phase) -> Phase:
            """Resolve next phase: agent request takes priority over default."""
            if memory.requested_next_phase:
                requested = memory.requested_next_phase
                memory.requested_next_phase = None  # consume the request
                phase_map = {
                    "plan": Phase.PLAN,
                    "act": Phase.ACT,
                    "review": Phase.REVIEW,
                    "report": Phase.REPORT,
                }
                return phase_map.get(requested, _default_next(current))
            return _default_next(current)

        next_phase = _resolve_next(current_phase)
        last_review_feedback: Optional[str] = None

        while next_phase != Phase.REPORT and transition_count < self._MAX_PHASE_TRANSITIONS:
            transition_count += 1
            phase_key = next_phase.value
            phase_visit_counts[phase_key] = phase_visit_counts.get(phase_key, 0) + 1

            # Loop prevention: max 3 visits per phase
            if phase_visit_counts[phase_key] > 3:
                logger.warning(
                    "Phase %s visited %d times for agent %s — forcing REPORT",
                    phase_key, phase_visit_counts[phase_key], config.agent_id,
                )
                break

            # Run the phase
            extra_ctx = None
            if next_phase == Phase.ACT and last_review_feedback:
                extra_ctx = last_review_feedback
                last_review_feedback = None

            if next_phase == Phase.REVIEW:
                # Run review with its own cycle logic
                review_phase_obj = self._phase_config.effective_review_phase or Phase.REVIEW
                review_retry_phase = self._phase_config.effective_review_retry_phase
                review_result = await self._run_review_cycle(
                    phase_runner=phase_runner,
                    review_phase=review_phase_obj,
                    review_retry_phase=review_retry_phase,
                    node=node,
                    workflow=workflow,
                    config=config,
                    input=input,
                    context_manager=context_manager,
                    memory=memory,
                    on_orchestrator_tool=on_orchestrator_tool,
                    execution_metadata=execution_metadata,
                    total_usage=total_usage,
                    phases_hitting_limit=phases_hitting_limit,
                )
                current_phase = Phase.REVIEW
                # If review failed and agent didn't request a specific next phase,
                # default to ACT retry with review feedback
                review_passed = self._evaluate_review(review_result, memory)
                if not review_passed and not memory.requested_next_phase:
                    last_review_feedback = self._extract_review_feedback(
                        review_result, memory,
                    )
                    memory.review_checklist = []  # clear for next cycle
            else:
                await self._execute_phase_with_bookkeeping(
                    phase_runner=phase_runner,
                    phase=next_phase,
                    node=node,
                    workflow=workflow,
                    config=config,
                    input=input,
                    context_manager=context_manager,
                    memory=memory,
                    on_orchestrator_tool=on_orchestrator_tool,
                    execution_metadata=execution_metadata,
                    total_usage=total_usage,
                    phases_hitting_limit=phases_hitting_limit,
                    extra_context=extra_ctx,
                )
                current_phase = next_phase

            next_phase = _resolve_next(current_phase)

        if transition_count >= self._MAX_PHASE_TRANSITIONS:
            logger.warning(
                "Max phase transitions (%d) reached for agent %s — forcing REPORT",
                self._MAX_PHASE_TRANSITIONS, config.agent_id,
            )

        # 9. Always run REPORT as the final phase
        report_result = await self._execute_phase_with_bookkeeping(
            phase_runner=phase_runner,
            phase=Phase.REPORT,
            node=node,
            workflow=workflow,
            config=config,
            input=input,
            context_manager=context_manager,
            memory=memory,
            on_orchestrator_tool=on_orchestrator_tool,
            execution_metadata=execution_metadata,
            total_usage=total_usage,
            phases_hitting_limit=phases_hitting_limit,
        )

        # 10. Validate output
        if memory.submitted_report:
            validation = self._output_validator.validate(
                output=memory.submitted_report,
                schema=input.output_schema,
                role=config.role,
                sub_role=config.sub_role,
            )
            if not validation.is_valid:
                for err_msg in validation.errors:
                    logger.warning(
                        "Output validation failed for agent %s: %s",
                        config.agent_id, err_msg,
                    )
                    errors.append(ErrorEntry(
                        source=ErrorSource.SYSTEM,
                        name="agent_runtime",
                        error_type="output_validation",
                        message=err_msg,
                        recoverable=True,
                    ))
            for warn_msg in validation.warnings:
                errors.append(ErrorEntry(
                    source=ErrorSource.SYSTEM,
                    name="agent_runtime",
                    error_type="output_validation_warning",
                    message=warn_msg,
                    recoverable=True,
                ))

        # 11. Build final output
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

    @staticmethod
    def _detect_entry_phase(tool_calls: List[ToolCall]) -> Phase:
        """Detect which phase the agent entered based on its entry tool calls.

        - ``create_todo_list`` or ``update_todo_list`` → PLAN
        - Anything else → ACT
        """
        plan_tools = {"create_todo_list", "update_todo_list"}
        for tc in tool_calls:
            if tc.name in plan_tools:
                return Phase.PLAN
        return Phase.ACT

    def _build_adaptive_direct_output(
        self,
        task_id: str,
        config: AgentConfig,
        content: str,
        total_usage: TokenUsage,
        execution_metadata: ExecutionMetadata,
        errors: List[ErrorEntry],
        start_time: float,
    ) -> AgentOutput:
        """Build output for adaptive-flow direct answer (text-only, no tools)."""
        execution_metadata.total_duration_ms = (time.time() - start_time) * 1000
        execution_metadata.phase_outputs["direct_answer"] = content
        execution_metadata.tools_called = []

        summary = content.strip()
        if len(summary) > 500:
            summary = summary[:500] + "..."

        return AgentOutput(
            task_id=task_id,
            agent_id=config.agent_id,
            role=config.role,
            sub_role=config.sub_role,
            status="completed",
            summary=summary,
            findings={"direct_answer": content, "answer": content},
            errors=errors,
            token_usage=total_usage,
            execution_metadata=execution_metadata,
        )

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

        if input.output_schema:
            if input.direct_answer_schema_policy is not None:
                # Per-role policy set → allow the router LLM to decide
                pass
            elif cfg.force_full_workflow_if_output_schema:
                # No per-role policy → use global gate (backward compatible)
                decision.update({
                    "reason": "Output schema is required for this task.",
                    "policy_reason": "output_schema_requires_full_workflow",
                })
                return decision

        routing_model_config = ModelConfig(
            temperature=config.model_config.temperature,
            top_p=config.model_config.top_p,
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
        policy = input.direct_answer_schema_policy
        use_schema_enforce = (
            policy == "enforce"
            and input.output_schema is not None
        )

        max_tokens = cfg.direct_answer_max_tokens
        if use_schema_enforce:
            max_tokens = max(max_tokens, 2048)

        model_config = ModelConfig(
            temperature=config.model_config.temperature,
            top_p=config.model_config.top_p,
            max_tokens=max_tokens,
        )

        if use_schema_enforce:
            messages = self._build_direct_answer_schema_messages(input)
        else:
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

        if use_schema_enforce:
            return self._parse_schema_enforced_result(raw_content, cfg, input)
        else:
            return self._parse_bypass_result(raw_content, cfg)

    def _parse_bypass_result(
        self, raw_content: str, cfg: SimpleQueryBypassConfig,
    ) -> Dict[str, Any]:
        """Parse direct-answer result in bypass / free-form mode."""
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

    def _parse_schema_enforced_result(
        self,
        raw_content: str,
        cfg: SimpleQueryBypassConfig,
        input: AgentInput,
    ) -> Dict[str, Any]:
        """Parse direct-answer result in schema-enforce mode."""
        parsed = self._parse_json_object(raw_content)

        if not isinstance(parsed, dict):
            return {
                "answer": "",
                "confidence": 0.0,
                "needs_full_workflow": True,
                "reason": "Schema-enforced direct answer was not valid JSON.",
                "parser_status": "invalid_json",
                "raw_response_preview": raw_content[:300],
                "escalate_to_full_workflow": cfg.allow_escalation_to_full_workflow,
            }

        confidence = self._normalize_confidence(parsed.get("confidence", 0.0))
        needs_full_workflow = self._coerce_bool(
            parsed.get("needs_full_workflow"), default=False,
        )
        reason = str(parsed.get("reason", "")).strip() or "Schema-enforced direct answer."

        schema_output = parsed.get("output")
        if not isinstance(schema_output, dict):
            return {
                "answer": "",
                "confidence": 0.0,
                "needs_full_workflow": True,
                "reason": "Schema-enforced output missing 'output' dict.",
                "parser_status": "missing_output",
                "raw_response_preview": raw_content[:300],
                "escalate_to_full_workflow": cfg.allow_escalation_to_full_workflow,
            }

        # Validate against output schema if validator available
        if input.output_schema and self._output_validator:
            validation = self._output_validator.validate(
                schema_output, input.output_schema,
            )
            if not validation.is_valid:
                return {
                    "answer": "",
                    "confidence": 0.0,
                    "needs_full_workflow": True,
                    "reason": (
                        "Schema validation failed: "
                        + "; ".join(validation.errors[:3])
                    ),
                    "parser_status": "schema_validation_failed",
                    "raw_response_preview": raw_content[:300],
                    "escalate_to_full_workflow": cfg.allow_escalation_to_full_workflow,
                }

        # Extract answer/summary from schema output for the summary field
        answer = (
            str(schema_output.get("answer", "")).strip()
            or str(schema_output.get("summary", "")).strip()
        )

        escalate = cfg.allow_escalation_to_full_workflow and (
            needs_full_workflow
            or confidence < cfg.route_confidence_threshold
        )

        return {
            "answer": answer,
            "confidence": confidence,
            "needs_full_workflow": needs_full_workflow,
            "reason": reason,
            "parser_status": "ok",
            "raw_response_preview": raw_content[:300],
            "escalate_to_full_workflow": escalate,
            "schema_output": schema_output,
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
        ]

        # Include domain tool descriptions so router can assess tool relevance
        domain_tools = [t for t in (input.tools or []) if not t.is_framework_tool]
        if domain_tools:
            tool_lines = [f"  {t.to_description_text()}" for t in domain_tools]
            user_payload.append(
                f"Available domain tools ({len(domain_tools)}):\n"
                + "\n".join(tool_lines)
            )

        if input.direct_answer_schema_policy:
            user_payload.append(
                f"Direct-answer schema policy: {input.direct_answer_schema_policy}"
            )
        user_payload.extend([
            "",
            "Respond with JSON only using this schema:",
            '{"mode":"direct_answer|full_workflow","confidence":0.0,'
            '"reason":"...","requires_external_data":false}',
        ])
        return [
            Message(
                role=MessageRole.SYSTEM,
                content=(
                    "You are a routing controller. Decide if a task should be "
                    "answered directly without tools or run through a full "
                    "multi-phase workflow. Consider the available domain tools — "
                    "if any could provide relevant data or functionality for the "
                    "task, prefer full_workflow. Return JSON only."
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

    def _build_direct_answer_schema_messages(self, input: AgentInput) -> List[Message]:
        """Build prompt for schema-enforced direct-answer mode."""
        schema_str = json.dumps(input.output_schema, indent=2, ensure_ascii=False)

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

        context_parts.extend([
            f"\nOutput schema:\n{schema_str}",
            "\nRespond with JSON only. Your 'output' field MUST conform to the schema above.",
            '{"output":{...schema-compliant...},"confidence":0.0,'
            '"needs_full_workflow":false,"reason":"..."}',
        ])
        return [
            Message(
                role=MessageRole.SYSTEM,
                content=(
                    "Answer the user directly without tools. "
                    "Your response 'output' field must conform to the provided JSON schema. "
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

        # Schema-enforced output goes directly into findings (renderable by user-report.js)
        schema_output = direct_result.get("schema_output")
        if isinstance(schema_output, dict):
            findings = schema_output
            summary = (
                str(schema_output.get("answer", "")).strip()
                or str(schema_output.get("summary", "")).strip()
                or answer
                or reason
                or "Direct-answer path completed."
            )
        else:
            summary = answer or reason or "Direct-answer path completed."
            findings = {
                "direct_answer": answer,
                "confidence": confidence,
                "reason": reason,
            }

        if len(summary) > 500:
            summary = summary[:500] + "..."

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
        report_result: Optional[PhaseResult],
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

        # Use submitted report if available, else use last phase content
        report_content = report_result.content if report_result else None
        findings = {}
        if memory.submitted_report:
            findings = memory.submitted_report
        elif report_content:
            findings = {"report_text": report_content}

        # Build summary from findings + report content
        summary = report_content or "Agent completed without producing a text summary."
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
