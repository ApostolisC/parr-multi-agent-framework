"""
Orchestrator for the Agentic Framework.

The orchestrator is the ONLY component that manages agent lifecycles.
It is code-level (not an LLM). It handles:
- Creating agent instances from AgentConfig + AgentInput
- Running the phase lifecycle for each agent (via AgentRuntime)
- Managing the agent tree (parent-child relationships)
- Handling sub-agent spawning (spawn_agent tool calls from agents)
- Managing suspension for wait_for_agents operations
- Enforcing budget limits
- Publishing events to the event bus
- Handling cancellation (immediate, total)
- Maintaining the trace store
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from .agent_runtime import AgentRuntime
from .budget_tracker import (
    BudgetExceededException,
    BudgetTracker,
    ChildBudgetAllocator,
)
from .agent_coordinator import AgentCoordinator
from .output_validator import OutputValidator
from .event_bus import EventBus, EventBridge, InMemoryEventSink
from .event_types import (
    agent_cancelled,
    agent_failed,
    budget_exceeded as budget_exceeded_event,
    # TODO(v2): Import agent_suspended and agent_resumed from event_types
    # when async agent execution with real suspension/resumption is added.
    # Currently unused — v1 runs all agents synchronously.
)
from .persistence import AgentFileStore, WorkflowFileStore
from .trace_store import TraceStore
from .core_types import (
    AdaptiveFlowConfig,
    AgentConfig,
    AgentInput,
    AgentMessage,
    AgentNode,
    AgentOutput,
    AgentStatus,
    BudgetConfig,
    BudgetUsage,
    CostConfig,
    EffortLevel,
    ErrorEntry,
    ErrorSource,
    ExecutionMetadata,
    ModelConfig,
    Phase,
    PhaseConfig,
    SimpleQueryBypassConfig,
    StallDetectionConfig,
    ToolCall,
    ToolDef,
    ToolResult,
    TokenUsage,
    TraceEntry,
    PoliciesConfig,
    SpawnPolicy,
    WorkflowExecution,
    WorkflowStatus,
    generate_id,
    normalise_keys as _normalise_keys,
    strip_heavy_fields_in_place,
    utc_now,
)
from .event_types import (
    batch_progress as batch_progress_event,
    batch_started as batch_started_event,
    spawn_started as spawn_started_event,
    spawn_validation as spawn_validation_event,
)
from .protocols import DomainAdapter, EventSink, TextSummarizer, ToolCallingLLM

logger = logging.getLogger(__name__)


def _classify_child_failure(exc: Exception) -> str:
    """Classify a child agent exception into a human-readable failure type."""
    exc_type = type(exc).__name__
    exc_str = str(exc).lower()

    if isinstance(exc, BudgetExceededException):
        return "budget_exhausted"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if "timeout" in exc_str:
        return "timeout"
    if "rate" in exc_str and "limit" in exc_str:
        return "rate_limit"
    if "429" in exc_str:
        return "rate_limit"
    if "content_filter" in exc_type.lower() or "content filter" in exc_str:
        return "content_filter"
    if "cancel" in exc_str or isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    if "connection" in exc_str or "network" in exc_str:
        return "connection_error"
    return "execution_error"


class Orchestrator:
    """
    Top-level entry point for running agent workflows.

    Usage:
        orchestrator = Orchestrator(llm=llm, event_sink=sink)
        output = await orchestrator.start_workflow(
            task="Analyze privacy risks...",
            role="risk_analyst",
            tools=[...],
            budget=BudgetConfig(max_tokens=100000),
        )
    """

    def __init__(
        self,
        llm: ToolCallingLLM,
        event_sink: Optional[EventSink] = None,
        domain_adapter: Optional[DomainAdapter] = None,
        cost_config: Optional[CostConfig] = None,
        max_review_cycles: int = 2,
        phase_limits: Optional[Dict[Phase, int]] = None,
        default_budget: Optional[BudgetConfig] = None,
        stream: bool = False,
        stall_config: Optional[StallDetectionConfig] = None,
        simple_query_bypass: Optional[SimpleQueryBypassConfig] = None,
        wait_for_agents_timeout: Optional[float] = None,
        persist_dir: Optional[str] = None,
        phase_config: Optional[PhaseConfig] = None,
        child_allocator: Optional[ChildBudgetAllocator] = None,
        output_validator: Optional[OutputValidator] = None,
        coordinator: Optional[AgentCoordinator] = None,
        adaptive_config: Optional[AdaptiveFlowConfig] = None,
        summarizer: Optional[TextSummarizer] = None,
        policies_config: Optional[PoliciesConfig] = None,
    ) -> None:
        self._llm = llm
        self._domain_adapter = domain_adapter
        self._cost_config = cost_config
        self._output_validator = output_validator
        self._coordinator = coordinator or AgentCoordinator()
        self._max_review_cycles = max_review_cycles
        self._phase_limits = phase_limits
        self._phase_config = phase_config
        self._default_budget = default_budget or BudgetConfig()
        self._stream = stream
        self._stall_config = stall_config
        self._simple_query_bypass = simple_query_bypass or SimpleQueryBypassConfig()
        self._wait_for_agents_timeout = wait_for_agents_timeout
        self._persist_dir = persist_dir
        self._adaptive_config = adaptive_config
        self._summarizer = summarizer
        self._policies_config = policies_config or PoliciesConfig()

        # Internal event bus
        self._event_bus = EventBus()

        # Bridge to external event sink
        self._event_sink = event_sink or InMemoryEventSink()
        self._event_bridge = EventBridge(self._event_bus, self._event_sink)

        # Budget tracker
        self._budget_tracker = BudgetTracker(cost_config, child_allocator=child_allocator)

        # Active workflows
        self._workflows: Dict[str, WorkflowExecution] = {}
        self._trace_stores: Dict[str, TraceStore] = {}
        # Background child agent tasks: {task_id: asyncio.Task}
        self._pending_tasks: Dict[str, asyncio.Task] = {}
        # Message read cursors: {task_id: last_read_index}
        self._message_cursors: Dict[str, int] = {}

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def start_workflow(
        self,
        task: str,
        role: str,
        sub_role: Optional[str] = None,
        system_prompt: str = "",
        model: str = "",
        model_config: Optional[Dict[str, Any]] = None,
        tools: Optional[List[ToolDef]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        raw_data: Optional[Dict[str, Any]] = None,
        rag_results: Optional[List[Dict[str, Any]]] = None,
        additional_context: Optional[str] = None,
        budget: Optional[BudgetConfig] = None,
        effort_level: Optional[int] = None,
        workflow_id: Optional[str] = None,
    ) -> AgentOutput:
        """
        Start a new workflow with a root agent.

        This is the primary entry point. It creates a workflow, sets up
        the root agent, and runs it to completion.

        Args:
            task: The task description for the root agent.
            role: Role identifier for the root agent.
            sub_role: Optional sub-role.
            system_prompt: Base system prompt (or resolved from adapter).
            model: LLM model identifier.
            model_config: Temperature, top_p, max_tokens overrides.
            tools: Domain-specific tools available to the agent.
            output_schema: Expected output JSON schema.
            raw_data: Structured domain data.
            rag_results: Pre-fetched document search results.
            additional_context: Free-form context.
            budget: Budget limits for this workflow.

        Returns:
            AgentOutput from the root agent.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        if not task or not task.strip():
            raise ValueError("'task' must be a non-empty string.")
        if not role or not role.strip():
            raise ValueError("'role' must be a non-empty string.")

        workflow_budget = budget or self._default_budget

        # Store workflow-level effort for sub-agent inheritance
        self._workflow_effort_level = effort_level
        logger.info(
            "[PARR] start_workflow: effort_level=%s (stored as _workflow_effort_level)",
            effort_level,
        )

        # Resolve config from adapter if available
        config = self._resolve_agent_config(role, sub_role, system_prompt, model, model_config)

        # Resolve tools from adapter if not provided
        if tools is None and self._domain_adapter:
            tools = self._domain_adapter.get_domain_tools(role, sub_role)
        tools = tools or []

        # Resolve output schema from adapter if not provided
        if output_schema is None and self._domain_adapter:
            output_schema = self._domain_adapter.get_output_schema(role, sub_role)

        # Create workflow (use caller-provided ID if given, to keep
        # persistence and external tracking in sync)
        wf_kwargs: Dict[str, Any] = {"global_budget": workflow_budget}
        if workflow_id:
            wf_kwargs["workflow_id"] = workflow_id
        workflow = WorkflowExecution(**wf_kwargs)
        self._workflows[workflow.workflow_id] = workflow

        # Create trace store for this workflow
        trace_store = TraceStore()
        self._trace_stores[workflow.workflow_id] = trace_store

        # Connect event bridge
        self._event_bridge.connect(workflow.workflow_id)

        # Build available roles description for spawn_agent tool
        roles_desc = self._build_roles_description()

        # Set up optional file-system persistence
        wf_store: Optional[WorkflowFileStore] = None
        root_store: Optional[AgentFileStore] = None
        if self._persist_dir:
            try:
                wf_store = WorkflowFileStore(self._persist_dir, workflow.workflow_id)
                wf_store.save_workflow_info(
                    workflow_id=workflow.workflow_id,
                    status="running",
                    budget={
                        "max_tokens": workflow_budget.max_tokens,
                        "max_cost": workflow_budget.max_cost,
                        "max_duration_ms": workflow_budget.max_duration_ms,
                    },
                    created_at=workflow.created_at.isoformat(),
                )
            except Exception as e:
                logger.error(f"Failed to initialise persistence: {e}", exc_info=True)
                wf_store = None

        try:
            # Create root agent node
            root_node = AgentNode(
                task_id=generate_id(),
                agent_id=config.agent_id,
                config=config,
                budget=workflow_budget,
                depth=0,
            )
            workflow.root_task_id = root_node.task_id
            workflow.agent_tree[root_node.task_id] = root_node

            # Set up root persistence store
            if wf_store:
                try:
                    root_store = wf_store.create_root_store(root_node.task_id)
                    root_store.save_agent_info(
                        task_id=root_node.task_id,
                        agent_id=config.agent_id,
                        role=config.role,
                        sub_role=config.sub_role,
                        task=task,
                        status="running",
                        depth=0,
                        model=config.model,
                        effort_level=effort_level,
                    )
                    wf_store.save_workflow_info(
                        workflow_id=workflow.workflow_id,
                        status="running",
                        root_task_id=root_node.task_id,
                        budget={
                            "max_tokens": workflow_budget.max_tokens,
                            "max_cost": workflow_budget.max_cost,
                            "max_duration_ms": workflow_budget.max_duration_ms,
                        },
                        created_at=workflow.created_at.isoformat(),
                    )
                except Exception as e:
                    logger.error(f"Failed to persist root agent info: {e}", exc_info=True)

            # Add to trace
            trace_store.add_entry(TraceEntry(
                task_id=root_node.task_id,
                agent_id=config.agent_id,
                role=config.role,
                sub_role=config.sub_role,
                task_description=task[:200],
            ))

            # Auto-inject domain context for the root agent
            if self._domain_adapter and hasattr(self._domain_adapter, 'get_initial_context'):
                try:
                    adapter_context = self._domain_adapter.get_initial_context(role, sub_role)
                    if adapter_context:
                        raw_data = {**(raw_data or {}), **adapter_context}
                except Exception as e:
                    logger.warning("Failed to get initial context from adapter: %s", e)

            # Resolve direct-answer schema policy for the root agent
            root_da_policy = None
            if self._domain_adapter and hasattr(
                self._domain_adapter, 'get_direct_answer_schema_policy',
            ):
                try:
                    root_da_policy = self._domain_adapter.get_direct_answer_schema_policy(
                        role, sub_role,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to resolve direct_answer_schema_policy: %s", e,
                    )

            # Build agent input
            agent_input = AgentInput(
                task=task,
                tools=tools,
                output_schema=output_schema,
                raw_data=raw_data,
                rag_results=rag_results,
                additional_context=additional_context,
                budget=workflow_budget,
                trace_snapshot=trace_store.get_snapshot(root_node.task_id),
                effort_level=effort_level,
                direct_answer_schema_policy=root_da_policy,
            )

            # Build report template handler from adapter
            report_template_handler = self._build_report_template_handler()

            # Create runtime
            runtime = AgentRuntime(
                llm=self._llm,
                budget_tracker=self._budget_tracker,
                event_bus=self._event_bus,
                max_review_cycles=self._max_review_cycles,
                phase_limits=self._phase_limits,
                available_roles_description=roles_desc,
                report_template_handler=report_template_handler,
                stream=self._stream,
                stall_config=self._stall_config,
                simple_query_bypass=self._simple_query_bypass,
                budget_config=workflow_budget,
                agent_file_store=root_store,
                effort_level=effort_level,
                output_validator=self._output_validator,
                adaptive_config=self._adaptive_config,
                phase_config=self._phase_config,
                summarizer=self._summarizer,
            )

            # Execute with orchestrator tool handling
            output = await runtime.execute(
                config=config,
                input=agent_input,
                node=root_node,
                workflow=workflow,
                on_orchestrator_tool=lambda tc, te: self._handle_orchestrator_tool(
                    tc, root_node, workflow, trace_store, roles_desc,
                    wf_store, tool_executor=te,
                ),
            )

            _SUCCESS_STATUSES = ("completed", "degraded")

            # Update trace
            trace_store.update_status(
                root_node.task_id,
                AgentStatus.COMPLETED if output.status in _SUCCESS_STATUSES else AgentStatus.FAILED,
                output_summary=output.summary[:200],
            )

            # Update workflow status
            workflow.status = (
                WorkflowStatus.COMPLETED
                if output.status in _SUCCESS_STATUSES
                else WorkflowStatus.FAILED
            )

            # Persist final state
            if wf_store:
                try:
                    wf_store.update_workflow_status(workflow.status.value)
                except Exception as e:
                    logger.error(f"Failed to persist workflow status: {e}")
                    output.errors.append(ErrorEntry(
                        source=ErrorSource.SYSTEM,
                        name="orchestrator",
                        error_type="persist_status_failed",
                        message=f"Failed to persist workflow status: {e}",
                        recoverable=True,
                    ))

            # Persist output if adapter is available
            if self._domain_adapter and output.status in _SUCCESS_STATUSES:
                try:
                    self._domain_adapter.persist_output(workflow.workflow_id, output)
                except Exception as e:
                    logger.error(f"Failed to persist workflow output: {e}")
                    output.errors.append(ErrorEntry(
                        source=ErrorSource.SYSTEM,
                        name="orchestrator",
                        error_type="persist_output_failed",
                        message=f"Failed to persist workflow output: {e}",
                        recoverable=True,
                    ))

            return output

        except Exception as e:
            logger.error(f"Workflow {workflow.workflow_id} failed: {e}", exc_info=True)
            workflow.status = WorkflowStatus.FAILED
            if wf_store:
                try:
                    wf_store.update_workflow_status("failed")
                except Exception:
                    pass
            raise

        finally:
            # Cancel and clean up any still-running child tasks
            await self._cancel_pending_tasks()
            self._event_bridge.disconnect_all()

    async def cancel_workflow(self, workflow_id: str) -> None:
        """
        Cancel a running workflow. Immediate and total.

        Walks the entire agent tree and cancels everything.
        No agent continues after cancellation.
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            logger.warning(f"Cannot cancel unknown workflow: {workflow_id}")
            return

        workflow.status = WorkflowStatus.CANCELLED

        # Cancel pending background tasks first
        await self._cancel_pending_tasks()

        for task_id, node in workflow.agent_tree.items():
            if node.status in (AgentStatus.RUNNING, AgentStatus.SUSPENDED):
                node.status = AgentStatus.CANCELLED
                await self._event_bus.publish(agent_cancelled(
                    workflow_id=workflow_id,
                    task_id=task_id,
                    agent_id=node.agent_id,
                ))

        # Update trace
        trace_store = self._trace_stores.get(workflow_id)
        if trace_store:
            for entry in trace_store.get_full_trace():
                if entry.status == AgentStatus.RUNNING:
                    trace_store.update_status(entry.task_id, AgentStatus.CANCELLED)

        logger.info(f"Workflow {workflow_id} cancelled")

    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowExecution]:
        """Get the current state of a workflow."""
        return self._workflows.get(workflow_id)

    def get_trace(self, workflow_id: str) -> Optional[TraceStore]:
        """Get the trace store for a workflow."""
        return self._trace_stores.get(workflow_id)

    # -----------------------------------------------------------------------
    # Orchestrator tool handling
    # -----------------------------------------------------------------------

    async def _handle_orchestrator_tool(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
        *,
        tool_executor: Optional[Any] = None,
    ) -> ToolResult:
        """
        Handle orchestrator-level tool calls (spawn_agent, wait_for_agents, etc.).

        ``tool_executor`` is the per-agent ToolExecutor, plumbed through
        from the PhaseRunner callback so that ``batch_operations`` can
        dispatch arbitrary registered tools (not just orchestrator tools).
        """
        handlers = {
            "batch_operations": self._handle_batch_operations,
            "spawn_agent": self._handle_spawn_agent,
            "wait_for_agents": self._handle_wait_for_agents,
            "get_agent_result": self._handle_get_agent_result,
            "get_agent_results_all": self._handle_get_agent_results_all,
            "get_agent_phase_output": self._handle_get_agent_phase_output,
            "send_message": self._handle_send_message,
            "read_messages": self._handle_read_messages,
            "set_shared_state": self._handle_set_shared_state,
            "get_shared_state": self._handle_get_shared_state,
        }

        handler = handlers.get(tool_call.name)
        if not handler:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Unknown orchestrator tool: {tool_call.name}",
            )

        try:
            result = await handler(
                tool_call, parent_node, workflow, trace_store, roles_description,
                wf_store, tool_executor=tool_executor,
            )

            # --- Per-tool stripping for orchestrator tools ---
            if result.success:
                from .tool_registry import ToolRegistry
                registry = tool_executor._registry if tool_executor else None
                if registry:
                    td = registry.get(tool_call.name)
                    if td:
                        effective_strip = tool_call.arguments.get(
                            "strip_input_after_dispatch",
                            td.strip_input_after_dispatch,
                        )
                        if effective_strip and td.heavy_input_fields:
                            strip_heavy_fields_in_place(
                                tool_call.arguments, td.heavy_input_fields,
                            )
            return result

        except BudgetExceededException as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Budget exceeded while executing {tool_call.name}: {e}",
            )
        except Exception as e:
            logger.error(
                f"Orchestrator tool {tool_call.name} failed: {e}",
                exc_info=True,
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Orchestrator tool {tool_call.name} failed: {e}",
            )

    # -----------------------------------------------------------------------
    # batch_operations — generic tool dispatcher
    # -----------------------------------------------------------------------

    async def _handle_batch_operations(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
        *,
        tool_executor: Optional[Any] = None,
    ) -> ToolResult:
        """Handle batch_operations — a generic tool dispatcher.

        Dispatches each ``op`` in the ``operations`` array to the
        appropriate handler: orchestrator tools route through this
        class's handler table, regular tools route through the
        ToolExecutor.

        When any ``spawn_agent`` ops succeed, the method automatically
        calls ``wait_for_agents`` on the collected task_ids before
        returning, replacing each spawn op result with the full
        sub-agent report. This collapses the spawn→wait→synthesize
        pattern from 3 turns to 2.
        """
        operations = tool_call.arguments.get("operations", [])
        if not isinstance(operations, list) or not operations:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error="batch_operations requires a non-empty 'operations' array.",
            )

        parallel = bool(tool_call.arguments.get("parallel", False))

        # Emit batch_started so the UI can show the card immediately
        await self._event_bus.publish(batch_started_event(
            workflow_id=workflow.workflow_id,
            task_id=parent_node.task_id,
            agent_id=parent_node.agent_id,
            batch_tool_call_id=tool_call.id,
            operations=operations,
        ))
        registry = tool_executor._registry if tool_executor else None

        # -- per-op dispatch closure --
        async def _dispatch_op(
            i: int, op: Dict[str, Any],
        ) -> Dict[str, Any]:
            op_type = str(op.get("op", "")).strip()
            if not op_type:
                return {
                    "op_index": i, "tool": "", "success": False,
                    "error": "Missing 'op' field.",
                }

            # Guard: nested batch_operations
            if op_type == "batch_operations":
                return {
                    "op_index": i, "tool": op_type, "success": False,
                    "error": "Nested batch_operations are not supported.",
                }

            # Guard: explicit wait_for_agents
            if op_type == "wait_for_agents":
                return {
                    "op_index": i, "tool": op_type, "success": False,
                    "error": (
                        "wait_for_agents cannot be batched explicitly; "
                        "batch_operations auto-waits for any spawn_agent "
                        "ops it dispatches."
                    ),
                }

            # Look up the tool
            tool_def = registry.get(op_type) if registry else None
            if tool_def is None:
                return {
                    "op_index": i, "tool": op_type, "success": False,
                    "error": f"Unknown tool '{op_type}'. Check the tool name and try again.",
                }

            # Phase availability check for inner ops
            if tool_def.phase_availability and parent_node.current_phase:
                if parent_node.current_phase not in tool_def.phase_availability:
                    avail = ", ".join(
                        p.name for p in tool_def.phase_availability
                    )
                    return {
                        "op_index": i, "tool": op_type, "success": False,
                        "error": (
                            f"Tool '{op_type}' is not available in the "
                            f"{parent_node.current_phase.name} phase. "
                            f"Available in: {avail}."
                        ),
                    }

            # Build synthetic ToolCall (exclude meta-keys from arguments)
            _META_KEYS = {"op", "strip_input_after_dispatch"}
            synth_args = {k: v for k, v in op.items() if k not in _META_KEYS}

            # Normalise common camelCase→snake_case from LLMs that emit
            # JavaScript-style keys (e.g. taskDescription → task_description).
            synth_args = _normalise_keys(synth_args)

            synth = ToolCall(
                id=f"{tool_call.id}::{i}",
                name=op_type,
                arguments=synth_args,
            )

            try:
                if tool_def.is_orchestrator_tool:
                    # Route through orchestrator handler table (spawn_agent, etc.)
                    result = await self._handle_orchestrator_tool(
                        synth, parent_node, workflow, trace_store,
                        roles_description, wf_store,
                        tool_executor=tool_executor,
                    )
                else:
                    # Route through ToolExecutor (memory ops, domain tools, etc.)
                    if tool_executor is None:
                        return {
                            "op_index": i, "tool": op_type, "success": False,
                            "error": (
                                f"Cannot dispatch '{op_type}': no tool executor "
                                f"available in this context."
                            ),
                        }
                    result = await tool_executor.execute(synth)
            except Exception as exc:
                logger.error(
                    f"batch_operations op[{i}] {op_type} raised: {exc}",
                    exc_info=True,
                )
                return {
                    "op_index": i, "tool": op_type, "success": False,
                    "error": str(exc),
                }

            # Parse result content for structured data
            result_payload: Any = result.content
            if isinstance(result_payload, str):
                try:
                    result_payload = json.loads(result_payload)
                except (json.JSONDecodeError, TypeError):
                    pass  # keep as string

            op_result: Dict[str, Any] = {
                "op_index": i,
                "tool": op_type,
                "success": result.success,
            }
            if result.success:
                op_result["result"] = result_payload
            else:
                op_result["error"] = result.error or "Unknown error"

            # -- per-op stripping --
            effective_strip = op.get(
                "strip_input_after_dispatch",
                tool_def.strip_input_after_dispatch,
            )
            if effective_strip and result.success and tool_def.heavy_input_fields:
                strip_heavy_fields_in_place(op, tool_def.heavy_input_fields)

            return op_result

        # -- dispatch all ops (sequential or parallel) --
        total_ops = len(operations)
        if parallel:
            op_results: List[Dict[str, Any]] = await asyncio.gather(
                *(_dispatch_op(i, op) for i, op in enumerate(operations, 1)),
            )
            # Emit progress for all completed ops at once (parallel)
            for idx, op_result in enumerate(op_results):
                await self._event_bus.publish(batch_progress_event(
                    workflow_id=workflow.workflow_id,
                    task_id=parent_node.task_id,
                    agent_id=parent_node.agent_id,
                    batch_tool_call_id=tool_call.id,
                    total_ops=total_ops,
                    completed_ops=idx + 1,
                    current_op=op_result,
                ))
        else:
            op_results = []
            for i, op in enumerate(operations, 1):
                op_result = await _dispatch_op(i, op)
                op_results.append(op_result)
                # Emit progress after each op (sequential)
                await self._event_bus.publish(batch_progress_event(
                    workflow_id=workflow.workflow_id,
                    task_id=parent_node.task_id,
                    agent_id=parent_node.agent_id,
                    batch_tool_call_id=tool_call.id,
                    total_ops=total_ops,
                    completed_ops=len(op_results),
                    current_op=op_result,
                ))

        # -- auto-wait for spawned agents (D3) --
        spawned_task_ids: List[str] = []
        spawn_op_indices: List[int] = []
        for idx, op_result in enumerate(op_results):
            if op_result.get("tool") == "spawn_agent" and op_result.get("success"):
                result_payload = op_result.get("result", {})
                if isinstance(result_payload, str):
                    try:
                        result_payload = json.loads(result_payload)
                    except (json.JSONDecodeError, TypeError):
                        result_payload = {}
                task_id = result_payload.get("task_id") if isinstance(result_payload, dict) else None
                if task_id:
                    spawned_task_ids.append(task_id)
                    spawn_op_indices.append(idx)

        if spawned_task_ids:
            wait_call = ToolCall(
                id=f"{tool_call.id}::auto_wait",
                name="wait_for_agents",
                arguments={"task_ids": spawned_task_ids, "detail_level": "full"},
            )
            wait_result = await self._handle_wait_for_agents(
                wait_call, parent_node, workflow, trace_store,
                roles_description, wf_store,
            )
            if wait_result.success:
                try:
                    agent_outputs = json.loads(wait_result.content)
                except (json.JSONDecodeError, TypeError):
                    agent_outputs = {}
                # agent_outputs is keyed by task_id
                for i, idx in enumerate(spawn_op_indices):
                    tid = spawned_task_ids[i]
                    if isinstance(agent_outputs, dict) and tid in agent_outputs:
                        op_results[idx]["result"] = agent_outputs[tid]
                    op_results[idx]["auto_waited"] = True
            else:
                for idx in spawn_op_indices:
                    op_results[idx]["wait_error"] = wait_result.error
                    op_results[idx]["auto_waited"] = False

        # Build final response
        any_success = any(r.get("success") for r in op_results)
        return ToolResult(
            tool_call_id=tool_call.id,
            success=any_success,
            content=json.dumps(op_results, indent=2, default=str),
            error=None if any_success else "All operations failed.",
        )

    # -----------------------------------------------------------------------
    # Agent lifecycle tools
    # -----------------------------------------------------------------------

    async def _handle_spawn_agent(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
        **_kw: Any,
    ) -> ToolResult:
        """Handle spawn_agent tool call.

        Launches the child agent as a background ``asyncio.Task`` and returns
        immediately with the child's ``task_id``.  The parent can continue
        working (or spawn more children) and later call ``wait_for_agents``
        to collect results.  By default blocks until the child finishes
        and returns the full output inline (``blocking=true``).
        """
        args = _normalise_keys(tool_call.arguments)
        role = args.get("role", "")
        sub_role = args.get("sub_role")
        task_description = args.get("task_description", "")

        # Validate required fields
        if not role or not role.strip():
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error="spawn_agent requires a non-empty 'role'.",
            )
        if not task_description or not task_description.strip():
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error="spawn_agent requires a non-empty 'task_description'.",
            )

        # Validate depth limit
        if parent_node.depth + 1 >= parent_node.budget.max_agent_depth:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Maximum agent nesting depth ({parent_node.budget.max_agent_depth}) "
                       f"exceeded. Current depth: {parent_node.depth}.",
            )

        # Validate total children count
        if len(parent_node.children) >= parent_node.budget.max_sub_agents_total:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Maximum total sub-agents ({parent_node.budget.max_sub_agents_total}) "
                       f"reached. You have already spawned {len(parent_node.children)} "
                       f"sub-agent(s). Consolidate their results and proceed.",
            )

        # Validate parallel count
        running_children = sum(
            1 for tid in parent_node.children
            if tid in workflow.agent_tree
            and workflow.agent_tree[tid].status == AgentStatus.RUNNING
        )
        if running_children >= parent_node.budget.max_parallel_agents:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Maximum parallel agents ({parent_node.budget.max_parallel_agents}) "
                       f"reached. Wait for existing agents to complete first.",
            )

        # Same-role spawn policy check
        description = args.get("description", "")
        parent_role = (parent_node.config.role, parent_node.config.sub_role)
        child_role_tuple = (role, sub_role)
        same_role = parent_role == child_role_tuple
        if same_role:
            policy = self._policies_config.same_role_spawn_policy
            if policy == SpawnPolicy.DENY:
                await self._emit_spawn_validation_event(
                    workflow, parent_node, role, sub_role,
                    task_description, description,
                    decision="denied",
                    reason="Same-role spawning is blocked by policy.",
                    policy_mode="deny",
                )
                return ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    content="",
                    error=(
                        "Cannot spawn an agent with the same role/sub_role "
                        "as yourself. Perform the task directly instead of "
                        "delegating to a clone."
                    ),
                )
            elif policy == SpawnPolicy.CONSULT:
                decision, reasoning = await self._consult_on_spawn(
                    parent_node, workflow, role, sub_role,
                    task_description, trace_store,
                )
                await self._emit_spawn_validation_event(
                    workflow, parent_node, role, sub_role,
                    task_description, description,
                    decision=decision, reason=reasoning,
                    policy_mode="consult",
                )
                if decision == "denied":
                    return ToolResult(
                        tool_call_id=tool_call.id,
                        success=False,
                        content="",
                        error=(
                            f"Spawn denied by consultant: {reasoning} "
                            f"Perform the task directly."
                        ),
                    )
            # SpawnPolicy.WARN — fall through; warning injected in ToolResult below

        # Resolve child config
        child_config = self._resolve_agent_config(role, sub_role)
        if not child_config.system_prompt and not self._domain_adapter:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Role '{role}' with sub_role '{sub_role}' not found. "
                       f"No domain adapter configured to resolve roles.",
            )

        # Calculate child budget
        child_budget = self._budget_tracker.calculate_child_budget(parent_node)

        # Resolve child tools
        child_tools: List[ToolDef] = []
        if self._domain_adapter:
            child_tools = self._domain_adapter.get_domain_tools(role, sub_role)

        # Resolve child output schema
        child_schema = None
        if self._domain_adapter:
            child_schema = self._domain_adapter.get_output_schema(role, sub_role)

        # Resolve child direct-answer schema policy
        child_da_policy = None
        if self._domain_adapter and hasattr(self._domain_adapter, 'get_direct_answer_schema_policy'):
            child_da_policy = self._domain_adapter.get_direct_answer_schema_policy(role, sub_role)

        # Create child node
        child_node = AgentNode(
            task_id=generate_id(),
            agent_id=child_config.agent_id,
            parent_task_id=parent_node.task_id,
            config=child_config,
            budget=child_budget,
            depth=parent_node.depth + 1,
            description=description,
        )
        workflow.agent_tree[child_node.task_id] = child_node
        parent_node.children.append(child_node.task_id)

        # Add to trace
        trace_store.add_entry(TraceEntry(
            task_id=child_node.task_id,
            agent_id=child_config.agent_id,
            role=role,
            sub_role=sub_role,
            parent_task_id=parent_node.task_id,
            task_description=task_description[:200],
        ))
        trace_store.add_child(parent_node.task_id, child_node.task_id)

        # Effort level resolution order:
        # 1. LLM's explicit effort_level in spawn_agent args (capped at parent)
        # 2. Domain adapter's default_effort for the child's role/sub_role
        # 3. Inherit parent's effort level (except Level 0 is never inherited)
        parent_effort = getattr(self, "_workflow_effort_level", None)
        child_effort_raw = args.get("effort_level")
        if child_effort_raw is not None:
            child_effort = int(child_effort_raw)
            # Cap: children cannot exceed the workflow's effort level
            if parent_effort is not None and child_effort > parent_effort:
                child_effort = parent_effort
        else:
            # Check domain adapter for role-specific default
            adapter_default = None
            if self._domain_adapter and hasattr(self._domain_adapter, "get_default_effort"):
                try:
                    adapter_default = self._domain_adapter.get_default_effort(role, sub_role)
                except Exception:
                    pass
            if adapter_default is not None:
                child_effort = int(adapter_default)
                if parent_effort is not None and child_effort > parent_effort:
                    child_effort = parent_effort
            elif parent_effort is not None and parent_effort != EffortLevel.MINIMAL:
                # Level 0 is internal-only — never inherited automatically.
                child_effort = parent_effort
            else:
                child_effort = None

        logger.info(
            "[PARR] _handle_spawn_agent: child_effort=%s (raw=%s, parent_effort=%s, capped=%s) for role=%s/%s",
            child_effort, child_effort_raw, parent_effort,
            child_effort_raw is not None and int(child_effort_raw) != child_effort,
            args.get("role"), args.get("sub_role"),
        )

        # review_mode lets the parent agent control how strictly the child
        # is reviewed: "thorough" (default), "lenient", or "skip".
        # When effort_level is provided it takes precedence over review_mode.
        child_review_mode = args.get("review_mode", "thorough")

        # Auto-inject domain context for spawned agents
        child_raw_data = args.get("raw_data") or {}
        if self._domain_adapter and hasattr(self._domain_adapter, 'get_initial_context'):
            try:
                adapter_context = self._domain_adapter.get_initial_context(role, sub_role)
                if adapter_context:
                    child_raw_data = {**child_raw_data, **adapter_context}
            except Exception as e:
                logger.warning("Failed to get initial context for child agent: %s", e)

        # Build child input
        child_input = AgentInput(
            task=task_description,
            tools=child_tools,
            output_schema=child_schema,
            raw_data=child_raw_data or None,
            additional_context=args.get("additional_context"),
            budget=child_budget,
            trace_snapshot=trace_store.get_snapshot(child_node.task_id),
            effort_level=child_effort,
            direct_answer_schema_policy=child_da_policy,
        )

        # Sub-agents get limited review cycles to avoid costly retry loops.
        # Configurable via budget.max_child_review_cycles; defaults to 1.
        max_child_reviews = parent_node.budget.max_child_review_cycles
        if max_child_reviews is None:
            max_child_reviews = 1
        child_review_cycles = min(self._max_review_cycles, max_child_reviews)

        # Set up child file store (nested under parent)
        child_file_store: Optional[AgentFileStore] = None
        if wf_store:
            try:
                child_file_store = wf_store.create_child_store(
                    parent_task_id=parent_node.task_id,
                    child_task_id=child_node.task_id,
                    role=role,
                )
                child_file_store.save_agent_info(
                    task_id=child_node.task_id,
                    agent_id=child_config.agent_id,
                    role=role,
                    sub_role=sub_role,
                    task=task_description,
                    status="running",
                    depth=child_node.depth,
                    parent_task_id=parent_node.task_id,
                    model=child_config.model,
                    effort_level=child_effort,
                )
                # Register child in parent's sub_agents.json
                parent_store = wf_store.get_store(parent_node.task_id)
                if parent_store:
                    parent_store.register_child(
                        task_id=child_node.task_id,
                        agent_id=child_config.agent_id,
                        role=role,
                        sub_role=sub_role,
                        task_description=task_description,
                    )
            except Exception as e:
                logger.error(f"Failed to persist child agent info: {e}", exc_info=True)

        # Inherit phase config for child, adjusting review cycles
        child_phase_config = None
        if self._phase_config is not None:
            child_phase_config = PhaseConfig(
                phases=self._phase_config.phases,
                phase_limits=self._phase_config.phase_limits,
                phase_prompts=self._phase_config.phase_prompts,
                max_review_cycles=min(
                    child_review_cycles, self._phase_config.max_review_cycles,
                ),
                review_phase=self._phase_config.review_phase,
                review_retry_phase=self._phase_config.review_retry_phase,
            )

        child_runtime = AgentRuntime(
            llm=self._llm,
            budget_tracker=self._budget_tracker,
            event_bus=self._event_bus,
            max_review_cycles=child_review_cycles,
            phase_limits=self._phase_limits,
            available_roles_description=roles_description,
            report_template_handler=self._build_report_template_handler(),
            stream=self._stream,
            stall_config=self._stall_config,
            simple_query_bypass=self._simple_query_bypass,
            budget_config=child_budget,
            agent_file_store=child_file_store,
            review_mode=child_review_mode,
            effort_level=child_effort,
            output_validator=self._output_validator,
            adaptive_config=self._adaptive_config,
            phase_config=child_phase_config if self._phase_config else None,
            summarizer=self._summarizer,
        )

        # Launch child execution as a background task so the parent can
        # continue working (or spawn more agents) concurrently.
        async def _run_child() -> AgentOutput:
            try:
                output = await child_runtime.execute(
                    config=child_config,
                    input=child_input,
                    node=child_node,
                    workflow=workflow,
                    on_orchestrator_tool=lambda tc, te: self._handle_orchestrator_tool(
                        tc, child_node, workflow, trace_store, roles_description,
                        wf_store, tool_executor=te,
                    ),
                )
                _child_success = output.status in ("completed", "degraded")
                trace_store.update_status(
                    child_node.task_id,
                    AgentStatus.COMPLETED if _child_success else AgentStatus.FAILED,
                    output_summary=output.summary[:200],
                )
                # Update child status in parent's sub_agents.json
                if wf_store:
                    try:
                        parent_store = wf_store.get_store(parent_node.task_id)
                        if parent_store:
                            parent_store.update_child_status(
                                child_node.task_id,
                                "completed" if _child_success else "failed",
                            )
                    except Exception:
                        pass
                return output
            except Exception as e:
                logger.error(
                    f"Child agent {child_node.agent_id} failed: {e}",
                    exc_info=True,
                )
                # Classify the failure for the parent agent
                failure_type = _classify_child_failure(e)
                # Build a minimal AgentOutput so parent gets structured info
                failed_output = AgentOutput(
                    task_id=child_node.task_id,
                    agent_id=child_config.agent_id,
                    role=child_config.role,
                    sub_role=child_config.sub_role,
                    status="failed",
                    summary=f"Agent failed: {failure_type} — {str(e)[:200]}",
                    findings={},
                    errors=[ErrorEntry(
                        source=ErrorSource.SYSTEM,
                        name="child_agent",
                        error_type=failure_type,
                        message=str(e)[:500],
                        recoverable=failure_type in (
                            "timeout", "rate_limit", "budget_exhausted",
                        ),
                    )],
                    token_usage=TokenUsage(),
                    execution_metadata=ExecutionMetadata(
                        execution_path="adaptive",
                    ),
                )
                child_node.status = AgentStatus.FAILED
                child_node.result = failed_output
                trace_store.update_status(
                    child_node.task_id,
                    AgentStatus.FAILED,
                    output_summary=str(e)[:200],
                )
                if wf_store:
                    try:
                        parent_store = wf_store.get_store(parent_node.task_id)
                        if parent_store:
                            parent_store.update_child_status(
                                child_node.task_id, "failed",
                            )
                    except Exception:
                        pass
                return failed_output

        task = asyncio.create_task(
            _run_child(),
            name=f"agent-{child_node.task_id[:8]}",
        )
        self._pending_tasks[child_node.task_id] = task

        # Emit spawn_started so the UI can show the agent immediately
        # (before the blocking await delays the tool_executed event).
        blocking = args.get("blocking", True)
        await self._event_bus.publish(spawn_started_event(
            workflow_id=workflow.workflow_id,
            task_id=parent_node.task_id,
            agent_id=parent_node.agent_id,
            child_task_id=child_node.task_id,
            child_role=role,
            child_sub_role=sub_role,
            child_description=description,
            blocking=blocking,
        ))

        # Same-role warning (emitted regardless of blocking mode)
        warn_suffix = ""
        if same_role and self._policies_config.same_role_spawn_policy == SpawnPolicy.WARN:
            warn_suffix = (
                "\n\n⚠ SAME-ROLE SPAWN WARNING: You spawned an agent with "
                "the same role as yourself. Next time, consider doing the "
                "work directly unless you have a specific reason to delegate "
                "(e.g., parallel sub-tasks or context window management)."
            )
            await self._emit_spawn_validation_event(
                workflow, parent_node, role, sub_role,
                task_description, description,
                decision="warned",
                reason="Same-role spawn proceeded with warning.",
                policy_mode="warn",
            )

        # Blocking mode (default): await child completion and return full output
        if blocking:
            try:
                await task
            except Exception as exc:
                logger.error(
                    f"Blocking spawn_agent child {child_node.task_id[:8]} "
                    f"raised: {exc}",
                    exc_info=True,
                )
                # _run_child stores a failed AgentOutput on child_node,
                # so we fall through to _collect_child_result below.

            # Collect and return the child result (same path as wait_for_agents)
            result_dict = self._collect_child_result(
                child_node.task_id, workflow,
            )
            child_status = result_dict.get("status", "failed")
            child_succeeded = child_status in ("completed", "degraded")
            if warn_suffix:
                result_dict["warn"] = warn_suffix

            return ToolResult(
                tool_call_id=tool_call.id,
                success=child_succeeded,
                content=json.dumps(result_dict, default=str),
                error=None if child_succeeded else (
                    f"Spawned agent {role}/{sub_role} finished with "
                    f"status '{child_status}'."
                ),
            )

        # Non-blocking mode: return immediately with task_id
        spawn_msg = (
            f"Agent spawned and running in the background. "
            f"Use wait_for_agents with task_id '{child_node.task_id}' "
            f"to collect results when ready.{warn_suffix}"
        )
        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            content=json.dumps({
                "task_id": child_node.task_id,
                "agent_id": child_config.agent_id,
                "role": role,
                "sub_role": sub_role,
                "description": description,
                "status": "spawned",
                "message": spawn_msg,
            }),
        )

    # -----------------------------------------------------------------------
    # Spawn policy helpers
    # -----------------------------------------------------------------------

    async def _emit_spawn_validation_event(
        self,
        workflow: WorkflowExecution,
        parent_node: AgentNode,
        child_role: str,
        child_sub_role: Optional[str],
        child_task: str,
        child_description: str,
        decision: str,
        reason: str,
        policy_mode: str,
    ) -> None:
        """Emit a spawn_validation event for UI observability."""
        event = spawn_validation_event(
            workflow_id=workflow.workflow_id,
            task_id=parent_node.task_id,
            agent_id=parent_node.agent_id,
            child_role=child_role,
            child_sub_role=child_sub_role,
            child_task=child_task,
            policy_mode=policy_mode,
            decision=decision,
            reason=reason,
        )
        await self._event_bus.publish(event)

    async def _consult_on_spawn(
        self,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        child_role: str,
        child_sub_role: Optional[str],
        child_task_description: str,
        trace_store: TraceStore,
    ) -> tuple:
        """Run a Level 0 consultant to validate a same-role spawn.

        Returns (decision, reasoning) where decision is "approved" or "denied".
        """
        from pathlib import Path

        # Build lean context packet
        # Approximate tools called from trace children count + budget usage
        entry = trace_store.get_entry(parent_node.task_id)
        tools_called = len(entry.children) if entry else 0
        budget_remaining_pct = 100.0
        if parent_node.budget.max_tokens and parent_node.budget_consumed.tokens:
            budget_remaining_pct = max(
                0.0,
                (1.0 - parent_node.budget_consumed.tokens / parent_node.budget.max_tokens) * 100,
            )

        # Load prompts from domain_agents/
        prompts_dir = Path(__file__).parent / "domain_agents" / "consultant"
        try:
            base_prompt = (prompts_dir / "base.md").read_text(encoding="utf-8")
            case_template = (prompts_dir / "cases" / "same_role_spawn.md").read_text(encoding="utf-8")
        except FileNotFoundError as e:
            logger.warning(f"Consultant prompt not found: {e}; defaulting to approved")
            return ("approved", f"Consultant prompts missing ({e}); defaulting to approved.")

        user_message = case_template.format(
            parent_role=parent_node.config.role or "(none)",
            parent_sub_role=parent_node.config.sub_role or "(none)",
            parent_task=(getattr(parent_node.config, 'task_description', '') or '')[:300],
            parent_phase=parent_node.current_phase.name if parent_node.current_phase else "unknown",
            parent_iteration=getattr(parent_node, '_iteration_count', '?'),
            tools_called_count=tools_called,
            budget_remaining_pct=f"{budget_remaining_pct:.0f}%",
            child_role=child_role,
            child_sub_role=child_sub_role or "(none)",
            child_task=child_task_description[:300],
        )

        # Single LLM call — no framework overhead
        model = self._policies_config.consultant_model or parent_node.config.model or "default"
        try:
            from .core_types import Message, MessageRole, ModelConfig as MC
            response = await self._llm.chat_with_tools(
                messages=[
                    Message(role=MessageRole.SYSTEM, content=base_prompt),
                    Message(role=MessageRole.USER, content=user_message),
                ],
                tools=[],
                model=model,
                model_config=MC(
                    temperature=self._policies_config.consultant_temperature,
                    max_tokens=self._policies_config.consultant_max_tokens,
                ),
            )
            text = (response.content or "").strip()
            if text.lower().startswith("approved"):
                return ("approved", text)
            else:
                return ("denied", text)
        except Exception as e:
            logger.warning(f"Consultant call failed: {e}; defaulting to approved")
            return ("approved", f"Consultant unavailable ({e}); defaulting to approved.")

    # -----------------------------------------------------------------------
    # wait_for_agents
    # -----------------------------------------------------------------------

    async def _handle_wait_for_agents(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
        **_kw: Any,
    ) -> ToolResult:
        """Handle wait_for_agents tool call.

        Awaits all requested child tasks concurrently via ``asyncio.gather``.
        Tasks that have already finished are collected immediately.
        A configurable timeout prevents indefinite blocking.

        Honors a ``detail_level`` argument (default ``"summary"``) that
        controls how verbose each per-agent payload is. Compact summaries
        slash synthesis cost when many children are spawned in parallel.
        """
        task_ids = tool_call.arguments.get("task_ids", [])
        detail_level = str(tool_call.arguments.get("detail_level", "summary")).lower()
        if detail_level not in ("summary", "full"):
            detail_level = "summary"

        if not task_ids:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error="wait_for_agents requires a non-empty 'task_ids' list.",
            )

        # Validate all task IDs are children
        for tid in task_ids:
            if tid not in parent_node.children:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    content="",
                    error=f"Task {tid} is not a child of this agent.",
                )

        # Await any tasks that are still pending, with optional timeout
        still_running = [
            self._pending_tasks[tid]
            for tid in task_ids
            if tid in self._pending_tasks and not self._pending_tasks[tid].done()
        ]
        timed_out = False
        if still_running:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*still_running, return_exceptions=True),
                    timeout=self._wait_for_agents_timeout,
                )
            except asyncio.TimeoutError:
                timed_out = True
                logger.warning(
                    f"wait_for_agents timed out after "
                    f"{self._wait_for_agents_timeout}s for tasks {task_ids}"
                )

        # Collect results for all requested children
        raw_results = {
            tid: self._collect_child_result(tid, workflow)
            for tid in task_ids
        }
        if detail_level == "summary":
            results = {
                tid: self._summarize_child_result(r)
                for tid, r in raw_results.items()
            }
        else:
            results = raw_results

        if timed_out:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content=json.dumps(results, indent=2, default=str),
                error=(
                    f"wait_for_agents timed out after "
                    f"{self._wait_for_agents_timeout}s. "
                    f"Some agents may still be running. "
                    f"Partial results are included in the content."
                ),
            )

        # Check for failed/degraded children and inject recovery guidance
        failed_entries = [
            (tid, r) for tid, r in results.items()
            if r.get("status") in ("failed", "degraded", "cancelled")
        ]
        content = json.dumps(results, indent=2, default=str)
        if failed_entries:
            recovery_pct = parent_node.budget.parent_recovery_budget_pct
            failure_details = []
            for tid, r in failed_entries:
                ftype = r.get("failure_type", "unknown")
                role = r.get("role", "unknown")
                err_msg = ""
                if r.get("errors"):
                    for e in r["errors"]:
                        if isinstance(e, dict) and e.get("message"):
                            err_msg = e["message"][:100]
                            break
                failure_details.append(
                    f"  - {tid[:8]} (role: {role}): {ftype}"
                    + (f" — {err_msg}" if err_msg else "")
                )
            content += (
                f"\n\n[RECOVERY NOTICE] {len(failed_entries)} sub-agent(s) "
                f"failed or returned degraded results:\n"
                + "\n".join(failure_details)
                + f"\n\nRecovery budget: ~{recovery_pct:.0%} of original reserved.\n"
                f"IMPORTANT: Analyze each failure and decide your next action. "
                f"State your reasoning clearly. Options:\n"
                f"(1) Synthesize from partial/successful results if coverage is adequate\n"
                f"(2) Respawn a replacement agent with narrower scope or different approach\n"
                f"(3) Perform the failed task yourself using your own tools\n"
                f"Explain which option you chose and why."
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            content=content,
        )

    def _collect_child_result(
        self, task_id: str, workflow: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Collect the result for a single child task after awaiting."""
        pending = self._pending_tasks.pop(task_id, None)
        if pending is not None and pending.done():
            if pending.cancelled():
                return {
                    "task_id": task_id,
                    "status": "cancelled",
                    "error": "Agent task was cancelled.",
                    "failure_type": "cancelled",
                }
            # Even if there was an exception, _run_child now stores
            # a failed AgentOutput on the node, so fall through to
            # the child.result check below.

        child = workflow.agent_tree.get(task_id)
        if child and child.result:
            result_dict = child.result.to_dict()
            # Enrich with failure_type from errors if failed
            if child.result.status == "failed" and child.result.errors:
                for err in child.result.errors:
                    if hasattr(err, "error_type"):
                        result_dict["failure_type"] = err.error_type
                        break
            return result_dict
        if child:
            return {
                "task_id": task_id,
                "status": child.status.value,
                "error": "No result available",
            }
        return {"task_id": task_id, "error": "Agent not found"}

    @staticmethod
    def _summarize_child_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Compact a full AgentOutput dict to its lightweight summary view.

        Keeps the fields a parent agent typically needs to decide whether
        to drill down: status, role, prose summary, finding counts,
        token usage, and any error info. Drops the full findings dict and
        phase outputs — those can be fetched on demand via
        ``get_agent_phase_output`` if the parent decides they're worth
        the extra tokens.
        """
        if not isinstance(result, dict):
            return result

        summary: Dict[str, Any] = {
            "task_id": result.get("task_id"),
            "status": result.get("status"),
            "role": result.get("role"),
            "summary": result.get("summary"),
            "_detail_level": "summary",
        }
        sub_role = result.get("sub_role")
        if sub_role:
            summary["sub_role"] = sub_role

        # Finding counts (drop the bulky structured data)
        findings = result.get("findings")
        if isinstance(findings, dict) and findings:
            counts: Dict[str, int] = {}
            for k, v in findings.items():
                if isinstance(v, list):
                    counts[k] = len(v)
                elif isinstance(v, dict):
                    counts[k] = len(v)
                else:
                    counts[k] = 1
            summary["finding_counts"] = counts

        # Recommendations stay (usually a short list)
        if result.get("recommendations"):
            summary["recommendations"] = result["recommendations"]

        # Token usage (small but useful for cost tracking)
        if "token_usage" in result:
            summary["token_usage"] = result["token_usage"]

        # Error info (always retain for failure handling)
        if result.get("errors"):
            summary["errors"] = result["errors"]
        if result.get("failure_type"):
            summary["failure_type"] = result["failure_type"]

        # Phase metadata (lightweight: counts + names, not contents)
        meta = result.get("execution_metadata") or {}
        phase_outputs = meta.get("phase_outputs") or {}
        if phase_outputs:
            summary["available_phase_outputs"] = list(phase_outputs.keys())

        return summary

    async def _handle_get_agent_phase_output(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
        **_kw: Any,
    ) -> ToolResult:
        """Return the full text of one phase for one completed child."""
        task_id = tool_call.arguments.get("task_id", "")
        phase = str(tool_call.arguments.get("phase", "")).lower().strip()

        if not task_id or not phase:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error="get_agent_phase_output requires both 'task_id' and 'phase'.",
            )
        if phase not in ("plan", "act", "review", "report"):
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Unknown phase '{phase}'. Expected plan|act|review|report.",
            )

        child = workflow.agent_tree.get(task_id)
        if not child:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"No agent found with task_id {task_id}.",
            )
        if child.status == AgentStatus.RUNNING:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Agent {task_id} is still running. Use wait_for_agents first.",
            )
        if child.result is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Agent {task_id} has no result.",
            )

        phase_outputs = (
            child.result.execution_metadata.phase_outputs or {}
        )
        text = phase_outputs.get(phase)
        if not text:
            available = ", ".join(phase_outputs.keys()) or "(none)"
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=(
                    f"Phase '{phase}' has no recorded output for agent {task_id}. "
                    f"Available phases: {available}."
                ),
            )

        payload = {
            "task_id": task_id,
            "role": child.config.role,
            "sub_role": child.config.sub_role,
            "phase": phase,
            "output": text,
        }
        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            content=json.dumps(payload, indent=2, default=str),
        )

    async def _handle_get_agent_result(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
        **_kw: Any,
    ) -> ToolResult:
        """Handle get_agent_result tool call."""
        task_id = tool_call.arguments.get("task_id", "")

        child = workflow.agent_tree.get(task_id)
        if not child:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"No agent found with task_id {task_id}.",
            )

        if child.status == AgentStatus.RUNNING:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Agent {task_id} is still running. Use wait_for_agents first.",
            )

        if child.result is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                content="",
                error=f"Agent {task_id} has no result (status: {child.status.value}).",
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            content=json.dumps(child.result.to_dict(), indent=2, default=str),
        )

    async def _handle_get_agent_results_all(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
        **_kw: Any,
    ) -> ToolResult:
        """Handle get_agent_results_all tool call."""
        results = {}
        for tid in parent_node.children:
            child = workflow.agent_tree.get(tid)
            if child and child.result:
                results[tid] = {
                    "role": child.config.role,
                    "sub_role": child.config.sub_role,
                    "status": child.result.status,
                    "summary": child.result.summary,
                    "findings": child.result.findings,
                }
            elif child:
                results[tid] = {
                    "role": child.config.role,
                    "status": child.status.value,
                }

        if not results:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=True,
                content="No child agents have been spawned.",
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            content=json.dumps(results, indent=2, default=str),
        )

    # -----------------------------------------------------------------------
    # Coordination handlers (message passing + shared state)
    # -----------------------------------------------------------------------

    async def _handle_send_message(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
        **_kw: Any,
    ) -> ToolResult:
        """Handle send_message tool call."""
        args = tool_call.arguments
        to_task_id = args.get("to_task_id", "")
        content = args.get("content", "")
        message_type = args.get("message_type", "info")
        data = args.get("data") or {}

        if not to_task_id:
            return ToolResult(
                tool_call_id=tool_call.id, success=False, content="",
                error="send_message requires a non-empty 'to_task_id'.",
            )
        if not content:
            return ToolResult(
                tool_call_id=tool_call.id, success=False, content="",
                error="send_message requires a non-empty 'content'.",
            )

        # Check recipient exists
        if to_task_id not in workflow.agent_tree:
            return ToolResult(
                tool_call_id=tool_call.id, success=False, content="",
                error=f"No agent found with task_id '{to_task_id}'.",
            )

        # Permission check
        if not self._coordinator.can_send_message(
            parent_node.task_id, to_task_id, workflow,
        ):
            return ToolResult(
                tool_call_id=tool_call.id, success=False, content="",
                error=(
                    f"Cannot send message to agent '{to_task_id}'. "
                    f"Messages are only allowed between parent/child and sibling agents."
                ),
            )

        message = self._coordinator.send_message(
            from_task_id=parent_node.task_id,
            to_task_id=to_task_id,
            content=content,
            message_type=message_type,
            data=data,
        )

        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            content=json.dumps({
                "message_id": message.message_id,
                "to_task_id": to_task_id,
                "status": "delivered",
            }),
        )

    async def _handle_read_messages(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
        **_kw: Any,
    ) -> ToolResult:
        """Handle read_messages tool call."""
        cursor = self._message_cursors.get(parent_node.task_id, 0)
        messages = self._coordinator.read_messages(parent_node.task_id, cursor)

        # Advance cursor
        new_cursor = cursor + len(messages)
        self._message_cursors[parent_node.task_id] = new_cursor

        if not messages:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=True,
                content="No new messages.",
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            content=json.dumps(
                [m.to_dict() for m in messages],
                indent=2,
                default=str,
            ),
        )

    async def _handle_set_shared_state(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
        **_kw: Any,
    ) -> ToolResult:
        """Handle set_shared_state tool call."""
        args = tool_call.arguments
        key = args.get("key", "")
        value = args.get("value")

        if not key:
            return ToolResult(
                tool_call_id=tool_call.id, success=False, content="",
                error="set_shared_state requires a non-empty 'key'.",
            )

        # Permission check
        if not self._coordinator.can_access_state(
            parent_node.task_id, key, "write", workflow,
        ):
            return ToolResult(
                tool_call_id=tool_call.id, success=False, content="",
                error=f"Access denied: cannot write shared state key '{key}'.",
            )

        self._coordinator.set_shared_state(
            workflow_id=workflow.workflow_id,
            key=key,
            value=value,
            set_by=parent_node.task_id,
        )

        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            content=json.dumps({
                "key": key,
                "status": "stored",
            }),
        )

    async def _handle_get_shared_state(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
        **_kw: Any,
    ) -> ToolResult:
        """Handle get_shared_state tool call."""
        key = tool_call.arguments.get("key")

        # Permission check
        check_key = key or "*"
        if not self._coordinator.can_access_state(
            parent_node.task_id, check_key, "read", workflow,
        ):
            return ToolResult(
                tool_call_id=tool_call.id, success=False, content="",
                error=f"Access denied: cannot read shared state key '{check_key}'.",
            )

        result = self._coordinator.get_shared_state(
            workflow_id=workflow.workflow_id,
            key=key,
        )

        if result is None and key is not None:
            return ToolResult(
                tool_call_id=tool_call.id,
                success=True,
                content=json.dumps({"key": key, "value": None, "exists": False}),
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            content=json.dumps(
                {"key": key, "value": result} if key else {"state": result},
                indent=2,
                default=str,
            ),
        )

    # -----------------------------------------------------------------------
    # Pending-task lifecycle helpers
    # -----------------------------------------------------------------------

    async def _cancel_pending_tasks(self) -> None:
        """Cancel all pending background child tasks and await their completion.

        Loops until no new tasks remain — handles the race where a child
        agent spawns another child just before being cancelled.
        """
        _MAX_ROUNDS = 5  # safety cap to avoid infinite loops
        for _ in range(_MAX_ROUNDS):
            tasks_snapshot = list(self._pending_tasks.values())
            if not tasks_snapshot:
                break
            for task in tasks_snapshot:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks_snapshot, return_exceptions=True)
            # Remove the tasks we just processed
            done_ids = [
                tid for tid, t in self._pending_tasks.items()
                if t in tasks_snapshot
            ]
            for tid in done_ids:
                self._pending_tasks.pop(tid, None)
            # If no new tasks were added during await, we're done
            if not self._pending_tasks:
                break
        # Final safety clear
        self._pending_tasks.clear()

    # -----------------------------------------------------------------------
    # Config resolution
    # -----------------------------------------------------------------------

    def _resolve_agent_config(
        self,
        role: str,
        sub_role: Optional[str] = None,
        system_prompt: str = "",
        model: str = "",
        model_config: Optional[Dict[str, Any]] = None,
    ) -> AgentConfig:
        """
        Resolve agent config from explicit params or domain adapter.

        Explicit params take precedence over adapter-provided values.
        """
        from .core_types import ModelConfig

        # Try adapter first
        if self._domain_adapter:
            try:
                adapter_config = self._domain_adapter.get_role_config(role, sub_role)
                # Merge: explicit params override adapter
                return AgentConfig(
                    agent_id=generate_id(),
                    role=role,
                    sub_role=sub_role,
                    system_prompt=system_prompt or adapter_config.system_prompt,
                    model=model or adapter_config.model,
                    model_config=ModelConfig(
                        **(model_config or {})
                    ) if model_config else adapter_config.model_config,
                )
            except Exception as e:
                logger.debug(f"Adapter config resolution failed for {role}: {e}")

        # Fall back to explicit params
        return AgentConfig(
            agent_id=generate_id(),
            role=role,
            sub_role=sub_role,
            system_prompt=system_prompt,
            model=model,
            model_config=ModelConfig(**(model_config or {})),
        )

    def _build_report_template_handler(self) -> Optional[Callable]:
        """
        Build a report template handler from the domain adapter.

        Returns a callable (role, sub_role) -> str that delegates to the adapter's
        get_report_template method, or None if no adapter is configured.
        """
        if not self._domain_adapter:
            return None

        def handler(role: str = "", sub_role: Optional[str] = None) -> str:
            try:
                template = self._domain_adapter.get_report_template(role, sub_role)
                return template or (
                    "No specific report template is defined for this role. "
                    "Use your best judgment to structure the report based on "
                    "your findings and the output schema."
                )
            except Exception as e:
                logger.debug(f"Failed to get report template for {role}: {e}")
                return (
                    "Could not retrieve report template. Use your best judgment "
                    "to structure the report based on your findings."
                )

        return handler

    def _build_roles_description(self) -> str:
        """Build the available roles description for the spawn_agent tool."""
        if not self._domain_adapter:
            return "No roles are configured. Spawning agents is not available."

        try:
            roles = self._domain_adapter.list_available_roles()
            if not roles:
                return "No roles available for spawning."

            lines = ["Available roles for spawning:\n"]
            for role_info in roles:
                lines.append(f"Role: {role_info.get('role', 'unknown')} — {role_info.get('description', '')}")
                sub_roles = role_info.get("sub_roles", [])
                if sub_roles:
                    for sr in sub_roles:
                        lines.append(f"  Sub-role: {sr}")
                lines.append("")

            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Failed to build roles description: {e}")
            return "Role catalog is not available."
