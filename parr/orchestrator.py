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
    Phase,
    PhaseConfig,
    SimpleQueryBypassConfig,
    StallDetectionConfig,
    ToolCall,
    ToolDef,
    ToolResult,
    TokenUsage,
    TraceEntry,
    WorkflowExecution,
    WorkflowStatus,
    generate_id,
    utc_now,
)
from .protocols import DomainAdapter, EventSink, ToolCallingLLM

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
            )

            # Execute with orchestrator tool handling
            output = await runtime.execute(
                config=config,
                input=agent_input,
                node=root_node,
                workflow=workflow,
                on_orchestrator_tool=lambda tc: self._handle_orchestrator_tool(
                    tc, root_node, workflow, trace_store, roles_desc,
                    wf_store,
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
    ) -> ToolResult:
        """
        Handle orchestrator-level tool calls (spawn_agent, wait_for_agents, etc.).
        """
        handlers = {
            "spawn_agent": self._handle_spawn_agent,
            "wait_for_agents": self._handle_wait_for_agents,
            "get_agent_result": self._handle_get_agent_result,
            "get_agent_results_all": self._handle_get_agent_results_all,
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
            return await handler(
                tool_call, parent_node, workflow, trace_store, roles_description,
                wf_store,
            )
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

    async def _handle_spawn_agent(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
    ) -> ToolResult:
        """Handle spawn_agent tool call.

        Launches the child agent as a background ``asyncio.Task`` and returns
        immediately with the child's ``task_id``.  The parent can continue
        working (or spawn more children) and later call ``wait_for_agents``
        to collect results.
        """
        args = tool_call.arguments
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
                    on_orchestrator_tool=lambda tc: self._handle_orchestrator_tool(
                        tc, child_node, workflow, trace_store, roles_description,
                        wf_store,
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

        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            content=json.dumps({
                "task_id": child_node.task_id,
                "agent_id": child_config.agent_id,
                "role": role,
                "sub_role": sub_role,
                "status": "spawned",
                "message": (
                    f"Agent spawned and running in the background. "
                    f"Use wait_for_agents with task_id '{child_node.task_id}' "
                    f"to collect results when ready."
                ),
            }),
        )

    async def _handle_wait_for_agents(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
    ) -> ToolResult:
        """Handle wait_for_agents tool call.

        Awaits all requested child tasks concurrently via ``asyncio.gather``.
        Tasks that have already finished are collected immediately.
        A configurable timeout prevents indefinite blocking.
        """
        task_ids = tool_call.arguments.get("task_ids", [])

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
        results = {
            tid: self._collect_child_result(tid, workflow)
            for tid in task_ids
        }

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

    async def _handle_get_agent_result(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
        wf_store: Optional[WorkflowFileStore] = None,
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
