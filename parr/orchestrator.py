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
from .budget_tracker import BudgetExceededException, BudgetTracker
from .event_bus import EventBus, EventBridge, InMemoryEventSink
from .event_types import (
    agent_cancelled,
    agent_failed,
    budget_exceeded as budget_exceeded_event,
    # TODO(v2): Import agent_suspended and agent_resumed from event_types
    # when async agent execution with real suspension/resumption is added.
    # Currently unused — v1 runs all agents synchronously.
)
from .trace_store import TraceStore
from .core_types import (
    AgentConfig,
    AgentInput,
    AgentNode,
    AgentOutput,
    AgentStatus,
    BudgetConfig,
    BudgetUsage,
    CostConfig,
    ErrorEntry,
    ErrorSource,
    ExecutionMetadata,
    Phase,
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
    ) -> None:
        self._llm = llm
        self._domain_adapter = domain_adapter
        self._cost_config = cost_config
        self._max_review_cycles = max_review_cycles
        self._phase_limits = phase_limits
        self._default_budget = default_budget or BudgetConfig()
        self._stream = stream

        # Internal event bus
        self._event_bus = EventBus()

        # Bridge to external event sink
        self._event_sink = event_sink or InMemoryEventSink()
        self._event_bridge = EventBridge(self._event_bus, self._event_sink)

        # Budget tracker
        self._budget_tracker = BudgetTracker(cost_config)

        # Active workflows
        self._workflows: Dict[str, WorkflowExecution] = {}
        self._trace_stores: Dict[str, TraceStore] = {}

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
        """
        workflow_budget = budget or self._default_budget

        # Resolve config from adapter if available
        config = self._resolve_agent_config(role, sub_role, system_prompt, model, model_config)

        # Resolve tools from adapter if not provided
        if tools is None and self._domain_adapter:
            tools = self._domain_adapter.get_domain_tools(role, sub_role)
        tools = tools or []

        # Resolve output schema from adapter if not provided
        if output_schema is None and self._domain_adapter:
            output_schema = self._domain_adapter.get_output_schema(role, sub_role)

        # Create workflow
        workflow = WorkflowExecution(
            global_budget=workflow_budget,
        )
        self._workflows[workflow.workflow_id] = workflow

        # Create trace store for this workflow
        trace_store = TraceStore()
        self._trace_stores[workflow.workflow_id] = trace_store

        # Connect event bridge
        self._event_bridge.connect(workflow.workflow_id)

        # Build available roles description for spawn_agent tool
        roles_desc = self._build_roles_description()

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

            # Add to trace
            trace_store.add_entry(TraceEntry(
                task_id=root_node.task_id,
                agent_id=config.agent_id,
                role=config.role,
                sub_role=config.sub_role,
                task_description=task[:200],
            ))

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
            )

            # Execute with orchestrator tool handling
            output = await runtime.execute(
                config=config,
                input=agent_input,
                node=root_node,
                workflow=workflow,
                on_orchestrator_tool=lambda tc: self._handle_orchestrator_tool(
                    tc, root_node, workflow, trace_store, roles_desc,
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

            # Persist output if adapter is available
            if self._domain_adapter and output.status in _SUCCESS_STATUSES:
                try:
                    self._domain_adapter.persist_output(workflow.workflow_id, output)
                except Exception as e:
                    logger.error(f"Failed to persist workflow output: {e}")

            return output

        except Exception as e:
            logger.error(f"Workflow {workflow.workflow_id} failed: {e}", exc_info=True)
            workflow.status = WorkflowStatus.FAILED
            raise

        finally:
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
    ) -> ToolResult:
        """
        Handle orchestrator-level tool calls (spawn_agent, wait_for_agents, etc.).
        """
        handlers = {
            "spawn_agent": self._handle_spawn_agent,
            "wait_for_agents": self._handle_wait_for_agents,
            "get_agent_result": self._handle_get_agent_result,
            "get_agent_results_all": self._handle_get_agent_results_all,
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
    ) -> ToolResult:
        """Handle spawn_agent tool call."""
        args = tool_call.arguments
        role = args.get("role", "")
        sub_role = args.get("sub_role")
        task_description = args.get("task_description", "")

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

        # Build child input
        child_input = AgentInput(
            task=task_description,
            tools=child_tools,
            output_schema=child_schema,
            raw_data=args.get("raw_data"),
            additional_context=args.get("additional_context"),
            budget=child_budget,
            trace_snapshot=trace_store.get_snapshot(child_node.task_id),
        )

        # Create child runtime and execute synchronously.
        # Sub-agents get at most 1 review cycle to avoid costly retry loops
        # that rarely converge (sub-agents typically lack the data sources
        # to satisfy a reviewer's demands for more detail).
        child_review_cycles = min(self._max_review_cycles, 1)
        child_runtime = AgentRuntime(
            llm=self._llm,
            budget_tracker=self._budget_tracker,
            event_bus=self._event_bus,
            max_review_cycles=child_review_cycles,
            phase_limits=self._phase_limits,
            available_roles_description=roles_description,
            report_template_handler=self._build_report_template_handler(),
            stream=self._stream,
        )

        child_output = await child_runtime.execute(
            config=child_config,
            input=child_input,
            node=child_node,
            workflow=workflow,
            on_orchestrator_tool=lambda tc: self._handle_orchestrator_tool(
                tc, child_node, workflow, trace_store, roles_description,
            ),
        )

        # Update trace
        _child_success = child_output.status in ("completed", "degraded")
        trace_store.update_status(
            child_node.task_id,
            AgentStatus.COMPLETED if _child_success else AgentStatus.FAILED,
            output_summary=child_output.summary[:200],
        )

        # Return child output as tool result
        output_summary = json.dumps(child_output.to_dict(), indent=2, default=str)
        return ToolResult(
            tool_call_id=tool_call.id,
            success=_child_success,
            content=output_summary,
            error=None if _child_success else child_output.summary,
        )

    async def _handle_wait_for_agents(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
    ) -> ToolResult:
        """Handle wait_for_agents tool call."""
        task_ids = tool_call.arguments.get("task_ids", [])

        # Validate all task IDs are children
        for tid in task_ids:
            if tid not in parent_node.children:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    content="",
                    error=f"Task {tid} is not a child of this agent.",
                )

        # Since we execute children synchronously, they should already be complete
        # This tool is mainly useful for future async execution support
        results = {}
        for tid in task_ids:
            child = workflow.agent_tree.get(tid)
            if child and child.result:
                results[tid] = child.result.to_dict()
            elif child:
                results[tid] = {
                    "task_id": tid,
                    "status": child.status.value,
                    "error": "No result available",
                }
            else:
                results[tid] = {"task_id": tid, "error": "Agent not found"}

        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            content=json.dumps(results, indent=2, default=str),
        )

    async def _handle_get_agent_result(
        self,
        tool_call: ToolCall,
        parent_node: AgentNode,
        workflow: WorkflowExecution,
        trace_store: TraceStore,
        roles_description: str,
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
