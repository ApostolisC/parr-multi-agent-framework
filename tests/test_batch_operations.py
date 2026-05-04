"""Tests for batch_operations — phase checks, key normalisation, event emissions."""
import asyncio
import json

import pytest

from parr.core_types import (
    AgentConfig,
    AgentNode,
    BudgetConfig,
    Phase,
    PoliciesConfig,
    SpawnPolicy,
    ToolCall,
    ToolDef,
    ToolResult,
    WorkflowExecution,
    generate_id,
)
from parr.orchestrator import Orchestrator
from parr.event_bus import EventBus, InMemoryEventSink
from parr.tool_registry import ToolRegistry
from parr.tool_executor import ToolExecutor
from parr.trace_store import TraceStore

from tests.conftest import MockLLM, make_budget


def _make_test_tool(name: str, phase: Phase = Phase.ACT) -> ToolDef:
    return ToolDef(
        name=name,
        description=f"Test tool: {name}",
        parameters={"type": "object", "properties": {}},
        handler=lambda: "ok",
        phase_availability=[phase],
    )


class TestBatchPhaseCheck:
    """batch_operations must check phase availability for each inner op."""

    @pytest.mark.asyncio
    async def test_spawn_blocked_in_report_phase(self):
        """spawn_agent ops in batch_operations should fail in REPORT phase."""
        orch = Orchestrator(
            llm=MockLLM(),
            default_budget=make_budget(),
        )
        parent = AgentNode(
            agent_id=generate_id(),
            config=AgentConfig(role="test", system_prompt="test"),
            budget=make_budget(),
            current_phase=Phase.REPORT,
        )
        wf = WorkflowExecution()
        wf.agent_tree[parent.task_id] = parent
        ts = TraceStore()

        # Build a registry with spawn_agent (ACT-only)
        from parr.framework_tools import build_agent_management_tools
        agent_tools = build_agent_management_tools("")
        registry = ToolRegistry()
        for t in agent_tools:
            registry.register(t)
        executor = ToolExecutor(registry)
        executor.set_phase(Phase.REPORT)

        batch_call = ToolCall(
            id="batch1",
            name="batch_operations",
            arguments={
                "operations": [
                    {"op": "spawn_agent", "role": "test", "task_description": "do stuff"},
                ],
            },
        )

        result = await orch._handle_batch_operations(
            batch_call, parent, wf, ts, "", None,
            tool_executor=executor,
        )

        ops = json.loads(result.content)
        assert isinstance(ops, list)
        assert ops[0]["success"] is False
        assert "REPORT" in ops[0]["error"]
        assert "ACT" in ops[0]["error"]

    @pytest.mark.asyncio
    async def test_submit_report_blocked_in_act_phase(self):
        """submit_report ops (REPORT-only) should fail in ACT phase."""
        orch = Orchestrator(
            llm=MockLLM(),
            default_budget=make_budget(),
        )
        parent = AgentNode(
            agent_id=generate_id(),
            config=AgentConfig(role="test", system_prompt="test"),
            budget=make_budget(),
            current_phase=Phase.ACT,
        )
        wf = WorkflowExecution()
        wf.agent_tree[parent.task_id] = parent
        ts = TraceStore()

        from parr.framework_tools import (
            AgentWorkingMemory,
            build_agent_management_tools,
            build_report_tools,
        )
        registry = ToolRegistry()
        mem = AgentWorkingMemory()
        for t in build_agent_management_tools(""):
            registry.register(t)
        for t in build_report_tools(mem):
            registry.register(t)
        executor = ToolExecutor(registry)
        executor.set_phase(Phase.ACT)

        batch_call = ToolCall(
            id="batch1",
            name="batch_operations",
            arguments={
                "operations": [
                    {"op": "submit_report", "report_markdown": "test"},
                ],
            },
        )

        result = await orch._handle_batch_operations(
            batch_call, parent, wf, ts, "", None,
            tool_executor=executor,
        )

        ops = json.loads(result.content)
        assert ops[0]["success"] is False
        assert "ACT" in ops[0]["error"]

    @pytest.mark.asyncio
    async def test_mixed_ops_partial_success(self):
        """Some ops succeed, some fail — both results returned."""
        orch = Orchestrator(
            llm=MockLLM(),
            default_budget=make_budget(),
        )
        parent = AgentNode(
            agent_id=generate_id(),
            config=AgentConfig(role="test", system_prompt="test"),
            budget=make_budget(),
            current_phase=Phase.ACT,
        )
        wf = WorkflowExecution()
        wf.agent_tree[parent.task_id] = parent
        ts = TraceStore()

        # Register a tool available in ACT and one in REPORT
        act_tool = ToolDef(
            name="act_tool",
            description="Available in ACT",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "act result",
            phase_availability=[Phase.ACT],
        )
        report_tool = ToolDef(
            name="report_tool",
            description="Available in REPORT",
            parameters={"type": "object", "properties": {}},
            handler=lambda: "report result",
            phase_availability=[Phase.REPORT],
        )

        from parr.framework_tools import build_agent_management_tools
        registry = ToolRegistry()
        for t in build_agent_management_tools(""):
            registry.register(t)
        registry.register(act_tool)
        registry.register(report_tool)
        executor = ToolExecutor(registry)
        executor.set_phase(Phase.ACT)

        batch_call = ToolCall(
            id="batch1",
            name="batch_operations",
            arguments={
                "operations": [
                    {"op": "act_tool"},
                    {"op": "report_tool"},
                ],
            },
        )

        result = await orch._handle_batch_operations(
            batch_call, parent, wf, ts, "", None,
            tool_executor=executor,
        )

        ops = json.loads(result.content)
        assert len(ops) == 2
        assert ops[0]["success"] is True  # act_tool works in ACT
        assert ops[1]["success"] is False  # report_tool blocked in ACT


class TestBatchKeyNormalisation:
    @pytest.mark.asyncio
    async def test_camel_case_normalised_in_batch_ops(self):
        """camelCase keys in batch op arguments should be normalised."""
        orch = Orchestrator(
            llm=MockLLM(),
            default_budget=make_budget(),
        )
        parent = AgentNode(
            agent_id=generate_id(),
            config=AgentConfig(role="test", system_prompt="test"),
            budget=make_budget(),
            current_phase=Phase.ACT,
        )
        wf = WorkflowExecution()
        wf.agent_tree[parent.task_id] = parent
        ts = TraceStore()

        received_args = {}
        def capture_handler(name: str = "", effort_level: int = 0):
            received_args["name"] = name
            received_args["effort_level"] = effort_level
            return "captured"

        tool = ToolDef(
            name="capture_tool",
            description="Captures arguments",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "effort_level": {"type": "integer"},
                },
            },
            handler=capture_handler,
            phase_availability=[Phase.ACT],
        )

        from parr.framework_tools import build_agent_management_tools
        registry = ToolRegistry()
        for t in build_agent_management_tools(""):
            registry.register(t)
        registry.register(tool)
        executor = ToolExecutor(registry)
        executor.set_phase(Phase.ACT)

        batch_call = ToolCall(
            id="batch1",
            name="batch_operations",
            arguments={
                "operations": [
                    {"op": "capture_tool", "name": "test", "effortLevel": 3},
                ],
            },
        )

        result = await orch._handle_batch_operations(
            batch_call, parent, wf, ts, "", None,
            tool_executor=executor,
        )

        ops = json.loads(result.content)
        assert ops[0]["success"] is True
