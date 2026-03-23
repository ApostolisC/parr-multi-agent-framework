"""Integration tests — full workflow through Orchestrator -> AgentRuntime -> PhaseRunner."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from parr.core_types import (
    AgentConfig,
    AgentNode,
    AgentOutput,
    BudgetConfig,
    ErrorEntry,
    ErrorSource,
    Phase,
    ToolCall,
    ToolDef,
    ToolResult,
    WorkflowExecution,
)
from parr.event_bus import EventBus, InMemoryEventSink
from parr.orchestrator import Orchestrator
from parr.tests.mock_llm import (
    MockToolCallingLLM,
    make_text_response,
    make_tool_call_response,
)


# =========================================================================
# Helpers
# =========================================================================

def _simple_llm_responses():
    """LLM responses for a minimal PLAN -> ACT -> REVIEW -> REPORT workflow."""
    return [
        # PLAN: just text, finishes planning
        make_text_response("Plan: Analyze the data and report findings."),
        # ACT: call log_finding then finish
        make_tool_call_response(
            "log_finding",
            {"category": "risk", "content": "Found a potential issue", "source": "test data"},
        ),
        make_text_response("Analysis complete."),
        # REVIEW: pass
        make_text_response("REVIEW_PASSED"),
        # REPORT: submit report
        make_tool_call_response(
            "submit_report",
            {"summary": "Test analysis complete.", "findings": {"risks": ["issue_1"]}},
        ),
    ]


def _make_orchestrator(llm=None, **kwargs):
    """Create an orchestrator with a mock LLM."""
    if llm is None:
        llm = MockToolCallingLLM(_simple_llm_responses())
    return Orchestrator(llm=llm, **kwargs)


# =========================================================================
# 1. Full workflow test: Orchestrator -> AgentRuntime -> PhaseRunner
# =========================================================================

class TestFullWorkflow:

    @pytest.mark.asyncio
    async def test_simple_workflow_completes(self):
        """A minimal workflow runs through all phases and returns output."""
        orch = _make_orchestrator()
        output = await orch.start_workflow(
            task="Analyze the test data.",
            role="analyst",
            system_prompt="You are a test analyst.",
            model="test-model",
            budget=BudgetConfig(max_tokens=50000),
        )
        assert output.status in ("completed", "degraded")
        assert output.task_id
        assert output.agent_id

    @pytest.mark.asyncio
    async def test_workflow_with_tool_calls(self):
        """Workflow that uses domain tools produces findings."""
        async def search_handler(query: str = "") -> str:
            return f"Results for: {query}"

        search_tool = ToolDef(
            name="search_data",
            description="Search test data.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=search_handler,
            phase_availability=[Phase.ACT],
        )

        llm = MockToolCallingLLM([
            make_text_response("Plan: search for data then report."),
            make_tool_call_response("search_data", {"query": "test"}),
            make_tool_call_response(
                "log_finding",
                {"category": "data", "content": "Found data", "source": "search"},
            ),
            make_text_response("Done with analysis."),
            make_text_response("REVIEW_PASSED"),
            make_tool_call_response(
                "submit_report",
                {"summary": "Found data.", "findings": {"data": ["result"]}},
            ),
        ])

        orch = _make_orchestrator(llm=llm)
        output = await orch.start_workflow(
            task="Search and analyze.",
            role="analyst",
            system_prompt="You are a test analyst.",
            model="test-model",
            tools=[search_tool],
            budget=BudgetConfig(max_tokens=100000),
        )
        assert output.status in ("completed", "degraded")

    @pytest.mark.asyncio
    async def test_workflow_event_emission(self):
        """Events are emitted during workflow execution."""
        sink = InMemoryEventSink()
        orch = _make_orchestrator(event_sink=sink)
        await orch.start_workflow(
            task="Analyze data.",
            role="analyst",
            system_prompt="Analyst.",
            model="test-model",
            budget=BudgetConfig(max_tokens=50000),
        )
        # Should have at least agent_started event
        events = sink.get_events()
        assert len(events) > 0
        event_types = {e.event_type for e in events}
        assert "agent_started" in event_types


# =========================================================================
# 2. Budget exhaustion mid-phase
# =========================================================================

class TestBudgetExhaustion:

    @pytest.mark.asyncio
    async def test_tiny_budget_still_produces_output(self):
        """Even with minimal budget, we get a structured output (possibly degraded)."""
        orch = _make_orchestrator()
        output = await orch.start_workflow(
            task="Analyze data.",
            role="analyst",
            system_prompt="Analyst.",
            model="test-model",
            budget=BudgetConfig(max_tokens=500),
        )
        # With very low budget, agent may fail or degrade but should return
        assert output.task_id
        assert output.status in ("completed", "degraded", "failed", "partial")


# =========================================================================
# 3. Error injection tests
# =========================================================================

class TestErrorInjection:

    @pytest.mark.asyncio
    async def test_llm_failure_produces_output(self):
        """If the LLM raises, the framework catches it and returns a failed output."""
        class FailingLLM:
            async def chat_with_tools(self, *args, **kwargs):
                raise RuntimeError("LLM connection failed")

        orch = Orchestrator(llm=FailingLLM())
        output = await orch.start_workflow(
            task="Test.",
            role="analyst",
            system_prompt="Test.",
            model="test-model",
            budget=BudgetConfig(max_tokens=10000),
        )
        # Framework catches LLM errors and returns structured output
        assert output.task_id
        assert output.status in ("failed", "degraded")

    @pytest.mark.asyncio
    async def test_tool_crash_is_handled(self):
        """A domain tool that raises should be caught and reported, not crash the workflow."""
        async def crashing_handler(**kwargs) -> str:
            raise ValueError("Tool crashed!")

        crash_tool = ToolDef(
            name="crasher",
            description="A tool that crashes.",
            parameters={"type": "object", "properties": {}},
            handler=crashing_handler,
            phase_availability=[Phase.ACT],
        )

        llm = MockToolCallingLLM([
            make_text_response("Plan: use the tool."),
            make_tool_call_response("crasher", {}),
            make_text_response("Tool failed, moving on."),
            make_text_response("REVIEW_PASSED"),
            make_tool_call_response(
                "submit_report",
                {"summary": "Partial analysis.", "findings": {}},
            ),
        ])

        orch = _make_orchestrator(llm=llm)
        output = await orch.start_workflow(
            task="Use the crasher tool.",
            role="analyst",
            system_prompt="Analyst.",
            model="test-model",
            tools=[crash_tool],
            budget=BudgetConfig(max_tokens=50000),
        )
        # Should complete despite the tool crash
        assert output.task_id
        assert output.status in ("completed", "degraded", "failed", "partial")


# =========================================================================
# 4. Silent failure fixes (2.1) — persist errors surfaced
# =========================================================================

class TestSilentFailureFixes:

    @pytest.mark.asyncio
    async def test_persist_output_error_surfaced(self):
        """If domain adapter's persist_output raises, error appears in output.errors."""

        class FailingAdapter:
            def resolve_role(self, role, sub_role=None):
                return None
            def get_domain_tools(self, role, sub_role=None):
                return []
            def get_output_schema(self, role, sub_role=None):
                return None
            def get_report_template(self, role, sub_role=None):
                return None
            def persist_output(self, workflow_id, output):
                raise IOError("Disk full")

        orch = _make_orchestrator(domain_adapter=FailingAdapter())
        output = await orch.start_workflow(
            task="Test.",
            role="analyst",
            system_prompt="Test.",
            model="test-model",
            budget=BudgetConfig(max_tokens=50000),
        )
        persist_errors = [
            e for e in output.errors
            if e.error_type == "persist_output_failed"
        ]
        assert len(persist_errors) == 1
        assert "Disk full" in persist_errors[0].message

    def test_event_bus_handler_error_callback(self):
        """EventBus on_handler_error callback is invoked on handler failure."""
        errors = []

        def capture_error(event, exc):
            errors.append((event, exc))

        bus = EventBus(on_handler_error=capture_error)

        async def bad_handler(event):
            raise ValueError("Handler boom")

        from parr.event_types import FrameworkEvent

        bus.subscribe("wf-1", bad_handler)

        async def _run():
            await bus.publish(FrameworkEvent(
                workflow_id="wf-1",
                task_id="t-1",
                agent_id="a-1",
                event_type="test_event",
            ))

        asyncio.get_event_loop().run_until_complete(_run())
        assert len(errors) == 1
        assert isinstance(errors[0][1], ValueError)


# =========================================================================
# 5. Race condition fix (2.2) — cancel_pending_tasks
# =========================================================================

class TestCancelPendingTasks:

    @pytest.mark.asyncio
    async def test_cancel_handles_empty(self):
        """Cancellation with no pending tasks completes cleanly."""
        orch = _make_orchestrator()
        await orch._cancel_pending_tasks()
        assert len(orch._pending_tasks) == 0

    @pytest.mark.asyncio
    async def test_cancel_handles_running_tasks(self):
        """Cancellation properly awaits running tasks."""
        orch = _make_orchestrator()

        async def slow_task():
            await asyncio.sleep(10)

        orch._pending_tasks["t-1"] = asyncio.create_task(slow_task())
        orch._pending_tasks["t-2"] = asyncio.create_task(slow_task())

        await orch._cancel_pending_tasks()
        assert len(orch._pending_tasks) == 0


# =========================================================================
# 6. Async file I/O (2.3)
# =========================================================================

class TestAsyncFileIO:

    @pytest.mark.asyncio
    async def test_async_write_json(self, tmp_path):
        from parr.persistence import _awrite_json, _aread_json

        path = tmp_path / "test.json"
        await _awrite_json(path, {"key": "value"})
        data = await _aread_json(path)
        assert data == {"key": "value"}

    @pytest.mark.asyncio
    async def test_agent_store_async_methods(self, tmp_path):
        from parr.persistence import AgentFileStore

        store = AgentFileStore(tmp_path / "agent")
        await store.async_save_agent_info(
            task_id="t-1",
            agent_id="a-1",
            role="analyst",
            task="test task",
        )
        info = store.read_agent_info()
        assert info["task_id"] == "t-1"
        assert info["role"] == "analyst"

    @pytest.mark.asyncio
    async def test_workflow_store_async_methods(self, tmp_path):
        from parr.persistence import WorkflowFileStore

        wf = WorkflowFileStore(tmp_path, "wf-1")
        await wf.async_save_workflow_info(
            workflow_id="wf-1",
            status="running",
        )
        info = wf.read_workflow_info()
        assert info["workflow_id"] == "wf-1"
        assert info["status"] == "running"

        await wf.async_update_workflow_status("completed")
        info = wf.read_workflow_info()
        assert info["status"] == "completed"


# =========================================================================
# 7. Type safety (2.4) — Literal types
# =========================================================================

class TestTypeSafety:

    def test_agent_output_status_literal(self):
        """AgentOutput.status accepts valid literal values."""
        output = AgentOutput(task_id="t", agent_id="a", role="r")
        assert output.status == "completed"

        output2 = AgentOutput(task_id="t", agent_id="a", role="r", status="degraded")
        assert output2.status == "degraded"

    def test_agent_message_type_literal(self):
        from parr.core_types import AgentMessage
        msg = AgentMessage(message_type="data")
        assert msg.message_type == "data"

    def test_literal_type_exports(self):
        """New type aliases are importable from the package."""
        from parr import (
            AgentOutputStatus,
            ExecutionPath,
            MessageType,
            PlanContextStatus,
            PlanContextType,
        )
        # Verify they are Literal types (check they exist)
        assert AgentOutputStatus is not None
        assert ExecutionPath is not None
        assert MessageType is not None
        assert PlanContextStatus is not None
        assert PlanContextType is not None
