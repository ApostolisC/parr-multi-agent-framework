"""Tests for incremental persistence (live updates during phase execution).

Covers:
1. PhaseRunner on_tool_persisted callback invocation
2. Callback receives correct tool records
3. Callback errors are caught and don't break phase execution
4. AgentRuntime wires up incremental persistence with file_store
5. _persist_phase no longer re-appends tool calls
6. Adaptive flow entry tool calls persisted incrementally
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from parr.core_types import (
    AdaptiveFlowConfig,
    AgentConfig,
    AgentInput,
    AgentNode,
    BudgetConfig,
    ExecutionMetadata,
    LLMResponse,
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
from parr.agent_runtime import AgentRuntime
from parr.budget_tracker import BudgetTracker
from parr.context_manager import ContextManager
from parr.event_bus import EventBus
from parr.phase_runner import PhaseRunner, PhaseResult
from parr.tool_executor import ToolExecutor
from parr.tool_registry import ToolRegistry
from parr.tests.mock_llm import (
    MockToolCallingLLM,
    make_text_response,
    make_tool_call_response,
    make_multi_tool_response,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_config(role="test_agent", system_prompt="You are a test agent."):
    return AgentConfig(
        agent_id=generate_id(),
        role=role,
        system_prompt=system_prompt,
        model="test-model",
        model_config=ModelConfig(temperature=0.3, max_tokens=1024),
    )


def _make_input(task="Test task.", tools=None):
    return AgentInput(task=task, tools=tools or [])


def _make_node(config=None, budget=None):
    config = config or _make_config()
    budget = budget or BudgetConfig(max_tokens=100000)
    return AgentNode(
        task_id=generate_id(),
        agent_id=config.agent_id,
        config=config,
        budget=budget,
    )


def _make_workflow():
    return WorkflowExecution(
        workflow_id=generate_id(),
        root_task_id=generate_id(),
    )


def _make_domain_tool(name="search_data"):
    async def handler(query: str = "") -> str:
        return f"Results for: {query}"

    return ToolDef(
        name=name,
        description=f"Search with {name}.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=handler,
        phase_availability=[Phase.ACT],
    )


def _make_phase_runner(
    llm=None,
    registry=None,
    on_tool_persisted=None,
):
    """Build a minimal PhaseRunner with the given callback."""
    llm = llm or MockToolCallingLLM([make_text_response("done")])
    registry = registry or ToolRegistry()
    budget = BudgetTracker()
    ctx = ContextManager()
    event_bus = EventBus()
    return PhaseRunner(
        llm=llm,
        tool_executor=ToolExecutor(registry=registry),
        tool_registry=registry,
        budget_tracker=budget,
        context_manager=ctx,
        event_bus=event_bus,
        on_tool_persisted=on_tool_persisted,
    )


def _make_runtime(llm, adaptive_config=None, budget_config=None):
    budget_config = budget_config or BudgetConfig(max_tokens=100000)
    tracker = BudgetTracker()
    event_bus = EventBus()
    return AgentRuntime(
        llm=llm,
        budget_tracker=tracker,
        event_bus=event_bus,
        adaptive_config=adaptive_config,
        budget_config=budget_config,
        simple_query_bypass=SimpleQueryBypassConfig(enabled=False),
    )


# =========================================================================
# 1. PhaseRunner callback invocation
# =========================================================================

class TestPhaseRunnerCallback:
    """Verify PhaseRunner invokes on_tool_persisted after each tool call."""

    @pytest.mark.asyncio
    async def test_callback_invoked_per_tool_call(self):
        """Callback fires once per tool call with the correct record."""
        tool = _make_domain_tool("search_data")
        registry = ToolRegistry()
        registry.register(tool)

        # LLM: call tool once, then text-only
        llm = MockToolCallingLLM([
            make_tool_call_response("search_data", {"query": "test"}),
            make_text_response("done"),
        ])

        persisted_records: List[Dict[str, Any]] = []

        def on_persist(record):
            persisted_records.append(record)

        runner = _make_phase_runner(llm=llm, registry=registry, on_tool_persisted=on_persist)
        node = _make_node()
        workflow = _make_workflow()
        config = node.config
        input_ = _make_input(tools=[tool])

        await runner.run(
            phase=Phase.ACT,
            node=node,
            workflow=workflow,
            config=config,
            input=input_,
        )

        assert len(persisted_records) == 1
        assert persisted_records[0]["name"] == "search_data"
        assert persisted_records[0]["success"] is True
        assert persisted_records[0]["phase"] == "act"

    @pytest.mark.asyncio
    async def test_callback_multiple_tool_calls(self):
        """Callback fires for each tool call in multi-tool responses."""
        tool_a = _make_domain_tool("tool_a")
        tool_b = _make_domain_tool("tool_b")
        registry = ToolRegistry()
        registry.register(tool_a)
        registry.register(tool_b)

        # LLM: two tools in one response, then text
        llm = MockToolCallingLLM([
            make_multi_tool_response([
                {"name": "tool_a", "arguments": {"query": "a"}},
                {"name": "tool_b", "arguments": {"query": "b"}},
            ]),
            make_text_response("done"),
        ])

        persisted_records: List[Dict[str, Any]] = []
        runner = _make_phase_runner(
            llm=llm, registry=registry,
            on_tool_persisted=lambda r: persisted_records.append(r),
        )
        node = _make_node()
        workflow = _make_workflow()

        await runner.run(
            phase=Phase.ACT, node=node, workflow=workflow,
            config=node.config, input=_make_input(tools=[tool_a, tool_b]),
        )

        assert len(persisted_records) == 2
        assert persisted_records[0]["name"] == "tool_a"
        assert persisted_records[1]["name"] == "tool_b"

    @pytest.mark.asyncio
    async def test_callback_across_iterations(self):
        """Callback fires across multiple LLM iterations."""
        tool = _make_domain_tool("search_data")
        registry = ToolRegistry()
        registry.register(tool)

        # LLM: tool call → tool call → text
        llm = MockToolCallingLLM([
            make_tool_call_response("search_data", {"query": "q1"}),
            make_tool_call_response("search_data", {"query": "q2"}),
            make_text_response("done"),
        ])

        persisted_records: List[Dict[str, Any]] = []
        runner = _make_phase_runner(
            llm=llm, registry=registry,
            on_tool_persisted=lambda r: persisted_records.append(r),
        )
        node = _make_node()
        workflow = _make_workflow()

        await runner.run(
            phase=Phase.ACT, node=node, workflow=workflow,
            config=node.config, input=_make_input(tools=[tool]),
        )

        assert len(persisted_records) == 2
        assert persisted_records[0]["arguments"]["query"] == "q1"
        assert persisted_records[1]["arguments"]["query"] == "q2"

    @pytest.mark.asyncio
    async def test_no_callback_when_none(self):
        """No error when on_tool_persisted is None."""
        tool = _make_domain_tool("search_data")
        registry = ToolRegistry()
        registry.register(tool)

        llm = MockToolCallingLLM([
            make_tool_call_response("search_data", {"query": "test"}),
            make_text_response("done"),
        ])

        runner = _make_phase_runner(llm=llm, registry=registry, on_tool_persisted=None)
        node = _make_node()
        workflow = _make_workflow()

        # Should not raise
        result = await runner.run(
            phase=Phase.ACT, node=node, workflow=workflow,
            config=node.config, input=_make_input(tools=[tool]),
        )
        assert len(result.tool_calls_made) == 1

    @pytest.mark.asyncio
    async def test_callback_error_does_not_break_phase(self):
        """If the callback raises, the phase continues normally."""
        tool = _make_domain_tool("search_data")
        registry = ToolRegistry()
        registry.register(tool)

        llm = MockToolCallingLLM([
            make_tool_call_response("search_data", {"query": "test"}),
            make_text_response("done"),
        ])

        def exploding_callback(record):
            raise RuntimeError("Persistence failure!")

        runner = _make_phase_runner(
            llm=llm, registry=registry,
            on_tool_persisted=exploding_callback,
        )
        node = _make_node()
        workflow = _make_workflow()

        # Phase should complete despite callback error
        result = await runner.run(
            phase=Phase.ACT, node=node, workflow=workflow,
            config=node.config, input=_make_input(tools=[tool]),
        )
        assert result.content == "done"
        assert len(result.tool_calls_made) == 1

    @pytest.mark.asyncio
    async def test_callback_receives_failed_tool_result(self):
        """Callback fires even for failed tool calls."""
        async def failing_handler(query: str = "") -> str:
            raise ValueError("Tool error!")

        tool = ToolDef(
            name="bad_tool",
            description="A tool that fails.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
            handler=failing_handler,
            phase_availability=[Phase.ACT],
        )
        registry = ToolRegistry()
        registry.register(tool)

        llm = MockToolCallingLLM([
            make_tool_call_response("bad_tool", {"query": "test"}),
            make_text_response("done"),
        ])

        persisted_records: List[Dict[str, Any]] = []
        runner = _make_phase_runner(
            llm=llm, registry=registry,
            on_tool_persisted=lambda r: persisted_records.append(r),
        )
        node = _make_node()
        workflow = _make_workflow()

        await runner.run(
            phase=Phase.ACT, node=node, workflow=workflow,
            config=node.config, input=_make_input(tools=[tool]),
        )

        assert len(persisted_records) == 1
        assert persisted_records[0]["success"] is False
        assert persisted_records[0]["error"] is not None


# =========================================================================
# 2. PhaseRunner.run_continuation with callback
# =========================================================================

class TestRunContinuationCallback:
    """Callback also works with run_continuation (adaptive flow)."""

    @pytest.mark.asyncio
    async def test_continuation_fires_callback(self):
        """run_continuation uses the same callback as run."""
        tool = _make_domain_tool("search_data")
        registry = ToolRegistry()
        registry.register(tool)

        llm = MockToolCallingLLM([
            make_tool_call_response("search_data", {"query": "test"}),
            make_text_response("done"),
        ])

        persisted_records: List[Dict[str, Any]] = []
        runner = _make_phase_runner(
            llm=llm, registry=registry,
            on_tool_persisted=lambda r: persisted_records.append(r),
        )
        node = _make_node()
        workflow = _make_workflow()

        initial_messages = [
            Message(role=MessageRole.SYSTEM, content="System prompt"),
            Message(role=MessageRole.USER, content="Do the task"),
        ]

        await runner.run_continuation(
            phase=Phase.ACT,
            node=node,
            workflow=workflow,
            config=node.config,
            input=_make_input(tools=[tool]),
            initial_messages=initial_messages,
            initial_tool_calls=[],
            initial_iteration=1,
        )

        assert len(persisted_records) == 1
        assert persisted_records[0]["name"] == "search_data"


# =========================================================================
# 3. AgentRuntime incremental persistence wiring
# =========================================================================

class TestAgentRuntimeIncrementalPersistence:
    """AgentRuntime creates the callback and wires it to PhaseRunner."""

    @pytest.mark.asyncio
    async def test_file_store_receives_incremental_tool_calls(self):
        """When file_store is set, tool calls are written after each execution."""
        tool = _make_domain_tool("search_data")

        # LLM for adaptive flow:
        # 1. Entry: call domain tool
        # 2. ACT continuation: call log_finding
        # 3. ACT continuation: text (ACT done)
        # 4. REPORT: submit_report
        # 5. REPORT: text (done)
        llm = MockToolCallingLLM([
            make_tool_call_response("search_data", {"query": "test"}),
            make_tool_call_response("log_finding",
                                    {"category": "test", "content": "Found it"}),
            make_text_response("ACT complete"),
            make_tool_call_response("submit_report",
                                    {"answer": "The answer", "summary": "Summary"}),
            make_text_response("REPORT complete"),
        ])

        runtime = _make_runtime(
            llm=llm,
            adaptive_config=AdaptiveFlowConfig(enabled=True),
        )

        # Mock file_store
        mock_store = MagicMock()
        mock_store.append_tool_calls = MagicMock()
        mock_store.save_memory = MagicMock()
        mock_store.save_phase_conversation = MagicMock()
        mock_store.save_output = MagicMock()
        mock_store.update_agent_status = MagicMock()
        runtime._file_store = mock_store

        config = _make_config()
        input_ = _make_input(task="Search for test data", tools=[tool])
        node = _make_node(config=config)
        workflow = _make_workflow()

        await runtime.execute(config, input_, node, workflow)

        # append_tool_calls should have been called incrementally
        # At minimum: entry tool call + ACT tool call(s) + REPORT tool call
        call_count = mock_store.append_tool_calls.call_count
        assert call_count >= 2, (
            f"Expected at least 2 incremental append_tool_calls, got {call_count}"
        )

        # Each call should have a single-item list (incremental)
        for call in mock_store.append_tool_calls.call_args_list:
            records = call[0][0]  # First positional arg
            assert len(records) == 1, (
                f"Expected single-item incremental append, got {len(records)}"
            )

    @pytest.mark.asyncio
    async def test_no_callback_without_file_store(self):
        """When file_store is None, no callback is created."""
        tool = _make_domain_tool("search_data")

        llm = MockToolCallingLLM([
            make_tool_call_response("search_data", {"query": "test"}),
            make_tool_call_response("log_finding",
                                    {"category": "test", "content": "result"}),
            make_text_response("ACT done"),
            make_tool_call_response("submit_report",
                                    {"answer": "Answer", "summary": "Sum"}),
            make_text_response("REPORT done"),
        ])

        runtime = _make_runtime(
            llm=llm,
            adaptive_config=AdaptiveFlowConfig(enabled=True),
        )
        # Ensure no file_store
        runtime._file_store = None

        config = _make_config()
        input_ = _make_input(task="Test", tools=[tool])
        node = _make_node(config=config)
        workflow = _make_workflow()

        # Should complete without errors
        output = await runtime.execute(config, input_, node, workflow)
        assert output is not None


# =========================================================================
# 4. _persist_phase does not duplicate tool calls
# =========================================================================

class TestPersistPhaseNoDuplication:
    """_persist_phase should NOT call append_tool_calls anymore."""

    def test_persist_phase_skips_append_tool_calls(self):
        """_persist_phase writes conversation + memory but not tool calls."""
        llm = MockToolCallingLLM([make_text_response("done")])
        runtime = _make_runtime(llm=llm)

        mock_store = MagicMock()
        mock_store.save_phase_conversation = MagicMock()
        mock_store.append_tool_calls = MagicMock()
        mock_store.save_memory = MagicMock()
        runtime._file_store = mock_store

        from parr.framework_tools import AgentWorkingMemory
        memory = AgentWorkingMemory()
        phase_result = PhaseResult(
            phase=Phase.ACT,
            content="done",
            iterations=2,
            tool_calls_made=[
                {"name": "tool_a", "phase": "act", "success": True},
                {"name": "tool_b", "phase": "act", "success": True},
            ],
        )

        runtime._persist_phase(phase_result, memory)

        # save_phase_conversation should be called
        mock_store.save_phase_conversation.assert_called_once()

        # save_memory should be called
        mock_store.save_memory.assert_called_once()

        # append_tool_calls should NOT be called (handled incrementally)
        mock_store.append_tool_calls.assert_not_called()


# =========================================================================
# 5. Memory persistence during callbacks
# =========================================================================

class TestMemoryPersistence:
    """Memory is persisted incrementally so UI shows findings/todos live."""

    @pytest.mark.asyncio
    async def test_memory_saved_after_each_tool_call(self):
        """save_memory is called after every tool call, not just at phase end."""
        tool = _make_domain_tool("search_data")

        llm = MockToolCallingLLM([
            # Entry: domain tool
            make_tool_call_response("search_data", {"query": "q1"}),
            # ACT: another domain tool + log_finding
            make_tool_call_response("search_data", {"query": "q2"}),
            make_tool_call_response("log_finding",
                                    {"category": "test", "content": "found"}),
            make_text_response("ACT done"),
            # REPORT
            make_tool_call_response("submit_report",
                                    {"answer": "Answer", "summary": "Sum"}),
            make_text_response("done"),
        ])

        runtime = _make_runtime(
            llm=llm,
            adaptive_config=AdaptiveFlowConfig(enabled=True),
        )

        mock_store = MagicMock()
        mock_store.append_tool_calls = MagicMock()
        mock_store.save_memory = MagicMock()
        mock_store.save_phase_conversation = MagicMock()
        mock_store.save_output = MagicMock()
        mock_store.update_agent_status = MagicMock()
        runtime._file_store = mock_store

        config = _make_config()
        input_ = _make_input(task="Test", tools=[tool])
        node = _make_node(config=config)
        workflow = _make_workflow()

        await runtime.execute(config, input_, node, workflow)

        # save_memory should be called at least as many times as tool calls
        # (each tool call triggers incremental save via callback)
        memory_calls = mock_store.save_memory.call_count
        tool_calls = mock_store.append_tool_calls.call_count
        assert memory_calls >= tool_calls, (
            f"save_memory ({memory_calls}) should be called at least as often "
            f"as append_tool_calls ({tool_calls})"
        )
