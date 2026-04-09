"""Tests for pluggable agent coordination (AgentCoordinator, message passing, shared state)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from parr.agent_coordinator import AgentCoordinator
from parr.core_types import (
    AgentConfig,
    AgentMessage,
    AgentNode,
    BudgetConfig,
    Phase,
    ToolCall,
    ToolResult,
    WorkflowExecution,
    generate_id,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_workflow_with_tree():
    """Create a workflow with a parent and two children for testing."""
    wf = WorkflowExecution()

    parent = AgentNode(
        task_id="parent-1",
        agent_id="parent-agent",
        config=AgentConfig(role="coordinator"),
        depth=0,
    )
    child_a = AgentNode(
        task_id="child-a",
        agent_id="child-a-agent",
        parent_task_id="parent-1",
        config=AgentConfig(role="analyst"),
        depth=1,
    )
    child_b = AgentNode(
        task_id="child-b",
        agent_id="child-b-agent",
        parent_task_id="parent-1",
        config=AgentConfig(role="reviewer"),
        depth=1,
    )
    parent.children = ["child-a", "child-b"]

    wf.root_task_id = parent.task_id
    wf.agent_tree = {
        "parent-1": parent,
        "child-a": child_a,
        "child-b": child_b,
    }
    return wf, parent, child_a, child_b


# =========================================================================
# 1. AgentMessage dataclass
# =========================================================================

class TestAgentMessage:

    def test_default_values(self):
        msg = AgentMessage()
        assert msg.message_id  # auto-generated
        assert msg.from_task_id == ""
        assert msg.to_task_id == ""
        assert msg.content == ""
        assert msg.message_type == "info"
        assert msg.data == {}

    def test_to_dict(self):
        msg = AgentMessage(
            from_task_id="a",
            to_task_id="b",
            content="hello",
            message_type="data",
            data={"key": "value"},
        )
        d = msg.to_dict()
        assert d["from_task_id"] == "a"
        assert d["to_task_id"] == "b"
        assert d["content"] == "hello"
        assert d["message_type"] == "data"
        assert d["data"] == {"key": "value"}
        assert "timestamp" in d

    def test_custom_message_types(self):
        for mt in ("info", "request", "response", "warning", "data"):
            msg = AgentMessage(message_type=mt)
            assert msg.message_type == mt


# =========================================================================
# 2. AgentCoordinator base — message permission
# =========================================================================

class TestCanSendMessage:

    def test_parent_to_child(self):
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        coord = AgentCoordinator()
        assert coord.can_send_message("parent-1", "child-a", wf) is True
        assert coord.can_send_message("parent-1", "child-b", wf) is True

    def test_child_to_parent(self):
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        coord = AgentCoordinator()
        assert coord.can_send_message("child-a", "parent-1", wf) is True

    def test_sibling_to_sibling(self):
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        coord = AgentCoordinator()
        assert coord.can_send_message("child-a", "child-b", wf) is True
        assert coord.can_send_message("child-b", "child-a", wf) is True

    def test_unknown_sender(self):
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        coord = AgentCoordinator()
        assert coord.can_send_message("unknown", "child-a", wf) is False

    def test_unknown_recipient(self):
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        coord = AgentCoordinator()
        assert coord.can_send_message("parent-1", "unknown", wf) is False

    def test_unrelated_agents(self):
        """Agents with different parents cannot message each other."""
        wf = WorkflowExecution()
        root = AgentNode(task_id="root", depth=0)
        a = AgentNode(task_id="a", parent_task_id="root", depth=1)
        b = AgentNode(task_id="b", parent_task_id="root", depth=1)
        # c is child of a, d is child of b — not siblings
        c = AgentNode(task_id="c", parent_task_id="a", depth=2)
        d = AgentNode(task_id="d", parent_task_id="b", depth=2)
        root.children = ["a", "b"]
        a.children = ["c"]
        b.children = ["d"]
        wf.agent_tree = {"root": root, "a": a, "b": b, "c": c, "d": d}

        coord = AgentCoordinator()
        assert coord.can_send_message("c", "d", wf) is False


# =========================================================================
# 3. AgentCoordinator — send and read messages
# =========================================================================

class TestSendReadMessages:

    def test_send_and_read(self):
        coord = AgentCoordinator()
        msg = coord.send_message("a", "b", "hello", "info")
        assert msg.from_task_id == "a"
        assert msg.to_task_id == "b"
        assert msg.content == "hello"

        messages = coord.read_messages("b")
        assert len(messages) == 1
        assert messages[0].content == "hello"

    def test_read_with_cursor(self):
        coord = AgentCoordinator()
        coord.send_message("a", "b", "msg1")
        coord.send_message("a", "b", "msg2")
        coord.send_message("a", "b", "msg3")

        first_batch = coord.read_messages("b", since_index=0)
        assert len(first_batch) == 3

        second_batch = coord.read_messages("b", since_index=2)
        assert len(second_batch) == 1
        assert second_batch[0].content == "msg3"

        empty_batch = coord.read_messages("b", since_index=3)
        assert len(empty_batch) == 0

    def test_read_empty_mailbox(self):
        coord = AgentCoordinator()
        messages = coord.read_messages("nonexistent")
        assert messages == []

    def test_send_with_data(self):
        coord = AgentCoordinator()
        msg = coord.send_message("a", "b", "data payload", "data", {"key": 42})
        assert msg.data == {"key": 42}

        messages = coord.read_messages("b")
        assert messages[0].data == {"key": 42}

    def test_multiple_recipients(self):
        coord = AgentCoordinator()
        coord.send_message("a", "b", "for b")
        coord.send_message("a", "c", "for c")

        b_msgs = coord.read_messages("b")
        c_msgs = coord.read_messages("c")
        assert len(b_msgs) == 1
        assert len(c_msgs) == 1
        assert b_msgs[0].content == "for b"
        assert c_msgs[0].content == "for c"

    def test_on_message_sent_hook(self):
        sent = []

        class TrackingCoordinator(AgentCoordinator):
            def on_message_sent(self, message):
                sent.append(message)

        coord = TrackingCoordinator()
        coord.send_message("a", "b", "tracked")
        assert len(sent) == 1
        assert sent[0].content == "tracked"


# =========================================================================
# 4. AgentCoordinator — shared state
# =========================================================================

class TestSharedState:

    def test_set_and_get(self):
        coord = AgentCoordinator()
        coord.set_shared_state("wf-1", "risk_level", "high", "agent-a")
        assert coord.get_shared_state("wf-1", "risk_level") == "high"

    def test_get_all(self):
        coord = AgentCoordinator()
        coord.set_shared_state("wf-1", "k1", "v1", "agent-a")
        coord.set_shared_state("wf-1", "k2", "v2", "agent-b")

        all_state = coord.get_shared_state("wf-1")
        assert all_state == {"k1": "v1", "k2": "v2"}

    def test_get_missing_key(self):
        coord = AgentCoordinator()
        assert coord.get_shared_state("wf-1", "missing") is None

    def test_get_missing_workflow(self):
        coord = AgentCoordinator()
        assert coord.get_shared_state("nonexistent") == {}
        assert coord.get_shared_state("nonexistent", "key") is None

    def test_overwrite(self):
        coord = AgentCoordinator()
        coord.set_shared_state("wf-1", "key", "old", "a")
        coord.set_shared_state("wf-1", "key", "new", "b")
        assert coord.get_shared_state("wf-1", "key") == "new"

    def test_separate_workflows(self):
        coord = AgentCoordinator()
        coord.set_shared_state("wf-1", "key", "val-1", "a")
        coord.set_shared_state("wf-2", "key", "val-2", "b")
        assert coord.get_shared_state("wf-1", "key") == "val-1"
        assert coord.get_shared_state("wf-2", "key") == "val-2"

    def test_complex_values(self):
        coord = AgentCoordinator()
        coord.set_shared_state("wf-1", "findings", [{"risk": "high", "count": 3}], "a")
        assert coord.get_shared_state("wf-1", "findings") == [{"risk": "high", "count": 3}]

    def test_can_access_state_default(self):
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        coord = AgentCoordinator()
        assert coord.can_access_state("parent-1", "key", "read", wf) is True
        assert coord.can_access_state("child-a", "key", "write", wf) is True

    def test_can_access_state_unknown_agent(self):
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        coord = AgentCoordinator()
        assert coord.can_access_state("unknown", "key", "read", wf) is False

    def test_on_state_changed_hook(self):
        changes = []

        class TrackingCoordinator(AgentCoordinator):
            def on_state_changed(self, workflow_id, key, value, changed_by):
                changes.append((workflow_id, key, value, changed_by))

        coord = TrackingCoordinator()
        coord.set_shared_state("wf-1", "k", "v", "agent-a")
        assert len(changes) == 1
        assert changes[0] == ("wf-1", "k", "v", "agent-a")


# =========================================================================
# 5. AgentCoordinator — lifecycle helpers
# =========================================================================

class TestLifecycleHelpers:

    def test_clear_workflow(self):
        coord = AgentCoordinator()
        coord.set_shared_state("wf-1", "key", "val", "a")
        coord.clear_workflow("wf-1")
        assert coord.get_shared_state("wf-1") == {}

    def test_clear_agent(self):
        coord = AgentCoordinator()
        coord.send_message("a", "b", "msg")
        coord.clear_agent("b")
        assert coord.read_messages("b") == []


# =========================================================================
# 6. Custom coordinator (subclass)
# =========================================================================

class TestCustomCoordinator:

    def test_restrict_messaging(self):
        """Only allow parent -> child, not sibling -> sibling."""
        class StrictCoordinator(AgentCoordinator):
            def can_send_message(self, from_id, to_id, workflow):
                from_node = workflow.agent_tree.get(from_id)
                to_node = workflow.agent_tree.get(to_id)
                if not from_node or not to_node:
                    return False
                # Only parent -> child
                return to_id in from_node.children

        wf, parent, child_a, child_b = _make_workflow_with_tree()
        coord = StrictCoordinator()

        assert coord.can_send_message("parent-1", "child-a", wf) is True
        assert coord.can_send_message("child-a", "parent-1", wf) is False
        assert coord.can_send_message("child-a", "child-b", wf) is False

    def test_restrict_state_access(self):
        """Only allow root agent to write."""
        class RootWriteCoordinator(AgentCoordinator):
            def can_access_state(self, task_id, key, operation, workflow):
                if operation == "write":
                    return task_id == workflow.root_task_id
                return task_id in workflow.agent_tree

        wf, parent, child_a, child_b = _make_workflow_with_tree()
        coord = RootWriteCoordinator()

        assert coord.can_access_state("parent-1", "key", "write", wf) is True
        assert coord.can_access_state("child-a", "key", "write", wf) is False
        assert coord.can_access_state("child-a", "key", "read", wf) is True

    def test_key_namespaced_access(self):
        """Only allow agents to write to their own namespace."""
        class NamespacedCoordinator(AgentCoordinator):
            def can_access_state(self, task_id, key, operation, workflow):
                if operation == "write":
                    return key.startswith(f"{task_id}:")
                return True

        wf, parent, child_a, child_b = _make_workflow_with_tree()
        coord = NamespacedCoordinator()

        assert coord.can_access_state("child-a", "child-a:data", "write", wf) is True
        assert coord.can_access_state("child-a", "child-b:data", "write", wf) is False
        assert coord.can_access_state("child-a", "child-b:data", "read", wf) is True


# =========================================================================
# 7. Orchestrator integration — send_message handler
# =========================================================================

class TestOrchestratorSendMessage:

    def _make_orchestrator(self, coordinator=None):
        from parr.orchestrator import Orchestrator

        llm = MagicMock()
        return Orchestrator(llm=llm, coordinator=coordinator)

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        from parr.orchestrator import Orchestrator
        from parr.trace_store import TraceStore

        orch = self._make_orchestrator()
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        trace = TraceStore()

        tc = ToolCall(
            id="tc-1", name="send_message",
            arguments={"to_task_id": "child-a", "content": "hello child"},
        )
        result = await orch._handle_orchestrator_tool(
            tc, parent, wf, trace, "",
        )
        assert result.success is True
        data = json.loads(result.content)
        assert data["to_task_id"] == "child-a"
        assert data["status"] == "delivered"

    @pytest.mark.asyncio
    async def test_send_message_unknown_recipient(self):
        from parr.trace_store import TraceStore

        orch = self._make_orchestrator()
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        trace = TraceStore()

        tc = ToolCall(
            id="tc-1", name="send_message",
            arguments={"to_task_id": "nonexistent", "content": "hello"},
        )
        result = await orch._handle_orchestrator_tool(
            tc, parent, wf, trace, "",
        )
        assert result.success is False
        assert "No agent found" in result.error

    @pytest.mark.asyncio
    async def test_send_message_permission_denied(self):
        from parr.trace_store import TraceStore

        # Unrelated agents
        wf = WorkflowExecution()
        root = AgentNode(task_id="root", depth=0)
        a = AgentNode(task_id="a", parent_task_id="root", depth=1)
        b = AgentNode(task_id="b", parent_task_id="root", depth=1)
        c = AgentNode(task_id="c", parent_task_id="a", depth=2)
        d = AgentNode(task_id="d", parent_task_id="b", depth=2)
        root.children = ["a", "b"]
        a.children = ["c"]
        b.children = ["d"]
        wf.agent_tree = {"root": root, "a": a, "b": b, "c": c, "d": d}

        orch = self._make_orchestrator()
        trace = TraceStore()

        tc = ToolCall(
            id="tc-1", name="send_message",
            arguments={"to_task_id": "d", "content": "hello"},
        )
        result = await orch._handle_orchestrator_tool(
            tc, c, wf, trace, "",
        )
        assert result.success is False
        assert "Cannot send message" in result.error

    @pytest.mark.asyncio
    async def test_send_message_empty_content(self):
        from parr.trace_store import TraceStore

        orch = self._make_orchestrator()
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        trace = TraceStore()

        tc = ToolCall(
            id="tc-1", name="send_message",
            arguments={"to_task_id": "child-a", "content": ""},
        )
        result = await orch._handle_orchestrator_tool(
            tc, parent, wf, trace, "",
        )
        assert result.success is False
        assert "non-empty" in result.error


# =========================================================================
# 8. Orchestrator integration — read_messages handler
# =========================================================================

class TestOrchestratorReadMessages:

    def _make_orchestrator(self, coordinator=None):
        from parr.orchestrator import Orchestrator

        llm = MagicMock()
        return Orchestrator(llm=llm, coordinator=coordinator)

    @pytest.mark.asyncio
    async def test_read_messages_empty(self):
        from parr.trace_store import TraceStore

        orch = self._make_orchestrator()
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        trace = TraceStore()

        tc = ToolCall(id="tc-1", name="read_messages", arguments={})
        result = await orch._handle_orchestrator_tool(
            tc, child_a, wf, trace, "",
        )
        assert result.success is True
        assert "No new messages" in result.content

    @pytest.mark.asyncio
    async def test_read_messages_with_cursor(self):
        from parr.trace_store import TraceStore

        orch = self._make_orchestrator()
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        trace = TraceStore()

        # Send two messages to child_a
        send1 = ToolCall(
            id="s1", name="send_message",
            arguments={"to_task_id": "child-a", "content": "msg1"},
        )
        send2 = ToolCall(
            id="s2", name="send_message",
            arguments={"to_task_id": "child-a", "content": "msg2"},
        )
        await orch._handle_orchestrator_tool(send1, parent, wf, trace, "")
        await orch._handle_orchestrator_tool(send2, parent, wf, trace, "")

        # First read — gets both
        read1 = ToolCall(id="r1", name="read_messages", arguments={})
        result1 = await orch._handle_orchestrator_tool(
            read1, child_a, wf, trace, "",
        )
        messages = json.loads(result1.content)
        assert len(messages) == 2

        # Second read — no new messages
        read2 = ToolCall(id="r2", name="read_messages", arguments={})
        result2 = await orch._handle_orchestrator_tool(
            read2, child_a, wf, trace, "",
        )
        assert "No new messages" in result2.content

        # Send another — third read gets only the new one
        send3 = ToolCall(
            id="s3", name="send_message",
            arguments={"to_task_id": "child-a", "content": "msg3"},
        )
        await orch._handle_orchestrator_tool(send3, parent, wf, trace, "")

        read3 = ToolCall(id="r3", name="read_messages", arguments={})
        result3 = await orch._handle_orchestrator_tool(
            read3, child_a, wf, trace, "",
        )
        messages3 = json.loads(result3.content)
        assert len(messages3) == 1
        assert messages3[0]["content"] == "msg3"


# =========================================================================
# 9. Orchestrator integration — shared state handlers
# =========================================================================

class TestOrchestratorSharedState:

    def _make_orchestrator(self, coordinator=None):
        from parr.orchestrator import Orchestrator

        llm = MagicMock()
        return Orchestrator(llm=llm, coordinator=coordinator)

    @pytest.mark.asyncio
    async def test_set_and_get_shared_state(self):
        from parr.trace_store import TraceStore

        orch = self._make_orchestrator()
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        trace = TraceStore()

        # Set state
        set_tc = ToolCall(
            id="tc-1", name="set_shared_state",
            arguments={"key": "risk_level", "value": "high"},
        )
        set_result = await orch._handle_orchestrator_tool(
            set_tc, parent, wf, trace, "",
        )
        assert set_result.success is True
        data = json.loads(set_result.content)
        assert data["key"] == "risk_level"
        assert data["status"] == "stored"

        # Get state (by key)
        get_tc = ToolCall(
            id="tc-2", name="get_shared_state",
            arguments={"key": "risk_level"},
        )
        get_result = await orch._handle_orchestrator_tool(
            get_tc, child_a, wf, trace, "",
        )
        assert get_result.success is True
        data = json.loads(get_result.content)
        assert data["key"] == "risk_level"
        assert data["value"] == "high"

    @pytest.mark.asyncio
    async def test_get_all_shared_state(self):
        from parr.trace_store import TraceStore

        orch = self._make_orchestrator()
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        trace = TraceStore()

        # Set multiple keys
        for key, val in [("k1", "v1"), ("k2", "v2")]:
            tc = ToolCall(
                id=f"tc-{key}", name="set_shared_state",
                arguments={"key": key, "value": val},
            )
            await orch._handle_orchestrator_tool(tc, parent, wf, trace, "")

        # Get all
        get_tc = ToolCall(id="tc-get", name="get_shared_state", arguments={})
        result = await orch._handle_orchestrator_tool(
            get_tc, child_a, wf, trace, "",
        )
        data = json.loads(result.content)
        assert data["state"]["k1"] == "v1"
        assert data["state"]["k2"] == "v2"

    @pytest.mark.asyncio
    async def test_get_missing_key(self):
        from parr.trace_store import TraceStore

        orch = self._make_orchestrator()
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        trace = TraceStore()

        get_tc = ToolCall(
            id="tc-1", name="get_shared_state",
            arguments={"key": "nonexistent"},
        )
        result = await orch._handle_orchestrator_tool(
            get_tc, parent, wf, trace, "",
        )
        data = json.loads(result.content)
        assert data["exists"] is False
        assert data["value"] is None

    @pytest.mark.asyncio
    async def test_set_empty_key(self):
        from parr.trace_store import TraceStore

        orch = self._make_orchestrator()
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        trace = TraceStore()

        tc = ToolCall(
            id="tc-1", name="set_shared_state",
            arguments={"key": "", "value": "test"},
        )
        result = await orch._handle_orchestrator_tool(
            tc, parent, wf, trace, "",
        )
        assert result.success is False
        assert "non-empty" in result.error

    @pytest.mark.asyncio
    async def test_custom_coordinator_blocks_write(self):
        from parr.trace_store import TraceStore

        class ReadOnlyCoordinator(AgentCoordinator):
            def can_access_state(self, task_id, key, operation, workflow):
                if operation == "write" and task_id != workflow.root_task_id:
                    return False
                return True

        orch = self._make_orchestrator(coordinator=ReadOnlyCoordinator())
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        trace = TraceStore()

        # Parent can write
        tc = ToolCall(
            id="tc-1", name="set_shared_state",
            arguments={"key": "data", "value": "from_parent"},
        )
        result = await orch._handle_orchestrator_tool(
            tc, parent, wf, trace, "",
        )
        assert result.success is True

        # Child cannot write
        tc2 = ToolCall(
            id="tc-2", name="set_shared_state",
            arguments={"key": "data", "value": "from_child"},
        )
        result2 = await orch._handle_orchestrator_tool(
            tc2, child_a, wf, trace, "",
        )
        assert result2.success is False
        assert "Access denied" in result2.error


# =========================================================================
# 10. Orchestrator constructor integration
# =========================================================================

class TestOrchestratorConstructor:

    def test_default_coordinator(self):
        from parr.orchestrator import Orchestrator

        llm = MagicMock()
        orch = Orchestrator(llm=llm)
        assert isinstance(orch._coordinator, AgentCoordinator)

    def test_custom_coordinator(self):
        from parr.orchestrator import Orchestrator

        class MyCoordinator(AgentCoordinator):
            pass

        llm = MagicMock()
        coord = MyCoordinator()
        orch = Orchestrator(llm=llm, coordinator=coord)
        assert orch._coordinator is coord


# =========================================================================
# 11. Framework tools registration
# =========================================================================

class TestCoordinationTools:

    def test_build_coordination_tools(self):
        from parr.framework_tools import build_coordination_tools

        tools = build_coordination_tools()
        names = {t.name for t in tools}
        assert names == {"send_message", "read_messages", "set_shared_state", "get_shared_state"}

    def test_all_are_orchestrator_tools(self):
        from parr.framework_tools import build_coordination_tools

        for tool in build_coordination_tools():
            assert tool.is_orchestrator_tool is True
            assert tool.is_framework_tool is True

    def test_read_tools_are_read_only(self):
        from parr.framework_tools import build_coordination_tools

        tools = {t.name: t for t in build_coordination_tools()}
        assert tools["read_messages"].is_read_only is True
        assert tools["get_shared_state"].is_read_only is True

    def test_phase_availability(self):
        from parr.framework_tools import build_coordination_tools

        tools = {t.name: t for t in build_coordination_tools()}
        # Write tools: ACT only
        assert tools["send_message"].phase_availability == [Phase.ACT]
        assert tools["set_shared_state"].phase_availability == [Phase.ACT]
        # Read tools: all phases
        all_phases = [Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT]
        assert tools["read_messages"].phase_availability == all_phases
        assert tools["get_shared_state"].phase_availability == all_phases


# =========================================================================
# 12. Edge cases
# =========================================================================

class TestEdgeCases:

    def test_message_to_self(self):
        """Agent cannot message itself (not parent/child/sibling)."""
        wf = WorkflowExecution()
        node = AgentNode(task_id="a", depth=0)
        wf.agent_tree = {"a": node}

        coord = AgentCoordinator()
        # Self is not in own children, not own parent, no shared parent
        assert coord.can_send_message("a", "a", wf) is False

    def test_shared_state_none_value(self):
        coord = AgentCoordinator()
        coord.set_shared_state("wf-1", "key", None, "a")
        # Value is None but key exists — get_shared_state returns None
        assert coord.get_shared_state("wf-1", "key") is None
        # But the key IS in the store
        all_state = coord.get_shared_state("wf-1")
        assert "key" in all_state

    def test_message_ordering(self):
        """Messages are ordered by send time."""
        coord = AgentCoordinator()
        for i in range(5):
            coord.send_message("a", "b", f"msg-{i}")

        messages = coord.read_messages("b")
        assert [m.content for m in messages] == [
            "msg-0", "msg-1", "msg-2", "msg-3", "msg-4"
        ]

    def test_coordinator_clear_does_not_affect_other_workflows(self):
        coord = AgentCoordinator()
        coord.set_shared_state("wf-1", "key", "val1", "a")
        coord.set_shared_state("wf-2", "key", "val2", "b")
        coord.clear_workflow("wf-1")
        assert coord.get_shared_state("wf-1") == {}
        assert coord.get_shared_state("wf-2", "key") == "val2"

    @pytest.mark.asyncio
    async def test_send_message_with_data_payload(self):
        from parr.orchestrator import Orchestrator
        from parr.trace_store import TraceStore

        llm = MagicMock()
        orch = Orchestrator(llm=llm)
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        trace = TraceStore()

        tc = ToolCall(
            id="tc-1", name="send_message",
            arguments={
                "to_task_id": "child-a",
                "content": "risk data",
                "message_type": "data",
                "data": {"risks": [{"severity": "high"}]},
            },
        )
        result = await orch._handle_orchestrator_tool(
            tc, parent, wf, trace, "",
        )
        assert result.success is True

        # Read and verify data is preserved
        read_tc = ToolCall(id="r1", name="read_messages", arguments={})
        read_result = await orch._handle_orchestrator_tool(
            read_tc, child_a, wf, trace, "",
        )
        messages = json.loads(read_result.content)
        assert messages[0]["data"]["risks"][0]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_set_shared_state_complex_value(self):
        from parr.orchestrator import Orchestrator
        from parr.trace_store import TraceStore

        llm = MagicMock()
        orch = Orchestrator(llm=llm)
        wf, parent, child_a, child_b = _make_workflow_with_tree()
        trace = TraceStore()

        tc = ToolCall(
            id="tc-1", name="set_shared_state",
            arguments={"key": "findings", "value": [1, 2, {"nested": True}]},
        )
        result = await orch._handle_orchestrator_tool(
            tc, parent, wf, trace, "",
        )
        assert result.success is True

        get_tc = ToolCall(
            id="tc-2", name="get_shared_state",
            arguments={"key": "findings"},
        )
        get_result = await orch._handle_orchestrator_tool(
            get_tc, parent, wf, trace, "",
        )
        data = json.loads(get_result.content)
        assert data["value"] == [1, 2, {"nested": True}]
