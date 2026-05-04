"""Tests for spawn policies — deny, warn, consult modes.

These tests exercise the policy logic in the Orchestrator without
making any LLM calls. The mock LLM returns scripted responses.
"""
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
    WorkflowExecution,
    generate_id,
)
from parr.orchestrator import Orchestrator
from parr.event_bus import EventBus, InMemoryEventSink
from parr.trace_store import TraceStore

from tests.conftest import MockLLM, make_budget


def _make_orchestrator(policy: SpawnPolicy) -> Orchestrator:
    return Orchestrator(
        llm=MockLLM(),
        policies_config=PoliciesConfig(same_role_spawn_policy=policy),
        default_budget=make_budget(),
    )


def _make_spawn_call(role: str, sub_role: str = None, task: str = "test") -> ToolCall:
    args = {"role": role, "task_description": task}
    if sub_role:
        args["sub_role"] = sub_role
    args["blocking"] = False  # non-blocking for policy tests
    return ToolCall(id=generate_id(), name="spawn_agent", arguments=args)


def _make_parent_node(role: str, sub_role: str = None, depth: int = 0) -> AgentNode:
    return AgentNode(
        agent_id=generate_id(),
        config=AgentConfig(role=role, sub_role=sub_role, system_prompt="test"),
        budget=make_budget(),
        depth=depth,
    )


class TestSameRoleDenyPolicy:
    @pytest.mark.asyncio
    async def test_same_role_blocked(self):
        orch = _make_orchestrator(SpawnPolicy.DENY)
        parent = _make_parent_node("analyst", "risk")
        wf = WorkflowExecution()
        wf.agent_tree[parent.task_id] = parent
        ts = TraceStore()

        call = _make_spawn_call("analyst", "risk")
        result = await orch._handle_spawn_agent(
            call, parent, wf, ts, "", None,
        )

        assert not result.success
        assert "same role" in result.error.lower()

    @pytest.mark.asyncio
    async def test_different_sub_role_allowed(self):
        """Same role but different sub_role should NOT trigger policy."""
        orch = _make_orchestrator(SpawnPolicy.DENY)
        parent = _make_parent_node("analyst", "risk")
        wf = WorkflowExecution()
        wf.agent_tree[parent.task_id] = parent
        ts = TraceStore()

        call = _make_spawn_call("analyst", "controls")
        result = await orch._handle_spawn_agent(
            call, parent, wf, ts, "", None,
        )
        # Should succeed (or fail for another reason, but NOT same-role)
        if not result.success:
            assert "same role" not in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_sub_role_vs_sub_role_allowed(self):
        """Parent has no sub_role, child has sub_role — not same role."""
        orch = _make_orchestrator(SpawnPolicy.DENY)
        parent = _make_parent_node("analyst", None)
        wf = WorkflowExecution()
        wf.agent_tree[parent.task_id] = parent
        ts = TraceStore()

        call = _make_spawn_call("analyst", "risk")
        result = await orch._handle_spawn_agent(
            call, parent, wf, ts, "", None,
        )
        if not result.success:
            assert "same role" not in result.error.lower()


class TestSameRoleWarnPolicy:
    @pytest.mark.asyncio
    async def test_same_role_not_blocked(self):
        """Warn mode should NOT block same-role spawns (unlike deny mode).

        The spawn may still fail for other reasons (no domain adapter to
        resolve the role), but the error should NOT mention same-role policy.
        """
        orch = _make_orchestrator(SpawnPolicy.WARN)
        parent = _make_parent_node("analyst", "risk")
        parent.current_phase = Phase.ACT
        wf = WorkflowExecution()
        wf.agent_tree[parent.task_id] = parent
        ts = TraceStore()

        call = _make_spawn_call("analyst", "risk")
        result = await orch._handle_spawn_agent(
            call, parent, wf, ts, "", None,
        )

        # May fail for role resolution, but NOT for same-role policy
        if not result.success:
            assert "same role" not in result.error.lower()
            assert "cannot spawn" not in result.error.lower()


class TestDepthLimitEnforcement:
    @pytest.mark.asyncio
    async def test_depth_limit_blocks_spawn(self):
        orch = _make_orchestrator(SpawnPolicy.WARN)
        parent = _make_parent_node("analyst", "risk", depth=3)
        parent.budget = make_budget(max_agent_depth=4)
        parent.current_phase = Phase.ACT
        wf = WorkflowExecution()
        wf.agent_tree[parent.task_id] = parent
        ts = TraceStore()

        call = _make_spawn_call("other_role")
        result = await orch._handle_spawn_agent(
            call, parent, wf, ts, "", None,
        )

        assert not result.success
        assert "depth" in result.error.lower()


class TestTotalChildrenLimit:
    @pytest.mark.asyncio
    async def test_max_children_blocks_spawn(self):
        orch = _make_orchestrator(SpawnPolicy.WARN)
        parent = _make_parent_node("analyst")
        parent.budget = make_budget(max_sub_agents_total=2)
        parent.current_phase = Phase.ACT
        parent.children = ["child1", "child2"]  # already at max
        wf = WorkflowExecution()
        wf.agent_tree[parent.task_id] = parent
        ts = TraceStore()

        call = _make_spawn_call("other_role")
        result = await orch._handle_spawn_agent(
            call, parent, wf, ts, "", None,
        )

        assert not result.success
        assert "sub-agents" in result.error.lower() or "maximum" in result.error.lower()
