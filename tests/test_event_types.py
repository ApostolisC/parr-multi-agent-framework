"""Tests for event_types.py — all event factory functions."""
import pytest
from parr.event_types import (
    agent_started,
    agent_completed,
    agent_failed,
    batch_progress,
    batch_started,
    budget_exceeded,
    budget_warning,
    llm_call_completed,
    phase_completed,
    phase_started,
    spawn_started,
    spawn_validation,
    tool_executed,
)


class TestLifecycleEvents:
    def test_agent_started(self):
        ev = agent_started("w1", "t1", "a1", role="analyst", depth=0)
        assert ev.event_type == "agent_started"
        assert ev.data["role"] == "analyst"
        assert ev.data["depth"] == 0

    def test_agent_completed(self):
        ev = agent_completed("w1", "t1", "a1", summary="Done", token_usage={"input": 100})
        assert ev.event_type == "agent_completed"

    def test_agent_failed(self):
        ev = agent_failed("w1", "t1", "a1", reason="boom")
        assert ev.event_type == "agent_failed"
        assert ev.data["reason"] == "boom"


class TestPhaseEvents:
    def test_phase_started(self):
        ev = phase_started("w1", "t1", "a1", phase="act")
        assert ev.event_type == "phase_started"
        assert ev.data["phase"] == "act"

    def test_phase_completed(self):
        ev = phase_completed("w1", "t1", "a1", phase="report", iterations=3)
        assert ev.event_type == "phase_completed"
        assert ev.data["iterations"] == 3


class TestBatchEvents:
    def test_batch_started(self):
        ops = [{"op": "spawn_agent"}, {"op": "log_finding"}]
        ev = batch_started("w1", "t1", "a1", "tc1", ops)
        assert ev.event_type == "batch_started"
        assert ev.data["total_ops"] == 2
        assert ev.data["batch_tool_call_id"] == "tc1"
        assert len(ev.data["operations"]) == 2
        assert ev.data["operations"][0]["op"] == "spawn_agent"

    def test_batch_progress(self):
        ev = batch_progress(
            "w1", "t1", "a1", "tc1",
            total_ops=5, completed_ops=2,
            current_op={"tool": "spawn_agent", "success": True},
        )
        assert ev.event_type == "batch_progress"
        assert ev.data["total_ops"] == 5
        assert ev.data["completed_ops"] == 2
        assert ev.data["op"]["tool"] == "spawn_agent"


class TestSpawnEvents:
    def test_spawn_started(self):
        ev = spawn_started(
            "w1", "t1", "a1",
            child_task_id="ct1",
            child_role="analyst",
            child_sub_role="risk",
            child_description="Assess risks",
            blocking=True,
        )
        assert ev.event_type == "spawn_started"
        assert ev.data["child_task_id"] == "ct1"
        assert ev.data["blocking"] is True

    def test_spawn_validation(self):
        ev = spawn_validation(
            "w1", "t1", "a1",
            child_role="analyst",
            child_sub_role="risk",
            child_task="do analysis",
            policy_mode="consult",
            decision="denied",
            reason="Agent has not done enough work",
        )
        assert ev.event_type == "spawn_validation"
        assert ev.data["decision"] == "denied"
        assert ev.data["policy_mode"] == "consult"
        assert ev.data["child_task"][:10] == "do analysi"


class TestEventStructure:
    def test_all_events_have_required_fields(self):
        """Every event must have workflow_id, task_id, agent_id, event_type, timestamp, data."""
        events = [
            agent_started("w", "t", "a", role="r"),
            agent_completed("w", "t", "a", summary="ok", token_usage={}),
            phase_started("w", "t", "a", phase="act"),
            batch_started("w", "t", "a", "tc1", [{"op": "x"}]),
            batch_progress("w", "t", "a", "tc1", 1, 1, {}),
            spawn_started("w", "t", "a", "ct", "r", None, "", True),
            spawn_validation("w", "t", "a", "r", None, "t", "warn", "warned", "ok"),
        ]
        for ev in events:
            d = ev.to_dict()
            assert "workflow_id" in d
            assert "task_id" in d
            assert "agent_id" in d
            assert "event_type" in d
            assert "timestamp" in d
            assert "data" in d
            assert isinstance(d["data"], dict)
