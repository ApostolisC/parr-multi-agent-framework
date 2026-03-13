"""Tests for debug UI metric derivation helpers."""

from parr.debug_ui.server import _aggregate_metrics, _compute_agent_metrics


def test_compute_agent_metrics_prefers_final_token_usage():
    agent = {
        "info": {"status": "completed", "task": "Analyze market dynamics."},
        "conversation": {
            "plan": {
                "content": "Created an execution plan.",
                "iterations": 1,
                "tool_calls_count": 1,
                "hit_iteration_limit": False,
            },
        },
        "tool_calls": [
            {
                "name": "create_todo_list",
                "phase": "plan",
                "arguments": {"items": [{"description": "Search data"}]},
                "result_content": "Created todo list with 1 items.",
                "success": True,
            },
        ],
        "memory": {
            "todo_list": [
                {"index": 0, "description": "Search data", "completed": True},
            ],
        },
        "output": {
            "status": "completed",
            "token_usage": {
                "input_tokens": 1200,
                "output_tokens": 300,
                "total_tokens": 1500,
                "total_cost": 0.01,
            },
            "execution_metadata": {"total_duration_ms": 2500},
        },
        "sub_agents": {},
        "sub_agents_summary": [],
    }

    metrics = _compute_agent_metrics(agent)

    assert metrics["tokens"]["input"] == 1200
    assert metrics["tokens"]["output"] == 300
    assert metrics["tokens"]["total"] == 1500
    assert metrics["tokens"]["is_estimated"] is False
    assert metrics["context"]["estimated_tokens"] > 0
    assert metrics["activity"]["state"] == "done"
    assert metrics["todo"]["total"] == 1
    assert metrics["todo"]["completed"] == 1


def test_compute_agent_metrics_estimates_live_running_state():
    agent = {
        "info": {"status": "running", "task": "Coordinate sub-agents."},
        "conversation": {
            "plan": {
                "content": "Plan complete.",
                "iterations": 1,
                "tool_calls_count": 1,
                "hit_iteration_limit": False,
            },
            "act": {
                "content": None,
                "iterations": 2,
                "tool_calls_count": 1,
                "hit_iteration_limit": False,
            },
        },
        "tool_calls": [
            {
                "name": "spawn_agent",
                "phase": "act",
                "arguments": {"task": "research"},
                "result_content": "Spawned agent child-1.",
                "success": True,
            },
            {
                "name": "wait_for_agents",
                "phase": "act",
                "arguments": {},
                "result_content": "Waiting for running agents.",
                "success": True,
            },
        ],
        "memory": {},
        "output": {},
        "sub_agents": {"child-1": {"info": {"status": "running"}}},
        "sub_agents_summary": [],
    }

    metrics = _compute_agent_metrics(agent)

    assert metrics["tokens"]["total"] > 0
    assert metrics["tokens"]["is_estimated"] is True
    assert metrics["context"]["estimated_tokens"] > 0
    assert metrics["activity"]["state"] == "waiting"
    assert metrics["activity"]["current_phase"] == "act"
    assert "waiting for 1 sub-agent" in metrics["activity"]["current_doing"]


def test_aggregate_metrics_rolls_up_context_and_activity_counts():
    child = {
        "info": {"status": "completed", "task": "Child task"},
        "conversation": {"plan": {"content": "Child content", "iterations": 1, "tool_calls_count": 0}},
        "tool_calls": [],
        "memory": {},
        "output": {"status": "completed", "token_usage": {"total_tokens": 400}},
        "sub_agents": {},
        "sub_agents_summary": [],
    }
    child["metrics"] = _compute_agent_metrics(child)

    parent = {
        "info": {"status": "running", "task": "Parent task"},
        "conversation": {"plan": {"content": "Parent content", "iterations": 1, "tool_calls_count": 0}},
        "tool_calls": [],
        "memory": {},
        "output": {},
        "sub_agents": {"child": child},
        "sub_agents_summary": [],
    }
    parent["metrics"] = _compute_agent_metrics(parent)

    agg = _aggregate_metrics(parent)

    assert agg["agent_count"] == 2
    assert agg["context"]["estimated_tokens"] >= parent["metrics"]["context"]["estimated_tokens"]
    assert agg["agent_states"]["working"] >= 1
    assert agg["agent_states"]["done"] >= 1
