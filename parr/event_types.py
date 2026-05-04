"""
Framework Event Types.

Structured event definitions for real-time observability. The orchestrator
publishes these events through the EventSink protocol. The application layer
subscribes to receive them via WebSocket, SSE, or any transport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------

@dataclass
class FrameworkEvent:
    """Base event emitted by the framework."""
    workflow_id: str
    task_id: str
    agent_id: str
    event_type: str
    timestamp: str = field(default_factory=_utc_now_iso)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "data": self.data,
        }


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

def agent_started(
    workflow_id: str, task_id: str, agent_id: str,
    role: str, sub_role: Optional[str] = None, depth: int = 0,
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="agent_started",
        data={"role": role, "sub_role": sub_role, "depth": depth},
    )


def agent_completed(
    workflow_id: str, task_id: str, agent_id: str,
    summary: str, token_usage: Dict[str, int],
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="agent_completed",
        data={"summary": summary, "token_usage": token_usage},
    )


def agent_failed(
    workflow_id: str, task_id: str, agent_id: str,
    reason: str,
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="agent_failed",
        data={"reason": reason},
    )


def agent_cancelled(
    workflow_id: str, task_id: str, agent_id: str,
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="agent_cancelled",
    )


def agent_suspended(
    workflow_id: str, task_id: str, agent_id: str,
    waiting_for: List[str],
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="agent_suspended",
        data={"waiting_for": waiting_for},
    )


def agent_resumed(
    workflow_id: str, task_id: str, agent_id: str,
    results_received: List[str],
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="agent_resumed",
        data={"results_received": results_received},
    )


# ---------------------------------------------------------------------------
# Phase events
# ---------------------------------------------------------------------------

def phase_started(
    workflow_id: str, task_id: str, agent_id: str,
    phase: str, context: str | None = None,
) -> FrameworkEvent:
    data: dict[str, Any] = {"phase": phase}
    if context:
        data["context"] = context
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="phase_started",
        data=data,
    )


def phase_completed(
    workflow_id: str, task_id: str, agent_id: str,
    phase: str, iterations: int = 0,
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="phase_completed",
        data={"phase": phase, "iterations": iterations},
    )


def review_override(
    workflow_id: str, task_id: str, agent_id: str,
    retry_count: int, reason: str,
    failed_criteria: list[dict] | None = None,
) -> FrameworkEvent:
    """Emitted when review did not pass but the agent proceeds to Report."""
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="review_override",
        data={
            "phase": "review",
            "retry_count": retry_count,
            "reason": reason,
            "failed_criteria": failed_criteria or [],
        },
    )


def phase_iteration_limit(
    workflow_id: str, task_id: str, agent_id: str,
    phase: str, limit: int,
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="phase_iteration_limit",
        data={"phase": phase, "limit": limit},
    )


def phase_injection(
    workflow_id: str, task_id: str, agent_id: str,
    phase: str, kind: str, content: str,
    iteration: int | None = None,
) -> FrameworkEvent:
    """Emitted when the framework injects a user-role message into the
    agent's conversation mid-phase (iteration advisories, warnings,
    budget nudges, stall warnings, mandatory-tool nudges, circuit-breaker
    messages, or the review-retry feedback passed via ``extra_context``).

    ``kind`` is a short machine-readable label (e.g.
    ``iteration_advisory``, ``iteration_warning``, ``progress``,
    ``budget_warning``, ``stall_warning``, ``tool_warning``,
    ``tool_disabled``, ``mandatory_nudge``, ``extra_context``).
    """
    data: Dict[str, Any] = {
        "phase": phase, "kind": kind, "content": content,
    }
    if iteration is not None:
        data["iteration"] = iteration
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="phase_injection",
        data=data,
    )


# ---------------------------------------------------------------------------
# Execution events
# ---------------------------------------------------------------------------

def llm_call_completed(
    workflow_id: str, task_id: str, agent_id: str,
    phase: str, iteration: int,
    input_tokens: int, output_tokens: int,
    response_content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    cumulative_tokens: int = 0,
    model: Optional[str] = None,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cost: Optional[float] = None,
) -> FrameworkEvent:
    data: Dict[str, Any] = {
        "phase": phase, "iteration": iteration,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if response_content is not None:
        data["response_content"] = response_content
    if tool_calls is not None:
        data["tool_calls"] = tool_calls
    if cumulative_tokens > 0:
        data["cumulative_tokens"] = cumulative_tokens
    if model is not None:
        data["model"] = model
    # Cache metrics — emitted only when non-zero so the event payload
    # stays compact when caching is disabled or unsupported.
    if cache_read_input_tokens:
        data["cache_read_input_tokens"] = cache_read_input_tokens
    if cache_creation_input_tokens:
        data["cache_creation_input_tokens"] = cache_creation_input_tokens
    # The provider-aware cost computed by CostConfig (cache-discounted
    # when applicable). Subscribers can use this directly instead of
    # re-deriving cost from raw tokens.
    if cost is not None:
        data["cost"] = cost
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="llm_call_completed",
        data=data,
    )


def tool_executed(
    workflow_id: str, task_id: str, agent_id: str,
    phase: str, tool_name: str, success: bool,
    error: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
    result_content: Optional[str] = None,
) -> FrameworkEvent:
    data: Dict[str, Any] = {
        "phase": phase, "tool": tool_name, "success": success, "error": error,
    }
    if arguments is not None:
        data["arguments"] = arguments
    if result_content is not None:
        data["result_content"] = result_content
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="tool_executed",
        data=data,
    )


def context_compacted(
    workflow_id: str, task_id: str, agent_id: str,
    phase: str,
    compaction_type: str = "soft",
    before_tokens: int = 0,
    after_tokens: int = 0,
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="context_compacted",
        data={
            "phase": phase,
            "compaction_type": compaction_type,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
        },
    )


# ---------------------------------------------------------------------------
# Streaming events
# ---------------------------------------------------------------------------

def agent_token(
    workflow_id: str, task_id: str, agent_id: str,
    phase: str, token: str,
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="agent_token",
        data={"phase": phase, "token": token},
    )


def agent_thinking(
    workflow_id: str, task_id: str, agent_id: str,
    content: str,
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="agent_thinking",
        data={"content": content},
    )


# ---------------------------------------------------------------------------
# Budget events
# ---------------------------------------------------------------------------

def budget_warning(
    workflow_id: str, task_id: str, agent_id: str,
    consumed_tokens: int, max_tokens: Optional[int],
    consumed_cost: float, max_cost: Optional[float],
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="budget_warning",
        data={
            "consumed_tokens": consumed_tokens, "max_tokens": max_tokens,
            "consumed_cost": consumed_cost, "max_cost": max_cost,
        },
    )


def budget_exceeded(
    workflow_id: str, task_id: str, agent_id: str,
    reason: str,
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="budget_exceeded",
        data={"reason": reason},
    )


def batch_started(
    workflow_id: str, task_id: str, agent_id: str,
    batch_tool_call_id: str,
    operations: List[Dict[str, Any]],
) -> FrameworkEvent:
    """Emitted when batch_operations begins — before any ops execute."""
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="batch_started",
        data={
            "batch_tool_call_id": batch_tool_call_id,
            "total_ops": len(operations),
            "operations": [
                {"op": op.get("op", ""), "index": i}
                for i, op in enumerate(operations, 1)
            ],
        },
    )


def batch_progress(
    workflow_id: str, task_id: str, agent_id: str,
    batch_tool_call_id: str,
    total_ops: int,
    completed_ops: int,
    current_op: Dict[str, Any],
) -> FrameworkEvent:
    """Emitted each time an op inside batch_operations completes."""
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="batch_progress",
        data={
            "batch_tool_call_id": batch_tool_call_id,
            "total_ops": total_ops,
            "completed_ops": completed_ops,
            "op": current_op,
        },
    )


def spawn_started(
    workflow_id: str, task_id: str, agent_id: str,
    child_task_id: str,
    child_role: str,
    child_sub_role: Optional[str],
    child_description: str,
    blocking: bool,
) -> FrameworkEvent:
    """Emitted when spawn_agent initiates a child — before blocking await."""
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="spawn_started",
        data={
            "child_task_id": child_task_id,
            "child_role": child_role,
            "child_sub_role": child_sub_role,
            "child_description": child_description,
            "blocking": blocking,
        },
    )


def spawn_validation(
    workflow_id: str, task_id: str, agent_id: str,
    child_role: str,
    child_sub_role: Optional[str],
    child_task: str,
    policy_mode: str,
    decision: str,
    reason: str,
) -> FrameworkEvent:
    """Emitted when a same-role spawn attempt triggers policy evaluation."""
    return FrameworkEvent(
        workflow_id=workflow_id, task_id=task_id, agent_id=agent_id,
        event_type="spawn_validation",
        data={
            "child_role": child_role,
            "child_sub_role": child_sub_role,
            "child_task": child_task[:200],
            "policy_mode": policy_mode,
            "decision": decision,
            "reason": reason,
        },
    )
