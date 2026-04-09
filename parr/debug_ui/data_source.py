"""
Data source abstraction for the PARR Debug UI.

Defines the ``UIDataSource`` protocol and a ``FileSystemDataSource``
implementation that reads session data from the persistence directory.
"""

from __future__ import annotations

import json
import logging
import os
import time
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class UIDataSource(Protocol):
    """Interface for providing session data to the debug UI."""

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all workflow sessions with summary info."""
        ...

    def get_session(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get full session detail for a workflow."""
        ...


# ---------------------------------------------------------------------------
# File-system helpers (stateless, module-level)
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Any:
    """Read and parse a JSON file, returning None on failure.

    On Windows, retries on ``PermissionError`` (WinError 5) which can
    occur when another thread is writing to the same file.
    """
    last_perm_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            return None
        except PermissionError as exc:
            last_perm_err = exc
            if os.name != "nt":
                break  # Only retry on Windows
            time.sleep(0.03 * (attempt + 1))  # 30ms, 60ms, 90ms
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read {path}: {e}")
            return None
    if last_perm_err is not None:
        logger.warning(f"Failed to read {path} after retries: {last_perm_err}")
    return None


def _json_text(value: Any) -> str:
    """Serialize arbitrary values to text for lightweight size estimates."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _chars_to_tokens(text_chars: int) -> int:
    """
    Approximate token count from text length.

    Uses ~4 chars/token heuristic, which is sufficient for live UI estimates.
    """
    if text_chars <= 0:
        return 0
    return max(1, ceil(text_chars / 4))


def _infer_current_phase(conv: Dict[str, Any], status: str) -> Optional[str]:
    """Infer the active phase from persisted phase snapshots."""
    phase_order = ["plan", "act", "review", "report"]
    running_like = status in {"running", "spawned", "queued"}
    if not conv:
        return "plan" if running_like else None

    ordered_present = [p for p in phase_order if p in conv]
    if not ordered_present:
        ordered_present = list(conv.keys())
    if not ordered_present:
        return "plan" if running_like else None

    if running_like:
        for p in phase_order:
            if p not in conv:
                return p
            pdata = conv.get(p) or {}
            if pdata.get("iterations", 0) > 0 and not pdata.get("content"):
                return p
            if pdata.get("hit_iteration_limit", False):
                return p

    return ordered_present[-1]


def _estimate_context_metrics(
    info: Dict[str, Any],
    conv: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
    output: Dict[str, Any],
) -> Dict[str, int]:
    """Estimate prompt/context footprint from persisted task, phases, tools, and outputs."""
    context_chars = 0
    output_chars = 0

    task_text = _json_text(info.get("task"))
    context_chars += len(task_text)

    for phase_data in conv.values():
        content_text = _json_text((phase_data or {}).get("content"))
        context_chars += len(content_text)
        output_chars += len(content_text)

    for tc in tool_calls:
        args_text = _json_text(tc.get("arguments"))
        result_text = _json_text(tc.get("result_content") or tc.get("result"))
        error_text = _json_text(tc.get("error"))
        context_chars += len(args_text) + len(result_text) + len(error_text)
        output_chars += len(result_text)

    phase_outputs = ((output.get("execution_metadata") or {}).get("phase_outputs") or {})
    for v in phase_outputs.values():
        v_text = _json_text(v)
        context_chars += len(v_text)
        output_chars += len(v_text)

    return {
        "chars": context_chars,
        "estimated_tokens": _chars_to_tokens(context_chars),
        "estimated_output_tokens": _chars_to_tokens(output_chars),
    }


def _estimate_todo_metrics(
    memory: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute todo counters; fallback to tool-call signal if memory is not yet persisted."""
    todo_items = memory.get("todo_list")
    if isinstance(todo_items, list):
        total = len(todo_items)
        completed = sum(1 for item in todo_items if (item or {}).get("completed"))
        return {
            "total": total,
            "completed": completed,
            "pending": max(0, total - completed),
            "source": "memory",
        }

    todo_tools = {
        "create_todo_list",
        "update_todo_list",
        "get_todo_list",
        "mark_todo_complete",
        "batch_mark_todo_complete",
    }
    seen = [tc for tc in tool_calls if tc.get("name") in todo_tools]
    return {
        "total": 0,
        "completed": 0,
        "pending": 0,
        "source": "tools" if seen else "none",
        "signals": len(seen),
    }


def _derive_activity(
    *,
    info: Dict[str, Any],
    conv: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
    sub_agents: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive working/waiting state and current action hints for live UI."""
    raw_status = (info.get("status") or output.get("status") or "unknown")
    status = str(raw_status).lower()
    current_phase = _infer_current_phase(conv, status)

    running_children = 0
    for child in (sub_agents or {}).values():
        child_status = str(((child.get("info") or {}).get("status") or "")).lower()
        if child_status in {"running", "spawned", "queued"}:
            running_children += 1

    last_tool = tool_calls[-1] if tool_calls else {}
    last_tool_name = last_tool.get("name")

    if status in {"running"}:
        if running_children and last_tool_name in {"wait_for_agents", "get_agent_results"}:
            state = "waiting"
            current_doing = f"waiting for {running_children} sub-agent(s)"
        elif last_tool_name:
            state = "working"
            current_doing = f"calling {last_tool_name}"
        elif current_phase:
            state = "working"
            current_doing = f"executing {current_phase} phase"
        else:
            state = "working"
            current_doing = "initializing"
    elif status in {"spawned", "queued"}:
        state = "waiting"
        current_doing = "queued for execution"
    elif status in {"completed"}:
        state = "done"
        current_doing = "completed"
    elif status in {"failed", "error", "cancelled"}:
        state = "failed"
        current_doing = status
    else:
        state = "idle"
        current_doing = status

    return {
        "state": state,
        "status": status,
        "current_phase": current_phase,
        "current_doing": current_doing,
        "last_tool": last_tool_name,
        "running_children": running_children,
    }


# ---------------------------------------------------------------------------
# Disk reading functions
# ---------------------------------------------------------------------------

def _list_sessions_from_disk(persist_dir: Path) -> List[Dict[str, Any]]:
    """List all workflow sessions in the persistence directory."""
    sessions = []
    if not persist_dir.exists():
        return sessions
    for child in persist_dir.iterdir():
        if not child.is_dir():
            continue
        workflow = _read_json(child / "workflow.json")
        agent = _read_json(child / "agent.json")
        task_text = ""
        if agent:
            raw_task = agent.get("task", "")
            task_text = (raw_task[:120] + "...") if len(raw_task) > 120 else raw_task
        # Prefer workflow status (authoritative) over agent status
        wf_status = (workflow or {}).get("status")
        ag_status = agent.get("status") if agent else None
        effective_status = wf_status or ag_status
        sessions.append({
            "workflow_id": child.name,
            "workflow": workflow,
            "agent_summary": {
                "role": agent.get("role") if agent else None,
                "sub_role": agent.get("sub_role") if agent else None,
                "status": effective_status,
                "task": task_text,
                "model": agent.get("model") if agent else None,
            },
        })
    # Sort by creation time (newest first), falling back to dir name
    sessions.sort(
        key=lambda s: (s.get("workflow") or {}).get("created_at", ""),
        reverse=True,
    )
    return sessions


def _read_agent_tree(agent_dir: Path) -> Dict[str, Any]:
    """Recursively read an agent and all its sub-agents from disk."""
    agent: Dict[str, Any] = {}

    agent["info"] = _read_json(agent_dir / "agent.json")
    agent["conversation"] = _read_json(agent_dir / "conversation.json")
    agent["tool_calls"] = _read_json(agent_dir / "tool_calls.json")
    agent["llm_calls"] = _read_json(agent_dir / "llm_calls.json")
    agent["output"] = _read_json(agent_dir / "output.json")
    agent["sub_agents_summary"] = _read_json(agent_dir / "sub_agents.json")

    # Memory
    memory_dir = agent_dir / "memory"
    if memory_dir.exists():
        memory = {}
        for f in sorted(memory_dir.iterdir()):
            if f.suffix == ".json":
                memory[f.stem] = _read_json(f)
        agent["memory"] = memory
    else:
        agent["memory"] = {}

    # Recursive sub-agents
    sa_dir = agent_dir / "sub_agents"
    if sa_dir.exists() and sa_dir.is_dir():
        children = {}
        for child_dir in sorted(sa_dir.iterdir()):
            if child_dir.is_dir():
                children[child_dir.name] = _read_agent_tree(child_dir)
        agent["sub_agents"] = children
    else:
        agent["sub_agents"] = {}

    # Cross-reference sub_agents_summary to fix stale status.
    # When an agent is killed (e.g. budget exceeded), agent.json may still
    # say "running" but sub_agents.json on the parent has the real status.
    sa_summary = agent.get("sub_agents_summary") or []
    if sa_summary and agent["sub_agents"]:
        summary_by_id = {}
        for entry in sa_summary:
            # Match by task_id (folder name) or agent_id
            tid = entry.get("task_id", "")
            aid = entry.get("agent_id", "")
            summary_by_id[tid] = entry
            if aid:
                summary_by_id[aid] = entry
        for child_key, child in agent["sub_agents"].items():
            child_info = child.get("info") or {}
            match = summary_by_id.get(child_key) or summary_by_id.get(child_info.get("task_id", ""))
            if match and child_info.get("status") == "running" and match.get("status") != "running":
                child_info["status"] = match["status"]

    # Compute metrics from available data
    agent["metrics"] = _compute_agent_metrics(agent)

    return agent


def _compute_agent_metrics(agent: Dict[str, Any]) -> Dict[str, Any]:
    """Derive metrics from persisted agent data for the debug UI."""
    metrics: Dict[str, Any] = {}
    info = agent.get("info") or {}

    # -- Tool stats --
    tool_calls = agent.get("tool_calls") or []
    total_tools = len(tool_calls)
    tool_ok = sum(1 for tc in tool_calls if tc.get("success") is not False)
    tool_fail = total_tools - tool_ok
    tool_names: Dict[str, int] = {}
    for tc in tool_calls:
        n = tc.get("name", "unknown")
        tool_names[n] = tool_names.get(n, 0) + 1
    metrics["tools"] = {
        "total": total_tools,
        "success": tool_ok,
        "failed": tool_fail,
        "by_name": tool_names,
    }

    # -- Phase stats --
    conv = agent.get("conversation") or {}
    total_iterations = 0
    phase_detail = {}
    for phase_name, phase_data in conv.items():
        iters = phase_data.get("iterations", 0)
        tc_count = phase_data.get("tool_calls_count", 0)
        total_iterations += iters
        phase_detail[phase_name] = {
            "iterations": iters,
            "tool_calls": tc_count,
            "hit_limit": phase_data.get("hit_iteration_limit", False),
            "has_content": bool(phase_data.get("content")),
        }
    metrics["phases"] = {
        "completed": list(conv.keys()),
        "total_iterations": total_iterations,
        "detail": phase_detail,
    }

    # -- Token / context / cost / duration --
    output = agent.get("output") or {}
    token_usage = output.get("token_usage") or {}
    exec_meta = output.get("execution_metadata") or {}
    context = _estimate_context_metrics(info, conv, tool_calls, output)

    input_tokens = int(token_usage.get("input_tokens") or 0)
    output_tokens = int(token_usage.get("output_tokens") or 0)
    total_tokens = int(token_usage.get("total_tokens") or (input_tokens + output_tokens))
    estimated = False
    if total_tokens <= 0:
        # No final usage yet (running agent): provide a live estimate.
        estimated = True
        output_tokens = context.get("estimated_output_tokens", 0)
        # Approximate prompt/context tokens as a function of current context.
        input_tokens = max(context.get("estimated_tokens", 0), output_tokens // 2)
        total_tokens = input_tokens + output_tokens

    metrics["tokens"] = {
        "input": input_tokens,
        "output": output_tokens,
        "total": total_tokens,
        "cost": float(token_usage.get("total_cost", 0) or 0),
        "is_estimated": estimated,
    }
    metrics["context"] = {
        "chars": context.get("chars", 0),
        "estimated_tokens": context.get("estimated_tokens", 0),
    }
    metrics["duration_ms"] = exec_meta.get("total_duration_ms", 0)
    metrics["iterations_per_phase"] = exec_meta.get("iterations_per_phase") or {}

    # -- Sub-agent count --
    subs = agent.get("sub_agents") or {}
    sub_summary = agent.get("sub_agents_summary") or []
    metrics["sub_agents"] = {
        "count": max(len(subs), len(sub_summary)),
        "ids": list(subs.keys()),
    }
    metrics["todo"] = _estimate_todo_metrics(agent.get("memory") or {}, tool_calls)
    metrics["activity"] = _derive_activity(
        info=info,
        conv=conv,
        tool_calls=tool_calls,
        sub_agents=subs,
        output=output,
    )

    return metrics


def _aggregate_metrics(agent: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively aggregate metrics across an agent and all sub-agents."""
    m = agent.get("metrics") or {}
    tokens = dict(m.get("tokens") or {"input": 0, "output": 0, "total": 0, "cost": 0})
    tools = dict(m.get("tools") or {"total": 0, "success": 0, "failed": 0})
    context = dict(m.get("context") or {"chars": 0, "estimated_tokens": 0})
    activity_state = ((m.get("activity") or {}).get("state") or "idle")
    agent_states = {
        "working": 1 if activity_state == "working" else 0,
        "waiting": 1 if activity_state == "waiting" else 0,
        "done": 1 if activity_state == "done" else 0,
        "failed": 1 if activity_state == "failed" else 0,
        "idle": 1 if activity_state == "idle" else 0,
    }
    total_iterations = (m.get("phases") or {}).get("total_iterations", 0)
    duration_ms = m.get("duration_ms", 0)
    agent_count = 1

    for _id, child in (agent.get("sub_agents") or {}).items():
        child_agg = _aggregate_metrics(child)
        tokens["input"] += child_agg["tokens"]["input"]
        tokens["output"] += child_agg["tokens"]["output"]
        tokens["total"] += child_agg["tokens"]["total"]
        tokens["cost"] += child_agg["tokens"]["cost"]
        tools["total"] += child_agg["tools"]["total"]
        tools["success"] += child_agg["tools"]["success"]
        tools["failed"] += child_agg["tools"]["failed"]
        context["chars"] += child_agg["context"]["chars"]
        context["estimated_tokens"] += child_agg["context"]["estimated_tokens"]
        for key in agent_states:
            agent_states[key] += child_agg["agent_states"].get(key, 0)
        total_iterations += child_agg["total_iterations"]
        duration_ms = max(duration_ms, child_agg["duration_ms"])
        agent_count += child_agg["agent_count"]

    return {
        "tokens": tokens,
        "tools": tools,
        "context": context,
        "agent_states": agent_states,
        "total_iterations": total_iterations,
        "duration_ms": duration_ms,
        "agent_count": agent_count,
    }


def _read_session_from_disk(
    persist_dir: Path, workflow_id: str,
) -> Optional[Dict[str, Any]]:
    """Read full session data for a workflow."""
    wf_dir = persist_dir / workflow_id
    if not wf_dir.exists():
        return None

    agent_tree = _read_agent_tree(wf_dir)
    workflow = _read_json(wf_dir / "workflow.json")

    # Build global aggregated metrics
    global_metrics = _aggregate_metrics(agent_tree)
    # Attach budget info from workflow.json for progress bars
    budget_cfg = (workflow or {}).get("budget") or (workflow or {}).get("budget_config") or {}
    global_metrics["budget_limits"] = {
        "max_tokens": budget_cfg.get("max_tokens", 0),
        "max_cost": budget_cfg.get("max_cost", 0),
        "max_tool_calls": budget_cfg.get("max_tool_calls", 0),
    }

    return {
        "workflow_id": workflow_id,
        "workflow": workflow,
        "agent_tree": agent_tree,
        "global_metrics": global_metrics,
    }


# ---------------------------------------------------------------------------
# FileSystemDataSource
# ---------------------------------------------------------------------------

class FileSystemDataSource:
    """Reads session data from the framework's persistence directory."""

    def __init__(self, persist_dir: str | Path) -> None:
        self._persist_dir = Path(persist_dir).resolve()
        self._persist_dir.mkdir(parents=True, exist_ok=True)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return _list_sessions_from_disk(self._persist_dir)

    def get_session(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        return _read_session_from_disk(self._persist_dir, workflow_id)

    @property
    def persist_dir(self) -> Path:
        return self._persist_dir
