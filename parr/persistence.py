"""
File-System Persistence for the Agentic Framework.

Provides optional, hierarchical file-system persistence for workflows
and agents. When enabled (via ``persist_dir``), every agent's working
state is saved incrementally to disk, producing a self-contained folder
tree that mirrors the agent hierarchy:

    <persist_dir>/<workflow_id>/
    ├── workflow.json               # Workflow-level metadata
    ├── agent.json                  # Root agent config, task, status
    ├── conversation.json           # Phase-by-phase conversations
    ├── tool_calls.json             # Chronological tool call log
    ├── memory/
    │   ├── todo_list.json          # Working memory: todo items
    │   ├── findings.json           # Working memory: findings
    │   ├── review.json             # Working memory: review checklist
    │   └── report.json             # Working memory: submitted report
    ├── output.json                 # Final AgentOutput
    ├── sub_agents.json             # Direct children summary only
    └── sub_agents/                 # One subfolder per child
        └── <role>_<short_id>/      # Recursive: same structure inside
            ├── agent.json
            ├── conversation.json
            ├── ...

Design decisions:
- **Hierarchical**: Each sub-agent lives in ``sub_agents/<role>_<id>/``
  under its parent. Parent's ``sub_agents.json`` tracks only its direct
  children; each child's subfolder carries its own ``sub_agents.json``
  for *its* children, and so on recursively.
- **Incremental**: Data is saved after each phase, tool call, and
  lifecycle event — not just at workflow completion.
- **Optional**: Persistence is activated only when the orchestrator is
  created with a ``persist_dir``.  All existing behaviour is unchanged
  when persistence is disabled.
- **Synchronous I/O**: File writes are fast (small JSON files) so we
  use synchronous ``pathlib`` operations. No asyncio file I/O needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Fallback serialiser for objects that ``json.dumps`` cannot handle."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if hasattr(obj, "value"):  # Enum
        return obj.value
    return str(obj)


def _write_json(path: Path, data: Any) -> None:
    """Atomically write *data* as pretty-printed JSON to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(
            json.dumps(data, indent=2, default=_json_default, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)
    except Exception:
        # Clean up temp file on failure
        tmp.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> Any:
    """Read and parse a JSON file. Returns ``None`` if the file is missing."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Per-agent file store
# ---------------------------------------------------------------------------

class AgentFileStore:
    """
    Manages the on-disk folder for a **single** agent.

    Provides incremental save helpers for every data type the framework
    produces.  The folder layout is:

        <agent_dir>/
        ├── agent.json
        ├── conversation.json
        ├── tool_calls.json
        ├── memory/
        │   ├── todo_list.json
        │   ├── findings.json
        │   ├── review.json
        │   └── report.json
        ├── output.json
        ├── sub_agents.json
        └── sub_agents/
    """

    def __init__(self, agent_dir: Path) -> None:
        self._dir = agent_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "memory").mkdir(exist_ok=True)
        # Initialise empty lists for incremental appending
        self._tool_calls: List[Dict[str, Any]] = []
        self._conversations: Dict[str, Any] = {}
        self._sub_agents_summary: List[Dict[str, Any]] = []

    @property
    def path(self) -> Path:
        """Root directory for this agent."""
        return self._dir

    # -- Agent info ---------------------------------------------------------

    def save_agent_info(
        self,
        *,
        task_id: str,
        agent_id: str,
        role: str,
        sub_role: Optional[str] = None,
        task: str = "",
        status: str = "running",
        depth: int = 0,
        parent_task_id: Optional[str] = None,
        model: str = "",
    ) -> None:
        """Save (or overwrite) the agent's identity and task description."""
        _write_json(self._dir / "agent.json", {
            "task_id": task_id,
            "agent_id": agent_id,
            "role": role,
            "sub_role": sub_role,
            "task": task,
            "status": status,
            "depth": depth,
            "parent_task_id": parent_task_id,
            "model": model,
        })

    def update_agent_status(self, status: str) -> None:
        """Update only the status field in agent.json."""
        path = self._dir / "agent.json"
        data = _read_json(path) or {}
        data["status"] = status
        _write_json(path, data)

    # -- Phase conversations ------------------------------------------------

    def save_phase_conversation(
        self,
        phase: str,
        *,
        content: Optional[str] = None,
        iterations: int = 0,
        hit_iteration_limit: bool = False,
        tool_calls_made: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Append a phase entry to ``conversation.json``.

        Each phase gets one top-level key in the JSON object.
        """
        self._conversations[phase] = {
            "content": content,
            "iterations": iterations,
            "hit_iteration_limit": hit_iteration_limit,
            "tool_calls_count": len(tool_calls_made) if tool_calls_made else 0,
        }
        _write_json(self._dir / "conversation.json", self._conversations)

    # -- Tool calls ---------------------------------------------------------

    def append_tool_calls(self, calls: List[Dict[str, Any]]) -> None:
        """Append tool call records to ``tool_calls.json``."""
        self._tool_calls.extend(calls)
        _write_json(self._dir / "tool_calls.json", self._tool_calls)

    # -- Working memory -----------------------------------------------------

    def save_todo_list(self, items: List[Dict[str, Any]]) -> None:
        """Save the current todo list snapshot."""
        _write_json(self._dir / "memory" / "todo_list.json", items)

    def save_findings(self, findings: List[Dict[str, Any]]) -> None:
        """Save the current findings snapshot."""
        _write_json(self._dir / "memory" / "findings.json", findings)

    def save_review(self, review_items: List[Dict[str, Any]]) -> None:
        """Save the review checklist snapshot."""
        _write_json(self._dir / "memory" / "review.json", review_items)

    def save_report(self, report: Optional[Dict[str, Any]]) -> None:
        """Save the submitted report."""
        _write_json(self._dir / "memory" / "report.json", report)

    def save_memory(self, memory: Any) -> None:
        """
        Persist the full ``AgentWorkingMemory`` state.

        Accepts the framework's ``AgentWorkingMemory`` object and saves
        each component to its own file inside ``memory/``.
        """
        self.save_todo_list([
            {
                "index": t.index,
                "description": t.description,
                "priority": t.priority,
                "completed": t.completed,
                "completion_summary": t.completion_summary,
            }
            for t in getattr(memory, "todo_list", [])
        ])
        self.save_findings([
            {
                "category": f.category,
                "content": f.content,
                "source": f.source,
                "confidence": f.confidence,
            }
            for f in getattr(memory, "findings", [])
        ])
        self.save_review([
            {
                "criterion": r.criterion,
                "rating": r.rating,
                "justification": r.justification,
            }
            for r in getattr(memory, "review_checklist", [])
        ])
        self.save_report(getattr(memory, "submitted_report", None))

    # -- Final output -------------------------------------------------------

    def save_output(self, output: Any) -> None:
        """
        Persist the agent's ``AgentOutput``.

        Accepts the framework's ``AgentOutput`` object (which has a
        ``to_dict()`` method).
        """
        data = output.to_dict() if hasattr(output, "to_dict") else output
        _write_json(self._dir / "output.json", data)

    # -- Sub-agents ---------------------------------------------------------

    def register_child(
        self,
        *,
        task_id: str,
        agent_id: str,
        role: str,
        sub_role: Optional[str] = None,
        task_description: str = "",
    ) -> None:
        """
        Record a newly spawned child in ``sub_agents.json``.

        This file only tracks **direct** children — one level deep.
        Each child's own subfolder carries its own ``sub_agents.json``
        for its children.
        """
        self._sub_agents_summary.append({
            "task_id": task_id,
            "agent_id": agent_id,
            "role": role,
            "sub_role": sub_role,
            "task_description": task_description[:200],
            "status": "spawned",
        })
        _write_json(self._dir / "sub_agents.json", self._sub_agents_summary)

    def update_child_status(self, task_id: str, status: str) -> None:
        """Update a child's status in ``sub_agents.json``."""
        for entry in self._sub_agents_summary:
            if entry["task_id"] == task_id:
                entry["status"] = status
                break
        _write_json(self._dir / "sub_agents.json", self._sub_agents_summary)

    def create_child_store(self, role: str, task_id: str) -> "AgentFileStore":
        """
        Create and return an ``AgentFileStore`` for a child agent.

        The child's folder is placed under ``sub_agents/<role>_<short_id>/``
        inside this agent's directory.
        """
        import re
        # Remove any character that is unsafe for file-system paths across
        # platforms (Windows: :*?"<>|, all: / \, control chars).
        safe_role = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", role)[:30]
        short_id = task_id[:8]
        child_dir = self._dir / "sub_agents" / f"{safe_role}_{short_id}"
        return AgentFileStore(child_dir)

    # -- Read helpers (for recovery / inspection) ---------------------------

    def read_agent_info(self) -> Optional[Dict[str, Any]]:
        """Read agent.json."""
        return _read_json(self._dir / "agent.json")

    def read_output(self) -> Optional[Dict[str, Any]]:
        """Read output.json."""
        return _read_json(self._dir / "output.json")

    def read_sub_agents(self) -> Optional[List[Dict[str, Any]]]:
        """Read sub_agents.json."""
        return _read_json(self._dir / "sub_agents.json")

    def read_memory(self) -> Dict[str, Any]:
        """Read all memory files."""
        return {
            "todo_list": _read_json(self._dir / "memory" / "todo_list.json"),
            "findings": _read_json(self._dir / "memory" / "findings.json"),
            "review": _read_json(self._dir / "memory" / "review.json"),
            "report": _read_json(self._dir / "memory" / "report.json"),
        }


# ---------------------------------------------------------------------------
# Workflow-level file store
# ---------------------------------------------------------------------------

class WorkflowFileStore:
    """
    Manages the top-level workflow folder.

    The root agent's data lives directly inside the workflow folder
    (the workflow folder *is* the root agent's folder).  Sub-agents
    are nested recursively under ``sub_agents/``.
    """

    def __init__(self, persist_dir: str | Path, workflow_id: str) -> None:
        self._base = Path(persist_dir)
        self._workflow_id = workflow_id
        self._workflow_dir = self._base / workflow_id
        self._workflow_dir.mkdir(parents=True, exist_ok=True)
        # Map task_id → AgentFileStore
        self._stores: Dict[str, AgentFileStore] = {}

    @property
    def workflow_dir(self) -> Path:
        """Root directory for this workflow."""
        return self._workflow_dir

    # -- Workflow-level info ------------------------------------------------

    def save_workflow_info(
        self,
        *,
        workflow_id: str,
        status: str = "running",
        root_task_id: Optional[str] = None,
        budget: Optional[Dict[str, Any]] = None,
        created_at: Optional[str] = None,
    ) -> None:
        """Save workflow-level metadata to ``workflow.json``."""
        _write_json(self._workflow_dir / "workflow.json", {
            "workflow_id": workflow_id,
            "status": status,
            "root_task_id": root_task_id,
            "budget": budget,
            "created_at": created_at,
        })

    def update_workflow_status(self, status: str) -> None:
        """Update the status in ``workflow.json``."""
        path = self._workflow_dir / "workflow.json"
        data = _read_json(path) or {}
        data["status"] = status
        _write_json(path, data)

    # -- Agent store management ---------------------------------------------

    def create_root_store(self, task_id: str) -> AgentFileStore:
        """
        Create and register the root agent's file store.

        The root agent writes directly into the workflow directory.
        """
        store = AgentFileStore(self._workflow_dir)
        self._stores[task_id] = store
        return store

    def create_child_store(
        self, parent_task_id: str, child_task_id: str, role: str,
    ) -> AgentFileStore:
        """
        Create a child agent's file store under its parent's directory.

        Returns the new ``AgentFileStore`` and registers it by task_id.
        """
        parent_store = self._stores.get(parent_task_id)
        if parent_store is None:
            raise ValueError(
                f"Cannot create child store: parent task_id "
                f"'{parent_task_id}' not registered."
            )
        child_store = parent_store.create_child_store(role, child_task_id)
        self._stores[child_task_id] = child_store
        return child_store

    def get_store(self, task_id: str) -> Optional[AgentFileStore]:
        """Retrieve the ``AgentFileStore`` for a given task_id."""
        return self._stores.get(task_id)

    # -- Read helpers -------------------------------------------------------

    def read_workflow_info(self) -> Optional[Dict[str, Any]]:
        """Read workflow.json."""
        return _read_json(self._workflow_dir / "workflow.json")
