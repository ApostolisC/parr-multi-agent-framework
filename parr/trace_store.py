"""
Trace Store for the Agentic Framework.

Append-only, read-only for agents. Only the orchestrator writes to the trace.
Agents receive trace snapshots — they never modify the trace directly.

In-memory for v1. Persistence to DB is an adapter concern for later.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .core_types import AgentNode, AgentStatus, TraceEntry, utc_now

logger = logging.getLogger(__name__)


class TraceStore:
    """
    Append-only execution trace for a workflow.

    Provides a complete, ordered record of all agent activity within
    a workflow execution. Used for observability, debugging, and
    providing context to agents about their siblings' work.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, TraceEntry] = {}  # task_id -> entry
        self._order: List[str] = []  # insertion order

    def add_entry(self, entry: TraceEntry) -> None:
        """
        Add a new trace entry. Only callable by the orchestrator.

        Raises ValueError if task_id already exists (entries are immutable
        once added — only status and output_summary can be updated).
        """
        if entry.task_id in self._entries:
            raise ValueError(
                f"Trace entry '{entry.task_id}' already exists. "
                f"Use update_status() to modify."
            )
        self._entries[entry.task_id] = entry
        self._order.append(entry.task_id)
        logger.debug(
            f"Trace: added {entry.role}:{entry.sub_role or ''} "
            f"task={entry.task_id}"
        )

    def update_status(
        self,
        task_id: str,
        status: AgentStatus,
        output_summary: Optional[str] = None,
    ) -> None:
        """Update the status (and optionally summary) of a trace entry."""
        entry = self._entries.get(task_id)
        if entry is None:
            logger.warning(f"Trace: cannot update unknown task_id={task_id}")
            return
        entry.status = status
        if output_summary is not None:
            entry.output_summary = output_summary
        if status in (AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.CANCELLED):
            entry.completed_at = utc_now()

    def add_child(self, parent_task_id: str, child_task_id: str) -> None:
        """Record a parent-child relationship."""
        entry = self._entries.get(parent_task_id)
        if entry:
            entry.children.append(child_task_id)

    def get_entry(self, task_id: str) -> Optional[TraceEntry]:
        """Get a single trace entry by task_id."""
        return self._entries.get(task_id)

    def get_snapshot(self, for_agent_task_id: str) -> List[TraceEntry]:
        """
        Get a read-only snapshot of relevant trace entries for an agent.

        Returns entries visible to this agent: its parent, its siblings,
        and completed entries from the broader workflow. Excludes the
        agent's own entry to avoid self-reference.
        """
        agent_entry = self._entries.get(for_agent_task_id)
        if agent_entry is None:
            return []

        relevant = []
        for task_id in self._order:
            if task_id == for_agent_task_id:
                continue  # Exclude self
            entry = self._entries[task_id]
            # Include: parent, siblings (same parent), completed agents
            if (
                entry.task_id == agent_entry.parent_task_id
                or entry.parent_task_id == agent_entry.parent_task_id
                or entry.status in (AgentStatus.COMPLETED, AgentStatus.FAILED)
            ):
                relevant.append(entry)

        return relevant

    def get_full_trace(self) -> List[TraceEntry]:
        """Get the complete trace in insertion order."""
        return [self._entries[tid] for tid in self._order]

    def get_children(self, parent_task_id: str) -> List[TraceEntry]:
        """Get all direct children of a task."""
        return [
            self._entries[tid]
            for tid in self._order
            if self._entries[tid].parent_task_id == parent_task_id
        ]

    @property
    def size(self) -> int:
        return len(self._entries)
