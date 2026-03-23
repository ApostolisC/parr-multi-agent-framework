"""
Agent Coordinator for the Agentic Framework.

Pluggable coordination layer for inter-agent communication during workflow
execution.  Provides two mechanisms:

1. **Message passing** -- agents send messages to parent/children/siblings.
2. **Shared state** -- per-workflow key-value store accessible by all agents.

Pluggability:
    - **AgentCoordinator**: Override permission checks and hooks to customise
      routing, access control, logging, or side-effects.
    - The default implementation allows messages between related agents
      (parent<->child, siblings) and unrestricted shared-state access.

Example::

    class AuditCoordinator(AgentCoordinator):
        def on_message_sent(self, message):
            audit_log.info("Agent %s -> %s: %s",
                           message.from_task_id, message.to_task_id,
                           message.content[:80])

    orchestrator = Orchestrator(llm=llm, coordinator=AuditCoordinator())
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .core_types import AgentMessage, AgentNode, WorkflowExecution, generate_id

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """Base class for pluggable agent coordination.

    Override any method to customise message routing, shared-state access
    control, or coordination side-effects.  The default implementation:

    - Allows messages between parent<->child and siblings (same parent).
    - Allows unrestricted read/write to the per-workflow shared state.
    - No-ops for all hooks (``on_message_sent``, ``on_state_changed``).

    Built-in storage
    ~~~~~~~~~~~~~~~~
    The coordinator maintains internal state that the orchestrator delegates
    to:

    - ``_mailboxes``: ``{task_id: [AgentMessage, ...]}`` -- per-agent inboxes.
    - ``_shared_state``: ``{workflow_id: {key: value, ...}}`` -- per-workflow
      key-value store.

    The orchestrator never touches these directly -- it always calls the
    coordinator's public API so that subclasses can intercept.
    """

    def __init__(self) -> None:
        # Per-agent message inboxes: task_id -> list of AgentMessage
        self._mailboxes: Dict[str, List[AgentMessage]] = {}
        # Per-workflow shared state: workflow_id -> {key: value}
        self._shared_state: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Message passing
    # ------------------------------------------------------------------

    def can_send_message(
        self,
        from_task_id: str,
        to_task_id: str,
        workflow: WorkflowExecution,
    ) -> bool:
        """Check whether *from_task_id* may send a message to *to_task_id*.

        Default policy: allow if the agents are parent<->child or siblings
        (share the same parent).  Override for custom routing rules.
        """
        from_node = workflow.agent_tree.get(from_task_id)
        to_node = workflow.agent_tree.get(to_task_id)
        if from_node is None or to_node is None:
            return False

        # Parent -> child
        if to_task_id in from_node.children:
            return True
        # Child -> parent
        if from_node.parent_task_id == to_task_id:
            return True
        # Siblings (same parent)
        if (
            from_node.parent_task_id is not None
            and from_node.parent_task_id == to_node.parent_task_id
        ):
            return True

        return False

    def send_message(
        self,
        from_task_id: str,
        to_task_id: str,
        content: str,
        message_type: str = "info",
        data: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Store a message in the recipient's inbox.

        Returns the created :class:`AgentMessage`.  Callers should verify
        permissions via :meth:`can_send_message` before calling this.
        """
        message = AgentMessage(
            from_task_id=from_task_id,
            to_task_id=to_task_id,
            content=content,
            message_type=message_type,
            data=data or {},
        )
        self._mailboxes.setdefault(to_task_id, []).append(message)
        self.on_message_sent(message)
        return message

    def read_messages(
        self,
        task_id: str,
        since_index: int = 0,
    ) -> List[AgentMessage]:
        """Return messages for *task_id* starting from *since_index*.

        This is a non-destructive read -- messages remain in the mailbox.
        Agents track their own read cursor via ``since_index``.
        """
        mailbox = self._mailboxes.get(task_id, [])
        return mailbox[since_index:]

    def on_message_sent(self, message: AgentMessage) -> None:
        """Hook called after a message is delivered.  Override for logging,
        filtering, or triggering side-effects."""

    # ------------------------------------------------------------------
    # Shared state
    # ------------------------------------------------------------------

    def can_access_state(
        self,
        task_id: str,
        key: str,
        operation: str,
        workflow: WorkflowExecution,
    ) -> bool:
        """Check whether *task_id* may perform *operation* on shared state
        *key*.

        Args:
            task_id: The requesting agent's task ID.
            key: The shared-state key.
            operation: ``"read"`` or ``"write"``.
            workflow: The current workflow execution.

        Default: allow all agents in the workflow.  Override for
        per-key or per-agent access control.
        """
        return task_id in workflow.agent_tree

    def set_shared_state(
        self,
        workflow_id: str,
        key: str,
        value: Any,
        set_by: str,
    ) -> None:
        """Write a key-value pair to the per-workflow shared state.

        Callers should verify permissions via :meth:`can_access_state`
        before calling this.
        """
        store = self._shared_state.setdefault(workflow_id, {})
        store[key] = value
        self.on_state_changed(workflow_id, key, value, set_by)

    def get_shared_state(
        self,
        workflow_id: str,
        key: Optional[str] = None,
    ) -> Any:
        """Read from the per-workflow shared state.

        If *key* is ``None``, returns the entire state dict (shallow copy).
        If *key* is provided, returns the value or ``None`` if not set.
        """
        store = self._shared_state.get(workflow_id, {})
        if key is None:
            return dict(store)
        return store.get(key)

    def on_state_changed(
        self,
        workflow_id: str,
        key: str,
        value: Any,
        changed_by: str,
    ) -> None:
        """Hook called after a shared-state write.  Override for logging,
        notifications, or reactive side-effects."""

    # ------------------------------------------------------------------
    # Lifecycle helpers (called by orchestrator)
    # ------------------------------------------------------------------

    def clear_workflow(self, workflow_id: str) -> None:
        """Clean up all coordination state for a completed workflow."""
        self._shared_state.pop(workflow_id, None)
        # Mailboxes are keyed by task_id -- we'd need to know which task_ids
        # belong to the workflow.  The orchestrator calls this after the tree
        # is no longer needed, so we accept the minor memory leak for
        # simplicity.  A production subclass can override to do full cleanup.

    def clear_agent(self, task_id: str) -> None:
        """Clean up mailbox for a completed agent."""
        self._mailboxes.pop(task_id, None)
