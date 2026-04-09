"""
Event Bus for the Agentic Framework.

In-process pub/sub system for framework events. The orchestrator publishes
events here, and the application layer subscribes to receive them.

For standalone testing, this provides an InMemoryEventSink that collects
all events for inspection. In production, the adapter layer provides a
transport-specific EventSink (e.g., WebSocket, SSE).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .event_types import FrameworkEvent

logger = logging.getLogger(__name__)


@dataclass
class Subscription:
    """A registered event subscription."""
    id: str
    workflow_id: str
    handler: Callable[[FrameworkEvent], Coroutine]


class EventBus:
    """
    In-process event bus for the framework.

    Supports workflow-scoped subscriptions. Events are dispatched to all
    subscribers for the matching workflow_id.

    Async-safe for concurrent publish/subscribe via asyncio.Lock.

    An optional ``on_handler_error`` callback is invoked when a subscriber's
    handler raises an exception.  The default logs and continues; callers
    can provide a custom callback to collect or re-raise errors.
    """

    def __init__(
        self,
        on_handler_error: Optional[Callable[[FrameworkEvent, Exception], None]] = None,
    ) -> None:
        self._subscriptions: Dict[str, List[Subscription]] = {}
        self._next_id: int = 0
        self._lock = asyncio.Lock()
        self._on_handler_error = on_handler_error

    async def publish(self, event: FrameworkEvent) -> None:
        """Publish an event to all subscribers for this workflow."""
        async with self._lock:
            subs = list(self._subscriptions.get(event.workflow_id, []))
        for sub in subs:
            try:
                await sub.handler(event)
            except Exception as e:
                logger.error(
                    f"Event handler error for {event.event_type}: {e}",
                    exc_info=True,
                )
                if self._on_handler_error is not None:
                    self._on_handler_error(event, e)

    def subscribe(
        self,
        workflow_id: str,
        handler: Callable[[FrameworkEvent], Coroutine],
    ) -> Subscription:
        """Subscribe to events for a workflow."""
        self._next_id += 1
        sub = Subscription(
            id=f"sub_{self._next_id}",
            workflow_id=workflow_id,
            handler=handler,
        )
        if workflow_id not in self._subscriptions:
            self._subscriptions[workflow_id] = []
        self._subscriptions[workflow_id].append(sub)
        return sub

    def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a subscription."""
        subs = self._subscriptions.get(subscription.workflow_id, [])
        self._subscriptions[subscription.workflow_id] = [
            s for s in subs if s.id != subscription.id
        ]

    def clear_workflow(self, workflow_id: str) -> None:
        """Remove all subscriptions for a workflow."""
        self._subscriptions.pop(workflow_id, None)


class InMemoryEventSink:
    """
    Simple event sink that collects events in memory.

    Useful for testing and debugging. Implements the EventSink protocol.
    """

    def __init__(self) -> None:
        self.events: List[FrameworkEvent] = []

    async def emit(self, event: Dict[str, Any]) -> None:
        """Collect the event."""
        if isinstance(event, FrameworkEvent):
            self.events.append(event)
        else:
            # Convert dict to FrameworkEvent
            self.events.append(FrameworkEvent(**event))

    def get_events(self, event_type: Optional[str] = None) -> List[FrameworkEvent]:
        """Get collected events, optionally filtered by type."""
        if event_type:
            return [e for e in self.events if e.event_type == event_type]
        return list(self.events)

    def clear(self) -> None:
        """Clear all collected events."""
        self.events.clear()

    @property
    def count(self) -> int:
        return len(self.events)


class EventBridge:
    """
    Bridges the EventBus to an EventSink (protocol implementation).

    The orchestrator uses the EventBus internally. This bridge subscribes
    to the bus and forwards events to the adapter's EventSink.

    An optional concurrency limit (``max_concurrent``) provides lightweight
    backpressure so a slow sink doesn't allow unbounded concurrent emits.
    """

    _MAX_CONCURRENT_EMITS = 32  # sensible default, no knob needed yet

    def __init__(self, bus: EventBus, sink: Any) -> None:
        """
        Args:
            bus: The framework's internal EventBus.
            sink: An object implementing the EventSink protocol (has async emit()).
        """
        self._bus = bus
        self._sink = sink
        self._subscriptions: List[Subscription] = []
        self._failure_count: int = 0
        self._last_error: Optional[Exception] = None
        self._semaphore = asyncio.Semaphore(self._MAX_CONCURRENT_EMITS)

    def connect(self, workflow_id: str) -> None:
        """Start forwarding events for a workflow to the sink."""
        sub = self._bus.subscribe(workflow_id, self._forward)
        self._subscriptions.append(sub)

    def disconnect_all(self) -> None:
        """Stop forwarding all events."""
        for sub in self._subscriptions:
            self._bus.unsubscribe(sub)
        self._subscriptions.clear()

    async def _forward(self, event: FrameworkEvent) -> None:
        """Forward an event to the sink with backpressure."""
        async with self._semaphore:
            try:
                await self._sink.emit(event.to_dict())
            except Exception as e:
                self._failure_count += 1
                self._last_error = e
                logger.warning(
                    f"EventBridge forward error (failure #{self._failure_count}): {e}",
                    exc_info=True,
                )

    @property
    def failure_count(self) -> int:
        """Number of failed event forwards since creation."""
        return self._failure_count

    @property
    def last_error(self) -> Optional[Exception]:
        """The most recent forwarding error, if any."""
        return self._last_error
