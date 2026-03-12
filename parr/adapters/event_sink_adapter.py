"""
EventSink Adapters for the Agentic Framework.

Implementations of the EventSink protocol for different transports:
- LoggingEventSink: Logs events via Python logging (debugging/development)
- WebSocketEventSink: Forwards events to a WebSocket send callback
- CompositeEventSink: Fans out events to multiple sinks

Usage:
    # Logging only
    sink = LoggingEventSink()

    # WebSocket
    sink = WebSocketEventSink(send_callback=ws.send_json)

    # Both
    sink = CompositeEventSink([
        LoggingEventSink(),
        WebSocketEventSink(send_callback=ws.send_json),
    ])

    orchestrator = Orchestrator(llm=llm, event_sink=sink)
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List

logger = logging.getLogger(__name__)


class LoggingEventSink:
    """
    EventSink that logs all framework events.

    Useful for debugging and development. Events are logged at INFO level
    with a structured format showing event type, agent, and key data.
    """

    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

    async def emit(self, event: Dict[str, Any]) -> None:
        """Log the event."""
        event_type = event.get("event_type", "unknown")
        agent_id = event.get("agent_id", "?")[:8]
        workflow_id = event.get("workflow_id", "?")[:8]
        data = event.get("data", {})

        # Format key data fields compactly
        data_summary = ""
        if "phase" in data:
            data_summary += f" phase={data['phase']}"
        if "role" in data:
            data_summary += f" role={data['role']}"
        if "total_tokens" in data:
            data_summary += f" tokens={data['total_tokens']}"
        if "reason" in data:
            data_summary += f" reason={data['reason']}"
        if "tool" in data:
            data_summary += f" tool={data['tool']}"

        logger.log(
            self._log_level,
            f"[{event_type}] wf={workflow_id} agent={agent_id}{data_summary}",
        )


class WebSocketEventSink:
    """
    EventSink that forwards framework events to a WebSocket send callback.

    The callback should accept a dict and send it as JSON to connected
    WebSocket clients.
    """

    def __init__(
        self, send_callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """
        Args:
            send_callback: Async callable that sends an event dict to the client.
                           Typically ``websocket.send_json`` or a custom broadcast function.
        """
        self._send = send_callback

    async def emit(self, event: Dict[str, Any]) -> None:
        """Forward the event to the WebSocket callback."""
        try:
            await self._send(event)
        except Exception as e:
            logger.error(
                f"WebSocket send failed for {event.get('event_type', '?')}: {e}"
            )


class CompositeEventSink:
    """
    EventSink that fans out events to multiple child sinks.

    Errors in one sink do not prevent delivery to others.
    """

    def __init__(self, sinks: List[Any]) -> None:
        """
        Args:
            sinks: List of EventSink-compatible objects (each has async emit()).
        """
        self._sinks = list(sinks)
        self._failure_counts: Dict[int, int] = {}  # sink id(obj) -> count

    async def emit(self, event: Dict[str, Any]) -> None:
        """Forward the event to all child sinks."""
        for sink in self._sinks:
            try:
                await sink.emit(event)
            except Exception as e:
                key = id(sink)
                self._failure_counts[key] = self._failure_counts.get(key, 0) + 1
                logger.warning(
                    f"CompositeEventSink: child sink {type(sink).__name__} failed "
                    f"(failure #{self._failure_counts[key]}) for "
                    f"{event.get('event_type', '?')}: {e}"
                )

    def add_sink(self, sink: Any) -> None:
        """Add a sink at runtime."""
        self._sinks.append(sink)

    def remove_sink(self, sink: Any) -> None:
        """Remove a sink at runtime."""
        self._sinks = [s for s in self._sinks if s is not sink]
