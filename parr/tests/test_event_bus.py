"""Tests for the EventBus, InMemoryEventSink, and EventBridge."""

import asyncio

import pytest

from parr.event_bus import EventBridge, EventBus, InMemoryEventSink
from parr.event_types import FrameworkEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    workflow_id: str = "wf-1",
    event_type: str = "test_event",
    task_id: str = "t-1",
    agent_id: str = "a-1",
    **data,
) -> FrameworkEvent:
    return FrameworkEvent(
        workflow_id=workflow_id,
        task_id=task_id,
        agent_id=agent_id,
        event_type=event_type,
        data=data,
    )


# ===================================================================
# EventBus
# ===================================================================


class TestEventBusSubscribeUnsubscribe:
    def test_subscribe_returns_subscription(self, event_bus):
        sub = event_bus.subscribe("wf-1", lambda e: None)
        assert sub.workflow_id == "wf-1"
        assert sub.id.startswith("sub_")

    def test_subscribe_increments_id(self, event_bus):
        s1 = event_bus.subscribe("wf-1", lambda e: None)
        s2 = event_bus.subscribe("wf-1", lambda e: None)
        assert s1.id != s2.id

    def test_unsubscribe_removes_subscription(self, event_bus):
        sub = event_bus.subscribe("wf-1", lambda e: None)
        event_bus.unsubscribe(sub)
        assert event_bus._subscriptions.get("wf-1") == []

    def test_unsubscribe_only_target(self, event_bus):
        s1 = event_bus.subscribe("wf-1", lambda e: None)
        s2 = event_bus.subscribe("wf-1", lambda e: None)
        event_bus.unsubscribe(s1)
        remaining = event_bus._subscriptions["wf-1"]
        assert len(remaining) == 1
        assert remaining[0].id == s2.id


class TestEventBusClearWorkflow:
    def test_clear_removes_all_subs(self, event_bus):
        event_bus.subscribe("wf-1", lambda e: None)
        event_bus.subscribe("wf-1", lambda e: None)
        event_bus.clear_workflow("wf-1")
        assert "wf-1" not in event_bus._subscriptions

    def test_clear_unknown_workflow_is_noop(self, event_bus):
        event_bus.clear_workflow("nonexistent")  # should not raise

    def test_clear_does_not_affect_other_workflows(self, event_bus):
        event_bus.subscribe("wf-1", lambda e: None)
        event_bus.subscribe("wf-2", lambda e: None)
        event_bus.clear_workflow("wf-1")
        assert "wf-2" in event_bus._subscriptions


class TestEventBusPublish:
    @pytest.mark.asyncio
    async def test_publish_dispatches_to_matching_workflow(self, event_bus):
        received = []

        async def handler(e):
            received.append(e)

        event_bus.subscribe("wf-1", handler)
        event = _make_event(workflow_id="wf-1")
        await event_bus.publish(event)
        assert len(received) == 1
        assert received[0] is event

    @pytest.mark.asyncio
    async def test_publish_ignores_other_workflows(self, event_bus):
        received = []

        async def handler(e):
            received.append(e)

        event_bus.subscribe("wf-2", handler)
        await event_bus.publish(_make_event(workflow_id="wf-1"))
        assert received == []

    @pytest.mark.asyncio
    async def test_publish_dispatches_to_multiple_subscribers(self, event_bus):
        calls = {"a": 0, "b": 0}

        async def handler_a(e):
            calls["a"] += 1

        async def handler_b(e):
            calls["b"] += 1

        event_bus.subscribe("wf-1", handler_a)
        event_bus.subscribe("wf-1", handler_b)
        await event_bus.publish(_make_event())
        assert calls == {"a": 1, "b": 1}

    @pytest.mark.asyncio
    async def test_publish_handler_error_does_not_propagate(self, event_bus):
        async def bad_handler(e):
            raise RuntimeError("boom")

        received = []

        async def good_handler(e):
            received.append(e)

        event_bus.subscribe("wf-1", bad_handler)
        event_bus.subscribe("wf-1", good_handler)
        await event_bus.publish(_make_event())
        # Good handler still called despite earlier handler raising
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_publish_no_subscribers_is_noop(self, event_bus):
        await event_bus.publish(_make_event())  # should not raise


# ===================================================================
# InMemoryEventSink
# ===================================================================


class TestInMemoryEventSinkEmit:
    @pytest.mark.asyncio
    async def test_emit_framework_event(self, event_sink):
        ev = _make_event()
        await event_sink.emit(ev)
        assert event_sink.count == 1
        assert event_sink.events[0] is ev

    @pytest.mark.asyncio
    async def test_emit_dict_converts_to_framework_event(self, event_sink):
        d = {
            "workflow_id": "wf-1",
            "task_id": "t-1",
            "agent_id": "a-1",
            "event_type": "dict_event",
        }
        await event_sink.emit(d)
        assert event_sink.count == 1
        stored = event_sink.events[0]
        assert isinstance(stored, FrameworkEvent)
        assert stored.event_type == "dict_event"

    @pytest.mark.asyncio
    async def test_emit_multiple_events(self, event_sink):
        for i in range(5):
            await event_sink.emit(_make_event(event_type=f"type_{i}"))
        assert event_sink.count == 5


class TestInMemoryEventSinkGetEvents:
    @pytest.mark.asyncio
    async def test_get_events_returns_all(self, event_sink):
        await event_sink.emit(_make_event(event_type="a"))
        await event_sink.emit(_make_event(event_type="b"))
        assert len(event_sink.get_events()) == 2

    @pytest.mark.asyncio
    async def test_get_events_filters_by_type(self, event_sink):
        await event_sink.emit(_make_event(event_type="keep"))
        await event_sink.emit(_make_event(event_type="drop"))
        await event_sink.emit(_make_event(event_type="keep"))
        filtered = event_sink.get_events("keep")
        assert len(filtered) == 2
        assert all(e.event_type == "keep" for e in filtered)

    @pytest.mark.asyncio
    async def test_get_events_filter_no_match(self, event_sink):
        await event_sink.emit(_make_event(event_type="x"))
        assert event_sink.get_events("y") == []

    @pytest.mark.asyncio
    async def test_get_events_returns_copy(self, event_sink):
        await event_sink.emit(_make_event())
        result = event_sink.get_events()
        result.clear()
        assert event_sink.count == 1


class TestInMemoryEventSinkClearAndCount:
    def test_count_empty(self, event_sink):
        assert event_sink.count == 0

    @pytest.mark.asyncio
    async def test_clear(self, event_sink):
        await event_sink.emit(_make_event())
        await event_sink.emit(_make_event())
        event_sink.clear()
        assert event_sink.count == 0
        assert event_sink.events == []


# ===================================================================
# EventBridge
# ===================================================================


class TestEventBridgeConnectDisconnect:
    def test_connect_registers_subscription(self, event_bus, event_sink):
        bridge = EventBridge(event_bus, event_sink)
        bridge.connect("wf-1")
        assert len(bridge._subscriptions) == 1
        assert bridge._subscriptions[0].workflow_id == "wf-1"

    def test_connect_multiple_workflows(self, event_bus, event_sink):
        bridge = EventBridge(event_bus, event_sink)
        bridge.connect("wf-1")
        bridge.connect("wf-2")
        assert len(bridge._subscriptions) == 2

    def test_disconnect_all_clears(self, event_bus, event_sink):
        bridge = EventBridge(event_bus, event_sink)
        bridge.connect("wf-1")
        bridge.connect("wf-2")
        bridge.disconnect_all()
        assert bridge._subscriptions == []

    @pytest.mark.asyncio
    async def test_disconnect_stops_forwarding(self, event_bus, event_sink):
        bridge = EventBridge(event_bus, event_sink)
        bridge.connect("wf-1")
        bridge.disconnect_all()
        await event_bus.publish(_make_event(workflow_id="wf-1"))
        assert event_sink.count == 0


class TestEventBridgeForwarding:
    @pytest.mark.asyncio
    async def test_forwards_event_to_sink(self, event_bus, event_sink):
        bridge = EventBridge(event_bus, event_sink)
        bridge.connect("wf-1")
        event = _make_event(workflow_id="wf-1")
        await event_bus.publish(event)
        assert event_sink.count == 1
        # Bridge calls event.to_dict(), so sink receives a dict → converted back
        stored = event_sink.events[0]
        assert stored.event_type == "test_event"
        assert stored.workflow_id == "wf-1"

    @pytest.mark.asyncio
    async def test_forwards_only_connected_workflows(self, event_bus, event_sink):
        bridge = EventBridge(event_bus, event_sink)
        bridge.connect("wf-1")
        await event_bus.publish(_make_event(workflow_id="wf-2"))
        assert event_sink.count == 0

    @pytest.mark.asyncio
    async def test_forwards_multiple_events(self, event_bus, event_sink):
        bridge = EventBridge(event_bus, event_sink)
        bridge.connect("wf-1")
        for _ in range(3):
            await event_bus.publish(_make_event(workflow_id="wf-1"))
        assert event_sink.count == 3


class TestEventBridgeErrorHandling:
    @pytest.mark.asyncio
    async def test_sink_error_increments_failure_count(self, event_bus):
        class FailingSink:
            async def emit(self, event):
                raise IOError("network down")

        bridge = EventBridge(event_bus, FailingSink())
        bridge.connect("wf-1")
        assert bridge.failure_count == 0

        await event_bus.publish(_make_event(workflow_id="wf-1"))
        assert bridge.failure_count == 1

    @pytest.mark.asyncio
    async def test_last_error_tracks_most_recent(self, event_bus):
        class FailingSink:
            def __init__(self):
                self.call_count = 0

            async def emit(self, event):
                self.call_count += 1
                raise ValueError(f"error-{self.call_count}")

        bridge = EventBridge(event_bus, FailingSink())
        bridge.connect("wf-1")

        await event_bus.publish(_make_event(workflow_id="wf-1"))
        await event_bus.publish(_make_event(workflow_id="wf-1"))

        assert bridge.failure_count == 2
        assert "error-2" in str(bridge.last_error)

    @pytest.mark.asyncio
    async def test_last_error_is_none_initially(self, event_bus, event_sink):
        bridge = EventBridge(event_bus, event_sink)
        assert bridge.last_error is None

    @pytest.mark.asyncio
    async def test_sink_error_does_not_propagate(self, event_bus):
        class FailingSink:
            async def emit(self, event):
                raise RuntimeError("kaboom")

        bridge = EventBridge(event_bus, FailingSink())
        bridge.connect("wf-1")
        # Should not raise
        await event_bus.publish(_make_event(workflow_id="wf-1"))

    @pytest.mark.asyncio
    async def test_sink_error_does_not_block_bus(self, event_bus):
        """Other bus subscribers still receive the event even if the bridge fails."""
        received = []

        async def direct_handler(e):
            received.append(e)

        class FailingSink:
            async def emit(self, event):
                raise RuntimeError("fail")

        bridge = EventBridge(event_bus, FailingSink())
        bridge.connect("wf-1")
        event_bus.subscribe("wf-1", direct_handler)

        await event_bus.publish(_make_event(workflow_id="wf-1"))
        assert bridge.failure_count == 1
        assert len(received) == 1
