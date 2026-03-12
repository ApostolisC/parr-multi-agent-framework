"""Tests for parr.trace_store — TraceStore append-only execution trace."""

import logging

import pytest

from parr.core_types import AgentStatus, TraceEntry
from parr.trace_store import TraceStore


# -- Helpers -----------------------------------------------------------------


def _entry(task_id: str, **kwargs) -> TraceEntry:
    """Create a TraceEntry with a fixed task_id and optional overrides."""
    return TraceEntry(task_id=task_id, **kwargs)


# -- Empty store edge cases --------------------------------------------------


class TestEmptyStore:
    def test_size_is_zero(self):
        store = TraceStore()
        assert store.size == 0

    def test_get_entry_returns_none(self):
        store = TraceStore()
        assert store.get_entry("nonexistent") is None

    def test_get_full_trace_returns_empty(self):
        store = TraceStore()
        assert store.get_full_trace() == []

    def test_get_children_returns_empty(self):
        store = TraceStore()
        assert store.get_children("nonexistent") == []

    def test_get_snapshot_returns_empty(self):
        store = TraceStore()
        assert store.get_snapshot("nonexistent") == []


# -- add_entry ---------------------------------------------------------------


class TestAddEntry:
    def test_adds_entry(self):
        store = TraceStore()
        entry = _entry("t1", role="analyst")
        store.add_entry(entry)
        assert store.size == 1
        assert store.get_entry("t1") is entry

    def test_duplicate_raises_value_error(self):
        store = TraceStore()
        store.add_entry(_entry("t1"))
        with pytest.raises(ValueError, match="already exists"):
            store.add_entry(_entry("t1"))

    def test_multiple_entries(self):
        store = TraceStore()
        store.add_entry(_entry("t1"))
        store.add_entry(_entry("t2"))
        store.add_entry(_entry("t3"))
        assert store.size == 3


# -- update_status -----------------------------------------------------------


class TestUpdateStatus:
    def test_updates_status(self):
        store = TraceStore()
        store.add_entry(_entry("t1"))
        store.update_status("t1", AgentStatus.COMPLETED)
        assert store.get_entry("t1").status == AgentStatus.COMPLETED

    def test_updates_output_summary(self):
        store = TraceStore()
        store.add_entry(_entry("t1"))
        store.update_status("t1", AgentStatus.COMPLETED, output_summary="done")
        entry = store.get_entry("t1")
        assert entry.output_summary == "done"

    def test_does_not_overwrite_summary_when_none(self):
        store = TraceStore()
        store.add_entry(_entry("t1"))
        store.update_status("t1", AgentStatus.RUNNING, output_summary="partial")
        store.update_status("t1", AgentStatus.COMPLETED)
        assert store.get_entry("t1").output_summary == "partial"

    def test_sets_completed_at_for_completed(self):
        store = TraceStore()
        store.add_entry(_entry("t1"))
        assert store.get_entry("t1").completed_at is None
        store.update_status("t1", AgentStatus.COMPLETED)
        assert store.get_entry("t1").completed_at is not None

    def test_sets_completed_at_for_failed(self):
        store = TraceStore()
        store.add_entry(_entry("t1"))
        store.update_status("t1", AgentStatus.FAILED)
        assert store.get_entry("t1").completed_at is not None

    def test_sets_completed_at_for_cancelled(self):
        store = TraceStore()
        store.add_entry(_entry("t1"))
        store.update_status("t1", AgentStatus.CANCELLED)
        assert store.get_entry("t1").completed_at is not None

    def test_does_not_set_completed_at_for_running(self):
        store = TraceStore()
        store.add_entry(_entry("t1"))
        store.update_status("t1", AgentStatus.RUNNING)
        assert store.get_entry("t1").completed_at is None

    def test_does_not_set_completed_at_for_suspended(self):
        store = TraceStore()
        store.add_entry(_entry("t1"))
        store.update_status("t1", AgentStatus.SUSPENDED)
        assert store.get_entry("t1").completed_at is None

    def test_unknown_task_id_warns(self, caplog):
        store = TraceStore()
        with caplog.at_level(logging.WARNING, logger="parr.trace_store"):
            store.update_status("missing", AgentStatus.COMPLETED)
        assert "unknown task_id" in caplog.text

    def test_unknown_task_id_does_not_raise(self):
        store = TraceStore()
        store.update_status("missing", AgentStatus.COMPLETED)


# -- add_child ---------------------------------------------------------------


class TestAddChild:
    def test_records_child(self):
        store = TraceStore()
        store.add_entry(_entry("parent"))
        store.add_entry(_entry("child", parent_task_id="parent"))
        store.add_child("parent", "child")
        assert "child" in store.get_entry("parent").children

    def test_multiple_children(self):
        store = TraceStore()
        store.add_entry(_entry("parent"))
        store.add_child("parent", "c1")
        store.add_child("parent", "c2")
        assert store.get_entry("parent").children == ["c1", "c2"]

    def test_unknown_parent_does_not_raise(self):
        store = TraceStore()
        store.add_child("missing", "child")


# -- get_entry ---------------------------------------------------------------


class TestGetEntry:
    def test_returns_entry(self):
        store = TraceStore()
        entry = _entry("t1", role="analyst")
        store.add_entry(entry)
        assert store.get_entry("t1") is entry

    def test_returns_none_for_missing(self):
        store = TraceStore()
        assert store.get_entry("nope") is None


# -- get_full_trace ----------------------------------------------------------


class TestGetFullTrace:
    def test_returns_all_in_insertion_order(self):
        store = TraceStore()
        ids = ["t1", "t2", "t3"]
        for tid in ids:
            store.add_entry(_entry(tid))
        trace = store.get_full_trace()
        assert [e.task_id for e in trace] == ids

    def test_returns_actual_entry_objects(self):
        store = TraceStore()
        entry = _entry("t1")
        store.add_entry(entry)
        assert store.get_full_trace()[0] is entry


# -- get_children ------------------------------------------------------------


class TestGetChildren:
    def test_returns_direct_children(self):
        store = TraceStore()
        store.add_entry(_entry("parent"))
        store.add_entry(_entry("c1", parent_task_id="parent"))
        store.add_entry(_entry("c2", parent_task_id="parent"))
        store.add_entry(_entry("other"))
        children = store.get_children("parent")
        assert [c.task_id for c in children] == ["c1", "c2"]

    def test_does_not_return_grandchildren(self):
        store = TraceStore()
        store.add_entry(_entry("root"))
        store.add_entry(_entry("child", parent_task_id="root"))
        store.add_entry(_entry("grandchild", parent_task_id="child"))
        children = store.get_children("root")
        assert [c.task_id for c in children] == ["child"]

    def test_returns_empty_for_no_children(self):
        store = TraceStore()
        store.add_entry(_entry("leaf"))
        assert store.get_children("leaf") == []

    def test_preserves_insertion_order(self):
        store = TraceStore()
        store.add_entry(_entry("p"))
        for i in range(5):
            store.add_entry(_entry(f"c{i}", parent_task_id="p"))
        children = store.get_children("p")
        assert [c.task_id for c in children] == [f"c{i}" for i in range(5)]


# -- size property -----------------------------------------------------------


class TestSizeProperty:
    def test_increments_on_add(self):
        store = TraceStore()
        assert store.size == 0
        store.add_entry(_entry("t1"))
        assert store.size == 1
        store.add_entry(_entry("t2"))
        assert store.size == 2

    def test_unaffected_by_updates(self):
        store = TraceStore()
        store.add_entry(_entry("t1"))
        store.update_status("t1", AgentStatus.COMPLETED)
        assert store.size == 1


# -- get_snapshot ------------------------------------------------------------


class TestGetSnapshot:
    def test_excludes_self(self):
        store = TraceStore()
        store.add_entry(_entry("t1", parent_task_id="root"))
        snapshot = store.get_snapshot("t1")
        assert all(e.task_id != "t1" for e in snapshot)

    def test_includes_parent(self):
        store = TraceStore()
        store.add_entry(_entry("root", role="orchestrator"))
        store.add_entry(_entry("child", parent_task_id="root"))
        snapshot = store.get_snapshot("child")
        ids = [e.task_id for e in snapshot]
        assert "root" in ids

    def test_includes_siblings(self):
        store = TraceStore()
        store.add_entry(_entry("root"))
        store.add_entry(_entry("s1", parent_task_id="root"))
        store.add_entry(_entry("s2", parent_task_id="root"))
        store.add_entry(_entry("s3", parent_task_id="root"))
        snapshot = store.get_snapshot("s2")
        ids = [e.task_id for e in snapshot]
        assert "s1" in ids
        assert "s3" in ids

    def test_includes_completed_entries(self):
        store = TraceStore()
        store.add_entry(_entry("done", status=AgentStatus.COMPLETED))
        store.add_entry(_entry("me"))
        snapshot = store.get_snapshot("me")
        ids = [e.task_id for e in snapshot]
        assert "done" in ids

    def test_includes_failed_entries(self):
        store = TraceStore()
        store.add_entry(_entry("failed", status=AgentStatus.FAILED))
        store.add_entry(_entry("me"))
        snapshot = store.get_snapshot("me")
        ids = [e.task_id for e in snapshot]
        assert "failed" in ids

    def test_excludes_running_non_sibling(self):
        store = TraceStore()
        store.add_entry(_entry("other_parent"))
        store.add_entry(
            _entry("unrelated", parent_task_id="other_parent", status=AgentStatus.RUNNING)
        )
        store.add_entry(_entry("root"))
        store.add_entry(_entry("me", parent_task_id="root"))
        snapshot = store.get_snapshot("me")
        ids = [e.task_id for e in snapshot]
        assert "unrelated" not in ids

    def test_returns_empty_for_unknown_task(self):
        store = TraceStore()
        store.add_entry(_entry("t1"))
        assert store.get_snapshot("nonexistent") == []

    def test_preserves_insertion_order(self):
        store = TraceStore()
        store.add_entry(_entry("root"))
        store.add_entry(_entry("s1", parent_task_id="root"))
        store.add_entry(_entry("s2", parent_task_id="root"))
        store.add_entry(_entry("s3", parent_task_id="root"))
        snapshot = store.get_snapshot("s2")
        ids = [e.task_id for e in snapshot]
        assert ids == ["root", "s1", "s3"]

    def test_no_duplicates_when_parent_is_also_completed(self):
        store = TraceStore()
        store.add_entry(
            _entry("root", status=AgentStatus.COMPLETED)
        )
        store.add_entry(_entry("child", parent_task_id="root"))
        snapshot = store.get_snapshot("child")
        ids = [e.task_id for e in snapshot]
        # root matches as both parent and completed; should appear only once
        assert ids.count("root") == 1

    def test_complex_scenario(self):
        """Realistic multi-level workflow snapshot."""
        store = TraceStore()
        # Root orchestrator
        store.add_entry(_entry("root", role="orchestrator"))
        # Two children of root
        store.add_entry(
            _entry("a1", role="analyst", parent_task_id="root", status=AgentStatus.COMPLETED)
        )
        store.add_entry(
            _entry("a2", role="reviewer", parent_task_id="root", status=AgentStatus.RUNNING)
        )
        # Unrelated running agent under a different parent
        store.add_entry(_entry("other_root", role="orchestrator2"))
        store.add_entry(
            _entry("b1", role="writer", parent_task_id="other_root", status=AgentStatus.RUNNING)
        )

        snapshot = store.get_snapshot("a2")
        ids = [e.task_id for e in snapshot]
        # Should include: root (parent), a1 (sibling + completed)
        assert "root" in ids
        assert "a1" in ids
        # Should NOT include: b1 (unrelated, running)
        assert "b1" not in ids
        # other_root is not a sibling nor completed — excluded
        assert "other_root" not in ids
