"""Tests for parr.compaction_strategy — pluggable context compaction."""

from __future__ import annotations

from collections import Counter
from typing import Any, List, Optional

import pytest

from parr.compaction_strategy import CompactionStrategy, DEFAULT_CHARS_PER_TOKEN
from parr.context_manager import ContextManager
from parr.core_types import Message, MessageRole, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: MessageRole, content: str = "", **kwargs) -> Message:
    return Message(role=role, content=content or None, **kwargs)


def _system(content: str) -> Message:
    return _msg(MessageRole.SYSTEM, content)


def _user(content: str) -> Message:
    return _msg(MessageRole.USER, content)


def _assistant(content: str = "", tool_calls=None) -> Message:
    return Message(role=MessageRole.ASSISTANT, content=content or None, tool_calls=tool_calls)


def _tool(content: str, tool_call_id: str = "tc-1") -> Message:
    return Message(role=MessageRole.TOOL, content=content, tool_call_id=tool_call_id)


def _tc(name: str, args: dict = None, tc_id: str = "tc-1") -> ToolCall:
    return ToolCall(id=tc_id, name=name, arguments=args or {})


def _make_long_messages(n_groups: int = 6, content_size: int = 80) -> list[Message]:
    msgs = [_system("S" * 40), _user("U" * 40)]
    for i in range(n_groups):
        tid = f"tc-{i}"
        tc = _tc("some_tool", {"q": "x" * content_size}, tc_id=tid)
        msgs.append(_assistant(tool_calls=[tc]))
        msgs.append(_tool("R" * content_size, tool_call_id=tid))
    return msgs


# ===================================================================
# 1. Default strategy matches ContextManager behavior
# ===================================================================

class TestDefaultBehavior:

    def test_estimate_tokens_basic(self):
        s = CompactionStrategy(chars_per_token=4.0)
        msgs = [_system("a" * 100)]
        assert s.estimate_tokens(msgs) == 25

    def test_estimate_tokens_with_overhead(self):
        s = CompactionStrategy(chars_per_token=4.0, tool_schema_overhead=200)
        msgs = [_system("a" * 40)]
        assert s.estimate_tokens(msgs) == 10 + 200

    def test_compact_under_threshold_unchanged(self):
        s = CompactionStrategy(max_context_tokens=500, soft_compaction_pct=0.40)
        msgs = [_system("short"), _user("short")]
        result = s.compact_if_needed(msgs)
        assert result is msgs

    def test_compact_over_threshold(self):
        s = CompactionStrategy(
            max_context_tokens=200,
            soft_compaction_pct=0.20,
            chars_per_token=4.0,
        )
        msgs = _make_long_messages(n_groups=6)
        result = s.compact_if_needed(msgs)
        assert len(result) < len(msgs)
        assert result[0].role == MessageRole.SYSTEM
        assert result[1].role == MessageRole.USER

    def test_truncate_under_threshold_unchanged(self):
        s = CompactionStrategy(max_context_tokens=1000, hard_truncation_pct=0.65)
        msgs = [_system("short"), _user("short")]
        result = s.truncate_if_needed(msgs)
        assert result == msgs

    def test_truncate_over_threshold(self):
        s = CompactionStrategy(
            max_context_tokens=200,
            soft_compaction_pct=0.05,
            hard_truncation_pct=0.10,
            chars_per_token=4.0,
        )
        msgs = _make_long_messages(n_groups=8)
        result = s.truncate_if_needed(msgs)
        assert len(result) < len(msgs)

    def test_pair_messages_groups_correctly(self):
        tc = _tc("tool_a", {}, tc_id="tc-1")
        msgs = [
            _assistant(tool_calls=[tc]),
            _tool("result", tool_call_id="tc-1"),
            _assistant("standalone"),
        ]
        groups = CompactionStrategy.pair_messages(msgs)
        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 1

    def test_should_preserve_group_findings(self):
        s = CompactionStrategy()
        tc = _tc("log_finding", {"category": "risk"})
        group = [_assistant(tool_calls=[tc]), _tool("ok")]
        assert s.should_preserve_group(group) is True

    def test_should_preserve_group_non_findings(self):
        s = CompactionStrategy()
        tc = _tc("search", {"q": "test"})
        group = [_assistant(tool_calls=[tc]), _tool("results")]
        assert s.should_preserve_group(group) is False

    def test_summarize_dropped_counts_tools(self):
        s = CompactionStrategy()
        msgs = [
            _assistant(tool_calls=[_tc("search", {}, tc_id="a")]),
            _assistant(tool_calls=[_tc("search", {}, tc_id="b")]),
            _assistant(tool_calls=[_tc("read_file", {}, tc_id="c")]),
        ]
        summary = s.summarize_dropped(msgs)
        assert "search: called 2 time(s)" in summary
        assert "read_file: called 1 time(s)" in summary

    def test_summarize_dropped_empty(self):
        s = CompactionStrategy()
        assert s.summarize_dropped([]) == ""


# ===================================================================
# 2. Custom token estimation
# ===================================================================

class TestCustomTokenEstimation:

    def test_custom_chars_per_token(self):
        """Different chars_per_token changes token estimates."""
        s_default = CompactionStrategy(chars_per_token=4.0)
        s_tight = CompactionStrategy(chars_per_token=2.0)
        msgs = [_system("a" * 100)]
        assert s_default.estimate_tokens(msgs) == 25
        assert s_tight.estimate_tokens(msgs) == 50

    def test_subclass_override_estimate_tokens(self):
        """Subclass can use a custom tokenizer."""

        class FixedTokenEstimator(CompactionStrategy):
            def estimate_tokens(self, messages):
                # Fixed: 10 tokens per message
                return len(messages) * 10

        s = FixedTokenEstimator(max_context_tokens=100)
        msgs = [_system("x" * 1000), _user("y" * 1000)]
        assert s.estimate_tokens(msgs) == 20  # 2 messages * 10


# ===================================================================
# 3. Custom preservation rules
# ===================================================================

class TestCustomPreservation:

    def test_subclass_preserve_search_results(self):
        """Subclass can preserve search results instead of just findings."""

        class PreserveSearchStrategy(CompactionStrategy):
            def should_preserve_group(self, group):
                for msg in group:
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            if tc.name in ("search", "log_finding", "get_findings"):
                                return True
                return False

        s = PreserveSearchStrategy(
            max_context_tokens=200,
            soft_compaction_pct=0.10,
            chars_per_token=4.0,
        )
        msgs = [_system("S" * 40), _user("U" * 40)]

        # Add non-search groups
        for i in range(4):
            tc = _tc("other_tool", {"data": "x" * 80}, tc_id=f"tc-{i}")
            msgs.append(_assistant(tool_calls=[tc]))
            msgs.append(_tool("R" * 80, tool_call_id=tc.id))

        # Add a search group in the middle
        search_tc = _tc("search", {"q": "important"}, tc_id="tc-search")
        msgs.insert(4, _assistant(tool_calls=[search_tc]))
        msgs.insert(5, _tool("search results", tool_call_id="tc-search"))

        # Add more groups to make recent (last 3)
        for i in range(10, 13):
            tc = _tc("other_tool", {"data": "x" * 80}, tc_id=f"tc-{i}")
            msgs.append(_assistant(tool_calls=[tc]))
            msgs.append(_tool("R" * 80, tool_call_id=tc.id))

        result = s.compact_if_needed(msgs)
        # Search group should be preserved
        has_search = any(
            m.tool_calls and any(tc.name == "search" for tc in m.tool_calls)
            for m in result
        )
        assert has_search


# ===================================================================
# 4. Custom summarization
# ===================================================================

class TestCustomSummarization:

    def test_subclass_custom_summary(self):
        """Subclass can provide custom summarization."""

        class VerboseSummaryStrategy(CompactionStrategy):
            def summarize_dropped(self, messages):
                count = sum(
                    len(m.tool_calls) for m in messages if m.tool_calls
                )
                if count == 0:
                    return ""
                return f"[CUSTOM] Dropped {count} tool calls from history."

        s = VerboseSummaryStrategy()
        msgs = [
            _assistant(tool_calls=[_tc("a", {}, tc_id="1")]),
            _assistant(tool_calls=[_tc("b", {}, tc_id="2"), _tc("c", {}, tc_id="3")]),
        ]
        summary = s.summarize_dropped(msgs)
        assert "[CUSTOM] Dropped 3 tool calls" in summary


# ===================================================================
# 5. Complete override
# ===================================================================

class TestCompleteOverride:

    def test_override_compact_if_needed(self):
        """Subclass can completely replace compaction logic."""

        class NeverCompactStrategy(CompactionStrategy):
            def compact_if_needed(self, messages, soft_fraction=None):
                return messages  # Never compact

        s = NeverCompactStrategy(
            max_context_tokens=50,
            soft_compaction_pct=0.01,
            chars_per_token=4.0,
        )
        msgs = _make_long_messages(n_groups=10)
        result = s.compact_if_needed(msgs)
        assert result is msgs  # No compaction

    def test_override_truncate_if_needed(self):
        """Subclass can completely replace truncation logic."""

        class AggressiveTruncateStrategy(CompactionStrategy):
            def truncate_if_needed(self, messages, max_fraction=None):
                # Always keep only system + user + last message
                if len(messages) > 3:
                    return messages[:2] + messages[-1:]
                return messages

        s = AggressiveTruncateStrategy()
        msgs = _make_long_messages(n_groups=6)
        result = s.truncate_if_needed(msgs)
        assert len(result) == 3
        assert result[0].role == MessageRole.SYSTEM
        assert result[1].role == MessageRole.USER


# ===================================================================
# 6. Integration with ContextManager
# ===================================================================

class TestContextManagerIntegration:

    def test_custom_strategy_via_constructor(self):
        """ContextManager accepts a custom CompactionStrategy."""

        class CountingStrategy(CompactionStrategy):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.compact_calls = 0
                self.truncate_calls = 0

            def compact_if_needed(self, messages, soft_fraction=None):
                self.compact_calls += 1
                return messages

            def truncate_if_needed(self, messages, max_fraction=None):
                self.truncate_calls += 1
                return messages

        strategy = CountingStrategy(max_context_tokens=1000)
        cm = ContextManager(compaction_strategy=strategy)

        msgs = [_system("test"), _user("test")]
        cm.compact_if_needed(msgs)
        cm.truncate_if_needed(msgs)

        assert strategy.compact_calls == 1
        assert strategy.truncate_calls == 1

    def test_strategy_accessible_via_property(self):
        strategy = CompactionStrategy(max_context_tokens=50000)
        cm = ContextManager(compaction_strategy=strategy)
        assert cm.compaction_strategy is strategy

    def test_default_strategy_created_from_params(self):
        """When no strategy provided, one is created from constructor params."""
        cm = ContextManager(
            max_context_tokens=50000,
            chars_per_token=3.0,
            soft_compaction_pct=0.30,
        )
        s = cm.compaction_strategy
        assert s.max_context_tokens == 50000
        assert s._chars_per_token == 3.0
        assert s._soft_compaction_pct == 0.30

    def test_estimate_tokens_delegates_to_strategy(self):
        """ContextManager.estimate_tokens delegates to strategy."""

        class AlwaysFifty(CompactionStrategy):
            def estimate_tokens(self, messages):
                return 50

        cm = ContextManager(compaction_strategy=AlwaysFifty())
        assert cm.estimate_tokens([_system("anything")]) == 50

    def test_backward_compat_pair_messages(self):
        """ContextManager._pair_messages still works via delegation."""
        tc = _tc("tool_a", {}, tc_id="tc-1")
        msgs = [
            _assistant(tool_calls=[tc]),
            _tool("result", tool_call_id="tc-1"),
        ]
        groups = ContextManager._pair_messages(msgs)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_backward_compat_contains_findings(self):
        """ContextManager._contains_findings still works via delegation."""
        cm = ContextManager()
        tc = _tc("log_finding", {"category": "risk"})
        msg = _assistant(tool_calls=[tc])
        assert cm._contains_findings(msg) is True

    def test_backward_compat_summarize_dropped(self):
        """ContextManager._summarize_dropped still works via delegation."""
        cm = ContextManager()
        msgs = [_assistant(tool_calls=[_tc("search", {}, tc_id="a")])]
        summary = cm._summarize_dropped(msgs)
        assert "search: called 1 time(s)" in summary


# ===================================================================
# 7. Max context tokens property
# ===================================================================

class TestMaxContextTokens:

    def test_max_context_tokens_property(self):
        s = CompactionStrategy(max_context_tokens=64000)
        assert s.max_context_tokens == 64000

    def test_context_manager_inherits_from_strategy(self):
        strategy = CompactionStrategy(max_context_tokens=32000)
        cm = ContextManager(compaction_strategy=strategy)
        assert cm._max_context_tokens == 32000
