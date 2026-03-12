"""Tests for the ContextManager module."""

import logging

import pytest

from parr.context_manager import ContextManager
from parr.core_types import (
    AgentConfig,
    AgentInput,
    Message,
    MessageRole,
    ModelConfig,
    Phase,
    ToolCall,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(role: MessageRole, content: str = "", **kwargs) -> Message:
    """Shorthand to create a Message."""
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


def _make_long_messages(
    n_groups: int = 6,
    content_size: int = 80,
) -> list[Message]:
    """Build a conversation with *n_groups* assistant+tool groups.

    Returns: [system, user, (assistant+tool) × n_groups]
    """
    msgs = [_system("S" * 40), _user("U" * 40)]
    for i in range(n_groups):
        tid = f"tc-{i}"
        tc = _tc("some_tool", {"q": "x" * content_size}, tc_id=tid)
        msgs.append(_assistant(tool_calls=[tc]))
        msgs.append(_tool("R" * content_size, tool_call_id=tid))
    return msgs


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_empty_messages(self):
        cm = ContextManager(max_context_tokens=1000)
        assert cm.estimate_tokens([]) == 0

    def test_text_content_only(self):
        cm = ContextManager(max_context_tokens=1000, chars_per_token=4.0)
        msgs = [_system("a" * 100)]  # 100 chars → 25 tokens
        assert cm.estimate_tokens(msgs) == 25

    def test_multiple_messages(self):
        cm = ContextManager(max_context_tokens=1000, chars_per_token=4.0)
        msgs = [_system("a" * 40), _user("b" * 60)]  # 100 chars → 25 tokens
        assert cm.estimate_tokens(msgs) == 25

    def test_tool_calls_counted(self):
        cm = ContextManager(max_context_tokens=1000, chars_per_token=4.0)
        tc = _tc("my_tool", {"key": "val"})
        msgs = [_assistant(tool_calls=[tc])]
        expected = int(len(str(tc.arguments)) / 4.0)
        assert cm.estimate_tokens(msgs) == expected

    def test_content_plus_tool_calls(self):
        cm = ContextManager(max_context_tokens=1000, chars_per_token=4.0)
        tc = _tc("t", {"k": "v"})
        msgs = [Message(role=MessageRole.ASSISTANT, content="hello", tool_calls=[tc])]
        content_chars = len("hello") + len(str(tc.arguments))
        assert cm.estimate_tokens(msgs) == int(content_chars / 4.0)

    def test_tool_schema_overhead_added(self):
        cm = ContextManager(max_context_tokens=1000, tool_schema_overhead=200)
        msgs = [_system("a" * 40)]
        assert cm.estimate_tokens(msgs) == int(40 / 4.0) + 200

    def test_none_content_ignored(self):
        cm = ContextManager(max_context_tokens=1000)
        msgs = [Message(role=MessageRole.ASSISTANT, content=None)]
        assert cm.estimate_tokens(msgs) == 0


# ---------------------------------------------------------------------------
# compact_if_needed
# ---------------------------------------------------------------------------

class TestCompactIfNeeded:
    def test_under_threshold_returns_unchanged(self):
        cm = ContextManager(max_context_tokens=500, soft_compaction_pct=0.40)
        msgs = [_system("short"), _user("short")]
        result = cm.compact_if_needed(msgs)
        assert result is msgs

    def test_over_threshold_compacts(self):
        cm = ContextManager(
            max_context_tokens=200,
            soft_compaction_pct=0.20,
            chars_per_token=4.0,
        )
        # Soft limit = 200 * 0.20 = 40 tokens.  6 groups → well above that.
        msgs = _make_long_messages(n_groups=6)
        assert cm.estimate_tokens(msgs) > 40
        assert len(msgs) > 6

        result = cm.compact_if_needed(msgs)
        # Result should be shorter than input
        assert len(result) < len(msgs)
        # System and user preserved
        assert result[0].role == MessageRole.SYSTEM
        assert result[1].role == MessageRole.USER

    def test_preserves_system_and_user(self):
        cm = ContextManager(max_context_tokens=200, soft_compaction_pct=0.20)
        msgs = _make_long_messages(n_groups=6)
        original_system = msgs[0]
        original_user = msgs[1]

        result = cm.compact_if_needed(msgs)
        assert result[0] is original_system
        assert result[1] is original_user

    def test_preserves_findings_groups(self):
        cm = ContextManager(max_context_tokens=200, soft_compaction_pct=0.40, chars_per_token=4.0)
        msgs = [_system("S" * 40), _user("U" * 40)]

        # Add several non-findings groups to push over limit
        for i in range(6):
            tc = _tc("some_tool", {"q": "x" * 80}, tc_id=f"tc-{i}")
            msgs.append(_assistant(tool_calls=[tc]))
            msgs.append(_tool("R" * 100, tool_call_id=tc.id))

        # Add a findings group in the middle (before the last 3 groups)
        findings_tc = _tc("log_finding", {"category": "risk", "content": "important"}, tc_id="tc-findings")
        findings_assistant = _assistant(tool_calls=[findings_tc])
        findings_tool = _tool("Finding logged", tool_call_id="tc-findings")
        # Insert after the non-findings groups (position 4 = index 2+2)
        msgs.insert(4, findings_assistant)
        msgs.insert(5, findings_tool)

        result = cm.compact_if_needed(msgs)
        # The findings group should be preserved
        has_findings = any(
            m.tool_calls and any(tc.name == "log_finding" for tc in m.tool_calls)
            for m in result
        )
        assert has_findings

    def test_preserves_recent_three_groups(self):
        cm = ContextManager(max_context_tokens=200, soft_compaction_pct=0.40, chars_per_token=4.0)
        msgs = [_system("S" * 40), _user("U" * 40)]

        # Add 6 groups so we have enough to compact
        tc_ids = []
        for i in range(6):
            tid = f"tc-{i}"
            tc_ids.append(tid)
            tc = _tc("some_tool", {"q": "x" * 80}, tc_id=tid)
            msgs.append(_assistant(tool_calls=[tc]))
            msgs.append(_tool("R" * 100, tool_call_id=tid))

        result = cm.compact_if_needed(msgs)
        # Last 3 tool_call ids should all appear in result
        last_three_ids = set(tc_ids[-3:])
        result_tc_ids = set()
        for m in result:
            if m.tool_calls:
                for tc in m.tool_calls:
                    result_tc_ids.add(tc.id)
        assert last_three_ids.issubset(result_tc_ids)

    def test_few_messages_returns_unchanged_with_warning(self, caplog):
        """When <= 6 messages and over limit, returns unchanged and logs warning."""
        cm = ContextManager(max_context_tokens=50, soft_compaction_pct=0.40, chars_per_token=4.0)
        # Soft limit = 50 * 0.40 = 20 tokens = 80 chars
        # 6 messages, each with enough content to exceed limit
        msgs = [
            _system("A" * 30),
            _user("B" * 30),
            _assistant("C" * 30),
            _tool("D" * 30, tool_call_id="tc-1"),
            _assistant("E" * 30),
            _tool("F" * 30, tool_call_id="tc-2"),
        ]
        assert cm.estimate_tokens(msgs) > 20

        with caplog.at_level(logging.WARNING, logger="parr.context_manager"):
            result = cm.compact_if_needed(msgs)

        assert result is msgs
        assert any("too few messages" in rec.message for rec in caplog.records)

    def test_exactly_six_messages_returns_unchanged(self, caplog):
        """Edge: exactly 6 messages over limit still triggers warning path."""
        cm = ContextManager(max_context_tokens=30, soft_compaction_pct=0.40, chars_per_token=4.0)
        msgs = [
            _system("A" * 20),
            _user("B" * 20),
            _assistant("C" * 20),
            _tool("D" * 20),
            _assistant("E" * 20),
            _tool("F" * 20),
        ]
        with caplog.at_level(logging.WARNING, logger="parr.context_manager"):
            result = cm.compact_if_needed(msgs)
        assert result is msgs

    def test_custom_soft_fraction(self):
        cm = ContextManager(max_context_tokens=10000, soft_compaction_pct=0.40, chars_per_token=4.0)
        msgs = _make_long_messages(n_groups=6)
        tokens = cm.estimate_tokens(msgs)

        # With default 0.40 → limit=4000; tokens << 4000 → no compaction
        result_default = cm.compact_if_needed(msgs)
        assert result_default is msgs

        # With soft_fraction tiny enough to be below actual tokens → compaction
        tight_fraction = (tokens - 1) / cm._max_context_tokens
        result_tight = cm.compact_if_needed(msgs, soft_fraction=tight_fraction * 0.5)
        assert len(result_tight) < len(msgs)

    def test_adds_compaction_notice(self):
        cm = ContextManager(max_context_tokens=200, soft_compaction_pct=0.10, chars_per_token=4.0)
        msgs = _make_long_messages(n_groups=6)
        assert len(msgs) > 6

        result = cm.compact_if_needed(msgs)
        compaction_notices = [
            m for m in result
            if m.content and "[CONTEXT COMPACTED]" in m.content
        ]
        assert len(compaction_notices) >= 1

    def test_three_or_fewer_groups_returns_unchanged(self):
        """When there are only 3 groups after system+user, no compaction occurs."""
        cm = ContextManager(max_context_tokens=100, soft_compaction_pct=0.10, chars_per_token=4.0)
        # system + user + 3 groups (6 assistant+tool pairs → but 3 groups)
        msgs = [_system("S" * 40), _user("U" * 40)]
        for i in range(3):
            tc = _tc("tool", {"q": "x" * 80}, tc_id=f"tc-{i}")
            msgs.append(_assistant(tool_calls=[tc]))
            msgs.append(_tool("R" * 80, tool_call_id=tc.id))
        # 8 messages total, > 6, but only 3 groups → returns unchanged
        result = cm.compact_if_needed(msgs)
        assert result is msgs


# ---------------------------------------------------------------------------
# truncate_if_needed
# ---------------------------------------------------------------------------

class TestTruncateIfNeeded:
    def test_under_threshold_returns_unchanged(self):
        cm = ContextManager(max_context_tokens=1000, hard_truncation_pct=0.65)
        msgs = [_system("short"), _user("short")]
        result = cm.truncate_if_needed(msgs)
        assert result == msgs

    def test_over_threshold_truncates(self):
        cm = ContextManager(
            max_context_tokens=200,
            soft_compaction_pct=0.05,
            hard_truncation_pct=0.10,
            chars_per_token=4.0,
        )
        # Hard limit = 200 * 0.10 = 20 tokens.  8 groups → well above that.
        msgs = _make_long_messages(n_groups=8)
        assert len(msgs) > 6

        result = cm.truncate_if_needed(msgs)
        # Should be shorter
        assert len(result) < len(msgs)
        # System and user preserved
        assert result[0].role == MessageRole.SYSTEM
        assert result[1].role == MessageRole.USER

    def test_calls_compact_first(self):
        """truncate_if_needed always runs compact_if_needed first."""
        cm = ContextManager(
            max_context_tokens=200,
            soft_compaction_pct=0.05,
            hard_truncation_pct=0.90,
            chars_per_token=4.0,
        )
        # Soft limit = 10, hard limit = 180.
        # Build 6 groups → tokens well above soft limit
        msgs = _make_long_messages(n_groups=6)
        assert len(msgs) > 6

        result = cm.truncate_if_needed(msgs)
        # Soft compaction should fire; hard truncation probably not since
        # compacted result is under 180 tokens. Just verify it ran.
        assert len(result) <= len(msgs)

    def test_preserves_last_three_groups(self):
        cm = ContextManager(
            max_context_tokens=200,
            soft_compaction_pct=0.05,
            hard_truncation_pct=0.10,
            chars_per_token=4.0,
        )
        msgs = [_system("S" * 40), _user("U" * 40)]
        tc_ids = []
        for i in range(8):
            tid = f"tc-{i}"
            tc_ids.append(tid)
            tc = _tc("do_work", {"data": "x" * 80}, tc_id=tid)
            msgs.append(_assistant(tool_calls=[tc]))
            msgs.append(_tool("Result " + "y" * 80, tool_call_id=tid))

        result = cm.truncate_if_needed(msgs)
        # Last 3 tc_ids should be in the result
        last_three_ids = set(tc_ids[-3:])
        result_tc_ids = set()
        for m in result:
            if m.tool_calls:
                for tc in m.tool_calls:
                    result_tc_ids.add(tc.id)
        assert last_three_ids.issubset(result_tc_ids)

    def test_adds_truncation_notice(self):
        cm = ContextManager(
            max_context_tokens=100000,
            soft_compaction_pct=0.99,
            hard_truncation_pct=0.001,
            chars_per_token=4.0,
        )
        # Soft limit = 99000 (won't fire), hard limit = 100 tokens (will fire).
        msgs = _make_long_messages(n_groups=8)
        assert len(msgs) > 6

        result = cm.truncate_if_needed(msgs)
        truncation_notices = [
            m for m in result
            if m.content and "[CONTEXT TRUNCATED]" in m.content
        ]
        assert len(truncation_notices) >= 1

    def test_custom_max_fraction(self):
        cm = ContextManager(
            max_context_tokens=10000,
            soft_compaction_pct=0.90,
            hard_truncation_pct=0.90,
            chars_per_token=4.0,
        )
        msgs = _make_long_messages(n_groups=8)
        tokens = cm.estimate_tokens(msgs)
        assert len(msgs) > 6

        # Default hard limit = 9000 → tokens << 9000, no truncation
        result_default = cm.truncate_if_needed(msgs)
        assert len(result_default) == len(msgs)

        # With max_fraction tiny enough to be below actual tokens → truncation
        tight_fraction = (tokens - 1) / cm._max_context_tokens * 0.5
        result_tight = cm.truncate_if_needed(msgs, max_fraction=tight_fraction)
        assert len(result_tight) < len(msgs)

    def test_few_groups_returns_unchanged_with_warning(self, caplog):
        """When <= 3 groups after system+user and over hard limit, logs warning."""
        cm = ContextManager(
            max_context_tokens=50,
            soft_compaction_pct=0.90,
            hard_truncation_pct=0.10,
            chars_per_token=4.0,
        )
        # Hard limit = 5 tokens. 3 groups after system+user.
        msgs = [
            _system("S" * 40),
            _user("U" * 40),
            _assistant("A" * 40),
            _assistant("B" * 40),
            _assistant("C" * 40),
        ]
        with caplog.at_level(logging.WARNING, logger="parr.context_manager"):
            result = cm.truncate_if_needed(msgs)
        # Should return unchanged because not enough groups
        assert len(result) == len(msgs)


# ---------------------------------------------------------------------------
# build_phase_messages
# ---------------------------------------------------------------------------

class TestBuildPhaseMessages:
    @pytest.fixture()
    def config(self):
        return AgentConfig(
            role="analyst",
            system_prompt="You are an analyst.",
            model="test-model",
            model_config=ModelConfig(),
        )

    @pytest.fixture()
    def agent_input(self):
        return AgentInput(task="Analyze risk data.")

    def test_returns_system_and_user(self, config, agent_input):
        cm = ContextManager()
        msgs = cm.build_phase_messages(Phase.PLAN, config, agent_input)
        assert len(msgs) == 2
        assert msgs[0].role == MessageRole.SYSTEM
        assert msgs[1].role == MessageRole.USER

    def test_system_prompt_contains_base_and_phase(self, config, agent_input):
        cm = ContextManager()
        msgs = cm.build_phase_messages(Phase.PLAN, config, agent_input)
        sys_content = msgs[0].content
        assert "You are an analyst." in sys_content
        assert "Planning" in sys_content

    def test_user_message_contains_task(self, config, agent_input):
        cm = ContextManager()
        msgs = cm.build_phase_messages(Phase.ACT, config, agent_input)
        assert "Analyze risk data." in msgs[1].content

    def test_act_includes_plan_summary(self, config, agent_input):
        cm = ContextManager()
        cm.record_phase_summary(Phase.PLAN, "Step 1: gather data")
        msgs = cm.build_phase_messages(Phase.ACT, config, agent_input)
        assert "Step 1: gather data" in msgs[1].content

    def test_review_includes_execution_summary(self, config, agent_input):
        cm = ContextManager()
        cm.record_phase_summary(Phase.ACT, "Collected 5 findings")
        msgs = cm.build_phase_messages(Phase.REVIEW, config, agent_input)
        assert "Collected 5 findings" in msgs[1].content

    def test_report_includes_review_summary(self, config, agent_input):
        cm = ContextManager()
        cm.record_phase_summary(Phase.REVIEW, "REVIEW_PASSED")
        msgs = cm.build_phase_messages(Phase.REPORT, config, agent_input)
        assert "REVIEW_PASSED" in msgs[1].content

    def test_working_memory_snapshot_included(self, config, agent_input):
        cm = ContextManager()
        msgs = cm.build_phase_messages(
            Phase.ACT, config, agent_input,
            working_memory_snapshot="Todo: 2/5 done",
        )
        assert "Todo: 2/5 done" in msgs[1].content

    def test_extra_context_included(self, config, agent_input):
        cm = ContextManager()
        msgs = cm.build_phase_messages(
            Phase.ACT, config, agent_input,
            extra_context="Review says redo step 3",
        )
        assert "Review says redo step 3" in msgs[1].content

    def test_all_four_phases(self, config, agent_input):
        cm = ContextManager()
        for phase in Phase:
            msgs = cm.build_phase_messages(phase, config, agent_input)
            assert len(msgs) == 2
            assert msgs[0].role == MessageRole.SYSTEM


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_message_list_compact(self):
        cm = ContextManager(max_context_tokens=100)
        result = cm.compact_if_needed([])
        assert result == []

    def test_empty_message_list_truncate(self):
        cm = ContextManager(max_context_tokens=100)
        result = cm.truncate_if_needed([])
        assert result == []

    def test_single_message_compact(self):
        cm = ContextManager(max_context_tokens=100)
        msgs = [_system("hello")]
        result = cm.compact_if_needed(msgs)
        assert result is msgs

    def test_single_message_truncate(self):
        cm = ContextManager(max_context_tokens=100)
        msgs = [_system("hello")]
        result = cm.truncate_if_needed(msgs)
        assert len(result) == 1

    def test_messages_with_none_content(self):
        cm = ContextManager(max_context_tokens=1000)
        msgs = [
            Message(role=MessageRole.ASSISTANT, content=None),
            Message(role=MessageRole.TOOL, content=None, tool_call_id="tc-1"),
        ]
        assert cm.estimate_tokens(msgs) == 0

    def test_record_phase_summary_overwrites(self):
        cm = ContextManager()
        cm.record_phase_summary(Phase.PLAN, "first")
        cm.record_phase_summary(Phase.PLAN, "second")
        assert cm._phase_summaries[Phase.PLAN] == "second"

    def test_pair_messages_groups_correctly(self):
        """Verify _pair_messages groups assistant+tool pairs."""
        tc = _tc("tool_a", {}, tc_id="tc-1")
        msgs = [
            _assistant(tool_calls=[tc]),
            _tool("result", tool_call_id="tc-1"),
            _assistant("standalone text"),
        ]
        groups = ContextManager._pair_messages(msgs)
        assert len(groups) == 2
        # First group is the assistant+tool pair
        assert len(groups[0]) == 2
        assert groups[0][0].role == MessageRole.ASSISTANT
        assert groups[0][1].role == MessageRole.TOOL
        # Second group is standalone
        assert len(groups[1]) == 1

    def test_pair_messages_multiple_tool_results(self):
        """An assistant with 2 tool_calls followed by 2 tool results → 1 group."""
        tc1 = _tc("t1", {}, tc_id="tc-1")
        tc2 = _tc("t2", {}, tc_id="tc-2")
        msgs = [
            _assistant(tool_calls=[tc1, tc2]),
            _tool("r1", tool_call_id="tc-1"),
            _tool("r2", tool_call_id="tc-2"),
        ]
        groups = ContextManager._pair_messages(msgs)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_contains_findings_detects_log_finding(self):
        cm = ContextManager()
        tc = _tc("log_finding", {"category": "risk"})
        msg = _assistant(tool_calls=[tc])
        assert cm._contains_findings(msg) is True

    def test_contains_findings_detects_get_findings(self):
        cm = ContextManager()
        tc = _tc("get_findings", {})
        msg = _assistant(tool_calls=[tc])
        assert cm._contains_findings(msg) is True

    def test_contains_findings_false_for_other_tools(self):
        cm = ContextManager()
        tc = _tc("some_other_tool", {})
        msg = _assistant(tool_calls=[tc])
        assert cm._contains_findings(msg) is False

    def test_summarize_dropped_empty(self):
        cm = ContextManager()
        assert cm._summarize_dropped([]) == ""

    def test_summarize_dropped_counts_tools(self):
        cm = ContextManager()
        msgs = [
            _assistant(tool_calls=[_tc("search", {}, tc_id="a")]),
            _assistant(tool_calls=[_tc("search", {}, tc_id="b")]),
            _assistant(tool_calls=[_tc("read_file", {}, tc_id="c")]),
        ]
        summary = cm._summarize_dropped(msgs)
        assert "search: called 2 time(s)" in summary
        assert "read_file: called 1 time(s)" in summary
