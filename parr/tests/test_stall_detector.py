"""Tests for parr.stall_detector — StallDetector pluggable stall detection."""

from __future__ import annotations

from typing import Any, List, Optional

import pytest

from parr.core_types import Phase, StallDetectionConfig, ToolCall, ToolDef
from parr.stall_detector import StallDetector, StallVerdict
from parr.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(
    name: str,
    is_framework_tool: bool = False,
    is_read_only: bool = False,
    marks_progress: bool = False,
    **overrides: Any,
) -> ToolDef:
    return ToolDef(
        name=name,
        description=f"Test tool {name}",
        parameters={"type": "object", "properties": {}},
        is_framework_tool=is_framework_tool,
        is_read_only=is_read_only,
        marks_progress=marks_progress,
        **overrides,
    )


def _tc(name: str, **kwargs: Any) -> ToolCall:
    return ToolCall(id=f"call_{name}", name=name, arguments=kwargs)


def _make_detector(
    tools: Optional[List[ToolDef]] = None,
    config: Optional[StallDetectionConfig] = None,
) -> StallDetector:
    registry = ToolRegistry()
    for t in (tools or []):
        registry.register(t)
    return StallDetector(registry=registry, config=config)


# Default framework tools for most tests
_DEFAULT_TOOLS = [
    _make_tool("get_todo_list", is_framework_tool=True, is_read_only=True),
    _make_tool("get_findings", is_framework_tool=True, is_read_only=True),
    _make_tool("log_finding", is_framework_tool=True, marks_progress=True),
    _make_tool("mark_todo_complete", is_framework_tool=True, marks_progress=True),
    _make_tool("search_documents"),  # domain tool
    _make_tool("analyze_data"),  # domain tool
]


# ===================================================================
# 1. No stall on domain tool calls
# ===================================================================

class TestDomainToolCalls:

    def test_domain_tool_never_stalls(self):
        detector = _make_detector(_DEFAULT_TOOLS)
        for i in range(20):
            verdict = detector.check_iteration([_tc("search_documents", query=f"q_{i}")])
            assert not verdict.is_stalled

    def test_mixed_domain_and_framework_never_stalls(self):
        detector = _make_detector(_DEFAULT_TOOLS)
        for i in range(20):
            verdict = detector.check_iteration([
                _tc("get_todo_list"),
                _tc("search_documents", query=f"q_{i}"),
            ])
            assert not verdict.is_stalled


# ===================================================================
# 2. Read-only stall detection
# ===================================================================

class TestReadOnlyStall:

    def test_read_only_stall_triggers(self):
        config = StallDetectionConfig(max_framework_stall_iterations=3)
        detector = _make_detector(_DEFAULT_TOOLS, config=config)

        for _ in range(3):
            verdict = detector.check_iteration([_tc("get_todo_list")])

        assert verdict.is_stalled
        assert verdict.reason == "read_only_stall"

    def test_read_only_stall_warning_before_trigger(self):
        config = StallDetectionConfig(max_framework_stall_iterations=5)
        detector = _make_detector(_DEFAULT_TOOLS, config=config)

        warned = False
        for _ in range(5):
            verdict = detector.check_iteration([_tc("get_todo_list")])
            if verdict.should_warn:
                warned = True
                assert "STALL WARNING" in verdict.warning_message

        assert warned, "Expected a warning before force-completion"

    def test_read_only_resets_on_domain_tool(self):
        config = StallDetectionConfig(max_framework_stall_iterations=3)
        detector = _make_detector(_DEFAULT_TOOLS, config=config)

        # 2 read-only iterations
        detector.check_iteration([_tc("get_todo_list")])
        detector.check_iteration([_tc("get_findings")])

        # Domain tool resets counter
        verdict = detector.check_iteration([_tc("search_documents")])
        assert not verdict.is_stalled

        # 2 more read-only — still under threshold
        detector.check_iteration([_tc("get_todo_list")])
        verdict = detector.check_iteration([_tc("get_todo_list")])
        assert not verdict.is_stalled

    def test_read_only_resets_on_progress_tool(self):
        config = StallDetectionConfig(max_framework_stall_iterations=3)
        detector = _make_detector(_DEFAULT_TOOLS, config=config)

        detector.check_iteration([_tc("get_todo_list")])
        detector.check_iteration([_tc("get_todo_list")])

        # Progress tool is not read-only → resets read-only counter
        verdict = detector.check_iteration([_tc("log_finding", category="x", content="y", source="z")])
        assert not verdict.is_stalled


# ===================================================================
# 3. Framework-only loop detection
# ===================================================================

class TestFrameworkOnlyLoop:

    def test_fw_only_loop_triggers(self):
        config = StallDetectionConfig(max_fw_only_consecutive_iterations=4)
        detector = _make_detector([
            _make_tool("create_todo_list", is_framework_tool=True),
            _make_tool("update_todo_list", is_framework_tool=True),
        ], config=config)

        for _ in range(4):
            verdict = detector.check_iteration([_tc("create_todo_list")])

        assert verdict.is_stalled
        assert verdict.reason == "framework_only_loop"

    def test_progress_tool_resets_fw_counter(self):
        config = StallDetectionConfig(max_fw_only_consecutive_iterations=4)
        detector = _make_detector(_DEFAULT_TOOLS, config=config)

        # 3 non-progress framework iterations
        for _ in range(3):
            detector.check_iteration([_tc("get_todo_list")])

        # Progress tool resets the fw-only counter
        detector.check_iteration([_tc("log_finding", category="x", content="y", source="z")])

        # 3 more won't trigger (reset happened)
        for _ in range(3):
            verdict = detector.check_iteration([_tc("get_todo_list")])

        assert not verdict.is_stalled


# ===================================================================
# 4. Duplicate call detection
# ===================================================================

class TestDuplicateDetection:

    def test_duplicate_loop_triggers(self):
        config = StallDetectionConfig(max_duplicate_call_iterations=3)
        detector = _make_detector(_DEFAULT_TOOLS, config=config)

        # First call establishes the signature
        detector.check_iteration([_tc("search_documents", query="test")])

        # 3 consecutive duplicates trigger stall
        for _ in range(3):
            verdict = detector.check_iteration([_tc("search_documents", query="test")])

        assert verdict.is_stalled
        assert verdict.reason == "duplicate_loop"

    def test_duplicate_warning_at_two(self):
        config = StallDetectionConfig(max_duplicate_call_iterations=5)
        detector = _make_detector(_DEFAULT_TOOLS, config=config)

        detector.check_iteration([_tc("search_documents", query="test")])

        warned = False
        for _ in range(5):
            verdict = detector.check_iteration([_tc("search_documents", query="test")])
            if verdict.should_warn:
                warned = True
                assert "DUPLICATE" in verdict.warning_message

        assert warned

    def test_different_args_not_duplicate(self):
        config = StallDetectionConfig(max_duplicate_call_iterations=3)
        detector = _make_detector(_DEFAULT_TOOLS, config=config)

        for i in range(10):
            verdict = detector.check_iteration([_tc("search_documents", query=f"query_{i}")])

        assert not verdict.is_stalled

    def test_duplicate_resets_on_new_call(self):
        config = StallDetectionConfig(max_duplicate_call_iterations=4)
        detector = _make_detector(_DEFAULT_TOOLS, config=config)

        detector.check_iteration([_tc("search_documents", query="test")])
        detector.check_iteration([_tc("search_documents", query="test")])
        detector.check_iteration([_tc("search_documents", query="test")])

        # Different call resets
        detector.check_iteration([_tc("analyze_data")])
        detector.check_iteration([_tc("search_documents", query="test")])
        verdict = detector.check_iteration([_tc("search_documents", query="test")])
        assert not verdict.is_stalled


# ===================================================================
# 5. Custom ToolDef flags
# ===================================================================

class TestToolDefFlags:

    def test_custom_tool_is_read_only(self):
        """Custom domain tool with is_read_only=True is treated as read-only."""
        config = StallDetectionConfig(max_framework_stall_iterations=3)
        tools = [
            _make_tool("my_status_check", is_framework_tool=True, is_read_only=True),
        ]
        detector = _make_detector(tools, config=config)

        for _ in range(3):
            verdict = detector.check_iteration([_tc("my_status_check")])

        assert verdict.is_stalled

    def test_custom_tool_marks_progress(self):
        """Custom tool with marks_progress=True resets fw-only counter."""
        config = StallDetectionConfig(max_fw_only_consecutive_iterations=4)
        tools = [
            _make_tool("get_todo_list", is_framework_tool=True, is_read_only=True),
            _make_tool("my_custom_save", is_framework_tool=True, marks_progress=True),
        ]
        detector = _make_detector(tools, config=config)

        for _ in range(3):
            detector.check_iteration([_tc("get_todo_list")])

        # Custom progress tool resets counter
        detector.check_iteration([_tc("my_custom_save")])

        for _ in range(3):
            verdict = detector.check_iteration([_tc("get_todo_list")])

        assert not verdict.is_stalled


# ===================================================================
# 6. Subclassing for custom behavior
# ===================================================================

class TestCustomDetector:

    def test_override_is_progress_tool(self):
        """Subclass can declare custom tools as progress-making."""

        class CustomDetector(StallDetector):
            def is_progress_tool(self, tool_name, tool_def):
                if tool_name == "run_analysis":
                    return True
                return super().is_progress_tool(tool_name, tool_def)

        tools = [
            _make_tool("run_analysis", is_framework_tool=True),
            _make_tool("get_todo_list", is_framework_tool=True, is_read_only=True),
        ]
        registry = ToolRegistry()
        for t in tools:
            registry.register(t)

        config = StallDetectionConfig(max_fw_only_consecutive_iterations=4)
        detector = CustomDetector(registry=registry, config=config)

        for _ in range(3):
            detector.check_iteration([_tc("get_todo_list")])

        # run_analysis is treated as progress by custom detector
        detector.check_iteration([_tc("run_analysis")])

        for _ in range(3):
            verdict = detector.check_iteration([_tc("get_todo_list")])

        assert not verdict.is_stalled

    def test_override_is_read_only_tool(self):
        """Subclass can declare custom tools as read-only."""

        class StrictDetector(StallDetector):
            def is_read_only_tool(self, tool_name, tool_def):
                if tool_name == "get_status":
                    return True
                return super().is_read_only_tool(tool_name, tool_def)

        tools = [_make_tool("get_status", is_framework_tool=True)]
        registry = ToolRegistry()
        for t in tools:
            registry.register(t)

        config = StallDetectionConfig(max_framework_stall_iterations=3)
        detector = StrictDetector(registry=registry, config=config)

        for _ in range(3):
            verdict = detector.check_iteration([_tc("get_status")])

        assert verdict.is_stalled

    def test_override_check_iteration_completely(self):
        """Subclass can completely replace stall logic."""

        class NeverStallDetector(StallDetector):
            def check_iteration(self, tool_calls):
                return StallVerdict()  # Never stall

        tools = [_make_tool("get_todo_list", is_framework_tool=True, is_read_only=True)]
        registry = ToolRegistry()
        for t in tools:
            registry.register(t)

        detector = NeverStallDetector(registry=registry)

        for _ in range(100):
            verdict = detector.check_iteration([_tc("get_todo_list")])
            assert not verdict.is_stalled


# ===================================================================
# 7. Reset behavior
# ===================================================================

class TestReset:

    def test_reset_clears_stall_state(self):
        config = StallDetectionConfig(max_framework_stall_iterations=3)
        detector = _make_detector(_DEFAULT_TOOLS, config=config)

        # Build up stall state
        detector.check_iteration([_tc("get_todo_list")])
        detector.check_iteration([_tc("get_todo_list")])

        # Reset
        detector.reset()

        # Counter starts fresh — 2 more iterations won't trigger
        detector.check_iteration([_tc("get_todo_list")])
        verdict = detector.check_iteration([_tc("get_todo_list")])
        assert not verdict.is_stalled

    def test_reset_clears_duplicate_state(self):
        config = StallDetectionConfig(max_duplicate_call_iterations=3)
        detector = _make_detector(_DEFAULT_TOOLS, config=config)

        detector.check_iteration([_tc("search_documents", query="test")])
        detector.check_iteration([_tc("search_documents", query="test")])

        detector.reset()

        # First after reset establishes new signature — won't be duplicate
        detector.check_iteration([_tc("search_documents", query="test")])
        verdict = detector.check_iteration([_tc("search_documents", query="test")])
        # Only 1 consecutive duplicate after reset, not enough
        assert not verdict.is_stalled


# ===================================================================
# 8. Legacy fallback (tools without flags)
# ===================================================================

class TestLegacyFallback:

    def test_legacy_read_only_tool_detected(self):
        """Tools named get_todo_list without is_read_only flag still detected."""
        config = StallDetectionConfig(max_framework_stall_iterations=3)
        tools = [
            # No is_read_only flag, but name matches legacy set
            _make_tool("get_todo_list", is_framework_tool=True),
        ]
        detector = _make_detector(tools, config=config)

        for _ in range(3):
            verdict = detector.check_iteration([_tc("get_todo_list")])

        assert verdict.is_stalled

    def test_legacy_progress_tool_detected(self):
        """Tools named log_finding without marks_progress flag still detected."""
        # Use high duplicate threshold so duplicate detection doesn't interfere
        config = StallDetectionConfig(
            max_fw_only_consecutive_iterations=5,
            max_duplicate_call_iterations=20,
        )
        tools = [
            _make_tool("get_todo_list", is_framework_tool=True),
            # No marks_progress flag, but name matches legacy set
            _make_tool("log_finding", is_framework_tool=True),
        ]
        detector = _make_detector(tools, config=config)

        for _ in range(4):
            detector.check_iteration([_tc("get_todo_list")])

        # Legacy progress tool resets counter
        detector.check_iteration([_tc("log_finding", category="x", content="y", source="z")])

        for _ in range(4):
            verdict = detector.check_iteration([_tc("get_todo_list")])

        assert not verdict.is_stalled


# ===================================================================
# 9. StallVerdict dataclass
# ===================================================================

class TestStallVerdict:

    def test_default_verdict_is_not_stalled(self):
        v = StallVerdict()
        assert not v.is_stalled
        assert v.reason is None
        assert not v.should_warn
        assert v.warning_message is None

    def test_stalled_verdict(self):
        v = StallVerdict(is_stalled=True, reason="read_only_stall")
        assert v.is_stalled
        assert v.reason == "read_only_stall"

    def test_warning_verdict(self):
        v = StallVerdict(should_warn=True, warning_message="Watch out!")
        assert not v.is_stalled
        assert v.should_warn
        assert v.warning_message == "Watch out!"
