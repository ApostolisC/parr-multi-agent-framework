"""
Pluggable Stall Detection for the Agentic Framework.

Detects when agents are stuck in loops and not making progress.
The default implementation uses ToolDef flags (is_read_only, marks_progress)
and duplicate call detection. Subclass StallDetector to customize behavior.

Three types of stall are detected:
1. Read-only stall: agent only calls read-only tools (get_todo_list, etc.)
2. Framework-only loop: agent calls only framework tools without progress
3. Duplicate call loop: agent repeats the exact same tool calls
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .core_types import StallDetectionConfig, ToolCall, ToolDef
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


# Legacy frozen sets kept for backward compatibility with tools that
# don't yet have is_read_only / marks_progress set on their ToolDef.
_LEGACY_READ_ONLY_TOOLS = frozenset({
    "get_todo_list",
    "get_findings",
})

_LEGACY_PROGRESS_TOOLS = frozenset({
    "log_finding",
    "batch_log_findings",
    "mark_todo_complete",
    "batch_mark_todo_complete",
    "review_checklist",
    "submit_report",
})


def _hash_tool_call(name: str, arguments: dict) -> str:
    """Produce a stable hash for a (tool_name, arguments) pair."""
    key = name + ":" + json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.md5(key.encode()).hexdigest()


@dataclass
class StallVerdict:
    """Result of a stall check for a single iteration."""
    is_stalled: bool = False
    reason: Optional[str] = None  # "read_only_stall", "framework_only_loop", "duplicate_loop"
    should_warn: bool = False
    warning_message: Optional[str] = None


class StallDetector:
    """
    Detects when an agent is stuck in non-productive loops.

    The default implementation checks three conditions:
    1. Read-only stall: consecutive iterations with only read-only tools
    2. Framework-only loop: consecutive iterations with only framework
       tools that don't mark progress
    3. Duplicate call loop: consecutive iterations repeating exact same calls

    Subclass and override methods to customize stall detection logic.
    For example, you might want to:
    - Change how tools are classified (override is_read_only / is_progress)
    - Add domain-specific stall patterns (override check_iteration)
    - Disable certain detectors (override and return StallVerdict())

    Example::

        class LenientStallDetector(StallDetector):
            def is_progress_tool(self, tool_name, tool_def):
                # Treat my custom analysis tool as progress
                if tool_name == "run_analysis":
                    return True
                return super().is_progress_tool(tool_name, tool_def)

        # Or via ToolDef flags (preferred, no subclassing needed):
        my_tool = ToolDef(
            name="run_analysis",
            marks_progress=True,  # Stall detector will recognize this
            ...
        )
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: Optional[StallDetectionConfig] = None,
    ) -> None:
        self._registry = registry
        self._config = config or StallDetectionConfig()
        self.reset()

    def reset(self) -> None:
        """Reset all stall detection state. Called at the start of each phase."""
        self._consecutive_stall_iterations = 0
        self._consecutive_fw_only_iterations = 0
        self._recent_iteration_signatures: List[Set[str]] = []
        self._consecutive_duplicate_iterations = 0
        self._stall_warning_given = False
        self._duplicate_warning_given = False

    @property
    def config(self) -> StallDetectionConfig:
        return self._config

    # -------------------------------------------------------------------
    # Tool classification — override these for custom behavior
    # -------------------------------------------------------------------

    def is_read_only_tool(self, tool_name: str, tool_def: Optional[ToolDef]) -> bool:
        """
        Determine if a tool is read-only (no state change).

        Checks ToolDef.is_read_only first, then falls back to legacy set.
        Override to add custom read-only tools without modifying ToolDef.
        """
        if tool_def is not None and tool_def.is_read_only:
            return True
        return tool_name in _LEGACY_READ_ONLY_TOOLS

    def is_progress_tool(self, tool_name: str, tool_def: Optional[ToolDef]) -> bool:
        """
        Determine if a tool represents genuine work output.

        Checks ToolDef.marks_progress first, then falls back to legacy set.
        Override to add custom progress tools without modifying ToolDef.
        """
        if tool_def is not None and tool_def.marks_progress:
            return True
        return tool_name in _LEGACY_PROGRESS_TOOLS

    def is_domain_tool(self, tool_name: str, tool_def: Optional[ToolDef]) -> bool:
        """
        Determine if a tool is a domain (non-framework) tool.

        Domain tool calls always reset stall counters.
        """
        if tool_def is None:
            return True  # Unknown tool = assume domain
        return not tool_def.is_framework_tool

    # -------------------------------------------------------------------
    # Main check — override for completely custom stall logic
    # -------------------------------------------------------------------

    def check_iteration(self, tool_calls: List[ToolCall]) -> StallVerdict:
        """
        Check a completed iteration for stall conditions.

        Args:
            tool_calls: The tool calls made in this iteration.

        Returns:
            StallVerdict indicating whether the agent is stalled.
        """
        tool_names = {tc.name for tc in tool_calls}
        tool_defs = {name: self._registry.get(name) for name in tool_names}

        # Check if any domain tools were called
        has_domain_calls = any(
            self.is_domain_tool(name, tool_defs.get(name))
            for name in tool_names
        )

        # --- Duplicate call detection ---
        dup_verdict = self._check_duplicates(tool_calls)
        if dup_verdict.is_stalled:
            return dup_verdict

        # Domain tools always reset stall counters
        if has_domain_calls:
            self._consecutive_stall_iterations = 0
            self._consecutive_fw_only_iterations = 0
            # Still surface duplicate warnings even for domain tools
            if dup_verdict.should_warn:
                return dup_verdict
            return StallVerdict()

        # --- Framework-only iteration analysis ---
        has_progress = any(
            self.is_progress_tool(name, tool_defs.get(name))
            for name in tool_names
        )

        all_read_only = all(
            self.is_read_only_tool(name, tool_defs.get(name))
            for name in tool_names
        )

        # Progress tools reset the framework-only counter
        if has_progress:
            self._consecutive_fw_only_iterations = 0
        else:
            self._consecutive_fw_only_iterations += 1

        # Read-only check
        if all_read_only:
            self._consecutive_stall_iterations += 1
        else:
            self._consecutive_stall_iterations = 0

        # --- Check thresholds ---

        # Warning before force-completion
        approaching_stall = (
            self._consecutive_stall_iterations == self._config.max_framework_stall_iterations - 2
            or self._consecutive_fw_only_iterations == self._config.max_fw_only_consecutive_iterations - 2
        )
        if approaching_stall and not self._stall_warning_given:
            self._stall_warning_given = True
            return StallVerdict(
                is_stalled=False,
                should_warn=True,
                warning_message=(
                    "[STALL WARNING] You have been calling only "
                    "framework tools (get_todo_list, get_findings, "
                    "etc.) without making progress with domain tools. "
                    "This pattern suggests you may be stuck. Consider: "
                    "(1) calling domain tools to fetch or process data, "
                    "(2) using log_finding or mark_todo_complete to "
                    "record progress, or (3) producing your final "
                    "response. The phase will be force-completed if "
                    "no domain tool progress is detected soon."
                ),
            )

        # Force-completion
        hit_read_only = self._consecutive_stall_iterations >= self._config.max_framework_stall_iterations
        hit_fw_only = self._consecutive_fw_only_iterations >= self._config.max_fw_only_consecutive_iterations

        if hit_read_only or hit_fw_only:
            reason = "read_only_stall" if hit_read_only else "framework_only_loop"
            return StallVerdict(is_stalled=True, reason=reason)

        return StallVerdict()

    # -------------------------------------------------------------------
    # Duplicate detection (internal)
    # -------------------------------------------------------------------

    def _check_duplicates(self, tool_calls: List[ToolCall]) -> StallVerdict:
        """Check for duplicate tool call patterns across iterations."""
        iter_sigs: Set[str] = set()
        for tc in tool_calls:
            iter_sigs.add(_hash_tool_call(tc.name, tc.arguments))

        all_seen_before = False
        if self._recent_iteration_signatures:
            historical_sigs: Set[str] = set()
            for prev_sigs in self._recent_iteration_signatures[-self._config.duplicate_call_window:]:
                historical_sigs.update(prev_sigs)
            all_seen_before = iter_sigs.issubset(historical_sigs)

        self._recent_iteration_signatures.append(iter_sigs)

        if all_seen_before:
            self._consecutive_duplicate_iterations += 1
        else:
            self._consecutive_duplicate_iterations = 0

        # Warning at 2 consecutive duplicates
        if self._consecutive_duplicate_iterations == 2 and not self._duplicate_warning_given:
            self._duplicate_warning_given = True
            return StallVerdict(
                is_stalled=False,
                should_warn=True,
                warning_message=(
                    "[DUPLICATE CALL WARNING] You have been making the "
                    "same tool calls repeatedly for 2 iterations. This "
                    "pattern suggests a loop. Consider: (1) using "
                    "different parameters, (2) trying a different tool, "
                    "or (3) producing your final response if you have "
                    "enough data. The phase will be force-completed if "
                    "this pattern continues."
                ),
            )

        # Force-completion
        if self._consecutive_duplicate_iterations >= self._config.max_duplicate_call_iterations:
            return StallVerdict(is_stalled=True, reason="duplicate_loop")

        return StallVerdict()
