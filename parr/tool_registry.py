"""
Tool Registry for the Agentic Framework.

Manages tool registration and provides phase-based filtering.
Phase-specific tool filtering is enforced by the orchestrator through this
registry, not by the prompt — even if the LLM requests a tool it shouldn't
have in the current phase, the registry won't return it.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .core_types import Phase, ToolDef

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Central registry for all tools available in a workflow.

    Tools are registered once when the workflow starts. The phase runner
    queries the registry to get the tools available in each phase.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        """Register a tool. Raises ValueError if name already registered."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._ensure_strip_param(tool)
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def override(self, tool: ToolDef) -> Optional[ToolDef]:
        """
        Replace an existing tool registration.

        This is the mechanism for overriding framework tools with custom
        implementations. If the tool doesn't exist yet, it is registered
        as new.

        Args:
            tool: The new ToolDef to register (replaces any existing tool
                  with the same name).

        Returns:
            The previous ToolDef if one was replaced, or None if this was
            a new registration.
        """
        previous = self._tools.get(tool.name)
        self._ensure_strip_param(tool)
        self._tools[tool.name] = tool
        if previous is not None:
            logger.info(f"Overrode tool: {tool.name}")
        else:
            logger.debug(f"Registered tool (via override): {tool.name}")
        return previous

    def register_many(self, tools: List[ToolDef]) -> None:
        """Register multiple tools at once."""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional[ToolDef]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_for_phase(self, phase: Phase, role: Optional[str] = None) -> List[ToolDef]:
        """
        Get tools available in a specific phase.

        Args:
            phase: The current execution phase.
            role: Optional role filter (for future per-role tool restrictions).

        Returns:
            List of ToolDef available in the given phase.
        """
        return [
            tool for tool in self._tools.values()
            if phase in tool.phase_availability
        ]

    def get_visible_for_phase(self, phase: Phase) -> List[ToolDef]:
        """
        Get tools whose descriptions should be shown (but not callable) in a phase.

        A tool is visible when:
        1. It has the phase in its explicit ``phase_visibility`` list, OR
        2. Auto-inference: ``phase_visibility`` is empty AND the tool is a
           domain tool callable in ACT AND the requested phase is PLAN.

        Tools already callable in the phase are excluded.
        """
        callable_names = {
            t.name for t in self._tools.values()
            if phase in t.phase_availability
        }
        visible: List[ToolDef] = []
        for tool in self._tools.values():
            if tool.name in callable_names:
                continue
            if tool.phase_visibility:
                if phase in tool.phase_visibility:
                    visible.append(tool)
            else:
                # Auto-inference: domain tools callable in ACT → visible in PLAN
                if (
                    not tool.is_framework_tool
                    and Phase.ACT in tool.phase_availability
                    and phase == Phase.PLAN
                ):
                    visible.append(tool)
        return visible

    def get_mandatory_for_phase(self, phase: Phase) -> List[ToolDef]:
        """Get tools that MUST be called during a phase."""
        return [
            tool for tool in self._tools.values()
            if tool.mandatory_in_phases and phase in tool.mandatory_in_phases
        ]

    def get_all(self) -> List[ToolDef]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_for_entry(self) -> List[ToolDef]:
        """Get tools available in the adaptive-flow entry call.

        Returns all tools except phase-specific ones that only make sense
        during REVIEW or REPORT (review_checklist, get_report_template,
        submit_report).
        """
        _ENTRY_EXCLUDED = {"review_checklist", "get_report_template", "submit_report"}
        return [
            tool for tool in self._tools.values()
            if tool.name not in _ENTRY_EXCLUDED
        ]

    def get_orchestrator_tools(self) -> List[ToolDef]:
        """Get tools that require orchestrator-level handling (spawn, wait, etc.)."""
        return [
            tool for tool in self._tools.values()
            if tool.is_orchestrator_tool
        ]

    @staticmethod
    def _ensure_strip_param(tool: ToolDef) -> None:
        """Auto-inject ``strip_input_after_dispatch`` into the tool schema.

        Every tool gets this optional boolean so the LLM can override
        per-call whether heavy input fields are stripped from its
        conversation history after dispatch.  Idempotent — skips tools
        that already declare the parameter explicitly.
        """
        props = tool.parameters.get("properties")
        if not isinstance(props, dict):
            return
        if "strip_input_after_dispatch" in props:
            return  # already explicitly defined
        props["strip_input_after_dispatch"] = {
            "type": "boolean",
            "description": (
                "Override whether this tool's heavy input fields are "
                "stripped from your conversation history after dispatch. "
                "Defaults to the tool's configured behavior. Set to false "
                "on a spawn_agent call if you genuinely need to refer "
                "back to the spawn args later."
            ),
        }

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    @property
    def tool_names(self) -> List[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())
