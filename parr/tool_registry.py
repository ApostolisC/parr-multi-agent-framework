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
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

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

    def get_mandatory_for_phase(self, phase: Phase) -> List[ToolDef]:
        """Get tools that MUST be called during a phase."""
        return [
            tool for tool in self._tools.values()
            if tool.mandatory_in_phases and phase in tool.mandatory_in_phases
        ]

    def get_all(self) -> List[ToolDef]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_orchestrator_tools(self) -> List[ToolDef]:
        """Get tools that require orchestrator-level handling (spawn, wait, etc.)."""
        return [
            tool for tool in self._tools.values()
            if tool.is_orchestrator_tool
        ]

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    @property
    def tool_names(self) -> List[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())
