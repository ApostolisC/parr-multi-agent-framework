"""
Tests for the three-tier tool access model (none / visible / callable).

Covers:
- ToolDef.phase_visibility field and to_description_text()
- ToolRegistry.get_visible_for_phase() with auto-inference and explicit config
- Routing prompt tool descriptions
- Context manager visible_tools parameter
- Config loader/validator for phase_visibility
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from parr.core_types import (
    AgentConfig,
    AgentInput,
    Message,
    MessageRole,
    ModelConfig,
    Phase,
    ToolDef,
)
from parr.tool_registry import ToolRegistry
from parr.context_manager import ContextManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _registry(tools: list[ToolDef]) -> ToolRegistry:
    """Create a ToolRegistry with the given tools."""
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg

def _make_tool(
    name: str,
    description: str = "A test tool.",
    phase_availability: list | None = None,
    phase_visibility: list | None = None,
    is_framework_tool: bool = False,
    params: dict | None = None,
) -> ToolDef:
    return ToolDef(
        name=name,
        description=description,
        parameters=params or {"type": "object", "properties": {"q": {"type": "string"}}},
        handler=lambda **kw: "ok",
        phase_availability=phase_availability or [Phase.ACT],
        phase_visibility=phase_visibility or [],
        is_framework_tool=is_framework_tool,
    )


def _make_config() -> AgentConfig:
    return AgentConfig(
        agent_id="a1",
        role="tester",
        model="test-model",
        system_prompt="You are a test agent.",
        model_config=ModelConfig(),
    )


def _make_input(tools: list[ToolDef] | None = None) -> AgentInput:
    return AgentInput(
        task="Test task.",
        tools=tools or [],
    )


# ===========================================================================
# Group 1: ToolDef — phase_visibility + to_description_text
# ===========================================================================


class TestToolDefVisibility:
    def test_phase_visibility_default_empty(self):
        tool = ToolDef(
            name="t", description="d",
            parameters={"type": "object", "properties": {}},
        )
        assert tool.phase_visibility == []

    def test_phase_visibility_explicit(self):
        tool = _make_tool("t", phase_visibility=[Phase.PLAN, Phase.REVIEW])
        assert tool.phase_visibility == [Phase.PLAN, Phase.REVIEW]

    def test_to_description_text_with_params(self):
        tool = _make_tool(
            "search_kb",
            description="Search the knowledge base.",
            params={"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}},
        )
        text = tool.to_description_text()
        assert text.startswith("- search_kb:")
        assert "Search the knowledge base." in text
        assert "query" in text
        assert "top_k" in text

    def test_to_description_text_no_params(self):
        tool = _make_tool(
            "simple",
            description="A simple tool.",
            params={"type": "object", "properties": {}},
        )
        text = tool.to_description_text()
        assert "(params: none)" in text

    def test_to_description_text_format(self):
        tool = _make_tool("my_tool", description="Does things.")
        text = tool.to_description_text()
        assert text.startswith("- my_tool: Does things.")


# ===========================================================================
# Group 2: ToolRegistry.get_visible_for_phase
# ===========================================================================


class TestRegistryVisibility:
    def test_auto_infer_act_visible_in_plan(self):
        """Domain tools callable in ACT are auto-visible in PLAN."""
        tool = _make_tool("rag_search", phase_availability=[Phase.ACT])
        reg = _registry([tool])
        visible = reg.get_visible_for_phase(Phase.PLAN)
        assert len(visible) == 1
        assert visible[0].name == "rag_search"

    def test_auto_infer_does_not_affect_framework_tools(self):
        """Framework tools are NOT auto-visible in PLAN."""
        tool = _make_tool("create_todo", phase_availability=[Phase.ACT], is_framework_tool=True)
        reg = _registry([tool])
        visible = reg.get_visible_for_phase(Phase.PLAN)
        assert len(visible) == 0

    def test_auto_infer_only_for_plan_phase(self):
        """Auto-inference only applies to PLAN, not REVIEW/REPORT."""
        tool = _make_tool("rag_search", phase_availability=[Phase.ACT])
        reg = _registry([tool])
        assert len(reg.get_visible_for_phase(Phase.REVIEW)) == 0
        assert len(reg.get_visible_for_phase(Phase.REPORT)) == 0

    def test_explicit_visibility_overrides_auto(self):
        """When phase_visibility is set, auto-inference is skipped."""
        tool = _make_tool(
            "rag_search",
            phase_availability=[Phase.ACT],
            phase_visibility=[Phase.REVIEW],
        )
        reg = _registry([tool])
        # Explicit: visible in REVIEW only
        assert len(reg.get_visible_for_phase(Phase.REVIEW)) == 1
        # Not auto-visible in PLAN (explicit overrides auto)
        assert len(reg.get_visible_for_phase(Phase.PLAN)) == 0

    def test_explicit_visibility_in_multiple_phases(self):
        """Tool can be visible in multiple explicitly configured phases."""
        tool = _make_tool(
            "rag_search",
            phase_availability=[Phase.ACT],
            phase_visibility=[Phase.PLAN, Phase.REVIEW],
        )
        reg = _registry([tool])
        assert len(reg.get_visible_for_phase(Phase.PLAN)) == 1
        assert len(reg.get_visible_for_phase(Phase.REVIEW)) == 1

    def test_callable_tools_excluded_from_visible(self):
        """Tools already callable in a phase are NOT in visible list."""
        tool = _make_tool("search", phase_availability=[Phase.PLAN, Phase.ACT])
        reg = _registry([tool])
        visible = reg.get_visible_for_phase(Phase.PLAN)
        assert len(visible) == 0  # Already callable in PLAN

    def test_empty_registry_returns_empty(self):
        reg = ToolRegistry()
        assert reg.get_visible_for_phase(Phase.PLAN) == []

    def test_no_domain_tools_returns_empty(self):
        """Only framework tools → no visible tools."""
        fw = _make_tool("todo", phase_availability=[Phase.ACT], is_framework_tool=True)
        reg = _registry([fw])
        assert len(reg.get_visible_for_phase(Phase.PLAN)) == 0

    def test_multiple_domain_tools_all_visible(self):
        """Multiple domain tools auto-visible in PLAN."""
        t1 = _make_tool("search_kb", phase_availability=[Phase.ACT])
        t2 = _make_tool("get_doc", phase_availability=[Phase.ACT])
        reg = _registry([t1, t2])
        visible = reg.get_visible_for_phase(Phase.PLAN)
        names = {t.name for t in visible}
        assert names == {"search_kb", "get_doc"}

    def test_tool_not_callable_in_act_not_auto_visible(self):
        """Tools only in PLAN/REVIEW (not ACT) are NOT auto-visible."""
        tool = _make_tool("plan_tool", phase_availability=[Phase.PLAN])
        reg = _registry([tool])
        # Already callable in PLAN, not auto-visible elsewhere
        assert len(reg.get_visible_for_phase(Phase.REVIEW)) == 0


# ===========================================================================
# Group 3: Routing prompt tool descriptions
# ===========================================================================


class TestRoutingPrompt:
    def _build_routing_messages(self, input: AgentInput):
        """Minimal routing message builder mirroring agent_runtime logic."""
        from parr.agent_runtime import AgentRuntime
        from parr.core_types import SimpleQueryBypassConfig

        # Create a minimal runtime to access _build_routing_messages
        mock_llm = MagicMock()
        runtime = AgentRuntime.__new__(AgentRuntime)
        runtime._simple_query_bypass = SimpleQueryBypassConfig()
        return runtime._build_routing_messages(input)

    def test_routing_includes_tool_descriptions(self):
        tools = [
            _make_tool("search_kb", description="Search the knowledge base."),
            _make_tool("get_doc", description="Retrieve a document by ID."),
        ]
        messages = self._build_routing_messages(_make_input(tools=tools))
        user_msg = messages[1].content
        assert "search_kb" in user_msg
        assert "Search the knowledge base." in user_msg
        assert "get_doc" in user_msg
        assert "Available domain tools (2)" in user_msg

    def test_routing_without_tools_no_section(self):
        messages = self._build_routing_messages(_make_input(tools=[]))
        user_msg = messages[1].content
        assert "Available domain tools" not in user_msg

    def test_routing_excludes_framework_tools(self):
        tools = [
            _make_tool("search_kb", description="Domain tool."),
            _make_tool("create_todo", description="Framework tool.", is_framework_tool=True),
        ]
        messages = self._build_routing_messages(_make_input(tools=tools))
        user_msg = messages[1].content
        assert "search_kb" in user_msg
        assert "create_todo" not in user_msg
        assert "Available domain tools (1)" in user_msg

    def test_routing_system_prompt_mentions_tools(self):
        tools = [_make_tool("t")]
        messages = self._build_routing_messages(_make_input(tools=tools))
        sys_msg = messages[0].content
        assert "domain tools" in sys_msg.lower()


# ===========================================================================
# Group 4: Context manager visible_tools parameter
# ===========================================================================


class TestContextManagerVisibility:
    def _build_messages(self, visible_tools=None):
        cm = ContextManager(max_context_tokens=10000)
        return cm.build_phase_messages(
            phase=Phase.PLAN,
            config=_make_config(),
            input=_make_input(),
            visible_tools=visible_tools,
        )

    def test_plan_prompt_includes_visible_tool_descriptions(self):
        tools = [_make_tool("search_kb", description="Search KB.")]
        messages = self._build_messages(visible_tools=tools)
        system_msg = messages[0].content
        assert "search_kb" in system_msg
        assert "Search KB." in system_msg
        assert "callable in the execution phase" in system_msg

    def test_act_phase_no_visible_tools_section(self):
        """ACT phase with no visible tools has no visibility section."""
        cm = ContextManager(max_context_tokens=10000)
        messages = cm.build_phase_messages(
            phase=Phase.ACT,
            config=_make_config(),
            input=_make_input(),
            visible_tools=[],
        )
        system_msg = messages[0].content
        assert "callable in the execution phase, not in this phase" not in system_msg

    def test_visible_tools_none_no_crash(self):
        """visible_tools=None is safe."""
        messages = self._build_messages(visible_tools=None)
        system_msg = messages[0].content
        assert "callable in the execution phase, not in this phase" not in system_msg

    def test_visible_tools_empty_no_section(self):
        """Empty list produces no section."""
        messages = self._build_messages(visible_tools=[])
        system_msg = messages[0].content
        assert "callable in the execution phase, not in this phase" not in system_msg

    def test_backward_compat_no_visible_param(self):
        """Old call without visible_tools still works."""
        cm = ContextManager(max_context_tokens=10000)
        messages = cm.build_phase_messages(
            phase=Phase.PLAN,
            config=_make_config(),
            input=_make_input(),
        )
        assert len(messages) == 2  # system + user

    def test_plan_prompt_references_tools(self):
        """PLAN prompt instructs agent to consider available tools."""
        messages = self._build_messages(visible_tools=None)
        system_msg = messages[0].content
        assert "available tools" in system_msg.lower()


# ===========================================================================
# Group 5: Config validation
# ===========================================================================


class TestConfigValidation:
    def test_yaml_phase_visibility_invalid_phase(self):
        from parr.config.config_validator import validate_tools_config

        tools = {
            "my_tool": {
                "description": "A tool.",
                "parameters": {"type": "object", "properties": {}},
                "phase_visibility": ["plan", "invalid_phase"],
            }
        }
        errors = validate_tools_config(tools, handler_names=["my_tool"])
        assert any("invalid_phase" in e and "phase_visibility" in e for e in errors)

    def test_yaml_phase_visibility_not_list(self):
        from parr.config.config_validator import validate_tools_config

        tools = {
            "my_tool": {
                "description": "A tool.",
                "parameters": {"type": "object", "properties": {}},
                "phase_visibility": "plan",
            }
        }
        errors = validate_tools_config(tools, handler_names=["my_tool"])
        assert any("phase_visibility" in e and "must be a list" in e for e in errors)

    def test_yaml_phase_visibility_valid(self):
        from parr.config.config_validator import validate_tools_config

        tools = {
            "my_tool": {
                "description": "A tool.",
                "parameters": {"type": "object", "properties": {}},
                "phase_visibility": ["plan", "review"],
            }
        }
        errors = validate_tools_config(tools, handler_names=["my_tool"])
        assert not any("phase_visibility" in e for e in errors)

    def test_yaml_no_phase_visibility_backward_compat(self):
        from parr.config.config_validator import validate_tools_config

        tools = {
            "my_tool": {
                "description": "A tool.",
                "parameters": {"type": "object", "properties": {}},
            }
        }
        errors = validate_tools_config(tools, handler_names=["my_tool"])
        assert not any("phase_visibility" in e for e in errors)

    def test_config_loader_phase_visibility(self):
        """Config loader parses phase_visibility from YAML."""
        from parr.config.config_loader import _build_tools_from_yaml, _PHASE_MAP

        async def _handler(**kw):
            return "ok"

        raw_tools = {
            "my_tool": {
                "description": "A tool.",
                "parameters": {"type": "object", "properties": {}},
                "phase_visibility": ["plan", "review"],
            }
        }
        result = _build_tools_from_yaml(raw_tools, {"my_tool": _handler})
        tool = result["my_tool"]
        assert Phase.PLAN in tool.phase_visibility
        assert Phase.REVIEW in tool.phase_visibility
        assert Phase.ACT not in tool.phase_visibility

    def test_config_loader_no_phase_visibility_defaults_empty(self):
        """Missing phase_visibility defaults to empty list."""
        from parr.config.config_loader import _build_tools_from_yaml

        async def _handler(**kw):
            return "ok"

        raw_tools = {
            "my_tool": {
                "description": "A tool.",
                "parameters": {"type": "object", "properties": {}},
            }
        }
        result = _build_tools_from_yaml(raw_tools, {"my_tool": _handler})
        assert result["my_tool"].phase_visibility == []
