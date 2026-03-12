"""Comprehensive tests for parr.tool_registry.ToolRegistry."""

import pytest

from parr.core_types import Phase, ToolDef
from parr.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(
    name: str = "tool_a",
    phase_availability=None,
    mandatory_in_phases=None,
    is_orchestrator_tool: bool = False,
    **overrides,
) -> ToolDef:
    """Build a minimal ToolDef for registry tests (no handler needed)."""
    return ToolDef(
        name=name,
        description=f"Description for {name}",
        parameters={"type": "object", "properties": {}},
        phase_availability=phase_availability if phase_availability is not None else list(Phase),
        mandatory_in_phases=mandatory_in_phases,
        is_orchestrator_tool=is_orchestrator_tool,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def plan_only_tool():
    return _make_tool("plan_only", phase_availability=[Phase.PLAN])


@pytest.fixture
def act_review_tool():
    return _make_tool("act_review", phase_availability=[Phase.ACT, Phase.REVIEW])


@pytest.fixture
def orchestrator_tool():
    return _make_tool("spawn_agent", is_orchestrator_tool=True)


@pytest.fixture
def mandatory_tool():
    return _make_tool(
        "submit_report",
        mandatory_in_phases=[Phase.REPORT],
        phase_availability=[Phase.REPORT],
    )


# ---------------------------------------------------------------------------
# 1. register()
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_single_tool(self, registry):
        tool = _make_tool("alpha")
        registry.register(tool)
        assert registry.has_tool("alpha")

    def test_register_returns_none(self, registry):
        result = registry.register(_make_tool("alpha"))
        assert result is None

    def test_register_duplicate_raises(self, registry):
        registry.register(_make_tool("dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_make_tool("dup"))

    def test_register_different_names_ok(self, registry):
        registry.register(_make_tool("a"))
        registry.register(_make_tool("b"))
        assert registry.has_tool("a")
        assert registry.has_tool("b")

    def test_register_preserves_tool_object(self, registry):
        tool = _make_tool("exact")
        registry.register(tool)
        assert registry.get("exact") is tool


# ---------------------------------------------------------------------------
# 2. register_many()
# ---------------------------------------------------------------------------

class TestRegisterMany:
    def test_register_many_adds_all(self, registry):
        tools = [_make_tool("x"), _make_tool("y"), _make_tool("z")]
        registry.register_many(tools)
        assert set(registry.tool_names) == {"x", "y", "z"}

    def test_register_many_empty_list(self, registry):
        registry.register_many([])
        assert registry.get_all() == []

    def test_register_many_duplicate_in_batch_raises(self, registry):
        tools = [_make_tool("same"), _make_tool("same")]
        with pytest.raises(ValueError, match="already registered"):
            registry.register_many(tools)

    def test_register_many_duplicate_with_existing_raises(self, registry):
        registry.register(_make_tool("existing"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register_many([_make_tool("new"), _make_tool("existing")])

    def test_register_many_partial_failure_leaves_earlier_tools(self, registry):
        """First tool in the batch is registered before the duplicate raises."""
        tools = [_make_tool("first"), _make_tool("first")]
        with pytest.raises(ValueError):
            registry.register_many(tools)
        assert registry.has_tool("first")


# ---------------------------------------------------------------------------
# 3. get()
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_existing(self, registry):
        tool = _make_tool("found")
        registry.register(tool)
        assert registry.get("found") is tool

    def test_get_missing_returns_none(self, registry):
        assert registry.get("nonexistent") is None

    def test_get_after_multiple_registers(self, registry):
        t1 = _make_tool("t1")
        t2 = _make_tool("t2")
        registry.register_many([t1, t2])
        assert registry.get("t1") is t1
        assert registry.get("t2") is t2


# ---------------------------------------------------------------------------
# 4. get_for_phase()
# ---------------------------------------------------------------------------

class TestGetForPhase:
    def test_returns_tools_available_in_phase(self, registry, plan_only_tool, act_review_tool):
        registry.register_many([plan_only_tool, act_review_tool])
        result = registry.get_for_phase(Phase.PLAN)
        assert plan_only_tool in result
        assert act_review_tool not in result

    def test_tool_available_in_multiple_phases(self, registry, act_review_tool):
        registry.register(act_review_tool)
        assert act_review_tool in registry.get_for_phase(Phase.ACT)
        assert act_review_tool in registry.get_for_phase(Phase.REVIEW)
        assert act_review_tool not in registry.get_for_phase(Phase.PLAN)

    def test_default_phase_availability_all_phases(self, registry):
        tool = _make_tool("all_phases")  # default = all phases
        registry.register(tool)
        for phase in Phase:
            assert tool in registry.get_for_phase(phase)

    def test_no_tools_in_phase(self, registry, plan_only_tool):
        registry.register(plan_only_tool)
        assert registry.get_for_phase(Phase.ACT) == []

    def test_role_parameter_accepted(self, registry, plan_only_tool):
        """role kwarg is accepted even though filtering isn't implemented yet."""
        registry.register(plan_only_tool)
        result = registry.get_for_phase(Phase.PLAN, role="researcher")
        assert plan_only_tool in result

    def test_empty_phase_availability(self, registry):
        tool = _make_tool("no_phase", phase_availability=[])
        registry.register(tool)
        for phase in Phase:
            assert tool not in registry.get_for_phase(phase)


# ---------------------------------------------------------------------------
# 5. get_mandatory_for_phase()
# ---------------------------------------------------------------------------

class TestGetMandatoryForPhase:
    def test_returns_mandatory_tools(self, registry, mandatory_tool):
        registry.register(mandatory_tool)
        result = registry.get_mandatory_for_phase(Phase.REPORT)
        assert mandatory_tool in result

    def test_excludes_non_mandatory(self, registry, plan_only_tool, mandatory_tool):
        registry.register_many([plan_only_tool, mandatory_tool])
        result = registry.get_mandatory_for_phase(Phase.REPORT)
        assert mandatory_tool in result
        assert plan_only_tool not in result

    def test_mandatory_in_different_phase(self, registry, mandatory_tool):
        registry.register(mandatory_tool)
        assert registry.get_mandatory_for_phase(Phase.PLAN) == []

    def test_no_mandatory_tools(self, registry, plan_only_tool):
        registry.register(plan_only_tool)
        assert registry.get_mandatory_for_phase(Phase.PLAN) == []

    def test_multiple_mandatory_phases(self, registry):
        tool = _make_tool(
            "multi_mandatory",
            mandatory_in_phases=[Phase.PLAN, Phase.REVIEW],
        )
        registry.register(tool)
        assert tool in registry.get_mandatory_for_phase(Phase.PLAN)
        assert tool in registry.get_mandatory_for_phase(Phase.REVIEW)
        assert tool not in registry.get_mandatory_for_phase(Phase.ACT)


# ---------------------------------------------------------------------------
# 6. get_all()
# ---------------------------------------------------------------------------

class TestGetAll:
    def test_returns_all_registered(self, registry):
        tools = [_make_tool("a"), _make_tool("b"), _make_tool("c")]
        registry.register_many(tools)
        result = registry.get_all()
        assert len(result) == 3
        for t in tools:
            assert t in result

    def test_returns_new_list(self, registry):
        registry.register(_make_tool("a"))
        list1 = registry.get_all()
        list2 = registry.get_all()
        assert list1 is not list2
        assert list1 == list2


# ---------------------------------------------------------------------------
# 7. get_orchestrator_tools()
# ---------------------------------------------------------------------------

class TestGetOrchestratorTools:
    def test_returns_orchestrator_tools(self, registry, orchestrator_tool):
        registry.register(orchestrator_tool)
        result = registry.get_orchestrator_tools()
        assert orchestrator_tool in result

    def test_excludes_non_orchestrator(self, registry, plan_only_tool, orchestrator_tool):
        registry.register_many([plan_only_tool, orchestrator_tool])
        result = registry.get_orchestrator_tools()
        assert orchestrator_tool in result
        assert plan_only_tool not in result

    def test_no_orchestrator_tools(self, registry, plan_only_tool):
        registry.register(plan_only_tool)
        assert registry.get_orchestrator_tools() == []

    def test_multiple_orchestrator_tools(self, registry):
        t1 = _make_tool("spawn", is_orchestrator_tool=True)
        t2 = _make_tool("wait", is_orchestrator_tool=True)
        registry.register_many([t1, t2])
        result = registry.get_orchestrator_tools()
        assert len(result) == 2
        assert t1 in result
        assert t2 in result


# ---------------------------------------------------------------------------
# 8. has_tool()
# ---------------------------------------------------------------------------

class TestHasTool:
    def test_returns_true_for_registered(self, registry):
        registry.register(_make_tool("present"))
        assert registry.has_tool("present") is True

    def test_returns_false_for_missing(self, registry):
        assert registry.has_tool("absent") is False

    def test_case_sensitive(self, registry):
        registry.register(_make_tool("CaseSensitive"))
        assert registry.has_tool("CaseSensitive") is True
        assert registry.has_tool("casesensitive") is False


# ---------------------------------------------------------------------------
# 9. tool_names property
# ---------------------------------------------------------------------------

class TestToolNames:
    def test_returns_all_names(self, registry):
        registry.register_many([_make_tool("a"), _make_tool("b")])
        names = registry.tool_names
        assert set(names) == {"a", "b"}

    def test_returns_list(self, registry):
        assert isinstance(registry.tool_names, list)

    def test_order_matches_insertion(self, registry):
        for name in ["x", "y", "z"]:
            registry.register(_make_tool(name))
        assert registry.tool_names == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# 10. Empty registry
# ---------------------------------------------------------------------------

class TestEmptyRegistry:
    def test_get_returns_none(self, registry):
        assert registry.get("anything") is None

    def test_get_all_returns_empty(self, registry):
        assert registry.get_all() == []

    def test_get_for_phase_returns_empty(self, registry):
        for phase in Phase:
            assert registry.get_for_phase(phase) == []

    def test_get_mandatory_for_phase_returns_empty(self, registry):
        for phase in Phase:
            assert registry.get_mandatory_for_phase(phase) == []

    def test_get_orchestrator_tools_returns_empty(self, registry):
        assert registry.get_orchestrator_tools() == []

    def test_has_tool_returns_false(self, registry):
        assert registry.has_tool("anything") is False

    def test_tool_names_returns_empty(self, registry):
        assert registry.tool_names == []
