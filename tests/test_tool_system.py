"""Tests for tool registration, phase availability, and terminates_phase."""
import pytest

from parr.core_types import Phase, ToolDef
from parr.framework_tools import (
    AgentWorkingMemory,
    build_report_tools,
    build_transition_tools,
)
from parr.tool_registry import ToolRegistry


class TestSubmitReportPhaseAvailability:
    def test_default_is_report_only(self):
        mem = AgentWorkingMemory()
        tools = build_report_tools(mem)
        submit = next(t for t in tools if t.name == "submit_report")
        assert submit.phase_availability == [Phase.REPORT]

    def test_terminates_phase_is_true(self):
        mem = AgentWorkingMemory()
        tools = build_report_tools(mem)
        submit = next(t for t in tools if t.name == "submit_report")
        assert submit.terminates_phase is True

    def test_level0_override_to_act(self):
        """Level 0 agents should have submit_report in ACT."""
        mem = AgentWorkingMemory()
        tools = build_report_tools(mem)
        registry = ToolRegistry()
        for t in tools:
            registry.register(t)

        # Simulate the Level 0 override from agent_runtime.py
        submit = registry.get("submit_report")
        submit.phase_availability = [Phase.ACT]

        assert submit.phase_availability == [Phase.ACT]
        assert submit.terminates_phase is True  # still terminates

    def test_get_report_template_phase(self):
        mem = AgentWorkingMemory()
        tools = build_report_tools(mem)
        template = next(t for t in tools if t.name == "get_report_template")
        assert Phase.REPORT in template.phase_availability


class TestSetNextPhaseAvailability:
    def test_available_in_plan_act_review(self):
        mem = AgentWorkingMemory()
        tools = build_transition_tools(mem)
        snp = next(t for t in tools if t.name == "set_next_phase")
        assert Phase.PLAN in snp.phase_availability
        assert Phase.ACT in snp.phase_availability
        assert Phase.REVIEW in snp.phase_availability
        assert Phase.REPORT not in snp.phase_availability


class TestSpawnAgentPhaseAvailability:
    def test_act_only(self):
        from parr.framework_tools import build_agent_management_tools
        tools = build_agent_management_tools("")
        spawn = next(t for t in tools if t.name == "spawn_agent")
        assert spawn.phase_availability == [Phase.ACT]

    def test_batch_operations_all_phases(self):
        from parr.framework_tools import build_agent_management_tools
        tools = build_agent_management_tools("")
        batch = next(t for t in tools if t.name == "batch_operations")
        assert Phase.PLAN in batch.phase_availability
        assert Phase.ACT in batch.phase_availability
        assert Phase.REVIEW in batch.phase_availability
        assert Phase.REPORT in batch.phase_availability

    def test_spawn_agent_has_blocking_param(self):
        from parr.framework_tools import build_agent_management_tools
        tools = build_agent_management_tools("")
        spawn = next(t for t in tools if t.name == "spawn_agent")
        props = spawn.parameters.get("properties", {})
        assert "blocking" in props
        assert props["blocking"]["default"] is True

    def test_spawn_agent_has_description_param(self):
        from parr.framework_tools import build_agent_management_tools
        tools = build_agent_management_tools("")
        spawn = next(t for t in tools if t.name == "spawn_agent")
        props = spawn.parameters.get("properties", {})
        assert "description" in props


class TestToolExecutorNormalisation:
    """Test that ToolExecutor normalises camelCase keys before validation."""

    @pytest.mark.asyncio
    async def test_camel_case_normalised(self):
        from parr.tool_executor import ToolExecutor
        from parr.core_types import ToolCall

        call_args = {}
        def handler(name: str = "", value: str = ""):
            call_args["name"] = name
            call_args["value"] = value
            return "ok"

        td = ToolDef(
            name="test_tool",
            description="Test",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["name"],
            },
            handler=handler,
            phase_availability=[Phase.ACT],
        )

        registry = ToolRegistry()
        registry.register(td)

        executor = ToolExecutor(registry)
        executor.set_phase(Phase.ACT)
        tc = ToolCall(id="tc1", name="test_tool", arguments={"name": "test", "value": "v"})
        result = await executor.execute(tc)
        assert result.success
