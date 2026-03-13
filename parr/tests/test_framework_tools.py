"""Tests for framework tool helper behavior."""

from parr.core_types import Phase
from parr.framework_tools import (
    AgentWorkingMemory,
    TodoItem,
    build_plan_tools,
    build_report_tools,
)


def _tool_by_name(tools, name):
    for tool in tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"Tool {name!r} not found")


def test_get_todo_list_uses_ascii_separator():
    memory = AgentWorkingMemory()
    memory.todo_list = [
        TodoItem(
            index=0,
            description="Collect DPIA phases",
            priority="high",
            completed=True,
            completion_summary="Found 8 phases",
        ),
    ]

    text = memory.get_todo_list()

    assert " - Found 8 phases" in text
    assert "—" not in text


def test_get_report_template_falls_back_to_default_role():
    memory = AgentWorkingMemory()

    def handler(role, sub_role=None):
        if role == "researcher" and not sub_role:
            return "Researcher template"
        return None

    tools = build_report_tools(
        memory=memory,
        report_template_handler=handler,
        default_role="researcher",
        default_sub_role=None,
    )
    get_template = _tool_by_name(tools, "get_report_template")

    # LLM may pass a human-readable label instead of role ID.
    result = get_template.handler(role="research_assistant", sub_role=None)

    assert result == "Researcher template"


def test_create_todo_list_is_available_but_not_mandatory():
    memory = AgentWorkingMemory()
    tools = build_plan_tools(memory)
    create_tool = _tool_by_name(tools, "create_todo_list")

    assert Phase.PLAN in create_tool.phase_availability
    assert not create_tool.mandatory_in_phases
