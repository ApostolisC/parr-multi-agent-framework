"""Tests for parr.core_types — enums, dataclasses, helpers."""

import time
from datetime import datetime, timezone

import pytest

from parr.core_types import (
    AgentNode,
    AgentOutput,
    AgentStatus,
    BudgetConfig,
    BudgetUsage,
    Confidence,
    CostConfig,
    ErrorEntry,
    ErrorSource,
    ExecutionMetadata,
    LLMResponse,
    Message,
    MessageRole,
    ModelPricing,
    Phase,
    PlanStepStatus,
    TokenUsage,
    ToolCall,
    ToolDef,
    ToolResult,
    TraceEntry,
    WorkflowExecution,
    WorkflowStatus,
    generate_id,
    utc_now,
)


# -- Helpers -----------------------------------------------------------------


class TestGenerateId:
    def test_returns_string(self):
        assert isinstance(generate_id(), str)

    def test_uniqueness(self):
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100


class TestUtcNow:
    def test_returns_utc_datetime(self):
        now = utc_now()
        assert isinstance(now, datetime)
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc

    def test_is_recent(self):
        before = datetime.now(timezone.utc)
        now = utc_now()
        after = datetime.now(timezone.utc)
        assert before <= now <= after


# -- Enums -------------------------------------------------------------------


class TestPhase:
    def test_values(self):
        assert Phase.PLAN.value == "plan"
        assert Phase.ACT.value == "act"
        assert Phase.REVIEW.value == "review"
        assert Phase.REPORT.value == "report"

    def test_is_str(self):
        assert isinstance(Phase.PLAN, str)
        assert Phase.PLAN == "plan"

    def test_member_count(self):
        assert len(Phase) == 4


class TestAgentStatus:
    def test_values(self):
        expected = {
            "RUNNING": "running",
            "SUSPENDED": "suspended",
            "COMPLETED": "completed",
            "FAILED": "failed",
            "CANCELLED": "cancelled",
        }
        for name, value in expected.items():
            assert AgentStatus[name].value == value

    def test_is_str(self):
        assert AgentStatus.RUNNING == "running"


class TestWorkflowStatus:
    def test_values(self):
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"


class TestMessageRole:
    def test_values(self):
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"


class TestErrorSource:
    def test_values(self):
        assert ErrorSource.TOOL.value == "tool"
        assert ErrorSource.AGENT.value == "agent"
        assert ErrorSource.SYSTEM.value == "system"


class TestConfidence:
    def test_values(self):
        assert Confidence.HIGH.value == "high"
        assert Confidence.MEDIUM.value == "medium"
        assert Confidence.LOW.value == "low"


class TestPlanStepStatus:
    def test_values(self):
        assert PlanStepStatus.PENDING.value == "pending"
        assert PlanStepStatus.IN_PROGRESS.value == "in_progress"
        assert PlanStepStatus.COMPLETED.value == "completed"
        assert PlanStepStatus.FAILED.value == "failed"
        assert PlanStepStatus.SKIPPED.value == "skipped"


# -- ToolCall / ToolResult ---------------------------------------------------


class TestToolCall:
    def test_construction(self):
        tc = ToolCall(id="tc-1", name="search", arguments={"q": "test"})
        assert tc.id == "tc-1"
        assert tc.name == "search"
        assert tc.arguments == {"q": "test"}


class TestToolResult:
    def test_success(self):
        tr = ToolResult(tool_call_id="tc-1", success=True, content="ok")
        assert tr.success is True
        assert tr.error is None

    def test_failure(self):
        tr = ToolResult(tool_call_id="tc-1", success=False, content="", error="boom")
        assert tr.success is False
        assert tr.error == "boom"

    def test_repr_success(self):
        tr = ToolResult(tool_call_id="tc-1", success=True, content="ok")
        r = repr(tr)
        assert "ok" in r
        assert "tc-1" in r

    def test_repr_failure(self):
        tr = ToolResult(tool_call_id="tc-1", success=False, content="", error="boom")
        r = repr(tr)
        assert "err=boom" in r


# -- Message -----------------------------------------------------------------


class TestMessage:
    def test_minimal(self):
        m = Message(role=MessageRole.USER, content="hello")
        assert m.role == MessageRole.USER
        assert m.content == "hello"
        assert m.tool_calls is None
        assert m.tool_call_id is None

    def test_tool_message(self):
        m = Message(role=MessageRole.TOOL, content="result", tool_call_id="tc-1")
        assert m.tool_call_id == "tc-1"

    def test_assistant_with_tool_calls(self):
        tc = ToolCall(id="tc-1", name="fn", arguments={})
        m = Message(role=MessageRole.ASSISTANT, tool_calls=[tc])
        assert m.tool_calls == [tc]


# -- LLMResponse -------------------------------------------------------------


class TestLLMResponse:
    def test_defaults(self):
        r = LLMResponse()
        assert r.content is None
        assert r.tool_calls is None
        assert r.usage is None
        assert r.raw_message is None

    def test_has_tool_calls_false_when_none(self):
        assert LLMResponse().has_tool_calls() is False

    def test_has_tool_calls_false_when_empty(self):
        assert LLMResponse(tool_calls=[]).has_tool_calls() is False

    def test_has_tool_calls_true(self):
        tc = ToolCall(id="1", name="fn", arguments={})
        assert LLMResponse(tool_calls=[tc]).has_tool_calls() is True

    def test_repr_no_tools(self):
        r = repr(LLMResponse(content="hello world"))
        assert "tools=0" in r
        assert "hello world" in r

    def test_repr_with_tools(self):
        tc = ToolCall(id="1", name="fn", arguments={})
        r = repr(LLMResponse(tool_calls=[tc]))
        assert "tools=1" in r

    def test_repr_truncates_long_content(self):
        long_content = "x" * 100
        r = repr(LLMResponse(content=long_content))
        # content_preview is capped at 40 chars
        assert len(r) < 100


# -- TokenUsage --------------------------------------------------------------


class TestTokenUsage:
    def test_defaults(self):
        tu = TokenUsage()
        assert tu.input_tokens == 0
        assert tu.output_tokens == 0
        assert tu.total_cost == 0.0

    def test_total_tokens_property(self):
        tu = TokenUsage(input_tokens=100, output_tokens=50)
        assert tu.total_tokens == 150

    def test_repr(self):
        tu = TokenUsage(input_tokens=10, output_tokens=20, total_cost=0.005)
        r = repr(tu)
        assert "in=10" in r
        assert "out=20" in r
        assert "$0.0050" in r


# -- ToolDef -----------------------------------------------------------------


class TestToolDef:
    def test_to_llm_schema(self):
        td = ToolDef(
            name="my_tool",
            description="does stuff",
            parameters={"type": "object", "properties": {}},
        )
        schema = td.to_llm_schema()
        assert schema == {
            "name": "my_tool",
            "description": "does stuff",
            "parameters": {"type": "object", "properties": {}},
        }

    def test_defaults(self):
        td = ToolDef(name="t", description="d", parameters={})
        assert td.handler is None
        assert td.phase_availability == list(Phase)
        assert td.is_framework_tool is False
        assert td.is_orchestrator_tool is False
        assert td.timeout_ms == 30000
        assert td.max_calls_per_phase is None
        assert td.retry_on_failure is False
        assert td.max_retries == 0

    def test_repr(self):
        td = ToolDef(
            name="t",
            description="d",
            parameters={},
            phase_availability=[Phase.PLAN, Phase.ACT],
        )
        r = repr(td)
        assert "t" in r
        assert "plan" in r
        assert "act" in r

    def test_to_llm_schema_excludes_internal_fields(self):
        td = ToolDef(
            name="t",
            description="d",
            parameters={"type": "object"},
            handler=lambda: None,
            is_framework_tool=True,
            timeout_ms=5000,
        )
        schema = td.to_llm_schema()
        assert "handler" not in schema
        assert "is_framework_tool" not in schema
        assert "timeout_ms" not in schema
        assert set(schema.keys()) == {"name", "description", "parameters"}


# -- BudgetConfig ------------------------------------------------------------


class TestBudgetConfig:
    def test_defaults(self):
        bc = BudgetConfig()
        assert bc.max_tokens is None
        assert bc.max_cost is None
        assert bc.max_duration_ms is None
        assert bc.max_agent_depth == 3
        assert bc.max_parallel_agents == 3
        assert bc.max_sub_agents_total == 3
        assert bc.inherit_remaining is True
        assert bc.child_budget_fraction == 0.5

    def test_frozen(self):
        bc = BudgetConfig()
        with pytest.raises(AttributeError):
            bc.max_tokens = 999

    def test_custom_values(self):
        bc = BudgetConfig(max_tokens=1000, max_cost=1.5, max_duration_ms=5000)
        assert bc.max_tokens == 1000
        assert bc.max_cost == 1.5
        assert bc.max_duration_ms == 5000


# -- BudgetUsage -------------------------------------------------------------


class TestBudgetUsage:
    def test_defaults(self):
        bu = BudgetUsage()
        assert bu.tokens == 0
        assert bu.cost == 0.0
        assert bu.started_at.tzinfo == timezone.utc

    def test_elapsed_ms_positive(self):
        bu = BudgetUsage()
        time.sleep(0.05)  # 50 ms
        elapsed = bu.elapsed_ms
        assert elapsed >= 40  # allow some scheduling slack

    def test_elapsed_ms_type(self):
        bu = BudgetUsage()
        assert isinstance(bu.elapsed_ms, float)


# -- AgentOutput -------------------------------------------------------------


class TestAgentOutput:
    def _make_output(self, **overrides):
        defaults = dict(
            task_id="task-1",
            agent_id="agent-1",
            role="analyst",
            summary="all good",
            findings={"key": "value"},
            artifacts=["report.pdf"],
            token_usage=TokenUsage(input_tokens=100, output_tokens=50, total_cost=0.01),
        )
        defaults.update(overrides)
        return AgentOutput(**defaults)

    def test_to_dict_keys(self):
        out = self._make_output()
        d = out.to_dict()
        expected_keys = {
            "task_id", "agent_id", "role", "sub_role", "status",
            "summary", "findings", "artifacts", "errors",
            "recommendations", "token_usage", "execution_metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        out = self._make_output()
        d = out.to_dict()
        assert d["task_id"] == "task-1"
        assert d["agent_id"] == "agent-1"
        assert d["role"] == "analyst"
        assert d["status"] == "completed"
        assert d["summary"] == "all good"
        assert d["findings"] == {"key": "value"}
        assert d["artifacts"] == ["report.pdf"]
        assert d["sub_role"] is None
        assert d["recommendations"] is None

    def test_to_dict_token_usage(self):
        out = self._make_output()
        tu = out.to_dict()["token_usage"]
        assert tu["input_tokens"] == 100
        assert tu["output_tokens"] == 50
        assert tu["total_tokens"] == 150
        assert tu["total_cost"] == 0.01

    def test_to_dict_errors_serialized(self):
        err = ErrorEntry(
            source=ErrorSource.TOOL,
            name="bad_tool",
            error_type="RuntimeError",
            message="it broke",
        )
        out = self._make_output(errors=[err])
        errors = out.to_dict()["errors"]
        assert len(errors) == 1
        assert errors[0] == {
            "source": "tool",
            "name": "bad_tool",
            "error_type": "RuntimeError",
            "message": "it broke",
        }

    def test_to_dict_execution_metadata(self):
        meta = ExecutionMetadata(
            phases_completed=["plan", "act"],
            total_duration_ms=1234.5,
            execution_path="direct_answer",
            routing_decision={"selected_path": "direct_answer"},
        )
        out = self._make_output(execution_metadata=meta)
        em = out.to_dict()["execution_metadata"]
        assert em["phases_completed"] == ["plan", "act"]
        assert em["total_duration_ms"] == 1234.5
        assert em["execution_path"] == "direct_answer"
        assert em["routing_decision"] == {"selected_path": "direct_answer"}

    def test_defaults(self):
        out = AgentOutput(task_id="t", agent_id="a", role="r")
        assert out.status == "completed"
        assert out.summary == ""
        assert out.findings == {}
        assert out.artifacts == []
        assert out.errors == []
        assert out.recommendations is None

    def test_repr(self):
        out = self._make_output()
        r = repr(out)
        assert "task-1" in r
        assert "analyst" in r
        assert "completed" in r


# -- TraceEntry --------------------------------------------------------------


class TestTraceEntry:
    def test_defaults(self):
        te = TraceEntry()
        assert te.agent_id == ""
        assert te.role == ""
        assert te.sub_role is None
        assert te.parent_task_id is None
        assert te.task_description == ""
        assert te.status == AgentStatus.RUNNING
        assert te.output_summary is None
        assert te.completed_at is None
        assert te.children == []
        # task_id and started_at are auto-generated
        assert isinstance(te.task_id, str)
        assert te.started_at.tzinfo == timezone.utc

    def test_custom_values(self):
        te = TraceEntry(
            task_id="t-1",
            agent_id="a-1",
            role="analyst",
            parent_task_id="root",
            status=AgentStatus.COMPLETED,
        )
        assert te.task_id == "t-1"
        assert te.agent_id == "a-1"
        assert te.role == "analyst"
        assert te.parent_task_id == "root"
        assert te.status == AgentStatus.COMPLETED


# -- CostConfig --------------------------------------------------------------


class TestCostConfig:
    def test_calculate_cost_known_model(self, cost_config):
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = cost_config.calculate_cost("test-model", usage)
        # input: 1000/1000 * 0.01 = 0.01
        # output: 500/1000 * 0.03 = 0.015
        assert cost == pytest.approx(0.025)

    def test_calculate_cost_unknown_model_non_strict(self):
        cc = CostConfig(models={})
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        cost = cc.calculate_cost("unknown-model", usage)
        assert cost == 0.0

    def test_calculate_cost_unknown_model_strict(self):
        cc = CostConfig(models={})
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        with pytest.raises(ValueError, match="No pricing data"):
            cc.calculate_cost("unknown-model", usage, strict=True)

    def test_calculate_cost_zero_usage(self, cost_config):
        usage = TokenUsage()
        assert cost_config.calculate_cost("test-model", usage) == 0.0


# -- AgentNode ---------------------------------------------------------------


class TestAgentNode:
    def test_defaults(self):
        node = AgentNode()
        assert isinstance(node.task_id, str)
        assert node.agent_id == ""
        assert node.parent_task_id is None
        assert node.status == AgentStatus.RUNNING
        assert node.current_phase is None
        assert node.children == []
        assert node.result is None
        assert node.depth == 0
        assert node.review_attempts == 0

    def test_from_fixture(self, agent_node):
        assert agent_node.status == AgentStatus.RUNNING
        assert agent_node.config.role == "test_analyst"
        assert agent_node.budget.max_tokens == 50000


# -- WorkflowExecution -------------------------------------------------------


class TestWorkflowExecution:
    def test_defaults(self):
        wf = WorkflowExecution()
        assert isinstance(wf.workflow_id, str)
        assert wf.root_task_id is None
        assert wf.status == WorkflowStatus.RUNNING
        assert wf.agent_tree == {}
        assert wf.created_at.tzinfo == timezone.utc

    def test_from_fixture(self, workflow):
        assert workflow.global_budget.max_tokens == 50000
        assert workflow.status == WorkflowStatus.RUNNING
