"""Tests for the adaptive agent-controlled phase flow.

Covers:
1. AdaptiveFlowConfig dataclass
2. Entry phase detection (_detect_entry_phase)
3. set_next_phase tool + AgentWorkingMemory
4. ToolRegistry.get_for_entry()
5. ContextManager.build_entry_messages()
6. PhaseRunner.run_continuation()
7. AgentRuntime._run_adaptive_flow() — direct answer, light work, deep work
8. Orchestrator propagation
9. Config loader/validator for adaptive_flow
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from parr.core_types import (
    AdaptiveFlowConfig,
    AgentConfig,
    AgentInput,
    AgentNode,
    AgentOutput,
    BudgetConfig,
    ExecutionMetadata,
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
    Phase,
    PhaseConfig,
    SimpleQueryBypassConfig,
    StallDetectionConfig,
    ToolCall,
    ToolDef,
    ToolResult,
    TokenUsage,
    WorkflowExecution,
    generate_id,
)
from parr.agent_runtime import AgentRuntime, EXECUTION_PATH_ADAPTIVE
from parr.budget_tracker import BudgetTracker
from parr.context_manager import ContextManager
from parr.event_bus import EventBus, InMemoryEventSink
from parr.framework_tools import AgentWorkingMemory, build_transition_tools
from parr.orchestrator import Orchestrator
from parr.tool_registry import ToolRegistry
from parr.tests.mock_llm import (
    MockToolCallingLLM,
    make_text_response,
    make_tool_call_response,
)


# =========================================================================
# Helpers
# =========================================================================

def _make_config(role="test_agent", system_prompt="You are a test agent."):
    return AgentConfig(
        agent_id=generate_id(),
        role=role,
        system_prompt=system_prompt,
        model="test-model",
        model_config=ModelConfig(temperature=0.3, max_tokens=1024),
    )


def _make_input(task="Test task.", tools=None, output_schema=None):
    return AgentInput(
        task=task,
        tools=tools or [],
        output_schema=output_schema,
    )


def _make_node(config=None, budget=None):
    config = config or _make_config()
    budget = budget or BudgetConfig(max_tokens=100000)
    return AgentNode(
        task_id=generate_id(),
        agent_id=config.agent_id,
        config=config,
        budget=budget,
    )


def _make_workflow():
    return WorkflowExecution(
        workflow_id=generate_id(),
        root_task_id=generate_id(),
    )


def _make_domain_tool(name="search_data"):
    async def handler(query: str = "") -> str:
        return f"Results for: {query}"

    return ToolDef(
        name=name,
        description=f"Search with {name}.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        handler=handler,
        phase_availability=[Phase.ACT],
    )


def _make_runtime(llm, adaptive_config=None, budget_config=None):
    budget_config = budget_config or BudgetConfig(max_tokens=100000)
    tracker = BudgetTracker()
    event_bus = EventBus()
    return AgentRuntime(
        llm=llm,
        budget_tracker=tracker,
        event_bus=event_bus,
        adaptive_config=adaptive_config,
        budget_config=budget_config,
        simple_query_bypass=SimpleQueryBypassConfig(enabled=False),
    )


# =========================================================================
# 1. AdaptiveFlowConfig dataclass
# =========================================================================

class TestAdaptiveFlowConfig:

    def test_defaults(self):
        cfg = AdaptiveFlowConfig()
        assert cfg.enabled is True
        assert cfg.entry_phase_limit == 3

    def test_custom_values(self):
        cfg = AdaptiveFlowConfig(enabled=False, entry_phase_limit=5)
        assert cfg.enabled is False
        assert cfg.entry_phase_limit == 5

    def test_frozen(self):
        cfg = AdaptiveFlowConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = False


# =========================================================================
# 2. Entry phase detection
# =========================================================================

class TestDetectEntryPhase:

    def test_create_todo_list_detects_plan(self):
        tc = [ToolCall(id="c1", name="create_todo_list", arguments={})]
        assert AgentRuntime._detect_entry_phase(tc) == Phase.PLAN

    def test_update_todo_list_detects_plan(self):
        tc = [ToolCall(id="c1", name="update_todo_list", arguments={})]
        assert AgentRuntime._detect_entry_phase(tc) == Phase.PLAN

    def test_domain_tool_detects_act(self):
        tc = [ToolCall(id="c1", name="search_data", arguments={})]
        assert AgentRuntime._detect_entry_phase(tc) == Phase.ACT

    def test_log_finding_detects_act(self):
        tc = [ToolCall(id="c1", name="log_finding", arguments={})]
        assert AgentRuntime._detect_entry_phase(tc) == Phase.ACT

    def test_mixed_plan_and_domain_detects_plan(self):
        """If both planning and domain tools used, PLAN takes priority."""
        tc = [
            ToolCall(id="c1", name="search_data", arguments={}),
            ToolCall(id="c2", name="create_todo_list", arguments={}),
        ]
        assert AgentRuntime._detect_entry_phase(tc) == Phase.PLAN

    def test_empty_list_detects_act(self):
        assert AgentRuntime._detect_entry_phase([]) == Phase.ACT


# =========================================================================
# 3. set_next_phase tool + AgentWorkingMemory
# =========================================================================

class TestSetNextPhase:

    def test_memory_default_none(self):
        mem = AgentWorkingMemory()
        assert mem.requested_next_phase is None

    def test_set_next_phase_stores_value(self):
        mem = AgentWorkingMemory()
        result = mem.set_next_phase("review", "Quality check needed")
        assert mem.requested_next_phase == "review"
        assert "review" in result

    def test_set_next_phase_report(self):
        mem = AgentWorkingMemory()
        mem.set_next_phase("report", "Ready to report")
        assert mem.requested_next_phase == "report"

    def test_build_transition_tools_returns_tool(self):
        mem = AgentWorkingMemory()
        tools = build_transition_tools(mem)
        assert len(tools) == 1
        assert tools[0].name == "set_next_phase"

    def test_transition_tool_phase_availability(self):
        mem = AgentWorkingMemory()
        tools = build_transition_tools(mem)
        assert Phase.PLAN in tools[0].phase_availability
        assert Phase.ACT in tools[0].phase_availability
        assert Phase.REVIEW in tools[0].phase_availability
        assert Phase.REPORT not in tools[0].phase_availability

    def test_transition_tool_accepts_all_phases(self):
        mem = AgentWorkingMemory()
        tools = build_transition_tools(mem)
        handler = tools[0].handler
        for phase in ("plan", "act", "review", "report"):
            mem.requested_next_phase = None
            result = handler(phase=phase, reason="test")
            assert mem.requested_next_phase == phase
            assert phase in result

    def test_transition_tool_rejects_invalid_phase(self):
        mem = AgentWorkingMemory()
        tools = build_transition_tools(mem)
        handler = tools[0].handler
        result = handler(phase="invalid", reason="test")
        assert "Error" in result
        assert mem.requested_next_phase is None

    def test_transition_tool_handler(self):
        mem = AgentWorkingMemory()
        tools = build_transition_tools(mem)
        handler = tools[0].handler
        result = handler(phase="review", reason="Need QA")
        assert mem.requested_next_phase == "review"
        assert "review" in result


# =========================================================================
# 4. ToolRegistry.get_for_entry()
# =========================================================================

class TestGetForEntry:

    def test_excludes_review_report_tools(self):
        registry = ToolRegistry()
        # Register a mix of tools
        from parr.framework_tools import (
            build_plan_tools,
            build_act_tools,
            build_review_tools,
            build_report_tools,
        )
        mem = AgentWorkingMemory()
        for t in build_plan_tools(mem):
            registry.register(t)
        for t in build_act_tools(mem):
            registry.register(t)
        for t in build_review_tools(mem):
            registry.register(t)
        for t in build_report_tools(mem, None, None):
            registry.register(t)

        entry_tools = registry.get_for_entry()
        entry_names = {t.name for t in entry_tools}

        # These should be excluded
        assert "review_checklist" not in entry_names
        assert "get_report_template" not in entry_names
        assert "submit_report" not in entry_names

        # These should be included
        assert "create_todo_list" in entry_names
        assert "log_finding" in entry_names

    def test_includes_domain_tools(self):
        registry = ToolRegistry()
        domain = _make_domain_tool("my_search")
        registry.register(domain)

        entry_tools = registry.get_for_entry()
        entry_names = {t.name for t in entry_tools}
        assert "my_search" in entry_names


# =========================================================================
# 5. ContextManager.build_entry_messages()
# =========================================================================

class TestBuildEntryMessages:

    def test_returns_system_and_user_messages(self):
        cm = ContextManager()
        config = _make_config()
        input = _make_input(task="Do something.")
        tools = [_make_domain_tool()]

        messages = cm.build_entry_messages(config, input, tools)
        assert len(messages) == 2
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[1].role == MessageRole.USER

    def test_system_prompt_includes_tool_descriptions(self):
        cm = ContextManager()
        config = _make_config()
        tool = _make_domain_tool("my_custom_tool")
        input = _make_input(task="Test.")

        messages = cm.build_entry_messages(config, input, [tool])
        system_content = messages[0].content
        assert "my_custom_tool" in system_content

    def test_user_message_includes_task(self):
        cm = ContextManager()
        config = _make_config()
        input = _make_input(task="Analyze DPIA risks.")

        messages = cm.build_entry_messages(config, input, [])
        user_content = messages[1].content
        assert "Analyze DPIA risks" in user_content

    def test_output_schema_notice(self):
        cm = ContextManager()
        config = _make_config()
        input = _make_input(
            task="Generate report.",
            output_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
        )

        messages = cm.build_entry_messages(config, input, [])
        system_content = messages[0].content
        assert "JSON schema" in system_content

    def test_entry_prompt_mentions_batch_calls(self):
        """ENTRY_PROMPT should encourage batch/parallel tool calls."""
        cm = ContextManager()
        config = _make_config()
        input = _make_input(task="Explore.")
        messages = cm.build_entry_messages(config, input, [_make_domain_tool()])
        system_content = messages[0].content
        assert "multiple tools" in system_content.lower() or "batch" in system_content.lower()

    def test_entry_prompt_no_clarification_steps(self):
        """ENTRY_PROMPT should warn against planning clarification steps."""
        cm = ContextManager()
        config = _make_config()
        input = _make_input(task="Investigate.")
        messages = cm.build_entry_messages(config, input, [_make_domain_tool()])
        system_content = messages[0].content
        assert "cannot interact with the user" in system_content.lower() or \
               "cannot ask the user" in system_content.lower()

    def test_entry_prompt_broad_questions_guidance(self):
        """ENTRY_PROMPT should guide exploration for broad/open-ended questions."""
        cm = ContextManager()
        config = _make_config()
        input = _make_input(task="What risks exist?")
        messages = cm.build_entry_messages(config, input, [_make_domain_tool()])
        system_content = messages[0].content
        # Should mention exploring or casting a wide net
        assert "explore" in system_content.lower()


# =========================================================================
# 6. AgentRuntime adaptive flow — Direct Answer
# =========================================================================

class TestAdaptiveDirectAnswer:

    @pytest.mark.asyncio
    async def test_text_only_response_produces_direct_answer(self):
        """When LLM responds with text only (no tool calls), result is direct answer."""
        llm = MockToolCallingLLM([
            make_text_response("The answer is 42."),
        ])
        runtime = _make_runtime(llm, AdaptiveFlowConfig())
        config = _make_config()
        node = _make_node(config)
        workflow = _make_workflow()
        input = _make_input(task="What is the answer?")

        output = await runtime.execute(config, input, node, workflow)

        assert output.status == "completed"
        assert output.execution_metadata.execution_path == "adaptive"
        assert output.execution_metadata.detected_mode == "direct_answer"
        assert "42" in output.summary

    @pytest.mark.asyncio
    async def test_direct_answer_no_tools_called(self):
        """Direct answer should have no tool calls in metadata."""
        llm = MockToolCallingLLM([
            make_text_response("Simple answer."),
        ])
        runtime = _make_runtime(llm, AdaptiveFlowConfig())
        config = _make_config()
        node = _make_node(config)
        workflow = _make_workflow()

        output = await runtime.execute(config, _make_input(), node, workflow)
        assert output.execution_metadata.tools_called == []

    @pytest.mark.asyncio
    async def test_direct_answer_single_llm_call(self):
        """Direct answer should use exactly 1 LLM call."""
        llm = MockToolCallingLLM([
            make_text_response("Quick answer."),
        ])
        runtime = _make_runtime(llm, AdaptiveFlowConfig())
        config = _make_config()
        node = _make_node(config)
        workflow = _make_workflow()

        await runtime.execute(config, _make_input(), node, workflow)
        assert llm.call_count == 1


# =========================================================================
# 7. AgentRuntime adaptive flow — Light Work (ACT → REPORT)
# =========================================================================

class TestAdaptiveLightWork:

    @pytest.mark.asyncio
    async def test_domain_tool_triggers_act_then_report(self):
        """Calling a domain tool in entry → ACT detected → continues → REPORT."""
        llm = MockToolCallingLLM([
            # Entry call: agent calls search_data (domain tool)
            make_tool_call_response("search_data", {"query": "risks"}),
            # ACT continuation: agent calls log_finding then finishes
            make_tool_call_response(
                "log_finding",
                {"category": "risk", "content": "Found risk", "source": "search"},
            ),
            make_text_response("Analysis complete."),
            # REPORT phase
            make_tool_call_response(
                "submit_report",
                {"summary": "Risk found.", "findings": {"risks": ["r1"]}},
            ),
        ])
        runtime = _make_runtime(llm, AdaptiveFlowConfig())
        config = _make_config()
        node = _make_node(config)
        workflow = _make_workflow()
        tool = _make_domain_tool("search_data")

        output = await runtime.execute(
            config, _make_input(tools=[tool]), node, workflow,
        )

        assert output.status in ("completed", "degraded")
        assert output.execution_metadata.detected_mode == "light_work"
        phases = output.execution_metadata.phases_completed
        assert "act" in phases
        assert "report" in phases
        # PLAN and REVIEW should NOT be in phases
        assert "plan" not in phases

    @pytest.mark.asyncio
    async def test_light_work_skips_review_by_default(self):
        """Without set_next_phase('review'), REVIEW is skipped."""
        llm = MockToolCallingLLM([
            make_tool_call_response("search_data", {"query": "test"}),
            make_tool_call_response(
                "log_finding",
                {"category": "data", "content": "Data found", "source": "test"},
            ),
            make_text_response("Done."),
            make_tool_call_response(
                "submit_report",
                {"summary": "Done.", "findings": {}},
            ),
        ])
        runtime = _make_runtime(llm, AdaptiveFlowConfig())
        config = _make_config()
        node = _make_node(config)
        workflow = _make_workflow()

        output = await runtime.execute(
            config, _make_input(tools=[_make_domain_tool()]), node, workflow,
        )
        assert "review" not in output.execution_metadata.phases_completed


# =========================================================================
# 8. AgentRuntime adaptive flow — Deep Work (PLAN → ACT → REPORT)
# =========================================================================

class TestAdaptiveDeepWork:

    @pytest.mark.asyncio
    async def test_create_todo_list_triggers_plan_then_act_report(self):
        """Calling create_todo_list in entry → PLAN → ACT → REPORT."""
        llm = MockToolCallingLLM([
            # Entry: agent creates todo list
            make_tool_call_response(
                "create_todo_list",
                {"items": [{"description": "Step 1"}, {"description": "Step 2"}]},
            ),
            # PLAN continuation: agent finishes planning
            make_text_response("Plan complete: will execute step 1 and 2."),
            # ACT phase
            make_tool_call_response(
                "log_finding",
                {"category": "data", "content": "Step 1 done", "source": "analysis"},
            ),
            make_text_response("All steps done."),
            # REPORT phase
            make_tool_call_response(
                "submit_report",
                {"summary": "Completed analysis.", "findings": {"steps": ["done"]}},
            ),
        ])
        runtime = _make_runtime(llm, AdaptiveFlowConfig())
        config = _make_config()
        node = _make_node(config)
        workflow = _make_workflow()

        output = await runtime.execute(config, _make_input(), node, workflow)

        assert output.status in ("completed", "degraded")
        assert output.execution_metadata.detected_mode == "deep_work"
        phases = output.execution_metadata.phases_completed
        assert "plan" in phases
        assert "act" in phases
        assert "report" in phases


# =========================================================================
# 9. Legacy flow preserved when adaptive disabled
# =========================================================================

class TestLegacyFlowPreserved:

    @pytest.mark.asyncio
    async def test_no_adaptive_config_uses_legacy(self):
        """Without adaptive_config, the legacy router path is used."""
        llm = MockToolCallingLLM([
            # Router call (legacy)
            make_text_response('{"mode":"full_workflow","confidence":0.9,"reason":"test"}'),
            # PLAN
            make_text_response("Plan done."),
            # ACT
            make_tool_call_response(
                "log_finding",
                {"category": "x", "content": "y", "source": "z"},
            ),
            make_text_response("Act done."),
            # REVIEW
            make_text_response("REVIEW_PASSED"),
            # REPORT
            make_tool_call_response(
                "submit_report",
                {"summary": "Done.", "findings": {}},
            ),
        ])
        runtime = _make_runtime(llm, adaptive_config=None)
        config = _make_config()
        node = _make_node(config)
        workflow = _make_workflow()

        output = await runtime.execute(config, _make_input(), node, workflow)
        # Legacy path uses "full_workflow" not "adaptive"
        assert output.execution_metadata.execution_path != "adaptive"

    @pytest.mark.asyncio
    async def test_adaptive_disabled_uses_legacy(self):
        """With adaptive_config.enabled=False, legacy path is used."""
        llm = MockToolCallingLLM([
            make_text_response('{"mode":"full_workflow","confidence":0.9,"reason":"test"}'),
            make_text_response("Plan."),
            make_text_response("Act."),
            make_text_response("REVIEW_PASSED"),
            make_text_response("Report."),
        ])
        cfg = AdaptiveFlowConfig(enabled=False)
        runtime = _make_runtime(llm, adaptive_config=cfg)
        config = _make_config()
        node = _make_node(config)
        workflow = _make_workflow()

        output = await runtime.execute(config, _make_input(), node, workflow)
        assert output.execution_metadata.execution_path != "adaptive"


# =========================================================================
# 10. Budget tracking in adaptive flow
# =========================================================================

class TestAdaptiveBudget:

    @pytest.mark.asyncio
    async def test_usage_tracked_across_phases(self):
        """Token usage accumulates across entry call + continuation phases."""
        llm = MockToolCallingLLM([
            make_text_response("Direct answer.", input_tokens=100, output_tokens=50),
        ])
        runtime = _make_runtime(llm, AdaptiveFlowConfig())
        config = _make_config()
        node = _make_node(config)
        workflow = _make_workflow()

        output = await runtime.execute(config, _make_input(), node, workflow)
        assert output.token_usage.input_tokens >= 100
        assert output.token_usage.output_tokens >= 50


# =========================================================================
# 11. Orchestrator propagation
# =========================================================================

class TestOrchestratorAdaptiveConfig:

    @pytest.mark.asyncio
    async def test_orchestrator_passes_adaptive_config(self):
        """Orchestrator with adaptive_config runs the adaptive flow."""
        llm = MockToolCallingLLM([
            # Entry: direct answer
            make_text_response("Here is the answer."),
        ])
        orch = Orchestrator(
            llm=llm,
            adaptive_config=AdaptiveFlowConfig(),
            simple_query_bypass=SimpleQueryBypassConfig(enabled=False),
        )
        output = await orch.start_workflow(
            task="Quick question.",
            role="tester",
            system_prompt="Test prompt.",
            model="test-model",
            budget=BudgetConfig(max_tokens=100000),
        )
        assert output.execution_metadata.execution_path == "adaptive"
        assert output.execution_metadata.detected_mode == "direct_answer"


# =========================================================================
# 12. Config loader/validator
# =========================================================================

class TestConfigValidation:

    def test_validate_adaptive_flow_valid(self):
        from parr.config.config_validator import _validate_adaptive_flow
        errors = []
        _validate_adaptive_flow({"enabled": True, "entry_phase_limit": 5}, errors)
        assert errors == []

    def test_validate_adaptive_flow_invalid_enabled(self):
        from parr.config.config_validator import _validate_adaptive_flow
        errors = []
        _validate_adaptive_flow({"enabled": "yes"}, errors)
        assert len(errors) == 1
        assert "boolean" in errors[0]

    def test_validate_adaptive_flow_invalid_limit(self):
        from parr.config.config_validator import _validate_adaptive_flow
        errors = []
        _validate_adaptive_flow({"entry_phase_limit": -1}, errors)
        assert len(errors) == 1
        assert "positive integer" in errors[0]

    def test_validate_adaptive_flow_empty_ok(self):
        from parr.config.config_validator import _validate_adaptive_flow
        errors = []
        _validate_adaptive_flow({}, errors)
        assert errors == []

    def test_build_adaptive_flow_config(self):
        from parr.config.config_loader import _build_adaptive_flow_config
        cfg = _build_adaptive_flow_config({"enabled": True, "entry_phase_limit": 7})
        assert cfg is not None
        assert cfg.enabled is True
        assert cfg.entry_phase_limit == 7

    def test_build_adaptive_flow_config_empty(self):
        from parr.config.config_loader import _build_adaptive_flow_config
        cfg = _build_adaptive_flow_config({})
        assert cfg is None


# =========================================================================
# 13. Exports
# =========================================================================

class TestPlanPromptQuality:
    """PLAN phase prompt should not allow non-actionable clarification steps."""

    def test_plan_prompt_no_clarify_steps(self):
        from parr.context_manager import PHASE_PROMPTS
        plan_prompt = PHASE_PROMPTS[Phase.PLAN].lower()
        assert "cannot interact with the user" in plan_prompt or \
               "cannot ask" in plan_prompt

    def test_plan_prompt_mentions_batch_calls(self):
        from parr.context_manager import PHASE_PROMPTS
        plan_prompt = PHASE_PROMPTS[Phase.PLAN].lower()
        assert "batch" in plan_prompt


class TestExports:

    def test_adaptive_flow_config_exported(self):
        import parr
        assert hasattr(parr, "AdaptiveFlowConfig")
        assert parr.AdaptiveFlowConfig is AdaptiveFlowConfig
