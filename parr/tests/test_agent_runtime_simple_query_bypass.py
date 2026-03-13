"""Tests for AgentRuntime simple-query bypass routing."""

from __future__ import annotations

import pytest

from parr.agent_runtime import AgentRuntime
from parr.budget_tracker import BudgetTracker
from parr.core_types import (
    AgentConfig,
    AgentInput,
    AgentNode,
    BudgetConfig,
    CostConfig,
    ModelConfig,
    ModelPricing,
    SimpleQueryBypassConfig,
    WorkflowExecution,
)
from parr.event_bus import EventBus
from parr.tests.mock_llm import MockToolCallingLLM, make_text_response


def _make_runtime(llm: MockToolCallingLLM, bypass_cfg: SimpleQueryBypassConfig | None = None) -> AgentRuntime:
    cost_config = CostConfig(models={
        "test-model": ModelPricing(
            input_price_per_1k=0.001,
            output_price_per_1k=0.002,
            context_window=128000,
        )
    })
    return AgentRuntime(
        llm=llm,
        budget_tracker=BudgetTracker(cost_config=cost_config),
        event_bus=EventBus(),
        simple_query_bypass=bypass_cfg or SimpleQueryBypassConfig(),
        budget_config=BudgetConfig(max_tokens=200000),
    )


def _make_context(output_schema: dict | None = None):
    config = AgentConfig(
        role="researcher",
        system_prompt="You are a researcher.",
        model="test-model",
        model_config=ModelConfig(temperature=0.2, top_p=1.0, max_tokens=1024),
    )
    budget = BudgetConfig(max_tokens=200000)
    node = AgentNode(agent_id=config.agent_id, config=config, budget=budget)
    workflow = WorkflowExecution(global_budget=budget)
    workflow.agent_tree[node.task_id] = node
    agent_input = AgentInput(
        task="What are the phases of DPIA?",
        output_schema=output_schema,
        budget=budget,
    )
    return config, agent_input, node, workflow


@pytest.mark.asyncio
async def test_simple_query_uses_direct_answer_path():
    llm = MockToolCallingLLM([
        make_text_response(
            '{"mode":"direct_answer","confidence":0.92,'
            '"reason":"Simple factual query","requires_external_data":false}',
            input_tokens=10,
            output_tokens=20,
        ),
        make_text_response(
            '{"answer":"DPIA phases are preparation, assessment, mitigation, and review.",'
            '"confidence":0.95,"needs_full_workflow":false,"reason":"Known answer"}',
            input_tokens=12,
            output_tokens=45,
        ),
    ])
    runtime = _make_runtime(llm)
    config, agent_input, node, workflow = _make_context()

    output = await runtime.execute(config, agent_input, node, workflow)

    assert output.status == "completed"
    assert output.execution_metadata.execution_path == "direct_answer"
    assert output.execution_metadata.phases_completed == []
    assert output.execution_metadata.tools_called == []
    assert output.execution_metadata.routing_decision["selected_path"] == "direct_answer"
    assert "DPIA phases" in output.summary
    assert llm.call_count == 2
    assert llm.calls_log[0]["tool_count"] == 0
    assert llm.calls_log[1]["tool_count"] == 0


@pytest.mark.asyncio
async def test_invalid_router_output_falls_back_to_full_workflow():
    llm = MockToolCallingLLM([
        make_text_response("not-json"),
        make_text_response("Plan complete."),
        make_text_response("Act complete."),
        make_text_response("REVIEW_PASSED"),
        make_text_response("Final report"),
    ])
    runtime = _make_runtime(llm)
    config, agent_input, node, workflow = _make_context()

    output = await runtime.execute(config, agent_input, node, workflow)

    assert output.execution_metadata.execution_path == "full_workflow"
    assert output.execution_metadata.phases_completed == ["plan", "act", "review", "report"]
    assert output.execution_metadata.routing_decision["policy_reason"] == "invalid_router_output"
    assert llm.call_count == 5


@pytest.mark.asyncio
async def test_direct_path_uncertain_escalates_to_full_workflow():
    llm = MockToolCallingLLM([
        make_text_response(
            '{"mode":"direct_answer","confidence":0.96,'
            '"reason":"Looks simple","requires_external_data":false}'
        ),
        make_text_response(
            '{"answer":"","confidence":0.20,"needs_full_workflow":true,'
            '"reason":"Need deeper retrieval"}'
        ),
        make_text_response("Plan complete."),
        make_text_response("Act complete."),
        make_text_response("REVIEW_PASSED"),
        make_text_response("Final report after escalation"),
    ])
    runtime = _make_runtime(llm)
    config, agent_input, node, workflow = _make_context()

    output = await runtime.execute(config, agent_input, node, workflow)

    assert output.execution_metadata.execution_path == "full_workflow"
    assert output.execution_metadata.routing_decision["escalated_after_direct_answer"] is True
    assert output.execution_metadata.routing_decision["selected_path"] == "full_workflow"
    assert output.execution_metadata.phases_completed == ["plan", "act", "review", "report"]
    assert llm.call_count == 6


@pytest.mark.asyncio
async def test_output_schema_policy_forces_full_workflow_without_router_call():
    llm = MockToolCallingLLM([
        make_text_response("Plan complete."),
        make_text_response("Act complete."),
        make_text_response("REVIEW_PASSED"),
        make_text_response("Final report"),
    ])
    runtime = _make_runtime(llm)
    config, agent_input, node, workflow = _make_context(output_schema={"type": "object"})

    output = await runtime.execute(config, agent_input, node, workflow)

    assert output.execution_metadata.execution_path == "full_workflow"
    assert output.execution_metadata.routing_decision["policy_reason"] == "output_schema_requires_full_workflow"
    assert output.execution_metadata.phases_completed == ["plan", "act", "review", "report"]
    assert llm.call_count == 4
