"""Shared fixtures for framework tests."""

import pytest

from parr.core_types import (
    AgentConfig,
    AgentInput,
    AgentNode,
    BudgetConfig,
    CostConfig,
    ModelConfig,
    ModelPricing,
    Phase,
    ToolDef,
    WorkflowExecution,
)
from parr.adapters import ReferenceDomainAdapter
from parr.budget_tracker import BudgetTracker
from parr.event_bus import EventBus, InMemoryEventSink


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def event_sink():
    return InMemoryEventSink()


@pytest.fixture
def cost_config():
    return CostConfig(models={
        "test-model": ModelPricing(
            input_price_per_1k=0.01,
            output_price_per_1k=0.03,
            context_window=128000,
        ),
    })


@pytest.fixture
def budget_config():
    return BudgetConfig(
        max_tokens=50000,
        max_cost=5.0,
        max_duration_ms=60000,
        max_agent_depth=3,
        max_parallel_agents=3,
    )


@pytest.fixture
def budget_tracker(cost_config):
    return BudgetTracker(cost_config=cost_config)


@pytest.fixture
def agent_config():
    return AgentConfig(
        role="test_analyst",
        system_prompt="You are a test analyst. Analyze the provided data thoroughly.",
        model="test-model",
        model_config=ModelConfig(temperature=0.7, max_tokens=4096),
    )


@pytest.fixture
def agent_input():
    return AgentInput(
        task="Analyze the test data and produce a report.",
        budget=BudgetConfig(max_tokens=50000),
    )


@pytest.fixture
def agent_node(agent_config, budget_config):
    return AgentNode(
        agent_id=agent_config.agent_id,
        config=agent_config,
        budget=budget_config,
    )


@pytest.fixture
def workflow(budget_config):
    wf = WorkflowExecution(global_budget=budget_config)
    return wf


@pytest.fixture
def sample_domain_tool():
    """A simple domain tool for testing."""
    async def _handler(query: str = "") -> str:
        return f"Search results for: {query}"

    return ToolDef(
        name="search_data",
        description="Search the test dataset.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
        handler=_handler,
        phase_availability=[Phase.ACT],
    )


@pytest.fixture
def domain_adapter(agent_config, sample_domain_tool):
    adapter = ReferenceDomainAdapter()
    adapter.register_role(
        role="test_analyst",
        config=agent_config,
        tools=[sample_domain_tool],
        output_schema={
            "type": "object",
            "properties": {
                "analysis": {"type": "string"},
                "risks": {"type": "array", "items": {"type": "string"}},
            },
        },
        report_template="## Test Report\n- Key findings\n- Risk assessment\n- Recommendations",
        description="Analyzes test data and produces risk reports",
    )
    return adapter
