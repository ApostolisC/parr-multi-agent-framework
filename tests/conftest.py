"""Shared fixtures for PARR framework tests.

All tests use mock LLMs — zero API calls, zero cost.
"""
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from parr.core_types import (
    AgentConfig,
    AgentInput,
    AgentNode,
    BudgetConfig,
    CostConfig,
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
    ModelPricing,
    Phase,
    PoliciesConfig,
    SpawnPolicy,
    ToolCall,
    ToolDef,
    ToolResult,
    WorkflowExecution,
    generate_id,
)
from parr.event_bus import EventBus, InMemoryEventSink
from parr.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockLLM:
    """A mock ToolCallingLLM that returns scripted responses.

    Usage:
        llm = MockLLM([
            LLMResponse(content="I'll log a finding", tool_calls=[...]),
            LLMResponse(content="Done", tool_calls=[]),
        ])
    """

    def __init__(self, responses: Optional[List[LLMResponse]] = None):
        self._responses = list(responses or [])
        self._call_index = 0
        self.calls: List[Dict[str, Any]] = []

    def add_response(self, response: LLMResponse):
        self._responses.append(response)

    async def chat_with_tools(
        self,
        messages: List[Message],
        tools: List[Any] = None,
        model: str = None,
        model_config: ModelConfig = None,
        **kwargs,
    ) -> LLMResponse:
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "model": model,
            "model_config": model_config,
            "kwargs": kwargs,
        })
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        # Default: text-only response that ends the phase
        return LLMResponse(content="No more scripted responses.", tool_calls=[])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tool_call(name: str, arguments: dict = None, call_id: str = None) -> ToolCall:
    return ToolCall(
        id=call_id or generate_id(),
        name=name,
        arguments=arguments or {},
    )


def make_text_response(text: str) -> LLMResponse:
    return LLMResponse(content=text, tool_calls=[])


def make_tool_response(text: str, tool_calls: List[ToolCall]) -> LLMResponse:
    return LLMResponse(content=text, tool_calls=tool_calls)


def make_budget(max_tokens=1000000, max_cost=10.0, max_agent_depth=4,
                max_sub_agents_total=10, max_parallel_agents=5) -> BudgetConfig:
    return BudgetConfig(
        max_tokens=max_tokens,
        max_cost=max_cost,
        max_agent_depth=max_agent_depth,
        max_sub_agents_total=max_sub_agents_total,
        max_parallel_agents=max_parallel_agents,
    )


def make_cost_config() -> CostConfig:
    return CostConfig(models={
        "test-model": ModelPricing(
            input_price_per_1k=0.001,
            output_price_per_1k=0.002,
            context_window=128000,
        ),
    })


def make_workflow(budget: BudgetConfig = None) -> WorkflowExecution:
    wf = WorkflowExecution()
    if budget:
        wf.global_budget = budget
    return wf


def make_agent_node(
    role: str = "test_role",
    sub_role: str = None,
    depth: int = 0,
    budget: BudgetConfig = None,
) -> AgentNode:
    node = AgentNode(
        agent_id=generate_id(),
        config=AgentConfig(role=role, sub_role=sub_role),
        budget=budget or make_budget(),
        depth=depth,
    )
    return node


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def event_sink():
    return InMemoryEventSink()
