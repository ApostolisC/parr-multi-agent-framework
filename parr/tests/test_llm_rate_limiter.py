"""Tests for the centralized LLM queue/rate-limiter wrapper."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from parr.adapters.llm_rate_limiter import RateLimitedToolCallingLLM
from parr.config import ConfigError, create_orchestrator_from_config, load_config
from parr.core_types import LLMRateLimitConfig, Message, MessageRole, ModelConfig
from parr.tests.mock_llm import make_text_response


class _ObservedLLM:
    """Simple LLM stub that records call start order/times/concurrency."""

    def __init__(self, sleep_seconds: float = 0.0) -> None:
        self._sleep_seconds = sleep_seconds
        self._lock = asyncio.Lock()
        self._active = 0
        self.max_active = 0
        self.started_labels: List[str] = []
        self.started_times: List[float] = []

    async def chat_with_tools(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        model: str,
        model_config: ModelConfig,
        stream: bool = False,
        on_token: Any = None,
    ):
        label = messages[-1].content if messages else ""
        now = time.monotonic()
        async with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            self.started_labels.append(str(label))
            self.started_times.append(now)

        try:
            if self._sleep_seconds > 0:
                await asyncio.sleep(self._sleep_seconds)
            return make_text_response(f"ok:{label}", input_tokens=1, output_tokens=1)
        finally:
            async with self._lock:
                self._active -= 1


async def _invoke_llm(llm, label: str) -> None:
    await llm.chat_with_tools(
        messages=[Message(role=MessageRole.USER, content=label)],
        tools=[],
        model="test-model",
        model_config=ModelConfig(),
    )


def _write_minimal_config(config_dir: Path, budget_yaml: str) -> None:
    (config_dir / "system_prompts").mkdir(parents=True, exist_ok=True)
    (config_dir / "system_prompts" / "analyst.md").write_text(
        "You are an analyst.", encoding="utf-8"
    )
    (config_dir / "roles.yaml").write_text(
        "\n".join(
            [
                "roles:",
                "  analyst:",
                "    model: test-model",
                "    system_prompt: system_prompts/analyst.md",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "models.yaml").write_text(
        "\n".join(
            [
                "models:",
                "  test-model:",
                "    input_price_per_1k: 0.001",
                "    output_price_per_1k: 0.002",
                "    context_window: 128000",
            ]
        ),
        encoding="utf-8",
    )
    (config_dir / "budget.yaml").write_text(budget_yaml, encoding="utf-8")


@pytest.mark.asyncio
async def test_rate_limiter_enforces_max_concurrency_one() -> None:
    inner = _ObservedLLM(sleep_seconds=0.03)
    limited = RateLimitedToolCallingLLM(
        llm=inner,
        config=LLMRateLimitConfig(enabled=True, max_concurrent_requests=1),
    )

    await asyncio.gather(*[_invoke_llm(limited, str(i)) for i in range(5)])

    assert inner.max_active == 1


@pytest.mark.asyncio
async def test_rate_limiter_enforces_rolling_window_and_fifo() -> None:
    inner = _ObservedLLM(sleep_seconds=0.0)
    limited = RateLimitedToolCallingLLM(
        llm=inner,
        config=LLMRateLimitConfig(
            enabled=True,
            max_requests_per_window=2,
            window_seconds=0.2,
            max_concurrent_requests=5,
        ),
    )

    await asyncio.gather(*[_invoke_llm(limited, str(i)) for i in range(4)])

    assert inner.started_labels == ["0", "1", "2", "3"]
    assert (inner.started_times[2] - inner.started_times[0]) >= 0.16


@pytest.mark.asyncio
async def test_rate_limiter_enforces_token_window() -> None:
    inner = _ObservedLLM(sleep_seconds=0.0)
    limited = RateLimitedToolCallingLLM(
        llm=inner,
        config=LLMRateLimitConfig(
            enabled=True,
            max_tokens_per_window=2500,
            window_seconds=0.2,
            max_concurrent_requests=5,
        ),
    )

    big_payload = "x" * 4000
    await asyncio.gather(*[_invoke_llm(limited, big_payload) for _ in range(2)])

    assert len(inner.started_times) == 2
    assert (inner.started_times[1] - inner.started_times[0]) >= 0.16


def test_config_loader_wraps_user_llm_when_llm_rate_limit_enabled(tmp_path: Path) -> None:
    _write_minimal_config(
        tmp_path,
        "\n".join(
            [
                "budget_defaults: {}",
                "phase_limits: {}",
                "llm_rate_limit:",
                "  enabled: true",
                "  max_concurrent_requests: 1",
            ]
        ),
    )

    inner = _ObservedLLM()
    orchestrator = create_orchestrator_from_config(config_dir=tmp_path, llm=inner)

    assert isinstance(orchestrator._llm, RateLimitedToolCallingLLM)


def test_config_loader_rejects_invalid_llm_rate_limit(tmp_path: Path) -> None:
    _write_minimal_config(
        tmp_path,
        "\n".join(
            [
                "budget_defaults: {}",
                "phase_limits: {}",
                "llm_rate_limit:",
                "  enabled: true",
            ]
        ),
    )

    with pytest.raises(ConfigError, match="llm_rate_limit"):
        load_config(tmp_path)


def test_config_loader_parses_simple_query_bypass(tmp_path: Path) -> None:
    _write_minimal_config(
        tmp_path,
        "\n".join(
            [
                "budget_defaults: {}",
                "phase_limits: {}",
                "llm_rate_limit:",
                "  enabled: false",
                "simple_query_bypass:",
                "  enabled: true",
                "  route_confidence_threshold: 0.9",
                "  force_full_workflow_if_output_schema: true",
                "  allow_escalation_to_full_workflow: false",
                "  direct_answer_max_tokens: 384",
            ]
        ),
    )

    bundle = load_config(tmp_path)
    cfg = bundle.simple_query_bypass

    assert cfg.enabled is True
    assert cfg.route_confidence_threshold == pytest.approx(0.9)
    assert cfg.force_full_workflow_if_output_schema is True
    assert cfg.allow_escalation_to_full_workflow is False
    assert cfg.direct_answer_max_tokens == 384


def test_config_loader_rejects_invalid_simple_query_bypass(tmp_path: Path) -> None:
    _write_minimal_config(
        tmp_path,
        "\n".join(
            [
                "budget_defaults: {}",
                "phase_limits: {}",
                "llm_rate_limit:",
                "  enabled: false",
                "simple_query_bypass:",
                "  route_confidence_threshold: 1.5",
            ]
        ),
    )

    with pytest.raises(ConfigError, match="simple_query_bypass"):
        load_config(tmp_path)
