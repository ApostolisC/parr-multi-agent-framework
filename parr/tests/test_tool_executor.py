"""Tests for parr.tool_executor – ToolExecutor."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict
from unittest.mock import patch

import pytest

from parr.core_types import Phase, ToolCall, ToolDef, ToolResult
from parr.tool_executor import DOCUMENT_CONTENT_TOOLS, ToolExecutor
from parr.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(*tools: ToolDef) -> ToolRegistry:
    """Return a fresh registry with the given tools registered."""
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _call(name: str = "my_tool", call_id: str = "c1", **kwargs: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=kwargs)


async def _async_echo(**kwargs: Any) -> str:
    return f"echo:{kwargs}"


def _sync_echo(**kwargs: Any) -> str:
    return f"sync:{kwargs}"


def _simple_tool(
    name: str = "my_tool",
    handler=_async_echo,
    phase_availability=None,
    parameters=None,
    **overrides: Any,
) -> ToolDef:
    """Build a minimal ToolDef with sensible defaults for testing."""
    if parameters is None:
        parameters = {"type": "object", "properties": {}}
    return ToolDef(
        name=name,
        description="A test tool.",
        parameters=parameters,
        handler=handler,
        phase_availability=phase_availability or list(Phase),
        **overrides,
    )


# ===================================================================
# 1. Basic execution
# ===================================================================

class TestBasicExecution:
    """Handler is invoked and a successful ToolResult is returned."""

    @pytest.mark.asyncio
    async def test_async_handler(self):
        tool = _simple_tool(handler=_async_echo)
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is True
        assert result.tool_call_id == "c1"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_sync_handler(self):
        tool = _simple_tool(handler=_sync_echo)
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is True
        assert "sync:" in result.content

    @pytest.mark.asyncio
    async def test_arguments_forwarded(self):
        async def handler(query: str = "") -> str:
            return query.upper()

        tool = _simple_tool(
            handler=handler,
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        )
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call(query="hello"))
        assert result.success is True
        assert result.content == "HELLO"


# ===================================================================
# 2. Unknown tool
# ===================================================================

class TestUnknownTool:

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self):
        executor = ToolExecutor(_make_registry())  # empty registry
        result = await executor.execute(_call(name="nonexistent"))
        assert result.success is False
        assert "Unknown tool" in result.error
        assert "nonexistent" in result.error


# ===================================================================
# 3. Phase availability
# ===================================================================

class TestPhaseAvailability:

    @pytest.mark.asyncio
    async def test_tool_not_available_in_phase(self):
        tool = _simple_tool(phase_availability=[Phase.PLAN])
        executor = ToolExecutor(_make_registry(tool))
        executor.set_phase(Phase.ACT)
        result = await executor.execute(_call())
        assert result.success is False
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_tool_available_in_phase(self):
        tool = _simple_tool(phase_availability=[Phase.ACT])
        executor = ToolExecutor(_make_registry(tool))
        executor.set_phase(Phase.ACT)
        result = await executor.execute(_call())
        assert result.success is True

    @pytest.mark.asyncio
    async def test_no_phase_set_skips_check(self):
        """When no phase is set, availability check is skipped."""
        tool = _simple_tool(phase_availability=[Phase.PLAN])
        executor = ToolExecutor(_make_registry(tool))
        # _current_phase is None → condition is falsy → skip
        result = await executor.execute(_call())
        assert result.success is True


# ===================================================================
# 4. Rate limiting (max_calls_per_phase)
# ===================================================================

class TestRateLimiting:

    @pytest.mark.asyncio
    async def test_respects_max_calls(self):
        tool = _simple_tool(max_calls_per_phase=2)
        executor = ToolExecutor(_make_registry(tool))
        executor.set_phase(Phase.ACT)

        r1 = await executor.execute(_call())
        r2 = await executor.execute(_call())
        r3 = await executor.execute(_call())

        assert r1.success is True
        assert r2.success is True
        assert r3.success is False
        assert "maximum" in r3.error

    @pytest.mark.asyncio
    async def test_failed_calls_count_toward_limit(self):
        """Rate limit increments BEFORE execution, so failures count."""
        call_count = 0

        async def failing_then_ok(**kw: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient")
            return "ok"

        tool = _simple_tool(handler=failing_then_ok, max_calls_per_phase=2)
        executor = ToolExecutor(_make_registry(tool))
        executor.set_phase(Phase.ACT)

        r1 = await executor.execute(_call())  # fails, count → 1
        assert r1.success is False

        r2 = await executor.execute(_call())  # succeeds, count → 2
        assert r2.success is True

        r3 = await executor.execute(_call())  # blocked by rate limit
        assert r3.success is False
        assert "maximum" in r3.error

    @pytest.mark.asyncio
    async def test_set_phase_resets_counts(self):
        tool = _simple_tool(max_calls_per_phase=1)
        executor = ToolExecutor(_make_registry(tool))
        executor.set_phase(Phase.ACT)

        r1 = await executor.execute(_call())
        assert r1.success is True

        # Switch phase → counts reset
        executor.set_phase(Phase.REVIEW)
        r2 = await executor.execute(_call())
        assert r2.success is True


# ===================================================================
# 5. Orchestrator tool
# ===================================================================

class TestOrchestratorTool:

    @pytest.mark.asyncio
    async def test_orchestrator_tool_returns_error(self):
        tool = _simple_tool(is_orchestrator_tool=True)
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is False
        assert "orchestrator" in result.error.lower()


# ===================================================================
# 6. No handler
# ===================================================================

class TestNoHandler:

    @pytest.mark.asyncio
    async def test_no_handler_returns_error(self):
        tool = _simple_tool(handler=None)
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is False
        assert "no handler" in result.error.lower()


# ===================================================================
# 7. Input validation
# ===================================================================

class TestInputValidation:

    @pytest.mark.asyncio
    async def test_invalid_arguments(self):
        tool = _simple_tool(
            parameters={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
        )
        executor = ToolExecutor(_make_registry(tool))
        # Missing required arg + wrong type
        result = await executor.execute(_call(count="not-an-int"))
        assert result.success is False
        assert "Input validation failed" in result.error

    @pytest.mark.asyncio
    async def test_valid_arguments_pass(self):
        async def handler(count: int = 0) -> str:
            return str(count)

        tool = _simple_tool(
            handler=handler,
            parameters={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            },
        )
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call(count=42))
        assert result.success is True
        assert result.content == "42"

    @pytest.mark.asyncio
    async def test_empty_parameters_skips_validation(self):
        """Tools with no parameters schema always pass input validation."""
        tool = _simple_tool(parameters={})
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is True


# ===================================================================
# 8. Output validation
# ===================================================================

class TestOutputValidation:

    @pytest.mark.asyncio
    async def test_invalid_output(self):
        async def handler(**kw: Any) -> str:
            return "not-a-dict"

        tool = _simple_tool(
            handler=handler,
            output_schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
            },
        )
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is False
        assert "Output validation failed" in result.error

    @pytest.mark.asyncio
    async def test_valid_output(self):
        async def handler(**kw: Any) -> Dict[str, str]:
            return {"result": "ok"}

        tool = _simple_tool(
            handler=handler,
            output_schema={
                "type": "object",
                "properties": {"result": {"type": "string"}},
                "required": ["result"],
            },
        )
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is True
        parsed = json.loads(result.content)
        assert parsed == {"result": "ok"}

    @pytest.mark.asyncio
    async def test_no_output_schema_skips_validation(self):
        """When output_schema is None, any handler return value is accepted."""
        async def handler(**kw: Any) -> int:
            return 123

        tool = _simple_tool(handler=handler, output_schema=None)
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is True
        assert result.content == "123"


# ===================================================================
# 9. Timeout
# ===================================================================

class TestTimeout:

    @pytest.mark.asyncio
    async def test_async_handler_timeout(self):
        async def slow(**kw: Any) -> str:
            await asyncio.sleep(10)
            return "done"

        tool = _simple_tool(handler=slow, timeout_ms=50)
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is False
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_sync_handler_timeout(self):
        import time

        def slow_sync(**kw: Any) -> str:
            time.sleep(10)
            return "done"

        tool = _simple_tool(handler=slow_sync, timeout_ms=50)
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is False
        assert "timed out" in result.error


# ===================================================================
# 10. Retry
# ===================================================================

class TestRetry:

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        attempts = 0

        async def flaky(**kw: Any) -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("transient failure")
            return "success"

        tool = _simple_tool(
            handler=flaky, retry_on_failure=True, max_retries=2, timeout_ms=5000,
        )
        executor = ToolExecutor(_make_registry(tool))

        # Patch asyncio.sleep to avoid real backoff waits
        with patch("parr.tool_executor.asyncio.sleep", return_value=None):
            result = await executor.execute(_call())

        assert result.success is True
        assert result.content == "success"
        assert attempts == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_respects_max_retries(self):
        async def always_fail(**kw: Any) -> str:
            raise RuntimeError("permanent failure")

        tool = _simple_tool(
            handler=always_fail, retry_on_failure=True, max_retries=2,
        )
        executor = ToolExecutor(_make_registry(tool))

        with patch("parr.tool_executor.asyncio.sleep", return_value=None):
            result = await executor.execute(_call())

        assert result.success is False
        assert "permanent failure" in result.error

    @pytest.mark.asyncio
    async def test_no_retry_when_flag_is_false(self):
        attempts = 0

        async def fail_once(**kw: Any) -> str:
            nonlocal attempts
            attempts += 1
            raise RuntimeError("boom")

        tool = _simple_tool(
            handler=fail_once, retry_on_failure=False, max_retries=5,
        )
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())

        assert result.success is False
        assert attempts == 1  # no retries even though max_retries=5

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        attempts = 0

        async def slow_then_fast(**kw: Any) -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                # Block until cancelled by timeout — unaffected by sleep patch
                await asyncio.get_running_loop().create_future()
            return "ok"

        tool = _simple_tool(
            handler=slow_then_fast,
            retry_on_failure=True,
            max_retries=2,
            timeout_ms=50,
        )
        executor = ToolExecutor(_make_registry(tool))

        with patch("parr.tool_executor.asyncio.sleep", return_value=None):
            result = await executor.execute(_call())

        assert result.success is True
        assert attempts == 2


# ===================================================================
# 11. Untrusted content wrapping
# ===================================================================

class TestUntrustedContentWrapping:

    @pytest.mark.asyncio
    async def test_wraps_when_flag_set(self):
        tool = _simple_tool(wraps_untrusted_content=True)
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert "<untrusted_document_content>" in result.content
        assert "</untrusted_document_content>" in result.content

    @pytest.mark.asyncio
    async def test_wraps_legacy_document_content_tools(self):
        """Tools in DOCUMENT_CONTENT_TOOLS are wrapped even without the flag."""
        legacy_name = next(iter(DOCUMENT_CONTENT_TOOLS))

        async def handler(**kw: Any) -> str:
            return "doc content"

        tool = _simple_tool(
            name=legacy_name,
            handler=handler,
            wraps_untrusted_content=False,
        )
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call(name=legacy_name))
        assert "<untrusted_document_content>" in result.content
        assert "doc content" in result.content

    @pytest.mark.asyncio
    async def test_no_wrapping_for_normal_tool(self):
        tool = _simple_tool(wraps_untrusted_content=False)
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert "<untrusted_document_content>" not in result.content


# ===================================================================
# 12. Serialization
# ===================================================================

class TestSerialization:

    @pytest.mark.asyncio
    async def test_dict_with_output_schema_uses_json_dumps(self):
        async def handler(**kw: Any) -> dict:
            return {"key": "value", "num": 1}

        tool = _simple_tool(
            handler=handler,
            output_schema={"type": "object"},
        )
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is True
        parsed = json.loads(result.content)
        assert parsed == {"key": "value", "num": 1}

    @pytest.mark.asyncio
    async def test_list_with_output_schema_uses_json_dumps(self):
        async def handler(**kw: Any) -> list:
            return [1, 2, 3]

        tool = _simple_tool(
            handler=handler,
            output_schema={"type": "array", "items": {"type": "integer"}},
        )
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is True
        assert json.loads(result.content) == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_non_dict_uses_str(self):
        async def handler(**kw: Any) -> int:
            return 42

        tool = _simple_tool(handler=handler)
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.content == "42"

    @pytest.mark.asyncio
    async def test_dict_without_output_schema_uses_str(self):
        async def handler(**kw: Any) -> dict:
            return {"a": 1}

        tool = _simple_tool(handler=handler, output_schema=None)
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        # No output_schema → str() is used
        assert result.content == str({"a": 1})


# ===================================================================
# Static method unit tests
# ===================================================================

class TestStaticHelpers:

    def test_validate_input_returns_none_for_valid(self):
        tool = _simple_tool(
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            },
        )
        assert ToolExecutor._validate_input(tool, {"x": 1}) is None

    def test_validate_input_returns_error_for_invalid(self):
        tool = _simple_tool(
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )
        err = ToolExecutor._validate_input(tool, {"x": "bad"})
        assert err is not None
        assert "Input validation failed" in err

    def test_validate_output_returns_none_when_no_schema(self):
        tool = _simple_tool(output_schema=None)
        assert ToolExecutor._validate_output(tool, "anything") is None

    def test_validate_output_returns_error_for_invalid(self):
        tool = _simple_tool(
            output_schema={"type": "string"},
        )
        err = ToolExecutor._validate_output(tool, 123)
        assert err is not None
        assert "Output validation failed" in err

    def test_serialize_output_json(self):
        tool = _simple_tool(output_schema={"type": "object"})
        s = ToolExecutor._serialize_output(tool, {"a": 1})
        assert json.loads(s) == {"a": 1}

    def test_serialize_output_str_fallback(self):
        tool = _simple_tool(output_schema=None)
        assert ToolExecutor._serialize_output(tool, 99) == "99"

    def test_wrap_untrusted(self):
        wrapped = ToolExecutor._wrap_untrusted("hello")
        assert wrapped == (
            "<untrusted_document_content>\nhello\n</untrusted_document_content>"
        )
