"""Tests for tool lifecycle hooks: ToolMiddleware, ToolContext, and ToolRegistry.override()."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

from parr.core_types import Phase, ToolCall, ToolContext, ToolDef, ToolMiddleware, ToolResult
from parr.tool_executor import ToolExecutor
from parr.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(*tools: ToolDef) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _call(name: str = "my_tool", call_id: str = "c1", **kwargs: Any) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=kwargs)


async def _async_echo(**kwargs: Any) -> str:
    return f"echo:{kwargs}"


def _simple_tool(
    name: str = "my_tool",
    handler=_async_echo,
    phase_availability=None,
    parameters=None,
    **overrides: Any,
) -> ToolDef:
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


# ---------------------------------------------------------------------------
# Test middleware implementations
# ---------------------------------------------------------------------------

class RecordingMiddleware(ToolMiddleware):
    """Records all hook invocations for assertion."""

    def __init__(self) -> None:
        self.pre_calls: List[str] = []
        self.post_calls: List[str] = []
        self.error_calls: List[str] = []

    async def pre_call(self, tool_call, tool_def, context):
        self.pre_calls.append(tool_call.name)
        return tool_call

    async def post_call(self, result, tool_call, tool_def, context):
        self.post_calls.append(tool_call.name)
        return result

    async def on_error(self, error, tool_call, tool_def, context, attempt, max_attempts):
        self.error_calls.append(f"{tool_call.name}:attempt={attempt}")
        return None


class ModifyingPreCallMiddleware(ToolMiddleware):
    """Modifies tool_call arguments in pre_call."""

    async def pre_call(self, tool_call, tool_def, context):
        new_args = {**tool_call.arguments, "injected": True}
        return ToolCall(id=tool_call.id, name=tool_call.name, arguments=new_args)


class ShortCircuitMiddleware(ToolMiddleware):
    """Returns a ToolResult in pre_call, skipping handler execution."""

    def __init__(self, content: str = "cached") -> None:
        self._content = content

    async def pre_call(self, tool_call, tool_def, context):
        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            content=self._content,
        )


class ModifyingPostCallMiddleware(ToolMiddleware):
    """Appends a suffix to successful results."""

    def __init__(self, suffix: str = ":modified") -> None:
        self._suffix = suffix

    async def post_call(self, result, tool_call, tool_def, context):
        if result.success:
            return ToolResult(
                tool_call_id=result.tool_call_id,
                success=True,
                content=result.content + self._suffix,
            )
        return result


class ErrorRecoveryMiddleware(ToolMiddleware):
    """Returns a fallback ToolResult on error, preventing retries."""

    async def on_error(self, error, tool_call, tool_def, context, attempt, max_attempts):
        return ToolResult(
            tool_call_id=tool_call.id,
            success=True,
            content=f"recovered from: {error}",
        )


class ContextInspectingMiddleware(ToolMiddleware):
    """Captures the ToolContext for inspection."""

    def __init__(self) -> None:
        self.captured_context: Optional[ToolContext] = None

    async def pre_call(self, tool_call, tool_def, context):
        self.captured_context = context
        return tool_call


class MetadataSharingMiddleware(ToolMiddleware):
    """Writes to context.metadata in pre_call, reads in post_call."""

    def __init__(self) -> None:
        self.post_metadata: Optional[Dict] = None

    async def pre_call(self, tool_call, tool_def, context):
        context.metadata["started"] = True
        return tool_call

    async def post_call(self, result, tool_call, tool_def, context):
        self.post_metadata = dict(context.metadata)
        return result


# ===================================================================
# 1. Basic middleware invocation
# ===================================================================

class TestBasicMiddleware:

    @pytest.mark.asyncio
    async def test_global_middleware_pre_and_post_called(self):
        mw = RecordingMiddleware()
        tool = _simple_tool()
        executor = ToolExecutor(_make_registry(tool), middleware=[mw])
        result = await executor.execute(_call())
        assert result.success is True
        assert mw.pre_calls == ["my_tool"]
        assert mw.post_calls == ["my_tool"]
        assert mw.error_calls == []

    @pytest.mark.asyncio
    async def test_per_tool_middleware_called(self):
        mw = RecordingMiddleware()
        tool = _simple_tool(middleware=[mw])
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is True
        assert mw.pre_calls == ["my_tool"]
        assert mw.post_calls == ["my_tool"]

    @pytest.mark.asyncio
    async def test_no_middleware_still_works(self):
        """Backward compatibility: no middleware = same behavior as before."""
        tool = _simple_tool()
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is True
        assert "echo:" in result.content


# ===================================================================
# 2. Middleware ordering
# ===================================================================

class TestMiddlewareOrdering:

    @pytest.mark.asyncio
    async def test_global_before_per_tool_in_pre_call(self):
        order = []

        class GlobalMW(ToolMiddleware):
            async def pre_call(self, tc, td, ctx):
                order.append("global")
                return tc

        class PerToolMW(ToolMiddleware):
            async def pre_call(self, tc, td, ctx):
                order.append("per_tool")
                return tc

        tool = _simple_tool(middleware=[PerToolMW()])
        executor = ToolExecutor(_make_registry(tool), middleware=[GlobalMW()])
        await executor.execute(_call())
        assert order == ["global", "per_tool"]

    @pytest.mark.asyncio
    async def test_post_call_runs_in_reverse_order(self):
        order = []

        class MW_A(ToolMiddleware):
            async def post_call(self, r, tc, td, ctx):
                order.append("A")
                return r

        class MW_B(ToolMiddleware):
            async def post_call(self, r, tc, td, ctx):
                order.append("B")
                return r

        tool = _simple_tool()
        executor = ToolExecutor(
            _make_registry(tool),
            middleware=[MW_A(), MW_B()],
        )
        await executor.execute(_call())
        # Reversed: B runs first, then A
        assert order == ["B", "A"]

    @pytest.mark.asyncio
    async def test_multiple_global_middleware_chained(self):
        mw1 = ModifyingPostCallMiddleware(":first")
        mw2 = ModifyingPostCallMiddleware(":second")
        tool = _simple_tool()
        executor = ToolExecutor(_make_registry(tool), middleware=[mw1, mw2])
        result = await executor.execute(_call())
        # post_call is reversed: mw2 runs first, then mw1
        assert result.content.endswith(":second:first")


# ===================================================================
# 3. Pre-call: argument modification
# ===================================================================

class TestPreCallModification:

    @pytest.mark.asyncio
    async def test_pre_call_can_modify_arguments(self):
        received_args = {}

        async def capture_handler(**kwargs):
            received_args.update(kwargs)
            return "ok"

        mw = ModifyingPreCallMiddleware()
        tool = _simple_tool(handler=capture_handler, middleware=[mw])
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call(query="test"))
        assert result.success is True
        assert received_args.get("injected") is True
        assert received_args.get("query") == "test"


# ===================================================================
# 4. Pre-call: short-circuit
# ===================================================================

class TestPreCallShortCircuit:

    @pytest.mark.asyncio
    async def test_short_circuit_skips_handler(self):
        handler_called = False

        async def handler(**kw):
            nonlocal handler_called
            handler_called = True
            return "from handler"

        mw = ShortCircuitMiddleware(content="from cache")
        tool = _simple_tool(handler=handler, middleware=[mw])
        executor = ToolExecutor(_make_registry(tool))
        result = await executor.execute(_call())
        assert result.success is True
        assert result.content == "from cache"
        assert handler_called is False

    @pytest.mark.asyncio
    async def test_short_circuit_still_runs_post_call(self):
        post_mw = RecordingMiddleware()
        cache_mw = ShortCircuitMiddleware(content="cached")
        tool = _simple_tool(middleware=[cache_mw])
        executor = ToolExecutor(_make_registry(tool), middleware=[post_mw])
        result = await executor.execute(_call())
        assert result.success is True
        assert result.content == "cached"
        # post_call should still be invoked
        assert post_mw.post_calls == ["my_tool"]


# ===================================================================
# 5. Post-call: result modification
# ===================================================================

class TestPostCallModification:

    @pytest.mark.asyncio
    async def test_post_call_can_modify_result(self):
        mw = ModifyingPostCallMiddleware(":appended")
        tool = _simple_tool()
        executor = ToolExecutor(_make_registry(tool), middleware=[mw])
        result = await executor.execute(_call())
        assert result.success is True
        assert result.content.endswith(":appended")

    @pytest.mark.asyncio
    async def test_post_call_receives_original_tool_call(self):
        captured_tc = None

        class CaptureMW(ToolMiddleware):
            async def post_call(self, r, tc, td, ctx):
                nonlocal captured_tc
                captured_tc = tc
                return r

        tool = _simple_tool()
        executor = ToolExecutor(_make_registry(tool), middleware=[CaptureMW()])
        await executor.execute(_call(name="my_tool", call_id="xyz"))
        assert captured_tc is not None
        assert captured_tc.id == "xyz"


# ===================================================================
# 6. On-error hook
# ===================================================================

class TestOnError:

    @pytest.mark.asyncio
    async def test_on_error_called_on_handler_exception(self):
        mw = RecordingMiddleware()

        async def failing_handler(**kw):
            raise RuntimeError("boom")

        tool = _simple_tool(handler=failing_handler)
        executor = ToolExecutor(_make_registry(tool), middleware=[mw])
        result = await executor.execute(_call())
        assert result.success is False
        assert mw.error_calls == ["my_tool:attempt=0"]

    @pytest.mark.asyncio
    async def test_on_error_can_recover(self):
        mw = ErrorRecoveryMiddleware()

        async def failing_handler(**kw):
            raise RuntimeError("crash")

        tool = _simple_tool(handler=failing_handler)
        executor = ToolExecutor(_make_registry(tool), middleware=[mw])
        result = await executor.execute(_call())
        assert result.success is True
        assert "recovered from" in result.content

    @pytest.mark.asyncio
    async def test_on_error_recovery_skips_retries(self):
        attempts = 0

        async def failing_handler(**kw):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("fail")

        mw = ErrorRecoveryMiddleware()
        tool = _simple_tool(
            handler=failing_handler,
            retry_on_failure=True,
            max_retries=3,
        )
        executor = ToolExecutor(_make_registry(tool), middleware=[mw])

        with patch("parr.tool_executor.asyncio.sleep", return_value=None):
            result = await executor.execute(_call())

        assert result.success is True
        assert attempts == 1  # Only first attempt, recovery stopped retries

    @pytest.mark.asyncio
    async def test_on_error_called_per_retry_attempt(self):
        mw = RecordingMiddleware()

        async def always_fail(**kw):
            raise RuntimeError("fail")

        tool = _simple_tool(
            handler=always_fail,
            retry_on_failure=True,
            max_retries=2,
        )
        executor = ToolExecutor(_make_registry(tool), middleware=[mw])

        with patch("parr.tool_executor.asyncio.sleep", return_value=None):
            result = await executor.execute(_call())

        assert result.success is False
        assert len(mw.error_calls) == 3  # attempt 0, 1, 2

    @pytest.mark.asyncio
    async def test_on_error_called_on_timeout(self):
        mw = RecordingMiddleware()

        async def slow(**kw):
            await asyncio.sleep(10)
            return "done"

        tool = _simple_tool(handler=slow, timeout_ms=50)
        executor = ToolExecutor(_make_registry(tool), middleware=[mw])
        result = await executor.execute(_call())
        assert result.success is False
        assert len(mw.error_calls) == 1


# ===================================================================
# 7. ToolContext
# ===================================================================

class TestToolContext:

    @pytest.mark.asyncio
    async def test_context_has_phase(self):
        mw = ContextInspectingMiddleware()
        tool = _simple_tool()
        executor = ToolExecutor(_make_registry(tool), middleware=[mw])
        executor.set_phase(Phase.ACT)
        await executor.execute(_call())
        assert mw.captured_context is not None
        assert mw.captured_context.phase == Phase.ACT

    @pytest.mark.asyncio
    async def test_context_has_agent_and_task_id(self):
        mw = ContextInspectingMiddleware()
        tool = _simple_tool()
        executor = ToolExecutor(
            _make_registry(tool),
            middleware=[mw],
            agent_id="agent-123",
            task_id="task-456",
        )
        await executor.execute(_call())
        assert mw.captured_context.agent_id == "agent-123"
        assert mw.captured_context.task_id == "task-456"

    @pytest.mark.asyncio
    async def test_context_has_tool_name(self):
        mw = ContextInspectingMiddleware()
        tool = _simple_tool(name="search")
        executor = ToolExecutor(_make_registry(tool), middleware=[mw])
        await executor.execute(_call(name="search"))
        assert mw.captured_context.tool_name == "search"

    @pytest.mark.asyncio
    async def test_context_metadata_shared_between_hooks(self):
        mw = MetadataSharingMiddleware()
        tool = _simple_tool()
        executor = ToolExecutor(_make_registry(tool), middleware=[mw])
        await executor.execute(_call())
        assert mw.post_metadata is not None
        assert mw.post_metadata.get("started") is True

    @pytest.mark.asyncio
    async def test_context_call_count(self):
        mw = ContextInspectingMiddleware()
        tool = _simple_tool(name="search")
        executor = ToolExecutor(_make_registry(tool), middleware=[mw])
        executor.set_phase(Phase.ACT)

        await executor.execute(_call(name="search", call_id="c1"))
        assert mw.captured_context.call_count == 1

        await executor.execute(_call(name="search", call_id="c2"))
        assert mw.captured_context.call_count == 2


# ===================================================================
# 8. Middleware error handling
# ===================================================================

class TestMiddlewareErrors:

    @pytest.mark.asyncio
    async def test_pre_call_exception_returns_error_result(self):
        class BrokenMW(ToolMiddleware):
            async def pre_call(self, tc, td, ctx):
                raise ValueError("middleware bug")

        tool = _simple_tool()
        executor = ToolExecutor(_make_registry(tool), middleware=[BrokenMW()])
        result = await executor.execute(_call())
        assert result.success is False
        assert "Middleware pre_call error" in result.error

    @pytest.mark.asyncio
    async def test_post_call_exception_returns_original_result(self):
        class BrokenPostMW(ToolMiddleware):
            async def post_call(self, r, tc, td, ctx):
                raise RuntimeError("post_call bug")

        tool = _simple_tool()
        executor = ToolExecutor(_make_registry(tool), middleware=[BrokenPostMW()])
        result = await executor.execute(_call())
        # Should still return the original successful result
        assert result.success is True

    @pytest.mark.asyncio
    async def test_on_error_exception_does_not_crash(self):
        class BrokenOnErrorMW(ToolMiddleware):
            async def on_error(self, err, tc, td, ctx, attempt, max_attempts):
                raise RuntimeError("on_error itself failed")

        async def failing_handler(**kw):
            raise RuntimeError("handler fail")

        tool = _simple_tool(handler=failing_handler)
        executor = ToolExecutor(_make_registry(tool), middleware=[BrokenOnErrorMW()])
        result = await executor.execute(_call())
        # Should still return the handler error, not crash
        assert result.success is False
        assert "handler fail" in result.error


# ===================================================================
# 9. Global middleware management
# ===================================================================

class TestMiddlewareManagement:

    def test_add_middleware(self):
        executor = ToolExecutor(ToolRegistry())
        mw = RecordingMiddleware()
        executor.add_middleware(mw)
        assert mw in executor.middleware

    def test_remove_middleware(self):
        mw = RecordingMiddleware()
        executor = ToolExecutor(ToolRegistry(), middleware=[mw])
        executor.remove_middleware(mw)
        assert mw not in executor.middleware

    def test_middleware_property_returns_copy(self):
        mw = RecordingMiddleware()
        executor = ToolExecutor(ToolRegistry(), middleware=[mw])
        mw_list = executor.middleware
        mw_list.clear()
        # Original should be unaffected
        assert len(executor.middleware) == 1


# ===================================================================
# 10. ToolRegistry.override()
# ===================================================================

class TestRegistryOverride:

    def test_override_existing_tool(self):
        registry = ToolRegistry()
        original = _simple_tool(name="search")
        registry.register(original)

        replacement = _simple_tool(name="search", handler=lambda **kw: "new")
        previous = registry.override(replacement)

        assert previous is original
        assert registry.get("search") is replacement

    def test_override_nonexistent_registers_new(self):
        registry = ToolRegistry()
        tool = _simple_tool(name="brand_new")
        previous = registry.override(tool)

        assert previous is None
        assert registry.get("brand_new") is tool

    @pytest.mark.asyncio
    async def test_overridden_tool_handler_is_used(self):
        registry = ToolRegistry()

        async def original_handler(**kw):
            return "original"

        async def custom_handler(**kw):
            return "custom"

        original = _simple_tool(name="search", handler=original_handler)
        registry.register(original)

        custom = _simple_tool(name="search", handler=custom_handler)
        registry.override(custom)

        executor = ToolExecutor(registry)
        result = await executor.execute(_call(name="search"))
        assert result.success is True
        assert result.content == "custom"

    def test_override_preserves_other_tools(self):
        registry = ToolRegistry()
        registry.register(_simple_tool(name="a"))
        registry.register(_simple_tool(name="b"))
        registry.override(_simple_tool(name="a"))
        assert registry.has_tool("a")
        assert registry.has_tool("b")
        assert len(registry.get_all()) == 2

    @pytest.mark.asyncio
    async def test_override_framework_tool(self):
        """Simulates overriding a built-in framework tool."""
        registry = ToolRegistry()

        async def default_handler(**kw):
            return "default behavior"

        async def custom_handler(**kw):
            return "custom behavior"

        fw_tool = _simple_tool(
            name="create_todo_list",
            handler=default_handler,
            is_framework_tool=True,
        )
        registry.register(fw_tool)

        custom_tool = _simple_tool(
            name="create_todo_list",
            handler=custom_handler,
            is_framework_tool=True,
        )
        previous = registry.override(custom_tool)

        assert previous is fw_tool
        executor = ToolExecutor(registry)
        result = await executor.execute(_call(name="create_todo_list"))
        assert result.content == "custom behavior"


# ===================================================================
# 11. Integration: middleware + retry + override together
# ===================================================================

class TestIntegration:

    @pytest.mark.asyncio
    async def test_middleware_with_retry_and_eventual_success(self):
        """Global middleware sees all retry attempts; handler eventually succeeds."""
        mw = RecordingMiddleware()
        attempts = 0

        async def flaky(**kw):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("transient")
            return "ok"

        tool = _simple_tool(
            handler=flaky,
            retry_on_failure=True,
            max_retries=2,
            timeout_ms=5000,
        )
        executor = ToolExecutor(_make_registry(tool), middleware=[mw])

        with patch("parr.tool_executor.asyncio.sleep", return_value=None):
            result = await executor.execute(_call())

        assert result.success is True
        assert result.content == "ok"
        # pre_call called once (before first attempt)
        assert len(mw.pre_calls) == 1
        # on_error called for each failed attempt
        assert len(mw.error_calls) == 2
        # post_call called once (after final success)
        assert len(mw.post_calls) == 1

    @pytest.mark.asyncio
    async def test_global_and_per_tool_middleware_combined(self):
        global_mw = ModifyingPostCallMiddleware(":global")
        per_tool_mw = ModifyingPostCallMiddleware(":per_tool")
        tool = _simple_tool(middleware=[per_tool_mw])
        executor = ToolExecutor(_make_registry(tool), middleware=[global_mw])
        result = await executor.execute(_call())
        # post_call reversed: per_tool_mw first, then global_mw
        assert result.content.endswith(":per_tool:global")
