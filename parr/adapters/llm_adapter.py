"""
ToolCallingLLM Adapters for the Agentic Framework.

Bridges existing LLM SDK clients (OpenAI, Azure OpenAI, Anthropic) to the
framework's ToolCallingLLM protocol by adding tool/function calling support
and translating between provider-specific formats and the framework's
Message/ToolCall/LLMResponse types.

All adapters use async clients to avoid blocking the event loop during
multi-agent workflows.

Usage:
    # OpenAI / Azure OpenAI (async)
    from openai import AsyncAzureOpenAI
    client = AsyncAzureOpenAI(azure_endpoint=..., api_key=..., api_version=...)
    llm = OpenAIToolCallingLLM(client=client, model="gpt-4o")

    # Anthropic (async)
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=...)
    llm = AnthropicToolCallingLLM(client=client, model="claude-3-5-sonnet-20241022")

    # Factory (creates async clients automatically)
    llm = create_tool_calling_llm("azure_openai", model="gpt-4o", endpoint=..., api_key=...)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ..budget_tracker import BudgetExceededException
from ..core_types import (
    CostConfig,
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
    TokenUsage,
    ToolCall,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retry configuration for transient API errors
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_RETRY_BACKOFF_SECONDS = [1.0, 2.0, 4.0, 8.0, 16.0]
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class ContentFilterError(RuntimeError):
    """Raised when an API call is blocked by the provider's content filter.

    This is NOT retryable by default because retrying with the same prompt
    will produce the same rejection. The caller should either soften the
    prompt or gracefully degrade.
    """

    def __init__(self, message: str, filter_details: Optional[Dict] = None):
        super().__init__(message)
        self.filter_details = filter_details or {}


def _is_content_filter_error(error: Exception) -> bool:
    """Detect whether an error is a content filter / responsible AI block.

    Checks structured error attributes first (reliable), then falls back
    to string matching for providers that don't expose structured codes.
    """
    # --- Structured attribute checks (most reliable) ---

    # OpenAI / Azure SDK: error.code or error.error.code
    error_code = getattr(error, "code", None) or ""
    if isinstance(error_code, str) and error_code.lower() in (
        "content_filter", "content_management_policy",
        "responsibleaipolicyviolation",
    ):
        return True

    # Some SDK errors nest details under an `error` dict/object
    inner = getattr(error, "error", None)
    if isinstance(inner, dict):
        inner_code = str(inner.get("code", "")).lower()
        inner_type = str(inner.get("type", "")).lower()
        if inner_code in ("content_filter", "responsibleaipolicyviolation"):
            return True
        if "content_filter" in inner_type:
            return True

    # Anthropic: error.type
    error_type = getattr(error, "type", None)
    if isinstance(error_type, str) and "content" in error_type.lower():
        return True

    # --- String-based fallback (less reliable, but catches edge cases) ---
    error_str = str(error).lower()
    if "content_filter" in error_str or "content management policy" in error_str:
        return True
    if "responsibleaipolicyviolation" in error_str:
        return True
    if "content policy" in error_str and "filtered" in error_str:
        return True
    return False


def _extract_filter_details(error: Exception) -> Dict:
    """Try to extract content filter result details from the error."""
    error_str = str(error)
    try:
        # Azure OpenAI errors often contain a JSON body
        import re
        match = re.search(r"content_filter_result['\"]:\s*(\{[^}]+\})", error_str)
        if match:
            return json.loads(match.group(1).replace("'", '"'))
    except Exception:
        pass
    return {}


def _is_retryable_error(error: Exception) -> bool:
    """Determine if an error is transient and worth retrying.

    Uses structured attributes first, falls back to string matching.
    SDK type checks are wrapped defensively so missing/changed SDK
    classes don't cause secondary failures.
    """
    # Content filter errors are NOT retryable with the same prompt
    if _is_content_filter_error(error):
        return False

    # Connection-level errors (DNS, timeout, reset, etc.)
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True

    # httpx-specific errors (may not be installed as a direct dependency,
    # but the OpenAI/Anthropic SDKs use httpx internally)
    try:
        import httpx
        if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout,
                              httpx.ReadTimeout, httpx.WriteTimeout)):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in _RETRYABLE_STATUS_CODES
    except (ImportError, AttributeError):
        pass

    # OpenAI SDK: APIStatusError.status_code
    # Anthropic SDK: APIStatusError.status_code
    # Check via getattr so we don't depend on SDK class hierarchy.
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        try:
            if int(status_code) in _RETRYABLE_STATUS_CODES:
                return True
        except (ValueError, TypeError):
            pass

    # OpenAI SDK v1+: error.code may be a string like "server_error"
    error_code = getattr(error, "code", None)
    if isinstance(error_code, str) and error_code.lower() in (
        "server_error", "rate_limit_exceeded", "service_unavailable",
    ):
        return True

    # Fallback: match common transient error string patterns
    error_str = str(error).lower()
    if any(pattern in error_str for pattern in [
        "connection", "timeout", "getaddrinfo", "name resolution",
        "temporary failure", "service unavailable", "rate limit",
    ]):
        return True

    return False


_CONTENT_FILTER_RETRIES = 1  # One retry for non-deterministic filter triggers
_CONTENT_FILTER_RETRY_DELAY = 2.0  # seconds


async def _with_retry(
    coro_factory: Callable[[], Any],
    provider_name: str,
) -> Any:
    """
    Execute an async operation with retry on transient failures.

    Args:
        coro_factory: A zero-argument callable that returns a new coroutine
                      each time (must be fresh per retry, not a single awaitable).
        provider_name: "OpenAI" or "Anthropic" for logging.

    Returns:
        The result of the successful call.

    Raises:
        RuntimeError wrapping the final exception after all retries exhausted.
    """
    last_error: Optional[Exception] = None
    _content_filter_attempts = 0
    for attempt in range(_MAX_RETRIES + 1):  # attempt 0 = first try
        try:
            return await coro_factory()
        except Exception as e:
            last_error = e
            # Content filter errors: retry once (filter can be non-deterministic,
            # especially Azure's jailbreak detector with agentic prompts).
            if _is_content_filter_error(e):
                _content_filter_attempts += 1
                if _content_filter_attempts <= _CONTENT_FILTER_RETRIES:
                    logger.warning(
                        f"{provider_name} content filter triggered (attempt "
                        f"{_content_filter_attempts}/{_CONTENT_FILTER_RETRIES + 1}), "
                        f"retrying in {_CONTENT_FILTER_RETRY_DELAY}s: {e}"
                    )
                    await asyncio.sleep(_CONTENT_FILTER_RETRY_DELAY)
                    continue
                details = _extract_filter_details(e)
                logger.warning(
                    f"{provider_name} content filter blocked the request: {e}"
                )
                raise ContentFilterError(
                    f"{provider_name} content filter blocked the request: {e}",
                    filter_details=details,
                ) from e

            # Preserve framework-internal exceptions — they should not
            # be wrapped in RuntimeError so that upstream handlers
            # (e.g. agent_runtime) can catch them by type.
            if isinstance(e, BudgetExceededException):
                raise

            if attempt < _MAX_RETRIES and _is_retryable_error(e):
                delay = _RETRY_BACKOFF_SECONDS[attempt]
                logger.warning(
                    f"{provider_name} API transient error (attempt {attempt + 1}/"
                    f"{_MAX_RETRIES + 1}), retrying in {delay}s: "
                    f"{type(e).__name__}: {e}"
                )
                await asyncio.sleep(delay)
            else:
                raise RuntimeError(
                    f"{provider_name} API call failed: {e}"
                ) from e

    # Should not reach here, but safety net
    raise RuntimeError(
        f"{provider_name} API call failed after {_MAX_RETRIES + 1} attempts: "
        f"{last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# OpenAI / Azure OpenAI adapter
# ---------------------------------------------------------------------------

class OpenAIToolCallingLLM:
    """
    ToolCallingLLM adapter for OpenAI-compatible APIs (OpenAI, Azure OpenAI).

    Wraps an ``openai.AsyncOpenAI`` or ``openai.AsyncAzureOpenAI`` client
    instance and translates between the framework's Message/ToolCall types
    and the OpenAI chat completions API with function calling.

    Uses async clients to avoid blocking the event loop during multi-agent
    workflows.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        cost_config: Optional[CostConfig] = None,
    ) -> None:
        """
        Args:
            client: An ``openai.AsyncOpenAI`` or ``openai.AsyncAzureOpenAI`` instance.
            model: Model name or Azure deployment name.
            cost_config: Optional pricing config for cost tracking.
        """
        self._client = client
        self._model = model
        self._cost_config = cost_config

    # -- public API (ToolCallingLLM protocol) --------------------------------

    async def chat_with_tools(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        model: str,
        model_config: ModelConfig,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        """Send messages with tool definitions to an OpenAI-compatible API."""
        api_messages = self._translate_messages(messages)
        api_tools = self._translate_tools(tools) if tools else None

        api_params: Dict[str, Any] = {
            "model": model or self._model,
            "messages": api_messages,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
        }

        # Newer models use max_completion_tokens instead of max_tokens
        if self._uses_max_completion_tokens(model or self._model):
            api_params["max_completion_tokens"] = model_config.max_tokens
        else:
            api_params["max_tokens"] = model_config.max_tokens

        if api_tools:
            api_params["tools"] = api_tools

        if stream:
            return await self._stream_response(api_params, on_token)

        start = time.time()
        try:
            response = await _with_retry(
                lambda: self._client.chat.completions.create(**api_params),
                "OpenAI",
            )
            latency_ms = (time.time() - start) * 1000
            return self._parse_response(response, latency_ms)
        except RuntimeError:
            raise  # Already wrapped by _with_retry
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.error(f"OpenAI API error after {latency_ms:.0f}ms: {e}")
            raise RuntimeError(f"OpenAI API call failed: {e}") from e

    # -- message translation -------------------------------------------------

    def _translate_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert framework Messages to OpenAI API format."""
        api_messages: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                api_messages.append({"role": "system", "content": msg.content or ""})

            elif msg.role == MessageRole.USER:
                api_messages.append({"role": "user", "content": msg.content or ""})

            elif msg.role == MessageRole.ASSISTANT:
                entry: Dict[str, Any] = {"role": "assistant"}
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments)
                                if isinstance(tc.arguments, dict)
                                else str(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                    # OpenAI requires content or null for assistant messages with tool_calls
                    entry["content"] = msg.content if msg.content else None
                else:
                    entry["content"] = msg.content or ""
                api_messages.append(entry)

            elif msg.role == MessageRole.TOOL:
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id or "",
                    "content": msg.content or "",
                })

        return api_messages

    def _translate_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert framework tool schemas to OpenAI function-calling format."""
        result = []
        for tool in tools:
            params = tool.get("parameters", {"type": "object", "properties": {}})
            # Azure OpenAI requires "properties" on every object-type schema.
            if isinstance(params, dict) and params.get("type") == "object" and "properties" not in params:
                params = {**params, "properties": {}}
            result.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": params,
                },
            })
        return result

    # -- response parsing ----------------------------------------------------

    def _parse_response(self, response: Any, latency_ms: float) -> LLMResponse:
        """Parse an OpenAI ChatCompletion response into framework LLMResponse."""
        choice = response.choices[0]
        message = choice.message

        # Extract text content
        content = message.content

        # Extract tool calls
        tool_calls: Optional[List[ToolCall]] = None
        if message.tool_calls:
            tool_calls = []
            for tc in message.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                ))

        # Token usage
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        if self._cost_config:
            usage.total_cost = self._cost_config.calculate_cost(
                self._model, usage
            )

        # Build raw_message for conversation history
        raw_message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            raw_message=raw_message,
        )

    # -- streaming -----------------------------------------------------------

    async def _stream_response(
        self,
        api_params: Dict[str, Any],
        on_token: Optional[Callable],
    ) -> LLMResponse:
        """Handle streaming responses from OpenAI API (async).

        The entire stream creation + chunk iteration is wrapped in an
        _attempt closure so that _with_retry can recover from transient
        failures that occur mid-stream (matching the Anthropic adapter).
        """
        api_params["stream"] = True
        api_params["stream_options"] = {"include_usage": True}

        start = time.time()

        async def _attempt() -> LLMResponse:
            content_parts: List[str] = []
            tool_call_accum: Dict[int, Dict[str, str]] = {}
            usage_data: Optional[Any] = None

            response_stream = await self._client.chat.completions.create(
                **api_params
            )

            async for chunk in response_stream:
                if not chunk.choices and hasattr(chunk, "usage") and chunk.usage:
                    usage_data = chunk.usage
                    continue

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                # Text content
                if delta.content:
                    content_parts.append(delta.content)
                    if on_token:
                        await on_token(delta.content)

                # Tool call deltas
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_call_accum:
                            tool_call_accum[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc_delta.id:
                            tool_call_accum[idx]["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_call_accum[idx]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_call_accum[idx]["arguments"] += tc_delta.function.arguments

            # Build result
            content = "".join(content_parts) or None

            tool_calls: Optional[List[ToolCall]] = None
            if tool_call_accum:
                tool_calls = []
                for _idx in sorted(tool_call_accum.keys()):
                    tc_data = tool_call_accum[_idx]
                    try:
                        arguments = json.loads(tc_data["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"raw": tc_data["arguments"]}
                    tool_calls.append(ToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=arguments,
                    ))

            usage = TokenUsage()
            if usage_data:
                usage.input_tokens = getattr(usage_data, "prompt_tokens", 0)
                usage.output_tokens = getattr(usage_data, "completion_tokens", 0)
                if self._cost_config:
                    usage.total_cost = self._cost_config.calculate_cost(
                        self._model, usage
                    )

            raw_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls,
            )

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                usage=usage,
                raw_message=raw_message,
            )

        try:
            return await _with_retry(_attempt, "OpenAI")
        except RuntimeError:
            raise
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.error(f"OpenAI streaming error after {latency_ms:.0f}ms: {e}")
            raise RuntimeError(f"OpenAI streaming call failed: {e}") from e

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _uses_max_completion_tokens(model: str) -> bool:
        """Check if model requires max_completion_tokens instead of max_tokens."""
        model_lower = model.lower()
        for pattern in ("o1", "o3", "o4", "gpt-5", "gpt5"):
            if pattern in model_lower:
                return True
        return False


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class AnthropicToolCallingLLM:
    """
    ToolCallingLLM adapter for the Anthropic Claude API.

    Wraps an ``anthropic.AsyncAnthropic`` client instance and translates
    between the framework's Message/ToolCall types and the Anthropic messages
    API with tool use.

    Uses async client to avoid blocking the event loop during multi-agent
    workflows.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        cost_config: Optional[CostConfig] = None,
    ) -> None:
        """
        Args:
            client: An ``anthropic.AsyncAnthropic`` instance.
            model: Model identifier (e.g. "claude-3-5-sonnet-20241022").
            cost_config: Optional pricing config for cost tracking.
        """
        self._client = client
        self._model = model
        self._cost_config = cost_config

    # -- public API (ToolCallingLLM protocol) --------------------------------

    async def chat_with_tools(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        model: str,
        model_config: ModelConfig,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        """Send messages with tool definitions to the Anthropic API."""
        system_prompt, api_messages = self._translate_messages(messages)
        api_tools = self._translate_tools(tools) if tools else []

        create_kwargs: Dict[str, Any] = {
            "model": model or self._model,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
            "messages": api_messages,
        }

        if system_prompt:
            create_kwargs["system"] = system_prompt
        if api_tools:
            create_kwargs["tools"] = api_tools

        if stream:
            return await self._stream_response(create_kwargs, on_token)

        start = time.time()
        try:
            response = await _with_retry(
                lambda: self._client.messages.create(**create_kwargs),
                "Anthropic",
            )
            latency_ms = (time.time() - start) * 1000
            return self._parse_response(response, latency_ms)
        except RuntimeError:
            raise  # Already wrapped by _with_retry
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.error(f"Anthropic API error after {latency_ms:.0f}ms: {e}")
            raise RuntimeError(f"Anthropic API call failed: {e}") from e

    # -- message translation -------------------------------------------------

    def _translate_messages(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert framework Messages to Anthropic API format.

        Returns (system_prompt, api_messages). System messages are extracted
        into the separate ``system`` parameter required by Anthropic.
        """
        system_parts: List[str] = []
        api_messages: List[Dict[str, Any]] = []

        # Collect tool results that need to be grouped into a user message
        pending_tool_results: List[Dict[str, Any]] = []

        def _flush_tool_results() -> None:
            """Emit accumulated tool_result blocks as a user message."""
            if pending_tool_results:
                api_messages.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                if msg.content:
                    system_parts.append(msg.content)

            elif msg.role == MessageRole.USER:
                content: Any = msg.content or ""
                # Merge pending tool results with this user message
                if pending_tool_results:
                    blocks = list(pending_tool_results)
                    pending_tool_results.clear()
                    if content:
                        blocks.append({"type": "text", "text": content})
                    api_messages.append({"role": "user", "content": blocks})
                else:
                    api_messages.append({"role": "user", "content": content})

            elif msg.role == MessageRole.ASSISTANT:
                # Flush any pending tool results before an assistant message
                _flush_tool_results()

                if msg.tool_calls:
                    content_blocks: List[Dict[str, Any]] = []
                    if msg.content:
                        content_blocks.append({"type": "text", "text": msg.content})
                    for tc in msg.tool_calls:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                    api_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    api_messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                    })

            elif msg.role == MessageRole.TOOL:
                # Anthropic expects tool_result inside user messages
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content or "",
                })

        # Flush any remaining tool results
        _flush_tool_results()

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, api_messages

    def _translate_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert framework tool schemas to Anthropic tool format."""
        result = []
        for tool in tools:
            schema = tool.get("parameters", {"type": "object", "properties": {}})
            if isinstance(schema, dict) and schema.get("type") == "object" and "properties" not in schema:
                schema = {**schema, "properties": {}}
            result.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": schema,
            })
        return result

    # -- response parsing ----------------------------------------------------

    def _parse_response(self, response: Any, latency_ms: float) -> LLMResponse:
        """Parse an Anthropic messages response into framework LLMResponse."""
        content_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        for block in response.content:
            if hasattr(block, "text"):
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        content = "\n".join(content_parts) if content_parts else None

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        if self._cost_config:
            usage.total_cost = self._cost_config.calculate_cost(
                self._model, usage
            )

        raw_message = Message(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls if tool_calls else None,
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            raw_message=raw_message,
        )

    # -- streaming -----------------------------------------------------------

    async def _stream_response(
        self,
        create_kwargs: Dict[str, Any],
        on_token: Optional[Callable],
    ) -> LLMResponse:
        """Handle streaming responses from Anthropic API (async)."""
        start = time.time()

        async def _attempt() -> LLMResponse:
            content_parts: List[str] = []
            tool_calls_acc: List[ToolCall] = []
            current_tool: Optional[Dict[str, Any]] = None
            input_tokens = 0
            output_tokens = 0

            async with self._client.messages.stream(**create_kwargs) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", None)

                    if event_type == "message_start":
                        if hasattr(event, "message") and hasattr(event.message, "usage"):
                            input_tokens = event.message.usage.input_tokens

                    elif event_type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool = {
                                "id": block.id,
                                "name": block.name,
                                "input_json": "",
                            }
                        elif block.type == "text":
                            pass  # Text deltas come in content_block_delta

                    elif event_type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "text"):
                            content_parts.append(delta.text)
                            if on_token:
                                await on_token(delta.text)
                        elif hasattr(delta, "partial_json"):
                            if current_tool is not None:
                                current_tool["input_json"] += delta.partial_json

                    elif event_type == "content_block_stop":
                        if current_tool is not None:
                            try:
                                arguments = json.loads(current_tool["input_json"])
                            except (json.JSONDecodeError, TypeError):
                                arguments = {}
                            tool_calls_acc.append(ToolCall(
                                id=current_tool["id"],
                                name=current_tool["name"],
                                arguments=arguments,
                            ))
                            current_tool = None

                    elif event_type == "message_delta":
                        if hasattr(event, "usage"):
                            output_tokens = event.usage.output_tokens

            content = "".join(content_parts) if content_parts else None
            usage = TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if self._cost_config:
                usage.total_cost = self._cost_config.calculate_cost(
                    self._model, usage
                )

            raw_message = Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls_acc if tool_calls_acc else None,
            )

            return LLMResponse(
                content=content,
                tool_calls=tool_calls_acc if tool_calls_acc else None,
                usage=usage,
                raw_message=raw_message,
            )

        try:
            return await _with_retry(_attempt, "Anthropic")
        except RuntimeError:
            raise
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.error(f"Anthropic streaming error after {latency_ms:.0f}ms: {e}")
            raise RuntimeError(f"Anthropic streaming call failed: {e}") from e


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tool_calling_llm(
    provider_type: str,
    model: str,
    cost_config: Optional[CostConfig] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a ToolCallingLLM adapter for the specified provider.

    Creates async clients to avoid blocking the event loop.

    Args:
        provider_type: One of "openai", "azure_openai", "anthropic".
        model: Model name or deployment name.
        cost_config: Optional pricing config.
        **kwargs: Provider-specific arguments:
            - openai: api_key, timeout
            - azure_openai: endpoint, api_key, api_version, timeout
            - anthropic: api_key, timeout

    Returns:
        A ToolCallingLLM-compatible adapter instance.
    """
    if provider_type in ("openai", "azure_openai"):
        if provider_type == "azure_openai":
            from openai import AsyncAzureOpenAI

            client = AsyncAzureOpenAI(
                azure_endpoint=kwargs.get("endpoint"),
                api_key=kwargs.get("api_key"),
                api_version=kwargs.get("api_version", "2024-02-15-preview"),
                timeout=kwargs.get("timeout", 120.0),
            )
        else:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=kwargs.get("api_key"),
                timeout=kwargs.get("timeout", 120.0),
            )

        return OpenAIToolCallingLLM(
            client=client,
            model=model,
            cost_config=cost_config,
        )

    elif provider_type == "anthropic":
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(
            api_key=kwargs.get("api_key"),
            timeout=kwargs.get("timeout", 120.0),
        )

        return AnthropicToolCallingLLM(
            client=client,
            model=model,
            cost_config=cost_config,
        )

    else:
        raise ValueError(
            f"Unsupported provider_type: {provider_type}. "
            f"Supported: 'openai', 'azure_openai', 'anthropic'"
        )
