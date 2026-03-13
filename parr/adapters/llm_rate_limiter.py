"""
Async global queue + rate limiter for ToolCallingLLM adapters.

This wrapper enforces FIFO admission and optional limits for:
- max concurrent in-flight LLM requests
- max requests per rolling time window

Waiting is async (`await`) and does not block the event loop thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Deque, Dict, List, Optional

from ..core_types import LLMRateLimitConfig, LLMResponse, Message, ModelConfig
from ..protocols import ToolCallingLLM

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMQueuePermit:
    """Permit metadata for a queued LLM request."""
    request_id: str
    queued_at: float
    granted_at: float
    queue_depth_at_enqueue: int

    @property
    def wait_seconds(self) -> float:
        return max(0.0, self.granted_at - self.queued_at)


@dataclass(frozen=True)
class _QueueEntry:
    request_id: str
    enqueued_at: float
    queue_depth_at_enqueue: int


class LLMCallQueue:
    """
    Fair FIFO scheduler for LLM calls.

    Guarantees that requests are admitted in enqueue order once both
    concurrency and rolling-window rate constraints are satisfied.
    """

    def __init__(
        self,
        config: LLMRateLimitConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._max_concurrent = config.max_concurrent_requests
        self._max_requests = config.max_requests_per_window
        self._max_tokens = config.max_tokens_per_window
        self._window_seconds = config.window_seconds
        self._max_queue_size = config.max_queue_size
        self._acquire_timeout_seconds = config.acquire_timeout_seconds

        self._condition = asyncio.Condition()
        self._wait_queue: Deque[_QueueEntry] = deque()
        self._request_timestamps: Deque[float] = deque()
        self._token_timestamps: Deque[tuple[float, int]] = deque()
        self._in_flight = 0

    @asynccontextmanager
    async def reserve(self, token_reservation: int = 0) -> AsyncIterator[LLMQueuePermit]:
        """Acquire a queue permit and release it automatically."""
        permit = await self.acquire(token_reservation=token_reservation)
        try:
            yield permit
        finally:
            await self.release()

    async def acquire(self, token_reservation: int = 0) -> LLMQueuePermit:
        """Wait until this request is admitted by queue/rate constraints."""
        token_reservation = max(0, token_reservation)
        now = self._clock()
        request_id = uuid.uuid4().hex
        deadline = None
        if self._acquire_timeout_seconds is not None:
            deadline = now + self._acquire_timeout_seconds

        async with self._condition:
            if self._max_queue_size is not None and len(self._wait_queue) >= self._max_queue_size:
                raise RuntimeError(
                    "LLM queue is full "
                    f"(max_queue_size={self._max_queue_size}); request rejected."
                )
            if self._max_tokens is not None and token_reservation > self._max_tokens:
                raise RuntimeError(
                    "LLM request token reservation exceeds queue token window limit "
                    f"({token_reservation} > {self._max_tokens})."
                )

            entry = _QueueEntry(
                request_id=request_id,
                enqueued_at=now,
                queue_depth_at_enqueue=len(self._wait_queue) + 1,
            )
            self._wait_queue.append(entry)
            self._condition.notify_all()

            try:
                while True:
                    now = self._clock()
                    self._prune_rate_window(now)
                    self._prune_token_window(now)
                    is_head = bool(self._wait_queue) and self._wait_queue[0] == entry

                    if is_head and self._has_capacity(now, token_reservation):
                        self._wait_queue.popleft()
                        self._in_flight += 1
                        if self._max_requests is not None:
                            self._request_timestamps.append(now)
                        if self._max_tokens is not None and token_reservation > 0:
                            self._token_timestamps.append((now, token_reservation))
                        self._condition.notify_all()
                        return LLMQueuePermit(
                            request_id=entry.request_id,
                            queued_at=entry.enqueued_at,
                            granted_at=now,
                            queue_depth_at_enqueue=entry.queue_depth_at_enqueue,
                        )

                    timeout = self._compute_wait_timeout(
                        now=now,
                        is_head=is_head,
                        deadline=deadline,
                        token_reservation=token_reservation,
                    )
                    if timeout is not None and timeout <= 0:
                        self._remove_entry(entry)
                        self._condition.notify_all()
                        raise TimeoutError(
                            "Timed out waiting for LLM queue permit "
                            f"(timeout={self._acquire_timeout_seconds}s)."
                        )

                    try:
                        if timeout is None:
                            await self._condition.wait()
                        else:
                            await asyncio.wait_for(self._condition.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        # Wake up and re-evaluate constraints/deadline.
                        pass
            except asyncio.CancelledError:
                self._remove_entry(entry)
                self._condition.notify_all()
                raise

    async def release(self) -> None:
        """Release one in-flight permit and wake pending waiters."""
        async with self._condition:
            if self._in_flight > 0:
                self._in_flight -= 1
            else:
                logger.warning("LLM queue release called with no in-flight requests.")
            self._condition.notify_all()

    def _remove_entry(self, entry: _QueueEntry) -> None:
        try:
            self._wait_queue.remove(entry)
        except ValueError:
            pass

    def _prune_rate_window(self, now: float) -> None:
        if self._max_requests is None:
            return
        cutoff = now - self._window_seconds
        while self._request_timestamps and self._request_timestamps[0] <= cutoff:
            self._request_timestamps.popleft()

    def _prune_token_window(self, now: float) -> None:
        if self._max_tokens is None:
            return
        cutoff = now - self._window_seconds
        while self._token_timestamps and self._token_timestamps[0][0] <= cutoff:
            self._token_timestamps.popleft()

    def _tokens_in_window(self, now: float) -> int:
        self._prune_token_window(now)
        return sum(tokens for _, tokens in self._token_timestamps)

    def _has_capacity(self, now: float, token_reservation: int) -> bool:
        if self._max_concurrent is not None and self._in_flight >= self._max_concurrent:
            return False
        if self._max_requests is None:
            request_capacity = True
        else:
            self._prune_rate_window(now)
            request_capacity = len(self._request_timestamps) < self._max_requests
        if not request_capacity:
            return False

        if self._max_tokens is None:
            return True
        used_tokens = self._tokens_in_window(now)
        return (used_tokens + token_reservation) <= self._max_tokens

    def _seconds_until_rate_capacity(self, now: float) -> float:
        if self._max_requests is None:
            return 0.0
        self._prune_rate_window(now)
        if len(self._request_timestamps) < self._max_requests:
            return 0.0
        oldest = self._request_timestamps[0]
        return max(0.0, (oldest + self._window_seconds) - now)

    def _seconds_until_token_capacity(self, now: float, token_reservation: int) -> float:
        if self._max_tokens is None:
            return 0.0

        self._prune_token_window(now)
        used = sum(tokens for _, tokens in self._token_timestamps)
        if (used + token_reservation) <= self._max_tokens:
            return 0.0

        removable = 0
        for ts, tokens in self._token_timestamps:
            removable += tokens
            if (used - removable + token_reservation) <= self._max_tokens:
                return max(0.0, (ts + self._window_seconds) - now)

        return float("inf")

    def _compute_wait_timeout(
        self,
        now: float,
        is_head: bool,
        deadline: Optional[float],
        token_reservation: int,
    ) -> Optional[float]:
        timeout: Optional[float] = None

        # Only the queue head needs timed wakeups for the rolling rate window.
        if is_head:
            rate_wait = self._seconds_until_rate_capacity(now)
            if rate_wait > 0:
                timeout = rate_wait
            token_wait = self._seconds_until_token_capacity(now, token_reservation)
            if token_wait > 0:
                timeout = token_wait if timeout is None else min(timeout, token_wait)

        if deadline is not None:
            remaining = deadline - now
            timeout = remaining if timeout is None else min(timeout, remaining)

        return timeout


class RateLimitedToolCallingLLM:
    """
    ToolCallingLLM wrapper that enforces queueing/rate limits before each call.
    """

    def __init__(
        self,
        llm: ToolCallingLLM,
        config: LLMRateLimitConfig,
    ) -> None:
        self._llm = llm
        self._config = config
        self._queue = LLMCallQueue(config)

    @property
    def inner_llm(self) -> ToolCallingLLM:
        return self._llm

    @property
    def config(self) -> LLMRateLimitConfig:
        return self._config

    async def chat_with_tools(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        model: str,
        model_config: ModelConfig,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        if not self._config.enabled:
            return await self._llm.chat_with_tools(
                messages=messages,
                tools=tools,
                model=model,
                model_config=model_config,
                stream=stream,
                on_token=on_token,
            )

        token_reservation = self._estimate_tokens(messages, tools, model_config)
        async with self._queue.reserve(token_reservation=token_reservation) as permit:
            if permit.wait_seconds > 0:
                logger.debug(
                    "LLM queue admitted request %s after %.0fms wait (enqueue_depth=%d)",
                    permit.request_id,
                    permit.wait_seconds * 1000,
                    permit.queue_depth_at_enqueue,
                )
            return await self._llm.chat_with_tools(
                messages=messages,
                tools=tools,
                model=model,
                model_config=model_config,
                stream=stream,
                on_token=on_token,
            )

    @staticmethod
    def _estimate_tokens(
        messages: List[Message],
        tools: List[Dict[str, Any]],
        model_config: ModelConfig,
    ) -> int:
        """Estimate total tokens for queue admission (prompt + likely completion)."""
        chars = 0
        for msg in messages:
            chars += len(msg.content or "")
            if msg.tool_call_id:
                chars += len(msg.tool_call_id)
            if msg.tool_calls:
                try:
                    chars += len(json.dumps(
                        [
                            {
                                "id": tc.id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                            }
                            for tc in msg.tool_calls
                        ],
                        default=str,
                    ))
                except Exception:
                    chars += 0

        if tools:
            try:
                chars += len(json.dumps(tools, default=str))
            except Exception:
                chars += 0

        prompt_tokens = max(1, int(chars / 4))
        completion_estimate = max(1, min(model_config.max_tokens, int(prompt_tokens * 0.5)))
        return prompt_tokens + completion_estimate
