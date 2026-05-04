"""
ResultCacheMiddleware — per-agent dedup of read-only tool results.

A ToolMiddleware that caches successful results of read-only tools and
short-circuits subsequent identical calls inside the same scope. Useful
for cutting cost when an agent (or its synthesis loop) re-issues the
same ``get_collection`` / ``get_findings`` / ``list_collection`` /
``get_todo_list`` call across iterations.

Generic and configurable — knows nothing about any specific tool or
domain. Cache scope, eligibility, and capacity are all controlled at
construction time so the framework, an adapter, or a config loader can
opt in without modifying tool definitions.

Eligibility (in order of precedence):
    1. ``deny_tools`` — explicit blocklist always wins.
    2. ``allow_tools`` — when set, only these tools are cached.
    3. ``cache_read_only`` — when True (default), any tool whose
       ``ToolDef.is_read_only`` is True is cached.

Scope:
    - ``"phase"`` (default): cache is cleared whenever the executing
      phase changes. Most agents read state, act on it, then re-read in
      the same phase, so phase scope is the safest default — there's no
      risk of returning a stale read from a previous phase.
    - ``"agent"``: cache persists for the lifetime of the executor.
      Use this for tools whose results are truly stable across phases
      (config lookups, immutable reference data).

Bounded capacity is enforced via a simple LRU. Set ``max_entries=None``
for unbounded (not recommended in long-running agents).

Cache stats are exposed via ``stats`` for cost-tracking integrations.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set, Union

from .core_types import Phase, ToolCall, ToolContext, ToolDef, ToolMiddleware, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class ResultCacheStats:
    """Counters for observability — read by tests and cost reports."""
    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0
    resets: int = 0
    skipped_ineligible: int = 0
    skipped_failures: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "evictions": self.evictions,
            "resets": self.resets,
            "skipped_ineligible": self.skipped_ineligible,
            "skipped_failures": self.skipped_failures,
        }


class ResultCacheMiddleware(ToolMiddleware):
    """
    Cache successful results of read-only tools and short-circuit duplicates.

    Args:
        scope: ``"phase"`` (default) clears the cache whenever the phase
            changes. ``"agent"`` keeps the cache for the executor's
            lifetime.
        max_entries: Cap on cached entries (LRU). ``None`` = unbounded.
            Defaults to 256.
        cache_read_only: When True (default), any tool with
            ``ToolDef.is_read_only=True`` is cached. Disable to opt
            out of automatic detection and rely solely on
            ``allow_tools``.
        allow_tools: Optional explicit allowlist of tool names. When
            set, ONLY tools in this set are cached (regardless of the
            read-only flag).
        deny_tools: Optional blocklist. Tools in this set are NEVER
            cached, even if they pass other eligibility rules.
        ignore_args: Optional iterable of argument names to drop from
            the cache key (e.g. timestamps, request IDs that don't
            affect output). Useful when a tool accepts a "trace_id"
            arg that's incidental to the result.
        enabled: Master switch. When False, the middleware behaves as
            a no-op so callers can keep it wired without measurable
            overhead.

    Example::

        from parr import ToolExecutor, ResultCacheMiddleware

        cache = ResultCacheMiddleware(scope="phase", max_entries=128)
        executor = ToolExecutor(registry, middleware=[cache])
        # ... run agent ...
        print(cache.stats.to_dict())
    """

    SCOPE_PHASE = "phase"
    SCOPE_AGENT = "agent"

    def __init__(
        self,
        *,
        scope: str = "phase",
        max_entries: Optional[int] = 256,
        cache_read_only: bool = True,
        allow_tools: Optional[Iterable[str]] = None,
        deny_tools: Optional[Iterable[str]] = None,
        ignore_args: Optional[Iterable[str]] = None,
        enabled: bool = True,
    ) -> None:
        if scope not in (self.SCOPE_PHASE, self.SCOPE_AGENT):
            raise ValueError(
                f"Invalid scope '{scope}'. Use 'phase' or 'agent'."
            )
        self._scope = scope
        self._max_entries = max_entries
        self._cache_read_only = cache_read_only
        self._allow_tools: Optional[Set[str]] = (
            set(allow_tools) if allow_tools is not None else None
        )
        self._deny_tools: Set[str] = set(deny_tools or ())
        self._ignore_args: Set[str] = set(ignore_args or ())
        self._enabled = enabled

        # OrderedDict gives O(1) LRU semantics via move_to_end + popitem.
        self._cache: "OrderedDict[str, ToolResult]" = OrderedDict()
        self._current_scope_phase: Optional[Phase] = None
        self.stats = ResultCacheStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Toggle the cache without recreating the middleware."""
        self._enabled = enabled

    def reset(self) -> None:
        """Clear all cached entries (manual cache invalidation hook)."""
        if self._cache:
            self.stats.resets += 1
        self._cache.clear()
        self._current_scope_phase = None

    def __len__(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # ToolMiddleware hooks
    # ------------------------------------------------------------------

    async def pre_call(
        self,
        tool_call: ToolCall,
        tool_def: ToolDef,
        context: ToolContext,
    ) -> Union[ToolCall, ToolResult]:
        if not self._enabled:
            return tool_call

        # Phase-scoped cache: drop everything when the phase rotates so
        # an agent never sees a snapshot of state from the previous phase.
        if self._scope == self.SCOPE_PHASE:
            if self._current_scope_phase != context.phase and self._cache:
                self._cache.clear()
                self.stats.resets += 1
            self._current_scope_phase = context.phase

        if not self._is_eligible(tool_call.name, tool_def):
            self.stats.skipped_ineligible += 1
            return tool_call

        key = self._make_key(tool_call, context)
        cached = self._cache.get(key)
        if cached is not None:
            # LRU bookkeeping — touch the entry so it stays warm.
            self._cache.move_to_end(key)
            self.stats.hits += 1
            logger.debug(
                "ResultCache HIT for tool=%s (key=%s, hits=%d)",
                tool_call.name, key[:80], self.stats.hits,
            )
            # Re-issue the cached ToolResult under the *current* call id
            # so the LLM correlates it with the call it just made.
            return ToolResult(
                tool_call_id=tool_call.id,
                success=cached.success,
                content=cached.content,
                error=cached.error,
            )

        self.stats.misses += 1
        # Stash the key so post_call can store without re-canonicalizing.
        context.metadata["__result_cache_key"] = key
        return tool_call

    async def post_call(
        self,
        result: ToolResult,
        tool_call: ToolCall,
        tool_def: ToolDef,
        context: ToolContext,
    ) -> ToolResult:
        if not self._enabled:
            return result
        if not result.success:
            # Never cache failures — they may be transient.
            self.stats.skipped_failures += 1
            return result

        key = context.metadata.pop("__result_cache_key", None)
        if key is None:
            # Either ineligible or short-circuited from cache; nothing to store.
            return result

        self._cache[key] = result
        self._cache.move_to_end(key)
        self.stats.stores += 1

        if self._max_entries is not None and len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)  # evict least-recently-used
            self.stats.evictions += 1

        return result

    # ------------------------------------------------------------------
    # Eligibility + key generation
    # ------------------------------------------------------------------

    def _is_eligible(self, tool_name: str, tool_def: ToolDef) -> bool:
        if tool_name in self._deny_tools:
            return False
        if self._allow_tools is not None:
            return tool_name in self._allow_tools
        if self._cache_read_only and tool_def.is_read_only:
            return True
        return False

    def _make_key(self, tool_call: ToolCall, context: ToolContext) -> str:
        """Build a stable cache key from tool name + canonical args.

        Phase is part of the key when scope='phase' so two phases that
        happen to call the same tool with the same args don't collide
        before the phase rotation has had a chance to flush the cache
        (defense in depth — the rotation in pre_call already handles
        the common case).
        """
        if self._ignore_args:
            args = {
                k: v for k, v in tool_call.arguments.items()
                if k not in self._ignore_args
            }
        else:
            args = tool_call.arguments
        try:
            arg_repr = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fall back to repr — still deterministic for the same input.
            arg_repr = repr(sorted(args.items())) if isinstance(args, dict) else repr(args)
        if self._scope == self.SCOPE_PHASE and context.phase is not None:
            return f"{context.phase.value}::{tool_call.name}::{arg_repr}"
        return f"{tool_call.name}::{arg_repr}"
