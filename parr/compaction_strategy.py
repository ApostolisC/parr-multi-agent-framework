"""
Pluggable Context Compaction Strategy for the Agentic Framework.

Manages how conversation history is compacted when it grows too large for
the context window. The default implementation uses rule-based soft/hard
truncation. Subclass CompactionStrategy to customize behavior.

Two levels of compaction:
1. Soft compaction: smart — preserves findings and recent work, summarizes rest
2. Hard truncation: aggressive — keeps only system, user, and last 3 groups

Example::

    class LLMSummarizationStrategy(CompactionStrategy):
        def summarize_dropped(self, messages):
            # Use an LLM to summarize instead of rule-based counting
            return my_llm.summarize(messages)

    class RAGStrategy(CompactionStrategy):
        def compact_if_needed(self, messages, soft_fraction=None):
            # Replace dropped messages with RAG-retrieved context
            ...

    # Or via ContextManager:
    strategy = CompactionStrategy(chars_per_token=3.5)  # Use tighter estimate
    context_mgr = ContextManager(compaction_strategy=strategy)
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import List, Optional

from .core_types import Message, MessageRole, ToolCall

logger = logging.getLogger(__name__)

# Default approximate chars per token for estimation (conservative).
DEFAULT_CHARS_PER_TOKEN = 4.0


class CompactionStrategy:
    """
    Pluggable strategy for context window compaction.

    The default implementation provides two tiers of compaction:
    1. Soft compaction: preserves findings + recent groups, summarizes rest
    2. Hard truncation: keeps only system + user + last 3 groups

    Subclass and override methods to customize compaction behavior.
    For example, you might want to:
    - Use a real tokenizer (override estimate_tokens)
    - Use LLM-based summarization (override summarize_dropped)
    - Change what gets preserved (override should_preserve_group)
    - Replace compaction entirely (override compact_if_needed / truncate_if_needed)
    """

    def __init__(
        self,
        max_context_tokens: int = 128000,
        tool_schema_overhead: int = 0,
        soft_compaction_pct: float = 0.40,
        hard_truncation_pct: float = 0.65,
        chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    ) -> None:
        self._max_context_tokens = max_context_tokens
        self._tool_schema_overhead = tool_schema_overhead
        self._soft_compaction_pct = soft_compaction_pct
        self._hard_truncation_pct = hard_truncation_pct
        self._chars_per_token = chars_per_token

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    # -------------------------------------------------------------------
    # Token estimation — override for real tokenizer (e.g., tiktoken)
    # -------------------------------------------------------------------

    def estimate_tokens(self, messages: List[Message]) -> int:
        """Estimate token count for a message sequence.

        Uses a rough heuristic of ~1 token per chars_per_token characters.
        Override to integrate a real tokenizer like ``tiktoken``.

        Includes a fixed overhead for tool schemas that are sent with every
        LLM call but aren't part of the message history.
        """
        total_chars = 0
        for msg in messages:
            if msg.content:
                total_chars += len(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    total_chars += len(str(tc.arguments))
        return int(total_chars / self._chars_per_token) + self._tool_schema_overhead

    # -------------------------------------------------------------------
    # Soft compaction — override for custom smart compaction
    # -------------------------------------------------------------------

    def compact_if_needed(
        self,
        messages: List[Message],
        soft_fraction: Optional[float] = None,
    ) -> List[Message]:
        """
        Soft compaction at configurable context usage threshold.

        Smarter than hard truncation: summarizes completed todo/tool results
        but preserves important content like findings and raw data.

        Operates on *message groups* to avoid breaking the assistant->tool-result
        pairing required by chat APIs.

        Preserves:
        - System prompt (first message)
        - Original user message with task + raw data (second message)
        - Groups containing findings (log_finding results)
        - Last 3 groups (recent work)

        Summarizes and drops:
        - Completed todo tool calls/results
        - Old groups beyond the last 3

        Override for completely custom compaction (e.g., RAG-based retrieval,
        LLM summarization, semantic deduplication).
        """
        if soft_fraction is None:
            soft_fraction = self._soft_compaction_pct
        token_limit = int(self._max_context_tokens * soft_fraction)
        current_tokens = self.estimate_tokens(messages)

        if current_tokens <= token_limit:
            return messages

        # Not enough messages to compact meaningfully — but still over limit.
        if len(messages) <= 6:
            logger.warning(
                f"Context ({current_tokens} tokens) exceeds soft limit "
                f"({token_limit}) but has too few messages ({len(messages)}) "
                f"to compact. Consider reducing system prompt or input size."
            )
            return messages

        logger.info(
            f"Soft compaction triggered: {current_tokens} tokens "
            f"exceeds soft limit of {token_limit}"
        )

        preserved_start = messages[:2]  # system + user

        # Group remaining messages into assistant+tool pairs
        groups = self.pair_messages(messages[2:])

        if len(groups) <= 3:
            return messages  # Not enough groups to compact

        # Keep last 3 groups as recent context
        middle_groups = groups[:-3]
        recent_groups = groups[-3:]

        # Identify groups containing important content vs droppable
        preserved_groups: List[List[Message]] = []
        droppable_messages: List[Message] = []
        for group in middle_groups:
            if self.should_preserve_group(group):
                preserved_groups.append(group)
            else:
                droppable_messages.extend(group)

        # Build summary of dropped work
        dropped_summary = self.summarize_dropped(droppable_messages)

        result = list(preserved_start)
        if dropped_summary:
            result.append(Message(
                role=MessageRole.USER,
                content=(
                    "[CONTEXT COMPACTED] Some earlier messages were summarised "
                    "to free context space. Your findings and recent work are "
                    "preserved. If you need details that appear missing, check "
                    "your working memory (get_findings / get_todo_list).\n\n"
                    f"{dropped_summary}"
                ),
            ))
        for group in preserved_groups:
            result.extend(group)
        for group in recent_groups:
            result.extend(group)

        new_tokens = self.estimate_tokens(result)
        logger.info(f"Soft compaction: {current_tokens} -> {new_tokens} tokens")
        return result

    # -------------------------------------------------------------------
    # Hard truncation — override for custom aggressive compaction
    # -------------------------------------------------------------------

    def truncate_if_needed(
        self,
        messages: List[Message],
        max_fraction: Optional[float] = None,
    ) -> List[Message]:
        """
        Hard truncation at configurable context usage threshold.

        First attempts soft compaction. If still over the hard limit,
        aggressively truncates to system + user + last 3 message groups.

        Operates on *message groups* to avoid breaking the assistant->tool-result
        pairing required by chat APIs.

        Override to replace the hard truncation behavior entirely.
        """
        # First try soft compaction
        messages = self.compact_if_needed(messages)

        if max_fraction is None:
            max_fraction = self._hard_truncation_pct
        token_limit = int(self._max_context_tokens * max_fraction)
        current_tokens = self.estimate_tokens(messages)

        if current_tokens <= token_limit:
            return messages

        logger.info(
            f"Hard truncation triggered: {current_tokens} tokens "
            f"exceeds limit of {token_limit}"
        )

        # Group messages after the preserved start
        groups = self.pair_messages(messages[2:])

        if len(groups) <= 3:
            logger.warning(
                f"Context ({current_tokens} tokens) exceeds hard limit "
                f"({token_limit}) but has too few message groups "
                f"({len(groups)}) to truncate further."
            )
            return messages  # Not enough to truncate

        preserved_start = messages[:2]  # system + user
        recent_groups = groups[-3:]  # last 3 groups

        # Build a summary of what was dropped
        dropped_messages: List[Message] = []
        for group in groups[:-3]:
            dropped_messages.extend(group)
        dropped_summary = self.summarize_dropped(dropped_messages)

        result = list(preserved_start)
        if dropped_summary:
            result.append(Message(
                role=MessageRole.USER,
                content=(
                    "[CONTEXT TRUNCATED] Earlier messages were aggressively "
                    "trimmed to stay within the context limit. Only your most "
                    "recent work is visible. Rely on your working memory "
                    "(get_findings / get_todo_list) for prior results.\n\n"
                    f"{dropped_summary}"
                ),
            ))
        for group in recent_groups:
            result.extend(group)

        new_tokens = self.estimate_tokens(result)
        logger.info(f"Hard truncation: {current_tokens} -> {new_tokens} tokens")
        return result

    # -------------------------------------------------------------------
    # Preservation rules — override to change what gets kept
    # -------------------------------------------------------------------

    def should_preserve_group(self, group: List[Message]) -> bool:
        """
        Determine if a message group should be preserved during compaction.

        Default: preserves groups containing findings (log_finding, get_findings).
        Override to change preservation rules (e.g., preserve search results,
        custom tool outputs, or messages with certain keywords).
        """
        return any(self._contains_findings(msg) for msg in group)

    def _contains_findings(self, msg: Message) -> bool:
        """Check if a message contains findings data worth preserving."""
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.name in ("log_finding", "get_findings"):
                    return True
        return False

    # -------------------------------------------------------------------
    # Summarization — override for LLM-based or custom summaries
    # -------------------------------------------------------------------

    def summarize_dropped(self, messages: List[Message]) -> str:
        """
        Create a brief summary of dropped messages.

        Default: rule-based tool call counting.
        Override for LLM-based summarization, semantic extraction, etc.
        """
        tool_calls = []
        for msg in messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(tc.name)

        if not tool_calls:
            return ""

        counts = Counter(tool_calls)
        lines = [f"- {name}: called {count} time(s)" for name, count in counts.items()]
        return "Tools called in earlier work:\n" + "\n".join(lines)

    # -------------------------------------------------------------------
    # Message grouping — utility for subclasses
    # -------------------------------------------------------------------

    @staticmethod
    def pair_messages(messages: List[Message]) -> List[List[Message]]:
        """Group messages into logical units that must stay together.

        An assistant message with ``tool_calls`` is paired with the subsequent
        tool-result messages (role == TOOL) that answer those calls.  All other
        messages form single-element groups.

        This prevents compaction / truncation from splitting an assistant
        tool-call message from its tool-result messages, which would produce
        an invalid conversation for the LLM API.
        """
        groups: List[List[Message]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
                # Start a new group with this assistant message
                group = [msg]
                i += 1
                # Collect all subsequent TOOL messages
                while i < len(messages) and messages[i].role == MessageRole.TOOL:
                    group.append(messages[i])
                    i += 1
                groups.append(group)
            else:
                groups.append([msg])
                i += 1
        return groups
