"""
Built-in Framework Tools.

These tools are provided by the framework and work identically regardless
of domain. They are automatically registered for every agent.

Categories:
- Plan management: create_todo_list, update_todo_list, get_todo_list, mark_todo_complete
- Findings management: log_finding, get_findings
- Agent management: spawn_agent, wait_for_agents, get_agent_result, get_agent_results_all
  (these are orchestrator-intercepted — their handlers are never called by ToolExecutor)
- Review tools: review_checklist
- Report tools: get_report_template, submit_report
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from .core_types import Confidence, Phase, ToolDef

if TYPE_CHECKING:
    from .protocols import TextSummarizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory state containers (per-agent, managed by the runtime)
# ---------------------------------------------------------------------------

@dataclass
class TodoItem:
    """A single item in an agent's todo list."""
    index: int
    description: str
    priority: str = "medium"  # "high", "medium", "low"
    completed: bool = False
    completion_summary: Optional[str] = None


@dataclass
class Finding:
    """A finding recorded during agent execution."""
    category: str
    content: str
    source: str
    confidence: str = "medium"


@dataclass
class ReviewItem:
    """A single criterion in the review checklist."""
    criterion: str
    rating: str  # "pass", "partial", "fail"
    justification: str


@dataclass
class CollectionItem:
    """A single item in a memory collection."""
    id: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryCollection:
    """A named, agent-created memory collection.

    Collections allow agents to persist structured knowledge that
    survives across phases and can be read by subsequent agents.
    Each collection has a name, description, optional extra context
    that describes the collection's purpose, and an ordered list
    of items.

    A version counter (``_items_version``) is bumped on every
    mutation. The framework's summarize-on-read cache is keyed by
    ``(version, instructions)`` so any change automatically
    invalidates stale summaries.
    """
    name: str
    description: str
    extra: str = ""  # Additional context about the collection
    items: List[CollectionItem] = field(default_factory=list)
    _next_id: int = field(default=0, repr=False)
    # Bumped on every mutation; used as the cache version for
    # summarize-on-read so summaries auto-invalidate when items change.
    _items_version: int = field(default=0, repr=False)
    # Cached summaries keyed by instructions string -> (version, summary).
    _summary_cache: Dict[str, Tuple[int, str]] = field(default_factory=dict, repr=False)

    def _bump_version(self) -> None:
        """Bump the items version, invalidating any cached summaries."""
        self._items_version += 1

    def add_item(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        item_id = self._next_id
        self._next_id += 1
        self.items.append(CollectionItem(
            id=item_id,
            content=content,
            metadata=metadata or {},
        ))
        self._bump_version()
        return item_id

    def get_item(self, item_id: int) -> Optional[CollectionItem]:
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def get_cached_summary(self, instructions: Optional[str]) -> Optional[str]:
        """Return a cached summary if it matches the current items version.

        Args:
            instructions: Caller-supplied summary directive (may be None).
                Used as the cache key — different instructions get
                different cache entries because they ask for different
                things.

        Returns:
            The cached summary string, or None if no cache entry exists
            or the cache is stale (items have changed since it was set).
        """
        key = instructions or ""
        entry = self._summary_cache.get(key)
        if entry is None:
            return None
        version, summary = entry
        if version != self._items_version:
            # Stale — drop it so the cache stays clean.
            self._summary_cache.pop(key, None)
            return None
        return summary

    def set_cached_summary(self, instructions: Optional[str], summary: str) -> None:
        """Store a summary in the cache at the current items version."""
        key = instructions or ""
        self._summary_cache[key] = (self._items_version, summary)

    def total_content_chars(self) -> int:
        """Total characters across all item contents.

        Used by ``AgentWorkingMemory.get_collection_async`` to decide
        whether the auto-summarize threshold is exceeded. Cheap — just
        sums string lengths, no formatting.
        """
        return sum(len(item.content or "") for item in self.items)

    def to_snapshot(self, max_items: Optional[int] = None) -> str:
        """Render collection as text for working memory snapshot.

        Args:
            max_items: When set and the collection contains more items
                than this, only the most recent ``max_items`` are
                rendered with their full content; older items are hidden
                behind a hint that points the agent at ``list_collection``
                / ``get_collection``. ``None`` (default) renders every
                item — preserves backwards-compatible behavior for
                callers and tests that expect the full dump.
        """
        header = f"## {self.name}\n{self.description}"
        if self.extra:
            header += f"\n{self.extra}"
        if not self.items:
            return f"{header}\n(empty)"

        total = len(self.items)
        if max_items is not None and max_items >= 0 and total > max_items:
            hidden = total - max_items
            visible = self.items[-max_items:] if max_items > 0 else []
            lines = [header]
            lines.append(
                f"  ({hidden} older item{'s' if hidden != 1 else ''} hidden — "
                f"call list_collection('{self.name}') for IDs/snippets or "
                f"get_collection('{self.name}') for the full dump.)"
            )
            for item in visible:
                meta_str = f" [{', '.join(f'{k}={v}' for k, v in item.metadata.items())}]" if item.metadata else ""
                lines.append(f"  [{item.id}]{meta_str} {item.content}")
            return "\n".join(lines)

        lines = [header]
        for item in self.items:
            meta_str = f" [{', '.join(f'{k}={v}' for k, v in item.metadata.items())}]" if item.metadata else ""
            lines.append(f"  [{item.id}]{meta_str} {item.content}")
        return "\n".join(lines)

    def to_inventory(self, snippet_chars: int = 80) -> str:
        """Render a lightweight inventory of items.

        Returns IDs, metadata, and short content snippets — never the
        full content. Used by the ``list_collection`` tool so agents
        can scan a large collection cheaply and decide which items
        warrant a follow-up ``get_collection`` call.
        """
        header = f"## {self.name} ({len(self.items)} items)\n{self.description}"
        if self.extra:
            header += f"\n{self.extra}"
        if not self.items:
            return f"{header}\n(empty)"
        lines = [header]
        for item in self.items:
            meta_str = (
                f" [{', '.join(f'{k}={v}' for k, v in item.metadata.items())}]"
                if item.metadata else ""
            )
            content = item.content or ""
            if snippet_chars > 0 and len(content) > snippet_chars:
                snippet = content[:snippet_chars].rstrip() + "…"
            else:
                snippet = content
            # Newlines in snippets break the one-item-per-line layout.
            snippet = snippet.replace("\n", " ")
            lines.append(f"  [{item.id}]{meta_str} {snippet}")
        return "\n".join(lines)


class AgentWorkingMemory:
    """
    In-memory working state for a single agent.

    Created by the agent runtime and passed to framework tool handlers.
    Holds the todo list, findings, review results, and submitted report.

    Args:
        snapshot_max_items_per_collection: Optional cap on how many items
            from each collection are rendered into the working memory
            snapshot per phase. ``None`` (default) renders every item —
            preserves prior behavior. When set, snapshots show only the
            most recent N items per collection plus a hint pointing the
            agent at ``list_collection`` for the full inventory or
            ``get_collection`` for the full dump. Wired through
            ``BudgetConfig.snapshot_max_items_per_collection``.
        summarizer: Optional ``TextSummarizer`` used by ``get_collection``
            and ``list_collection`` when the agent (or framework) asks
            for a summarized read. ``None`` disables summarize-on-read —
            tools always return raw content.
        summarize_threshold_chars: When the agent calls
            ``get_collection`` without an explicit ``summarize`` flag,
            the framework auto-summarizes if total collection content
            exceeds this many characters. ``0`` disables auto-summarize
            (callers must pass ``summarize=True`` explicitly). Wired
            through ``BudgetConfig.summarize_on_read_threshold_chars``.
        default_summary_instructions: Optional caller-supplied default
            instructions handed to the summarizer when the tool call
            does not specify any. Useful for steering the global
            shape of summaries (e.g. "preserve all quantified
            findings and source citations").
    """

    def __init__(
        self,
        snapshot_max_items_per_collection: Optional[int] = None,
        summarizer: Optional["TextSummarizer"] = None,
        summarize_threshold_chars: int = 0,
        default_summary_instructions: Optional[str] = None,
    ) -> None:
        self.todo_list: List[TodoItem] = []
        self.findings: List[Finding] = []
        self.review_checklist: List[ReviewItem] = []
        self.submitted_report: Optional[Dict[str, Any]] = None
        self.collections: Dict[str, MemoryCollection] = {}
        self.snapshot_max_items_per_collection = snapshot_max_items_per_collection
        self._summarizer = summarizer
        self.summarize_threshold_chars = max(0, summarize_threshold_chars)
        self.default_summary_instructions = default_summary_instructions
        # Adaptive phase routing: when an agent calls set_next_phase()
        # the requested phase is stored here. The runtime reads and
        # consumes (clears) this between phases. None means no override.
        self.requested_next_phase: Optional[str] = None

    # -- Todo operations --

    def create_todo_list(self, items: List[Dict[str, Any]]) -> str:
        """Create the agent's todo list."""
        if self.todo_list:
            return "Error: Todo list already exists. Use update_todo_list to modify."
        self.todo_list = [
            TodoItem(
                index=i,
                description=item.get("description", ""),
                priority=item.get("priority", "medium"),
            )
            for i, item in enumerate(items)
        ]
        return f"Created todo list with {len(self.todo_list)} items."

    def update_todo_list(
        self,
        add: Optional[List[Dict[str, Any]]] = None,
        remove: Optional[List[int]] = None,
        modify: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Modify the todo list."""
        changes = []
        if remove:
            remove_set = set(remove)
            self.todo_list = [t for t in self.todo_list if t.index not in remove_set]
            changes.append(f"removed {len(remove_set)} items")
        if modify:
            for mod in modify:
                idx = mod.get("index")
                for item in self.todo_list:
                    if item.index == idx:
                        if "description" in mod:
                            item.description = mod["description"]
                        if "priority" in mod:
                            item.priority = mod["priority"]
            changes.append(f"modified {len(modify)} items")
        if add:
            start_idx = max((t.index for t in self.todo_list), default=-1) + 1
            for i, item in enumerate(add):
                self.todo_list.append(
                    TodoItem(
                        index=start_idx + i,
                        description=item.get("description", ""),
                        priority=item.get("priority", "medium"),
                    )
                )
            changes.append(f"added {len(add)} items")
        return f"Updated todo list: {', '.join(changes)}. Total items: {len(self.todo_list)}."

    def get_todo_list(self) -> str:
        """Return the current todo list as formatted text."""
        if not self.todo_list:
            return "No todo items."
        lines = []
        for item in self.todo_list:
            status = "[x]" if item.completed else "[ ]"
            summary = f" - {item.completion_summary}" if item.completion_summary else ""
            lines.append(
                f"{status} {item.index}. [{item.priority}] {item.description}{summary}"
            )
        return "\n".join(lines)

    def mark_todo_complete(self, item_index: int, summary: str) -> str:
        """Mark a todo item as completed with a summary."""
        for item in self.todo_list:
            if item.index == item_index:
                item.completed = True
                item.completion_summary = summary
                return f"Marked item {item_index} as complete."
        return f"Error: No todo item with index {item_index}."

    def batch_mark_todo_complete(self, items: List[Dict[str, Any]]) -> str:
        """Mark multiple todo items as completed in one call."""
        results = []
        for entry in items:
            idx = entry.get("item_index")
            summary = entry.get("summary", "")
            found = False
            for item in self.todo_list:
                if item.index == idx:
                    item.completed = True
                    item.completion_summary = summary
                    found = True
                    break
            if found:
                results.append(f"Item {idx}: marked complete")
            else:
                results.append(f"Item {idx}: not found")
        return f"Batch complete: {len(results)} items processed.\n" + "\n".join(results)

    # -- Findings operations --

    def log_finding(
        self, category: str, content: str, source: str, confidence: str = "medium"
    ) -> str:
        """Record a finding."""
        self.findings.append(Finding(
            category=category, content=content,
            source=source, confidence=confidence,
        ))
        return f"Logged finding in category '{category}'. Total findings: {len(self.findings)}."

    def batch_log_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Record multiple findings in one call."""
        for f in findings:
            self.findings.append(Finding(
                category=f.get("category", "general"),
                content=f.get("content", ""),
                source=f.get("source", ""),
                confidence=f.get("confidence", "medium"),
            ))
        return f"Logged {len(findings)} findings. Total findings: {len(self.findings)}."

    def get_findings(self, category: Optional[str] = None) -> str:
        """Retrieve findings, optionally filtered by category."""
        filtered = self.findings
        if category:
            filtered = [f for f in filtered if f.category == category]
        if not filtered:
            return "No findings recorded." if not category else f"No findings in category '{category}'."
        lines = []
        for i, f in enumerate(filtered):
            lines.append(
                f"{i+1}. [{f.confidence}] [{f.category}] {f.content} (source: {f.source})"
            )
        return "\n".join(lines)

    # -- Review operations --

    def record_review_checklist(self, items: List[Dict[str, Any]]) -> str:
        """Record review checklist evaluation."""
        self.review_checklist = [
            ReviewItem(
                criterion=item.get("criterion", ""),
                rating=item.get("rating", "partial"),
                justification=item.get("justification", ""),
            )
            for item in items
        ]
        pass_count = sum(1 for r in self.review_checklist if r.rating == "pass")
        total = len(self.review_checklist)
        return f"Review checklist recorded: {pass_count}/{total} criteria passed."

    def get_review_summary(self) -> str:
        """Get a summary of the review checklist."""
        if not self.review_checklist:
            return "No review checklist recorded."
        lines = []
        for item in self.review_checklist:
            lines.append(f"[{item.rating.upper()}] {item.criterion}: {item.justification}")
        return "\n".join(lines)

    # -- Transition operations --

    def set_next_phase(self, phase: str, reason: str = "") -> str:
        """Record a phase transition request."""
        valid_phases = {"plan", "act", "review", "report"}
        if phase not in valid_phases:
            return f"Error: Invalid phase '{phase}'. Must be one of: {', '.join(sorted(valid_phases))}"
        self.requested_next_phase = phase
        return f"Transition registered: will go to '{phase}' after current phase. Reason: {reason}"

    # -- Report operations --

    def submit_report(self, report: Dict[str, Any]) -> str:
        """Submit the final report."""
        self.submitted_report = report
        return "Report submitted successfully."

    # -- Memory collections --

    def create_collection(
        self, name: str, description: str, extra: str = "",
        initial_items: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Create a new named memory collection, optionally with initial items."""
        key = name.strip().lower().replace(" ", "_")
        if key in self.collections:
            return f"Error: Collection '{name}' already exists. Use add_to_collection or edit_collection_item."
        col = MemoryCollection(name=name, description=description, extra=extra)
        if initial_items:
            for item in initial_items:
                content = item.get("content", "")
                metadata = item.get("metadata", {})
                col.add_item(content, metadata)
        self.collections[key] = col
        count_msg = f" with {len(initial_items)} items" if initial_items else ""
        return f"Created collection '{name}'{count_msg}."

    def list_collections(self) -> str:
        """List all available collections with their metadata."""
        if not self.collections:
            return "No collections."
        lines = []
        for key, col in self.collections.items():
            lines.append(
                f"- {col.name} ({len(col.items)} items): {col.description}"
                + (f"\n  Context: {col.extra}" if col.extra else "")
            )
        return "\n".join(lines)

    def get_collection(self, name: str) -> str:
        """Get all items from a collection (raw, no summarization)."""
        key = name.strip().lower().replace(" ", "_")
        col = self.collections.get(key)
        if not col:
            return f"Error: Collection '{name}' not found. Available: {', '.join(c.name for c in self.collections.values()) or 'none'}."
        return col.to_snapshot()

    async def get_collection_async(
        self,
        name: str,
        summarize: Optional[bool] = None,
        summary_instructions: Optional[str] = None,
    ) -> str:
        """Async read of a collection with optional summarize-on-read.

        Args:
            name: Collection name.
            summarize:
                * ``True``  — always summarize (cache when possible).
                * ``False`` — never summarize, return the raw snapshot.
                * ``None``  — auto: summarize when total content exceeds
                  ``self.summarize_threshold_chars`` (and a summarizer
                  is configured); otherwise return raw.
            summary_instructions: Optional directive passed to the
                summarizer (e.g. "keep all quantified findings"). Falls
                back to ``self.default_summary_instructions`` when not
                provided.

        Returns:
            Either the raw collection snapshot or a summary string.
            Falls back to the raw snapshot if the summarizer is missing
            or raises — the framework never lets summarization failures
            block the agent.
        """
        key = name.strip().lower().replace(" ", "_")
        col = self.collections.get(key)
        if not col:
            return (
                f"Error: Collection '{name}' not found. Available: "
                f"{', '.join(c.name for c in self.collections.values()) or 'none'}."
            )

        raw = col.to_snapshot()

        # Decide whether to summarize.
        should_summarize: bool
        if summarize is True:
            should_summarize = True
        elif summarize is False:
            should_summarize = False
        else:
            # Auto: only when a summarizer is configured AND threshold > 0
            # AND the collection content exceeds the threshold.
            should_summarize = bool(
                self._summarizer
                and self.summarize_threshold_chars > 0
                and col.total_content_chars() >= self.summarize_threshold_chars
            )

        if not should_summarize:
            return raw
        if self._summarizer is None:
            # Caller asked for a summary but none is configured — fall
            # back to raw with a clear hint instead of failing silently.
            return (
                "(summarize requested but no summarizer is configured — "
                "returning raw collection)\n\n" + raw
            )

        # Resolve instructions: explicit > default > None.
        instructions = summary_instructions or self.default_summary_instructions

        # Cache lookup keyed by (items_version, instructions).
        cached = col.get_cached_summary(instructions)
        if cached is not None:
            return cached

        # Live summarize.
        try:
            summary = await self._summarizer.summarize(
                text=raw,
                instructions=instructions,
            )
        except Exception as e:
            logger.warning(
                "[PARR] Summarizer failed for collection '%s': %s — "
                "falling back to raw content.",
                col.name, e,
            )
            return raw

        if not isinstance(summary, str) or not summary.strip():
            logger.warning(
                "[PARR] Summarizer returned empty/invalid output for "
                "collection '%s' — falling back to raw content.",
                col.name,
            )
            return raw

        col.set_cached_summary(instructions, summary)
        return summary

    def list_collection_items(self, name: str, snippet_chars: int = 80) -> str:
        """Return a lightweight inventory (IDs + metadata + snippets) for one collection.

        Used by the ``list_collection`` tool. Cheaper than ``get_collection``
        because it never returns the full content of any item — agents
        scan the inventory and follow up with ``get_collection`` only
        when they need a full read.
        """
        key = name.strip().lower().replace(" ", "_")
        col = self.collections.get(key)
        if not col:
            return f"Error: Collection '{name}' not found. Available: {', '.join(c.name for c in self.collections.values()) or 'none'}."
        return col.to_inventory(snippet_chars=snippet_chars)

    def add_to_collection(
        self, name: str, items: List[Dict[str, Any]],
    ) -> str:
        """Add one or more items to a collection."""
        key = name.strip().lower().replace(" ", "_")
        col = self.collections.get(key)
        if not col:
            return f"Error: Collection '{name}' not found."
        ids = []
        for item in items:
            content = item.get("content", "")
            metadata = item.get("metadata", {})
            item_id = col.add_item(content, metadata)
            ids.append(str(item_id))
        return f"Added {len(ids)} items to '{col.name}'. IDs: {', '.join(ids)}. Total: {len(col.items)}."

    def edit_collection_item(
        self, name: str, item_id: int,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Edit a specific item in a collection."""
        key = name.strip().lower().replace(" ", "_")
        col = self.collections.get(key)
        if not col:
            return f"Error: Collection '{name}' not found."
        item = col.get_item(item_id)
        if not item:
            return f"Error: Item {item_id} not found in '{col.name}'."
        if content is not None:
            item.content = content
        if metadata is not None:
            item.metadata.update(metadata)
        col._bump_version()
        return f"Updated item {item_id} in '{col.name}'."

    def remove_collection_item(self, name: str, item_id: int) -> str:
        """Remove a specific item from a collection."""
        key = name.strip().lower().replace(" ", "_")
        col = self.collections.get(key)
        if not col:
            return f"Error: Collection '{name}' not found."
        original_len = len(col.items)
        col.items = [i for i in col.items if i.id != item_id]
        if len(col.items) == original_len:
            return f"Error: Item {item_id} not found in '{col.name}'."
        col._bump_version()
        return f"Removed item {item_id} from '{col.name}'. Remaining: {len(col.items)}."

    async def batch_operations(self, operations: List[Dict[str, Any]]) -> str:
        """Execute multiple memory operations in a single call.

        Supported ops: create_todo_list, mark_todos_complete,
        log_findings, create_collection, add_to_collection,
        read_collections. Operations execute sequentially so
        dependencies work (e.g. create_collection then
        add_to_collection in the same call).

        Note: this method is async because ``read_collections`` may
        invoke the framework's summarizer, which is async. Sync ops
        run inline (no overhead). The tool handler awaits the result,
        so callers see a single string back as before.
        """
        results: List[str] = []
        for i, op in enumerate(operations, 1):
            op_type = op.get("op", "").strip()
            try:
                if op_type == "create_todo_list":
                    r = self.create_todo_list(op.get("items", []))
                elif op_type == "mark_todos_complete":
                    r = self.batch_mark_todo_complete(op.get("items", []))
                elif op_type == "log_findings":
                    r = self.batch_log_findings(op.get("findings", []))
                elif op_type == "create_collection":
                    r = self.create_collection(
                        op.get("collection_name", ""),
                        op.get("description", ""),
                        op.get("extra", ""),
                        op.get("initial_items"),
                    )
                elif op_type == "add_to_collection":
                    r = self.add_to_collection(
                        op.get("collection_name", ""),
                        op.get("items", []),
                    )
                elif op_type == "read_collections":
                    r = await self._batch_read_collections(
                        op.get("collections", []),
                    )
                else:
                    r = f"Unknown operation '{op_type}'"
            except Exception as e:
                r = f"Error in {op_type}: {e}"
            results.append(f"[{i}] {op_type}: {r}")
        return "\n".join(results)

    async def _batch_read_collections(
        self,
        reads: List[Dict[str, Any]],
    ) -> str:
        """Execute a batch of collection reads inside batch_operations.

        Each read entry supports:
            collection_name: str (required)
            summarize: Optional[bool] (default: None — auto)
            summary_instructions: Optional[str]

        Output is a single string with one section per read, delimited
        by ``=== <collection_name> ===`` headers so the agent can split
        them. Errors per read are inlined into that read's section
        instead of aborting the whole batch.
        """
        if not isinstance(reads, list) or not reads:
            return "(no collections requested)"
        sections: List[str] = []
        for entry in reads:
            if not isinstance(entry, dict):
                sections.append("=== invalid ===\n(read entry must be an object)")
                continue
            name = (entry.get("collection_name") or "").strip()
            if not name:
                sections.append("=== invalid ===\n(missing collection_name)")
                continue
            summarize = entry.get("summarize")
            instructions = entry.get("summary_instructions")
            try:
                content = await self.get_collection_async(
                    name,
                    summarize=summarize if isinstance(summarize, bool) else None,
                    summary_instructions=instructions if isinstance(instructions, str) else None,
                )
            except Exception as e:
                content = f"(error reading collection: {e})"
            sections.append(f"=== {name} ===\n{content}")
        return "\n\n".join(sections)

    def get_collections_snapshot(self) -> str:
        """Render all collections as text for working memory snapshot.

        Always returns a string (never None) so the agent always sees
        the collection inventory — including when no collections exist.
        Per-collection rendering is capped at
        ``self.snapshot_max_items_per_collection`` when set, so large
        collections do not balloon the snapshot every phase.
        """
        if not self.collections:
            return "No memory collections exist yet."
        # Inventory header — compact summary so agents know what's available
        inventory_lines = ["Available collections:"]
        for col in self.collections.values():
            inventory_lines.append(
                f"  - \"{col.name}\": {col.description} ({len(col.items)} items)"
            )
        inventory = "\n".join(inventory_lines)
        # Full content follows (capped per collection when configured)
        max_items = self.snapshot_max_items_per_collection
        parts = [col.to_snapshot(max_items=max_items) for col in self.collections.values()]
        return f"{inventory}\n\n{'—' * 40}\n\n" + "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tool definition builders
# ---------------------------------------------------------------------------

def build_plan_tools(memory: AgentWorkingMemory) -> List[ToolDef]:
    """Build tools available in the Plan phase."""
    return [
        ToolDef(
            name="create_todo_list",
            description=(
                "Create an ordered todo list for your execution plan. "
                "Each item should be a specific, actionable step. "
                "Call this ONCE to define your plan."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string", "description": "What to do"},
                                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            },
                            "required": ["description"],
                        },
                    },
                },
                "required": ["items"],
            },
            handler=lambda items: memory.create_todo_list(items),
            phase_availability=[Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT],
            is_framework_tool=True,
        ),
        ToolDef(
            name="update_todo_list",
            description="Add, remove, or modify items in the todo list.",
            parameters={
                "type": "object",
                "properties": {
                    "add": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            },
                            "required": ["description"],
                        },
                        "description": "Items to add.",
                    },
                    "remove": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Indices of items to remove.",
                    },
                    "modify": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "description": {"type": "string"},
                                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                            },
                            "required": ["index"],
                        },
                        "description": "Items to modify (by index).",
                    },
                },
            },
            handler=lambda add=None, remove=None, modify=None: memory.update_todo_list(add, remove, modify),
            phase_availability=[Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT],
            is_framework_tool=True,
        ),
        ToolDef(
            name="get_todo_list",
            description="View the current todo list with completion statuses.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: memory.get_todo_list(),
            phase_availability=[Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT],
            is_framework_tool=True,
            is_read_only=True,
        ),
    ]


def build_act_tools(memory: AgentWorkingMemory) -> List[ToolDef]:
    """Build tools available in the Act phase (in addition to plan tools that carry over)."""
    return [
        ToolDef(
            name="mark_todo_complete",
            description=(
                "Mark a todo item as completed. Provide the item index and "
                "a brief summary of what was accomplished."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "item_index": {"type": "integer", "description": "Index of the todo item."},
                    "summary": {"type": "string", "description": "Brief summary of completion."},
                },
                "required": ["item_index", "summary"],
            },
            handler=lambda item_index, summary: memory.mark_todo_complete(item_index, summary),
            phase_availability=[Phase.ACT],
            is_framework_tool=True,
            marks_progress=True,
        ),
        ToolDef(
            name="batch_mark_todo_complete",
            description=(
                "Mark multiple todo items as completed in a single call. "
                "Use this when you have finished several items and want to "
                "update them all at once to save iterations."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_index": {"type": "integer", "description": "Index of the todo item."},
                                "summary": {"type": "string", "description": "Brief summary of completion."},
                            },
                            "required": ["item_index", "summary"],
                        },
                        "description": "List of items to mark complete.",
                    },
                },
                "required": ["items"],
            },
            handler=lambda items: memory.batch_mark_todo_complete(items),
            phase_availability=[Phase.ACT],
            is_framework_tool=True,
            marks_progress=True,
        ),
        ToolDef(
            name="log_finding",
            description=(
                "Record a finding from your analysis. Findings are preserved "
                "across phases and used in the report. Use categories to organize "
                "(e.g., 'risk', 'gap', 'recommendation', 'data_point')."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Finding category."},
                    "content": {"type": "string", "description": "The finding content."},
                    "source": {"type": "string", "description": "Where this finding came from."},
                    "confidence": {
                        "type": "string", "enum": ["high", "medium", "low"],
                        "description": "Confidence level.",
                    },
                },
                "required": ["category", "content", "source"],
            },
            handler=lambda category, content, source, confidence="medium": (
                memory.log_finding(category, content, source, confidence)
            ),
            phase_availability=[Phase.ACT, Phase.REPORT],
            is_framework_tool=True,
            marks_progress=True,
        ),
        ToolDef(
            name="batch_log_findings",
            description=(
                "Record multiple findings in a single call. Use this when you "
                "have several findings to log at once to save iterations. "
                "Each finding needs category, content, source, and optional confidence."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string", "description": "Finding category."},
                                "content": {"type": "string", "description": "The finding content."},
                                "source": {"type": "string", "description": "Where this finding came from."},
                                "confidence": {
                                    "type": "string", "enum": ["high", "medium", "low"],
                                    "description": "Confidence level.",
                                },
                            },
                            "required": ["category", "content", "source"],
                        },
                        "description": "List of findings to log.",
                    },
                },
                "required": ["findings"],
            },
            handler=lambda findings: memory.batch_log_findings(findings),
            phase_availability=[Phase.ACT, Phase.REPORT],
            is_framework_tool=True,
            marks_progress=True,
        ),
        ToolDef(
            name="get_findings",
            description="Retrieve all logged findings, optionally filtered by category.",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter.",
                    },
                },
            },
            handler=lambda category=None: memory.get_findings(category),
            phase_availability=[Phase.ACT, Phase.REVIEW, Phase.REPORT],
            is_framework_tool=True,
            is_read_only=True,
        ),
    ]


def build_review_tools(memory: AgentWorkingMemory) -> List[ToolDef]:
    """Build tools available in the Review phase."""
    return [
        ToolDef(
            name="review_checklist",
            description=(
                "Record your evaluation of the work against completion criteria. "
                "Rate each criterion as 'pass', 'partial', or 'fail' with justification."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "criterion": {"type": "string"},
                                "rating": {"type": "string", "enum": ["pass", "partial", "fail"]},
                                "justification": {"type": "string"},
                            },
                            "required": ["criterion", "rating", "justification"],
                        },
                    },
                },
                "required": ["items"],
            },
            handler=lambda items: memory.record_review_checklist(items),
            phase_availability=[Phase.REVIEW],
            is_framework_tool=True,
            marks_progress=True,
        ),
    ]


def build_report_tools(
    memory: AgentWorkingMemory,
    report_template_handler: Optional[Callable] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    default_role: Optional[str] = None,
    default_sub_role: Optional[str] = None,
) -> List[ToolDef]:
    """
    Build tools available in the Report phase.

    Args:
        memory: Agent working memory.
        report_template_handler: Optional callback to fetch report template.
            Signature: (role: str, sub_role: Optional[str]) -> Optional[str]
            Typically wired to DomainAdapter.get_report_template() by the runtime.
            If None, get_report_template returns a sensible default message.
        output_schema: Optional JSON Schema for the report content. When
            provided, the ``submit_report`` tool's parameters will reflect
            this schema so the LLM knows the expected structure.
        default_role: Agent role fallback used when the LLM passes
            an unknown role identifier to get_report_template.
        default_sub_role: Agent sub-role fallback paired with default_role.
    """
    # Build submit_report parameters — use output_schema if provided,
    # otherwise fall back to open-ended additionalProperties.
    if output_schema:
        submit_params = dict(output_schema)  # shallow copy
    else:
        submit_params = {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
            "description": (
                "The structured report object. Pass all report fields "
                "directly as top-level properties."
            ),
        }

    def _resolve_report_template(role: str = "", sub_role: Optional[str] = None) -> str:
        if not report_template_handler:
            return (
                "No report template configured. Use your best judgment to "
                "structure the report based on your findings and the output schema."
            )

        requested_role = (role or "").strip() or (default_role or "")
        requested_sub_role = (
            sub_role
            if (sub_role is not None and str(sub_role).strip())
            else default_sub_role
        )

        template = report_template_handler(requested_role, requested_sub_role)
        if template:
            return template

        # LLMs occasionally pass human-readable labels instead of role IDs.
        # Fall back to the current agent role to avoid losing formatting guidance.
        if default_role and (
            requested_role != default_role or requested_sub_role != default_sub_role
        ):
            fallback = report_template_handler(default_role, default_sub_role)
            if fallback:
                return fallback

        return (
            "No specific report template is defined for this role. "
            "Use your best judgment to structure the report based on your findings "
            "and the output schema."
        )

    tools = [
        ToolDef(
            name="get_report_template",
            description=(
                "Fetch the structural and formatting instructions for your "
                "deliverable. Call this at the start of the report phase to "
                "understand the expected output format."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "description": "Your role identifier.",
                    },
                    "sub_role": {
                        "type": "string",
                        "description": "Optional sub-role identifier.",
                    },
                },
            },
            handler=_resolve_report_template,
            phase_availability=[Phase.REPORT],
            is_framework_tool=True,
        ),
        ToolDef(
            name="submit_report",
            description=(
                "Submit your final deliverable. Pass the report fields directly "
                "as parameters (e.g. title, summary, findings, etc.). "
                "This finalizes your work."
            ),
            parameters=submit_params,
            handler=lambda **kwargs: memory.submit_report(kwargs if kwargs else {}),
            phase_availability=[Phase.REPORT],
            is_framework_tool=True,
            marks_progress=True,
            terminates_phase=True,
        ),
    ]
    return tools


def build_collection_tools(memory: AgentWorkingMemory) -> List[ToolDef]:
    """Build memory collection tools available in all phases."""
    all_phases = [Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT]

    async def _get_collection_async(
        collection_name: str,
        summarize: Optional[bool] = None,
        summary_instructions: Optional[str] = None,
    ) -> str:
        # Real async handler — required because the underlying
        # ``get_collection_async`` is a coroutine. ToolExecutor uses
        # ``inspect.iscoroutinefunction`` to dispatch, so wrapping in
        # a sync lambda would return a raw coroutine object instead
        # of the resolved string.
        return await memory.get_collection_async(
            collection_name,
            summarize=summarize,
            summary_instructions=summary_instructions,
        )

    return [
        ToolDef(
            name="create_collection",
            description=(
                "Create a named memory collection to store structured knowledge. "
                "Collections persist across phases and are visible to subsequent agents. "
                "Use for: limitations, data gaps, decisions, context to pass forward, etc. "
                "Prefer passing initial_items here instead of a separate add_to_collection call."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Collection name (e.g., 'limitations', 'data_gaps', 'decisions').",
                    },
                    "description": {
                        "type": "string",
                        "description": "What this collection contains and when to reference it.",
                    },
                    "extra": {
                        "type": "string",
                        "description": "Optional additional context about the collection's purpose.",
                    },
                    "initial_items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "Item content."},
                                "metadata": {
                                    "type": "object",
                                    "description": "Optional key-value metadata.",
                                    "properties": {},
                                    "additionalProperties": True,
                                },
                            },
                            "required": ["content"],
                        },
                        "description": "Items to populate the collection with immediately.",
                    },
                },
                "required": ["collection_name", "description"],
            },
            handler=lambda collection_name, description, extra="", initial_items=None: memory.create_collection(collection_name, description, extra, initial_items),
            phase_availability=all_phases,
            is_framework_tool=True,
        ),
        ToolDef(
            name="list_collections",
            description="List all available memory collections with their descriptions and item counts.",
            parameters={"type": "object", "properties": {}},
            handler=lambda: memory.list_collections(),
            phase_availability=all_phases,
            is_framework_tool=True,
        ),
        ToolDef(
            name="get_collection",
            description=(
                "Get items from a memory collection.\n\n"
                "Optional summarize-on-read: pass summarize=true to "
                "have the framework return a faithful prose summary "
                "instead of the raw items. The summary preserves the "
                "context you specify in summary_instructions (or the "
                "framework default). Summaries are cached per "
                "collection until items change, so repeated reads in "
                "the same phase do not re-pay the summarizer cost. "
                "When summarize is omitted, the framework auto-summarizes "
                "above an internal size threshold; pass summarize=false "
                "to force the raw read."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string", "description": "Collection name."},
                    "summarize": {
                        "type": "boolean",
                        "description": (
                            "true=force summarize, false=force raw, "
                            "omit=auto (summarize when content is large)."
                        ),
                    },
                    "summary_instructions": {
                        "type": "string",
                        "description": (
                            "Optional directive describing what the "
                            "summary should preserve (e.g. \"keep all "
                            "quantified findings and source citations\")."
                        ),
                    },
                },
                "required": ["collection_name"],
            },
            handler=_get_collection_async,
            phase_availability=all_phases,
            is_framework_tool=True,
            is_read_only=True,
        ),
        ToolDef(
            name="list_collection",
            description=(
                "Cheap inventory view of ONE collection — returns each item's "
                "ID, metadata, and a short content snippet (no full content). "
                "Use this to scan large collections without paying for the "
                "full text, then call get_collection if you need the full "
                "items. Distinct from list_collections (plural), which "
                "lists every collection but no per-item detail."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string", "description": "Collection name."},
                    "snippet_chars": {
                        "type": "integer",
                        "description": (
                            "Max characters of each item's content to "
                            "include in the snippet. 0 = no snippet "
                            "(IDs/metadata only). Default 80."
                        ),
                        "default": 80,
                        "minimum": 0,
                    },
                },
                "required": ["collection_name"],
            },
            handler=lambda collection_name, snippet_chars=80: memory.list_collection_items(
                collection_name, snippet_chars=snippet_chars
            ),
            phase_availability=all_phases,
            is_framework_tool=True,
            is_read_only=True,
        ),
        ToolDef(
            name="add_to_collection",
            description=(
                "Add one or more items to a memory collection. "
                "Each item has content and optional metadata."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string", "description": "Collection name."},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "Item content."},
                                "metadata": {
                                    "type": "object",
                                    "description": "Optional key-value metadata (e.g., severity, status, source).",
                                    "properties": {},
                                    "additionalProperties": True,
                                },
                            },
                            "required": ["content"],
                        },
                    },
                },
                "required": ["collection_name", "items"],
            },
            handler=lambda collection_name, items: memory.add_to_collection(collection_name, items),
            phase_availability=all_phases,
            is_framework_tool=True,
        ),
        ToolDef(
            name="edit_collection_item",
            description="Edit a specific item in a collection by its ID. Only provided fields are updated.",
            parameters={
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string", "description": "Collection name."},
                    "item_id": {"type": "integer", "description": "ID of the item to edit."},
                    "content": {"type": "string", "description": "New content (optional, omit to keep current)."},
                    "metadata": {
                        "type": "object",
                        "description": "Metadata fields to update (merged with existing, optional).",
                        "properties": {},
                        "additionalProperties": True,
                    },
                },
                "required": ["collection_name", "item_id"],
            },
            handler=lambda collection_name, item_id, content=None, metadata=None: memory.edit_collection_item(collection_name, item_id, content, metadata),
            phase_availability=all_phases,
            is_framework_tool=True,
        ),
        ToolDef(
            name="remove_collection_item",
            description="Remove a specific item from a collection by its ID.",
            parameters={
                "type": "object",
                "properties": {
                    "collection_name": {"type": "string", "description": "Collection name."},
                    "item_id": {"type": "integer", "description": "ID of the item to remove."},
                },
                "required": ["collection_name", "item_id"],
            },
            handler=lambda collection_name, item_id: memory.remove_collection_item(collection_name, item_id),
            phase_availability=all_phases,
            is_framework_tool=True,
        ),
    ]


def build_agent_management_tools(available_roles_description: str) -> List[ToolDef]:
    """
    Build orchestrator-intercepted agent management tools.

    These tools have is_orchestrator_tool=True. The ToolExecutor will refuse
    to execute them directly — the orchestrator intercepts and handles them.
    """
    return [
        ToolDef(
            name="spawn_agent",
            description=(
                f"Spawn a new agent to handle a sub-task. "
                f"By default this call BLOCKS until the agent finishes and "
                f"returns its full output inline (no need to call wait_for_agents).\n\n"
                f"{available_roles_description}\n\n"
                f"Provide a clear task description. The spawned agent will go "
                f"through its own Plan/Act/Review/Report lifecycle.\n\n"
                f"Tip: use batch_operations with parallel=true to spawn multiple "
                f"agents concurrently in one call — more efficient than separate spawns."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "Role ID from the catalog."},
                    "sub_role": {"type": "string", "description": "Optional sub-role ID."},
                    "description": {
                        "type": "string",
                        "maxLength": 150,
                        "description": (
                            "One-line summary of what this agent will do. "
                            "Shown in the agent tree for observability. "
                            "Example: 'Assess GDPR Article 32 technical controls'"
                        ),
                    },
                    "task_description": {
                        "type": "string",
                        "description": "Clear description of what the agent should do.",
                    },
                    "plan_fragment": {
                        "type": "string",
                        "description": "Optional: the relevant portion of your plan for this agent.",
                    },
                    "raw_data": {
                        "type": "object",
                        "description": "Optional: structured data the agent needs.",
                        "properties": {},
                        "additionalProperties": True,
                    },
                    "additional_context": {
                        "type": "string",
                        "description": "Optional: any context that helps the agent.",
                    },
                    "errors_to_propagate": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": True,
                        },
                        "description": "Optional: known tool failures to avoid.",
                    },
                    "effort_level": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 4,
                        "description": (
                            "Controls the agent's phase pipeline and review strictness. "
                            "0 (Minimal): Act only — single LLM call. "
                            "1 (Quick): Act → Report. "
                            "2 (Focused): Plan → Act → Report. "
                            "3 (Thorough): full pipeline with lenient review. "
                            "4 (Rigorous): full pipeline with strict review + retries. "
                            "If omitted, inherits from the parent workflow."
                        ),
                    },
                    "review_mode": {
                        "type": "string",
                        "enum": ["thorough", "lenient", "skip"],
                        "description": (
                            "Legacy parameter — use effort_level instead. "
                            "Controls the spawned agent's review phase. "
                            "'thorough' (default): full review, retries on any fail or partial rating. "
                            "'lenient': review runs but only retries if output is fundamentally broken "
                            "(explicit 'fail' ratings); partial gaps are noted but don't trigger retry. "
                            "'skip': no review phase — go straight from act to report. "
                            "Use 'skip' for simple data-gathering tasks, 'lenient' for analysis tasks "
                            "where minor gaps are acceptable, 'thorough' for critical assessments."
                        ),
                    },
                    "blocking": {
                        "type": "boolean",
                        "description": (
                            "If true (default), the call blocks until the spawned "
                            "agent completes and returns its full output inline. "
                            "Set to false to spawn in the background and collect "
                            "results later via wait_for_agents."
                        ),
                        "default": True,
                    },
                },
                "required": ["role", "task_description"],
            },
            handler=None,  # Orchestrator handles this
            phase_availability=[Phase.ACT],
            is_framework_tool=True,
            is_orchestrator_tool=True,
            strip_input_after_dispatch=True,
            heavy_input_fields=[
                "raw_data", "additional_context",
                "errors_to_propagate", "plan_fragment",
            ],
        ),
        ToolDef(
            name="batch_operations",
            description=(
                "Batch multiple tool calls into a single round-trip. "
                "STRONGLY PREFERRED over calling tools one by one — each "
                "separate tool call costs a full LLM iteration.\n\n"
                "The 'op' field is the name of ANY registered tool. The "
                "remaining fields in each operation are passed as that "
                "tool's arguments.\n\n"
                "When spawn_agent ops are included, batch_operations "
                "automatically waits for all spawned agents to finish and "
                "returns their full reports in one tool message. You do NOT "
                "need to call wait_for_agents separately.\n\n"
                "Set parallel=true when ops are independent (e.g., multiple "
                "spawn_agent calls). Leave false (default) when ops have "
                "ordering dependencies.\n\n"
                "Examples:\n"
                "  Memory ops: {op: 'log_findings', findings: [...]}\n"
                "  Spawn:      {op: 'spawn_agent', role: '...', task_description: '...'}\n"
                "  Mixed:      combine any registered tools in one batch"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "description": (
                            "Ordered list of operations. Each must have an "
                            "'op' field naming the tool, plus that tool's "
                            "arguments as sibling fields."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {
                                    "type": "string",
                                    "description": (
                                        "Name of any registered tool to dispatch."
                                    ),
                                },
                            },
                            "required": ["op"],
                            "additionalProperties": True,
                        },
                        "minItems": 1,
                    },
                    "parallel": {
                        "type": "boolean",
                        "description": (
                            "If true, dispatch all ops concurrently. Set to "
                            "true when ops are independent (e.g., multiple "
                            "spawn_agent calls). Leave false (default) when "
                            "ops have ordering dependencies."
                        ),
                        "default": False,
                    },
                },
                "required": ["operations"],
            },
            handler=None,  # Orchestrator handles this
            phase_availability=[Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT],
            is_framework_tool=True,
            is_orchestrator_tool=True,
        ),
        ToolDef(
            name="wait_for_agents",
            description=(
                "Wait for one or more spawned agents to complete and get their results. "
                "Provide the task_ids returned by spawn_agent.\n\n"
                "By default returns a compact `summary` view (status, prose summary, "
                "finding counts, token usage). For deep dives, set "
                "`detail_level='full'` or call `get_agent_phase_output` on a specific "
                "task. The compact view dramatically reduces synthesis cost when you "
                "have many children."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Task IDs of agents to wait for.",
                    },
                    "detail_level": {
                        "type": "string",
                        "enum": ["summary", "full"],
                        "description": (
                            "How much per-agent detail to return. "
                            "'summary' (default) keeps only status, summary prose, "
                            "finding counts, and token usage. 'full' returns the "
                            "complete AgentOutput including full findings dict and "
                            "phase outputs."
                        ),
                        "default": "summary",
                    },
                },
                "required": ["task_ids"],
            },
            handler=None,
            phase_availability=[Phase.ACT],
            is_framework_tool=True,
            is_orchestrator_tool=True,
        ),
        ToolDef(
            name="get_agent_phase_output",
            description=(
                "Fetch the full text output of a specific phase (plan, act, review, "
                "or report) for a single completed child agent. Use this when "
                "wait_for_agents was called with detail_level='summary' and you need "
                "to inspect a particular phase in detail. Cheaper than detail_level="
                "'full' because it loads only one phase for one agent."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Task ID of the child agent.",
                    },
                    "phase": {
                        "type": "string",
                        "enum": ["plan", "act", "review", "report"],
                        "description": "Which phase output to retrieve.",
                    },
                },
                "required": ["task_id", "phase"],
            },
            handler=None,
            phase_availability=[Phase.ACT, Phase.REPORT],
            is_framework_tool=True,
            is_orchestrator_tool=True,
        ),
        ToolDef(
            name="get_agent_result",
            description=(
                "Get the result of a completed spawned agent. "
                "The agent must have finished — use wait_for_agents first if unsure."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID of the agent."},
                },
                "required": ["task_id"],
            },
            handler=None,
            phase_availability=[Phase.ACT, Phase.REPORT],
            is_framework_tool=True,
            is_orchestrator_tool=True,
        ),
        ToolDef(
            name="get_agent_results_all",
            description=(
                "Get the results of ALL spawned child agents. "
                "Returns summaries of every child agent's output."
            ),
            parameters={"type": "object", "properties": {}},
            handler=None,
            phase_availability=[Phase.ACT, Phase.REPORT],
            is_framework_tool=True,
            is_orchestrator_tool=True,
        ),
    ]


def build_coordination_tools() -> List[ToolDef]:
    """
    Build orchestrator-intercepted agent coordination tools.

    These tools enable inter-agent message passing and shared state
    during workflow execution.  Like the agent management tools, they
    have ``is_orchestrator_tool=True`` — the orchestrator intercepts and
    handles them.
    """
    return [
        ToolDef(
            name="send_message",
            description=(
                "Send a message to another agent in the same workflow. "
                "You can message your parent, children, or sibling agents. "
                "Use this to share intermediate findings, request information, "
                "or coordinate work."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "to_task_id": {
                        "type": "string",
                        "description": "Task ID of the recipient agent.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Human-readable message content.",
                    },
                    "message_type": {
                        "type": "string",
                        "enum": ["info", "request", "response", "warning", "data"],
                        "description": (
                            "Type of message. 'info' for general updates, "
                            "'request' to ask for something, 'response' to "
                            "reply, 'warning' for issues, 'data' for structured "
                            "data payloads. Defaults to 'info'."
                        ),
                    },
                    "data": {
                        "type": "object",
                        "description": "Optional: structured data payload.",
                    },
                },
                "required": ["to_task_id", "content"],
            },
            handler=None,
            phase_availability=[Phase.ACT],
            is_framework_tool=True,
            is_orchestrator_tool=True,
        ),
        ToolDef(
            name="read_messages",
            description=(
                "Read messages sent to you by other agents. Returns any new "
                "messages since the last read. Use this to check for updates, "
                "requests, or data from parent/child/sibling agents."
            ),
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=None,
            phase_availability=[Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT],
            is_framework_tool=True,
            is_orchestrator_tool=True,
            is_read_only=True,
        ),
        ToolDef(
            name="set_shared_state",
            description=(
                "Write a key-value pair to the workflow's shared state. "
                "All agents in the workflow can read shared state. "
                "Use this to publish data that multiple agents need."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "State key (e.g. 'risk_findings', 'config.threshold').",
                    },
                    "value": {
                        "description": "Value to store (any JSON-serializable type).",
                    },
                },
                "required": ["key", "value"],
            },
            handler=None,
            phase_availability=[Phase.ACT],
            is_framework_tool=True,
            is_orchestrator_tool=True,
        ),
        ToolDef(
            name="get_shared_state",
            description=(
                "Read from the workflow's shared state. "
                "Omit 'key' to get all shared state, or provide a specific "
                "key to get its value."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Optional: specific key to read. Omit to get all state.",
                    },
                },
            },
            handler=None,
            phase_availability=[Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT],
            is_framework_tool=True,
            is_orchestrator_tool=True,
            is_read_only=True,
        ),
    ]


def build_transition_tools(memory: AgentWorkingMemory) -> List[ToolDef]:
    """Build the set_next_phase tool for agent-controlled phase transitions."""
    return [
        ToolDef(
            name="set_next_phase",
            description=(
                "Control what happens after the current phase. You can transition "
                "to ANY phase:\n"
                "- 'plan': Go back to planning (e.g., after review reveals gaps "
                "  that need a revised plan)\n"
                "- 'act': Go back to execution (e.g., after review identifies "
                "  missing work, or to do additional research)\n"
                "- 'review': Run quality validation before reporting\n"
                "- 'report': Skip directly to final report (e.g., if you "
                "  disagree with a failed review and believe your work is adequate)\n\n"
                "Default transitions: after planning -> execution, "
                "after execution -> reporting.\n"
                "You have full autonomy over phase flow. Use your judgment."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "phase": {
                        "type": "string",
                        "enum": ["plan", "act", "review", "report"],
                        "description": "Which phase to transition to next.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation for this transition.",
                    },
                },
                "required": ["phase"],
            },
            handler=lambda phase, reason="": memory.set_next_phase(phase, reason),
            phase_availability=[Phase.PLAN, Phase.ACT, Phase.REVIEW],
            is_framework_tool=True,
        ),
    ]
