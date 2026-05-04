"""
Core Data Structures for the Agentic Multi-Agent Framework.

All types used across the framework are defined here. These are pure data
structures with no framework logic and no external dependencies beyond stdlib.

Every agent instance uses the same code path — differentiation comes from:
- the system prompt injected (AgentConfig)
- the tools made available (via ToolRegistry)
- the input data provided (AgentInput)
- the output schema expected (AgentOutput)
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
import logging
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def normalise_keys(d: dict) -> dict:
    """Normalise camelCase dict keys to snake_case.

    LLMs sometimes emit JavaScript-style keys (``taskDescription``) instead
    of the Python-style keys (``task_description``) that tool schemas expect.
    Only transforms top-level string keys; nested dicts/values are untouched.
    """
    out = {}
    for k, v in d.items():
        snake = _camel_to_snake(k) if isinstance(k, str) else k
        if snake not in out:
            out[snake] = v
    return out


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Phase(str, Enum):
    """The four phases every agent passes through."""
    PLAN = "plan"
    ACT = "act"
    REVIEW = "review"
    REPORT = "report"


class EffortLevel(IntEnum):
    """Controls which phases run and how strict review is.

    Level 0 (Minimal): Act only — single LLM call, no report.
    Level 1 (Quick):   Act → Report.
    Level 2 (Focused): Plan → Act → Report. **Default.**
    Level 3 (Thorough): Plan → Act → Review → Report (lenient review).
    Level 4 (Rigorous): Plan → Act → Review → Report (strict review + retries).
    """
    MINIMAL = 0
    QUICK = 1
    FOCUSED = 2
    THOROUGH = 3
    RIGOROUS = 4


class ReviewMode(str, Enum):
    """How strictly the Review phase evaluates the agent's work."""
    NONE = "none"        # No review phase
    LENIENT = "lenient"  # Partial ratings count as pass
    STRICT = "strict"    # Partial/fail trigger rework


# Effort level → (phases, review_mode, max_review_retries)
_EFFORT_SPECS: Dict[int, tuple] = {
    0: ([Phase.ACT], ReviewMode.NONE, 0),
    1: ([Phase.ACT, Phase.REPORT], ReviewMode.NONE, 0),
    2: ([Phase.PLAN, Phase.ACT, Phase.REPORT], ReviewMode.NONE, 0),
    3: ([Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT], ReviewMode.LENIENT, 0),
    4: ([Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT], ReviewMode.STRICT, 2),
}


def get_effort_spec(
    effort_level: int,
) -> tuple:
    """Return (phases, review_mode, max_review_retries) for an effort level."""
    return _EFFORT_SPECS.get(
        effort_level,
        _EFFORT_SPECS[2],  # default to Focused
    )


class AgentStatus(str, Enum):
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Literal type aliases for status fields
# ---------------------------------------------------------------------------

AgentOutputStatus = Literal["completed", "degraded", "failed", "partial"]
"""Status of an agent's output."""

PlanContextStatus = Literal["in_progress", "replanning"]
"""Status of a plan context."""

PlanContextType = Literal["full_plan", "plan_fragment"]
"""Type of plan context."""

ExecutionPath = Literal["full_workflow", "direct_answer", "adaptive"]
"""How the agent's execution was routed."""

AdaptiveMode = Literal["direct_answer", "light_work", "deep_work"]
"""Which mode the agent selected in adaptive flow."""

MessageType = Literal["info", "request", "response", "warning", "data"]
"""Type of inter-agent message."""

DirectAnswerSchemaPolicy = Literal["enforce", "bypass"]
"""Per-role policy for output schema handling during direct-answer bypass.

- ``"enforce"``: Direct answer must produce schema-compliant JSON output.
- ``"bypass"``:  Direct answer is free-form text (chat-like), schema ignored.

When not set (``None``), the global
``SimpleQueryBypassConfig.force_full_workflow_if_output_schema`` gate applies.
"""

ToolAccessLevel = Literal["none", "visible", "callable"]
"""Three-tier tool access per phase.

- ``"none"``:     Tool hidden entirely (not in schema, not in prompt).
- ``"visible"``:  Tool description shown in prompt (read-only awareness).
- ``"callable"``: Tool shown in LLM function schema and callable.
"""


class ErrorSource(str, Enum):
    TOOL = "tool"
    AGENT = "agent"
    SYSTEM = "system"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PlanStepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ---------------------------------------------------------------------------
# Message types (LLM conversation)
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    tool_call_id: str
    success: bool
    content: str
    error: Optional[str] = None

    def __repr__(self) -> str:
        status = "ok" if self.success else f"err={self.error}"
        return f"ToolResult(id={self.tool_call_id!r}, {status})"


@dataclass
class Message:
    """A single message in the LLM conversation."""
    role: MessageRole
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    # For tool-result messages, reference the tool_call_id
    tool_call_id: Optional[str] = None


@dataclass
class LLMResponse:
    """Response from an LLM chat_with_tools call."""
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    usage: Optional[TokenUsage] = None
    # The raw assistant message to append to conversation history
    raw_message: Optional[Message] = None

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def __repr__(self) -> str:
        tc = len(self.tool_calls) if self.tool_calls else 0
        content_preview = (self.content or "")[:40]
        return f"LLMResponse(tools={tc}, content={content_preview!r})"


@dataclass
class TokenUsage:
    """Token consumption from a single LLM call.

    ``input_tokens`` is the **total** number of input tokens charged for
    the request, regardless of cache state. The cache fields below are
    sub-totals carved out of ``input_tokens`` for cost-accounting:

    - ``cache_creation_input_tokens`` — input tokens that **wrote** to the
      provider's prompt cache (Anthropic explicit cache only). These are
      typically billed at a small premium over the base input price.
    - ``cache_read_input_tokens`` — input tokens **read from** the cache
      (Anthropic ``cache_read_input_tokens`` or Azure/OpenAI
      ``prompt_tokens_details.cached_tokens``). Billed at a steep discount
      (≈10% of base for Anthropic, ≈50% of base for Azure).

    Adapters that don't surface cache metrics leave both fields at 0,
    in which case ``CostConfig`` falls back to the pessimistic
    "all input billed at full price" calculation.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def uncached_input_tokens(self) -> int:
        """Input tokens that were neither written to nor read from the cache.

        This is what you would have been billed at full input price even
        with caching enabled. Useful for the live cost display.
        """
        cached = self.cache_creation_input_tokens + self.cache_read_input_tokens
        return max(0, self.input_tokens - cached)

    def __repr__(self) -> str:
        if self.cache_read_input_tokens or self.cache_creation_input_tokens:
            return (
                f"TokenUsage(in={self.input_tokens} "
                f"[cache_read={self.cache_read_input_tokens}, "
                f"cache_write={self.cache_creation_input_tokens}], "
                f"out={self.output_tokens}, cost=${self.total_cost:.4f})"
            )
        return (
            f"TokenUsage(in={self.input_tokens}, out={self.output_tokens}, "
            f"cost=${self.total_cost:.4f})"
        )


# ---------------------------------------------------------------------------
# Tool lifecycle hooks
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """
    Context passed to tool middleware hooks.

    Provides information about the current execution environment so
    middleware can make decisions based on phase, agent, and call history.
    """
    tool_name: str
    phase: Optional[Phase] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    call_count: int = 0  # How many times this tool has been called in this phase
    metadata: Dict[str, Any] = field(default_factory=dict)  # User-defined data passed between hooks


class ToolMiddleware:
    """
    Base class for tool lifecycle hooks.

    Subclass and override any method to intercept tool execution.
    All methods have sensible defaults (pass-through), so you only
    need to override what you care about.

    Middleware can be attached globally (on ToolExecutor) or per-tool
    (on ToolDef). Global middleware runs first (outer), per-tool
    middleware runs second (inner).

    Example::

        class LoggingMiddleware(ToolMiddleware):
            async def pre_call(self, tool_call, tool_def, context):
                print(f"Calling {tool_call.name} with {tool_call.arguments}")
                return tool_call  # pass through unchanged

            async def post_call(self, result, tool_call, tool_def, context):
                print(f"Result: {result.success}")
                return result  # pass through unchanged

        class CachingMiddleware(ToolMiddleware):
            def __init__(self):
                self._cache = {}

            async def pre_call(self, tool_call, tool_def, context):
                key = (tool_call.name, str(tool_call.arguments))
                if key in self._cache:
                    # Short-circuit: return cached ToolResult directly
                    return self._cache[key]
                return tool_call

            async def post_call(self, result, tool_call, tool_def, context):
                if result.success:
                    key = (tool_call.name, str(tool_call.arguments))
                    self._cache[key] = result
                return result
    """

    async def pre_call(
        self,
        tool_call: ToolCall,
        tool_def: "ToolDef",
        context: ToolContext,
    ) -> Union[ToolCall, ToolResult]:
        """
        Called before tool execution.

        Args:
            tool_call: The tool invocation from the LLM.
            tool_def: The tool definition.
            context: Execution context with phase, agent, and metadata.

        Returns:
            - A ToolCall (possibly modified) to continue execution.
            - A ToolResult to short-circuit execution (skip handler entirely).
        """
        return tool_call

    async def post_call(
        self,
        result: ToolResult,
        tool_call: ToolCall,
        tool_def: "ToolDef",
        context: ToolContext,
    ) -> ToolResult:
        """
        Called after successful tool execution.

        Args:
            result: The result from the handler (or from a pre_call short-circuit).
            tool_call: The original tool invocation.
            tool_def: The tool definition.
            context: Execution context.

        Returns:
            A ToolResult (possibly modified).
        """
        return result

    async def on_error(
        self,
        error: Exception,
        tool_call: ToolCall,
        tool_def: "ToolDef",
        context: ToolContext,
        attempt: int,
        max_attempts: int,
    ) -> Optional[ToolResult]:
        """
        Called when tool execution raises an exception.

        Args:
            error: The exception that was raised.
            tool_call: The original tool invocation.
            tool_def: The tool definition.
            context: Execution context.
            attempt: Current attempt number (0-based).
            max_attempts: Total allowed attempts.

        Returns:
            - None to let the default retry/fail logic proceed.
            - A ToolResult to short-circuit (skip remaining retries).
        """
        return None


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    """
    Definition of a tool that can be made available to agents.

    The handler is the actual Python callable that executes the tool.
    The parameters schema is a JSON Schema dict for the LLM.
    """
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    handler: Optional[Callable] = None
    phase_availability: List[Phase] = field(default_factory=lambda: list(Phase))
    # Phases where this tool's description is shown but not callable.
    # Default (empty): auto-inferred — domain tools callable in ACT are visible in PLAN.
    phase_visibility: List[Phase] = field(default_factory=list)
    mandatory_in_phases: Optional[List[Phase]] = None
    is_framework_tool: bool = False
    is_orchestrator_tool: bool = False  # spawn_agent, wait_for_agents, etc.
    timeout_ms: int = 30000
    max_calls_per_phase: Optional[int] = None
    # Validation & metadata (optional, backward-compatible defaults)
    output_schema: Optional[Dict[str, Any]] = None  # JSON Schema for return values
    category: Optional[str] = None                   # Logical grouping ("retrieval", "persistence")
    retry_on_failure: bool = False                   # Auto-retry on handler exception
    max_retries: int = 0                             # Retry attempts (only if retry_on_failure)
    wraps_untrusted_content: bool = False            # Results contain user-uploaded content
    cacheable: bool = False                          # Opt-in: cache results for identical args (off by default)
    # Stall-detector hints. Read-only tools (e.g., get_*) don't reset the
    # framework-only call counter; progress-marking tools (e.g., log_finding,
    # mark_todo_complete) do. See stall_detector.py for the full classification.
    is_read_only: bool = False
    marks_progress: bool = False
    # Per-tool stripping: after a successful call, heavy input fields are
    # replaced with "<stripped after dispatch>" in the assistant message's
    # tool_call.arguments. This saves tokens on subsequent iterations when
    # the conversation history is re-sent. Per-call override via an
    # auto-injected ``strip_input_after_dispatch`` argument.
    strip_input_after_dispatch: bool = False
    heavy_input_fields: List[str] = field(default_factory=list)
    # Per-tool middleware (logging, caching, redaction, etc.). Combined with
    # any executor-level middleware at call time. See ToolMiddleware in this
    # module and tool_executor._execute_with_middleware.
    middleware: Optional[List["ToolMiddleware"]] = None
    # When True, the phase ends immediately after a successful call to this
    # tool — no further LLM round-trips. Used by submit_report to avoid the
    # wasted "what next?" iteration after report submission.
    terminates_phase: bool = False

    def to_llm_schema(self) -> Dict[str, Any]:
        """Convert to the schema format sent to the LLM."""
        schema: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        return schema

    def to_description_text(self) -> str:
        """Text-only summary for non-callable contexts (visible-only phases)."""
        params = self.parameters.get("properties", {})
        param_names = ", ".join(params.keys()) if params else "none"
        return f"- {self.name}: {self.description} (params: {param_names})"

    def __repr__(self) -> str:
        phases = ",".join(p.value for p in self.phase_availability)
        return f"ToolDef(name={self.name!r}, phases=[{phases}])"


def strip_heavy_fields_in_place(
    args: Dict[str, Any], heavy_fields: List[str],
) -> None:
    """Replace heavy input fields with compact stubs in-place.

    Mutates ``args`` so that the parent agent's conversation history
    (which still references this dict) drops bulky payloads on subsequent
    iterations, saving input tokens.
    """
    for key in heavy_fields:
        if key in args and args[key]:
            args[key] = "<stripped after dispatch>"


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BudgetConfig:
    """Budget limits for an agent or workflow."""
    max_tokens: Optional[int] = None
    max_cost: Optional[float] = None
    max_duration_ms: Optional[int] = None
    max_agent_depth: int = 3
    max_parallel_agents: int = 3
    max_sub_agents_total: int = 3  # Max total children a single agent can spawn
    inherit_remaining: bool = True
    child_budget_fraction: float = 0.5  # Fraction of remaining budget for children
    parent_recovery_budget_pct: float = 0.10  # Fraction of budget reserved for parent recovery after sub-agents
    max_child_review_cycles: Optional[int] = None  # None = use parent's max_review_cycles
    context_soft_compaction_pct: float = 0.78  # Fraction of context for soft compaction (~100K of 128K)
    context_hard_truncation_pct: float = 0.65  # Fraction of context for hard truncation
    chars_per_token: float = 4.0  # Tuneable token estimation ratio (chars / token)
    # Cap on items rendered per memory collection in the working memory
    # snapshot. None = unlimited (legacy behavior). When set, only the
    # most recent N items per collection appear in the snapshot; agents
    # use list_collection / get_collection for the rest.
    snapshot_max_items_per_collection: Optional[int] = None
    # ResultCacheMiddleware: dedup successful read-only tool results to
    # cut cost when an agent re-issues identical reads in a phase.
    # When None, the middleware is not installed (legacy behavior).
    # When set to a non-None value, AgentRuntime installs the cache.
    result_cache_enabled: Optional[bool] = None
    result_cache_scope: str = "phase"  # "phase" or "agent"
    result_cache_max_entries: Optional[int] = 256
    # Summarize-on-read: when get_collection is called without an
    # explicit summarize flag, the framework auto-summarizes if total
    # collection content >= this threshold (chars). 0 disables auto —
    # the agent must pass summarize=True explicitly. Requires a
    # summarizer to be wired through the orchestrator/runtime.
    summarize_on_read_threshold_chars: int = 0
    # Optional default summary directive handed to the summarizer when
    # the tool call does not provide one. Useful for steering the global
    # shape of summaries (e.g. preserve quantified findings & sources).
    default_summary_instructions: Optional[str] = None


@dataclass
class BudgetUsage:
    """Tracked budget consumption."""
    tokens: int = 0
    cost: float = 0.0
    started_at: datetime = field(default_factory=utc_now)

    @property
    def elapsed_ms(self) -> float:
        delta = utc_now() - self.started_at
        return delta.total_seconds() * 1000


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

class SpawnPolicy(str, Enum):
    """How the framework handles same-role spawn attempts."""
    DENY = "deny"        # Hard block — return error
    WARN = "warn"        # Inject warning into result, proceed
    CONSULT = "consult"  # Run consultant agent, obey decision


@dataclass(frozen=True)
class PoliciesConfig:
    """Runtime behaviour policies (loaded from policies.yaml)."""
    same_role_spawn_policy: SpawnPolicy = SpawnPolicy.WARN
    consultant_model: Optional[str] = None   # Model for consult mode (from models.yaml)
    consultant_max_tokens: int = 512
    consultant_temperature: float = 0.1


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------

@dataclass
class ErrorEntry:
    """A recorded error from tool execution, agent failure, or system issue."""
    source: ErrorSource
    name: str
    error_type: str
    message: str
    recoverable: bool = True
    timestamp: datetime = field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# Plan context
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single step in an execution plan."""
    step_id: str = field(default_factory=generate_id)
    description: str = ""
    assigned_agent_role: Optional[str] = None
    status: PlanStepStatus = PlanStepStatus.PENDING
    output_summary: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # step_ids


@dataclass
class PlanContext:
    """Plan information passed to an agent."""
    plan: List[PlanStep] = field(default_factory=list)
    current_agent_assignment: Optional[str] = None
    status: PlanContextStatus = "in_progress"
    type: PlanContextType = "full_plan"


# ---------------------------------------------------------------------------
# Stall detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StallDetectionConfig:
    """Tuneable thresholds for PhaseRunner stall detection."""
    max_consecutive_tool_failures: int = 3
    max_framework_stall_iterations: int = 5
    max_fw_only_consecutive_iterations: int = 10
    max_duplicate_call_iterations: int = 4
    duplicate_call_window: int = 8


@dataclass
class PhaseConfig:
    """Configurable phase lifecycle for an agent.

    Controls which phases run, in what order, their iteration limits,
    custom prompts, and review retry behavior.

    Default: the standard PLAN -> ACT -> REVIEW -> REPORT sequence.

    Example::

        # Skip review:
        PhaseConfig(phases=[Phase.PLAN, Phase.ACT, Phase.REPORT])

        # Custom prompts:
        PhaseConfig(phase_prompts={Phase.ACT: "Custom act instructions..."})

        # No review retry:
        PhaseConfig(max_review_cycles=0)

        # Act-only:
        PhaseConfig(phases=[Phase.ACT])
    """
    phases: List[Phase] = field(
        default_factory=lambda: [Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT]
    )
    phase_limits: Optional[Dict[Phase, int]] = None
    phase_prompts: Optional[Dict[Phase, str]] = None
    max_review_cycles: int = 2
    # Which phase triggers review evaluation. None = auto-detect REVIEW in phases.
    review_phase: Optional[Phase] = None
    # Which phase to re-run on review failure. None = phase before review_phase.
    review_retry_phase: Optional[Phase] = None

    @property
    def effective_review_phase(self) -> Optional[Phase]:
        """The phase that triggers review evaluation."""
        if self.review_phase is not None:
            return self.review_phase if self.review_phase in self.phases else None
        return Phase.REVIEW if Phase.REVIEW in self.phases else None

    @property
    def effective_review_retry_phase(self) -> Optional[Phase]:
        """The phase re-run on review failure (before re-reviewing)."""
        if self.review_retry_phase is not None:
            return self.review_retry_phase if self.review_retry_phase in self.phases else None
        review = self.effective_review_phase
        if review is None:
            return None
        try:
            idx = self.phases.index(review)
        except ValueError:
            return None
        return self.phases[idx - 1] if idx > 0 else None


@dataclass
class OutputValidationResult:
    """Result from output validation.

    Returned by :class:`OutputValidator.validate()`. Multiple validators
    can contribute errors and warnings independently.
    """
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class LLMRateLimitConfig:
    """Global scheduling and throttling controls for LLM calls."""
    enabled: bool = False
    max_concurrent_requests: Optional[int] = None
    max_requests_per_window: Optional[int] = None
    max_tokens_per_window: Optional[int] = None
    window_seconds: float = 60.0
    max_queue_size: Optional[int] = None
    acquire_timeout_seconds: Optional[float] = None


@dataclass(frozen=True)
class SimpleQueryBypassConfig:
    """Runtime policy for bypassing full phases on simple questions."""
    enabled: bool = True
    route_confidence_threshold: float = 0.80
    force_full_workflow_if_output_schema: bool = True
    allow_escalation_to_full_workflow: bool = True
    direct_answer_max_tokens: int = 512


@dataclass(frozen=True)
class AdaptiveFlowConfig:
    """Configuration for agent-controlled adaptive phase flow.

    When enabled, the agent decides its own execution path from the very
    first LLM call. All tools are presented upfront and the agent's
    behavior (tool calls vs text-only) determines whether it enters
    PLAN, ACT, or answers directly.

    PLAN and REVIEW become optional — the agent decides when they're
    needed.  ACT and REPORT remain mandatory for tool-based work.
    """
    enabled: bool = True
    entry_phase_limit: int = 3


# ---------------------------------------------------------------------------
# Agent configuration (what defines an agent)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """LLM model parameters.

    ``prompt_caching`` is an optional per-role override for explicit
    cache markers (e.g. Anthropic ``cache_control``). When None, the LLM
    adapter falls back to its constructor default (set globally via the
    factory). Set ``True`` to force-enable or ``False`` to force-disable
    caching for a specific role / sub-role — useful when one agent's
    prompt changes too often to benefit from caching.
    """
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096
    prompt_caching: Optional[bool] = None


@dataclass(frozen=True)
class AgentConfig:
    """
    What defines an agent instance.

    Differentiation between agents comes exclusively from these fields.
    No special-case agent logic exists anywhere in the framework.
    """
    agent_id: str = field(default_factory=generate_id)
    role: str = ""
    sub_role: Optional[str] = None
    system_prompt: str = ""
    model: str = ""
    model_config: ModelConfig = field(default_factory=ModelConfig)


# ---------------------------------------------------------------------------
# Agent input (what an agent receives)
# ---------------------------------------------------------------------------

@dataclass
class AgentInput:
    """Everything an agent needs to perform its task."""
    task: str
    plan_context: Optional[PlanContext] = None
    trace_snapshot: List[TraceEntry] = field(default_factory=list)
    tools: List[ToolDef] = field(default_factory=list)
    output_schema: Optional[Dict[str, Any]] = None
    raw_data: Optional[Dict[str, Any]] = None
    rag_results: Optional[List[Dict[str, Any]]] = None
    additional_context: Optional[str] = None
    parent_errors: Optional[List[ErrorEntry]] = None
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    effort_level: Optional[int] = None
    # Per-role override for direct-answer JSON-schema enforcement.
    # "enforce" → require structured output even on direct-answer path;
    # "bypass"  → allow free-form text on direct-answer path;
    # None      → fall back to the global SimpleQueryBypassConfig gate.
    direct_answer_schema_policy: Optional[str] = None

    def __repr__(self) -> str:
        task_preview = self.task[:50]
        return f"AgentInput(task={task_preview!r}, tools={len(self.tools)})"


# ---------------------------------------------------------------------------
# Agent output (what every agent must produce)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionMetadata:
    """Metadata about how the agent executed."""
    phases_completed: List[str] = field(default_factory=list)
    iterations_per_phase: Dict[str, int] = field(default_factory=dict)
    sub_agents_spawned: List[str] = field(default_factory=list)  # task_ids
    tools_called: List[Dict[str, Any]] = field(default_factory=list)
    total_duration_ms: float = 0.0
    phase_outputs: Dict[str, str] = field(default_factory=dict)  # phase_name -> LLM text
    execution_path: ExecutionPath = "full_workflow"
    routing_decision: Optional[Dict[str, Any]] = None
    detected_mode: Optional[str] = None


@dataclass
class AgentOutput:
    """
    What every agent must produce.

    Even on failure, timeout, or cancellation, a structured output is produced.
    """
    task_id: str
    agent_id: str
    role: str
    sub_role: Optional[str] = None
    status: AgentOutputStatus = "completed"
    summary: str = ""
    findings: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    errors: List[ErrorEntry] = field(default_factory=list)
    recommendations: Optional[List[str]] = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    execution_metadata: ExecutionMetadata = field(default_factory=ExecutionMetadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "role": self.role,
            "sub_role": self.sub_role,
            "status": self.status,
            "summary": self.summary,
            "findings": self.findings,
            "artifacts": self.artifacts,
            "errors": [
                {"source": e.source.value, "name": e.name,
                 "error_type": e.error_type, "message": e.message}
                for e in self.errors
            ],
            "recommendations": self.recommendations,
            "token_usage": {
                "input_tokens": self.token_usage.input_tokens,
                "output_tokens": self.token_usage.output_tokens,
                "total_tokens": self.token_usage.total_tokens,
                "total_cost": self.token_usage.total_cost,
            },
            "execution_metadata": {
                "phases_completed": self.execution_metadata.phases_completed,
                "iterations_per_phase": self.execution_metadata.iterations_per_phase,
                "sub_agents_spawned": self.execution_metadata.sub_agents_spawned,
                "tools_called": self.execution_metadata.tools_called,
                "total_duration_ms": self.execution_metadata.total_duration_ms,
                "phase_outputs": self.execution_metadata.phase_outputs,
                "execution_path": self.execution_metadata.execution_path,
                "routing_decision": self.execution_metadata.routing_decision,
            },
        }

    def __repr__(self) -> str:
        return (
            f"AgentOutput(task={self.task_id!r}, role={self.role!r}, "
            f"status={self.status!r})"
        )


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

@dataclass
class TraceEntry:
    """
    A single entry in the execution trace.

    The trace is append-only and read-only for agents.
    Only the orchestrator writes to it.
    """
    task_id: str = field(default_factory=generate_id)
    agent_id: str = ""
    role: str = ""
    sub_role: Optional[str] = None
    parent_task_id: Optional[str] = None
    task_description: str = ""
    status: AgentStatus = AgentStatus.RUNNING
    output_summary: Optional[str] = None
    started_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    children: List[str] = field(default_factory=list)  # task_ids


# ---------------------------------------------------------------------------
# Agent tree (orchestrator state)
# ---------------------------------------------------------------------------

@dataclass
class AgentNode:
    """A node in the orchestrator's agent tree."""
    task_id: str = field(default_factory=generate_id)
    agent_id: str = ""
    parent_task_id: Optional[str] = None
    config: AgentConfig = field(default_factory=AgentConfig)
    status: AgentStatus = AgentStatus.RUNNING
    current_phase: Optional[Phase] = None
    children: List[str] = field(default_factory=list)  # task_ids
    result: Optional[AgentOutput] = None
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    budget_consumed: BudgetUsage = field(default_factory=BudgetUsage)
    depth: int = 0
    review_attempts: int = 0
    description: str = ""  # One-line summary provided by parent at spawn time


@dataclass
class AgentMessage:
    """A message sent between agents during workflow execution.

    Used by the coordination system for inter-agent communication.
    Messages can carry both human-readable content and structured data.
    """
    message_id: str = field(default_factory=generate_id)
    from_task_id: str = ""
    to_task_id: str = ""
    content: str = ""
    message_type: MessageType = "info"
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "from_task_id": self.from_task_id,
            "to_task_id": self.to_task_id,
            "content": self.content,
            "message_type": self.message_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class WorkflowExecution:
    """Top-level container for a complete workflow run."""
    workflow_id: str = field(default_factory=generate_id)
    root_task_id: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.RUNNING
    agent_tree: Dict[str, AgentNode] = field(default_factory=dict)
    global_budget: BudgetConfig = field(default_factory=BudgetConfig)
    budget_consumed: BudgetUsage = field(default_factory=BudgetUsage)
    created_at: datetime = field(default_factory=utc_now)


# ---------------------------------------------------------------------------
# Cost configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelPricing:
    """Pricing for a single model.

    ``cached_input_price_per_1k`` is optional and only used when the LLM
    adapter populates ``TokenUsage.cache_read_input_tokens``. If left at
    None, the framework falls back to charging cache-read tokens at the
    full ``input_price_per_1k`` (the safe, pessimistic default — never
    underestimates cost). Anthropic's explicit cache also writes a cache
    entry on the first hit; that write is billed via
    ``cache_write_price_per_1k`` if set, otherwise the full input price.
    """
    input_price_per_1k: float = 0.0
    output_price_per_1k: float = 0.0
    context_window: int = 128000
    cached_input_price_per_1k: Optional[float] = None
    cache_write_price_per_1k: Optional[float] = None


@dataclass
class CostConfig:
    """Pricing table for all models used in the framework."""
    models: Dict[str, ModelPricing] = field(default_factory=dict)

    def calculate_cost(self, model: str, usage: TokenUsage, strict: bool = False) -> float:
        """Calculate the cost for a given model and token usage.

        Cache-aware: when ``usage.cache_read_input_tokens`` or
        ``usage.cache_creation_input_tokens`` are non-zero, those slices
        of the input total are charged at the cached / write price if
        the model has them configured. Tokens not accounted for by either
        cache field are charged at the regular ``input_price_per_1k``.

        Adapters that don't surface cache metrics leave both cache fields
        at 0, in which case this reduces to the original
        ``input_tokens * input_price_per_1k`` calculation — i.e. fully
        backwards-compatible.

        Args:
            model: Model name to look up pricing for.
            usage: Token usage from the LLM call.
            strict: If True, raise ValueError when pricing is missing
                (use when a cost budget is configured so limits are enforced).
        """
        pricing = self.models.get(model)
        if not pricing:
            if strict:
                raise ValueError(
                    f"No pricing data for model '{model}'. "
                    f"Add it to models.yaml or disable cost budgets."
                )
            if not hasattr(self, '_warned_models'):
                self._warned_models: set = set()
            if model not in self._warned_models:
                self._warned_models.add(model)
                logger.warning(
                    "No pricing data for model '%s'. "
                    "Cost tracking will be inaccurate (returning 0.0).",
                    model,
                )
            return 0.0

        # Slice the input total into cached/write/uncached buckets.
        cache_read = max(0, usage.cache_read_input_tokens)
        cache_write = max(0, usage.cache_creation_input_tokens)
        # Cap the buckets so they can never exceed the total (defensive
        # against an adapter that double-counts).
        if cache_read + cache_write > usage.input_tokens:
            overflow = (cache_read + cache_write) - usage.input_tokens
            # Trim the smaller bucket first so cache_read (the larger
            # discount typically) stays accurate.
            if cache_write >= overflow:
                cache_write -= overflow
            else:
                overflow -= cache_write
                cache_write = 0
                cache_read = max(0, cache_read - overflow)
        uncached_in = max(0, usage.input_tokens - cache_read - cache_write)

        # Cached read: use the cached price if present, else fall back
        # to the full input price (pessimistic — never underestimate).
        cached_price = (
            pricing.cached_input_price_per_1k
            if pricing.cached_input_price_per_1k is not None
            else pricing.input_price_per_1k
        )
        # Cache write: use the explicit write price if present, else
        # fall back to the full input price.
        write_price = (
            pricing.cache_write_price_per_1k
            if pricing.cache_write_price_per_1k is not None
            else pricing.input_price_per_1k
        )

        input_cost = (
            (uncached_in / 1000) * pricing.input_price_per_1k
            + (cache_read / 1000) * cached_price
            + (cache_write / 1000) * write_price
        )
        output_cost = (usage.output_tokens / 1000) * pricing.output_price_per_1k
        return input_cost + output_cost
