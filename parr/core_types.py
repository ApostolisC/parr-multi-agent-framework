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

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Phase(str, Enum):
    """The four phases every agent passes through."""
    PLAN = "plan"
    ACT = "act"
    REVIEW = "review"
    REPORT = "report"


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
    """Token consumption from a single LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __repr__(self) -> str:
        return (
            f"TokenUsage(in={self.input_tokens}, out={self.output_tokens}, "
            f"cost=${self.total_cost:.4f})"
        )


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

    def to_llm_schema(self) -> Dict[str, Any]:
        """Convert to the schema format sent to the LLM."""
        schema: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        return schema

    def __repr__(self) -> str:
        phases = ",".join(p.value for p in self.phase_availability)
        return f"ToolDef(name={self.name!r}, phases=[{phases}])"


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    """Budget limits for an agent or workflow."""
    max_tokens: Optional[int] = None
    max_cost: Optional[float] = None
    max_duration_ms: Optional[int] = None
    max_agent_depth: int = 3
    max_parallel_agents: int = 3
    max_sub_agents_total: int = 3  # Max total children a single agent can spawn
    inherit_remaining: bool = True


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
    status: str = "in_progress"  # "in_progress" | "replanning"
    type: str = "full_plan"  # "full_plan" | "plan_fragment"


# ---------------------------------------------------------------------------
# Stall detection
# ---------------------------------------------------------------------------

@dataclass
class StallDetectionConfig:
    """Tuneable thresholds for PhaseRunner stall detection."""
    max_consecutive_tool_failures: int = 3
    max_framework_stall_iterations: int = 5
    max_fw_only_consecutive_iterations: int = 10
    max_duplicate_call_iterations: int = 4
    duplicate_call_window: int = 8


# ---------------------------------------------------------------------------
# Agent configuration (what defines an agent)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """LLM model parameters."""
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096


@dataclass
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
    status: str = "completed"  # "completed" | "degraded" | "failed" | "partial"
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

@dataclass
class ModelPricing:
    """Pricing for a single model."""
    input_price_per_1k: float = 0.0
    output_price_per_1k: float = 0.0
    context_window: int = 128000


@dataclass
class CostConfig:
    """Pricing table for all models used in the framework."""
    models: Dict[str, ModelPricing] = field(default_factory=dict)

    def calculate_cost(self, model: str, usage: TokenUsage, strict: bool = False) -> float:
        """Calculate the cost for a given model and token usage.

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
        input_cost = (usage.input_tokens / 1000) * pricing.input_price_per_1k
        output_cost = (usage.output_tokens / 1000) * pricing.output_price_per_1k
        return input_cost + output_cost
