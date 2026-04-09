"""
Agentic Multi-Agent Framework.

A generic, reusable framework for building multi-agent AI systems.
Domain-agnostic — all domain-specific behavior flows through adapter protocols.

Quick Start:
    from framework import Orchestrator, BudgetConfig, ToolDef

    orchestrator = Orchestrator(llm=my_llm_adapter, event_sink=my_event_sink)
    output = await orchestrator.start_workflow(
        task="Analyze the data...",
        role="analyst",
        system_prompt="You are a data analyst...",
        model="gpt-4o",
        tools=[my_search_tool, my_save_tool],
        budget=BudgetConfig(max_tokens=100000),
    )
"""

# Core types
from .core_types import (
    AdaptiveFlowConfig,
    AgentConfig,
    AgentInput,
    AgentMessage,
    AgentNode,
    AgentOutput,
    AgentOutputStatus,
    AgentStatus,
    BudgetConfig,
    BudgetUsage,
    Confidence,
    CostConfig,
    EffortLevel,
    ErrorEntry,
    ToolAccessLevel,
    ErrorSource,
    ExecutionMetadata,
    ExecutionPath,
    LLMRateLimitConfig,
    LLMResponse,
    Message,
    MessageRole,
    MessageType,
    ModelConfig,
    ModelPricing,
    OutputValidationResult,
    Phase,
    PhaseConfig,
    PlanContext,
    PlanContextStatus,
    PlanContextType,
    PlanStep,
    PlanStepStatus,
    ReviewMode,
    SimpleQueryBypassConfig,
    StallDetectionConfig,
    TokenUsage,
    ToolCall,
    ToolContext,
    ToolDef,
    ToolMiddleware,
    ToolResult,
    TraceEntry,
    WorkflowExecution,
    WorkflowStatus,
    generate_id,
    get_effort_spec,
    utc_now,
)

# Protocols (interfaces for adapters)
from .protocols import (
    DomainAdapter,
    DocumentSearchProvider,
    EventSink,
    ToolCallingLLM,
)

# Orchestrator (main entry point)
from .orchestrator import Orchestrator

# Agent runtime (for advanced usage / testing)
from .agent_runtime import AgentRuntime

# Event system
from .event_bus import EventBus, EventBridge, InMemoryEventSink
from .event_types import FrameworkEvent

# Budget
from .budget_tracker import (
    BudgetExceededException,
    BudgetTracker,
    ChildBudgetAllocator,
    EqualShareAllocator,
    FixedAllocator,
    FractionAllocator,
)

# Trace
from .trace_store import TraceStore

# Persistence
from .persistence import AgentFileStore, WorkflowFileStore

# Tool system
from .tool_registry import ToolRegistry
from .tool_executor import ToolExecutor

# Framework tools (for testing / custom setups)
from .framework_tools import AgentWorkingMemory, MemoryCollection

# Phase runner (for advanced usage)
from .phase_runner import CancelledException, PhaseRunner, PhaseResult

# Stall detection
from .stall_detector import StallDetector, StallVerdict

# Output validation
from .output_validator import (
    CompositeValidator,
    JsonSchemaValidator,
    OutputValidator,
)

# Agent coordination
from .agent_coordinator import AgentCoordinator

# Context manager & compaction
from .context_manager import ContextManager
from .compaction_strategy import CompactionStrategy

# Reference adapters
from .adapters import (
    AnthropicToolCallingLLM,
    CompositeEventSink,
    ContentFilterError,
    LoggingEventSink,
    RateLimitedToolCallingLLM,
    OpenAIToolCallingLLM,
    RAGDocumentSearchAdapter,
    ReferenceDomainAdapter,
    WebSocketEventSink,
    create_tool_calling_llm,
)

# Declarative config
from .config import (
    ConfigBundle,
    ConfigError,
    load_config,
    create_orchestrator_from_config,
)

__all__ = [
    # Entry point
    "Orchestrator",
    # Types
    "AgentConfig",
    "AgentInput",
    "AgentMessage",
    "AgentNode",
    "AgentOutput",
    "AgentOutputStatus",
    "AgentStatus",
    "BudgetConfig",
    "BudgetUsage",
    "Confidence",
    "CostConfig",
    "EffortLevel",
    "ErrorEntry",
    "ErrorSource",
    "ExecutionMetadata",
    "ExecutionPath",
    "LLMRateLimitConfig",
    "LLMResponse",
    "Message",
    "MessageRole",
    "MessageType",
    "ModelConfig",
    "ModelPricing",
    "OutputValidationResult",
    "Phase",
    "PhaseConfig",
    "PlanContext",
    "PlanContextStatus",
    "PlanContextType",
    "PlanStep",
    "PlanStepStatus",
    "ReviewMode",
    "SimpleQueryBypassConfig",
    "StallDetectionConfig",
    "TokenUsage",
    "ToolCall",
    "ToolContext",
    "ToolDef",
    "ToolMiddleware",
    "ToolResult",
    "TraceEntry",
    "WorkflowExecution",
    "WorkflowStatus",
    # Helpers
    "generate_id",
    "get_effort_spec",
    "utc_now",
    # Protocols
    "DomainAdapter",
    "DocumentSearchProvider",
    "EventSink",
    "ToolCallingLLM",
    # Runtime
    "AgentRuntime",
    "PhaseRunner",
    "PhaseResult",
    "CancelledException",
    # Events
    "EventBus",
    "EventBridge",
    "InMemoryEventSink",
    "FrameworkEvent",
    # Budget
    "BudgetExceededException",
    "BudgetTracker",
    "ChildBudgetAllocator",
    "EqualShareAllocator",
    "FixedAllocator",
    "FractionAllocator",
    # Trace
    "TraceStore",
    # Persistence
    "AgentFileStore",
    "WorkflowFileStore",
    # Tools
    "ToolRegistry",
    "ToolExecutor",
    "AgentWorkingMemory",
    "MemoryCollection",
    # Context
    "ContextManager",
    "CompactionStrategy",
    # Adapters
    "OpenAIToolCallingLLM",
    "AnthropicToolCallingLLM",
    "ContentFilterError",
    "RateLimitedToolCallingLLM",
    "create_tool_calling_llm",
    "ReferenceDomainAdapter",
    "RAGDocumentSearchAdapter",
    "LoggingEventSink",
    "WebSocketEventSink",
    "CompositeEventSink",
    # Config
    "ConfigBundle",
    "ConfigError",
    "load_config",
    "create_orchestrator_from_config",
]
