"""
Reference Adapters for the Agentic Framework.

Generic, reusable adapter implementations that bridge the framework's
Protocol interfaces to real infrastructure. Any application can use these
adapters by registering its domain-specific data (roles, tools, schemas).

Adapters:
    - OpenAIToolCallingLLM: ToolCallingLLM for OpenAI / Azure OpenAI APIs
    - AnthropicToolCallingLLM: ToolCallingLLM for Anthropic Claude API
    - create_tool_calling_llm: Factory for creating LLM adapters
    - ReferenceDomainAdapter: In-memory DomainAdapter with role registration
    - RAGDocumentSearchAdapter: DocumentSearchProvider wrapping a RAG service
    - LoggingEventSink: EventSink that logs events
    - WebSocketEventSink: EventSink that forwards to WebSocket
    - CompositeEventSink: EventSink that fans out to multiple sinks
"""

from .llm_adapter import (
    AnthropicToolCallingLLM,
    ContentFilterError,
    OpenAIToolCallingLLM,
    create_tool_calling_llm,
)
from .domain_adapter import ReferenceDomainAdapter
from .document_search_adapter import RAGDocumentSearchAdapter
from .event_sink_adapter import (
    CompositeEventSink,
    LoggingEventSink,
    WebSocketEventSink,
)

__all__ = [
    # LLM adapters
    "OpenAIToolCallingLLM",
    "AnthropicToolCallingLLM",
    "ContentFilterError",
    "create_tool_calling_llm",
    # Domain adapter
    "ReferenceDomainAdapter",
    # Document search adapter
    "RAGDocumentSearchAdapter",
    # Event sink adapters
    "LoggingEventSink",
    "WebSocketEventSink",
    "CompositeEventSink",
]
