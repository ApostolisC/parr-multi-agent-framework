"""
Interface Contracts (Protocols) for the Agentic Framework.

These Protocol classes define the extension points between the framework
and the adapter/application layers. Adapters implement these protocols
to plug domain-specific behavior into the generic framework.

The framework NEVER imports from the adapter. All adapter-specific behavior
flows through these registered interfaces.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from .core_types import (
    AgentConfig,
    AgentOutput,
    LLMResponse,
    Message,
    ModelConfig,
    ToolDef,
)


# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------

@runtime_checkable
class ToolCallingLLM(Protocol):
    """
    Interface for an LLM that supports tool/function calling.

    The adapter wraps the application's LLM factory/provider to implement
    this protocol, normalizing provider-specific tool call formats
    (OpenAI, Anthropic, etc.) into the framework's ToolCall type.
    """

    async def chat_with_tools(
        self,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        model: str,
        model_config: ModelConfig,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        """
        Send messages to the LLM with tool definitions.

        Args:
            messages: Conversation history.
            tools: Tool schemas (from ToolDef.to_llm_schema()).
            model: Model identifier.
            model_config: Temperature, top_p, max_tokens.
            stream: Whether to stream tokens.
            on_token: Callback for streamed tokens (if stream=True).

        Returns:
            LLMResponse containing either text content or tool_calls.
        """
        ...


# ---------------------------------------------------------------------------
# Event Sink
# ---------------------------------------------------------------------------

@runtime_checkable
class EventSink(Protocol):
    """
    Interface for receiving framework events.

    The application layer implements this to wire events to its transport
    (WebSocket, SSE, logging, etc.). The framework publishes events through
    this interface without knowing the transport mechanism.
    """

    async def emit(self, event: Dict[str, Any]) -> None:
        """
        Emit a framework event.

        Args:
            event: Event dict with at minimum:
                - workflow_id: str
                - task_id: str
                - agent_id: str
                - event_type: str
                - timestamp: str (ISO format)
                - data: dict
        """
        ...


# ---------------------------------------------------------------------------
# Domain Adapter
# ---------------------------------------------------------------------------

@runtime_checkable
class DomainAdapter(Protocol):
    """
    Interface for domain-specific configuration.

    The adapter layer implements this to provide role catalogs, domain tools,
    output schemas, and workflow templates specific to the application domain.
    """

    def get_role_config(self, role: str, sub_role: Optional[str] = None) -> AgentConfig:
        """
        Get the agent configuration for a role.

        Resolves system prompt, model, model_config from the domain's
        role catalog (e.g., database tables, config files).

        Args:
            role: Role identifier (e.g., "risk_analyst").
            sub_role: Optional sub-role (e.g., "risk_assessment").

        Returns:
            AgentConfig with system_prompt, model, model_config populated.
        """
        ...

    def get_domain_tools(self, role: str, sub_role: Optional[str] = None) -> List[ToolDef]:
        """
        Get domain-specific tools available to a role.

        These are tools beyond the framework's built-in tools (todo, findings,
        spawn_agent, etc.). Examples: search_documents, get_project_data.

        Args:
            role: Role identifier.
            sub_role: Optional sub-role.

        Returns:
            List of ToolDef with handlers attached.
        """
        ...

    def get_output_schema(self, role: str, sub_role: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the expected output JSON schema for a role.

        The report phase will validate the agent's deliverable against this.

        Args:
            role: Role identifier.
            sub_role: Optional sub-role.

        Returns:
            JSON Schema dict, or None if no schema enforcement.
        """
        ...

    def get_report_template(self, role: str, sub_role: Optional[str] = None) -> Optional[str]:
        """
        Get structural/formatting instructions for the report phase.

        Returns:
            Template string with instructions, or None.
        """
        ...

    def get_direct_answer_schema_policy(
        self, role: str, sub_role: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the direct-answer schema policy for a role.

        Controls how the output schema is handled during direct-answer bypass:
        - ``"enforce"``: Direct answer must produce schema-compliant JSON.
        - ``"bypass"``:  Direct answer is free-form text.
        - ``None``:      Use global ``force_full_workflow_if_output_schema`` gate.

        Args:
            role: Role identifier.
            sub_role: Optional sub-role.

        Returns:
            ``"enforce"``, ``"bypass"``, or ``None``.
        """
        ...

    def list_available_roles(self) -> List[Dict[str, Any]]:
        """
        List all available roles for the spawn_agent tool description.

        Returns:
            List of dicts with 'role', 'sub_roles', 'description' keys.
        """
        ...

    def get_initial_context(self, role: str, sub_role: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get pre-loaded domain data to inject into an agent's first phase.

        Called by the orchestrator for EVERY agent (root and spawned).
        Returns a dict that will be set as AgentInput.raw_data, appearing
        under "## Available Data" in the agent's first user message.

        This eliminates redundant tool calls — agents start with data
        already available instead of fetching it via tools.

        Args:
            role: Role identifier.
            sub_role: Optional sub-role.

        Returns:
            Dict of pre-loaded data keyed by tool name, or None.
        """
        ...

    def persist_output(self, workflow_id: str, agent_output: AgentOutput) -> None:
        """
        Persist the final workflow output to the application's storage.

        Called once per workflow completion by the orchestrator.

        Args:
            workflow_id: Workflow execution ID.
            agent_output: The root agent's final output.
        """
        ...


# ---------------------------------------------------------------------------
# Document Search Provider
# ---------------------------------------------------------------------------

@runtime_checkable
class DocumentSearchProvider(Protocol):
    """
    Interface for document retrieval (RAG).

    The adapter wraps the application's RAG service to implement this,
    enabling agents to search and read documents via tools.
    """

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search documents by semantic query.

        Args:
            query: Natural language search query.
            top_k: Maximum results to return.
            filters: Structured filters (e.g., {"data_category": "health_data"}).

        Returns:
            List of dicts with keys:
                - section_id: str
                - source_file: str
                - section_title: str
                - summary: str (3-5 sentences)
                - relevance_score: float
                - metadata: dict
        """
        ...

    async def get_section(self, section_id: str) -> Dict[str, Any]:
        """
        Get the full text of a specific document section.

        Args:
            section_id: Section identifier from search results.

        Returns:
            Dict with keys:
                - section_id: str
                - source_file: str
                - section_title: str
                - full_text: str
                - metadata: dict
        """
        ...


# ---------------------------------------------------------------------------
# Text Summarizer
# ---------------------------------------------------------------------------

@runtime_checkable
class TextSummarizer(Protocol):
    """
    Interface for a small text summarizer.

    Implementations are typically thin wrappers around a cheap LLM
    (e.g., gpt-4.1-nano) that compress long blocks of text down to a
    short, faithful prose summary. The framework uses this for
    summarize-on-read collection reads — when an agent calls
    ``get_collection(name, summarize=True)``, the framework hands the
    full collection text to the summarizer and returns the summary
    instead of the raw items.

    Implementations MUST be safe to call from a tool handler (no
    blocking I/O on the calling thread, no exceptions on transient
    failures — return None or raise so the framework can fall back
    to the full text).
    """

    async def summarize(
        self,
        text: str,
        instructions: Optional[str] = None,
    ) -> str:
        """
        Produce a faithful, terse prose summary of ``text``.

        Args:
            text: The text to summarize. May be long; the implementation
                is responsible for handling its own context window.
            instructions: Optional caller-supplied directive describing
                what the summary should preserve (e.g. "keep all
                quantified findings and source citations").

        Returns:
            A summary string. The framework treats this as the
            authoritative replacement for the original text — the
            implementation should aim for the smallest summary that
            preserves the context the caller asked for, with no hard
            word cap.
        """
        ...
