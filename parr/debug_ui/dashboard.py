"""
High-level dashboard for attaching the debug UI to a running Orchestrator.

Usage::

    from parr.debug_ui import PARRDashboard

    orch = Orchestrator(llm=llm, persist_dir="./sessions", ...)
    dashboard = PARRDashboard(orch, port=8080)
    dashboard.start_background()   # non-blocking
    await orch.start_workflow(task="...", role="researcher")
    dashboard.stop()
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Any, Optional

from .data_source import FileSystemDataSource
from .server import DebugServer, SSEHub, SSEEventSink


class PARRDashboard:
    """Attach a browser-based debug UI to a running Orchestrator instance.

    The dashboard extracts configuration (roles, tools, budget) from the
    orchestrator, wires SSE live events via :class:`CompositeEventSink`,
    and serves the debug interface on a background HTTP server.

    Requires the orchestrator to have ``persist_dir`` set so session data
    can be read from disk.
    """

    def __init__(
        self,
        orchestrator: Any,
        host: str = "localhost",
        port: int = 8080,
    ) -> None:
        persist_dir = orchestrator._persist_dir
        if not persist_dir:
            raise ValueError(
                "Orchestrator must have persist_dir set for the debug UI. "
                "Pass persist_dir='./sessions' when creating the Orchestrator."
            )

        # SSE infrastructure
        sse_hub = SSEHub()
        sse_sink = SSEEventSink(sse_hub)
        _wire_sse(orchestrator, sse_sink)

        # Data source + management data
        data_source = FileSystemDataSource(persist_dir)

        # Workflow runner closures
        async def runner(task: str, role: str) -> Any:
            return await orchestrator.start_workflow(task=task, role=role)

        async def continue_runner(task: str, role: str, additional_context: str) -> Any:
            return await orchestrator.start_workflow(
                task=task, role=role, additional_context=additional_context,
            )

        # Build server
        self._server = DebugServer(
            persist_dir=persist_dir,
            host=host,
            port=port,
            workflow_runner=runner,
            available_roles=_extract_available_roles(orchestrator),
            cancel_func=orchestrator.cancel_workflow,
            continue_func=continue_runner,
            role_details=_extract_role_details(orchestrator),
            tool_details=_extract_tool_details(orchestrator),
            budget_config=_extract_budget(orchestrator),
            sse_hub=sse_hub,
            data_source=data_source,
        )
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the dashboard (blocking — does not return until stopped)."""
        self._server.start()

    def start_background(self) -> None:
        """Start the dashboard in a background daemon thread (non-blocking)."""
        self._thread = threading.Thread(
            target=self._server.start,
            daemon=True,
            name="parr-debug-ui",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the dashboard."""
        self._server.stop()

    @property
    def url(self) -> str:
        """The URL where the dashboard is accessible."""
        return f"http://{self._server.host}:{self._server.port}"


# ---------------------------------------------------------------------------
# SSE wiring
# ---------------------------------------------------------------------------

def _wire_sse(orchestrator: Any, sse_sink: SSEEventSink) -> None:
    """Compose SSE sink into the orchestrator's event pipeline.

    Replaces the EventBridge's sink with a :class:`CompositeEventSink`
    that fans out to both the original sink and the SSE sink.
    """
    from parr.adapters.event_sink_adapter import CompositeEventSink

    bridge = orchestrator._event_bridge
    original_sink = bridge._sink
    bridge._sink = CompositeEventSink([original_sink, sse_sink])


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _extract_available_roles(orchestrator: Any) -> list[str]:
    """Extract role names from the orchestrator's domain adapter."""
    adapter = orchestrator._domain_adapter
    if adapter and hasattr(adapter, "_roles"):
        return list(adapter._roles.keys())
    return []


def _extract_role_details(orchestrator: Any) -> list[dict]:
    """Extract JSON-serializable role details from the orchestrator."""
    adapter = orchestrator._domain_adapter
    if not adapter or not hasattr(adapter, "_roles"):
        return []
    roles = []
    for role_name, entry in adapter._roles.items():
        role_info = {
            "name": role_name,
            "description": entry.description,
            "model": entry.config.model,
            "model_config": {
                "temperature": entry.config.model_config.temperature,
                "top_p": entry.config.model_config.top_p,
                "max_tokens": entry.config.model_config.max_tokens,
            },
            "tools": [t.name for t in entry.tools],
            "has_output_schema": entry.output_schema is not None,
            "has_report_template": entry.report_template is not None,
            "sub_roles": [
                {"name": sr_name, "description": sr_entry.description}
                for sr_name, sr_entry in entry.sub_roles.items()
            ],
        }
        roles.append(role_info)
    return roles


def _extract_tool_details(orchestrator: Any) -> list[dict]:
    """Extract JSON-serializable tool metadata from the orchestrator."""
    adapter = orchestrator._domain_adapter
    if not adapter or not hasattr(adapter, "_roles"):
        return []
    seen: set[str] = set()
    tools = []
    for entry in adapter._roles.values():
        for tool in entry.tools:
            if tool.name in seen:
                continue
            seen.add(tool.name)
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "category": tool.category,
                "is_framework_tool": tool.is_framework_tool,
                "is_read_only": tool.is_read_only,
                "phase_availability": [p.value for p in tool.phase_availability],
            })
    return tools


def _extract_budget(orchestrator: Any) -> dict:
    """Extract budget config from the orchestrator as a dict."""
    budget = orchestrator._default_budget
    if budget and dataclasses.is_dataclass(budget):
        return dataclasses.asdict(budget)
    return {}
