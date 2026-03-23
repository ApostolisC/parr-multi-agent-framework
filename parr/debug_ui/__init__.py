"""
PARR Debug UI — Lightweight session inspector.

Provides a browser-based debug interface for inspecting persisted
workflow sessions. Reads data from the framework's file-system
persistence layer (``persist_dir``).

Quick Start::

    from parr.debug_ui import start_server
    start_server(persist_dir="./sessions")

With workflow launching::

    async def my_runner(task, role):
        orchestrator = Orchestrator(llm=llm, persist_dir="./sessions")
        return await orchestrator.start_workflow(task=task, role=role)

    start_server(
        persist_dir="./sessions",
        workflow_runner=my_runner,
        available_roles=["researcher", "analyst"],
    )
"""

from .dashboard import PARRDashboard
from .data_source import UIDataSource, FileSystemDataSource
from .server import start_server, DebugServer, SSEHub, SSEEventSink

__all__ = [
    "PARRDashboard",
    "start_server",
    "DebugServer",
    "SSEHub",
    "SSEEventSink",
    "UIDataSource",
    "FileSystemDataSource",
]
