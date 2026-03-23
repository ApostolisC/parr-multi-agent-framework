"""
Debug UI HTTP Server.

Zero-dependency HTTP server that delegates session data reading to a
pluggable :class:`UIDataSource` and serves a single-page debug interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

from .data_source import UIDataSource, FileSystemDataSource

logger = logging.getLogger(__name__)

_HTML_PATH = Path(__file__).parent / "index.html"
_STATIC_DIR = Path(__file__).parent / "static"
_MIME_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}

# Events whose change is visible in the session list sidebar.
_SESSION_LIST_EVENTS = frozenset({
    "agent_started", "agent_completed", "agent_failed", "agent_cancelled",
})


# ---------------------------------------------------------------------------
# SSE Hub — thread-safe manager for Server-Sent Events connections
# ---------------------------------------------------------------------------

class SSEHub:
    """Thread-safe manager for SSE client connections.

    Each connected browser tab gets a bounded ``Queue`` that the
    ``broadcast`` method pushes lightweight event dicts into.  The SSE
    endpoint handler reads from the queue and writes to the HTTP response.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # {client_id: (Queue, workflow_filter|None)}
        self._clients: Dict[str, Tuple[queue.Queue, Optional[str]]] = {}

    def subscribe(self, workflow_id: Optional[str] = None) -> Tuple[str, queue.Queue]:
        """Register a new SSE client.  Returns ``(client_id, queue)``."""
        client_id = f"sse-{uuid.uuid4().hex[:12]}"
        q: queue.Queue = queue.Queue(maxsize=256)
        with self._lock:
            self._clients[client_id] = (q, workflow_id)
        return client_id, q

    def unsubscribe(self, client_id: str) -> None:
        """Remove a disconnected client."""
        with self._lock:
            self._clients.pop(client_id, None)

    def broadcast(self, event_dict: Dict[str, Any]) -> None:
        """Push an event into all matching client queues.

        Clients whose workflow filter does not match are skipped.
        Full queues silently drop the event (the 30 s polling safety net
        ensures the UI catches up).
        """
        wf_id = event_dict.get("workflow_id")
        with self._lock:
            clients = list(self._clients.values())
        for q, wf_filter in clients:
            if wf_filter is not None and wf_filter != wf_id:
                continue
            try:
                q.put_nowait(event_dict)
            except queue.Full:
                pass  # drop — client will catch up via polling

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)


class SSEEventSink:
    """``EventSink``-compatible adapter that bridges framework events to :class:`SSEHub`."""

    def __init__(self, hub: SSEHub) -> None:
        self._hub = hub

    async def emit(self, event: Dict[str, Any]) -> None:
        self._hub.broadcast(event)


# ---------------------------------------------------------------------------
# Background Workflow Runner
# ---------------------------------------------------------------------------

class _WorkflowRunner:
    """Runs async workflow functions in a background thread with its own event loop."""

    def __init__(
        self,
        runner_func: Callable,
        cancel_func: Optional[Callable] = None,
        continue_func: Optional[Callable] = None,
    ) -> None:
        self._runner = runner_func
        self._cancel_func = cancel_func
        self._continue_func = continue_func
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self, task: str, role: str) -> None:
        """Schedule a workflow run (fire-and-forget)."""
        asyncio.run_coroutine_threadsafe(
            self._safe_run(task, role), self._loop,
        )

    def start_with_context(self, task: str, role: str, additional_context: str) -> None:
        """Schedule a workflow with additional_context (fire-and-forget)."""
        if not self._continue_func:
            raise ValueError("No continue_func configured")
        asyncio.run_coroutine_threadsafe(
            self._safe_continue(task, role, additional_context), self._loop,
        )

    def cancel(self, workflow_id: str) -> bool:
        """Cancel a running workflow. Returns True on success."""
        if not self._cancel_func:
            return False
        future = asyncio.run_coroutine_threadsafe(
            self._safe_cancel(workflow_id), self._loop,
        )
        try:
            future.result(timeout=10.0)
            return True
        except Exception:
            return False

    async def _safe_run(self, task: str, role: str) -> None:
        try:
            await self._runner(task, role)
        except Exception as e:
            logger.error(f"Workflow runner failed: {e}", exc_info=True)

    async def _safe_continue(self, task: str, role: str, additional_context: str) -> None:
        try:
            await self._continue_func(task, role, additional_context)
        except Exception as e:
            logger.error(f"Workflow continue failed: {e}", exc_info=True)

    async def _safe_cancel(self, workflow_id: str) -> None:
        try:
            await self._cancel_func(workflow_id)
        except Exception as e:
            logger.error(f"Workflow cancel failed: {e}", exc_info=True)
            raise


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class _DebugHandler(BaseHTTPRequestHandler):
    """Routes requests to the debug UI and API endpoints."""

    # Injected via server class
    data_source: UIDataSource
    workflow_runner: Optional[_WorkflowRunner] = None
    available_roles: List[str] = []
    sse_hub: Optional[SSEHub] = None
    role_details: List[Dict] = []
    tool_details: List[Dict] = []
    budget_config: Dict = {}

    def handle(self) -> None:
        """Wrap default handle to suppress broken-pipe / connection-aborted."""
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # Client disconnected — normal during SSE or page refresh

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug(fmt, *args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, content: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    # -- Routing ------------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "" or path == "/":
            self._serve_index()
        elif path == "/api/sessions":
            self._api_list_sessions()
        elif path.startswith("/api/sessions/"):
            wf_id = path[len("/api/sessions/"):]
            if wf_id:
                self._api_get_session(wf_id)
            else:
                self._send_error_json(400, "Missing workflow_id")
        elif path == "/api/config":
            self._api_get_config()
        elif path == "/api/config/export":
            self._api_config_export()
        elif path == "/api/roles":
            self._api_list_roles()
        elif path.startswith("/api/roles/"):
            self._api_get_role(path[len("/api/roles/"):])
        elif path == "/api/tools":
            self._api_list_tools()
        elif path == "/api/budget":
            self._api_get_budget()
        elif path == "/api/events":
            self._api_sse_stream(parsed.query)
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/"):])
        else:
            self._send_error_json(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/sessions":
            self._api_start_session()
        elif path.startswith("/api/sessions/") and path.endswith("/cancel"):
            wf_id = path[len("/api/sessions/"):-len("/cancel")]
            self._api_cancel_session(wf_id)
        elif path.startswith("/api/sessions/") and path.endswith("/continue"):
            wf_id = path[len("/api/sessions/"):-len("/continue")]
            self._api_continue_session(wf_id)
        else:
            self._send_error_json(404, "Not found")

    # -- Handlers -----------------------------------------------------------

    def _serve_index(self) -> None:
        try:
            content = _HTML_PATH.read_bytes()
            self._send_html(content)
        except FileNotFoundError:
            self._send_error_json(500, "index.html not found")

    def _serve_static(self, rel_path: str) -> None:
        """Serve a file from the static directory with path traversal protection."""
        try:
            target = (_STATIC_DIR / rel_path).resolve()
            if not str(target).startswith(str(_STATIC_DIR.resolve())):
                self._send_error_json(403, "Forbidden")
                return
            if not target.is_file():
                self._send_error_json(404, "Not found")
                return
            content_type = _MIME_TYPES.get(target.suffix, "application/octet-stream")
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        except (FileNotFoundError, OSError):
            self._send_error_json(404, "Not found")

    def _api_list_sessions(self) -> None:
        sessions = self.data_source.list_sessions()
        self._send_json(sessions)

    def _api_get_session(self, workflow_id: str) -> None:
        data = self.data_source.get_session(workflow_id)
        if data is None:
            self._send_error_json(404, f"Session '{workflow_id}' not found")
        else:
            self._send_json(data)

    def _api_get_config(self) -> None:
        self._send_json({
            "can_start": self.workflow_runner is not None,
            "can_cancel": (
                self.workflow_runner is not None
                and self.workflow_runner._cancel_func is not None
            ),
            "can_continue": (
                self.workflow_runner is not None
                and self.workflow_runner._continue_func is not None
            ),
            "available_roles": self.available_roles,
            "sse_available": self.sse_hub is not None,
        })

    # -- Management APIs ----------------------------------------------------

    def _api_cancel_session(self, workflow_id: str) -> None:
        if self.workflow_runner is None:
            self._send_error_json(400, "No workflow_runner configured")
            return
        if not workflow_id:
            self._send_error_json(400, "Missing workflow_id")
            return
        try:
            success = self.workflow_runner.cancel(workflow_id)
            if success:
                self._send_json({"status": "cancelled", "workflow_id": workflow_id})
            else:
                self._send_error_json(400, "Cancellation not available")
        except Exception as e:
            self._send_error_json(500, f"Cancel failed: {e}")

    def _api_list_roles(self) -> None:
        self._send_json(self.role_details)

    def _api_get_role(self, name: str) -> None:
        if not name:
            self._send_error_json(400, "Missing role name")
            return
        for role in self.role_details:
            if role["name"] == name:
                self._send_json(role)
                return
        self._send_error_json(404, f"Role '{name}' not found")

    def _api_list_tools(self) -> None:
        self._send_json(self.tool_details)

    def _api_get_budget(self) -> None:
        self._send_json(self.budget_config)

    def _api_config_export(self) -> None:
        self._send_json({
            "roles": self.role_details,
            "tools": self.tool_details,
            "budget": self.budget_config,
            "available_roles": self.available_roles,
            "sse_available": self.sse_hub is not None,
        })

    # -- SSE Stream ---------------------------------------------------------

    def _sse_write(self, event_name: str, data: str) -> None:
        """Write a single SSE frame to the response."""
        self.wfile.write(f"event: {event_name}\ndata: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _api_sse_stream(self, query_string: str) -> None:
        """Hold the connection open and push events via Server-Sent Events."""
        if self.sse_hub is None:
            self._send_error_json(503, "SSE not available (no event_bus)")
            return

        # Optional workflow_id filter from query string
        params = parse_qs(query_string)
        wf_filter = params.get("workflow_id", [None])[0]

        client_id, q = self.sse_hub.subscribe(workflow_id=wf_filter)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            # Initial connected event
            self._sse_write("connected", json.dumps({"client_id": client_id}))

            while True:
                try:
                    event_dict = q.get(timeout=30)
                except queue.Empty:
                    # Keepalive comment (SSE spec: lines starting with ':')
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue

                event_type = event_dict.get("event_type", "")
                payload = json.dumps({
                    "workflow_id": event_dict.get("workflow_id", ""),
                    "event_type": event_type,
                    "task_id": event_dict.get("task_id", ""),
                    "timestamp": event_dict.get("timestamp", ""),
                }, ensure_ascii=False)

                # Always send session_update (detail view may need refresh)
                self._sse_write("session_update", payload)

                # Lifecycle events also affect the session list sidebar
                if event_type in _SESSION_LIST_EVENTS:
                    self._sse_write("session_list", payload)

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass  # client disconnected
        finally:
            self.sse_hub.unsubscribe(client_id)

    def _api_start_session(self) -> None:
        if self.workflow_runner is None:
            self._send_error_json(400, "No workflow_runner configured")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except (json.JSONDecodeError, ValueError):
            self._send_error_json(400, "Invalid JSON body")
            return

        task = body.get("task", "").strip()
        role = body.get("role", "").strip()

        if not task:
            self._send_error_json(400, "'task' is required")
            return
        if not role:
            self._send_error_json(400, "'role' is required")
            return

        self.workflow_runner.start(task, role)
        self._send_json({"status": "started", "task": task, "role": role})

    def _api_continue_session(self, workflow_id: str) -> None:
        """Continue a completed session with a follow-up message."""
        if self.workflow_runner is None:
            self._send_error_json(400, "No workflow_runner configured")
            return
        if not self.workflow_runner._continue_func:
            self._send_error_json(400, "Continue not available")
            return
        if not workflow_id:
            self._send_error_json(400, "Missing workflow_id")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except (json.JSONDecodeError, ValueError):
            self._send_error_json(400, "Invalid JSON body")
            return

        message = body.get("message", "").strip()
        if not message:
            self._send_error_json(400, "'message' is required")
            return

        # Read previous session to extract role and output
        data = self.data_source.get_session(workflow_id)
        if not data:
            self._send_error_json(404, f"Session '{workflow_id}' not found")
            return

        tree = data.get("agent_tree") or {}
        info = tree.get("info") or {}
        output = tree.get("output") or {}
        memory = tree.get("memory") or {}
        sub_agents = tree.get("sub_agents") or {}

        role = info.get("role", "").strip()
        if not role:
            self._send_error_json(400, "Cannot determine role from previous session")
            return

        # Build additional_context from previous session output
        additional_context = _build_continue_context(info, output, memory, sub_agents)

        self.workflow_runner.start_with_context(message, role, additional_context)
        self._send_json({
            "status": "started",
            "task": message,
            "role": role,
            "continued_from": workflow_id,
        })


def _build_continue_context(info: Dict, output: Dict,
                            memory: Optional[Dict] = None,
                            sub_agents: Optional[Dict] = None) -> str:
    """Build an additional_context string from a previous session's data."""
    original_task = info.get("task", "")
    summary = output.get("summary") or ""
    report = output.get("submitted_report")
    # Findings come from memory (a list), not output (a dict).
    memory = memory or {}
    findings = memory.get("findings") or []
    if not isinstance(findings, list):
        findings = []

    parts = [f"## Previous Session Context\n\nOriginal task: {original_task}"]
    if summary:
        parts.append(f"\nPrevious result summary:\n{summary}")
    if report:
        report_text = (
            report if isinstance(report, str)
            else json.dumps(report, ensure_ascii=False, indent=2)
        )
        # Truncate very long reports to keep context manageable
        if len(report_text) > 4000:
            report_text = report_text[:4000] + "\n... (truncated)"
        parts.append(f"\nPrevious report:\n{report_text}")
    if findings:
        findings_lines = []
        for f in findings[:20]:  # cap at 20 findings
            title = (f.get("category", "") or f.get("title", "")
                     or f.get("content", "") or str(f))
            findings_lines.append(f"- {title}")
        parts.append(f"\nPrevious findings:\n" + "\n".join(findings_lines))

    # Include sub-agent results so retries don't re-do completed work
    if sub_agents:
        sa_parts = []
        for sa_id, sa_data in sub_agents.items():
            sa_info = sa_data.get("info") or {}
            sa_output = sa_data.get("output") or {}
            sa_role = sa_info.get("role", sa_id)
            sa_task = sa_info.get("task", "")
            sa_status = sa_info.get("status", "unknown")
            sa_report = sa_output.get("submitted_report")
            sa_summary = sa_output.get("summary", "")

            sa_section = f"\n### Sub-agent: {sa_role} (status: {sa_status})\nTask: {sa_task}"
            if sa_summary:
                sa_section += f"\nSummary: {sa_summary}"
            if sa_report:
                report_text = (sa_report if isinstance(sa_report, str)
                              else json.dumps(sa_report, ensure_ascii=False, indent=2))
                if len(report_text) > 2000:
                    report_text = report_text[:2000] + "\n... (truncated)"
                sa_section += f"\nReport:\n{report_text}"
            sa_parts.append(sa_section)

        if sa_parts:
            parts.append("\n## Completed Sub-Agent Results\n"
                        "The following sub-agents already completed their work. "
                        "DO NOT re-spawn them \u2014 use their results directly.\n"
                        + "\n".join(sa_parts))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Threading Server
# ---------------------------------------------------------------------------

class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTPServer that handles each request in a new thread.

    Required for SSE: long-lived SSE connections must not block
    normal REST API requests.
    """
    daemon_threads = True


# ---------------------------------------------------------------------------
# Debug Server
# ---------------------------------------------------------------------------

class DebugServer:
    """
    Lightweight HTTP server for inspecting PARR framework sessions.

    Delegates session data reading to a :class:`UIDataSource` and serves
    a browser-based debug interface.
    """

    def __init__(
        self,
        persist_dir: str,
        host: str = "localhost",
        port: int = 8080,
        workflow_runner: Optional[Callable] = None,
        available_roles: Optional[List[str]] = None,
        cancel_func: Optional[Callable] = None,
        continue_func: Optional[Callable] = None,
        role_details: Optional[List[Dict]] = None,
        tool_details: Optional[List[Dict]] = None,
        budget_config: Optional[Dict] = None,
        sse_hub: Optional[SSEHub] = None,
        data_source: Optional[UIDataSource] = None,
    ) -> None:
        self.persist_dir = Path(persist_dir).resolve()
        self.host = host
        self.port = port
        self._workflow_runner = (
            _WorkflowRunner(
                workflow_runner,
                cancel_func=cancel_func,
                continue_func=continue_func,
            )
            if workflow_runner else None
        )
        self._available_roles = available_roles or []
        self._role_details = role_details or []
        self._tool_details = tool_details or []
        self._budget_config = budget_config or {}
        self._sse_hub = sse_hub
        self._data_source = data_source or FileSystemDataSource(self.persist_dir)

        # Ensure persist_dir exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Configure handler class with our state
        handler = type(
            "_BoundHandler",
            (_DebugHandler,),
            {
                "data_source": self._data_source,
                "workflow_runner": self._workflow_runner,
                "available_roles": self._available_roles,
                "role_details": self._role_details,
                "tool_details": self._tool_details,
                "budget_config": self._budget_config,
                "sse_hub": self._sse_hub,
            },
        )
        self._server = _ThreadingHTTPServer((host, port), handler)

    def start(self) -> None:
        """Start the debug server (blocking)."""
        url = f"http://{self.host}:{self.port}"
        print(f"PARR Debug UI running at {url}")
        print(f"Reading sessions from: {self.persist_dir}")
        if self._workflow_runner:
            print(f"Workflow launching: enabled ({len(self._available_roles)} roles)")
        else:
            print("Workflow launching: disabled (no workflow_runner provided)")
        if self._sse_hub:
            print("SSE live events: enabled")
        else:
            print("SSE live events: disabled (no event_bus)")
        print("Press Ctrl+C to stop.\n")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
            self._server.shutdown()

    def stop(self) -> None:
        """Stop the server."""
        self._server.shutdown()


def start_server(
    persist_dir: str,
    host: str = "localhost",
    port: int = 8080,
    workflow_runner: Optional[Callable] = None,
    available_roles: Optional[List[str]] = None,
    cancel_func: Optional[Callable] = None,
    continue_func: Optional[Callable] = None,
    role_details: Optional[List[Dict]] = None,
    tool_details: Optional[List[Dict]] = None,
    budget_config: Optional[Dict] = None,
    sse_hub: Optional[SSEHub] = None,
    data_source: Optional[UIDataSource] = None,
) -> None:
    """
    Start the PARR Debug UI server (blocking).

    Args:
        persist_dir: Path to the framework's persistence directory.
        host: Host to bind to (default: localhost).
        port: Port to listen on (default: 8080).
        workflow_runner: Optional async callable ``(task, role) -> AgentOutput``
            for starting workflows from the UI.
        available_roles: Role names shown in the UI dropdown when
            workflow_runner is provided.
        cancel_func: Optional async callable to cancel a workflow by ID.
        continue_func: Optional async callable
            ``(task, role, additional_context) -> AgentOutput`` for continuing
            a completed session with a follow-up message.
        role_details: Pre-serialized role details for the management API.
        tool_details: Pre-serialized tool details for the management API.
        budget_config: Pre-serialized budget config for the management API.
        sse_hub: Optional :class:`SSEHub` for real-time event push via SSE.
        data_source: Optional :class:`UIDataSource` for reading session data.
            Defaults to :class:`FileSystemDataSource` using ``persist_dir``.
    """
    server = DebugServer(
        persist_dir=persist_dir,
        host=host,
        port=port,
        workflow_runner=workflow_runner,
        available_roles=available_roles,
        cancel_func=cancel_func,
        continue_func=continue_func,
        role_details=role_details,
        tool_details=tool_details,
        budget_config=budget_config,
        sse_hub=sse_hub,
        data_source=data_source,
    )
    server.start()
