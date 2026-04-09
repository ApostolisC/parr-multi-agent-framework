# CLAUDE.md — PARR Multi-Agent Framework

## Project Overview

**PARR** (Plan-Act-Review-Report) is a Python multi-agent framework with structured lifecycle phases, budget enforcement, and hierarchical agent spawning.

- **Language:** Python 3.10+
- **Package:** `parr` (source in `parr/`)
- **Author:** Apostolis Ch
- **License:** MIT
- **Repo:** https://github.com/ApostolisC/parr-multi-agent-framework

## Architecture

```
parr/
  orchestrator.py      — Workflow management, agent tree, sub-agent spawning
  agent_runtime.py     — Single agent lifecycle execution (configurable phases via PhaseConfig)
  phase_runner.py      — Single-phase loop: LLM -> tools -> repeat
  core_types.py        — Data structures (enums, dataclasses, PhaseConfig)
  protocols.py         — Extension point interfaces (ToolCallingLLM, DomainAdapter, EventSink, DocumentSearchProvider)
  budget_tracker.py    — Token/cost/duration enforcement, pluggable child allocation
  context_manager.py   — Message history, phase message building (custom prompts/sequence support)
  compaction_strategy.py — Pluggable context compaction (soft/hard, token estimation)
  tool_registry.py     — Tool registration + phase filtering
  tool_executor.py     — Dispatch, validation, timeout, retry, middleware chain
  framework_tools.py   — Built-in tools (todos, findings, spawn_agent, set_next_phase) with stall flags
  stall_detector.py    — Pluggable stall detection (read-only, fw-only loop, duplicates)
  output_validator.py  — Pluggable output validation (JsonSchema, Composite, custom)
  agent_coordinator.py — Pluggable agent coordination (messages, shared state)
  persistence.py       — Hierarchical file-system persistence (sync + async I/O)
  event_bus.py         — In-process pub/sub event system (with error callbacks)
  event_types.py       — Event definitions
  trace_store.py       — Append-only execution trace
  adapters/
    llm_adapter.py           — OpenAI/Azure/Anthropic LLM implementations
    llm_rate_limiter.py      — FIFO queue + rate limiting
    domain_adapter.py        — In-memory role catalog (ReferenceDomainAdapter)
    document_search_adapter.py — RAG wrapper
    event_sink_adapter.py    — Logging, WebSocket, Composite event sinks
  config/
    config_loader.py         — YAML -> framework objects
    config_validator.py      — Schema validation
  debug_ui/
    server.py                — HTTP debug server (threading, SSE, static files, REST + management + continue APIs, ~680 lines)
    data_source.py           — UIDataSource protocol + FileSystemDataSource (all data-reading logic, ~510 lines)
    dashboard.py             — PARRDashboard: one-call attach debug UI to running Orchestrator (~190 lines)
    index.html               — Slim HTML shell (~50 lines, loads ES modules, chat input bar)
    __main__.py              — CLI entry point (wires SSEHub, SSEEventSink, FileSystemDataSource, management data, continue_func into server)
    static/
      style.css              — Extracted CSS (~1900 lines, dark theme, user view, chat bar, event stream)
      js/                    — 21 ES modules (state, utils, api, collapse, tabs, renderers, memory, report, output, chat, agent, agent-tree, global-overview, session-detail, sessions, start-form, sse, user-view, user-report, sub-agent-dashboard, main)
  tests/                     — 20 test files, 760 test cases
```

## Key Protocols (Extension Points)

- `ToolCallingLLM` — LLM provider abstraction (protocols.py)
- `DomainAdapter` — Role catalog, domain tools, output schemas (protocols.py)
- `EventSink` — Event emission target (protocols.py)
- `DocumentSearchProvider` — RAG/search abstraction (protocols.py)

## Commands

```bash
# Install
pip install -e ".[openai,anthropic]"

# Run tests
pytest parr/tests/ -v

# Run specific test file
pytest parr/tests/test_budget_tracker.py -v

# Run debug UI
python -m parr.debug_ui.server --persist-dir ./sessions --port 8099
```

## Code Conventions

- Pure Python with type hints throughout
- Async/await for all runtime operations (asyncio)
- Dataclasses for data structures (frozen where appropriate)
- Protocol classes for extension points (runtime_checkable)
- pytest + pytest-asyncio for testing
- No external UI dependencies (vanilla JS debug UI)

## Session Protocol

**CRITICAL: After EVERY prompt/response cycle, Claude MUST:**
1. Update `plan.md` — Mark completed items, update current step, add discovered items
2. Update memory files — Save stable patterns, decisions, and progress
3. Keep CLAUDE.md current if project structure changes

**Use subagents** (Task tool) for parallelizable work. Prefer:
- `Explore` subagent for codebase research
- `general-purpose` subagent for multi-step implementation tasks
- `Plan` subagent for architectural decisions
- Multiple parallel subagents when tasks are independent

## Current State

See `plan.md` for the living plan with current progress and next steps.
See memory files in `C:\Users\Apostolos\.claude\projects\C--Users-Apostolos-Documents-Documents-For-Synchronization-GitHub-parr-multi-agent-framework\memory\` for cross-session knowledge.

## Phase Flow Architecture

Agent-controlled adaptive flow (enabled via `AdaptiveFlowConfig`):
- Entry LLM call gets ALL tools; behavior determines path
- `set_next_phase(phase, reason)` tool lets agent navigate to any of: plan, act, review, report
- Loop prevention: max 3 visits per phase, max 12 total transitions
- Backward transitions supported (e.g., review → plan, review → act)
- Subagent failures classified into 7 types with structured error propagation

## Known Issues & Technical Debt

Tracked in `plan.md` under each work area. Key items:
- Debug UI uses inline onclick handlers (4 window bridges) — could migrate to event delegation
- SSE only available in live mode (with --config-dir); file-only mode still uses polling
- Management APIs (roles/tools/budget/cancel) only available in live mode; file-only mode returns empty data
- Runtime CRUD for roles/tools/budget deferred — would need file-system backed prompts/schemas
