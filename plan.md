# PARR Framework — Master Plan

> **Last updated:** 2026-03-16
> **Status:** Work Area 12 complete. All 8 gaps resolved: flexible phase flow, failure classification, reasoning visibility, review history, live spinner, prompt improvements.

---

## Work Area 12: Flow Integrity & Agent Autonomy — DONE

### Identified Gaps (8 items) → All Resolved

1. **Event ordering broken** → Fixed: `append_tool_calls()` auto-assigns `seq` + `timestamp`; user-view.js sorts by `seq`
2. **Only 1 review shown when 2+ happened** → Fixed: `save_phase_conversation()` stores `_history` list; both UI views show all iterations
3. **No agent reasoning lines** → Fixed: user-view.js reads `llm_calls.json` `response_content`, injects reasoning lines (<200 chars) into event stream
4. **No live "working" indicator** → Fixed: live spinner at bottom of user view when agent is running
5. **Agent doesn't reason about subagent results** → Fixed: ENTRY_PROMPT + ACT prompt have "Sub-agent Management" and "Reasoning visibility" sections
6. **Passive subagent failure handling** → Fixed: `_classify_child_failure()` in orchestrator.py classifies exceptions; failed agents get structured `AgentOutput` with failure_type; recovery notice has actionable guidance
7. **No finding extraction from subagent reports** → Fixed: prompts instruct agent to summarize/interpret subagent results and call log_finding
8. **Rigid phase flow** → Fixed: `set_next_phase` accepts all 4 phases (plan/act/review/report); agent-controlled loop in `_run_adaptive_flow()` with visit counts (max 3/phase, max 12 total)

### Additional UI improvements
- Sub-agent dashboard: 1-box for direct answer agents, 4-box for normal; rerun phases shown in amber
- Debug view: revisit count badge on phases with `_history`; flat view shows all historical visits
- User view: reasoning events, review `_history` iterations, live spinner

### Changes (10 files + 2 new tests)
- [x] `framework_tools.py`: `set_next_phase` validates all 4 phases; `build_transition_tools` includes REVIEW in availability
- [x] `agent_runtime.py`: Rewrote `_run_adaptive_flow()` — agent-controlled loop, `_MAX_PHASE_TRANSITIONS = 12`, visit counts, `_default_next()` + `_resolve_next()` inner functions
- [x] `orchestrator.py`: `_classify_child_failure()` (7 failure types), `_run_child()` creates structured failed output, enriched recovery notices
- [x] `persistence.py`: `save_phase_conversation()` `_history` pattern; `append_tool_calls()` auto `seq`/`timestamp`
- [x] `context_manager.py`: ENTRY_PROMPT "Phase Flow Control" + "Sub-agent Management" sections; ACT "Reasoning visibility" + "Phase flow"; REVIEW backward transition options
- [x] `debug_ui/static/js/user-view.js`: `seq`-based sorting, reasoning events from `llmCalls`, review `_history`, live spinner
- [x] `debug_ui/static/js/sub-agent-dashboard.js`: 1-box direct answer, rerun amber color, fail red
- [x] `debug_ui/static/js/renderers.js`: `_history` support in flat + iteration views, revisit badges
- [x] `debug_ui/static/style.css`: reasoning, spinner, segment variants CSS
- [x] `parr/tests/test_adaptive_flow.py`: Fixed + 2 new tests (all-phases acceptance, invalid phase rejection)
- **760/760 tests green**

---

## Work Area 11: LLM Call Logging + Error Display + Loop Visibility -- DONE

### Problem
Three issues surfaced after real-world testing:
1. No visibility into LLM calls: Debug mode only showed tool calls and final LLM text per phase. What was sent to the LLM and what it responded each iteration was invisible.
2. Errors not shown nicely: Failed tool calls showed a fail badge but no error text. Error banners joined all errors with semicolons. LLM errors (rate limits, connection) were invisible.
3. Post-sub-agent loops: Main agent loops calling read-only framework tools after sub-agents finish. Stall detection exists but warnings weren't visible in the UI.

### Changes (8 files)
- [x] `persistence.py`: `append_llm_calls()` + `async_append_llm_calls()`, `_llm_calls` list for incremental `llm_calls.json` storage
- [x] `phase_runner.py`: `llm_calls` on PhaseResult, `on_llm_call_persisted` callback, per-iteration record building, LLM error capture in try/except, stall warning annotation
- [x] `agent_runtime.py`: `_on_llm_call_persisted` closure, entry LLM call persistence in adaptive flow
- [x] `debug_ui/data_source.py`: Load `llm_calls.json` in `_read_agent_tree()`
- [x] `debug_ui/static/js/renderers.js`: `renderPhases()` accepts `llmCalls`, per-iteration collapsible blocks with token/response/tools, flat fallback for old sessions
- [x] `debug_ui/static/js/agent.js`: Pass `agent.llm_calls` to `renderPhases()`
- [x] `debug_ui/static/js/user-view.js`: Inline error text on failed domain events, structured per-error banner with source tags
- [x] `debug_ui/static/style.css`: CSS for iteration blocks (`.llm-call-block`), error/stall variants, error display classes
- **758/758 tests green**

---

## Work Area 10: Behavioral Fixes + UI Polish + Retry Improvement -- DONE

### Problem
Real-world testing revealed 6 issues:
1. Direct answer when user requests sub-agents ("use 3 subagents" still got direct answer)
2. Sub-agent output too short (2 sentences despite full PARR pipeline)
3. Emoji icons look AI-generated and unprofessional
4. Event ordering wrong (wait shows after dashboard, not before)
5. No live updates during wait (duration label stuck for 8+ minutes)
6. Retry creates new session without sub-agent results (re-does all work)

### Changes (5 files)
- [x] `context_manager.py`: Added delegation rule to ENTRY_PROMPT (forces spawn_agent when user requests delegation). Added depth behavioral guidance to Rules section + ACT prompt (substantive findings, not 2-sentence stubs).
- [x] `debug_ui/static/js/user-view.js`: Removed all emoji icons (domain tools: no icon, spawn: `+`, wait: `…`, finding: `■`, plan: no icon). Fixed event ordering: dashboard renders after last spawn event (not after wait). Indexed loop replaces for-of for event tracking.
- [x] `debug_ui/static/js/session-detail.js`: Live duration timer for running sessions (client-side elapsed time via `Date.now() - created_at`, 1-second interval, auto-cleanup).
- [x] `debug_ui/static/js/state.js`: Added `_durationTimer` field.
- [x] `debug_ui/server.py`: `_build_continue_context()` now accepts `sub_agents` param, recursively extracts role/task/status/summary/report from each sub-agent. Caller passes `tree.sub_agents`. Context includes "DO NOT re-spawn" instruction.
- **758/758 tests green**

---

## Work Area 9: Live UI Overhaul + LLM Retry Hardening -- DONE

### Problem
Real-world testing with 5 concurrent sub-agents revealed:
- All sub-agents failed with OpenAI connection errors (5 retries / 31s total too short)
- Sub-agent ACT phase tools invisible (adaptive flow "entry" phase not mapped)
- Output not chronological (per-phase rendering loses timeline)
- No live progress during wait_for_agents (8+ minutes of silence)
- Too chaotic: fade previews, badges, chevrons, thinking blocks create visual noise
- spawn_agent/wait_for_agents hidden from user view

### Changes (6 files + 1 new file)
- [x] `adapters/llm_adapter.py`: 12 retries (was 5), ~331s total backoff (was 31s), jitter (`delay * random.uniform(0.5, 1.5)`), Retry-After header parsing for 429 responses
- [x] `debug_ui/static/js/utils.js`: `classifyToolForUserView` returns 'spawn'/'wait' instead of 'hidden' for spawn_agent/wait_for_agents
- [x] `debug_ui/static/js/user-view.js`: **Major rewrite** — chronological event stream, `>` one-liners, "entry" phase mapping fix, spawn/wait rendering, sub-agent dashboard integration
- [x] `debug_ui/static/js/sub-agent-dashboard.js`: **New** — live dashboard with 4-segment phase progress bars per sub-agent, role/tools/status display, expandable to full event stream
- [x] `debug_ui/static/js/collapse.js`: Updated selectors from `.uv-block` to `.uv-evt` for collapse state persistence
- [x] `debug_ui/static/style.css`: Replaced `.uv-block-*` container/row/preview/fade with `.uv-evt-*` event stream. Added `.uv-sa-dashboard` with progress bar segments. Kept `.uv-plan-*` and `.uv-answer-*`
- **758/758 tests green**

---

## Work Area 8: Incremental Persistence + Adaptive Flow Fixes -- DONE

### Changes (4 files + 1 new test file)
- [x] Incremental persistence: `on_tool_persisted` callback on PhaseRunner writes tool_calls + memory to disk after each tool call (not just at phase completion)
- [x] Persistence crash fix: `save_memory()` handles `None` attributes with `(getattr(..., None) or [])` pattern
- [x] ENTRY_PROMPT improvements: "Broad questions" section, batch call guidance, "cannot interact with user" rule
- [x] Budget increase: `max_duration_ms` 600000 → 1800000 (30 min)
- [x] 21 new tests in `test_incremental_persistence.py`
- **758/758 tests green**

---

## Work Area 7: Adaptive Agent-Controlled Phase Flow -- DONE

### Problem
The fixed Router → PLAN → ACT → REVIEW → REPORT pipeline wastes tokens: a separate router LLM call makes worse decisions than the agent itself, the agent doesn't know its own tools (searched Weaviate to answer "what tools do you have?"), and all 4 phases run even for simple tasks (12 tool calls, only 2 useful in session 454421ef).

### Solution
Agent-controlled adaptive flow: the agent gets ALL tools from the first LLM call and its behavior determines the execution path.

Three modes (agent decides via behavior):
1. **Direct answer**: Text-only response in first call → done (1 LLM call)
2. **Light work**: Calls domain tools immediately → ACT → REPORT (skip PLAN, REVIEW)
3. **Deep work**: Creates todo list → PLAN → ACT → optionally REVIEW → REPORT

### Changes (12 files + 1 new test file)
- [x] `core_types.py`: `AdaptiveFlowConfig` dataclass, `AdaptiveMode` literal, updated `ExecutionPath` + `ExecutionMetadata.detected_mode`
- [x] `framework_tools.py`: `set_next_phase` tool via `build_transition_tools()`, `AgentWorkingMemory.requested_next_phase`
- [x] `tool_registry.py`: `get_for_entry()` method (excludes review_checklist, submit_report, get_report_template)
- [x] `context_manager.py`: `ENTRY_PROMPT` constant, `build_entry_messages()` method
- [x] `phase_runner.py`: `run_continuation()` method + private continuation params on `run()`
- [x] `agent_runtime.py`: `_run_adaptive_flow()`, `_detect_entry_phase()`, `_build_adaptive_direct_output()`, adaptive branch in `execute()`
- [x] `orchestrator.py`: `adaptive_config` param propagated to AgentRuntime
- [x] `config/config_loader.py`: Parse `adaptive_flow` from YAML, `_build_adaptive_flow_config()`
- [x] `config/config_validator.py`: `_validate_adaptive_flow()`, new param on `validate_config()`
- [x] `__init__.py`: Export `AdaptiveFlowConfig`
- [x] `debug_ui/static/js/output.js`: "adaptive" execution path label + detected_mode badge
- [x] `tests/test_adaptive_flow.py`: 38 tests (config, detection, tools, messages, direct/light/deep work, legacy compat, budget, orchestrator, config validation, exports)
- **737/737 tests green** (699 existing + 38 new)

---

## Work Area 6: Three-Tier Tool Access + Bug Fixes -- DONE

### Problem
Simple knowledge questions (e.g. "what is DPIA?") cost ~$0.05 through the full 4-phase workflow because `force_full_workflow_if_output_schema=True` blocks bypass for any role with an output_schema. The bypass mechanism exists but was never triggered.

### Solution
New per-role YAML field: `direct_answer_schema_policy: "enforce" | "bypass"`

Three modes:
1. **Full workflow** (default/None): global `force_full_workflow_if_output_schema` gate applies (backward compatible)
2. **Direct answer + enforce**: 2 LLM calls, output matches role's JSON schema
3. **Direct answer + bypass**: 2 LLM calls, free-form text (chat-like)

### Changes (9 files + 1 new test file)
- [x] `core_types.py`: Added `DirectAnswerSchemaPolicy` Literal type + `AgentInput.direct_answer_schema_policy` field
- [x] `adapters/domain_adapter.py`: Added fields to `RoleEntry`/`SubRoleEntry`, `register_role`/`register_sub_role` params, `get_direct_answer_schema_policy()` accessor
- [x] `protocols.py`: Added `get_direct_answer_schema_policy()` to DomainAdapter protocol
- [x] `orchestrator.py`: Wired policy resolution in `start_workflow()` and `_handle_orchestrator_tool()`
- [x] `agent_runtime.py`: Replaced blanket gate in `_route_execution_path()`, added `_build_direct_answer_schema_messages()`, `_parse_schema_enforced_result()`, `_parse_bypass_result()`, updated `_build_direct_output()` for schema findings, updated router prompt
- [x] `config/config_loader.py`: Parse `direct_answer_schema_policy` from roles/sub-roles YAML
- [x] `config/config_validator.py`: Validate "enforce"/"bypass" values for roles/sub-roles
- [x] `__init__.py`: Export `DirectAnswerSchemaPolicy`
- [x] `examples/research_assistant/config/roles.yaml`: Added `direct_answer_schema_policy: bypass`
- [x] `tests/test_direct_answer_schema_policy.py`: 19 tests covering backward compat, bypass/enforce modes, escalation, schema validation, sub-role inheritance/override, config validation, router prompt content, findings structure
- **668/668 tests green**

---

## Work Area 1: Framework Pluggability Improvements

### 1.1 Tool Lifecycle Hooks -- DONE
- [x] Added `ToolMiddleware` base class with `pre_call`, `post_call`, `on_error` hooks
- [x] Added `ToolContext` dataclass (phase, agent_id, task_id, call_count, metadata)
- [x] Middleware chain in ToolExecutor: global (outer) + per-tool (inner)
- [x] Pre-call can modify arguments or short-circuit with cached ToolResult
- [x] Post-call can modify results; on_error can recover or let retry proceed
- [x] `ToolDef.middleware` field for per-tool middleware
- [x] `ToolExecutor(middleware=[...])` + `add_middleware()`/`remove_middleware()` for global
- [x] `ToolRegistry.override()` for replacing framework tools
- [x] 34 new tests (test_tool_middleware.py), 412/412 full suite passing
- **Files changed:** `core_types.py`, `tool_executor.py`, `tool_registry.py`, `__init__.py`
- **Files added:** `tests/test_tool_middleware.py`

### 1.2 Pluggable Stall Detection -- DONE
- [x] Added `ToolDef.is_read_only` and `ToolDef.marks_progress` flags to `core_types.py`
- [x] Updated all framework tools in `framework_tools.py` with declarative flags
- [x] Created `StallDetector` class in new `stall_detector.py` with overridable methods
- [x] `StallVerdict` dataclass for check results (is_stalled, reason, should_warn, warning_message)
- [x] Three detection modes: read-only stall, framework-only loop, duplicate call detection
- [x] Legacy fallback via frozen sets for backward compatibility
- [x] Subclass-friendly: override `is_read_only_tool()`, `is_progress_tool()`, `is_domain_tool()`, or entire `check_iteration()`
- [x] Integrated into `PhaseRunner` — accepts optional `StallDetector`, creates default if none
- [x] Removed ~100 lines of inline stall logic from `phase_runner.py`
- [x] 24 new tests (test_stall_detector.py), 436/436 full suite passing
- **Files changed:** `core_types.py`, `framework_tools.py`, `phase_runner.py`, `__init__.py`
- **Files added:** `stall_detector.py`, `tests/test_stall_detector.py`

### 1.3 Pluggable Context Strategy -- DONE
- [x] Created `CompactionStrategy` class in new `compaction_strategy.py` with overridable methods
- [x] Extracted all compaction logic from `ContextManager`: soft/hard compaction, token estimation, message grouping
- [x] Overridable: `estimate_tokens()`, `compact_if_needed()`, `truncate_if_needed()`, `should_preserve_group()`, `summarize_dropped()`
- [x] `ContextManager` accepts optional `compaction_strategy` param, creates default if none
- [x] Full backward compatibility — all existing constructor params and methods still work
- [x] `ContextManager.compaction_strategy` property for access to the strategy
- [x] 26 new tests (test_compaction_strategy.py), 462/462 full suite passing
- **Files changed:** `context_manager.py`, `__init__.py`
- **Files added:** `compaction_strategy.py`, `tests/test_compaction_strategy.py`

### 1.4 Custom Phase Definitions -- DONE
- [x] Added `PhaseConfig` dataclass to `core_types.py` with configurable phases, limits, prompts, and review retry
- [x] `effective_review_phase` / `effective_review_retry_phase` properties for auto-detection
- [x] `ContextManager` accepts `phase_prompts` (custom per-phase LLM instructions) and `phase_sequence` (cross-phase context)
- [x] Custom prompts replace defaults per-phase; unoverridden phases keep defaults
- [x] Sequence-based cross-phase context: each phase gets predecessor's summary
- [x] Refactored `AgentRuntime.execute()` — configurable phase loop replaces hard-coded PLAN→ACT→REVIEW→REPORT
- [x] Extracted `_execute_phase_with_bookkeeping()` and `_run_review_cycle()` helpers
- [x] `AgentRuntime` accepts optional `PhaseConfig`; backward compat via legacy `max_review_cycles`/`phase_limits` params
- [x] `Orchestrator` accepts and passes through `PhaseConfig`; child agents inherit with capped review cycles
- [x] 43 new tests (test_phase_config.py), 505/505 full suite passing
- **Files changed:** `core_types.py`, `context_manager.py`, `agent_runtime.py`, `orchestrator.py`, `__init__.py`
- **Files added:** `tests/test_phase_config.py`

### 1.5 Budget Middleware & Policies -- DONE
- [x] Created `ChildBudgetAllocator` base class with `allocate()` override point
- [x] Extracted current logic into `FractionAllocator` (default, backward-compatible)
- [x] Added `EqualShareAllocator` (divides remaining budget equally among expected children)
- [x] Added `FixedAllocator` (fixed budget per child, ignores parent remaining)
- [x] `BudgetTracker` accepts optional `child_allocator` param, delegates `calculate_child_budget`
- [x] Made `check_limits` public (was `_check_limits`) for subclass overriding
- [x] Documented all BudgetTracker methods as subclass-friendly override points
- [x] `Orchestrator` accepts and passes through `child_allocator` param
- [x] 44 new tests (test_budget_policy.py), 549/549 full suite passing
- **Files changed:** `budget_tracker.py`, `orchestrator.py`, `__init__.py`
- **Files added:** `tests/test_budget_policy.py`

### 1.6 Output Validation Plugging -- DONE
- [x] Added `OutputValidationResult` dataclass to `core_types.py`
- [x] Created `OutputValidator` base class in new `output_validator.py`
- [x] Created `JsonSchemaValidator` (default, extracts existing jsonschema logic)
- [x] Created `CompositeValidator` (chains multiple validators)
- [x] `AgentRuntime` accepts optional `output_validator`, defaults to `JsonSchemaValidator`
- [x] Replaced inline jsonschema call with pluggable validator delegation
- [x] Validation now also surfaces warnings (not just errors) as ErrorEntries
- [x] `Orchestrator` accepts and passes through `output_validator` to all runtimes
- [x] 31 new tests (test_output_validator.py), 580/580 full suite passing
- **Files changed:** `core_types.py`, `agent_runtime.py`, `orchestrator.py`, `__init__.py`
- **Files added:** `output_validator.py`, `tests/test_output_validator.py`

### 1.7 Sub-agent Coordination -- DONE
- [x] Created `AgentMessage` dataclass in `core_types.py` for inter-agent messages
- [x] Created `AgentCoordinator` base class in new `agent_coordinator.py` with overridable methods
- [x] Message passing: `send_message` + `read_messages` tools with cursor-based polling
- [x] Shared state: `set_shared_state` + `get_shared_state` tools with per-workflow key-value store
- [x] Permission model: parent<->child and siblings by default; customizable via subclass
- [x] Hooks: `on_message_sent()` and `on_state_changed()` for logging/side-effects
- [x] `Orchestrator` accepts optional `coordinator` param, creates default `AgentCoordinator` if none
- [x] 4 new orchestrator tools registered in `framework_tools.py` and dispatched in `orchestrator.py`
- [x] 53 new tests (test_agent_coordinator.py), 633/633 full suite passing
- **Files changed:** `core_types.py`, `framework_tools.py`, `orchestrator.py`, `agent_runtime.py`, `__init__.py`
- **Files added:** `agent_coordinator.py`, `tests/test_agent_coordinator.py`

---

## Work Area 2: Bug Fixes & Hardening

### 2.1 Silent Failure Fixes -- DONE
- [x] `persist_output` errors surfaced as `ErrorEntry` in `output.errors` (instead of silent logging)
- [x] `persist_workflow_status` errors surfaced the same way
- [x] EventBus `on_handler_error` callback for handler exception propagation
- **Files changed:** `orchestrator.py`, `event_bus.py`

### 2.2 Race Condition in Sub-agent Cleanup -- DONE
- [x] Rewrote `_cancel_pending_tasks()` to loop until stable (max 5 rounds)
- [x] Handles tasks spawned during cancellation; removes processed tasks individually
- **Files changed:** `orchestrator.py`

### 2.3 Async File I/O -- DONE
- [x] Added `_awrite_json` / `_aread_json` module-level async helpers using `asyncio.to_thread()`
- [x] Added 8 `async_*` method variants to `AgentFileStore`
- [x] Added 2 `async_*` method variants to `WorkflowFileStore`
- [x] Backward-compatible: sync methods unchanged, async variants added alongside
- **Files changed:** `persistence.py`

### 2.4 Type Safety Improvements -- DONE
- [x] Added 5 `Literal` type aliases: `AgentOutputStatus`, `PlanContextStatus`, `PlanContextType`, `ExecutionPath`, `MessageType`
- [x] Applied Literal types to `AgentOutput.status`, `PlanContext.status`, `PlanContext.type`, `ExecutionMetadata.execution_path`, `AgentMessage.message_type`
- [x] Exported all type aliases from `parr/__init__.py`
- **Files changed:** `core_types.py`, `__init__.py`

### 2.5 Missing Integration Tests -- DONE
- [x] Full workflow test: Orchestrator -> AgentRuntime -> PhaseRunner (3 tests)
- [x] Budget exhaustion mid-phase test (1 test)
- [x] Error injection tests — LLM failures and tool crashes (2 tests)
- [x] Silent failure fix validation (2 tests)
- [x] Race condition fix validation (2 tests)
- [x] Async file I/O validation (3 tests)
- [x] Type safety validation (3 tests)
- [x] 16 new tests total, 649/649 full suite passing
- **Files added:** `tests/test_integration.py`

---

## Work Area 3: UI Transformation

### Phase 1 — Modularize Existing UI -- DONE
- [x] Extracted CSS into `static/style.css` (~950 lines)
- [x] Split JS into 14 ES modules in `static/js/` (state, utils, api, collapse, tabs, renderers, memory, report, output, chat, agent, session-detail, sessions, start-form, main)
- [x] Native ES modules via `<script type="module">` (no bundler)
- [x] Centralized mutable state in `state.js`; window bridges for inline onclick handlers
- [x] Added static file serving to `server.py` (`_serve_static` with path traversal protection)
- [x] `index.html` reduced from 2666 lines to 44 lines
- [x] 649/649 tests green (no regressions)
- **Files changed:** `debug_ui/index.html`, `debug_ui/server.py`
- **Files created:** `debug_ui/static/style.css`, `debug_ui/static/js/` (14 modules)

### Phase 2 — Real-time Events (SSE) -- DONE
- [x] `SSEHub` class: thread-safe client connection manager with bounded queues (256)
- [x] `SSEEventSink` class: implements `EventSink` protocol, bridges to SSEHub
- [x] `_ThreadingHTTPServer`: enables concurrent SSE + REST connections
- [x] `/api/events` SSE endpoint with keepalive, workflow filtering, event classification
- [x] Event classification: lifecycle events → `session_list` + `session_update`, others → `session_update` only
- [x] `DebugServer` accepts optional `sse_hub` param; `/api/config` reports `sse_available`
- [x] `__main__.py` creates SSEHub + SSEEventSink, wires into Orchestrator via `event_sink`
- [x] Frontend `sse.js` module: EventSource with auto-reconnect, named event listeners
- [x] SSE-aware polling: 30s safety net when SSE active, 2-3s fallback when not
- [x] Dual mode: live mode (with EventBus) enables SSE, file-only mode keeps polling
- [x] 649/649 tests green (no regressions)
- **Files changed:** `debug_ui/server.py`, `debug_ui/__main__.py`, `debug_ui/__init__.py`, `static/js/state.js`, `static/js/main.js`, `static/js/sessions.js`
- **Files created:** `static/js/sse.js`

### Phase 3 — Write APIs (Management) -- DONE
- [x] POST `/api/sessions/{id}/cancel` — Cancel running workflows (async bridge to Orchestrator)
- [x] GET `/api/roles` — List roles with config details (model, tools, sub-roles)
- [x] GET `/api/roles/{name}` — Single role detail
- [x] GET `/api/tools` — List registered tools with metadata (phase availability, read-only, category)
- [x] GET `/api/budget` — Get budget configuration (all BudgetConfig fields)
- [x] GET `/api/config/export` — Full config snapshot as JSON
- [x] GET `/api/config` updated with `can_cancel` flag
- [x] Cancel button in session detail header (visible when status=running)
- [x] _WorkflowRunner extended with cancel support (async scheduling via event loop)
- [x] Serialization helpers: `_serialize_role_details`, `_serialize_tool_details`, `_serialize_budget`
- [x] 649/649 tests green (no regressions)
- **Deferred:** Runtime CRUD for roles/tools/budget, YAML import (too complex for debug UI)
- **Files changed:** `debug_ui/server.py`, `debug_ui/__main__.py`, `static/js/sessions.js`, `static/js/session-detail.js`, `static/js/main.js`, `static/style.css`

### Phase 4 — Data Source Adapter -- DONE
- [x] `UIDataSource` protocol (runtime_checkable, `list_sessions()` + `get_session()`)
- [x] `FileSystemDataSource` — reads from persist_dir (current behavior, extracted from server.py)
- [x] Extracted ~437 lines of data-reading functions from server.py into data_source.py
- [x] Server handler uses `data_source: UIDataSource` (replaces `persist_dir: Path`)
- [x] `DebugServer` and `start_server` accept optional `data_source` param (defaults to FileSystemDataSource)
- [x] `__main__.py` creates FileSystemDataSource, passes to start_server
- [x] `__init__.py` exports UIDataSource, FileSystemDataSource
- [x] server.py: 1005 → 571 lines; data_source.py: 513 lines
- [x] 649/649 tests green (no regressions)
- **Files changed:** `debug_ui/server.py`, `debug_ui/__main__.py`, `debug_ui/__init__.py`, `tests/test_debug_ui_server_metrics.py`
- **Files created:** `debug_ui/data_source.py`

### Phase 5 — Dual Mode Packaging -- DONE
- [x] `PARRDashboard` class — attach debug UI to a running Orchestrator with one call
- [x] Library mode: `from parr.debug_ui import PARRDashboard` (extracts roles/tools/budget/events from Orchestrator)
- [x] `start()` (blocking), `start_background()` (daemon thread), `stop()`, `url` property
- [x] SSE wiring via `CompositeEventSink` — composites original sink with SSEEventSink on `EventBridge._sink`
- [x] Data extraction helpers: `_extract_role_details`, `_extract_tool_details`, `_extract_budget`, `_extract_available_roles`
- [x] Graceful handling when Orchestrator lacks domain_adapter or budget (returns empty data)
- [x] ValueError when Orchestrator has no `persist_dir`
- [x] `parr-ui` console script entry point in `pyproject.toml`
- [x] 649/649 tests green (no regressions)
- **Files created:** `debug_ui/dashboard.py` (~175 lines)
- **Files changed:** `debug_ui/__init__.py`, `pyproject.toml`

### Phase 6 — Enhanced Visualizations (Focused) -- DONE
- [x] Agent tree visualization — CSS-based hierarchy diagram with connector lines, clickable nodes
- [x] Global overview panel — token/cost breakdown by agent + tool distribution bar charts
- [x] Session search/filter — real-time substring matching on role, task, status, workflow_id
- [x] 649/649 tests green (no regressions)
- **Files created:** `debug_ui/static/js/agent-tree.js`, `debug_ui/static/js/global-overview.js`
- **Files changed:** `debug_ui/static/style.css`, `debug_ui/static/js/session-detail.js`, `debug_ui/static/js/sessions.js`, `debug_ui/static/js/state.js`, `debug_ui/static/js/main.js`, `debug_ui/index.html`

---

## Completed Work

### Analysis Phase (2026-03-13)
- [x] Full framework architecture analysis
- [x] All source files reviewed (7,244 LOC core, 16 test files)
- [x] Protocol/pluggability assessment
- [x] Persistence layer analysis
- [x] Error handling audit
- [x] Configuration system review
- [x] Type system analysis
- [x] Test coverage assessment
- [x] Full UI analysis (debug_ui/index.html, debug_ui/server.py)
- [x] UI transformation feasibility assessment
- [x] Created CLAUDE.md, plan.md, and memory files

### Work Area 1.1 Implementation (2026-03-13)
- [x] `ToolMiddleware` base class with `pre_call`, `post_call`, `on_error`
- [x] `ToolContext` dataclass for middleware execution context
- [x] Global + per-tool middleware chain in `ToolExecutor`
- [x] `ToolRegistry.override()` for framework tool replacement
- [x] 34 new tests, 412/412 full suite green
- [x] Updated plan.md, CLAUDE.md, memory files

### Work Area 1.2 Implementation (2026-03-13)
- [x] `ToolDef.is_read_only` and `ToolDef.marks_progress` declarative flags
- [x] All 8 framework tools annotated with flags in `framework_tools.py`
- [x] `StallDetector` class with overridable classification methods
- [x] `StallVerdict` dataclass for structured stall check results
- [x] Legacy frozen-set fallback for backward compatibility
- [x] Integrated into `PhaseRunner`, removed ~100 lines of inline logic
- [x] 24 new tests, 436/436 full suite green
- [x] Updated plan.md, CLAUDE.md, memory files

### Work Area 1.3 Implementation (2026-03-13)
- [x] `CompactionStrategy` class with overridable compaction methods
- [x] Extracted all compaction logic from `ContextManager` into the strategy
- [x] Overridable: estimate_tokens, compact_if_needed, truncate_if_needed, should_preserve_group, summarize_dropped
- [x] `ContextManager` delegates to strategy, full backward compat
- [x] 26 new tests, 462/462 full suite green
- [x] Updated plan.md, CLAUDE.md, memory files

### Work Area 1.4 Implementation (2026-03-13)
- [x] `PhaseConfig` dataclass with configurable phases, limits, prompts, and review retry
- [x] `effective_review_phase` / `effective_review_retry_phase` auto-detection properties
- [x] `ContextManager` accepts `phase_prompts` and `phase_sequence` for custom phase support
- [x] Refactored `AgentRuntime.execute()` to iterate over configurable phase list
- [x] Extracted `_execute_phase_with_bookkeeping()` and `_run_review_cycle()` helpers
- [x] `Orchestrator` passes through `PhaseConfig` to child agents
- [x] 43 new tests, 505/505 full suite green
- [x] Updated plan.md, CLAUDE.md, memory files

### Work Area 1.5 Implementation (2026-03-13)
- [x] `ChildBudgetAllocator` base class with `allocate()` override point
- [x] `FractionAllocator` (default): extracts current fraction-based allocation with recovery reserve
- [x] `EqualShareAllocator`: divides remaining budget equally among expected children
- [x] `FixedAllocator`: fixed budget per child, ignores parent's remaining budget
- [x] `BudgetTracker` accepts optional `child_allocator`, delegates `calculate_child_budget`
- [x] `check_limits` made public for subclass overriding (was `_check_limits`)
- [x] All BudgetTracker methods documented as subclass-friendly override points
- [x] `Orchestrator` accepts and passes through `child_allocator`
- [x] 44 new tests, 549/549 full suite green
- [x] Updated plan.md, CLAUDE.md, memory files

### Work Area 1.6 Implementation (2026-03-13)
- [x] `OutputValidationResult` dataclass in `core_types.py` (is_valid, errors, warnings)
- [x] `OutputValidator` base class with `validate()` override point in new `output_validator.py`
- [x] `JsonSchemaValidator` (default): extracts existing jsonschema validation logic
- [x] `CompositeValidator`: chains multiple validators, collects all errors/warnings
- [x] `AgentRuntime` accepts `output_validator`, replaces inline jsonschema with delegation
- [x] `Orchestrator` passes `output_validator` to all AgentRuntime instances
- [x] 31 new tests, 580/580 full suite green
- [x] Updated plan.md, CLAUDE.md, memory files

### Work Area 1.7 Implementation (2026-03-13)
- [x] `AgentMessage` dataclass in `core_types.py` (message_id, from/to task_ids, content, type, data, timestamp)
- [x] `AgentCoordinator` base class with overridable permission checks and hooks
- [x] Per-agent mailboxes with cursor-based polling (non-destructive reads)
- [x] Per-workflow shared state (key-value store with access control)
- [x] 4 new orchestrator tools: `send_message`, `read_messages`, `set_shared_state`, `get_shared_state`
- [x] Read tools available in all phases; write tools ACT-only
- [x] `Orchestrator` accepts `coordinator` param, delegates all coordination to it
- [x] 53 new tests, 633/633 full suite green
- [x] Updated plan.md, CLAUDE.md, memory files

### Work Area 2 Implementation (2026-03-13)
- [x] 2.1: `persist_output` errors surfaced as ErrorEntry (not silently swallowed); EventBus `on_handler_error` callback
- [x] 2.2: `_cancel_pending_tasks()` rewrote to loop until stable (max 5 rounds, handles mid-cancellation spawns)
- [x] 2.3: Async file I/O via `asyncio.to_thread()` — 8 async methods on AgentFileStore, 2 on WorkflowFileStore
- [x] 2.4: 5 Literal type aliases (AgentOutputStatus, PlanContextStatus, PlanContextType, ExecutionPath, MessageType)
- [x] 2.5: 16 integration tests (full workflow, budget exhaustion, error injection, silent failure, race condition, async I/O, type safety)
- [x] 649/649 full suite green
- [x] Updated plan.md, CLAUDE.md, memory files

---

### Work Area 2 Completed
- 2.1–2.5 all done. 16 new integration tests. 649/649 full suite green.

### Work Area 3 Phase 1 Implementation (2026-03-13)
- [x] Extracted CSS to `static/style.css` (~950 lines, verbatim)
- [x] Created 14 ES modules: state.js, utils.js, api.js, collapse.js, tabs.js, renderers.js, memory.js, report.js, output.js, chat.js (~500 lines, kept unified for mutual recursion), agent.js, session-detail.js, sessions.js, start-form.js, main.js
- [x] Centralized globals into `state` object exported from state.js
- [x] Window bridges for 3 inline onclick handlers (selectSession, toggleToolRow, toggleSpawnRow)
- [x] progressBarRow/pctColor in utils.js to avoid circular deps
- [x] Static file serving in server.py with _STATIC_DIR, _MIME_TYPES, _serve_static(), /static/ route, path traversal protection
- [x] index.html: 2666 → 44 lines (slim HTML shell + `<script type="module">`)
- [x] 649/649 full suite green

### Work Area 3 Phase 2 Implementation (2026-03-13)
- [x] SSEHub class (thread-safe, bounded Queue(256) per client, workflow filtering)
- [x] SSEEventSink (EventSink protocol adapter → SSEHub.broadcast)
- [x] _ThreadingHTTPServer (ThreadingMixIn + daemon_threads for concurrent SSE+REST)
- [x] /api/events SSE endpoint (30s keepalive, event classification, disconnect handling)
- [x] DebugServer + start_server accept sse_hub param; /api/config returns sse_available
- [x] __main__.py: create SSEHub + SSEEventSink, wire into Orchestrator via event_sink param
- [x] Frontend sse.js module: EventSource, auto-reconnect, named event listeners
- [x] SSE-aware polling: 30s safety net (SSE active) vs 2-3s (file-only fallback)
- [x] 649/649 full suite green

### Work Area 3 Phase 3 Implementation (2026-03-13)
- [x] POST `/api/sessions/{id}/cancel` — async bridge to `orchestrator.cancel_workflow`
- [x] GET `/api/roles`, `/api/roles/{name}` — role details (model, tools, sub-roles, model_config)
- [x] GET `/api/tools` — tool metadata (phase availability, read-only flag, category)
- [x] GET `/api/budget` — full BudgetConfig as JSON via `dataclasses.asdict`
- [x] GET `/api/config/export` — combined config snapshot
- [x] `_WorkflowRunner` extended with `cancel_func` + async cancel scheduling
- [x] Serialization helpers: `_serialize_role_details`, `_serialize_tool_details`, `_serialize_budget`
- [x] `_build_workflow_runner` returns orchestrator for management data extraction
- [x] Cancel button in session-detail header (visible when running + can_cancel)
- [x] 649/649 full suite green

### Work Area 3 Phase 4 Implementation (2026-03-13)
- [x] Created `data_source.py` with `UIDataSource` protocol + `FileSystemDataSource` class
- [x] Extracted ~437 lines of data-reading functions from server.py into data_source.py
- [x] Functions moved: `_read_json`, `_json_text`, `_chars_to_tokens`, `_infer_current_phase`, `_estimate_context_metrics`, `_estimate_todo_metrics`, `_derive_activity`, `_list_sessions_from_disk`, `_read_agent_tree`, `_compute_agent_metrics`, `_aggregate_metrics`, `_read_session_from_disk`
- [x] Handler class: replaced `persist_dir: Path` attribute with `data_source: UIDataSource`
- [x] `DebugServer` and `start_server` accept optional `data_source` param (defaults to FileSystemDataSource)
- [x] `__main__.py` creates `FileSystemDataSource`, passes to `start_server`
- [x] `__init__.py` exports `UIDataSource`, `FileSystemDataSource`
- [x] Updated test import (`test_debug_ui_server_metrics.py` → imports from `data_source`)
- [x] Cleaned up unused imports (`from math import ceil`, `from functools import partial`)
- [x] server.py: 1005 → 571 lines; data_source.py: 513 lines
- [x] 649/649 full suite green

### Work Area 3 Phase 5 Implementation (2026-03-13)
- [x] Created `dashboard.py` with `PARRDashboard` class (~175 lines)
- [x] `PARRDashboard(orchestrator, host, port)` — one-call attach to running Orchestrator
- [x] `start()` (blocking), `start_background()` (daemon thread), `stop()`, `url` property
- [x] SSE wiring: composites `CompositeEventSink([original_sink, sse_sink])` on `EventBridge._sink`
- [x] Data extraction helpers: `_extract_role_details`, `_extract_tool_details`, `_extract_budget`, `_extract_available_roles`
- [x] Graceful when Orchestrator has no domain_adapter or budget (returns empty data)
- [x] `ValueError` when Orchestrator has no `persist_dir`
- [x] `parr-ui` console script in `pyproject.toml` → `parr.debug_ui.__main__:main`
- [x] `__init__.py` exports `PARRDashboard`
- [x] 649/649 full suite green

---

### Work Area 3 Phase 6 Implementation (2026-03-13)
- [x] Agent tree visualization — CSS-based `<ul>/<li>` tree with `::before`/`::after` connector lines
- [x] Each tree node shows role, status badge, token count, cost; clickable to scroll to agent card
- [x] Global overview panel — collapsible `<details>` with per-agent horizontal bar charts (tokens + cost)
- [x] Session-wide tool distribution chart — sorted by count, horizontal bars
- [x] `flattenAgents()` helper recursively walks agent tree extracting per-agent metrics
- [x] Tree and overview only shown when session has sub-agents (single agent: no extra UI)
- [x] Session search/filter — `<input>` in sidebar, filters against role/sub_role/task/status/workflow_id
- [x] Real-time filtering as user types (via `input` event); empty query resets to full list
- [x] `navigateToAgent()` function opens and scrolls to agent's `<details>` card
- [x] 649/649 full suite green

---

## Work Area 4: UI Transformation — User View + Debug Toggle + Chat Support

### Round 1 — User View + Debug Improvements (Steps 1-5) -- DONE
- [x] **Step 1: State + View Mode Toggle** — Added `viewMode`, `lastSessionData`, `openPhases` to state.js. Toggle button in session-detail header. Conditional rendering: user mode hides metrics dashboard, progress bars, global overview, agent tree. Duration shown in user mode header.
- [x] **Step 2: Agent Card Branching** — `renderAgent()` in agent.js branches on `state.viewMode`. User mode: clean card with no tabs, no progress bars, calls `renderUserView()`. Debug mode: preserves full tabbed interface.
- [x] **Step 3: User Report Renderer** — New `user-report.js` (~150 lines). Polished report with: prominent answer prose, finding cards with collapsible metadata pills (Source, Evidence, Confidence), clean bullet lists for Gaps/Recommendations, compact Sources section, overall confidence badge. `togglePill()` function in main.js.
- [x] **Step 4: User View Renderer** — New `user-view.js` (~300 lines). Claude-like continuous stream: minimal header (role + status indicator with pulse animation), compact todo progress bar, content flows without phase labels, framework tools hidden (uses `isFrameworkTool()`), domain tools as collapsed inline blocks, sub-agents as collapsible cards, review badges (pass/fail) instead of full review text, error banners, final report via `renderUserReport()`. Recursive for nested sub-agents.
- [x] **Step 5: Debug View Improvements** — Phase sections in debug chat view are now collapsible. `cs-phase-wrap` with clickable toggle header (chevron + phase label + summary). `togglePhaseSection()` in collapse.js. State persistence via `openPhases` Set. Default: collapsed.
- [x] Exported `_buildLiveTodoList()` from chat.js for reuse in user-view.js
- [x] 6 window bridges total: selectSession, cancelSession, toggleToolRow, toggleSpawnRow, togglePhaseSection, togglePill, toggleViewMode, navigateToAgent
- [x] ~250 lines of CSS added: view toggle button, user view stream, status indicators, pulse animation, todo progress, review badges, report pills, sub-agent styling, collapsible phase sections
- [x] 649/649 tests green (no regressions)
- **Files created:** `debug_ui/static/js/user-view.js`, `debug_ui/static/js/user-report.js`
- **Files changed:** `state.js`, `session-detail.js`, `agent.js`, `chat.js`, `collapse.js`, `sessions.js`, `main.js`, `style.css`

### Round 2 — Chat Support (Steps 6-7) -- DONE
- [x] **Step 6: Chat Backend** — New `POST /api/sessions/{id}/continue` endpoint. `_WorkflowRunner` extended with `continue_func` param and `start_with_context(task, role, additional_context)` method. Handler reads previous session via `data_source.get_session()`, builds context with `_build_continue_context()` (original task + summary + report + findings), calls `start_with_context()`. `DebugServer` and `start_server` accept `continue_func`. `/api/config` returns `can_continue`. `__main__.py` creates `continue_runner` closure. `dashboard.py` creates `continue_runner` closure.
- [x] **Step 7: Chat Frontend** — Sticky chat input bar at bottom of `<main>`: textarea + send button. Visible when: session completed/failed + `can_continue`. `sendChatMessage()` in main.js: POST to `/api/sessions/{id}/continue`, sets `pendingContinuation` + `autoSelectNewest`, triggers `loadSessions()`. Enter-to-send (Shift+Enter for newline), auto-resize textarea. `chatChains` state for conversation chain tracking: `pendingContinuation → parentId`, resolved on `autoSelectNewest` in `renderSessionList()`. "Continued from..." link in session header. CSS: sticky bottom bar with gradient fade, input focus ring, chain link styling.
- [x] 649/649 tests green (no regressions)
- **Files changed:** `server.py`, `__main__.py`, `dashboard.py`, `index.html`, `main.js`, `session-detail.js`, `sessions.js`, `state.js`, `style.css`

### Step 8 — Narrative User View (Replace Raw Phase Content) -- DONE
- [x] **Step 8.1: Tool Classification** — Added `classifyToolForUserView(name)` to utils.js: returns `'finding'` | `'progress'` | `'hidden'` | `'domain'`. Updated `isFrameworkTool()` to include all 21 framework tools (was missing 7: get_report_template, send_message, read_messages, set_shared_state, get_shared_state, get_agent_results_all, get_agent_result).
- [x] **Step 8.2: Narrative Phase Rendering** — Complete rewrite of user-view.js (~690 lines). Replaced raw phase content dump with narrative activity stream:
  - `TOOL_SUMMARIES` — maps tool names to user-friendly one-liners with icons (🔍 search, 📄 read, 🌐 web, ⚙️ run, etc.)
  - `_renderPhaseNarrative()` — phase-specific rendering (plan: status line; act: tool stream; review: badge; report: skip JSON)
  - `_renderToolActivityStream()` — classifies + renders each tool per `classifyToolForUserView()`
  - `_renderInlineFinding()` — finding cards with category badge, content, source/confidence pills
  - `_renderProgressLine()` — "✓ Completed: ..." from mark_todo_complete, resolves todo descriptions
  - `_renderSmartToolLine()` — icon + summary + expandable detail + result preview
  - `_renderThinkingToggle()` — collapsible "Agent's thinking" reusing `.cs-phase-wrap[data-phid]` (zero collapse.js changes)
  - `_extractPreview()` — JSON-aware result preview extraction (title, name, summary, N results count)
  - XML wrapper stripping for `<untrusted_document_content>` tags around tool results
  - Truncated JSON fallback via regex extraction (title/name fields)
- [x] **Step 8.3: CSS** — ~130 lines of CSS for narrative components: phase status lines (`.uv-phase-status`), tool activity stream (`.uv-activity-*`), inline finding cards (`.uv-finding-*`) with color-coded confidence (green/yellow/red), progress lines (`.uv-progress-line`), thinking toggle (`.uv-thinking-*`)
- [x] 649/649 tests green (no regressions)
- [x] Visual testing in Chrome: 0-tool session (prose content), 19-tool session (smart summaries), 4-agent session (recursive sub-agents), debug toggle preserved
- **Files changed:** `utils.js`, `user-view.js`, `style.css`
- **No new files. No backend changes. No changes to collapse.js, state.js, main.js, agent.js, or any other JS module.**

### Bug Fixes (2026-03-14, session 21) -- DONE
- [x] **Bug 1: WinError 5 on Windows file rename** — `_write_json()` in persistence.py: `Path.replace()` fails on Windows when another thread holds the target file open (debug UI reading sub_agents.json during a write). **Fix**: Added retry loop (5 attempts, 50-250ms backoff) for `PermissionError` on Windows (`os.name == 'nt'`). Linux/Mac unaffected (atomic rename works even on open files).
- [x] **Bug 2: KeyError on findings[:20] slice** — `_build_continue_context()` in server.py: `output.findings` is a `Dict[str, Any]` (from AgentOutput), not a list. Slicing a dict with `[:20]` raises `KeyError: slice(None, 20, None)`. **Fix**: Changed to read findings from `memory["findings"]` (the list from persistence) instead of `output["findings"]` (the dict). Added `isinstance` guard. Also added `memory` parameter to `_build_continue_context()`.
- [x] 649/649 tests green (no regressions)
- **Files changed:** `persistence.py`, `debug_ui/server.py`

### Stricter Phase Prompts + UI Polish (2026-03-14, session 22) -- DONE
- [x] **Stricter PHASE_PROMPTS** — Rewrote all 4 prompts in `context_manager.py`: Plan must create todo or DIRECT_ANSWER signal (never answer directly), Act MUST always call `log_finding` (even for model-knowledge answers), Review accepts model_knowledge as valid source, Report synthesizes with explicit "Output quality rules" for comprehensive depth.
- [x] **ConnectionAbortedError** — Added `handle()` override to `_DebugHandler` in server.py wrapping `super().handle()` to catch `ConnectionAbortedError`, `ConnectionResetError`, `BrokenPipeError` (Windows SSE disconnects).
- [x] **Report prompt strengthened** — Report prompt now mandates DETAILED, COMPREHENSIVE output. Explicit rules: don't compress sub-agent findings, don't repeat answer content in key_findings/evidence, weave all detail into the answer field.
- [x] **Skip inline findings for parent agents** — Added `hasSubAgents` flag to user-view.js rendering pipeline. When agent has sub-agents, inline finding cards are suppressed (they were appearing above sub-agent spawn rows, breaking flow). Leaf agents still show inline findings.
- [x] **Thinking toggle content font** — Fixed `.uv-thinking-toggle .uv-content` CSS to use 11px monospace Consolas (was inheriting 13px from parent). Explicit selectors for p, li, h1-h3 descendants.
- [x] **User report overhaul** — Complete rewrite of `user-report.js`: removed expandable pill pattern (`_renderPillGroup`/`togglePill`), replaced with inline metadata spans. Added `_isModelKnowledge()` filter. Evidence section only shows for external sources. Summary suppressed when redundant (35-char prefix match or answer > 200 chars). Key findings only as fallback when no answer.
- [x] **Error retry button** — Added `retryFailedSession()` function in main.js + window bridge. Error banners in user-view.js show retry button when `state.config?.can_continue`.
- [x] **Read retry on Windows** — Both `data_source.py` and `persistence.py` `_read_json()` now retry 3× on `PermissionError` with 30/60/90ms backoff (matches `_write_json` pattern).
- [x] Chrome-verified: 713c2def (sub-agents), failed session (retry button), DPIA (direct prose), research (leaf findings), thinking toggle font, debug toggle
- [x] 649/649 tests green (no regressions)
- **Files changed:** `context_manager.py`, `debug_ui/server.py`, `debug_ui/static/js/user-view.js`, `debug_ui/static/js/user-report.js`, `debug_ui/static/js/main.js`, `debug_ui/static/style.css`, `debug_ui/data_source.py`, `persistence.py`

---

### Step 9 — Claude-Code-Style Compact Block UI (2026-03-14, session 23) -- DONE
- [x] **Complete visual overhaul** of user view to Claude-Code-style compact activity stream. Every intermediate event is a tiny one-line item with collapsed fade-out preview; only the final answer gets visual prominence.
- [x] **state.js** — Added `openBlocks: new Set()` for tracking expanded compact blocks
- [x] **collapse.js** — Added `toggleBlock(rowEl)` function + save/restore for `.uv-block[data-bid]` elements in `saveCollapseState()` / `restoreCollapseState()`
- [x] **main.js** — Added `toggleBlock` import and `window.toggleBlock` bridge
- [x] **style.css** — Added ~200 lines of new CSS:
  - `.uv-line-*` (simple status lines, no expand)
  - `.uv-block-*` (expandable compact blocks: row, chevron, icon, text, badge, preview, fade, detail)
  - `.uv-plan-*` (persistent plan checklist: header, items, done/current states, pulse animation)
  - `.uv-answer-*` (prominent answer section: gradient separator, 14px full-color prose)
  - CSS gradient fade technique: `max-height: 42px; overflow: hidden; opacity: 0.5` + `linear-gradient(transparent, var(--bg-card))` overlay
  - Chevron rotation on `.uv-block.open` via CSS transform
- [x] **user-view.js** — Complete rewrite (~540 lines, was ~690):
  - `_renderCompactTodo()` → `_renderPlanChecklist()`: Shows ALL items with check/circle markers, current item pulses, always visible, never disappears
  - `_renderInlineFinding()` → `_renderFindingBlock()`: Compact `.uv-block` with category header, fade preview of content
  - `_renderSmartToolLine()` + `_renderDomainToolCall()` → unified `_renderToolBlock()`: One line (chevron + icon + text + badge) + fade preview + expandable detail
  - `_renderThinkingToggle()` → `_renderThinkingBlock()`: Compact `.uv-block` with monospace label, fade preview
  - `_renderProgressLine()` → uses `.uv-line` class (no expand)
  - `_renderReviewBadge()` → inline `.uv-line` in `_renderPhaseNarrative()`
  - Final report → `.uv-answer-separator` + `.uv-answer-section` wrapper (14px, full color)
- [x] Chrome-verified: sub-agent session (713c2def), tools session (c80a6510), direct prose session (4834f763), failed session (3a617c8e), block expand/collapse, debug view toggle
- [x] 649/649 tests green (no regressions, no backend changes)
- **Files changed:** `state.js`, `collapse.js`, `main.js`, `style.css`, `user-view.js`
- **No new files. No backend changes.**

---

## Next Step

> All planned work complete. Work Areas 1-4 + Steps 8-9 + bug fixes + prompt/UI polish + compact block UI. 649/649 tests green.
