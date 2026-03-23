# PARR Multi-Agent Framework

A protocol-based Python framework for building multi-agent AI systems with a structured **Plan → Act → Review → Report** lifecycle.

Agents plan their work, execute with tools, self-review, and produce structured output — all with budget enforcement, stall detection, and real-time event streaming.

## What this solves

Building reliable multi-agent systems is hard. LLMs loop, forget their task, blow through token limits, and produce unstructured output. PARR handles this:

- **Structured lifecycle** — Every agent follows 4 phases. No open-ended loops.
- **Budget enforcement** — Token and cost limits checked before every LLM call. Child agents inherit remaining budget.
- **Stall detection** — Three independent mechanisms catch infinite loops and force graceful exits.
- **Tool validation** — JSON Schema input/output validation, timeouts, retries, phase-gated availability.
- **Protocol-based** — Swap LLM providers, add domain tools, change event transport without touching the engine.

## Table of contents

- [Install](#install)
- [Quick start — run the example](#quick-start)
- [Setup modes](#setup-modes)
  - [Option A: Config-driven (YAML)](#option-a-config-driven-yaml)
  - [Option B: Programmatic (Python only)](#option-b-programmatic-python-only)
- [Define domain tools](#define-domain-tools)
- [LLM providers](#llm-providers)
- [Debug UI](#debug-ui)
  - [Standalone CLI](#standalone-cli)
  - [Library mode (PARRDashboard)](#library-mode-parrdashboard)
  - [File-only browsing](#file-only-browsing)
- [Configuration reference](#configuration-reference)
- [Architecture](#architecture)
- [Extension points](#extension-points)
- [Testing with MockLLM](#testing-with-mockllm)

---

## Install

```bash
pip install -e ".[all]"
```

This installs both the `openai` and `anthropic` SDK extras. To install only what you need:

```bash
pip install -e "."               # core only (jsonschema, pyyaml)
pip install -e ".[openai]"       # + OpenAI/Azure
pip install -e ".[anthropic]"    # + Anthropic
```

## Quick start

Run the included research assistant example with a mock LLM (no API key needed):

```bash
python -m examples.research_assistant.run
```

Run it with a real LLM:

```bash
export AZURE_OPENAI_API_KEY=your-key
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_API_VERSION=2024-02-15-preview

python -m examples.research_assistant.run --live
```

Override the provider or model:

```bash
python -m examples.research_assistant.run --live --provider openai --model gpt-4o
python -m examples.research_assistant.run --live --provider anthropic --model claude-3-5-sonnet
```

---

## Setup modes

There are two ways to set up PARR. Both produce the same `Orchestrator` — pick whichever fits your project.

### Option A: Config-driven (YAML)

Best when you want roles, budgets, and models defined declaratively. Create a config directory:

```
my_project/
  config/
    roles.yaml
    models.yaml
    budget.yaml
    providers.yaml          # optional — auto-creates LLM from env vars
    system_prompts/
      my_role.md
    output_schemas/
      my_role.json
  tools.py
  main.py
```

**config/roles.yaml**

```yaml
roles:
  researcher:
    model: gpt-4o
    description: "Researches topics and produces structured reports."
    system_prompt: system_prompts/researcher.md
    output_schema: output_schemas/researcher.json
    tools:
      - search_documents
      - read_section
    model_config:
      temperature: 0.4
      max_tokens: 4096
```

**config/models.yaml**

```yaml
models:
  gpt-4o:
    input_price_per_1k: 0.0025
    output_price_per_1k: 0.01
    context_window: 128000
```

**config/budget.yaml**

```yaml
budget_defaults:
  max_tokens: 500000
  max_cost: 10.0
  max_duration_ms: 300000

phase_limits:
  plan: 10
  act: 30
  review: 10
  report: 10
```

**config/providers.yaml**

```yaml
default_provider: openai

providers:
  openai:
    api_key: ${OPENAI_API_KEY}

  # azure_openai:
  #   api_key: ${AZURE_OPENAI_API_KEY}
  #   endpoint: ${AZURE_OPENAI_ENDPOINT}
  #   api_version: ${AZURE_OPENAI_API_VERSION}

  # anthropic:
  #   api_key: ${ANTHROPIC_API_KEY}
```

**tools.py**

```python
import json
from parr import ToolDef, Phase

async def search_documents(query: str, top_k: int = 3) -> str:
    # Your search logic here
    return json.dumps({"results": []})

SEARCH_TOOL = ToolDef(
    name="search_documents",
    description="Search the document corpus.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "top_k": {"type": "integer", "description": "Max results."},
        },
        "required": ["query"],
    },
    handler=search_documents,
    phase_availability=[Phase.ACT],
)
```

**main.py**

```python
import asyncio
from parr.config import create_orchestrator_from_config
from tools import SEARCH_TOOL

async def main():
    orchestrator = create_orchestrator_from_config(
        config_dir="./config",
        tool_registry={"search_documents": SEARCH_TOOL},
    )
    # LLM is auto-created from providers.yaml + env vars

    output = await orchestrator.start_workflow(
        task="Research the impact of AI on healthcare.",
        role="researcher",
    )
    print(output.status, output.summary)

asyncio.run(main())
```

That's it. The config loader reads roles.yaml, resolves the system prompt file, attaches your tools, builds the LLM from providers.yaml, and gives you a ready-to-run orchestrator.

### Option B: Programmatic (Python only)

No YAML files needed. Everything is wired in code:

```python
import asyncio
from parr import (
    Orchestrator,
    BudgetConfig,
    ToolDef,
    Phase,
    AgentConfig,
)
from parr.adapters import (
    ReferenceDomainAdapter,
    LoggingEventSink,
    create_tool_calling_llm,
)

async def main():
    # 1. Create LLM adapter
    llm = create_tool_calling_llm(
        "openai",
        model="gpt-4o",
        api_key="sk-...",
    )

    # 2. Define tools
    async def search(query: str) -> str:
        return '{"results": ["item1", "item2"]}'

    search_tool = ToolDef(
        name="search",
        description="Search for information.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
            },
            "required": ["query"],
        },
        handler=search,
        phase_availability=[Phase.ACT],
    )

    # 3. Register roles
    adapter = ReferenceDomainAdapter()
    adapter.register_role(
        role="analyst",
        config=AgentConfig(
            role="analyst",
            system_prompt="You are a research analyst. Search for information and produce a report.",
            model="gpt-4o",
        ),
        tools=[search_tool],
    )

    # 4. Create orchestrator
    orchestrator = Orchestrator(
        llm=llm,
        domain_adapter=adapter,
        event_sink=LoggingEventSink(),
        default_budget=BudgetConfig(max_tokens=100000, max_cost=5.0),
    )

    # 5. Run
    output = await orchestrator.start_workflow(
        task="Find the latest trends in renewable energy.",
        role="analyst",
    )
    print(output.status, output.summary)

asyncio.run(main())
```

---

## Define domain tools

Every tool is a `ToolDef` with a JSON Schema for parameters and an async handler:

```python
from parr import ToolDef, Phase

async def save_note(title: str, content: str) -> str:
    # Do something with the note
    return '{"saved": true}'

note_tool = ToolDef(
    name="save_note",
    description="Save a research note.",
    parameters={
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["title", "content"],
    },
    handler=save_note,
    phase_availability=[Phase.ACT],          # only available during Act phase
    # Optional:
    # timeout_ms=30000,                      # handler timeout (default 30s)
    # max_calls_per_phase=10,                # limit calls per phase
    # retry_on_failure=True, max_retries=2,  # auto-retry on exception
    # is_read_only=True,                     # hint for stall detection
    # marks_progress=True,                   # this tool represents real work
)
```

Tool handlers receive keyword arguments matching the JSON Schema properties. They must return a JSON string.

The framework automatically provides built-in tools that agents use internally:
- `create_todo_list`, `mark_todo_complete` (Plan phase)
- `log_finding` (Act phase)
- `review_checklist` (Review phase)
- `submit_report` (Report phase)
- `spawn_agent`, `wait_for_agents` (multi-agent orchestration)

---

## LLM providers

Three providers are supported. Use `create_tool_calling_llm` or instantiate directly:

**OpenAI**

```python
from parr.adapters import create_tool_calling_llm

llm = create_tool_calling_llm("openai", model="gpt-4o", api_key="sk-...")
```

**Azure OpenAI**

```python
llm = create_tool_calling_llm(
    "azure_openai",
    model="gpt-4o",
    endpoint="https://your-resource.openai.azure.com",
    api_key="your-key",
    api_version="2024-02-15-preview",
)
```

**Anthropic**

```python
llm = create_tool_calling_llm("anthropic", model="claude-3-5-sonnet-20241022", api_key="sk-ant-...")
```

Or pass an existing async client:

```python
from openai import AsyncOpenAI
from parr.adapters import OpenAIToolCallingLLM

client = AsyncOpenAI(api_key="sk-...")
llm = OpenAIToolCallingLLM(client=client, model="gpt-4o")
```

When using config-driven setup, the LLM is created automatically from `providers.yaml` — you don't need any of the above.

---

## Debug UI

PARR includes a browser-based debug dashboard for inspecting agent sessions in real time. It shows the full agent hierarchy, per-agent metrics, tool call history, conversation logs, and token/cost breakdowns.

### Prerequisites

Enable file persistence by passing `persist_dir` to the orchestrator:

```python
orchestrator = Orchestrator(llm=llm, persist_dir="./sessions", ...)
# or in config-driven mode:
orchestrator = create_orchestrator_from_config(config_dir="./config", ...)
orchestrator._persist_dir = "./sessions"
```

### Standalone CLI

Browse persisted sessions (read-only — no LLM needed):

```bash
python -m parr.debug_ui --persist-dir ./sessions
```

Or using the installed entry point:

```bash
parr-ui --persist-dir ./sessions
```

Enable launching new workflows from the UI by pointing at a config directory:

```bash
python -m parr.debug_ui \
  --persist-dir ./sessions \
  --config-dir ./examples/research_assistant/config
```

Load real domain tools so launched workflows can actually call them:

```bash
python -m parr.debug_ui \
  --persist-dir ./sessions \
  --config-dir ./examples/research_assistant/config \
  --tools-module examples.research_assistant.run
```

CLI options:

| Flag | Default | Description |
|---|---|---|
| `--persist-dir` | (required) | Path to session data directory |
| `--config-dir` | None | Config directory to enable session creation |
| `--tools-module` | None | Python module with `ToolDef` objects for domain tools |
| `--host` | localhost | Server host |
| `--port` | 8080 | Server port |
| `--provider` | None | Override LLM provider |
| `--model` | None | Override LLM model |

### Library mode (PARRDashboard)

Attach the debug UI to a running orchestrator in your own code. This gives you live SSE updates as agents execute:

```python
import asyncio
from parr import Orchestrator, BudgetConfig
from parr.adapters import LoggingEventSink, create_tool_calling_llm
from parr.debug_ui import PARRDashboard

async def main():
    llm = create_tool_calling_llm("openai", model="gpt-4o", api_key="sk-...")

    orchestrator = Orchestrator(
        llm=llm,
        persist_dir="./sessions",       # required for debug UI
        event_sink=LoggingEventSink(),
        default_budget=BudgetConfig(max_tokens=100000),
    )

    # Start dashboard on background thread
    dashboard = PARRDashboard(orchestrator, port=8080)
    dashboard.start_background()
    print(f"Debug UI: {dashboard.url}")

    # Run your workflow — the UI updates live
    output = await orchestrator.start_workflow(task="...", role="analyst")

    dashboard.stop()

asyncio.run(main())
```

Open `http://localhost:8080` in your browser while the workflow runs.

### File-only browsing

If you just want to inspect sessions that were already persisted (e.g., from a production run), point the CLI at the directory:

```bash
python -m parr.debug_ui --persist-dir /path/to/saved/sessions
```

No config directory or API keys needed — the UI reads the JSON files directly.

---

## Configuration reference

See [BUILDING_AGENTS.md](BUILDING_AGENTS.md) for the full configuration reference, including:

- Role definitions with sub-roles
- System prompt files and report templates
- Output schema validation (JSON Schema)
- Budget tuning (token, cost, duration limits)
- Phase iteration limits
- Stall detection thresholds
- Rate limiting configuration
- Simple query bypass
- Tool middleware
- Inter-agent coordination

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Orchestrator                       │
│  Creates workflows, manages agent tree,              │
│  handles sub-agent spawning, enforces budgets        │
├──────────────────────────────────────────────────────┤
│                   AgentRuntime                        │
│  Drives a single agent through Plan→Act→Review→Report│
│  with review retry loops and context compaction       │
├──────────────────────────────────────────────────────┤
│                    PhaseRunner                        │
│  Single-phase loop: LLM call → tool calls → repeat   │
│  Stall detection, iteration limits                   │
├──────────────────────────────────────────────────────┤
│    ToolCallingLLM    │  DomainAdapter  │  EventSink  │
│    (protocol)        │  (protocol)     │  (protocol) │
│    OpenAI, Anthropic │  Your tools,    │  Logging,   │
│    or custom         │  roles, schemas │  WebSocket  │
└──────────────────────┴─────────────────┴─────────────┘
```

**The PARR lifecycle** — Every agent executes exactly 4 phases:

1. **Plan** — Create a todo list. Decide what steps to take.
2. **Act** — Work through the plan. Call domain tools, log findings.
3. **Review** — Self-evaluate using a checklist. If criteria fail, re-plan and re-execute (up to 2 cycles).
4. **Report** — Synthesize findings into structured output validated against a JSON Schema.

---

## Extension points

| Protocol / Class | Purpose | Included implementations |
|---|---|---|
| `ToolCallingLLM` | LLM with function calling | OpenAI, Azure OpenAI, Anthropic |
| `DomainAdapter` | Roles, tools, schemas | `ReferenceDomainAdapter` (in-memory) |
| `EventSink` | Real-time event streaming | Logging, WebSocket, Composite |
| `DocumentSearchProvider` | RAG retrieval | `RAGDocumentSearchAdapter` |
| `ToolMiddleware` | Pre/post tool call hooks | (subclass `ToolMiddleware`) |
| `StallDetector` | Custom stall detection | (subclass or configure thresholds) |
| `CompactionStrategy` | Context window management | (subclass for custom compaction) |
| `ChildBudgetAllocator` | Sub-agent budget splits | `FractionAllocator`, `EqualShareAllocator`, `FixedAllocator` |
| `OutputValidator` | Output validation | `JsonSchemaValidator`, `CompositeValidator` |
| `AgentCoordinator` | Inter-agent messaging | Default coordinator included |

---

## Testing with MockLLM

Use `MockToolCallingLLM` for deterministic tests without API calls:

```python
import asyncio
from parr import Orchestrator, Phase, ToolDef, BudgetConfig
from parr.adapters import ReferenceDomainAdapter
from parr import AgentConfig
from parr.tests.mock_llm import (
    MockToolCallingLLM,
    make_text_response,
    make_tool_call_response,
)

mock_llm = MockToolCallingLLM({
    Phase.PLAN: [
        make_tool_call_response("create_todo_list", {
            "items": [{"description": "Do the work"}]
        }),
        make_text_response("Plan ready."),
    ],
    Phase.ACT: [
        make_text_response("Work done."),
    ],
    Phase.REVIEW: [
        make_tool_call_response("review_checklist", {
            "items": [{"criterion": "Task complete", "rating": "pass", "justification": "Done."}]
        }),
        make_text_response("REVIEW_PASSED"),
    ],
    Phase.REPORT: [
        make_tool_call_response("submit_report", {
            "summary": "Task completed successfully.",
        }),
        make_text_response("Report submitted."),
    ],
})

async def test():
    adapter = ReferenceDomainAdapter()
    adapter.register_role(
        role="worker",
        config=AgentConfig(role="worker", system_prompt="Do tasks.", model="mock"),
    )
    orch = Orchestrator(llm=mock_llm, domain_adapter=adapter)
    output = await orch.start_workflow(task="Do something", role="worker")
    assert output.status == "completed"

asyncio.run(test())
```

---

## Project structure

```
parr/
  orchestrator.py         # Workflow management, agent tree
  agent_runtime.py        # Single agent PARR lifecycle
  phase_runner.py         # Single phase execution loop
  core_types.py           # Data structures (dataclasses)
  protocols.py            # Extension point interfaces
  budget_tracker.py       # Token/cost/duration enforcement
  framework_tools.py      # Built-in tools (todos, findings, review)
  context_manager.py      # Message history and compaction
  tool_registry.py        # Tool registration and phase filtering
  tool_executor.py        # Tool dispatch, validation, retry
  event_bus.py            # In-process pub/sub
  event_types.py          # Event definitions
  trace_store.py          # Append-only execution trace
  persistence.py          # File-system session persistence
  adapters/               # LLM, domain, event sink implementations
  config/                 # YAML loader, validator
  debug_ui/               # Browser-based debug dashboard
  tests/                  # Test infrastructure (MockLLM, fixtures)
examples/
  research_assistant/     # Config-driven example (mock + live)
  live_demo/              # Live LLM example with event streaming
```

## Requirements

- Python 3.10+
- `jsonschema` and `pyyaml` (core)
- `openai` (optional, for OpenAI/Azure adapter)
- `anthropic` (optional, for Anthropic adapter)

## License

MIT
