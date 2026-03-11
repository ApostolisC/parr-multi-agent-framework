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

## Quick start

```bash
pip install -e ".[all]"
```

Run the included example (no API key needed — uses a mock LLM):

```bash
python -m examples.research_assistant.run
```

Or with a real LLM:

```bash
export OPENAI_API_KEY=sk-...
python -m examples.research_assistant.run --live
```

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

## The PARR lifecycle

Every agent executes exactly 4 phases:

**Plan** — Create a todo list. Decide what steps to take and in what order.

**Act** — Work through the plan. Call domain tools, search documents, log findings, mark items complete.

**Review** — Self-evaluate using a checklist. If criteria fail, re-plan and re-execute (up to 2 cycles).

**Report** — Synthesize findings into structured output validated against a JSON Schema.

## Extension points

The framework defines 4 protocols. You implement them for your domain:

| Protocol | Purpose | Included adapters |
|---|---|---|
| `ToolCallingLLM` | LLM with function calling | OpenAI, Azure OpenAI, Anthropic |
| `DomainAdapter` | Roles, tools, schemas | `ReferenceDomainAdapter` (in-memory) |
| `EventSink` | Real-time event streaming | Logging, WebSocket, Composite |
| `DocumentSearchProvider` | RAG retrieval | `RAGDocumentSearchAdapter` |

## Configuration

Three YAML files define the operational parameters:

**roles.yaml** — Agent roles, models, tools, system prompts, output schemas

**models.yaml** — LLM pricing for cost tracking

**budget.yaml** — Token/cost/time limits and phase iteration caps

See [BUILDING_AGENTS.md](BUILDING_AGENTS.md) for the full configuration reference.

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
  adapters/               # LLM, domain, event sink implementations
  config/                 # YAML loader, validator
  tests/                  # Test infrastructure (MockLLM, fixtures)
examples/
  research_assistant/     # Working end-to-end example
```

## Requirements

- Python 3.10+
- `jsonschema` and `pyyaml` (core)
- `openai` (optional, for OpenAI/Azure adapter)
- `anthropic` (optional, for Anthropic adapter)

## License

MIT
