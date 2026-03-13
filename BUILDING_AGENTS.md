# Building Agents with PARR

This document covers everything needed to build, configure, and run agents. Read it top to bottom.

---

## 1. Core concepts

**Orchestrator** — Creates and manages workflows. Handles agent tree, sub-agent spawning, budget enforcement.

**Agent** — A unit of work. Defined by a role, system prompt, model, tools, and output schema. Executes the 4-phase lifecycle.

**Role** — A reusable agent template. Defined in `roles.yaml` or registered programmatically via `ReferenceDomainAdapter`.

**Sub-role** — A specialization of a role. Inherits the parent's config and overrides specific fields (prompt, tools, schema).

**Tool** — A function the agent can call during execution. Has a name, description, JSON Schema parameters, a handler, and phase restrictions.

**Workflow** — A single execution started by `orchestrator.start_workflow()`. Contains an agent tree (root + optional children).

---

## 2. Setup options

### Option A: Config-driven (recommended)

Create a config directory with three YAML files and content folders:

```
my_project/config/
  roles.yaml
  models.yaml
  budget.yaml
  system_prompts/
    my_role.md
  report_templates/
    my_role.md              # optional
  output_schemas/
    my_role.json            # optional
```

Load and run:

```python
from parr.config import create_orchestrator_from_config
from parr.adapters import LoggingEventSink

orchestrator = create_orchestrator_from_config(
    config_dir="my_project/config",
    tool_registry={"my_tool": my_tool_def},
    llm=my_llm,
    event_sink=LoggingEventSink(),
)

output = await orchestrator.start_workflow(
    task="Do the thing",
    role="my_role",
)
```

### Option B: Programmatic

Register roles directly in Python without YAML:

```python
from parr import (
    Orchestrator, AgentConfig, BudgetConfig, CostConfig,
    ModelConfig, ModelPricing, ToolDef, Phase,
)
from parr.adapters import ReferenceDomainAdapter, LoggingEventSink

adapter = ReferenceDomainAdapter()
adapter.register_role(
    role="analyst",
    config=AgentConfig(
        role="analyst",
        system_prompt="You are a data analyst...",
        model="gpt-4o",
        model_config=ModelConfig(temperature=0.5, max_tokens=4096),
    ),
    tools=[my_search_tool],
    output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
    report_template="Format your findings as a structured report.",
    description="Analyzes data",
)

orchestrator = Orchestrator(
    llm=my_llm,
    domain_adapter=adapter,
    event_sink=LoggingEventSink(),
    cost_config=CostConfig(models={
        "gpt-4o": ModelPricing(input_price_per_1k=0.005, output_price_per_1k=0.015),
    }),
    default_budget=BudgetConfig(max_tokens=100000, max_cost=5.0),
)

output = await orchestrator.start_workflow(task="Analyze X", role="analyst")
```

---

## 3. Defining tools

A tool is a `ToolDef` with a handler function. The handler receives keyword arguments matching its JSON Schema parameters.

```python
from parr import ToolDef, Phase

async def search(query: str, limit: int = 5) -> str:
    results = my_database.search(query, limit)
    return json.dumps(results)

search_tool = ToolDef(
    name="search",
    description="Search the dataset by keyword.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results"},
        },
        "required": ["query"],
    },
    handler=search,
    phase_availability=[Phase.ACT],          # only usable during Act phase
)
```

### Tool options

| Field | Type | Default | Purpose |
|---|---|---|---|
| `name` | str | required | Tool identifier |
| `description` | str | required | Shown to the LLM |
| `parameters` | dict | required | JSON Schema for arguments |
| `handler` | callable | required | Async or sync function |
| `phase_availability` | list | all phases | Which phases can use this tool |
| `mandatory_in_phases` | list | None | Agent must call this tool in these phases |
| `timeout_ms` | int | 30000 | Handler timeout |
| `max_calls_per_phase` | int | None | Rate limit per phase |
| `output_schema` | dict | None | JSON Schema to validate handler output |
| `retry_on_failure` | bool | False | Retry on handler exception |
| `max_retries` | int | 0 | Max retry attempts |
| `wraps_untrusted_content` | bool | False | Wrap output in safety tags |

### Phase availability

Tools are only available to the LLM in their declared phases. Built-in tools are pre-assigned:

| Phase | Built-in tools |
|---|---|
| Plan | `create_todo_list`, `update_todo_list`, `get_todo_list` |
| Act | `mark_todo_complete`, `log_finding`, `get_findings`, `spawn_agent`, `wait_for_agents` |
| Review | `review_checklist` |
| Report | `get_report_template`, `submit_report` |

Your domain tools typically go in Act only. Use `phase_availability=[Phase.ACT]`.

### Handler return values

Handlers can return:
- `str` — returned directly as tool result
- `dict` or `list` — serialized to JSON
- Any other type — converted via `str()`

If `output_schema` is set, the return value is validated against it before the agent sees it.

---

## 4. roles.yaml reference

```yaml
roles:
  role_name:
    model: gpt-4o                                  # required, must exist in models.yaml
    description: "What this role does"              # shown in spawn_agent tool
    system_prompt: system_prompts/role_name.md      # required, path relative to config dir
    report_template: report_templates/role_name.md  # optional
    output_schema: output_schemas/role_name.json    # optional, validates submit_report
    tools:                                          # optional, list of tool names
      - search
      - save
    model_config:                                   # optional
      temperature: 0.7
      top_p: 1.0
      max_tokens: 4096

    sub_roles:                                      # optional
      specialization:
        description: "Narrow focus"
        system_prompt: system_prompts/role_name__specialization.md  # override
        output_schema: output_schemas/role_name__specialization.json
        # tools, model, model_config — all optional overrides
        # omitted fields inherit from the parent role
```

### Sub-role inheritance

Sub-roles inherit everything from the parent. Only specified fields override:

| Field | If omitted in sub-role |
|---|---|
| `system_prompt` | Inherits parent's prompt |
| `model` | Inherits parent's model |
| `model_config` | Inherits parent's config |
| `tools` | Inherits parent's tools |
| `output_schema` | Inherits parent's schema |
| `report_template` | Inherits parent's template |

---

## 5. models.yaml reference

```yaml
models:
  gpt-4o:
    input_price_per_1k: 0.005
    output_price_per_1k: 0.015
    context_window: 128000

  gpt-4o-mini:
    input_price_per_1k: 0.00015
    output_price_per_1k: 0.0006
    context_window: 128000

  claude-3-5-sonnet:
    input_price_per_1k: 0.003
    output_price_per_1k: 0.015
    context_window: 200000
```

Pricing is used for cost tracking and budget enforcement. Update when provider pricing changes.

---

## 6. budget.yaml reference

```yaml
budget_defaults:
  max_tokens: 200000        # total token ceiling per workflow
  max_cost: 10.0            # dollar ceiling per workflow
  max_duration_ms: 300000   # wall-clock timeout (ms)
  max_agent_depth: 3        # max nesting for spawned agents
  max_parallel_agents: 3    # max concurrent children
  max_sub_agents_total: 10  # max total children per agent
  inherit_remaining: true   # children get fraction of parent's remaining budget

phase_limits:
  plan: 5       # max LLM calls in Plan phase
  act: 15       # max LLM calls in Act phase
  review: 5     # max LLM calls in Review phase
  report: 5     # max LLM calls in Report phase

llm_rate_limit:
  enabled: true
  max_concurrent_requests: 2
  max_tokens_per_minute: 100000
  max_requests_per_minute: 60   # alias for max_requests_per_window=60, window_seconds=60
  max_queue_size: 100
  acquire_timeout_seconds: 30
```

### Budget inheritance

When an agent spawns a child, the child receives 50% of the parent's remaining token and cost budget. The depth limit decreases by 1. This prevents runaway cost from recursive spawning.

### LLM queue and throttling

`llm_rate_limit` adds a **central async FIFO queue** for all `chat_with_tools` calls in an orchestrator instance. Requests wait with `await` (non-blocking) until:

- concurrent in-flight limit allows entry
- rolling request-window limit allows entry

Use this to smooth bursts and reduce provider 429s when many agents run at once.

### Per-workflow override

```python
output = await orchestrator.start_workflow(
    task="...",
    role="analyst",
    budget=BudgetConfig(max_tokens=50000, max_cost=2.0),
)
```

---

## 7. System prompts

System prompts are Markdown files loaded at runtime. They define the agent's persona and instructions.

```markdown
You are a security analyst specializing in vulnerability assessment.

Guidelines:
- Search for known CVEs related to the system under test.
- Assess severity using CVSS scoring.
- Log each vulnerability as a finding with its CVE ID.
- Recommend remediations ordered by severity.
```

The framework appends phase-specific instructions automatically. You do not need to describe the PARR lifecycle in your prompt.

---

## 8. Output schemas

JSON Schema files that validate the `submit_report` output. The agent sees this schema and must conform.

```json
{
  "type": "object",
  "properties": {
    "vulnerabilities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "cve_id": { "type": "string" },
          "severity": { "type": "string", "enum": ["critical", "high", "medium", "low"] },
          "description": { "type": "string" },
          "remediation": { "type": "string" }
        },
        "required": ["cve_id", "severity", "description"]
      }
    },
    "summary": { "type": "string" }
  },
  "required": ["vulnerabilities", "summary"]
}
```

If no schema is provided, the agent can submit any JSON structure.

---

## 9. Sub-agent spawning

During the Act phase, an agent can delegate subtasks to child agents using the `spawn_agent` tool:

```
spawn_agent(role="researcher", sub_role="deep_dive", task="Analyze X in detail")
```

The orchestrator:
1. Creates a child agent node in the tree
2. Allocates 50% of the parent's remaining budget
3. Runs the child through the full PARR lifecycle
4. Returns the child's output to the parent

The parent sees available roles via the `spawn_agent` tool description, which lists all registered roles and sub-roles.

### Limits

- `max_agent_depth` — Prevents infinite nesting (default: 3)
- `max_parallel_agents` — Max concurrent children (default: 3)
- `max_sub_agents_total` — Max total children per agent (default: 10)
- Children get max 1 review cycle (vs 2 for root agents)

---

## 10. LLM adapters

### Option A: Declarative via providers.yaml (recommended)

Add a `providers.yaml` to your config directory. API keys use `${ENV_VAR}` references resolved from environment variables at load time — never put actual secrets in the YAML file.

```yaml
# config/providers.yaml
default_provider: azure_openai

providers:
  azure_openai:
    api_key: ${AZURE_OPENAI_API_KEY}
    endpoint: ${AZURE_OPENAI_ENDPOINT}
    api_version: "2024-12-01-preview"

  openai:
    api_key: ${OPENAI_API_KEY}

  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
```

Set environment variables and run — the config loader creates the LLM adapter automatically:

```bash
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

```python
# No llm= needed — built from providers.yaml
orchestrator = create_orchestrator_from_config(
    config_dir="my_project/config",
    tool_registry={"my_tool": my_tool_def},
    event_sink=LoggingEventSink(),
)
```

Override provider or model at runtime:

```python
orchestrator = create_orchestrator_from_config(
    config_dir="my_project/config",
    tool_registry={"my_tool": my_tool_def},
    provider_override="openai",     # use OpenAI instead of default
    model_override="gpt-4o",        # override model from roles.yaml
)
```

**providers.yaml is optional.** If the file does not exist, pass `llm=` explicitly (see Option B).

Provider-specific fields:

| Provider | Required fields | Optional fields |
|---|---|---|
| `openai` | `api_key` | `timeout` |
| `azure_openai` | `api_key`, `endpoint` | `api_version`, `timeout` |
| `anthropic` | `api_key` | `timeout` |

### Option B: Programmatic

Create the LLM adapter directly in Python:

```python
from parr.adapters import create_tool_calling_llm

# OpenAI
llm = create_tool_calling_llm("openai", model="gpt-4o", api_key="sk-...")

# Azure OpenAI
llm = create_tool_calling_llm(
    "azure_openai",
    model="my-gpt4o-deployment",
    endpoint="https://my-resource.openai.azure.com",
    api_key="...",
    api_version="2024-02-15-preview",
)

# Anthropic
llm = create_tool_calling_llm("anthropic", model="claude-3-5-sonnet-20241022", api_key="...")
```

### Custom LLM

Implement the `ToolCallingLLM` protocol:

```python
from parr import ToolCallingLLM, LLMResponse, Message, ModelConfig

class MyLLM:
    async def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
        model: str,
        model_config: ModelConfig,
        stream: bool = False,
        on_token=None,
    ) -> LLMResponse:
        # Call your LLM, translate response to LLMResponse
        ...
```

---

## 11. Event handling

The framework emits events throughout execution. Subscribe for logging, UI updates, or monitoring.

### Built-in sinks

```python
from parr.adapters import LoggingEventSink, WebSocketEventSink, CompositeEventSink

# Logging
sink = LoggingEventSink()

# WebSocket
sink = WebSocketEventSink(send_callback=my_ws_send)

# Both
sink = CompositeEventSink([LoggingEventSink(), WebSocketEventSink(my_ws_send)])
```

### Custom sink

```python
class MySink:
    async def emit(self, event: dict) -> None:
        event_type = event["event_type"]
        # handle: agent_started, agent_completed, phase_started,
        #         phase_completed, llm_call_completed, tool_executed,
        #         budget_warning, budget_exceeded, ...
```

### Event types

| Event | When |
|---|---|
| `agent_started` | Agent begins execution |
| `agent_completed` | Agent finished successfully |
| `agent_failed` | Agent hit an unrecoverable error |
| `phase_started` | Entering a phase |
| `phase_completed` | Exiting a phase |
| `llm_call_completed` | LLM responded (tokens in data) |
| `tool_executed` | Tool call finished |
| `budget_warning` | 80% of budget consumed |
| `budget_exceeded` | Hard budget limit reached |
| `agent_token` | Streamed text token |

---

## 12. Testing with MockLLM

For deterministic tests, use the included `MockToolCallingLLM`:

```python
from parr import Phase
from parr.tests.mock_llm import (
    MockToolCallingLLM,
    make_text_response,
    make_tool_call_response,
)

llm = MockToolCallingLLM({
    Phase.PLAN: [
        make_tool_call_response("create_todo_list", {
            "items": [{"description": "Step 1"}, {"description": "Step 2"}]
        }),
        make_text_response("Plan ready."),
    ],
    Phase.ACT: [
        make_tool_call_response("my_tool", {"query": "test"}),
        make_tool_call_response("log_finding", {
            "category": "test", "content": "Found X", "source": "doc1"
        }),
        make_tool_call_response("mark_todo_complete", {"item_index": 0, "summary": "Done"}),
        make_tool_call_response("mark_todo_complete", {"item_index": 1, "summary": "Done"}),
        make_text_response("Act complete."),
    ],
    Phase.REVIEW: [
        make_tool_call_response("review_checklist", {
            "items": [{"criterion": "Work done", "rating": "pass", "justification": "OK"}]
        }),
        make_text_response("REVIEW_PASSED"),
    ],
    Phase.REPORT: [
        make_tool_call_response("submit_report", {"report": {"result": "All good"}}),
        make_text_response("Done."),
    ],
})
```

The mock detects the current phase from system prompt content and returns the scripted responses in order. When responses are exhausted, it returns a default text response that stops the loop.

---

## 13. Working memory

Each agent has an `AgentWorkingMemory` that persists across phases:

| Store | Written by | Read by |
|---|---|---|
| Todo list | `create_todo_list`, `mark_todo_complete` | All phases (context snapshot) |
| Findings | `log_finding` | Review, Report phases |
| Review results | `review_checklist` | Report phase, retry logic |
| Report | `submit_report` | Orchestrator (final output) |

Findings are carried into the Report phase and should be the primary data source for the final output.

---

## 14. Stall detection

The framework detects when an agent is stuck:

| Mechanism | Trigger | Action |
|---|---|---|
| Read-only loop | 5 consecutive iterations using only read tools | Force phase exit |
| Framework-only loop | 10 consecutive iterations with no domain tools | Force phase exit |
| Duplicate tool calls | 4 consecutive identical (tool, args) pairs | Force phase exit |

When a stall is detected, the framework exits the phase with the best output available and emits a `phase_iteration_limit` event.

Thresholds are configurable via `StallDetectionConfig` in `AgentConfig.stall_detection`.

---

## 15. Context management

The framework manages conversation history automatically:

- **Phase boundaries** — Previous phase output is compacted into a summary for the next phase. Raw conversation is not carried across.
- **Soft compaction** — At 40% of context window, older tool calls are summarized while findings are preserved.
- **Hard truncation** — At 65%, only system prompt + user message + last 3 message groups are kept.

Token estimation uses ~4 characters per token. This is approximate but prevents context overflow on all major models.
