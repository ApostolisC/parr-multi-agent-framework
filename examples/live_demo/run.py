"""
Live Demo — 100% AI, Full Transparency.

No mocks. Uses a real LLM via providers.yaml. A custom event sink prints
every framework event in real time so you can see exactly what the agent
does at each step: which tools it calls, what arguments it passes, what
the LLM returns, how many tokens each call costs, and how the final
report is assembled.

Requires:
    pip install -e ".[openai]"      # or .[all]

    Set environment variables (see .env.example) then run:
        python -m examples.live_demo.run

    Override provider or model:
        python -m examples.live_demo.run --provider openai --model gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import tempfile
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from parr import (
    AgentOutput,
    ExecutionMetadata,
    Phase,
    TokenUsage,
    ToolDef,
)
from parr.config import create_orchestrator_from_config

# ---------------------------------------------------------------------------
# Terminal colours (ANSI escape codes, works on modern terminals)
# ---------------------------------------------------------------------------

class _C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    WHITE = "\033[97m"


def _banner(text: str, colour: str = _C.CYAN) -> None:
    width = 72
    print(f"\n{colour}{_C.BOLD}{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}{_C.RESET}")


def _section(text: str, colour: str = _C.YELLOW) -> None:
    print(f"\n{colour}{_C.BOLD}--- {text} ---{_C.RESET}")


def _kv(key: str, value: Any, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{_C.DIM}{key}:{_C.RESET} {value}")


def _wrap(text: str, indent: int = 4, width: int = 68) -> str:
    pad = " " * indent
    return textwrap.fill(text, width=width, initial_indent=pad,
                         subsequent_indent=pad)


# ---------------------------------------------------------------------------
# Transparent Event Sink — prints every event to the console
# ---------------------------------------------------------------------------

class TransparentEventSink:
    """
    EventSink that prints every framework event in a human-readable format.
    Implements the EventSink protocol (async emit).

    All LLM responses and tool results are written **untruncated** to a log
    file in the system temp directory so nothing is ever lost.
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        self._phase_start_time: Optional[float] = None
        self._total_tokens = 0
        self._total_cost = 0.0
        self._llm_calls = 0
        self._tool_calls: List[Dict[str, Any]] = []

        # Log file — defaults to %TEMP%/parr_live_demo_<timestamp>.log
        if log_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(tempfile.gettempdir()) / f"parr_live_demo_{ts}.log"
        self._log_path = log_path
        self._log_file = open(self._log_path, "w", encoding="utf-8")
        self._log(f"PARR Live Demo Log — {datetime.now().isoformat()}\n")
        print(f"  {_C.DIM}Log file: {self._log_path}{_C.RESET}")

    def _log(self, text: str) -> None:
        """Write to the log file (untruncated)."""
        self._log_file.write(text + "\n")
        self._log_file.flush()

    def close(self) -> None:
        if self._log_file and not self._log_file.closed:
            self._log_file.close()

    async def emit(self, event: Dict[str, Any]) -> None:
        etype = event.get("event_type", "")
        data = event.get("data", {})

        if etype == "agent_started":
            _banner("AGENT STARTED", _C.GREEN)
            _kv("Role", data.get("role", "?"))
            if data.get("sub_role"):
                _kv("Sub-role", data["sub_role"])
            _kv("Depth", data.get("depth", 0))

        elif etype == "phase_started":
            phase = data.get("phase", "?").upper()
            self._phase_start_time = time.time()
            _banner(f"PHASE: {phase}", _C.MAGENTA)

        elif etype == "phase_completed":
            phase = data.get("phase", "?").upper()
            iters = data.get("iterations", "?")
            elapsed = ""
            if self._phase_start_time:
                elapsed = f" ({time.time() - self._phase_start_time:.1f}s)"
            print(f"\n  {_C.GREEN}{_C.BOLD}[PHASE COMPLETE]{_C.RESET} "
                  f"{phase} finished in {iters} iteration(s){elapsed}")

        elif etype == "phase_iteration_limit":
            phase = data.get("phase", "?").upper()
            limit = data.get("limit", "?")
            print(f"\n  {_C.RED}[ITERATION LIMIT]{_C.RESET} "
                  f"{phase} hit max iterations ({limit})")

        elif etype == "llm_call_completed":
            self._llm_calls += 1
            inp = data.get("input_tokens", 0)
            out = data.get("output_tokens", 0)
            total = inp + out
            self._total_tokens += total
            phase = data.get("phase", "?")
            iteration = data.get("iteration", "?")

            _section(f"LLM Call #{self._llm_calls}  (phase={phase}, iter={iteration})")
            _kv("Tokens", f"{inp} in + {out} out = {total} total")
            _kv("Cumulative", f"{self._total_tokens} tokens")

            self._log(f"\n{'='*72}")
            self._log(f"LLM Call #{self._llm_calls}  (phase={phase}, iter={iteration})")
            self._log(f"Tokens: {inp} in + {out} out = {total} total")

            # Show what the LLM said (full, no truncation)
            content = data.get("response_content")
            if content:
                print(f"\n  {_C.CYAN}LLM Response:{_C.RESET}")
                print(_wrap(content))
                self._log(f"\nLLM Response:\n{content}")

            # Show tool calls the LLM requested
            tool_calls = data.get("tool_calls")
            if tool_calls:
                print(f"\n  {_C.BLUE}Tool Calls Requested:{_C.RESET}")
                for tc in tool_calls:
                    name = tc.get("name", "?")
                    args = tc.get("arguments", {})
                    args_str = json.dumps(args, indent=2) if args else "{}"
                    print(f"    {_C.BOLD}{name}{_C.RESET}({args_str})")
                    self._log(f"\nTool Call: {name}({args_str})")

        elif etype == "tool_executed":
            tool = data.get("tool", "?")
            success = data.get("success", False)
            phase = data.get("phase", "?")
            args = data.get("arguments", {})
            result = data.get("result_content", "")

            status_sym = f"{_C.GREEN}OK{_C.RESET}" if success else f"{_C.RED}FAIL{_C.RESET}"
            status_txt = "OK" if success else "FAIL"
            print(f"\n  {_C.BLUE}[TOOL]{_C.RESET} {_C.BOLD}{tool}{_C.RESET} "
                  f"[{status_sym}]  (phase={phase})")

            self._log(f"\n--- Tool: {tool} [{status_txt}] (phase={phase}) ---")

            if args:
                args_display = json.dumps(args, indent=2)
                print(f"    {_C.DIM}Args:{_C.RESET} {args_display}")
                self._log(f"Args: {args_display}")

            if result:
                print(f"    {_C.DIM}Result:{_C.RESET} {result}")
                self._log(f"Result: {result}")

            if not success:
                error = data.get("error", "unknown")
                print(f"    {_C.RED}Error: {error}{_C.RESET}")
                self._log(f"Error: {error}")

            self._tool_calls.append({
                "tool": tool, "success": success, "phase": phase,
            })

        elif etype == "budget_warning":
            tokens = data.get("consumed_tokens", 0)
            max_t = data.get("max_tokens", "?")
            cost = data.get("consumed_cost", 0)
            print(f"\n  {_C.YELLOW}[BUDGET WARNING]{_C.RESET} "
                  f"Tokens: {tokens}/{max_t}, Cost: ${cost:.4f}")

        elif etype == "budget_exceeded":
            reason = data.get("reason", "unknown")
            print(f"\n  {_C.RED}{_C.BOLD}[BUDGET EXCEEDED]{_C.RESET} {reason}")

        elif etype == "context_compacted":
            print(f"  {_C.DIM}[context compacted in {data.get('phase', '?')}]{_C.RESET}")

        elif etype == "agent_completed":
            _banner("AGENT COMPLETED", _C.GREEN)
            summary = data.get("summary", "")
            if summary:
                print(_wrap(summary))
            usage = data.get("token_usage", {})
            if usage:
                _kv("Final tokens", f"{usage.get('input_tokens', 0)} in / "
                     f"{usage.get('output_tokens', 0)} out")
                _kv("Final cost", f"${usage.get('total_cost', 0):.4f}")

        elif etype == "agent_failed":
            _banner("AGENT FAILED", _C.RED)
            _kv("Reason", data.get("reason", "unknown"))

        # Silently skip streaming tokens and other minor events

    def print_summary(self) -> None:
        """Print a final statistics summary."""
        _section("Event Sink Statistics", _C.CYAN)
        _kv("Total LLM calls", self._llm_calls)
        _kv("Total tool calls", len(self._tool_calls))
        _kv("Total tokens observed", self._total_tokens)
        # Tool breakdown
        tools_by_name: Dict[str, int] = {}
        for tc in self._tool_calls:
            tools_by_name[tc["tool"]] = tools_by_name.get(tc["tool"], 0) + 1
        if tools_by_name:
            print(f"  {_C.DIM}Tool call breakdown:{_C.RESET}")
            for name, count in sorted(tools_by_name.items()):
                print(f"    {name}: {count}")


# ---------------------------------------------------------------------------
# Knowledge base — small curated corpus for the demo
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "id": "kb-01",
        "title": "Large Language Models — Capabilities Overview",
        "content": (
            "Large language models (LLMs) like GPT-4, Claude, and Gemini have "
            "demonstrated capabilities in text generation, translation, "
            "summarization, code generation, and multi-step reasoning. "
            "Benchmarks show GPT-4 scoring in the 90th percentile on the bar "
            "exam and achieving 86.4% on MMLU. However, they still struggle "
            "with precise arithmetic, long-horizon planning, and factual "
            "grounding without retrieval augmentation."
        ),
        "source": "AI Capabilities Report 2025",
    },
    {
        "id": "kb-02",
        "title": "Agentic AI — Multi-Agent Architectures",
        "content": (
            "Agentic AI systems use LLMs as reasoning engines within "
            "structured workflows. Common patterns include ReAct (reason + "
            "act), plan-then-execute, and multi-agent orchestration where "
            "specialized agents collaborate. Key challenges are stall "
            "detection (preventing infinite loops), budget management "
            "(controlling token spend), and tool validation (ensuring "
            "agents call tools correctly). Frameworks like PARR address "
            "these with structured lifecycles and protocol-based design."
        ),
        "source": "Multi-Agent Systems Survey, NeurIPS 2025",
    },
    {
        "id": "kb-03",
        "title": "Cost Economics of LLM Inference",
        "content": (
            "LLM inference costs have dropped 10x since 2023. GPT-4o-mini "
            "costs $0.15 per million input tokens compared to GPT-4's "
            "$30 per million in early 2023. Despite lower per-token costs, "
            "agentic workloads amplify spend through multi-turn "
            "conversations and tool-calling loops. A single agentic task "
            "can consume 20-50x more tokens than a single-shot prompt. "
            "Budget enforcement and cost tracking are essential for "
            "production deployments."
        ),
        "source": "LLM Economics Whitepaper, a16z, 2025",
    },
    {
        "id": "kb-04",
        "title": "Retrieval Augmented Generation (RAG)",
        "content": (
            "RAG combines LLMs with external knowledge retrieval to reduce "
            "hallucinations and ground responses in verifiable sources. "
            "Modern RAG pipelines use hybrid search (dense + sparse "
            "embeddings), re-ranking, and chunk-level citation tracking. "
            "Studies show RAG reduces hallucination rates from 15-20% to "
            "3-5% on knowledge-intensive tasks. The main limitation is "
            "retrieval quality — if relevant documents aren't found, the "
            "LLM falls back to parametric knowledge."
        ),
        "source": "RAG Best Practices, Google DeepMind, 2025",
    },
    {
        "id": "kb-05",
        "title": "Safety and Alignment of AI Agents",
        "content": (
            "Autonomous AI agents raise alignment concerns including "
            "goal drift (pursuing sub-goals that diverge from user intent), "
            "reward hacking, and unintended side effects from tool use. "
            "Mitigation strategies include human-in-the-loop oversight, "
            "constrained action spaces (tool whitelisting per phase), "
            "budget limits as hard stops, and structured review phases "
            "where agents self-evaluate before producing final output."
        ),
        "source": "AI Safety for Agentic Systems, Anthropic, 2025",
    },
]


# ---------------------------------------------------------------------------
# Domain tools
# ---------------------------------------------------------------------------

async def search_knowledge_base(query: str, max_results: int = 3) -> str:
    """Search the knowledge base by keyword relevance."""
    query_words = query.lower().split()
    scored = []
    for doc in KNOWLEDGE_BASE:
        text = f"{doc['title']} {doc['content']}".lower()
        score = sum(1 for w in query_words if w in text)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)

    results = scored[:max_results]
    if not results:
        return json.dumps({"results": [], "message": "No documents matched."})

    return json.dumps({
        "results": [
            {
                "id": doc["id"],
                "title": doc["title"],
                "snippet": doc["content"][:150] + "...",
                "source": doc["source"],
                "relevance": round(score / max(len(query_words), 1), 2),
            }
            for score, doc in results
        ],
    })


async def get_document(document_id: str) -> str:
    """Retrieve the full text of a document by its ID."""
    for doc in KNOWLEDGE_BASE:
        if doc["id"] == document_id:
            return json.dumps({
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "source": doc["source"],
            })
    return json.dumps({"error": f"Document '{document_id}' not found."})


SEARCH_TOOL = ToolDef(
    name="search_knowledge_base",
    description=(
        "Search the knowledge base for relevant documents. "
        "Returns ranked results with snippets."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return (default 3).",
            },
        },
        "required": ["query"],
    },
    handler=search_knowledge_base,
    phase_availability=[Phase.ACT],
)

GET_DOC_TOOL = ToolDef(
    name="get_document",
    description="Retrieve the full text of a document by its ID.",
    parameters={
        "type": "object",
        "properties": {
            "document_id": {
                "type": "string",
                "description": "The document ID from search results.",
            },
        },
        "required": ["document_id"],
    },
    handler=get_document,
    phase_availability=[Phase.ACT],
)


# ---------------------------------------------------------------------------
# Output printer
# ---------------------------------------------------------------------------

def print_final_output(output: AgentOutput) -> None:
    """Print a detailed breakdown of the AgentOutput."""
    _banner("FINAL OUTPUT", _C.WHITE)

    _section("Status & Identity")
    _kv("Status", output.status)
    _kv("Role", output.role)
    if output.sub_role:
        _kv("Sub-role", output.sub_role)

    _section("Token Usage")
    _kv("Input tokens", output.token_usage.input_tokens)
    _kv("Output tokens", output.token_usage.output_tokens)
    _kv("Total tokens", output.token_usage.total_tokens)
    _kv("Total cost", f"${output.token_usage.total_cost:.4f}")

    _section("Execution Metadata")
    meta = output.execution_metadata
    _kv("Phases completed", meta.phases_completed)
    _kv("Iterations per phase", meta.iterations_per_phase)
    _kv("Total duration", f"{meta.total_duration_ms:.0f} ms")
    _kv("Tools called", len(meta.tools_called))

    # Show tool call breakdown from metadata
    tool_summary: Dict[str, Dict[str, int]] = {}
    for tc in meta.tools_called:
        name = tc.get("name", "?")
        ok = tc.get("success", True)
        if name not in tool_summary:
            tool_summary[name] = {"ok": 0, "fail": 0}
        tool_summary[name]["ok" if ok else "fail"] += 1
    if tool_summary:
        print(f"\n  {_C.DIM}Tool call breakdown:{_C.RESET}")
        for name in sorted(tool_summary):
            s = tool_summary[name]
            print(f"    {name}: {s['ok']} ok, {s['fail']} fail")

    # Phase outputs (full, no truncation)
    if meta.phase_outputs:
        _section("Phase Outputs")
        for phase_name, phase_text in meta.phase_outputs.items():
            print(f"\n  {_C.BOLD}{phase_name.upper()}:{_C.RESET}")
            print(_wrap(phase_text))

    # Summary
    if output.summary:
        _section("Agent Summary")
        print(_wrap(output.summary))

    # Findings / submitted report
    if output.findings:
        _section("Submitted Report")
        formatted = json.dumps(output.findings, indent=2)
        print(formatted)

    # Errors
    if output.errors:
        _section("Errors")
        for e in output.errors:
            print(f"  {_C.RED}{e.error_type}: {e.message}{_C.RESET}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
) -> AgentOutput:
    config_dir = Path(__file__).parent / "config"

    # The transparent event sink is the star — it shows everything
    event_sink = TransparentEventSink()

    _kv("Log file", str(event_sink._log_path))

    orchestrator = create_orchestrator_from_config(
        config_dir=config_dir,
        tool_registry={
            "search_knowledge_base": SEARCH_TOOL,
            "get_document": GET_DOC_TOOL,
        },
        event_sink=event_sink,
        provider_override=provider,
        model_override=model,
    )

    task_text = task or (
        "Analyze the current state of agentic AI systems. "
        "What are the key capabilities, economic challenges, "
        "and safety considerations?"
    )

    _banner("LIVE DEMO — PARR Framework", _C.WHITE)
    print(f"  {_C.DIM}100% real AI — no mocks, full transparency{_C.RESET}")
    print(f"  {_C.DIM}Every LLM call, tool call, and decision is shown below.{_C.RESET}")
    _kv("Task", task_text)
    _kv("Config", str(config_dir))

    output = await orchestrator.start_workflow(
        task=task_text,
        role="analyst",
    )

    # Print the event sink's own stats
    event_sink.print_summary()

    # Print the full structured output
    print_final_output(output)

    # Save full output to the log file
    event_sink._log(f"\n{'='*72}")
    event_sink._log("FINAL AGENT OUTPUT (JSON):")
    event_sink._log(json.dumps(output.to_dict(), indent=2))
    log_path = event_sink._log_path
    event_sink.close()

    _section("Full output saved to", _C.GREEN)
    print(f"  {log_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="PARR Live Demo — 100% AI, Full Transparency",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help="Override default_provider (e.g. openai, azure_openai, anthropic)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model name (e.g. gpt-4o, gpt-4o-mini)",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Custom task for the agent (default: analyze agentic AI)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable framework debug logging",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)-8s %(name)s: %(message)s")

    asyncio.run(run(provider=args.provider, model=args.model, task=args.task))


if __name__ == "__main__":
    main()
