"""
Research Assistant — End-to-end example.

Demonstrates:
  - Config-driven setup (roles.yaml, models.yaml, budget.yaml)
  - Custom domain tools (search_documents, read_section)
  - Full PARR lifecycle (Plan → Act → Review → Report)
  - Real-time event streaming via LoggingEventSink
  - Budget tracking and cost reporting
  - Structured output validated against JSON Schema

Run with:
    python -m examples.research_assistant.run

Or with a real LLM (requires providers.yaml + environment variables):
    python -m examples.research_assistant.run --live

Override provider or model:
    python -m examples.research_assistant.run --live --provider openai --model gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from parr import (
    AgentOutput,
    BudgetConfig,
    Phase,
    ToolDef,
)
from parr.adapters import LoggingEventSink
from parr.config import load_config, create_orchestrator_from_config

from examples.research_assistant.rag_tools import (
    build_rag_tools,
    SEARCH_KB_TOOL,
    GET_DOCUMENT_TOOL,
)

# ---------------------------------------------------------------------------
# Simulated document corpus (stands in for a real search backend)
# ---------------------------------------------------------------------------

_CORPUS: List[Dict[str, Any]] = [
    {
        "id": "doc1:0",
        "title": "AI in Healthcare — Overview",
        "content": (
            "Artificial intelligence is transforming healthcare through improved "
            "diagnostics, drug discovery, and personalized treatment plans. "
            "Machine learning models can detect diseases from medical images with "
            "accuracy comparable to specialists."
        ),
        "source": "WHO Report 2025",
    },
    {
        "id": "doc1:1",
        "title": "AI in Healthcare — Diagnostics",
        "content": (
            "Deep learning models for radiology have shown 94% sensitivity in "
            "detecting lung nodules, compared to 88% for radiologists. However, "
            "false positive rates remain a concern. Integration with clinical "
            "workflows requires careful validation."
        ),
        "source": "WHO Report 2025, Chapter 3",
    },
    {
        "id": "doc2:0",
        "title": "Privacy Risks of Health AI",
        "content": (
            "AI systems processing patient data face risks including re-identification "
            "of anonymized records, model memorization of training data, and "
            "unauthorized access through adversarial attacks. GDPR Article 22 "
            "restricts automated decision-making in healthcare contexts."
        ),
        "source": "EU Health Data Space Whitepaper 2025",
    },
    {
        "id": "doc2:1",
        "title": "Bias in Medical AI",
        "content": (
            "Studies show that dermatology AI models trained primarily on light-skinned "
            "patients perform significantly worse on darker skin tones. This bias "
            "can lead to misdiagnosis and delayed treatment for underrepresented "
            "populations. Fairness auditing is now recommended before deployment."
        ),
        "source": "Lancet Digital Health, 2025",
    },
    {
        "id": "doc3:0",
        "title": "Regulatory Landscape — EU AI Act",
        "content": (
            "The EU AI Act classifies medical AI systems as high-risk, requiring "
            "conformity assessments, human oversight, and transparency documentation. "
            "Providers must establish risk management systems and maintain technical "
            "documentation throughout the AI system's lifecycle."
        ),
        "source": "EU AI Act Implementation Guide 2025",
    },
    {
        "id": "doc4:0",
        "title": "DPIA Phases - ICO Guidance",
        "content": (
            "A standard Data Protection Impact Assessment (DPIA) process includes: "
            "(1) identify whether a DPIA is needed, (2) describe the processing and data flows, "
            "(3) consult stakeholders, (4) assess necessity and proportionality, "
            "(5) identify privacy risks, (6) define mitigating controls, "
            "(7) document decisions and sign-off, and (8) review outcomes after implementation."
        ),
        "source": "ICO Guide to UK GDPR DPIA",
    },
    {
        "id": "doc4:1",
        "title": "DPIA Method Steps - EDPB",
        "content": (
            "EDPB guidance describes a comparable DPIA flow: preliminary screening, "
            "processing description and purpose analysis, risk assessment for data subjects, "
            "selection of safeguards, decision on residual risk and consultation where required, "
            "and periodic review."
        ),
        "source": "EDPB Guidelines on DPIA",
    },
]


# ---------------------------------------------------------------------------
# Domain tools — what agents can actually call
# ---------------------------------------------------------------------------

async def search_documents(query: str, top_k: int = 3) -> str:
    """Search the document corpus by keyword matching."""
    query_lower = query.lower()
    scored = []
    for doc in _CORPUS:
        text = f"{doc['title']} {doc['content']}".lower()
        # Simple relevance: count query word matches
        score = sum(1 for word in query_lower.split() if word in text)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = scored[:top_k]

    if not results:
        return json.dumps({"results": [], "message": "No documents matched the query."})

    return json.dumps({
        "results": [
            {
                "section_id": doc["id"],
                "title": doc["title"],
                "summary": doc["content"][:200] + "...",
                "source": doc["source"],
                "relevance_score": score / max(len(query_lower.split()), 1),
            }
            for score, doc in results
        ]
    })


async def read_section(section_id: str) -> str:
    """Read the full text of a document section."""
    for doc in _CORPUS:
        if doc["id"] == section_id:
            return json.dumps({
                "section_id": doc["id"],
                "title": doc["title"],
                "full_text": doc["content"],
                "source": doc["source"],
            })
    return json.dumps({"error": f"Section '{section_id}' not found."})


async def read_multiple_sections(section_ids: List[str]) -> str:
    """Read multiple document sections in one call. Deduplicates automatically."""
    seen = set()
    results = []
    not_found = []
    for sid in section_ids:
        if sid in seen:
            continue
        seen.add(sid)
        found = False
        for doc in _CORPUS:
            if doc["id"] == sid:
                results.append({
                    "section_id": doc["id"],
                    "title": doc["title"],
                    "full_text": doc["content"],
                    "source": doc["source"],
                })
                found = True
                break
        if not found:
            not_found.append(sid)
    response: Dict[str, Any] = {"sections": results}
    if not_found:
        response["not_found"] = not_found
    return json.dumps(response)


# Build ToolDef objects for the domain tools
SEARCH_TOOL = ToolDef(
    name="search_documents",
    description="Search the document corpus. Returns ranked results with summaries.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results (default: 3).",
            },
        },
        "required": ["query"],
    },
    handler=search_documents,
    phase_availability=[Phase.ACT],
)

READ_TOOL = ToolDef(
    name="read_section",
    description="Read the full text of a document section by its ID.",
    parameters={
        "type": "object",
        "properties": {
            "section_id": {
                "type": "string",
                "description": "The section ID returned by search_documents.",
            },
        },
        "required": ["section_id"],
    },
    handler=read_section,
    phase_availability=[Phase.ACT],
)

READ_MULTI_TOOL = ToolDef(
    name="read_multiple_sections",
    description=(
        "Read multiple document sections in one call. More efficient than "
        "calling read_section repeatedly. Deduplicates IDs automatically."
    ),
    parameters={
        "type": "object",
        "properties": {
            "section_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of section IDs returned by search_documents.",
            },
        },
        "required": ["section_ids"],
    },
    handler=read_multiple_sections,
    phase_availability=[Phase.ACT],
)


# ---------------------------------------------------------------------------
# Mock LLM for offline demo
# ---------------------------------------------------------------------------

def _build_mock_llm():
    """Build a MockToolCallingLLM that drives the full research workflow."""
    from parr.tests.mock_llm import (
        MockToolCallingLLM,
        make_text_response,
        make_tool_call_response,
    )

    return MockToolCallingLLM({
        Phase.PLAN: [
            # Agent creates a todo list
            make_tool_call_response("create_todo_list", {
                "items": [
                    {"description": "Search for documents about AI in healthcare"},
                    {"description": "Read the most relevant results in detail"},
                    {"description": "Log findings with sources"},
                ]
            }),
            make_text_response("Plan created. Ready to execute."),
        ],
        Phase.ACT: [
            # Step 1: search
            make_tool_call_response("search_documents", {
                "query": "AI healthcare diagnostics privacy risks",
                "top_k": 5,
            }),
            # Step 2: read a result
            make_tool_call_response("read_section", {"section_id": "doc1:1"}),
            # Step 3: log finding
            make_tool_call_response("log_finding", {
                "category": "diagnostics",
                "content": "Deep learning models show 94% sensitivity for lung nodule detection vs 88% for radiologists.",
                "source": "WHO Report 2025, Chapter 3",
                "confidence": "high",
            }),
            # Step 4: read another result
            make_tool_call_response("read_section", {"section_id": "doc2:0"}),
            # Step 5: log finding
            make_tool_call_response("log_finding", {
                "category": "privacy",
                "content": "AI systems face re-identification risks, model memorization, and adversarial attacks on patient data.",
                "source": "EU Health Data Space Whitepaper 2025",
                "confidence": "high",
            }),
            # Step 6: read bias doc
            make_tool_call_response("read_section", {"section_id": "doc2:1"}),
            # Step 7: log finding
            make_tool_call_response("log_finding", {
                "category": "bias",
                "content": "Dermatology AI models perform worse on darker skin tones due to training data imbalance.",
                "source": "Lancet Digital Health, 2025",
                "confidence": "high",
            }),
            # Step 8: mark todos complete
            make_tool_call_response("mark_todo_complete", {"item_index": 0, "summary": "Found 5 relevant documents"}),
            make_tool_call_response("mark_todo_complete", {"item_index": 1, "summary": "Read 3 key sections in detail"}),
            make_tool_call_response("mark_todo_complete", {"item_index": 2, "summary": "Logged 3 findings with sources"}),
            make_text_response("All tasks completed."),
        ],
        Phase.REVIEW: [
            make_tool_call_response("review_checklist", {
                "items": [
                    {"criterion": "All search steps completed", "rating": "pass", "justification": "Searched and read multiple documents."},
                    {"criterion": "Findings logged with sources", "rating": "pass", "justification": "3 findings logged with citations."},
                    {"criterion": "Key risks identified", "rating": "pass", "justification": "Diagnostics, privacy, and bias risks covered."},
                ]
            }),
            make_text_response("REVIEW_PASSED — all criteria met."),
        ],
        Phase.REPORT: [
            make_tool_call_response("submit_report", {
                "summary": "AI in healthcare shows strong diagnostic potential but carries significant privacy and bias risks that require regulatory attention.",
                "key_findings": [
                    {"finding": "Deep learning achieves 94% sensitivity for lung nodule detection, outperforming radiologists at 88%.", "source": "WHO Report 2025, Chapter 3"},
                    {"finding": "Patient data faces re-identification, memorization, and adversarial attack risks.", "source": "EU Health Data Space Whitepaper 2025"},
                    {"finding": "Dermatology AI models exhibit racial bias due to training data imbalance.", "source": "Lancet Digital Health, 2025"},
                ],
                "gaps": ["No data found on AI adoption rates by country", "Cost-effectiveness studies not available in corpus"],
                "recommendations": ["Mandate fairness audits before clinical deployment", "Implement differential privacy for training data", "Require human-in-the-loop for high-stakes diagnoses"],
            }),
            make_text_response("Report submitted."),
        ],
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_output(output: AgentOutput) -> None:
    """Print the agent output."""
    print("\n" + "=" * 70)
    print("WORKFLOW RESULT")
    print("=" * 70)
    print(f"  Status     : {output.status}")
    print(f"  Role       : {output.role}")
    print(f"  Tokens     : {output.token_usage.input_tokens} in / {output.token_usage.output_tokens} out")
    print(f"  Cost       : ${output.token_usage.total_cost:.4f}")

    if output.summary:
        print(f"\n  Summary:\n    {output.summary[:300]}")

    if output.findings:
        print("\n  Report:")
        print(json.dumps(output.findings, indent=2)[:1500])

    if output.errors:
        print("\n  Errors:")
        for e in output.errors:
            print(f"    {e.error_type}: {e.message}")

    print("=" * 70)


async def run(
    live: bool = False,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> AgentOutput:
    """Run the research assistant example."""
    config_dir = Path(__file__).parent / "config"

    # Base tool registry — in-memory corpus tools + Weaviate RAG tools
    tool_registry = {
        "search_documents": SEARCH_TOOL,
        "read_section": READ_TOOL,
        "read_multiple_sections": READ_MULTI_TOOL,
        **build_rag_tools(),
    }

    if live:
        # LLM is auto-created from providers.yaml + environment variables.
        # provider/model overrides are passed through to the config loader.
        orchestrator = create_orchestrator_from_config(
            config_dir=config_dir,
            tool_registry=tool_registry,
            event_sink=LoggingEventSink(),
            provider_override=provider,
            model_override=model,
        )
        print("Using live LLM (provider from providers.yaml)")
    else:
        llm = _build_mock_llm()
        print("Using mock LLM (offline demo)")
        orchestrator = create_orchestrator_from_config(
            config_dir=config_dir,
            tool_registry=tool_registry,
            llm=llm,
            event_sink=LoggingEventSink(),
        )

    print(f"Starting workflow: 'Research AI in healthcare'")
    print("-" * 70)

    # Run the workflow
    output = await orchestrator.start_workflow(
        task="Research the current state of AI in healthcare. Focus on diagnostic capabilities, privacy risks, and regulatory requirements.",
        role="researcher",
    )

    _print_output(output)
    return output


def main():
    parser = argparse.ArgumentParser(description="Research Assistant — PARR Framework Example")
    parser.add_argument("--live", action="store_true", help="Use real LLM via providers.yaml")
    parser.add_argument("--provider", type=str, default=None, help="Override default_provider (e.g., openai, azure_openai, anthropic)")
    parser.add_argument("--model", type=str, default=None, help="Override model name (e.g., gpt-4o, claude-3-5-sonnet)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)-8s %(name)s: %(message)s")

    asyncio.run(run(live=args.live, provider=args.provider, model=args.model))


if __name__ == "__main__":
    main()
