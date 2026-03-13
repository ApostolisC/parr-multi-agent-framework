"""
Generate sample session data and launch the Debug UI for testing.

Usage::

    python -m parr.debug_ui.test_ui
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_test_data(persist_dir: Path) -> None:
    """Generate two sample sessions: one completed, one running."""
    persist_dir.mkdir(parents=True, exist_ok=True)

    # ── Session 1: Completed research workflow ──────────────────────
    wf1_id = str(uuid.uuid4())
    task1_id = str(uuid.uuid4())
    child_task_id = str(uuid.uuid4())

    wf1_dir = persist_dir / wf1_id
    _write(wf1_dir / "workflow.json", {
        "workflow_id": wf1_id,
        "status": "completed",
        "root_task_id": task1_id,
        "budget": {"max_tokens": 100000, "max_cost": 5.0, "max_duration_ms": None},
        "created_at": _utc_iso(),
    })

    _write(wf1_dir / "agent.json", {
        "task_id": task1_id,
        "agent_id": str(uuid.uuid4()),
        "role": "researcher",
        "sub_role": None,
        "task": "Research the current state of AI in healthcare. Focus on diagnostic capabilities, privacy risks, and regulatory requirements.",
        "status": "completed",
        "depth": 0,
        "parent_task_id": None,
        "model": "gpt-4o",
    })

    _write(wf1_dir / "conversation.json", {
        "plan": {
            "content": "I need to research AI in healthcare across three dimensions: diagnostic capabilities, privacy risks, and regulatory requirements. Let me start by creating a structured research plan with search tasks for each area, then systematically work through each one.\n\nI'll begin by searching the document corpus for relevant materials on each topic, prioritizing recent publications and authoritative sources like WHO reports and EU regulatory documents.",
            "iterations": 2,
            "hit_iteration_limit": False,
            "tool_calls_count": 1,
        },
        "act": {
            "content": "I've completed all research tasks. Here's what I found:\n\n1. **Diagnostic Capabilities**: Deep learning models achieve 94% sensitivity for lung nodule detection, outperforming radiologists at 88%. Retinal scan AI can detect diabetic retinopathy with 97% accuracy.\n\n2. **Privacy Risks**: Major concerns include patient data re-identification through model memorization, adversarial attacks on training data, and lack of consent frameworks for federated learning.\n\n3. **Regulatory Landscape**: The EU AI Act classifies medical AI as high-risk, requiring conformity assessments, human oversight, and transparency documentation. FDA has cleared 521 AI/ML medical devices as of 2025.\n\nAll todo items have been completed and findings logged with citations.",
            "iterations": 8,
            "hit_iteration_limit": False,
            "tool_calls_count": 7,
        },
        "review": {
            "content": "Reviewing my work against the research criteria:\n\n\u2705 All search tasks completed — searched for diagnostics, privacy, and regulatory documents\n\u2705 Findings logged with proper citations — 3 findings with WHO, EU, and Lancet sources\n\u2705 Key risks identified — covered diagnostic bias, privacy vulnerabilities, and regulatory gaps\n\u2705 Multiple authoritative sources used — WHO Report 2025, EU AI Act, Lancet Digital Health\n\nREVIEW_PASSED — all criteria met. The research covers the requested scope with high-confidence findings.",
            "iterations": 2,
            "hit_iteration_limit": False,
            "tool_calls_count": 1,
        },
        "report": {
            "content": "Submitting final report with key findings, evidence-based recommendations, and source citations. The report covers all three requested areas with actionable policy recommendations for healthcare AI deployment.",
            "iterations": 2,
            "hit_iteration_limit": False,
            "tool_calls_count": 1,
        },
    })

    _write(wf1_dir / "tool_calls.json", [
        {
            "name": "create_todo_list",
            "phase": "plan",
            "arguments": {
                "items": [
                    {"description": "Search for documents about AI in healthcare"},
                    {"description": "Read the most relevant results in detail"},
                    {"description": "Log findings with sources"},
                ]
            },
            "result": "Created 3 todo items.",
            "success": True,
        },
        {
            "name": "search_documents",
            "phase": "act",
            "arguments": {"query": "AI healthcare diagnostics privacy risks", "top_k": 5},
            "result_content": '{"results": [{"title": "AI in Healthcare — Overview", "relevance_score": 0.85}, {"title": "Privacy Risks of Health AI", "relevance_score": 0.78}]}',
            "success": True,
        },
        {
            "name": "read_section",
            "phase": "act",
            "arguments": {"section_id": "doc1:1"},
            "result_content": '{"title": "AI in Healthcare — Diagnostics", "full_text": "Deep learning models for radiology have shown 94% sensitivity in detecting lung nodules."}',
            "success": True,
        },
        {
            "name": "log_finding",
            "phase": "act",
            "arguments": {
                "category": "diagnostics",
                "content": "Deep learning models show 94% sensitivity for lung nodule detection vs 88% for radiologists.",
                "source": "WHO Report 2025, Chapter 3",
                "confidence": "high",
            },
            "result": "Finding logged.",
            "success": True,
        },
        {
            "name": "read_section",
            "phase": "act",
            "arguments": {"section_id": "doc2:0"},
            "result_content": '{"title": "Privacy Risks of Health AI", "full_text": "AI systems face re-identification risks and model memorization."}',
            "success": True,
        },
        {
            "name": "log_finding",
            "phase": "act",
            "arguments": {
                "category": "privacy",
                "content": "AI systems face re-identification risks, model memorization, and adversarial attacks on patient data.",
                "source": "EU Health Data Space Whitepaper 2025",
                "confidence": "high",
            },
            "result": "Finding logged.",
            "success": True,
        },
        {
            "name": "mark_todo_complete",
            "phase": "act",
            "arguments": {"item_index": 0, "summary": "Found 5 relevant documents"},
            "result": "Todo item 0 marked complete.",
            "success": True,
        },
        {
            "name": "review_checklist",
            "phase": "review",
            "arguments": {
                "items": [
                    {"criterion": "All search steps completed", "rating": "pass", "justification": "Searched and read multiple documents."},
                    {"criterion": "Findings logged with sources", "rating": "pass", "justification": "3 findings logged with citations."},
                    {"criterion": "Key risks identified", "rating": "pass", "justification": "Diagnostics, privacy, and bias risks covered."},
                ]
            },
            "result": "Review checklist saved.",
            "success": True,
        },
        {
            "name": "submit_report",
            "phase": "report",
            "arguments": {
                "summary": "AI in healthcare shows strong diagnostic potential but carries significant privacy and bias risks.",
                "key_findings": [
                    {"finding": "Deep learning achieves 94% sensitivity for lung nodule detection.", "source": "WHO Report 2025"},
                    {"finding": "Patient data faces re-identification and memorization risks.", "source": "EU Whitepaper 2025"},
                ],
                "recommendations": ["Mandate fairness audits", "Implement differential privacy"],
            },
            "result": "Report submitted.",
            "success": True,
        },
    ])

    _write(wf1_dir / "memory" / "todo_list.json", [
        {"index": 0, "description": "Search for documents about AI in healthcare", "priority": "high", "completed": True, "completion_summary": "Found 5 relevant documents"},
        {"index": 1, "description": "Read the most relevant results in detail", "priority": "high", "completed": True, "completion_summary": "Read 3 key sections"},
        {"index": 2, "description": "Log findings with sources", "priority": "medium", "completed": True, "completion_summary": "Logged 3 findings with citations"},
    ])

    _write(wf1_dir / "memory" / "findings.json", [
        {"category": "diagnostics", "content": "Deep learning models show 94% sensitivity for lung nodule detection.", "source": "WHO Report 2025, Chapter 3", "confidence": "high"},
        {"category": "privacy", "content": "AI systems face re-identification risks and model memorization.", "source": "EU Health Data Space Whitepaper 2025", "confidence": "high"},
        {"category": "bias", "content": "Dermatology AI models perform worse on darker skin tones.", "source": "Lancet Digital Health, 2025", "confidence": "high"},
    ])

    _write(wf1_dir / "memory" / "review.json", [
        {"criterion": "All search steps completed", "rating": "pass", "justification": "Searched and read multiple documents."},
        {"criterion": "Findings logged with sources", "rating": "pass", "justification": "3 findings logged with citations."},
        {"criterion": "Key risks identified", "rating": "pass", "justification": "Diagnostics, privacy, and bias risks covered."},
    ])

    _write(wf1_dir / "memory" / "report.json", {
        "summary": "AI in healthcare shows strong diagnostic potential but carries significant privacy and bias risks.",
        "key_findings": [
            {"finding": "Deep learning achieves 94% sensitivity.", "source": "WHO Report 2025"},
            {"finding": "Patient data faces re-identification risks.", "source": "EU Whitepaper 2025"},
        ],
        "recommendations": ["Mandate fairness audits", "Implement differential privacy"],
    })

    _write(wf1_dir / "output.json", {
        "task_id": task1_id,
        "agent_id": str(uuid.uuid4()),
        "role": "researcher",
        "sub_role": None,
        "status": "completed",
        "summary": "AI in healthcare shows strong diagnostic potential but carries significant privacy and bias risks that require regulatory attention.",
        "findings": {
            "summary": "AI in healthcare shows strong diagnostic potential but carries significant privacy and bias risks.",
            "key_findings": [
                {"finding": "Deep learning achieves 94% sensitivity for lung nodule detection.", "source": "WHO Report 2025"},
                {"finding": "Patient data faces re-identification and memorization risks.", "source": "EU Whitepaper 2025"},
                {"finding": "Dermatology AI models exhibit racial bias.", "source": "Lancet Digital Health, 2025"},
            ],
        },
        "artifacts": [],
        "errors": [],
        "recommendations": ["Mandate fairness audits before clinical deployment", "Implement differential privacy for training data"],
        "token_usage": {
            "input_tokens": 24560,
            "output_tokens": 3840,
            "total_tokens": 28400,
            "total_cost": 0.3720,
        },
        "execution_metadata": {
            "phases_completed": ["plan", "act", "review", "report"],
            "iterations_per_phase": {"plan": 2, "act": 8, "review": 2, "report": 2},
            "sub_agents_spawned": [child_task_id],
            "tools_called": [],
            "total_duration_ms": 45200.0,
            "phase_outputs": {
                "plan": "Research plan created with 3 investigation areas: (1) AI diagnostic capabilities and accuracy benchmarks, (2) Privacy and data protection risks in health AI systems, (3) Regulatory landscape including EU AI Act and FDA clearance status. Each area has targeted search queries and specific document sections to review.",
                "act": "Completed all research tasks across 3 areas. Key findings: Deep learning achieves 94% sensitivity for lung nodule detection (WHO 2025). Patient data faces re-identification and model memorization risks (EU Whitepaper 2025). Dermatology AI shows racial bias in skin tone analysis (Lancet 2025). EU AI Act classifies medical AI as high-risk requiring conformity assessments. All 3 todo items completed with citations logged.",
                "review": "REVIEW_PASSED — All 3 criteria met: (1) All search and read steps completed across diagnostic, privacy, and regulatory domains. (2) 3 findings logged with authoritative citations from WHO, EU, and Lancet. (3) Key risks identified covering diagnostic bias, privacy vulnerabilities, and regulatory compliance gaps.",
                "report": "# AI in Healthcare — Research Report\n\n## Summary\nAI in healthcare shows strong diagnostic potential (94% sensitivity for lung nodules) but carries significant privacy risks (re-identification, model memorization) and faces strict regulatory requirements (EU AI Act high-risk classification).\n\n## Key Findings\n1. Deep learning achieves 94% sensitivity for lung nodule detection vs 88% for radiologists (WHO Report 2025)\n2. Patient data faces re-identification and adversarial attack risks (EU Health Data Space Whitepaper 2025)\n3. Dermatology AI models exhibit racial bias on darker skin tones (Lancet Digital Health 2025)\n\n## Recommendations\n- Mandate fairness audits before clinical deployment\n- Implement differential privacy for training data\n- Establish conformity assessment pipelines for EU AI Act compliance",
            },
        },
    })

    # Sub-agent for the completed session
    child_dir = wf1_dir / "sub_agents" / f"deep_dive_{child_task_id[:8]}"
    _write(wf1_dir / "sub_agents.json", [
        {
            "task_id": child_task_id,
            "agent_id": str(uuid.uuid4()),
            "role": "researcher",
            "sub_role": "deep_dive",
            "task_description": "Deep dive into EU AI Act implications for healthcare AI systems",
            "status": "completed",
        }
    ])

    _write(child_dir / "agent.json", {
        "task_id": child_task_id,
        "agent_id": str(uuid.uuid4()),
        "role": "researcher",
        "sub_role": "deep_dive",
        "task": "Deep dive into EU AI Act implications for healthcare AI systems. Analyze conformity requirements and human oversight obligations.",
        "status": "completed",
        "depth": 1,
        "parent_task_id": task1_id,
        "model": "gpt-4o-mini",
    })

    _write(child_dir / "conversation.json", {
        "plan": {"content": "I'll focus my deep dive on the EU AI Act's specific implications for healthcare AI. Key areas to investigate: conformity assessment requirements, human oversight obligations, and transparency mandates for high-risk medical AI systems.", "iterations": 1, "hit_iteration_limit": False, "tool_calls_count": 1},
        "act": {"content": "Completed analysis of the EU AI Act healthcare implications. The Act classifies all medical AI as high-risk under Annex III, requiring:\n\n1. **Conformity assessments** — third-party audits for diagnostic AI\n2. **Human oversight** — clinician-in-the-loop mandatory for treatment decisions\n3. **Data governance** — training data must be representative and bias-tested\n4. **Transparency** — patients must be informed when AI assists in diagnosis\n\nTimeline: Full enforcement begins January 2026 with 6-month grace period for existing devices.", "iterations": 3, "hit_iteration_limit": False, "tool_calls_count": 2},
        "review": {"content": "REVIEW_PASSED — Analysis covers all key regulatory dimensions with specific article references and implementation timelines.", "iterations": 1, "hit_iteration_limit": False, "tool_calls_count": 1},
        "report": {"content": "Submitting deep dive report on EU AI Act healthcare implications with 4 key regulatory requirements and implementation timeline.", "iterations": 1, "hit_iteration_limit": False, "tool_calls_count": 1},
    })

    _write(child_dir / "tool_calls.json", [
        {"name": "create_todo_list", "phase": "plan", "arguments": {"items": [{"description": "Analyze EU AI Act requirements"}]}, "result": "Created 1 todo item.", "success": True},
        {"name": "search_documents", "phase": "act", "arguments": {"query": "EU AI Act healthcare"}, "result_content": '{"results": [{"title": "Regulatory Landscape — EU AI Act"}]}', "success": True},
        {"name": "read_section", "phase": "act", "arguments": {"section_id": "doc3:0"}, "result_content": '{"title": "Regulatory Landscape", "full_text": "The EU AI Act classifies medical AI as high-risk..."}', "success": True},
    ])

    _write(child_dir / "memory" / "todo_list.json", [
        {"index": 0, "description": "Analyze EU AI Act requirements", "priority": "high", "completed": True, "completion_summary": "Full analysis complete"},
    ])
    _write(child_dir / "memory" / "findings.json", [
        {"category": "regulatory", "content": "EU AI Act classifies medical AI as high-risk, requiring conformity assessments.", "source": "EU AI Act Implementation Guide 2025", "confidence": "high"},
    ])

    _write(child_dir / "output.json", {
        "task_id": child_task_id,
        "agent_id": str(uuid.uuid4()),
        "role": "researcher",
        "sub_role": "deep_dive",
        "status": "completed",
        "summary": "EU AI Act classifies medical AI as high-risk with strict conformity and oversight requirements.",
        "findings": {"key_points": ["Conformity assessments required", "Human oversight mandatory"]},
        "artifacts": [],
        "errors": [],
        "recommendations": [],
        "token_usage": {"input_tokens": 8200, "output_tokens": 1400, "total_tokens": 9600, "total_cost": 0.089},
        "execution_metadata": {
            "phases_completed": ["plan", "act", "review", "report"],
            "iterations_per_phase": {"plan": 1, "act": 3, "review": 1, "report": 1},
            "sub_agents_spawned": [],
            "tools_called": [],
            "total_duration_ms": 12400.0,
        },
    })

    # ── Session 2: Running workflow ─────────────────────────────────
    wf2_id = str(uuid.uuid4())
    task2_id = str(uuid.uuid4())
    wf2_dir = persist_dir / wf2_id

    _write(wf2_dir / "workflow.json", {
        "workflow_id": wf2_id,
        "status": "running",
        "root_task_id": task2_id,
        "budget": {"max_tokens": 50000, "max_cost": 2.0, "max_duration_ms": 120000},
        "created_at": _utc_iso(),
    })

    _write(wf2_dir / "agent.json", {
        "task_id": task2_id,
        "agent_id": str(uuid.uuid4()),
        "role": "analyst",
        "sub_role": None,
        "task": "Analyze the competitive landscape for electric vehicle manufacturers in Europe, focusing on market share, technology adoption, and regulatory impact.",
        "status": "running",
        "depth": 0,
        "parent_task_id": None,
        "model": "claude-3-5-sonnet",
    })

    _write(wf2_dir / "conversation.json", {
        "plan": {
            "content": "Created analysis plan with 4 research areas.",
            "iterations": 2,
            "hit_iteration_limit": False,
            "tool_calls_count": 1,
        },
        "act": {
            "content": None,
            "iterations": 3,
            "hit_iteration_limit": False,
            "tool_calls_count": 2,
        },
    })

    _write(wf2_dir / "tool_calls.json", [
        {"name": "create_todo_list", "phase": "plan", "arguments": {"items": [{"description": "Research market share data"}, {"description": "Analyze technology trends"}, {"description": "Review regulatory impact"}]}, "result": "Created 3 todo items.", "success": True},
        {"name": "search_documents", "phase": "act", "arguments": {"query": "electric vehicle market share Europe 2025"}, "result_content": '{"results": [{"title": "EV Market Report Q1 2025"}]}', "success": True},
        {"name": "read_section", "phase": "act", "arguments": {"section_id": "ev1:0"}, "result_content": None, "success": False, "error": "Section 'ev1:0' not found in corpus."},
    ])

    _write(wf2_dir / "memory" / "todo_list.json", [
        {"index": 0, "description": "Research market share data", "priority": "high", "completed": False, "completion_summary": None},
        {"index": 1, "description": "Analyze technology trends", "priority": "high", "completed": False, "completion_summary": None},
        {"index": 2, "description": "Review regulatory impact", "priority": "medium", "completed": False, "completion_summary": None},
    ])
    _write(wf2_dir / "memory" / "findings.json", [])

    # ── Session 3: Failed workflow ──────────────────────────────────
    wf3_id = str(uuid.uuid4())
    task3_id = str(uuid.uuid4())
    wf3_dir = persist_dir / wf3_id

    _write(wf3_dir / "workflow.json", {
        "workflow_id": wf3_id,
        "status": "failed",
        "root_task_id": task3_id,
        "budget": {"max_tokens": 10000, "max_cost": 0.5},
        "created_at": _utc_iso(),
    })

    _write(wf3_dir / "agent.json", {
        "task_id": task3_id,
        "agent_id": str(uuid.uuid4()),
        "role": "researcher",
        "sub_role": None,
        "task": "Quick survey of quantum computing applications.",
        "status": "failed",
        "depth": 0,
        "parent_task_id": None,
        "model": "gpt-4o",
    })

    _write(wf3_dir / "conversation.json", {
        "plan": {
            "content": "Planning research on quantum computing.",
            "iterations": 5,
            "hit_iteration_limit": True,
            "tool_calls_count": 0,
        },
    })

    _write(wf3_dir / "tool_calls.json", [])

    _write(wf3_dir / "output.json", {
        "task_id": task3_id,
        "agent_id": str(uuid.uuid4()),
        "role": "researcher",
        "status": "failed",
        "summary": "Agent failed during plan phase — hit iteration limit without producing actionable plan.",
        "findings": {},
        "artifacts": [],
        "errors": [
            {"source": "agent", "error_type": "StallDetected", "message": "Plan phase hit iteration limit (5). Agent may be stuck in a loop.", "name": "stall_error"},
            {"source": "system", "error_type": "BudgetWarning", "message": "Used 8500/10000 tokens (85%) before failure.", "name": "budget_warning"},
        ],
        "recommendations": ["Increase plan phase iteration limit", "Improve system prompt specificity"],
        "token_usage": {"input_tokens": 6200, "output_tokens": 2300, "total_tokens": 8500, "total_cost": 0.112},
        "execution_metadata": {
            "phases_completed": ["plan"],
            "iterations_per_phase": {"plan": 5},
            "sub_agents_spawned": [],
            "tools_called": [],
            "total_duration_ms": 8900.0,
        },
    })

    _write(wf3_dir / "memory" / "todo_list.json", [])

    # ── Session 4: Budget-exceeded with stale sub-agents ────────────
    # Mimics a real failure where budget runs out mid-execution:
    # - workflow.json says "failed" but NO output.json exists
    # - agent.json still shows status "running" (never updated)
    # - sub_agents.json shows "failed" but child agent.json says "running"
    wf4_id = str(uuid.uuid4())
    task4_id = str(uuid.uuid4())
    sa4a_id = str(uuid.uuid4())
    sa4b_id = str(uuid.uuid4())
    wf4_dir = persist_dir / wf4_id

    _write(wf4_dir / "workflow.json", {
        "workflow_id": wf4_id,
        "status": "failed",
        "root_task_id": task4_id,
        "budget": {"max_tokens": 50000, "max_cost": 1.0, "max_duration_ms": 120000},
        "created_at": _utc_iso(),
    })

    _write(wf4_dir / "agent.json", {
        "task_id": task4_id,
        "agent_id": str(uuid.uuid4()),
        "role": "researcher",
        "sub_role": None,
        "task": "What is a data protection impact assessment (DPIA)? How is it conducted?",
        "status": "running",  # Intentionally stale — never updated before crash
        "depth": 0,
        "parent_task_id": None,
        "model": "gpt-4o",
    })

    _write(wf4_dir / "conversation.json", {
        "plan": {
            "content": "I'll research DPIAs by:\n1. Searching for DPIA methodology documents\n2. Understanding the legal basis under GDPR\n3. Reviewing practical implementation guides\n4. Comparing different assessment frameworks",
            "iterations": 3,
            "hit_iteration_limit": False,
            "tool_calls_count": 4,
        },
    })

    _write(wf4_dir / "tool_calls.json", [
        {"name": "create_todo_list", "phase": "plan", "arguments": {"items": [{"description": "Search DPIA methodology"}, {"description": "Review GDPR basis"}, {"description": "Compare frameworks"}]}, "result": "Created 3 todo items.", "success": True},
        {"name": "create_todo_list", "phase": "plan", "arguments": {"items": [{"description": "Deep dive GDPR Art 35"}]}, "result": "Created 1 todo item.", "success": True},
        {"name": "create_todo_list", "phase": "plan", "arguments": {"items": [{"description": "Review other methodologies"}]}, "result": "Created 1 todo item.", "success": True},
        {"name": "update_todo_list", "phase": "plan", "arguments": {"item_index": 0, "priority": "high"}, "result": "Updated todo item.", "success": True},
    ])

    _write(wf4_dir / "sub_agents.json", [
        {"task_id": sa4a_id, "agent_id": str(uuid.uuid4()), "role": "researcher", "sub_role": "deep_dive", "task_description": "Deep dive into GDPR Article 35", "status": "failed"},
        {"task_id": sa4b_id, "agent_id": str(uuid.uuid4()), "role": "researcher", "sub_role": "deep_dive", "task_description": "Deep dive into other methodologies", "status": "failed"},
    ])

    sa4a_dir = wf4_dir / "sub_agents" / f"researcher_{sa4a_id[:8]}"
    _write(sa4a_dir / "agent.json", {
        "task_id": sa4a_id,
        "agent_id": str(uuid.uuid4()),
        "role": "researcher",
        "sub_role": "deep_dive",
        "task": "Deep dive into GDPR Article 35 requirements for DPIAs",
        "status": "running",  # Stale — should be "failed"
        "depth": 1,
        "parent_task_id": task4_id,
        "model": "gpt-4o-mini",
    })
    _write(sa4a_dir / "conversation.json", {
        "plan": {"content": "I'll analyze GDPR Article 35 in detail.", "iterations": 1, "hit_iteration_limit": False, "tool_calls_count": 1},
    })
    _write(sa4a_dir / "tool_calls.json", [
        {"name": "create_todo_list", "phase": "plan", "arguments": {"items": [{"description": "Analyze Art 35"}]}, "result": "Created 1 todo item.", "success": True},
    ])
    _write(sa4a_dir / "memory" / "todo_list.json", [])

    sa4b_dir = wf4_dir / "sub_agents" / f"researcher_{sa4b_id[:8]}"
    _write(sa4b_dir / "agent.json", {
        "task_id": sa4b_id,
        "agent_id": str(uuid.uuid4()),
        "role": "researcher",
        "sub_role": "deep_dive",
        "task": "Deep dive into alternative DPIA methodologies (NIST, ISO 29134)",
        "status": "running",  # Stale
        "depth": 1,
        "parent_task_id": task4_id,
        "model": "gpt-4o-mini",
    })
    _write(sa4b_dir / "conversation.json", {
        "plan": {"content": "Researching NIST and ISO 29134 methodologies for privacy impact assessments.", "iterations": 1, "hit_iteration_limit": False, "tool_calls_count": 0},
    })
    _write(sa4b_dir / "tool_calls.json", [])
    _write(sa4b_dir / "memory" / "todo_list.json", [])

    # No output.json for wf4 or its sub-agents — budget exceeded before persistence

    _write(wf4_dir / "memory" / "todo_list.json", [
        {"index": 0, "description": "Search DPIA methodology", "priority": "high", "completed": False},
        {"index": 1, "description": "Review GDPR basis", "priority": "medium", "completed": False},
        {"index": 2, "description": "Compare frameworks", "priority": "medium", "completed": False},
    ])

    print(f"Generated 4 test sessions in: {persist_dir}")
    print(f"  - {wf1_id[:12]}... (completed, with sub-agent)")
    print(f"  - {wf2_id[:12]}... (running, in act phase)")
    print(f"  - {wf3_id[:12]}... (failed, stall detected)")
    print(f"  - {wf4_id[:12]}... (failed, budget exceeded, stale sub-agents)")


if __name__ == "__main__":
    import sys
    import shutil

    test_dir = Path("_debug_test_sessions")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    generate_test_data(test_dir)

    print(f"\nStarting Debug UI at http://localhost:8090")
    print("Press Ctrl+C to stop.\n")

    from parr.debug_ui import start_server
    start_server(persist_dir=str(test_dir), port=8090)
