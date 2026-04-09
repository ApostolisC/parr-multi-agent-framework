"""Tests for per-role direct_answer_schema_policy feature.

Covers: backward compatibility, bypass mode, enforce mode, schema validation,
escalation, sub-role inheritance/override, config loading, config validation,
router prompt content, and findings structure.
"""

from __future__ import annotations

import json

import pytest

from parr.adapters.domain_adapter import ReferenceDomainAdapter, RoleEntry, SubRoleEntry
from parr.agent_runtime import AgentRuntime
from parr.budget_tracker import BudgetTracker
from parr.config.config_validator import validate_config
from parr.core_types import (
    AgentConfig,
    AgentInput,
    AgentNode,
    BudgetConfig,
    CostConfig,
    ModelConfig,
    ModelPricing,
    SimpleQueryBypassConfig,
    WorkflowExecution,
)
from parr.event_bus import EventBus
from parr.output_validator import JsonSchemaValidator
from parr.tests.mock_llm import MockToolCallingLLM, make_text_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["summary"],
}


def _make_runtime(
    llm: MockToolCallingLLM,
    bypass_cfg: SimpleQueryBypassConfig | None = None,
    output_validator=None,
) -> AgentRuntime:
    cost_config = CostConfig(models={
        "test-model": ModelPricing(
            input_price_per_1k=0.001,
            output_price_per_1k=0.002,
            context_window=128000,
        )
    })
    return AgentRuntime(
        llm=llm,
        budget_tracker=BudgetTracker(cost_config=cost_config),
        event_bus=EventBus(),
        simple_query_bypass=bypass_cfg or SimpleQueryBypassConfig(),
        budget_config=BudgetConfig(max_tokens=200000),
        output_validator=output_validator,
    )


def _make_context(
    output_schema: dict | None = None,
    direct_answer_schema_policy: str | None = None,
):
    config = AgentConfig(
        role="researcher",
        system_prompt="You are a researcher.",
        model="test-model",
        model_config=ModelConfig(temperature=0.2, top_p=1.0, max_tokens=1024),
    )
    budget = BudgetConfig(max_tokens=200000)
    node = AgentNode(agent_id=config.agent_id, config=config, budget=budget)
    workflow = WorkflowExecution(global_budget=budget)
    workflow.agent_tree[node.task_id] = node
    agent_input = AgentInput(
        task="What is DPIA?",
        output_schema=output_schema,
        budget=budget,
        direct_answer_schema_policy=direct_answer_schema_policy,
    )
    return config, agent_input, node, workflow


# ---------------------------------------------------------------------------
# 1. Backward compatibility: no policy + output_schema → full workflow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backward_compat_no_policy_with_schema_forces_full_workflow():
    """When direct_answer_schema_policy is None (default), the global gate blocks bypass."""
    llm = MockToolCallingLLM([
        make_text_response("Plan complete."),
        make_text_response("Act complete."),
        make_text_response("REVIEW_PASSED"),
        make_text_response("Final report"),
    ])
    runtime = _make_runtime(llm)
    config, agent_input, node, workflow = _make_context(
        output_schema=_SIMPLE_SCHEMA,
        direct_answer_schema_policy=None,
    )

    output = await runtime.execute(config, agent_input, node, workflow)

    assert output.execution_metadata.execution_path == "full_workflow"
    assert output.execution_metadata.routing_decision["policy_reason"] == "output_schema_requires_full_workflow"
    assert llm.call_count == 4  # No router call


# ---------------------------------------------------------------------------
# 2. Bypass mode success: policy="bypass" + schema → direct answer, free-form
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bypass_mode_direct_answer_freeform():
    """Bypass mode allows direct answer with free-form text (ignores schema)."""
    llm = MockToolCallingLLM([
        # Router response
        make_text_response(
            '{"mode":"direct_answer","confidence":0.95,'
            '"reason":"Simple factual query","requires_external_data":false}',
            input_tokens=10, output_tokens=20,
        ),
        # Direct answer response (free-form)
        make_text_response(
            '{"answer":"DPIA is a Data Protection Impact Assessment.",'
            '"confidence":0.95,"needs_full_workflow":false,"reason":"Known answer"}',
            input_tokens=12, output_tokens=30,
        ),
    ])
    runtime = _make_runtime(llm)
    config, agent_input, node, workflow = _make_context(
        output_schema=_SIMPLE_SCHEMA,
        direct_answer_schema_policy="bypass",
    )

    output = await runtime.execute(config, agent_input, node, workflow)

    assert output.status == "completed"
    assert output.execution_metadata.execution_path == "direct_answer"
    assert output.execution_metadata.routing_decision["selected_path"] == "direct_answer"
    assert "DPIA" in output.summary
    assert output.findings.get("direct_answer") is not None
    assert llm.call_count == 2


# ---------------------------------------------------------------------------
# 3. Enforce mode success: policy="enforce" → schema-compliant findings
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enforce_mode_direct_answer_schema_compliant():
    """Enforce mode produces schema-compliant output in findings directly."""
    schema_output = {"summary": "DPIA explained.", "answer": "DPIA is a risk assessment tool."}
    llm = MockToolCallingLLM([
        # Router
        make_text_response(
            '{"mode":"direct_answer","confidence":0.95,'
            '"reason":"Simple query","requires_external_data":false}',
            input_tokens=10, output_tokens=20,
        ),
        # Direct answer with schema output
        make_text_response(
            json.dumps({
                "output": schema_output,
                "confidence": 0.95,
                "needs_full_workflow": False,
                "reason": "Known answer",
            }),
            input_tokens=15, output_tokens=60,
        ),
    ])
    runtime = _make_runtime(llm, output_validator=JsonSchemaValidator())
    config, agent_input, node, workflow = _make_context(
        output_schema=_SIMPLE_SCHEMA,
        direct_answer_schema_policy="enforce",
    )

    output = await runtime.execute(config, agent_input, node, workflow)

    assert output.status == "completed"
    assert output.execution_metadata.execution_path == "direct_answer"
    # findings IS the schema output dict directly
    assert output.findings == schema_output
    # answer field takes priority over summary for the output.summary
    assert output.summary == "DPIA is a risk assessment tool."
    assert llm.call_count == 2


# ---------------------------------------------------------------------------
# 4. Enforce invalid JSON → escalates to full workflow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enforce_invalid_json_escalates():
    """When enforce mode gets invalid JSON, it escalates to full workflow."""
    llm = MockToolCallingLLM([
        # Router → direct answer
        make_text_response(
            '{"mode":"direct_answer","confidence":0.95,'
            '"reason":"Simple","requires_external_data":false}',
        ),
        # Invalid JSON from LLM
        make_text_response("This is not JSON at all."),
        # Full workflow phases after escalation
        make_text_response("Plan complete."),
        make_text_response("Act complete."),
        make_text_response("REVIEW_PASSED"),
        make_text_response("Final report"),
    ])
    runtime = _make_runtime(llm, output_validator=JsonSchemaValidator())
    config, agent_input, node, workflow = _make_context(
        output_schema=_SIMPLE_SCHEMA,
        direct_answer_schema_policy="enforce",
    )

    output = await runtime.execute(config, agent_input, node, workflow)

    assert output.execution_metadata.execution_path == "full_workflow"
    assert output.execution_metadata.routing_decision["escalated_after_direct_answer"] is True
    assert llm.call_count == 6


# ---------------------------------------------------------------------------
# 5. Enforce escalation: LLM says needs_full_workflow=true
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enforce_escalation_needs_full_workflow():
    """Enforce mode respects needs_full_workflow=true from LLM."""
    llm = MockToolCallingLLM([
        # Router
        make_text_response(
            '{"mode":"direct_answer","confidence":0.95,'
            '"reason":"Simple","requires_external_data":false}',
        ),
        # LLM says needs full workflow
        make_text_response(json.dumps({
            "output": {"summary": "partial"},
            "confidence": 0.3,
            "needs_full_workflow": True,
            "reason": "Need deeper research",
        })),
        # Full workflow phases
        make_text_response("Plan complete."),
        make_text_response("Act complete."),
        make_text_response("REVIEW_PASSED"),
        make_text_response("Final report"),
    ])
    runtime = _make_runtime(llm, output_validator=JsonSchemaValidator())
    config, agent_input, node, workflow = _make_context(
        output_schema=_SIMPLE_SCHEMA,
        direct_answer_schema_policy="enforce",
    )

    output = await runtime.execute(config, agent_input, node, workflow)

    assert output.execution_metadata.execution_path == "full_workflow"
    assert output.execution_metadata.routing_decision["escalated_after_direct_answer"] is True


# ---------------------------------------------------------------------------
# 6. Enforce max_tokens: verify ≥2048
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enforce_mode_increases_max_tokens():
    """Enforce mode auto-increases max_tokens to at least 2048."""
    calls_model_config = []

    class InstrumentedLLM(MockToolCallingLLM):
        async def chat_with_tools(self, messages, tools, model, model_config, **kw):
            calls_model_config.append(model_config)
            return await super().chat_with_tools(messages, tools, model, model_config, **kw)

    llm = InstrumentedLLM([
        make_text_response(
            '{"mode":"direct_answer","confidence":0.95,'
            '"reason":"Simple","requires_external_data":false}',
        ),
        make_text_response(json.dumps({
            "output": {"summary": "test"},
            "confidence": 0.95,
            "needs_full_workflow": False,
            "reason": "ok",
        })),
    ])
    bypass_cfg = SimpleQueryBypassConfig(direct_answer_max_tokens=512)
    runtime = _make_runtime(llm, bypass_cfg=bypass_cfg, output_validator=JsonSchemaValidator())
    config, agent_input, node, workflow = _make_context(
        output_schema=_SIMPLE_SCHEMA,
        direct_answer_schema_policy="enforce",
    )

    await runtime.execute(config, agent_input, node, workflow)

    # Second call is the direct answer — should have max_tokens >= 2048
    assert len(calls_model_config) == 2
    direct_answer_mc = calls_model_config[1]
    assert direct_answer_mc.max_tokens >= 2048


# ---------------------------------------------------------------------------
# 7. Sub-role inheritance: inherits parent policy
# ---------------------------------------------------------------------------

def test_subrole_inherits_parent_policy():
    """Sub-role without its own policy inherits parent's."""
    adapter = ReferenceDomainAdapter()
    adapter.register_role(
        role="analyst",
        config=AgentConfig(role="analyst", system_prompt="...", model="gpt-4o"),
        direct_answer_schema_policy="bypass",
    )
    adapter.register_sub_role(
        role="analyst",
        sub_role="deep_dive",
        description="Deep analysis",
    )

    assert adapter.get_direct_answer_schema_policy("analyst", "deep_dive") == "bypass"


# ---------------------------------------------------------------------------
# 8. Sub-role override: own policy overrides parent
# ---------------------------------------------------------------------------

def test_subrole_overrides_parent_policy():
    """Sub-role with its own policy overrides the parent's."""
    adapter = ReferenceDomainAdapter()
    adapter.register_role(
        role="analyst",
        config=AgentConfig(role="analyst", system_prompt="...", model="gpt-4o"),
        direct_answer_schema_policy="bypass",
    )
    adapter.register_sub_role(
        role="analyst",
        sub_role="strict",
        description="Strict sub-role",
        direct_answer_schema_policy_override="enforce",
    )

    assert adapter.get_direct_answer_schema_policy("analyst", "strict") == "enforce"
    # Parent unchanged
    assert adapter.get_direct_answer_schema_policy("analyst") == "bypass"


# ---------------------------------------------------------------------------
# 9. No schema + policy: normal direct answer (policy irrelevant)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_schema_with_policy_normal_direct_answer():
    """When there's no output_schema, policy is irrelevant — normal bypass works."""
    llm = MockToolCallingLLM([
        make_text_response(
            '{"mode":"direct_answer","confidence":0.95,'
            '"reason":"Simple","requires_external_data":false}',
        ),
        make_text_response(
            '{"answer":"DPIA is Data Protection Impact Assessment.",'
            '"confidence":0.95,"needs_full_workflow":false,"reason":"ok"}',
        ),
    ])
    runtime = _make_runtime(llm)
    config, agent_input, node, workflow = _make_context(
        output_schema=None,
        direct_answer_schema_policy="enforce",  # should be ignored
    )

    output = await runtime.execute(config, agent_input, node, workflow)

    assert output.execution_metadata.execution_path == "direct_answer"
    assert "direct_answer" in output.findings  # Free-form, not schema output
    assert llm.call_count == 2


# ---------------------------------------------------------------------------
# 10. Router prompt includes policy info
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_router_prompt_includes_policy():
    """Routing messages mention the direct-answer schema policy."""
    captured_messages = []

    class CapturingLLM(MockToolCallingLLM):
        async def chat_with_tools(self, messages, tools, model, model_config, **kw):
            captured_messages.append(messages)
            return await super().chat_with_tools(messages, tools, model, model_config, **kw)

    llm = CapturingLLM([
        # Router → full workflow (so it stops after routing)
        make_text_response(
            '{"mode":"full_workflow","confidence":0.9,'
            '"reason":"Need research","requires_external_data":true}',
        ),
        make_text_response("Plan complete."),
        make_text_response("Act complete."),
        make_text_response("REVIEW_PASSED"),
        make_text_response("Final report"),
    ])
    runtime = _make_runtime(llm)
    config, agent_input, node, workflow = _make_context(
        output_schema=_SIMPLE_SCHEMA,
        direct_answer_schema_policy="bypass",
    )

    await runtime.execute(config, agent_input, node, workflow)

    # First call is routing
    assert len(captured_messages) >= 1
    routing_user_msg = captured_messages[0][-1].content
    assert "Direct-answer schema policy: bypass" in routing_user_msg


# ---------------------------------------------------------------------------
# 11. Schema output in findings: is dict directly (not wrapped)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_schema_output_in_findings_directly():
    """In enforce mode, findings IS the schema output dict (not wrapped)."""
    schema_output = {
        "summary": "GDPR overview.",
        "answer": "GDPR is the General Data Protection Regulation.",
        "key_findings": [{"finding": "Enacted 2018"}],
    }
    llm = MockToolCallingLLM([
        make_text_response(
            '{"mode":"direct_answer","confidence":0.95,'
            '"reason":"Known","requires_external_data":false}',
        ),
        make_text_response(json.dumps({
            "output": schema_output,
            "confidence": 0.95,
            "needs_full_workflow": False,
            "reason": "ok",
        })),
    ])
    runtime = _make_runtime(llm, output_validator=JsonSchemaValidator())
    config, agent_input, node, workflow = _make_context(
        output_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "answer": {"type": "string"},
                "key_findings": {"type": "array"},
            },
            "required": ["summary"],
        },
        direct_answer_schema_policy="enforce",
    )

    output = await runtime.execute(config, agent_input, node, workflow)

    assert output.findings == schema_output
    assert output.findings["key_findings"] == [{"finding": "Enacted 2018"}]
    assert "direct_answer" not in output.findings  # NOT wrapped in old format


# ---------------------------------------------------------------------------
# 12. Config validation: invalid values produce errors
# ---------------------------------------------------------------------------

def _validate_roles_only(roles):
    """Helper: run validate_config with minimal params, return errors."""
    from pathlib import Path

    models = {"gpt-4o": {"input_price_per_1k": 0.001, "output_price_per_1k": 0.002}}
    return validate_config(
        config_dir=Path("."),
        roles=roles,
        models=models,
        budget={},
        phase_limits={},
        llm_rate_limit=None,
        tool_names=[],
    )


def test_config_validation_invalid_policy_value():
    """Invalid direct_answer_schema_policy values produce validation errors."""
    roles = {
        "analyst": {
            "model": "gpt-4o",
            "system_prompt": "prompts/analyst.md",
            "direct_answer_schema_policy": "invalid_value",
        }
    }
    errors = _validate_roles_only(roles)

    policy_errors = [e for e in errors if "direct_answer_schema_policy" in e]
    assert len(policy_errors) >= 1
    assert "invalid_value" in policy_errors[0]


def test_config_validation_valid_policy_values():
    """Valid direct_answer_schema_policy values produce no extra errors."""
    for value in ("enforce", "bypass"):
        roles = {
            "analyst": {
                "model": "gpt-4o",
                "system_prompt": "prompts/analyst.md",
                "direct_answer_schema_policy": value,
            }
        }
        errors = _validate_roles_only(roles)

        policy_errors = [e for e in errors if "direct_answer_schema_policy" in e]
        assert len(policy_errors) == 0, f"Unexpected errors for policy={value}: {policy_errors}"


def test_config_validation_subrole_invalid_policy():
    """Invalid direct_answer_schema_policy on sub-role produces error."""
    roles = {
        "analyst": {
            "model": "gpt-4o",
            "system_prompt": "prompts/analyst.md",
            "sub_roles": {
                "strict": {
                    "direct_answer_schema_policy": "wrong",
                },
            },
        }
    }
    errors = _validate_roles_only(roles)

    policy_errors = [e for e in errors if "direct_answer_schema_policy" in e]
    assert len(policy_errors) >= 1
    assert "wrong" in policy_errors[0]


# ---------------------------------------------------------------------------
# 13. Config loading: YAML populates RoleEntry correctly
# ---------------------------------------------------------------------------

def test_domain_adapter_stores_policy():
    """register_role stores direct_answer_schema_policy on RoleEntry."""
    adapter = ReferenceDomainAdapter()
    adapter.register_role(
        role="researcher",
        config=AgentConfig(role="researcher", system_prompt="...", model="gpt-4o"),
        direct_answer_schema_policy="bypass",
    )

    assert adapter.get_direct_answer_schema_policy("researcher") == "bypass"


def test_domain_adapter_none_policy():
    """When no policy is set, get_direct_answer_schema_policy returns None."""
    adapter = ReferenceDomainAdapter()
    adapter.register_role(
        role="researcher",
        config=AgentConfig(role="researcher", system_prompt="...", model="gpt-4o"),
    )

    assert adapter.get_direct_answer_schema_policy("researcher") is None


def test_domain_adapter_unknown_role_returns_none():
    """Unknown role returns None for direct_answer_schema_policy."""
    adapter = ReferenceDomainAdapter()
    assert adapter.get_direct_answer_schema_policy("nonexistent") is None


# ---------------------------------------------------------------------------
# 14. Enforce with missing output field → escalates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enforce_missing_output_field_escalates():
    """When enforce mode LLM omits the 'output' field, escalation occurs."""
    llm = MockToolCallingLLM([
        make_text_response(
            '{"mode":"direct_answer","confidence":0.95,'
            '"reason":"Simple","requires_external_data":false}',
        ),
        # Missing "output" field
        make_text_response(json.dumps({
            "answer": "DPIA is ...",
            "confidence": 0.95,
            "needs_full_workflow": False,
            "reason": "ok",
        })),
        make_text_response("Plan complete."),
        make_text_response("Act complete."),
        make_text_response("REVIEW_PASSED"),
        make_text_response("Final report"),
    ])
    runtime = _make_runtime(llm, output_validator=JsonSchemaValidator())
    config, agent_input, node, workflow = _make_context(
        output_schema=_SIMPLE_SCHEMA,
        direct_answer_schema_policy="enforce",
    )

    output = await runtime.execute(config, agent_input, node, workflow)

    assert output.execution_metadata.execution_path == "full_workflow"
    assert output.execution_metadata.routing_decision["escalated_after_direct_answer"] is True


# ---------------------------------------------------------------------------
# 15. Enforce schema validation failure → escalates
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enforce_schema_validation_failure_escalates():
    """When enforce mode output fails schema validation, escalation occurs."""
    strict_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["summary", "count"],
    }
    llm = MockToolCallingLLM([
        make_text_response(
            '{"mode":"direct_answer","confidence":0.95,'
            '"reason":"Simple","requires_external_data":false}',
        ),
        # Output missing required "count" field
        make_text_response(json.dumps({
            "output": {"summary": "test"},  # missing "count"
            "confidence": 0.95,
            "needs_full_workflow": False,
            "reason": "ok",
        })),
        make_text_response("Plan complete."),
        make_text_response("Act complete."),
        make_text_response("REVIEW_PASSED"),
        make_text_response("Final report"),
    ])
    runtime = _make_runtime(llm, output_validator=JsonSchemaValidator())
    config, agent_input, node, workflow = _make_context(
        output_schema=strict_schema,
        direct_answer_schema_policy="enforce",
    )

    output = await runtime.execute(config, agent_input, node, workflow)

    assert output.execution_metadata.execution_path == "full_workflow"
    assert output.execution_metadata.routing_decision["escalated_after_direct_answer"] is True
