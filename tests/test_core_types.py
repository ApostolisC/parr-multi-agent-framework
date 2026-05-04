"""Tests for core_types.py — data structures, enums, helpers."""
import pytest
from parr.core_types import (
    AgentNode,
    BudgetConfig,
    Phase,
    PoliciesConfig,
    SpawnPolicy,
    ToolDef,
    normalise_keys,
    _camel_to_snake,
)


# ---------------------------------------------------------------------------
# normalise_keys / _camel_to_snake
# ---------------------------------------------------------------------------


class TestCamelToSnake:
    def test_simple(self):
        assert _camel_to_snake("taskDescription") == "task_description"

    def test_sub_role(self):
        assert _camel_to_snake("subRole") == "sub_role"

    def test_effort_level(self):
        assert _camel_to_snake("effortLevel") == "effort_level"

    def test_already_snake(self):
        assert _camel_to_snake("task_description") == "task_description"

    def test_single_word(self):
        assert _camel_to_snake("role") == "role"

    def test_acronym(self):
        assert _camel_to_snake("dpaConsultationRequired") == "dpa_consultation_required"

    def test_leading_caps(self):
        assert _camel_to_snake("HTTPResponse") == "http_response"


class TestNormaliseKeys:
    def test_basic(self):
        result = normalise_keys({"taskDescription": "x", "role": "y"})
        assert result == {"task_description": "x", "role": "y"}

    def test_snake_preferred_over_camel(self):
        """If both snake and camel forms exist, snake wins."""
        result = normalise_keys({"sub_role": "correct", "subRole": "wrong"})
        assert result["sub_role"] == "correct"

    def test_empty_dict(self):
        assert normalise_keys({}) == {}

    def test_nested_not_touched(self):
        """Only top-level keys are normalised."""
        inp = {"processingDescription": {"completenessLevel": "high"}}
        result = normalise_keys(inp)
        assert "processing_description" in result
        inner = result["processing_description"]
        assert "completenessLevel" in inner  # nested NOT normalised


# ---------------------------------------------------------------------------
# PoliciesConfig / SpawnPolicy
# ---------------------------------------------------------------------------


class TestPoliciesConfig:
    def test_defaults(self):
        pc = PoliciesConfig()
        assert pc.same_role_spawn_policy == SpawnPolicy.WARN
        assert pc.consultant_model is None
        assert pc.consultant_max_tokens == 512
        assert pc.consultant_temperature == 0.1

    def test_custom(self):
        pc = PoliciesConfig(
            same_role_spawn_policy=SpawnPolicy.CONSULT,
            consultant_model="gpt-4.1-nano",
        )
        assert pc.same_role_spawn_policy == SpawnPolicy.CONSULT
        assert pc.consultant_model == "gpt-4.1-nano"

    def test_spawn_policy_values(self):
        assert SpawnPolicy.DENY.value == "deny"
        assert SpawnPolicy.WARN.value == "warn"
        assert SpawnPolicy.CONSULT.value == "consult"

    def test_frozen(self):
        """PoliciesConfig is frozen — fields cannot be reassigned."""
        pc = PoliciesConfig()
        with pytest.raises(AttributeError):
            pc.same_role_spawn_policy = SpawnPolicy.DENY


# ---------------------------------------------------------------------------
# ToolDef
# ---------------------------------------------------------------------------


class TestToolDef:
    def test_terminates_phase_default(self):
        td = ToolDef(name="test", description="test", parameters={})
        assert td.terminates_phase is False

    def test_terminates_phase_set(self):
        td = ToolDef(
            name="submit_report",
            description="submit",
            parameters={},
            terminates_phase=True,
        )
        assert td.terminates_phase is True

    def test_not_frozen(self):
        """ToolDef is mutable — phase_availability can be overridden at runtime."""
        td = ToolDef(
            name="test",
            description="test",
            parameters={},
            phase_availability=[Phase.REPORT],
        )
        td.phase_availability = [Phase.ACT]
        assert td.phase_availability == [Phase.ACT]


# ---------------------------------------------------------------------------
# AgentNode
# ---------------------------------------------------------------------------


class TestAgentNode:
    def test_description_default(self):
        node = AgentNode()
        assert node.description == ""

    def test_description_set(self):
        node = AgentNode(description="Analyze GDPR controls")
        assert node.description == "Analyze GDPR controls"
