"""Tests for pluggable output validation (OutputValidator and built-in validators)."""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from parr.core_types import OutputValidationResult
from parr.output_validator import (
    CompositeValidator,
    JsonSchemaValidator,
    OutputValidator,
)


# =========================================================================
# 1. OutputValidationResult
# =========================================================================

class TestOutputValidationResult:

    def test_default_is_valid(self):
        result = OutputValidationResult()
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_invalid_with_errors(self):
        result = OutputValidationResult(
            is_valid=False,
            errors=["Missing field 'summary'"],
        )
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_valid_with_warnings(self):
        result = OutputValidationResult(
            is_valid=True,
            warnings=["Field 'notes' is deprecated"],
        )
        assert result.is_valid is True
        assert len(result.warnings) == 1


# =========================================================================
# 2. OutputValidator base class
# =========================================================================

class TestOutputValidatorBase:

    def test_base_passes_everything(self):
        validator = OutputValidator()
        result = validator.validate({"anything": "goes"})
        assert result.is_valid is True
        assert result.errors == []

    def test_base_ignores_schema(self):
        validator = OutputValidator()
        result = validator.validate(
            {"bad": "data"},
            schema={"type": "object", "required": ["missing_field"]},
        )
        assert result.is_valid is True

    def test_base_accepts_role_params(self):
        validator = OutputValidator()
        result = validator.validate(
            {"data": 1},
            role="analyst",
            sub_role="risk",
        )
        assert result.is_valid is True


# =========================================================================
# 3. JsonSchemaValidator
# =========================================================================

class TestJsonSchemaValidator:

    def test_passes_valid_output(self):
        validator = JsonSchemaValidator()
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "score": {"type": "integer"},
            },
            "required": ["summary"],
        }
        result = validator.validate(
            {"summary": "Test report", "score": 5},
            schema=schema,
        )
        assert result.is_valid is True
        assert result.errors == []

    def test_fails_missing_required_field(self):
        validator = JsonSchemaValidator()
        schema = {
            "type": "object",
            "required": ["summary"],
            "properties": {
                "summary": {"type": "string"},
            },
        }
        result = validator.validate({"score": 5}, schema=schema)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "summary" in result.errors[0]

    def test_fails_wrong_type(self):
        validator = JsonSchemaValidator()
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
            },
        }
        result = validator.validate({"score": "not_a_number"}, schema=schema)
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_passes_when_no_schema(self):
        validator = JsonSchemaValidator()
        result = validator.validate({"anything": "goes"}, schema=None)
        assert result.is_valid is True

    def test_passes_empty_schema(self):
        validator = JsonSchemaValidator()
        result = validator.validate({"anything": "goes"}, schema={})
        assert result.is_valid is True

    def test_complex_schema(self):
        validator = JsonSchemaValidator()
        schema = {
            "type": "object",
            "properties": {
                "risks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["description", "severity"],
                        "properties": {
                            "description": {"type": "string"},
                            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        },
                    },
                },
            },
            "required": ["risks"],
        }
        # Valid
        result = validator.validate(
            {"risks": [{"description": "Data leak", "severity": "high"}]},
            schema=schema,
        )
        assert result.is_valid is True

        # Invalid severity enum
        result = validator.validate(
            {"risks": [{"description": "Data leak", "severity": "extreme"}]},
            schema=schema,
        )
        assert result.is_valid is False


# =========================================================================
# 4. CompositeValidator
# =========================================================================

class TestCompositeValidator:

    def test_all_pass(self):
        v1 = OutputValidator()  # always passes
        v2 = OutputValidator()  # always passes
        composite = CompositeValidator([v1, v2])
        result = composite.validate({"data": 1})
        assert result.is_valid is True
        assert result.errors == []

    def test_one_fails(self):
        class FailValidator(OutputValidator):
            def validate(self, output, schema=None, role="", sub_role=None):
                return OutputValidationResult(
                    is_valid=False,
                    errors=["Custom check failed"],
                )

        composite = CompositeValidator([OutputValidator(), FailValidator()])
        result = composite.validate({"data": 1})
        assert result.is_valid is False
        assert "Custom check failed" in result.errors

    def test_collects_all_errors(self):
        class Fail1(OutputValidator):
            def validate(self, output, schema=None, role="", sub_role=None):
                return OutputValidationResult(
                    is_valid=False,
                    errors=["Error from validator 1"],
                )

        class Fail2(OutputValidator):
            def validate(self, output, schema=None, role="", sub_role=None):
                return OutputValidationResult(
                    is_valid=False,
                    errors=["Error from validator 2"],
                )

        composite = CompositeValidator([Fail1(), Fail2()])
        result = composite.validate({"data": 1})
        assert result.is_valid is False
        assert len(result.errors) == 2

    def test_collects_warnings(self):
        class WarnValidator(OutputValidator):
            def validate(self, output, schema=None, role="", sub_role=None):
                return OutputValidationResult(
                    is_valid=True,
                    warnings=["Consider adding more detail"],
                )

        composite = CompositeValidator([OutputValidator(), WarnValidator()])
        result = composite.validate({"data": 1})
        assert result.is_valid is True
        assert len(result.warnings) == 1

    def test_schema_passed_to_all(self):
        schemas_seen = []

        class TrackingValidator(OutputValidator):
            def validate(self, output, schema=None, role="", sub_role=None):
                schemas_seen.append(schema)
                return OutputValidationResult(is_valid=True)

        schema = {"type": "object"}
        composite = CompositeValidator([TrackingValidator(), TrackingValidator()])
        composite.validate({"data": 1}, schema=schema)
        assert all(s is schema for s in schemas_seen)

    def test_empty_validators_passes(self):
        composite = CompositeValidator([])
        result = composite.validate({"data": 1})
        assert result.is_valid is True

    def test_jsonschema_plus_custom(self):
        """Typical usage: jsonschema + domain validator."""
        class RiskScoreValidator(OutputValidator):
            def validate(self, output, schema=None, role="", sub_role=None):
                score = output.get("risk_score")
                if isinstance(score, (int, float)) and not (1 <= score <= 10):
                    return OutputValidationResult(
                        is_valid=False,
                        errors=["risk_score must be between 1 and 10"],
                    )
                return OutputValidationResult(is_valid=True)

        schema = {
            "type": "object",
            "required": ["risk_score"],
            "properties": {"risk_score": {"type": "integer"}},
        }
        composite = CompositeValidator([
            JsonSchemaValidator(),
            RiskScoreValidator(),
        ])

        # Valid
        result = composite.validate({"risk_score": 5}, schema=schema)
        assert result.is_valid is True

        # Fails schema (wrong type)
        result = composite.validate({"risk_score": "high"}, schema=schema)
        assert result.is_valid is False

        # Fails domain (out of range)
        result = composite.validate({"risk_score": 15}, schema=schema)
        assert result.is_valid is False
        assert "1 and 10" in result.errors[0]

        # Fails both
        result = composite.validate({}, schema=schema)
        assert result.is_valid is False


# =========================================================================
# 5. Custom validators (user-defined)
# =========================================================================

class TestCustomValidators:

    def test_role_specific_validator(self):
        class AnalystValidator(OutputValidator):
            def validate(self, output, schema=None, role="", sub_role=None):
                if role == "analyst" and "findings" not in output:
                    return OutputValidationResult(
                        is_valid=False,
                        errors=["Analyst reports must include 'findings'"],
                    )
                return OutputValidationResult(is_valid=True)

        validator = AnalystValidator()

        # Analyst without findings — fails
        result = validator.validate({"summary": "Done"}, role="analyst")
        assert result.is_valid is False

        # Analyst with findings — passes
        result = validator.validate(
            {"summary": "Done", "findings": []},
            role="analyst",
        )
        assert result.is_valid is True

        # Non-analyst without findings — passes
        result = validator.validate({"summary": "Done"}, role="coordinator")
        assert result.is_valid is True

    def test_warning_only_validator(self):
        class QualityAdvisor(OutputValidator):
            def validate(self, output, schema=None, role="", sub_role=None):
                warnings = []
                if len(str(output.get("summary", ""))) < 50:
                    warnings.append("Summary is very short — consider adding more detail")
                return OutputValidationResult(is_valid=True, warnings=warnings)

        validator = QualityAdvisor()
        result = validator.validate({"summary": "Brief"})
        assert result.is_valid is True
        assert len(result.warnings) == 1

    def test_cross_field_validator(self):
        class CrossFieldValidator(OutputValidator):
            def validate(self, output, schema=None, role="", sub_role=None):
                errors = []
                if output.get("severity") == "high" and not output.get("mitigation"):
                    errors.append("High severity findings must include mitigation")
                return OutputValidationResult(
                    is_valid=len(errors) == 0,
                    errors=errors,
                )

        validator = CrossFieldValidator()

        # High severity without mitigation — fails
        result = validator.validate({"severity": "high", "description": "Bad"})
        assert result.is_valid is False

        # High severity with mitigation — passes
        result = validator.validate({
            "severity": "high",
            "description": "Bad",
            "mitigation": "Fix it",
        })
        assert result.is_valid is True

        # Low severity without mitigation — passes
        result = validator.validate({"severity": "low", "description": "Minor"})
        assert result.is_valid is True


# =========================================================================
# 6. AgentRuntime integration
# =========================================================================

class TestAgentRuntimeIntegration:

    def test_runtime_default_validator(self):
        from parr.agent_runtime import AgentRuntime

        llm = MagicMock()
        bt = MagicMock()
        eb = MagicMock()
        rt = AgentRuntime(llm=llm, budget_tracker=bt, event_bus=eb)
        assert isinstance(rt._output_validator, JsonSchemaValidator)

    def test_runtime_custom_validator(self):
        from parr.agent_runtime import AgentRuntime

        llm = MagicMock()
        bt = MagicMock()
        eb = MagicMock()
        custom = OutputValidator()
        rt = AgentRuntime(
            llm=llm, budget_tracker=bt, event_bus=eb,
            output_validator=custom,
        )
        assert rt._output_validator is custom

    def test_runtime_composite_validator(self):
        from parr.agent_runtime import AgentRuntime

        llm = MagicMock()
        bt = MagicMock()
        eb = MagicMock()
        composite = CompositeValidator([JsonSchemaValidator(), OutputValidator()])
        rt = AgentRuntime(
            llm=llm, budget_tracker=bt, event_bus=eb,
            output_validator=composite,
        )
        assert isinstance(rt._output_validator, CompositeValidator)


# =========================================================================
# 7. Orchestrator integration
# =========================================================================

class TestOrchestratorIntegration:

    def test_orchestrator_stores_validator(self):
        from parr.orchestrator import Orchestrator

        llm = MagicMock()
        custom = OutputValidator()
        orch = Orchestrator(llm=llm, output_validator=custom)
        assert orch._output_validator is custom

    def test_orchestrator_default_no_validator(self):
        from parr.orchestrator import Orchestrator

        llm = MagicMock()
        orch = Orchestrator(llm=llm)
        assert orch._output_validator is None


# =========================================================================
# 8. Edge cases
# =========================================================================

class TestEdgeCases:

    def test_validate_empty_output(self):
        validator = JsonSchemaValidator()
        result = validator.validate({})
        assert result.is_valid is True

    def test_validate_with_additional_properties(self):
        validator = JsonSchemaValidator()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        result = validator.validate({"name": "test", "extra": "field"}, schema=schema)
        assert result.is_valid is False

    def test_composite_with_single_validator(self):
        composite = CompositeValidator([JsonSchemaValidator()])
        schema = {"type": "object", "required": ["x"]}
        result = composite.validate({"x": 1}, schema=schema)
        assert result.is_valid is True

    def test_validator_receives_all_params(self):
        received = {}

        class ParamTracker(OutputValidator):
            def validate(self, output, schema=None, role="", sub_role=None):
                received["output"] = output
                received["schema"] = schema
                received["role"] = role
                received["sub_role"] = sub_role
                return OutputValidationResult(is_valid=True)

        tracker = ParamTracker()
        data = {"key": "value"}
        s = {"type": "object"}
        tracker.validate(data, schema=s, role="analyst", sub_role="risk")

        assert received["output"] is data
        assert received["schema"] is s
        assert received["role"] == "analyst"
        assert received["sub_role"] == "risk"
