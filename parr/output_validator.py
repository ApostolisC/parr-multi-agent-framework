"""
Output Validator for the Agentic Framework.

Pluggable output validation for agent-submitted reports. The framework
validates the agent's output after all phases complete.

Pluggability:
    - **OutputValidator**: Override ``validate()`` to add domain-specific
      validation rules (e.g., cross-field checks, semantic validation,
      business rule enforcement).
    - **JsonSchemaValidator** (default): Validates against a JSON Schema.
    - **CompositeValidator**: Chains multiple validators — all must pass.

Example::

    class RiskScoreValidator(OutputValidator):
        def validate(self, output, schema=None, role="", sub_role=None):
            score = output.get("risk_score")
            if score is not None and not (1 <= score <= 10):
                return OutputValidationResult(
                    is_valid=False,
                    errors=["risk_score must be between 1 and 10"],
                )
            return OutputValidationResult(is_valid=True)

    validator = CompositeValidator([
        JsonSchemaValidator(),
        RiskScoreValidator(),
    ])
    orchestrator = Orchestrator(llm=llm, output_validator=validator)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .core_types import OutputValidationResult

logger = logging.getLogger(__name__)


class OutputValidator:
    """Base class for pluggable output validation.

    Override ``validate()`` to add custom validation logic. The default
    implementation is a no-op that passes all outputs.

    Built-in validators:
        - :class:`JsonSchemaValidator` (default): validates against a
          JSON Schema (preserves pre-existing framework behavior).
        - :class:`CompositeValidator`: chains multiple validators.

    Example::

        class MandatoryFieldsValidator(OutputValidator):
            def validate(self, output, schema=None, role="", sub_role=None):
                if "summary" not in output:
                    return OutputValidationResult(
                        is_valid=False,
                        errors=["Output must contain a 'summary' field"],
                    )
                return OutputValidationResult(is_valid=True)
    """

    def validate(
        self,
        output: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
        role: str = "",
        sub_role: Optional[str] = None,
    ) -> OutputValidationResult:
        """Validate agent output.

        Args:
            output: The submitted report (dict).
            schema: Optional JSON Schema (from output_schema config).
            role: The agent's role identifier.
            sub_role: Optional sub-role identifier.

        Returns:
            OutputValidationResult with is_valid, errors, and warnings.
        """
        return OutputValidationResult(is_valid=True)


class JsonSchemaValidator(OutputValidator):
    """Validates output against a JSON Schema.

    This is the default validator and preserves the framework's existing
    behavior. When no schema is provided, validation always passes.
    """

    def validate(
        self,
        output: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
        role: str = "",
        sub_role: Optional[str] = None,
    ) -> OutputValidationResult:
        if not schema:
            return OutputValidationResult(is_valid=True)

        try:
            from jsonschema import ValidationError as JsonSchemaValidationError
            from jsonschema import validate as jsonschema_validate

            jsonschema_validate(instance=output, schema=schema)
            return OutputValidationResult(is_valid=True)
        except JsonSchemaValidationError as e:
            return OutputValidationResult(
                is_valid=False,
                errors=[f"Schema validation failed: {e.message}"],
            )


class CompositeValidator(OutputValidator):
    """Chains multiple validators — all must pass for the output to be valid.

    Errors and warnings from all validators are collected regardless of
    whether earlier validators fail.

    Example::

        validator = CompositeValidator([
            JsonSchemaValidator(),
            MyDomainValidator(),
        ])
    """

    def __init__(self, validators: List[OutputValidator]) -> None:
        self._validators = list(validators)

    def validate(
        self,
        output: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
        role: str = "",
        sub_role: Optional[str] = None,
    ) -> OutputValidationResult:
        all_errors: List[str] = []
        all_warnings: List[str] = []

        for validator in self._validators:
            result = validator.validate(output, schema, role, sub_role)
            all_errors.extend(result.errors)
            all_warnings.extend(result.warnings)

        return OutputValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings,
        )
