"""
Configuration Validator for the Agentic Framework.

Validates declarative config files (roles.yaml, models.yaml, budget.yaml)
and all referenced content files (system prompts, output schemas, report templates).

All errors are collected before raising, so the developer sees every problem at once.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """
    Raised when declarative config validation fails.

    Contains a list of all validation errors found.
    """

    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        msg = f"Config validation failed with {len(errors)} error(s):\n"
        msg += "\n".join(f"  - {e}" for e in errors)
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _check_file_exists(
    config_dir: Path,
    rel_path: str,
    context: str,
    errors: List[str],
) -> bool:
    """Check that a referenced file exists. Returns True if it exists."""
    abs_path = config_dir / rel_path
    if not abs_path.is_file():
        errors.append(
            f"{context} references file '{rel_path}' which does not exist "
            f"(resolved: {abs_path})"
        )
        return False
    return True


def _check_json_valid(
    config_dir: Path,
    rel_path: str,
    context: str,
    errors: List[str],
) -> bool:
    """Check that a JSON file is valid. Returns True if valid."""
    abs_path = config_dir / rel_path
    if not abs_path.is_file():
        return False  # Already reported by _check_file_exists
    try:
        json.loads(abs_path.read_text(encoding="utf-8"))
        return True
    except (json.JSONDecodeError, ValueError) as e:
        errors.append(
            f"{context} file '{rel_path}' is not valid JSON: {e}"
        )
        return False


def _sorted_list(items: Any) -> str:
    """Format a collection as a sorted list string."""
    return str(sorted(items))


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate_config(
    config_dir: Path,
    roles: Dict[str, Dict[str, Any]],
    models: Dict[str, Dict[str, Any]],
    budget: Dict[str, Any],
    phase_limits: Dict[str, Any],
    llm_rate_limit: Optional[Dict[str, Any]],
    tool_names: List[str],
    simple_query_bypass: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Validate all config cross-references and required fields.

    Args:
        config_dir: Absolute path to the config directory.
        roles: Parsed roles dict from roles.yaml.
        models: Parsed models dict from models.yaml.
        budget: Parsed budget_defaults dict from budget.yaml.
        phase_limits: Parsed phase_limits dict from budget.yaml.
        llm_rate_limit: Parsed llm_rate_limit dict from budget.yaml.
        simple_query_bypass: Parsed simple_query_bypass dict from budget.yaml.
        tool_names: List of available tool names from the tool registry.

    Returns:
        List of error strings. Empty list means config is valid.
    """
    errors: List[str] = []
    available_models = set(models.keys())
    available_tools = set(tool_names)
    valid_phases = {"plan", "act", "review", "report"}

    # -- Validate phase limits -----------------------------------------------
    for phase_name in phase_limits:
        if phase_name not in valid_phases:
            errors.append(
                f"Phase limit references unknown phase '{phase_name}'. "
                f"Valid phases: {', '.join(sorted(valid_phases))}"
            )
        else:
            val = phase_limits[phase_name]
            if not isinstance(val, int) or val < 1:
                errors.append(
                    f"Phase limit for '{phase_name}' must be a positive integer, "
                    f"got {val!r}"
                )

    # -- Validate budget defaults --------------------------------------------
    _validate_budget(budget, errors)
    _validate_llm_rate_limit(llm_rate_limit or {}, errors)
    _validate_simple_query_bypass(simple_query_bypass or {}, errors)

    # -- Validate each role --------------------------------------------------
    for role_name, role_def in roles.items():
        ctx = f"Role '{role_name}'"

        # Role name must not contain double underscores (reserved for sub-role files)
        if "__" in role_name:
            errors.append(
                f"{ctx}: role name must not contain '__' "
                f"(reserved for sub-role file naming)"
            )

        # Required fields
        if not role_def.get("model"):
            errors.append(f"{ctx} is missing required field 'model'")

        if not role_def.get("system_prompt"):
            errors.append(f"{ctx} is missing required field 'system_prompt'")

        # Model exists in models.yaml
        model = role_def.get("model", "")
        if model and model not in available_models:
            errors.append(
                f"{ctx} references model '{model}' which is not defined in "
                f"models.yaml. Available models: {_sorted_list(available_models)}"
            )

        # System prompt file exists
        sp = role_def.get("system_prompt", "")
        if sp:
            _check_file_exists(config_dir, sp, ctx, errors)

        # Output schema file exists and is valid JSON
        os_path = role_def.get("output_schema")
        if os_path:
            if _check_file_exists(config_dir, os_path, ctx, errors):
                _check_json_valid(config_dir, os_path, ctx, errors)

        # Report template file exists
        rt = role_def.get("report_template")
        if rt:
            _check_file_exists(config_dir, rt, ctx, errors)

        # Tools exist in registry
        for tool_name in role_def.get("tools", []):
            if tool_name not in available_tools:
                errors.append(
                    f"{ctx} references tool '{tool_name}' which is not in the "
                    f"tool registry. Available tools: {_sorted_list(available_tools)}"
                )

        # Validate model_config if present
        mc = role_def.get("model_config")
        if mc:
            _validate_model_config(mc, ctx, errors)

        # -- Validate sub-roles ----------------------------------------------
        for sr_name, sr_def in role_def.get("sub_roles", {}).items():
            sr_ctx = f"Sub-role '{role_name}/{sr_name}'"

            if "__" in sr_name:
                errors.append(
                    f"{sr_ctx}: sub-role name must not contain '__' "
                    f"(reserved for file naming)"
                )

            # Model override
            sr_model = sr_def.get("model")
            if sr_model and sr_model not in available_models:
                errors.append(
                    f"{sr_ctx} references model '{sr_model}' which is not "
                    f"defined in models.yaml. Available models: "
                    f"{_sorted_list(available_models)}"
                )

            # System prompt override
            sr_sp = sr_def.get("system_prompt")
            if sr_sp:
                _check_file_exists(config_dir, sr_sp, sr_ctx, errors)

            # Output schema override
            sr_os = sr_def.get("output_schema")
            if sr_os:
                if _check_file_exists(config_dir, sr_os, sr_ctx, errors):
                    _check_json_valid(config_dir, sr_os, sr_ctx, errors)

            # Report template override
            sr_rt = sr_def.get("report_template")
            if sr_rt:
                _check_file_exists(config_dir, sr_rt, sr_ctx, errors)

            # Tools override
            for tool_name in sr_def.get("tools", []):
                if tool_name not in available_tools:
                    errors.append(
                        f"{sr_ctx} references tool '{tool_name}' which is not "
                        f"in the tool registry. Available tools: "
                        f"{_sorted_list(available_tools)}"
                    )

            # model_config override
            sr_mc = sr_def.get("model_config")
            if sr_mc:
                _validate_model_config(sr_mc, sr_ctx, errors)

    return errors


def _validate_budget(budget: Dict[str, Any], errors: List[str]) -> None:
    """Validate budget default values."""
    positive_int_fields = ["max_tokens", "max_duration_ms", "max_agent_depth", "max_parallel_agents"]
    for field_name in positive_int_fields:
        val = budget.get(field_name)
        if val is not None:
            if not isinstance(val, (int, float)) or val < 1:
                errors.append(
                    f"Budget field '{field_name}' must be a positive number, "
                    f"got {val!r}"
                )

    max_cost = budget.get("max_cost")
    if max_cost is not None:
        if not isinstance(max_cost, (int, float)) or max_cost <= 0:
            errors.append(
                f"Budget field 'max_cost' must be a positive number, "
                f"got {max_cost!r}"
            )


def _validate_model_config(
    mc: Dict[str, Any], context: str, errors: List[str]
) -> None:
    """Validate model_config fields."""
    temp = mc.get("temperature")
    if temp is not None and (not isinstance(temp, (int, float)) or temp < 0 or temp > 2):
        errors.append(
            f"{context} model_config.temperature must be between 0 and 2, "
            f"got {temp!r}"
        )

    top_p = mc.get("top_p")
    if top_p is not None and (not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1):
        errors.append(
            f"{context} model_config.top_p must be between 0 and 1, "
            f"got {top_p!r}"
        )

    max_tokens = mc.get("max_tokens")
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens < 1):
        errors.append(
            f"{context} model_config.max_tokens must be a positive integer, "
            f"got {max_tokens!r}"
        )


def _validate_llm_rate_limit(raw: Dict[str, Any], errors: List[str]) -> None:
    """Validate llm_rate_limit section in budget.yaml."""
    if not raw:
        return

    ctx = "llm_rate_limit"
    if not isinstance(raw, dict):
        errors.append(f"{ctx} must be a mapping/object, got {type(raw).__name__}")
        return

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        errors.append(f"{ctx}.enabled must be a boolean, got {enabled!r}")

    max_concurrent = raw.get("max_concurrent_requests")
    if max_concurrent is not None and (not isinstance(max_concurrent, int) or max_concurrent < 1):
        errors.append(
            f"{ctx}.max_concurrent_requests must be a positive integer, got {max_concurrent!r}"
        )

    max_per_window = raw.get("max_requests_per_window")
    max_per_minute = raw.get("max_requests_per_minute")
    if max_per_window is not None and (not isinstance(max_per_window, int) or max_per_window < 1):
        errors.append(
            f"{ctx}.max_requests_per_window must be a positive integer, got {max_per_window!r}"
        )
    if max_per_minute is not None and (not isinstance(max_per_minute, int) or max_per_minute < 1):
        errors.append(
            f"{ctx}.max_requests_per_minute must be a positive integer, got {max_per_minute!r}"
        )
    if max_per_window is not None and max_per_minute is not None:
        errors.append(
            f"{ctx} must set only one of 'max_requests_per_window' or "
            f"'max_requests_per_minute'"
        )

    max_tokens_window = raw.get("max_tokens_per_window")
    max_tokens_minute = raw.get("max_tokens_per_minute")
    if max_tokens_window is not None and (
        not isinstance(max_tokens_window, int) or max_tokens_window < 1
    ):
        errors.append(
            f"{ctx}.max_tokens_per_window must be a positive integer, got {max_tokens_window!r}"
        )
    if max_tokens_minute is not None and (
        not isinstance(max_tokens_minute, int) or max_tokens_minute < 1
    ):
        errors.append(
            f"{ctx}.max_tokens_per_minute must be a positive integer, got {max_tokens_minute!r}"
        )
    if max_tokens_window is not None and max_tokens_minute is not None:
        errors.append(
            f"{ctx} must set only one of 'max_tokens_per_window' or "
            f"'max_tokens_per_minute'"
        )

    window_seconds = raw.get("window_seconds")
    if window_seconds is not None and (
        not isinstance(window_seconds, (int, float)) or window_seconds <= 0
    ):
        errors.append(
            f"{ctx}.window_seconds must be a positive number, got {window_seconds!r}"
        )

    max_queue_size = raw.get("max_queue_size")
    if max_queue_size is not None and (not isinstance(max_queue_size, int) or max_queue_size < 1):
        errors.append(
            f"{ctx}.max_queue_size must be a positive integer, got {max_queue_size!r}"
        )

    acquire_timeout = raw.get("acquire_timeout_seconds")
    if acquire_timeout is not None and (
        not isinstance(acquire_timeout, (int, float)) or acquire_timeout <= 0
    ):
        errors.append(
            f"{ctx}.acquire_timeout_seconds must be a positive number, "
            f"got {acquire_timeout!r}"
        )

    has_limit = any(
        raw.get(name) is not None
        for name in (
            "max_concurrent_requests",
            "max_requests_per_window",
            "max_requests_per_minute",
            "max_tokens_per_window",
            "max_tokens_per_minute",
        )
    )
    if enabled and not has_limit:
        errors.append(
            f"{ctx} is enabled but no limits are configured. Set at least one of "
            f"'max_concurrent_requests', 'max_requests_per_window', "
            f"or 'max_requests_per_minute'."
        )


def _validate_simple_query_bypass(raw: Dict[str, Any], errors: List[str]) -> None:
    """Validate simple_query_bypass section in budget.yaml."""
    if not raw:
        return

    ctx = "simple_query_bypass"
    if not isinstance(raw, dict):
        errors.append(f"{ctx} must be a mapping/object, got {type(raw).__name__}")
        return

    enabled = raw.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        errors.append(f"{ctx}.enabled must be a boolean, got {enabled!r}")

    threshold = raw.get("route_confidence_threshold")
    if threshold is not None and (
        not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1
    ):
        errors.append(
            f"{ctx}.route_confidence_threshold must be between 0 and 1, got {threshold!r}"
        )

    force_schema = raw.get("force_full_workflow_if_output_schema")
    if force_schema is not None and not isinstance(force_schema, bool):
        errors.append(
            f"{ctx}.force_full_workflow_if_output_schema must be a boolean, "
            f"got {force_schema!r}"
        )

    allow_escalation = raw.get("allow_escalation_to_full_workflow")
    if allow_escalation is not None and not isinstance(allow_escalation, bool):
        errors.append(
            f"{ctx}.allow_escalation_to_full_workflow must be a boolean, "
            f"got {allow_escalation!r}"
        )

    max_tokens = raw.get("direct_answer_max_tokens")
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or max_tokens < 1
    ):
        errors.append(
            f"{ctx}.direct_answer_max_tokens must be a positive integer, got {max_tokens!r}"
        )


# ---------------------------------------------------------------------------
# Tools config validation
# ---------------------------------------------------------------------------

def validate_tools_config(
    tools: Dict[str, Dict[str, Any]],
    handler_names: List[str],
) -> List[str]:
    """
    Validate tools.yaml content.

    Args:
        tools: Parsed tools dict from tools.yaml (tool_name -> definition).
        handler_names: List of available handler names from tool_handlers.

    Returns:
        List of error strings. Empty list means valid.
    """
    errors: List[str] = []
    valid_phases = {"plan", "act", "review", "report"}
    available_handlers = set(handler_names)

    for tool_name, tool_def in tools.items():
        ctx = f"Tool '{tool_name}'"

        # Name validation
        if "__" in tool_name:
            errors.append(f"{ctx}: tool name must not contain '__'")

        # Required: description
        desc = tool_def.get("description")
        if not desc or not isinstance(desc, str):
            errors.append(
                f"{ctx} is missing required field 'description' (non-empty string)"
            )

        # Required: parameters
        params = tool_def.get("parameters")
        if not isinstance(params, dict):
            errors.append(
                f"{ctx} is missing required field 'parameters' "
                f"(must be a JSON Schema object)"
            )
        elif params.get("type") != "object":
            errors.append(
                f"{ctx} 'parameters.type' must be 'object', "
                f"got {params.get('type')!r}"
            )

        # Optional: phase_availability
        phases = tool_def.get("phase_availability")
        if phases is not None:
            if not isinstance(phases, list):
                errors.append(f"{ctx} 'phase_availability' must be a list")
            else:
                for p in phases:
                    if p not in valid_phases:
                        errors.append(
                            f"{ctx} references unknown phase '{p}'. "
                            f"Valid phases: {', '.join(sorted(valid_phases))}"
                        )

        # Optional: timeout_ms
        timeout = tool_def.get("timeout_ms")
        if timeout is not None and (not isinstance(timeout, int) or timeout < 1):
            errors.append(
                f"{ctx} 'timeout_ms' must be a positive integer, got {timeout!r}"
            )

        # Optional: max_calls_per_phase
        max_calls = tool_def.get("max_calls_per_phase")
        if max_calls is not None and (not isinstance(max_calls, int) or max_calls < 1):
            errors.append(
                f"{ctx} 'max_calls_per_phase' must be a positive integer, "
                f"got {max_calls!r}"
            )

        # Optional: retry_on_failure
        retry = tool_def.get("retry_on_failure")
        if retry is not None and not isinstance(retry, bool):
            errors.append(
                f"{ctx} 'retry_on_failure' must be a boolean, got {retry!r}"
            )

        # Optional: max_retries
        max_r = tool_def.get("max_retries")
        if max_r is not None and (not isinstance(max_r, int) or max_r < 0):
            errors.append(
                f"{ctx} 'max_retries' must be a non-negative integer, "
                f"got {max_r!r}"
            )

        # Optional: output_schema
        out_schema = tool_def.get("output_schema")
        if out_schema is not None:
            if not isinstance(out_schema, dict):
                errors.append(f"{ctx} 'output_schema' must be a JSON Schema dict")
            elif "type" not in out_schema:
                errors.append(f"{ctx} 'output_schema' must have a 'type' field")

        # Optional: category
        cat = tool_def.get("category")
        if cat is not None and (not isinstance(cat, str) or not cat.strip()):
            errors.append(
                f"{ctx} 'category' must be a non-empty string, got {cat!r}"
            )

        # Handler must exist
        if tool_name not in available_handlers:
            errors.append(
                f"{ctx} has no handler in tool_handlers. "
                f"Every tool in tools.yaml must have a corresponding Python handler."
            )

    return errors


# ---------------------------------------------------------------------------
# Providers config validation
# ---------------------------------------------------------------------------

_SUPPORTED_PROVIDERS = {"openai", "azure_openai", "anthropic"}

_REQUIRED_PROVIDER_FIELDS: Dict[str, List[str]] = {
    "openai": ["api_key"],
    "azure_openai": ["api_key", "endpoint"],
    "anthropic": ["api_key"],
}


def validate_providers_config(
    providers: Dict[str, Dict[str, Any]],
    default_provider: Optional[str],
) -> List[str]:
    """
    Validate providers.yaml content.

    Args:
        providers: Parsed providers dict (provider_type -> settings).
        default_provider: The default_provider value from the YAML root.

    Returns:
        List of error strings. Empty list means valid.
    """
    errors: List[str] = []

    if not providers:
        errors.append("providers.yaml must define at least one provider under 'providers'")
        return errors

    for provider_type, settings in providers.items():
        ctx = f"Provider '{provider_type}'"

        if provider_type not in _SUPPORTED_PROVIDERS:
            errors.append(
                f"{ctx} is not a supported provider type. "
                f"Supported: {', '.join(sorted(_SUPPORTED_PROVIDERS))}"
            )
            continue

        if not isinstance(settings, dict):
            errors.append(f"{ctx} must be a mapping of settings")
            continue

        # Check required fields
        required = _REQUIRED_PROVIDER_FIELDS.get(provider_type, [])
        for field in required:
            val = settings.get(field)
            if not val or not isinstance(val, str) or not val.strip():
                errors.append(f"{ctx} is missing required field '{field}'")

        # Validate timeout if present
        timeout = settings.get("timeout")
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                errors.append(
                    f"{ctx} 'timeout' must be a positive number, got {timeout!r}"
                )

    # Validate default_provider
    if default_provider is not None:
        if default_provider not in providers:
            available = ", ".join(sorted(providers.keys())) or "(none)"
            errors.append(
                f"default_provider '{default_provider}' is not defined under "
                f"'providers'. Available: {available}"
            )

    return errors


# ---------------------------------------------------------------------------
# Templates config validation
# ---------------------------------------------------------------------------

def validate_templates_config(
    templates: Dict[str, Dict[str, Any]],
    roles: Dict[str, Dict[str, Any]],
) -> List[str]:
    """
    Validate templates.yaml references against roles.yaml.

    Args:
        templates: Parsed templates dict from templates.yaml (template_id -> definition).
        roles: Parsed roles dict from roles.yaml (role_name -> definition).

    Returns:
        List of error strings. Empty list means valid.
    """
    errors: List[str] = []
    available_roles = set(roles.keys())

    # Build a set of valid role/sub_role pairs
    valid_sub_roles: Dict[str, set] = {}
    for role_name, role_def in roles.items():
        valid_sub_roles[role_name] = set(role_def.get("sub_roles", {}).keys())

    for tmpl_id, tmpl_def in templates.items():
        ctx = f"Template '{tmpl_id}'"

        # Required: name
        if not tmpl_def.get("name"):
            errors.append(f"{ctx} is missing required field 'name'")

        # Required: role
        role = tmpl_def.get("role")
        if not role:
            errors.append(f"{ctx} is missing required field 'role'")
            continue

        if role not in available_roles:
            errors.append(
                f"{ctx} references role '{role}' which is not defined in "
                f"roles.yaml. Available roles: {_sorted_list(available_roles)}"
            )
            continue

        # Optional: sub_role must exist under the referenced role
        sub_role = tmpl_def.get("sub_role")
        if sub_role:
            role_subs = valid_sub_roles.get(role, set())
            if sub_role not in role_subs:
                if role_subs:
                    errors.append(
                        f"{ctx} references sub_role '{sub_role}' which is not "
                        f"defined under role '{role}'. Available sub_roles: "
                        f"{_sorted_list(role_subs)}"
                    )
                else:
                    errors.append(
                        f"{ctx} references sub_role '{sub_role}' but role "
                        f"'{role}' has no sub_roles defined"
                    )

    return errors
