"""
Configuration Loader for the Agentic Framework.

Reads declarative YAML/JSON/Markdown config files and produces the framework
objects needed to create an Orchestrator. Zero changes to existing framework
code required — the loader is a pure convenience layer.

Two ways to provide domain tools:

1. **Legacy** (tool_registry): Pass complete ToolDef objects — metadata and
   handlers combined. tools.yaml is not needed.

2. **Declarative** (tool_handlers + tools.yaml): Tool metadata lives in
   tools.yaml, Python callables are passed via tool_handlers. The loader
   merges them into ToolDef objects.

Usage:
    from framework.config import load_config

    # Option 1 — legacy (no tools.yaml)
    bundle = load_config(
        config_dir="framework/config",
        tool_registry={"search_documents": my_search_tool, ...},
    )

    # Option 2 — declarative (tools.yaml + handlers)
    bundle = load_config(
        config_dir="framework/config",
        tool_handlers={"search_documents": my_search_fn, ...},
    )

    orchestrator = Orchestrator(
        llm=my_llm,
        domain_adapter=bundle.domain_adapter,
        cost_config=bundle.cost_config,
        phase_limits=bundle.phase_limits,
        default_budget=bundle.default_budget,
    )
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

from ..adapters.domain_adapter import ReferenceDomainAdapter
from ..core_types import (
    AgentConfig,
    BudgetConfig,
    CostConfig,
    LLMRateLimitConfig,
    ModelConfig,
    ModelPricing,
    Phase,
    SimpleQueryBypassConfig,
    StallDetectionConfig,
    ToolDef,
)
from .config_validator import (
    ConfigError,
    validate_config,
    validate_providers_config,
    validate_tools_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output bundle
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """Resolved LLM provider settings from providers.yaml."""
    provider_type: str
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigBundle:
    """Everything produced by loading the config folder."""
    domain_adapter: ReferenceDomainAdapter
    cost_config: CostConfig
    default_budget: BudgetConfig
    phase_limits: Dict[Phase, int]
    provider_config: Optional[ProviderConfig] = None
    stall_config: Optional[StallDetectionConfig] = None
    llm_rate_limit: Optional[LLMRateLimitConfig] = None
    simple_query_bypass: SimpleQueryBypassConfig = field(
        default_factory=SimpleQueryBypassConfig
    )


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load and parse a YAML file."""
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError([f"Expected a YAML mapping in {path}, got {type(data).__name__}"])
    return data


def _read_text_file(config_dir: Path, rel_path: str) -> str:
    """Read a text file relative to config_dir."""
    return (config_dir / rel_path).read_text(encoding="utf-8")


def _read_json_file(config_dir: Path, rel_path: str) -> Dict[str, Any]:
    """Read and parse a JSON file relative to config_dir."""
    return json.loads((config_dir / rel_path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Environment variable resolution
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _resolve_env_vars(
    data: Any,
    context: str = "providers.yaml",
) -> Any:
    """
    Walk a parsed YAML structure and replace ``${VAR_NAME}`` references in
    string values with the corresponding environment variable.

    Raises ConfigError if a referenced variable is not set.
    """
    if isinstance(data, str):
        missing: List[str] = []

        def _replacer(match: re.Match) -> str:
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                missing.append(var_name)
                return match.group(0)  # keep original for error reporting
            return value

        result = _ENV_VAR_PATTERN.sub(_replacer, data)
        if missing:
            raise ConfigError([
                f"{context}: environment variable '{v}' is not set"
                for v in missing
            ])
        return result

    if isinstance(data, dict):
        return {
            k: _resolve_env_vars(v, context=f"{context}.{k}")
            for k, v in data.items()
        }

    if isinstance(data, list):
        return [
            _resolve_env_vars(item, context=f"{context}[{i}]")
            for i, item in enumerate(data)
        ]

    # Numbers, booleans, None — pass through untouched
    return data


# ---------------------------------------------------------------------------
# Provider config loading
# ---------------------------------------------------------------------------

def _try_load_dotenv(config_dir: Path) -> None:
    """
    Attempt to load a ``.env`` file so that ``${VAR}`` references in
    ``providers.yaml`` can be resolved.  Searches from *config_dir* upward
    to the repository root (the first directory containing ``.git``,
    ``pyproject.toml``, or ``setup.py``).

    Requires the optional ``python-dotenv`` package.  If it is not installed
    the function is a silent no-op — users must then export env vars
    themselves.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]
    except ImportError:
        logger.debug(
            "python-dotenv not installed — skipping .env auto-load. "
            "Install it with: pip install python-dotenv"
        )
        return

    # Walk upward from config_dir looking for .env
    search = config_dir.resolve()
    for parent in [search, *search.parents]:
        env_path = parent / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)
            logger.info(f".env loaded from {env_path}")
            return
        # Stop at repo root indicators
        if any((parent / m).exists() for m in (".git", "pyproject.toml", "setup.py")):
            # Check this directory but don't go higher
            break

    logger.debug("No .env file found in parent directories.")


def _load_providers(
    config_dir: Path,
    provider_override: Optional[str] = None,
) -> Optional[ProviderConfig]:
    """
    Load ``providers.yaml`` if it exists and return a resolved ProviderConfig.

    Returns None if the file does not exist (backward-compatible).
    """
    providers_path = config_dir / "providers.yaml"
    if not providers_path.is_file():
        return None

    # Auto-load .env before resolving ${VAR} references
    _try_load_dotenv(config_dir)

    raw = _load_yaml(providers_path)
    raw_providers = raw.get("providers", {})
    default_provider = raw.get("default_provider")

    # Validate structure before env-var resolution
    errors = validate_providers_config(raw_providers, default_provider)
    if errors:
        raise ConfigError(errors)

    # Resolve environment variable references
    resolved_providers = _resolve_env_vars(raw_providers, context="providers")

    # Determine which provider to use
    provider_type = provider_override or default_provider
    if provider_type is None:
        # Use the first (only) provider if there's exactly one
        if len(resolved_providers) == 1:
            provider_type = next(iter(resolved_providers))
        else:
            raise ConfigError([
                "providers.yaml defines multiple providers but no "
                "'default_provider' is set. Set 'default_provider' or "
                "pass provider_override to select one."
            ])

    if provider_type not in resolved_providers:
        available = ", ".join(sorted(resolved_providers.keys()))
        raise ConfigError([
            f"Provider '{provider_type}' is not defined in providers.yaml. "
            f"Available: {available}"
        ])

    settings = resolved_providers[provider_type]

    # Build kwargs for create_tool_calling_llm
    kwargs: Dict[str, Any] = {}
    if provider_type == "azure_openai":
        kwargs["endpoint"] = settings["endpoint"]
        kwargs["api_key"] = settings["api_key"]
        kwargs["api_version"] = settings.get("api_version", "2024-12-01-preview")
    elif provider_type == "openai":
        kwargs["api_key"] = settings["api_key"]
    elif provider_type == "anthropic":
        kwargs["api_key"] = settings["api_key"]

    if "timeout" in settings:
        kwargs["timeout"] = float(settings["timeout"])

    return ProviderConfig(provider_type=provider_type, kwargs=kwargs)


def build_llm_from_provider_config(
    provider_config: ProviderConfig,
    model: str,
    cost_config: Optional[CostConfig] = None,
    llm_rate_limit: Optional[LLMRateLimitConfig] = None,
) -> Any:
    """
    Create a ToolCallingLLM instance from a resolved ProviderConfig.

    Args:
        provider_config: Resolved provider settings.
        model: Model name or deployment name.
        cost_config: Optional pricing config for cost tracking.
        llm_rate_limit: Optional queue/rate-limit settings for all LLM calls.

    Returns:
        A ToolCallingLLM-compatible adapter instance.
    """
    from ..adapters.llm_adapter import create_tool_calling_llm

    return create_tool_calling_llm(
        provider_type=provider_config.provider_type,
        model=model,
        cost_config=cost_config,
        llm_rate_limit=llm_rate_limit,
        **provider_config.kwargs,
    )


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load_config(
    config_dir: Union[str, Path],
    tool_registry: Optional[Dict[str, ToolDef]] = None,
    tool_handlers: Optional[Dict[str, Callable]] = None,
    provider_override: Optional[str] = None,
) -> ConfigBundle:
    """
    Load all declarative config files and build framework objects.

    Args:
        config_dir: Path to the config directory containing roles.yaml,
            models.yaml, budget.yaml, and content folders.
        tool_registry: *Legacy* — mapping of tool name -> ToolDef (with
            handlers).  Used when tools.yaml does NOT exist.
        provider_override: If set, override the ``default_provider``
            in providers.yaml with this value.
        tool_handlers: *Declarative* — mapping of tool name -> Python
            callable.  Used together with tools.yaml, which provides the
            metadata (description, parameters, constraints).

    Resolution:
        * If ``tools.yaml`` exists in config_dir → ``tool_handlers`` is used
          to attach Python callables to the metadata from YAML.
          ``tool_registry`` is ignored (with a warning if also provided).
        * If ``tools.yaml`` does NOT exist → ``tool_registry`` is used
          directly (legacy behaviour, fully backward-compatible).

    Returns:
        ConfigBundle with domain_adapter, cost_config, default_budget,
        and phase_limits ready for Orchestrator construction.

    Raises:
        ConfigError: If any validation check fails. All errors are collected
            and reported at once.
    """
    config_dir = Path(config_dir).resolve()
    tool_registry = tool_registry or {}
    tool_handlers = tool_handlers or {}

    # -- Check required files exist ------------------------------------------
    required_files = ["roles.yaml", "models.yaml", "budget.yaml"]
    missing = [f for f in required_files if not (config_dir / f).is_file()]
    if missing:
        raise ConfigError([
            f"Required config file not found: {config_dir / f}"
            for f in missing
        ])

    # -- Parse YAML files ----------------------------------------------------
    models_data = _load_yaml(config_dir / "models.yaml")
    budget_data = _load_yaml(config_dir / "budget.yaml")
    roles_data = _load_yaml(config_dir / "roles.yaml")

    raw_models = models_data.get("models", {})
    raw_budget = budget_data.get("budget_defaults", {})
    raw_phase_limits = budget_data.get("phase_limits", {})
    raw_stall = budget_data.get("stall_detection", {})
    raw_llm_rate_limit = budget_data.get("llm_rate_limit", {})
    raw_simple_query_bypass = budget_data.get("simple_query_bypass", {})
    raw_roles = roles_data.get("roles", {})

    # -- Resolve tools (declarative vs legacy) -------------------------------
    tools_yaml_path = config_dir / "tools.yaml"
    use_tools_yaml = tools_yaml_path.is_file()

    if use_tools_yaml:
        # Declarative path: build ToolDef objects from tools.yaml + handlers
        if tool_registry:
            logger.warning(
                "Both tool_registry and tools.yaml provided. "
                "tools.yaml takes precedence; tool_registry is ignored."
            )
        raw_tools = _load_yaml(tools_yaml_path).get("tools", {})

        # Validate tools.yaml content
        tools_errors = validate_tools_config(raw_tools, list(tool_handlers.keys()))
        # We collect these and combine with other errors below

        # Build ToolDef objects from YAML metadata + Python handlers
        built_tools = _build_tools_from_yaml(raw_tools, tool_handlers)
    else:
        # Legacy path: use tool_registry directly
        tools_errors = []
        built_tools = tool_registry

    # -- Validate everything -------------------------------------------------
    errors = validate_config(
        config_dir=config_dir,
        roles=raw_roles,
        models=raw_models,
        budget=raw_budget,
        phase_limits=raw_phase_limits,
        llm_rate_limit=raw_llm_rate_limit,
        tool_names=list(built_tools.keys()),
        simple_query_bypass=raw_simple_query_bypass,
    )
    errors.extend(tools_errors)
    if errors:
        raise ConfigError(errors)

    # -- Build CostConfig ----------------------------------------------------
    cost_config = _build_cost_config(raw_models)

    # -- Build BudgetConfig --------------------------------------------------
    default_budget = _build_budget_config(raw_budget)

    # -- Build phase limits --------------------------------------------------
    phase_limits = _build_phase_limits(raw_phase_limits)

    # -- Build stall detection config ----------------------------------------
    stall_config = _build_stall_config(raw_stall) if raw_stall else None

    # -- Build LLM rate-limit config -----------------------------------------
    llm_rate_limit = _build_llm_rate_limit_config(raw_llm_rate_limit)

    # -- Build simple-query bypass config ------------------------------------
    simple_query_bypass = _build_simple_query_bypass_config(raw_simple_query_bypass)

    # -- Build DomainAdapter -------------------------------------------------
    domain_adapter = _build_domain_adapter(
        config_dir=config_dir,
        raw_roles=raw_roles,
        tool_registry=built_tools,
    )

    # -- Load providers (optional) -------------------------------------------
    # Provider loading is deferred — only attempted when the caller needs
    # an auto-created LLM (i.e., no explicit llm= passed).  This avoids
    # failing on missing env vars when running in mock / offline mode.
    provider_config = None
    if provider_override is not None:
        # Explicit request → resolve now
        provider_config = _load_providers(config_dir, provider_override)

    logger.info(
        f"Config loaded: {len(raw_roles)} role(s), "
        f"{len(raw_models)} model(s), "
        f"{len(built_tools)} tool(s), "
        f"budget max_tokens={default_budget.max_tokens}"
        + (f", provider={provider_config.provider_type}" if provider_config else "")
        + (", llm_rate_limit=enabled" if llm_rate_limit and llm_rate_limit.enabled else "")
        + (", simple_query_bypass=enabled" if simple_query_bypass.enabled else "")
    )

    return ConfigBundle(
        domain_adapter=domain_adapter,
        cost_config=cost_config,
        default_budget=default_budget,
        phase_limits=phase_limits,
        provider_config=provider_config,
        stall_config=stall_config,
        llm_rate_limit=llm_rate_limit,
        simple_query_bypass=simple_query_bypass,
    )


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def _build_cost_config(raw_models: Dict[str, Any]) -> CostConfig:
    """Build CostConfig from parsed models dict."""
    models: Dict[str, ModelPricing] = {}
    for name, pricing in raw_models.items():
        models[name] = ModelPricing(
            input_price_per_1k=float(pricing.get("input_price_per_1k", 0.0)),
            output_price_per_1k=float(pricing.get("output_price_per_1k", 0.0)),
            context_window=int(pricing.get("context_window", 128000)),
        )
    return CostConfig(models=models)


def _build_budget_config(raw_budget: Dict[str, Any]) -> BudgetConfig:
    """Build BudgetConfig from parsed budget dict."""
    return BudgetConfig(
        max_tokens=raw_budget.get("max_tokens"),
        max_cost=raw_budget.get("max_cost"),
        max_duration_ms=raw_budget.get("max_duration_ms"),
        max_agent_depth=raw_budget.get("max_agent_depth", 3),
        max_parallel_agents=raw_budget.get("max_parallel_agents", 3),
        max_sub_agents_total=raw_budget.get("max_sub_agents_total", 3),
        inherit_remaining=raw_budget.get("inherit_remaining", True),
        child_budget_fraction=raw_budget.get("child_budget_fraction", 0.5),
        parent_recovery_budget_pct=raw_budget.get("parent_recovery_budget_pct", 0.10),
        max_child_review_cycles=raw_budget.get("max_child_review_cycles"),
        context_soft_compaction_pct=raw_budget.get("context_soft_compaction_pct", 0.40),
        context_hard_truncation_pct=raw_budget.get("context_hard_truncation_pct", 0.65),
    )


def _build_stall_config(raw_stall: Dict[str, Any]) -> StallDetectionConfig:
    """Build StallDetectionConfig from parsed budget.yaml stall_detection section."""
    return StallDetectionConfig(
        max_consecutive_tool_failures=raw_stall.get("max_consecutive_tool_failures", 3),
        max_framework_stall_iterations=raw_stall.get("max_framework_stall_iterations", 5),
        max_fw_only_consecutive_iterations=raw_stall.get("max_fw_only_consecutive_iterations", 10),
        max_duplicate_call_iterations=raw_stall.get("max_duplicate_call_iterations", 4),
        duplicate_call_window=raw_stall.get("duplicate_call_window", 8),
    )


def _build_llm_rate_limit_config(raw_limit: Dict[str, Any]) -> Optional[LLMRateLimitConfig]:
    """Build LLMRateLimitConfig from parsed budget.yaml llm_rate_limit section."""
    if not raw_limit:
        return None

    max_requests_per_window = raw_limit.get("max_requests_per_window")
    max_requests_per_minute = raw_limit.get("max_requests_per_minute")
    if max_requests_per_window is None and max_requests_per_minute is not None:
        max_requests_per_window = max_requests_per_minute
    max_tokens_per_window = raw_limit.get("max_tokens_per_window")
    max_tokens_per_minute = raw_limit.get("max_tokens_per_minute")
    if max_tokens_per_window is None and max_tokens_per_minute is not None:
        max_tokens_per_window = max_tokens_per_minute

    return LLMRateLimitConfig(
        enabled=bool(raw_limit.get("enabled", True)),
        max_concurrent_requests=raw_limit.get("max_concurrent_requests"),
        max_requests_per_window=max_requests_per_window,
        max_tokens_per_window=max_tokens_per_window,
        window_seconds=float(raw_limit.get("window_seconds", 60.0)),
        max_queue_size=raw_limit.get("max_queue_size"),
        acquire_timeout_seconds=raw_limit.get("acquire_timeout_seconds"),
    )


def _build_simple_query_bypass_config(raw_cfg: Dict[str, Any]) -> SimpleQueryBypassConfig:
    """Build SimpleQueryBypassConfig from parsed budget.yaml section."""
    if not raw_cfg:
        return SimpleQueryBypassConfig()

    return SimpleQueryBypassConfig(
        enabled=bool(raw_cfg.get("enabled", True)),
        route_confidence_threshold=float(raw_cfg.get("route_confidence_threshold", 0.80)),
        force_full_workflow_if_output_schema=bool(
            raw_cfg.get("force_full_workflow_if_output_schema", True)
        ),
        allow_escalation_to_full_workflow=bool(
            raw_cfg.get("allow_escalation_to_full_workflow", True)
        ),
        direct_answer_max_tokens=int(raw_cfg.get("direct_answer_max_tokens", 512)),
    )


def _build_phase_limits(raw_limits: Dict[str, Any]) -> Dict[Phase, int]:
    """Build phase limits dict from parsed YAML."""
    phase_map = {
        "plan": Phase.PLAN,
        "act": Phase.ACT,
        "review": Phase.REVIEW,
        "report": Phase.REPORT,
    }
    result: Dict[Phase, int] = {}
    for name, value in raw_limits.items():
        phase = phase_map.get(name)
        if phase:
            result[phase] = int(value)
    return result


def _build_domain_adapter(
    config_dir: Path,
    raw_roles: Dict[str, Any],
    tool_registry: Dict[str, ToolDef],
) -> ReferenceDomainAdapter:
    """Build and populate a ReferenceDomainAdapter from parsed roles."""
    adapter = ReferenceDomainAdapter()

    for role_name, role_def in raw_roles.items():
        # Read system prompt content
        system_prompt = _read_text_file(config_dir, role_def["system_prompt"])

        # Build ModelConfig
        mc_raw = role_def.get("model_config", {})
        model_config = ModelConfig(
            temperature=mc_raw.get("temperature", 0.7),
            top_p=mc_raw.get("top_p", 1.0),
            max_tokens=mc_raw.get("max_tokens", 4096),
        )

        # Build AgentConfig
        config = AgentConfig(
            role=role_name,
            system_prompt=system_prompt,
            model=role_def["model"],
            model_config=model_config,
        )

        # Resolve tools
        tools = _resolve_tools(role_def.get("tools", []), tool_registry)

        # Read output schema (optional)
        output_schema = None
        if role_def.get("output_schema"):
            output_schema = _read_json_file(config_dir, role_def["output_schema"])

        # Read report template (optional)
        report_template = None
        if role_def.get("report_template"):
            report_template = _read_text_file(config_dir, role_def["report_template"])

        # Register role
        adapter.register_role(
            role=role_name,
            config=config,
            tools=tools,
            output_schema=output_schema,
            report_template=report_template,
            description=role_def.get("description", ""),
        )

        # Register sub-roles
        for sr_name, sr_def in role_def.get("sub_roles", {}).items():
            _register_sub_role(
                adapter=adapter,
                config_dir=config_dir,
                role_name=role_name,
                sr_name=sr_name,
                sr_def=sr_def,
                parent_role_def=role_def,
                parent_config=config,
                tool_registry=tool_registry,
            )

    return adapter


def _register_sub_role(
    adapter: ReferenceDomainAdapter,
    config_dir: Path,
    role_name: str,
    sr_name: str,
    sr_def: Dict[str, Any],
    parent_role_def: Dict[str, Any],
    parent_config: AgentConfig,
    tool_registry: Dict[str, ToolDef],
) -> None:
    """Register a single sub-role under a parent role."""

    # Build config overrides if anything is specified
    config_overrides = None
    has_config_override = (
        sr_def.get("system_prompt")
        or sr_def.get("model")
        or sr_def.get("model_config")
    )
    if has_config_override:
        # Read sub-role system prompt (or inherit parent's)
        sr_system_prompt = parent_config.system_prompt
        if sr_def.get("system_prompt"):
            sr_system_prompt = _read_text_file(config_dir, sr_def["system_prompt"])

        # Build sub-role model config (inherit parent defaults)
        parent_mc = parent_role_def.get("model_config", {})
        sr_mc_raw = sr_def.get("model_config", {})
        sr_model_config = ModelConfig(
            temperature=sr_mc_raw.get("temperature", parent_mc.get("temperature", 0.7)),
            top_p=sr_mc_raw.get("top_p", parent_mc.get("top_p", 1.0)),
            max_tokens=sr_mc_raw.get("max_tokens", parent_mc.get("max_tokens", 4096)),
        )

        config_overrides = AgentConfig(
            role=role_name,
            sub_role=sr_name,
            system_prompt=sr_system_prompt,
            model=sr_def.get("model", parent_role_def["model"]),
            model_config=sr_model_config,
        )

    # Tools override (None = inherit parent)
    tools_override = None
    if "tools" in sr_def:
        tools_override = _resolve_tools(sr_def["tools"], tool_registry)

    # Output schema override
    output_schema_override = None
    if sr_def.get("output_schema"):
        output_schema_override = _read_json_file(config_dir, sr_def["output_schema"])

    # Report template override
    report_template_override = None
    if sr_def.get("report_template"):
        report_template_override = _read_text_file(config_dir, sr_def["report_template"])

    adapter.register_sub_role(
        role=role_name,
        sub_role=sr_name,
        description=sr_def.get("description", ""),
        config_overrides=config_overrides,
        tools_override=tools_override,
        output_schema_override=output_schema_override,
        report_template_override=report_template_override,
    )


def _resolve_tools(
    tool_names: List[str],
    tool_registry: Dict[str, ToolDef],
) -> List[ToolDef]:
    """Resolve tool name strings to ToolDef objects."""
    resolved = []
    for name in tool_names:
        tool = tool_registry.get(name)
        if tool is None:
            available = ", ".join(sorted(tool_registry.keys())) or "(none)"
            raise ConfigError(
                [f"Role references undefined tool '{name}'. Available tools: {available}"]
            )
        resolved.append(tool)
    return resolved


# ---------------------------------------------------------------------------
# tools.yaml builder
# ---------------------------------------------------------------------------

_PHASE_MAP = {
    "plan": Phase.PLAN,
    "act": Phase.ACT,
    "review": Phase.REVIEW,
    "report": Phase.REPORT,
}


def _build_tools_from_yaml(
    raw_tools: Dict[str, Dict[str, Any]],
    tool_handlers: Dict[str, Callable],
) -> Dict[str, ToolDef]:
    """
    Build ToolDef objects from tools.yaml metadata + Python handlers.

    Each entry in raw_tools is a tool name mapped to its YAML definition.
    The handler is looked up from tool_handlers by the same name.

    This function assumes validation has already passed — missing handlers
    and invalid field values are caught by validate_tools_config().
    """
    result: Dict[str, ToolDef] = {}

    for tool_name, tool_def in raw_tools.items():
        handler = tool_handlers.get(tool_name)

        # Parse phase_availability (default: all phases)
        phase_names = tool_def.get("phase_availability")
        if phase_names is not None:
            phase_availability = [
                _PHASE_MAP[p] for p in phase_names if p in _PHASE_MAP
            ]
        else:
            phase_availability = list(Phase)

        result[tool_name] = ToolDef(
            name=tool_name,
            description=tool_def.get("description", ""),
            parameters=tool_def.get("parameters", {"type": "object", "properties": {}}),
            handler=handler,
            phase_availability=phase_availability,
            timeout_ms=tool_def.get("timeout_ms", 30000),
            max_calls_per_phase=tool_def.get("max_calls_per_phase"),
            output_schema=tool_def.get("output_schema"),
            category=tool_def.get("category"),
            retry_on_failure=tool_def.get("retry_on_failure", False),
            max_retries=tool_def.get("max_retries", 0),
        )

    return result


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_orchestrator_from_config(
    config_dir: Union[str, Path],
    tool_registry: Optional[Dict[str, ToolDef]] = None,
    tool_handlers: Optional[Dict[str, Callable]] = None,
    llm: Any = None,
    event_sink: Any = None,
    max_review_cycles: int = 2,
    stream: bool = False,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Any:
    """
    Load config and create a fully-wired Orchestrator in one step.

    If ``llm`` is not provided but ``providers.yaml`` exists in the config
    directory, the LLM adapter is created automatically from environment
    variables referenced in that file.

    Args:
        config_dir: Path to the config directory.
        tool_registry: *Legacy* — Tool name -> ToolDef mapping.
        tool_handlers: *Declarative* — Tool name -> Python callable mapping
            (used with tools.yaml).
        llm: ToolCallingLLM instance.  If None, built from providers.yaml.
        event_sink: Optional EventSink instance.
        max_review_cycles: Max review retry attempts (default 2).
        stream: Whether to stream LLM tokens.
        provider_override: Override ``default_provider`` from providers.yaml.
        model_override: Model name to use when building LLM from providers.yaml.
            If not set, falls back to the first model in models.yaml.

    Returns:
        A configured Orchestrator instance.
    """
    # Import here to avoid circular imports
    from ..orchestrator import Orchestrator

    bundle = load_config(
        config_dir,
        tool_registry=tool_registry,
        tool_handlers=tool_handlers,
        provider_override=provider_override,
    )

    # Auto-build LLM from providers.yaml if no explicit llm provided
    provider_config = bundle.provider_config
    if llm is None:
        # Try to load providers.yaml if not already loaded
        if provider_config is None:
            provider_config = _load_providers(
                Path(config_dir).resolve(), provider_override,
            )
        if provider_config is not None:
            # Determine model: explicit override > first model in cost_config
            model = model_override
            if model is None and bundle.cost_config.models:
                model = next(iter(bundle.cost_config.models))
            if model is None:
                raise ConfigError([
                    "Cannot auto-create LLM: no model specified. "
                    "Pass model_override or define models in models.yaml."
                ])
            llm = build_llm_from_provider_config(
                provider_config,
                model,
                bundle.cost_config,
                llm_rate_limit=bundle.llm_rate_limit,
            )
            logger.info(
                f"LLM auto-created: provider={provider_config.provider_type}, "
                f"model={model}"
            )
    elif bundle.llm_rate_limit and bundle.llm_rate_limit.enabled:
        from ..adapters.llm_rate_limiter import RateLimitedToolCallingLLM
        if not isinstance(llm, RateLimitedToolCallingLLM):
            llm = RateLimitedToolCallingLLM(llm=llm, config=bundle.llm_rate_limit)

    return Orchestrator(
        llm=llm,
        event_sink=event_sink,
        domain_adapter=bundle.domain_adapter,
        cost_config=bundle.cost_config,
        phase_limits=bundle.phase_limits,
        default_budget=bundle.default_budget,
        max_review_cycles=max_review_cycles,
        stream=stream,
        stall_config=bundle.stall_config,
        simple_query_bypass=bundle.simple_query_bypass,
    )
