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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml

from ..adapters.domain_adapter import ReferenceDomainAdapter
from ..core_types import (
    AgentConfig,
    BudgetConfig,
    CostConfig,
    ModelConfig,
    ModelPricing,
    Phase,
    ToolDef,
)
from .config_validator import ConfigError, validate_config, validate_tools_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output bundle
# ---------------------------------------------------------------------------

@dataclass
class ConfigBundle:
    """Everything produced by loading the config folder."""
    domain_adapter: ReferenceDomainAdapter
    cost_config: CostConfig
    default_budget: BudgetConfig
    phase_limits: Dict[Phase, int]


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
# Core loader
# ---------------------------------------------------------------------------

def load_config(
    config_dir: Union[str, Path],
    tool_registry: Optional[Dict[str, ToolDef]] = None,
    tool_handlers: Optional[Dict[str, Callable]] = None,
) -> ConfigBundle:
    """
    Load all declarative config files and build framework objects.

    Args:
        config_dir: Path to the config directory containing roles.yaml,
            models.yaml, budget.yaml, and content folders.
        tool_registry: *Legacy* — mapping of tool name -> ToolDef (with
            handlers).  Used when tools.yaml does NOT exist.
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
        tool_names=list(built_tools.keys()),
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

    # -- Build DomainAdapter -------------------------------------------------
    domain_adapter = _build_domain_adapter(
        config_dir=config_dir,
        raw_roles=raw_roles,
        tool_registry=built_tools,
    )

    logger.info(
        f"Config loaded: {len(raw_roles)} role(s), "
        f"{len(raw_models)} model(s), "
        f"{len(built_tools)} tool(s), "
        f"budget max_tokens={default_budget.max_tokens}"
    )

    return ConfigBundle(
        domain_adapter=domain_adapter,
        cost_config=cost_config,
        default_budget=default_budget,
        phase_limits=phase_limits,
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
) -> Any:
    """
    Load config and create a fully-wired Orchestrator in one step.

    Args:
        config_dir: Path to the config directory.
        tool_registry: *Legacy* — Tool name -> ToolDef mapping.
        tool_handlers: *Declarative* — Tool name -> Python callable mapping
            (used with tools.yaml).
        llm: ToolCallingLLM instance.
        event_sink: Optional EventSink instance.
        max_review_cycles: Max review retry attempts (default 2).
        stream: Whether to stream LLM tokens.

    Returns:
        A configured Orchestrator instance.
    """
    # Import here to avoid circular imports
    from ..orchestrator import Orchestrator

    bundle = load_config(
        config_dir,
        tool_registry=tool_registry,
        tool_handlers=tool_handlers,
    )

    return Orchestrator(
        llm=llm,
        event_sink=event_sink,
        domain_adapter=bundle.domain_adapter,
        cost_config=bundle.cost_config,
        phase_limits=bundle.phase_limits,
        default_budget=bundle.default_budget,
        max_review_cycles=max_review_cycles,
        stream=stream,
    )
