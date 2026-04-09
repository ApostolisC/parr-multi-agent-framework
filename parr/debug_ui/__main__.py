"""
Launch the PARR Debug UI from the command line.

Usage::

    python -m parr.debug_ui --persist-dir ./sessions
    python -m parr.debug_ui --persist-dir ./sessions --port 9090

Enable session creation from the UI by pointing at a config directory::

    python -m parr.debug_ui --persist-dir ./sessions \\
        --config-dir ./examples/research_assistant/config

Full workflow with real domain tools (searches, reads, etc.)::

    python -m parr.debug_ui --persist-dir ./sessions \\
        --config-dir ./examples/research_assistant/config \\
        --tools-module examples.research_assistant.run
"""

import argparse
import dataclasses
import importlib
import logging

from parr.core_types import ToolDef

from .data_source import FileSystemDataSource
from .server import start_server, SSEHub, SSEEventSink

logger = logging.getLogger(__name__)


def _discover_tools_from_module(module_path: str) -> dict[str, ToolDef]:
    """
    Import *module_path* and collect every module-level ``ToolDef`` instance.

    Returns a ``{name: ToolDef}`` dict suitable for ``tool_registry``.
    """
    mod = importlib.import_module(module_path)
    tools: dict[str, ToolDef] = {}
    for attr_name in dir(mod):
        obj = getattr(mod, attr_name)
        if isinstance(obj, ToolDef) and obj.handler is not None:
            tools[obj.name] = obj
            logger.info(f"Discovered tool '{obj.name}' from {module_path}")
    return tools


def _build_workflow_runner(
    config_dir: str,
    persist_dir: str,
    provider: str | None,
    model: str | None,
    tools_module: str | None = None,
    event_sink: object | None = None,
):
    """
    Build an async workflow runner from a config directory.

    Returns ``(runner_func, available_roles, orchestrator)`` where
    *runner_func* is an ``async (task, role) -> AgentOutput`` callable,
    *available_roles* is the list of role names from ``roles.yaml``,
    and *orchestrator* is the Orchestrator instance for management APIs.
    """
    from pathlib import Path

    import yaml

    from parr.config import create_orchestrator_from_config

    config_path = Path(config_dir).resolve()

    # Extract role names directly from roles.yaml (no validation needed)
    roles_file = config_path / "roles.yaml"
    available_roles: list[str] = []
    if roles_file.exists():
        data = yaml.safe_load(roles_file.read_text(encoding="utf-8")) or {}
        available_roles = list(data.get("roles", {}).keys())

    # Discover domain tools from the specified Python module
    tool_registry: dict[str, ToolDef] | None = None
    if tools_module:
        tool_registry = _discover_tools_from_module(tools_module)
        if tool_registry:
            logger.info(f"Loaded {len(tool_registry)} tool(s) from {tools_module}: {list(tool_registry.keys())}")
        else:
            logger.warning(f"No ToolDef instances found in {tools_module}")

    # Try to build orchestrator from config.
    try:
        orchestrator = create_orchestrator_from_config(
            config_dir=config_dir,
            tool_registry=tool_registry,
            provider_override=provider,
            model_override=model,
            event_sink=event_sink,
        )
    except Exception as e:
        if tool_registry:
            # Tools were provided but config still failed — don't silently fall back
            raise
        # Fall back: build a minimal orchestrator without tool validation
        logger.warning(f"Full config load failed ({e}), building minimal orchestrator")
        orchestrator = _build_minimal_orchestrator(config_path, provider, model, event_sink)

    # Point persistence at the same directory the UI reads from
    orchestrator._persist_dir = persist_dir

    async def runner(task: str, role: str):
        return await orchestrator.start_workflow(task=task, role=role)

    async def continue_runner(task: str, role: str, additional_context: str):
        return await orchestrator.start_workflow(
            task=task, role=role, additional_context=additional_context,
        )

    return runner, continue_runner, available_roles, orchestrator


def _build_minimal_orchestrator(config_path, provider: str | None, model: str | None, event_sink: object | None = None):
    """Build an Orchestrator bypassing tool validation for CLI use."""
    import json

    from parr.orchestrator import Orchestrator
    from parr.core_types import ToolDef
    from parr.config.config_loader import (
        _load_yaml,
        _build_cost_config,
        _build_budget_config,
        _build_llm_rate_limit_config,
        _build_phase_limits,
        _build_stall_config,
        _build_domain_adapter,
        _load_providers,
        build_llm_from_provider_config,
    )

    models_data = _load_yaml(config_path / "models.yaml")
    budget_data = _load_yaml(config_path / "budget.yaml")
    roles_data = _load_yaml(config_path / "roles.yaml")

    raw_models = models_data.get("models", {})
    raw_budget = budget_data.get("budget_defaults", {})
    raw_phase_limits = budget_data.get("phase_limits", {})
    raw_stall = budget_data.get("stall_detection", {})
    raw_llm_rate_limit = budget_data.get("llm_rate_limit", {})
    raw_roles = roles_data.get("roles", {})

    cost_config = _build_cost_config(raw_models)
    default_budget = _build_budget_config(raw_budget)
    phase_limits = _build_phase_limits(raw_phase_limits)
    stall_config = _build_stall_config(raw_stall) if raw_stall else None
    llm_rate_limit = _build_llm_rate_limit_config(raw_llm_rate_limit)

    # Collect all tool names referenced by roles and create stub ToolDefs
    # so the domain adapter can load system prompts and output schemas
    all_tool_names: set[str] = set()
    for role_def in raw_roles.values():
        for t in role_def.get("tools", []):
            all_tool_names.add(t)
        for sr_def in role_def.get("sub_roles", {}).values():
            for t in sr_def.get("tools", []):
                all_tool_names.add(t)

    async def _stub_handler(**kwargs):
        return json.dumps({"error": "Tool not available in debug UI mode. Domain tools require the full application entry point (e.g. run.py)."})

    stub_registry = {
        name: ToolDef(
            name=name,
            description=f"(stub — domain tool not available in CLI mode)",
            parameters={"type": "object", "properties": {}},
            handler=_stub_handler,
        )
        for name in all_tool_names
    }

    domain_adapter = _build_domain_adapter(
        config_dir=config_path,
        raw_roles=raw_roles,
        tool_registry=stub_registry,
    )

    # Build LLM from providers.yaml
    provider_config = _load_providers(config_path, provider)
    if provider_config is None:
        raise RuntimeError(
            "No providers.yaml found in config directory. "
            "Cannot auto-create LLM for workflow launching."
        )
    llm_model = model
    if llm_model is None and cost_config.models:
        llm_model = next(iter(cost_config.models))
    if llm_model is None:
        raise RuntimeError("No model specified and none found in models.yaml.")

    llm = build_llm_from_provider_config(
        provider_config=provider_config,
        model=llm_model,
        cost_config=cost_config,
        llm_rate_limit=llm_rate_limit,
    )

    return Orchestrator(
        llm=llm,
        domain_adapter=domain_adapter,
        cost_config=cost_config,
        phase_limits=phase_limits,
        default_budget=default_budget,
        stall_config=stall_config,
        event_sink=event_sink,
    )


def _serialize_role_details(domain_adapter) -> list[dict]:
    """Extract JSON-serializable role details from a domain adapter."""
    roles = []
    for role_name, entry in domain_adapter._roles.items():
        role_info = {
            "name": role_name,
            "description": entry.description,
            "model": entry.config.model,
            "model_config": {
                "temperature": entry.config.model_config.temperature,
                "top_p": entry.config.model_config.top_p,
                "max_tokens": entry.config.model_config.max_tokens,
            },
            "tools": [t.name for t in entry.tools],
            "has_output_schema": entry.output_schema is not None,
            "has_report_template": entry.report_template is not None,
            "sub_roles": [
                {"name": sr_name, "description": sr_entry.description}
                for sr_name, sr_entry in entry.sub_roles.items()
            ],
        }
        roles.append(role_info)
    return roles


def _serialize_tool_details(domain_adapter) -> list[dict]:
    """Extract JSON-serializable tool metadata from a domain adapter."""
    seen: set[str] = set()
    tools = []
    for entry in domain_adapter._roles.values():
        for tool in entry.tools:
            if tool.name in seen:
                continue
            seen.add(tool.name)
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "category": tool.category,
                "is_framework_tool": tool.is_framework_tool,
                "is_read_only": tool.is_read_only,
                "phase_availability": [p.value for p in tool.phase_availability],
            })
    return tools


def _serialize_budget(budget_config) -> dict:
    """Convert a BudgetConfig dataclass to a JSON-serializable dict."""
    return dataclasses.asdict(budget_config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PARR Debug UI — inspect persisted agent sessions",
    )
    parser.add_argument(
        "--persist-dir",
        required=True,
        help="Path to the framework's persistence directory",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Path to a PARR config directory (roles.yaml, models.yaml, etc.) "
             "to enable launching new sessions from the UI",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind to (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Override default LLM provider (e.g. openai, anthropic)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override LLM model name (e.g. gpt-4o, claude-3-5-sonnet)",
    )
    parser.add_argument(
        "--tools-module",
        default=None,
        help="Python module that defines ToolDef objects for domain tools. "
             "e.g. examples.research_assistant.run — the module is imported "
             "and all module-level ToolDef instances are registered automatically.",
    )
    args = parser.parse_args()

    workflow_runner = None
    continue_func = None
    available_roles = None
    cancel_func = None
    role_details: list[dict] = []
    tool_details: list[dict] = []
    budget_config: dict = {}
    sse_hub = None

    if args.config_dir:
        sse_hub = SSEHub()
        event_sink = SSEEventSink(sse_hub)
        try:
            workflow_runner, continue_func, available_roles, orchestrator = _build_workflow_runner(
                config_dir=args.config_dir,
                persist_dir=args.persist_dir,
                provider=args.provider,
                model=args.model,
                tools_module=args.tools_module,
                event_sink=event_sink,
            )
            cancel_func = orchestrator.cancel_workflow
            if orchestrator._domain_adapter:
                role_details = _serialize_role_details(orchestrator._domain_adapter)
                tool_details = _serialize_tool_details(orchestrator._domain_adapter)
            budget_config = _serialize_budget(orchestrator._default_budget)
            print(f"Workflow runner loaded from: {args.config_dir}")
        except Exception as e:
            logger.error(f"Failed to load config from {args.config_dir}: {e}", exc_info=True)
            print(f"WARNING: Could not load config — session creation disabled.\n  Error: {e}")
            sse_hub = None  # disable SSE on config failure

    data_source = FileSystemDataSource(args.persist_dir)

    start_server(
        persist_dir=args.persist_dir,
        host=args.host,
        port=args.port,
        workflow_runner=workflow_runner,
        available_roles=available_roles,
        cancel_func=cancel_func,
        continue_func=continue_func,
        role_details=role_details,
        tool_details=tool_details,
        budget_config=budget_config,
        sse_hub=sse_hub,
        data_source=data_source,
    )


if __name__ == "__main__":
    main()
