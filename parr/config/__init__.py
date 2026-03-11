"""
Declarative Configuration for the Agentic Framework.

Load roles, models, budgets, and phase limits from YAML/JSON/Markdown files
instead of registering them imperatively in Python code.

Usage:
    from framework.config import load_config, ConfigBundle

    bundle = load_config(
        config_dir="path/to/config",
        tool_registry={"search_documents": my_search_tool},
    )

    # Use with Orchestrator
    orchestrator = Orchestrator(
        llm=my_llm,
        domain_adapter=bundle.domain_adapter,
        cost_config=bundle.cost_config,
        phase_limits=bundle.phase_limits,
        default_budget=bundle.default_budget,
    )

    # Or use the convenience factory
    from framework.config import create_orchestrator_from_config

    orchestrator = create_orchestrator_from_config(
        config_dir="path/to/config",
        tool_registry={"search_documents": my_search_tool},
        llm=my_llm,
    )
"""

from .config_loader import ConfigBundle, load_config, create_orchestrator_from_config
from .config_validator import ConfigError

__all__ = [
    "ConfigBundle",
    "ConfigError",
    "load_config",
    "create_orchestrator_from_config",
]
