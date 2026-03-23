"""
Reference DomainAdapter for the Agentic Framework.

A generic, in-memory domain adapter that applications populate at startup
with their roles, tools, schemas, and templates. This is the standard way
to connect any domain to the framework — not domain-specific itself.

Usage:
    from framework.adapters import ReferenceDomainAdapter
    from framework import AgentConfig, ToolDef, Phase

    adapter = ReferenceDomainAdapter()

    adapter.register_role(
        role="analyst",
        config=AgentConfig(
            role="analyst",
            system_prompt="You are a data analyst...",
            model="gpt-4o",
        ),
        tools=[my_search_tool, my_save_tool],
        output_schema={"type": "object", "properties": {...}},
        report_template="## Analysis Report\\n...",
        description="Analyzes data and produces structured reports",
    )

    adapter.register_sub_role(
        role="analyst",
        sub_role="risk_assessment",
        description="Specializes in risk analysis",
        config_overrides=AgentConfig(
            role="analyst",
            sub_role="risk_assessment",
            system_prompt="You are a risk analyst...",
            model="gpt-4o",
        ),
    )

    # Wire into orchestrator
    orchestrator = Orchestrator(llm=llm, domain_adapter=adapter)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..core_types import AgentConfig, AgentOutput, ToolDef

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class SubRoleEntry:
    """Override configuration for a sub-role."""
    description: str = ""
    config_overrides: Optional[AgentConfig] = None
    tools_override: Optional[List[ToolDef]] = None
    output_schema_override: Optional[Dict[str, Any]] = None
    report_template_override: Optional[str] = None
    direct_answer_schema_policy_override: Optional[str] = None


@dataclass
class RoleEntry:
    """Complete registration for a role."""
    config: AgentConfig
    tools: List[ToolDef] = field(default_factory=list)
    output_schema: Optional[Dict[str, Any]] = None
    report_template: Optional[str] = None
    description: str = ""
    sub_roles: Dict[str, SubRoleEntry] = field(default_factory=dict)
    direct_answer_schema_policy: Optional[str] = None


# ---------------------------------------------------------------------------
# ReferenceDomainAdapter
# ---------------------------------------------------------------------------

class ReferenceDomainAdapter:
    """
    Generic DomainAdapter that applications configure at startup.

    Roles and sub-roles are registered via ``register_role()`` and
    ``register_sub_role()``. The framework calls the protocol methods
    during workflow execution to resolve agent configurations.
    """

    def __init__(self) -> None:
        self._roles: Dict[str, RoleEntry] = {}
        self._output_handler: Optional[Callable[[str, AgentOutput], None]] = None

    # -- registration API ----------------------------------------------------

    def register_role(
        self,
        role: str,
        config: AgentConfig,
        tools: Optional[List[ToolDef]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        report_template: Optional[str] = None,
        description: str = "",
        direct_answer_schema_policy: Optional[str] = None,
    ) -> None:
        """
        Register a role with the adapter.

        Args:
            role: Unique role identifier.
            config: AgentConfig with system_prompt, model, model_config.
            tools: Domain-specific tools for this role.
            output_schema: JSON Schema for the role's report output.
            report_template: Formatting instructions for the report phase.
            description: Human-readable description (shown in spawn_agent tool).
            direct_answer_schema_policy: ``"enforce"`` or ``"bypass"`` for
                direct-answer schema handling; ``None`` uses global config.
        """
        if role in self._roles:
            raise ValueError(f"Role '{role}' is already registered")

        self._roles[role] = RoleEntry(
            config=config,
            tools=tools or [],
            output_schema=output_schema,
            report_template=report_template,
            description=description,
            direct_answer_schema_policy=direct_answer_schema_policy,
        )
        logger.debug(f"Registered role: {role}")

    def register_sub_role(
        self,
        role: str,
        sub_role: str,
        description: str = "",
        config_overrides: Optional[AgentConfig] = None,
        tools_override: Optional[List[ToolDef]] = None,
        output_schema_override: Optional[Dict[str, Any]] = None,
        report_template_override: Optional[str] = None,
        direct_answer_schema_policy_override: Optional[str] = None,
    ) -> None:
        """
        Register a sub-role under an existing role.

        Sub-role fields override the parent role's fields when present.
        ``None`` means inherit from the parent role.

        Args:
            role: Parent role identifier (must already be registered).
            sub_role: Sub-role identifier.
            description: Human-readable description.
            config_overrides: If set, replaces the parent's AgentConfig entirely.
            tools_override: If set, replaces the parent's tool list entirely.
            output_schema_override: If set, replaces the parent's output schema.
            report_template_override: If set, replaces the parent's report template.
            direct_answer_schema_policy_override: If set, overrides the parent's
                direct-answer schema policy.
        """
        entry = self._roles.get(role)
        if entry is None:
            raise ValueError(f"Role '{role}' not registered. Register it first.")
        if sub_role in entry.sub_roles:
            raise ValueError(f"Sub-role '{sub_role}' already registered under '{role}'")

        entry.sub_roles[sub_role] = SubRoleEntry(
            description=description,
            config_overrides=config_overrides,
            tools_override=tools_override,
            output_schema_override=output_schema_override,
            report_template_override=report_template_override,
            direct_answer_schema_policy_override=direct_answer_schema_policy_override,
        )
        logger.debug(f"Registered sub-role: {role}/{sub_role}")

    def set_output_handler(
        self, handler: Callable[[str, AgentOutput], None]
    ) -> None:
        """
        Set a callback for persisting workflow output.

        The handler receives ``(workflow_id, agent_output)`` when the
        orchestrator calls ``persist_output()``.

        Args:
            handler: Callable that persists the output.
        """
        self._output_handler = handler

    # -- DomainAdapter protocol methods --------------------------------------

    def get_role_config(
        self, role: str, sub_role: Optional[str] = None
    ) -> AgentConfig:
        """Get agent configuration for a role, with sub-role overrides."""
        entry = self._roles.get(role)
        if entry is None:
            raise ValueError(f"Unknown role: '{role}'")

        if sub_role and sub_role in entry.sub_roles:
            sr = entry.sub_roles[sub_role]
            if sr.config_overrides is not None:
                return sr.config_overrides
        return entry.config

    def get_domain_tools(
        self, role: str, sub_role: Optional[str] = None
    ) -> List[ToolDef]:
        """Get domain-specific tools for a role."""
        entry = self._roles.get(role)
        if entry is None:
            return []

        if sub_role and sub_role in entry.sub_roles:
            sr = entry.sub_roles[sub_role]
            if sr.tools_override is not None:
                return sr.tools_override
        return entry.tools

    def get_output_schema(
        self, role: str, sub_role: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the expected output JSON schema for a role."""
        entry = self._roles.get(role)
        if entry is None:
            return None

        if sub_role and sub_role in entry.sub_roles:
            sr = entry.sub_roles[sub_role]
            if sr.output_schema_override is not None:
                return sr.output_schema_override
        return entry.output_schema

    def get_report_template(
        self, role: str, sub_role: Optional[str] = None
    ) -> Optional[str]:
        """Get report formatting instructions for a role."""
        entry = self._roles.get(role)
        if entry is None:
            return None

        if sub_role and sub_role in entry.sub_roles:
            sr = entry.sub_roles[sub_role]
            if sr.report_template_override is not None:
                return sr.report_template_override
        return entry.report_template

    def get_direct_answer_schema_policy(
        self, role: str, sub_role: Optional[str] = None
    ) -> Optional[str]:
        """Get the direct-answer schema policy for a role.

        Returns ``"enforce"``, ``"bypass"``, or ``None`` (use global config).
        Sub-role override takes precedence when present.
        """
        entry = self._roles.get(role)
        if entry is None:
            return None

        if sub_role and sub_role in entry.sub_roles:
            sr = entry.sub_roles[sub_role]
            if sr.direct_answer_schema_policy_override is not None:
                return sr.direct_answer_schema_policy_override
        return entry.direct_answer_schema_policy

    def list_available_roles(self) -> List[Dict[str, Any]]:
        """List all registered roles for the spawn_agent tool description."""
        result: List[Dict[str, Any]] = []
        for role_name, entry in self._roles.items():
            role_info: Dict[str, Any] = {
                "role": role_name,
                "description": entry.description,
                "sub_roles": [],
            }
            for sr_name, sr_entry in entry.sub_roles.items():
                role_info["sub_roles"].append({
                    "name": sr_name,
                    "description": sr_entry.description,
                })
            result.append(role_info)
        return result

    def persist_output(
        self, workflow_id: str, agent_output: AgentOutput
    ) -> None:
        """Persist workflow output via the registered handler."""
        if self._output_handler:
            try:
                self._output_handler(workflow_id, agent_output)
            except Exception as e:
                logger.error(f"Output handler failed for workflow {workflow_id}: {e}")
        else:
            logger.info(
                f"Workflow {workflow_id} completed "
                f"(role={agent_output.role}, status={agent_output.status}). "
                f"No output handler registered."
            )

    # -- introspection -------------------------------------------------------

    def has_role(self, role: str) -> bool:
        """Check if a role is registered."""
        return role in self._roles

    @property
    def role_names(self) -> List[str]:
        """Get all registered role names."""
        return list(self._roles.keys())
