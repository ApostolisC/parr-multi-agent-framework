"""
Budget Tracker for the Agentic Framework.

Tracks token usage, cost, and duration at both per-agent and per-workflow
levels. Enforces budget limits before every LLM call.

Critical invariant: Budget is checked BEFORE every LLM call. No exceptions.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .core_types import (
    AgentNode,
    BudgetConfig,
    BudgetUsage,
    CostConfig,
    ModelPricing,
    TokenUsage,
    WorkflowExecution,
    utc_now,
)

logger = logging.getLogger(__name__)


class BudgetExceededException(Exception):
    """Raised when an agent or workflow exceeds its budget."""

    def __init__(
        self,
        message: str,
        agent_id: str = "",
        workflow_id: str = "",
        limit_type: str = "unknown",
    ):
        super().__init__(message)
        self.agent_id = agent_id
        self.workflow_id = workflow_id
        self.limit_type = limit_type  # "tokens", "cost", "duration", or "unknown"
        self.partial_phase_result = None  # Attached by PhaseRunner when available


class BudgetTracker:
    """
    Tracks and enforces budget limits for agents and workflows.

    Usage:
        tracker = BudgetTracker(cost_config)
        tracker.register_workflow(workflow)

        # Before every LLM call:
        tracker.check_budget(node)  # raises BudgetExceededException

        # After every LLM call:
        tracker.record_usage(node, workflow, token_usage, model)
    """

    def __init__(self, cost_config: Optional[CostConfig] = None) -> None:
        self._cost_config = cost_config or CostConfig()

    def check_budget(
        self,
        node: AgentNode,
        workflow: WorkflowExecution,
    ) -> None:
        """
        Check if the agent or workflow has exceeded its budget.

        Must be called BEFORE every LLM call.

        For child agents, also verifies that the parent's budget hasn't
        been exhausted since the child was spawned (dynamic check).

        Raises:
            BudgetExceededException: If any limit is exceeded.
        """
        # Check workflow-level limits
        self._check_limits(
            workflow.budget_consumed,
            workflow.global_budget,
            scope=f"workflow:{workflow.workflow_id}",
            agent_id=node.agent_id,
            workflow_id=workflow.workflow_id,
        )

        # Check parent budget dynamically (guards against concurrent overrun)
        if node.parent_task_id:
            parent = workflow.agent_tree.get(node.parent_task_id)
            if parent:
                self._check_limits(
                    parent.budget_consumed,
                    parent.budget,
                    scope=f"parent:{parent.agent_id}",
                    agent_id=node.agent_id,
                    workflow_id=workflow.workflow_id,
                )

        # Check agent-level limits
        self._check_limits(
            node.budget_consumed,
            node.budget,
            scope=f"agent:{node.agent_id}",
            agent_id=node.agent_id,
            workflow_id=workflow.workflow_id,
        )

    def record_usage(
        self,
        node: AgentNode,
        workflow: WorkflowExecution,
        usage: TokenUsage,
        model: str,
    ) -> float:
        """
        Record token usage after an LLM call.

        Updates both agent-level and workflow-level tracking.
        The passed ``usage`` object is NOT mutated — cost is calculated
        and applied to budget tracking without side-effects on the caller.

        Returns:
            The calculated cost for this LLM call.
        """
        # Use strict cost calculation when a cost budget is configured,
        # so missing pricing doesn't silently bypass cost limits.
        has_cost_budget = bool(
            (node.budget and node.budget.max_cost)
            or (workflow.global_budget and workflow.global_budget.max_cost)
        )
        cost = self._cost_config.calculate_cost(model, usage, strict=has_cost_budget)

        # Update both levels atomically — if either fails, roll back.
        prev_agent_tokens = node.budget_consumed.tokens
        prev_agent_cost = node.budget_consumed.cost
        prev_workflow_tokens = workflow.budget_consumed.tokens
        prev_workflow_cost = workflow.budget_consumed.cost
        try:
            node.budget_consumed.tokens += usage.total_tokens
            node.budget_consumed.cost += cost
            workflow.budget_consumed.tokens += usage.total_tokens
            workflow.budget_consumed.cost += cost
        except Exception:
            # Roll back both levels to keep them in sync
            node.budget_consumed.tokens = prev_agent_tokens
            node.budget_consumed.cost = prev_agent_cost
            workflow.budget_consumed.tokens = prev_workflow_tokens
            workflow.budget_consumed.cost = prev_workflow_cost
            raise

        logger.debug(
            f"Budget: agent={node.agent_id} "
            f"+{usage.total_tokens} tokens (+${cost:.4f}), "
            f"total={node.budget_consumed.tokens} tokens "
            f"(${node.budget_consumed.cost:.4f})"
        )

        return cost

    def check_warning_threshold(
        self,
        node: AgentNode,
        threshold: float = 0.8,
    ) -> Optional[str]:
        """
        Check if budget usage is approaching the limit.

        Returns a warning message if usage exceeds the threshold,
        or None if within safe limits.
        """
        warnings = []

        if node.budget.max_tokens:
            ratio = node.budget_consumed.tokens / node.budget.max_tokens
            if ratio >= threshold:
                warnings.append(
                    f"Token usage at {ratio:.0%} "
                    f"({node.budget_consumed.tokens}/{node.budget.max_tokens})"
                )

        if node.budget.max_cost:
            ratio = node.budget_consumed.cost / node.budget.max_cost
            if ratio >= threshold:
                warnings.append(
                    f"Cost at {ratio:.0%} "
                    f"(${node.budget_consumed.cost:.4f}/${node.budget.max_cost:.2f})"
                )

        if node.budget.max_duration_ms:
            elapsed = node.budget_consumed.elapsed_ms
            ratio = elapsed / node.budget.max_duration_ms
            if ratio >= threshold:
                warnings.append(
                    f"Duration at {ratio:.0%} "
                    f"({elapsed:.0f}ms/{node.budget.max_duration_ms}ms)"
                )

        return "; ".join(warnings) if warnings else None

    def calculate_child_budget(self, parent_node: AgentNode) -> BudgetConfig:
        """
        Calculate budget for a child agent.

        If inherit_remaining is True, the child gets a fraction of the
        parent's remaining budget. Otherwise, uses defaults.
        """
        if not parent_node.budget.inherit_remaining:
            return BudgetConfig(
                max_agent_depth=max(0, parent_node.budget.max_agent_depth - 1),
                max_parallel_agents=parent_node.budget.max_parallel_agents,
                max_sub_agents_total=parent_node.budget.max_sub_agents_total,
            )

        # Reserve a portion of remaining budget for parent recovery
        recovery_pct = parent_node.budget.parent_recovery_budget_pct
        allocatable_fraction = 1.0 - recovery_pct

        remaining_tokens = None
        if parent_node.budget.max_tokens:
            remaining = parent_node.budget.max_tokens - parent_node.budget_consumed.tokens
            allocatable = remaining * allocatable_fraction
            remaining_tokens = max(0, int(allocatable * parent_node.budget.child_budget_fraction))
            if remaining_tokens == 0:
                logger.warning(
                    f"Child budget for agent {parent_node.agent_id} has 0 "
                    f"remaining tokens (parent remaining: {remaining}). "
                    f"Child will immediately exceed budget."
                )

        remaining_cost = None
        if parent_node.budget.max_cost:
            remaining = parent_node.budget.max_cost - parent_node.budget_consumed.cost
            allocatable = remaining * allocatable_fraction
            remaining_cost = max(0.0, allocatable * parent_node.budget.child_budget_fraction)
            if remaining_cost == 0.0:
                logger.warning(
                    f"Child budget for agent {parent_node.agent_id} has $0 "
                    f"remaining cost (parent remaining: ${remaining:.4f}). "
                    f"Child will immediately exceed budget."
                )

        return BudgetConfig(
            max_tokens=remaining_tokens,
            max_cost=remaining_cost,
            max_duration_ms=parent_node.budget.max_duration_ms,
            max_agent_depth=max(0, parent_node.budget.max_agent_depth - 1),
            max_parallel_agents=parent_node.budget.max_parallel_agents,
            max_sub_agents_total=parent_node.budget.max_sub_agents_total,
            inherit_remaining=True,
        )

    def _check_limits(
        self,
        consumed: BudgetUsage,
        budget: BudgetConfig,
        scope: str,
        agent_id: str,
        workflow_id: str,
    ) -> None:
        """Check specific budget limits and raise if exceeded."""
        if budget.max_tokens and consumed.tokens >= budget.max_tokens:
            raise BudgetExceededException(
                f"Token limit exceeded for {scope}: "
                f"{consumed.tokens} >= {budget.max_tokens}",
                agent_id=agent_id,
                workflow_id=workflow_id,
                limit_type="tokens",
            )

        if budget.max_cost and consumed.cost >= budget.max_cost:
            raise BudgetExceededException(
                f"Cost limit exceeded for {scope}: "
                f"${consumed.cost:.4f} >= ${budget.max_cost:.2f}",
                agent_id=agent_id,
                workflow_id=workflow_id,
                limit_type="cost",
            )

        if budget.max_duration_ms:
            elapsed = consumed.elapsed_ms
            if elapsed >= budget.max_duration_ms:
                raise BudgetExceededException(
                    f"Duration limit exceeded for {scope}: "
                    f"{elapsed:.0f}ms >= {budget.max_duration_ms}ms",
                    agent_id=agent_id,
                    workflow_id=workflow_id,
                    limit_type="duration",
                )
