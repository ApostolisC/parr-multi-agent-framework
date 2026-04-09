"""Tests for pluggable budget policy (ChildBudgetAllocator and BudgetTracker overrides)."""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from parr.budget_tracker import (
    BudgetExceededException,
    BudgetTracker,
    ChildBudgetAllocator,
    EqualShareAllocator,
    FixedAllocator,
    FractionAllocator,
)
from parr.core_types import (
    AgentConfig,
    AgentNode,
    BudgetConfig,
    BudgetUsage,
    TokenUsage,
    WorkflowExecution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(
    budget: BudgetConfig | None = None,
    consumed_tokens: int = 0,
    consumed_cost: float = 0.0,
    children: list[str] | None = None,
    agent_id: str = "agent-1",
    task_id: str = "task-1",
) -> AgentNode:
    node = AgentNode(
        task_id=task_id,
        agent_id=agent_id,
        config=AgentConfig(agent_id=agent_id, model="test-model"),
        budget=budget or BudgetConfig(),
        budget_consumed=BudgetUsage(tokens=consumed_tokens, cost=consumed_cost),
    )
    if children:
        node.children = children
    return node


def _make_workflow(
    budget: BudgetConfig | None = None,
    consumed_tokens: int = 0,
    consumed_cost: float = 0.0,
) -> WorkflowExecution:
    wf = WorkflowExecution(global_budget=budget or BudgetConfig())
    wf.budget_consumed.tokens = consumed_tokens
    wf.budget_consumed.cost = consumed_cost
    return wf


# =========================================================================
# 1. ChildBudgetAllocator base class
# =========================================================================

class TestChildBudgetAllocatorBase:

    def test_base_raises_not_implemented(self):
        allocator = ChildBudgetAllocator()
        parent = _make_node(budget=BudgetConfig(max_tokens=1000))
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            allocator.allocate(parent)


# =========================================================================
# 2. FractionAllocator (default — preserves existing behavior)
# =========================================================================

class TestFractionAllocator:

    def test_inherits_fraction_of_remaining_tokens(self):
        allocator = FractionAllocator()
        parent = _make_node(
            budget=BudgetConfig(max_tokens=1000, child_budget_fraction=0.5),
            consumed_tokens=200,
        )
        child_budget = allocator.allocate(parent)
        # remaining = 800, allocatable = 800 * 0.9 = 720, child = 720 * 0.5 = 360
        assert child_budget.max_tokens == 360

    def test_inherits_fraction_of_remaining_cost(self):
        allocator = FractionAllocator()
        parent = _make_node(
            budget=BudgetConfig(max_cost=10.0, child_budget_fraction=0.5),
            consumed_cost=4.0,
        )
        child_budget = allocator.allocate(parent)
        # remaining = 6.0, allocatable = 6.0 * 0.9 = 5.4, child = 5.4 * 0.5 = 2.7
        assert child_budget.max_cost == pytest.approx(2.7)

    def test_inherits_duration_unchanged(self):
        allocator = FractionAllocator()
        parent = _make_node(budget=BudgetConfig(max_duration_ms=60_000))
        child_budget = allocator.allocate(parent)
        assert child_budget.max_duration_ms == 60_000

    def test_decrements_agent_depth(self):
        allocator = FractionAllocator()
        parent = _make_node(budget=BudgetConfig(max_agent_depth=3))
        child_budget = allocator.allocate(parent)
        assert child_budget.max_agent_depth == 2

    def test_zero_remaining_yields_zero(self):
        allocator = FractionAllocator()
        parent = _make_node(
            budget=BudgetConfig(max_tokens=500, child_budget_fraction=0.5),
            consumed_tokens=500,
        )
        child_budget = allocator.allocate(parent)
        assert child_budget.max_tokens == 0

    def test_over_consumed_yields_zero(self):
        allocator = FractionAllocator()
        parent = _make_node(
            budget=BudgetConfig(max_tokens=100, child_budget_fraction=0.5),
            consumed_tokens=200,
        )
        child_budget = allocator.allocate(parent)
        assert child_budget.max_tokens == 0

    def test_inherit_remaining_false_returns_defaults(self):
        allocator = FractionAllocator()
        parent = _make_node(
            budget=BudgetConfig(
                max_tokens=1000, max_cost=10.0,
                inherit_remaining=False, max_agent_depth=3,
            ),
        )
        child_budget = allocator.allocate(parent)
        assert child_budget.max_tokens is None
        assert child_budget.max_cost is None
        assert child_budget.max_agent_depth == 2

    def test_custom_fraction(self):
        allocator = FractionAllocator()
        parent = _make_node(
            budget=BudgetConfig(max_tokens=1000, child_budget_fraction=0.25),
            consumed_tokens=0,
        )
        child_budget = allocator.allocate(parent)
        # remaining = 1000, allocatable = 1000 * 0.9 = 900, child = 900 * 0.25 = 225
        assert child_budget.max_tokens == 225


# =========================================================================
# 3. EqualShareAllocator
# =========================================================================

class TestEqualShareAllocator:

    def test_splits_equally_among_expected(self):
        allocator = EqualShareAllocator(expected_children=4, recovery_pct=0.0)
        parent = _make_node(
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=0,
        )
        child_budget = allocator.allocate(parent)
        # 1000 / 4 = 250
        assert child_budget.max_tokens == 250

    def test_accounts_for_already_spawned(self):
        allocator = EqualShareAllocator(expected_children=4, recovery_pct=0.0)
        parent = _make_node(
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=200,
            children=["c1", "c2"],
        )
        child_budget = allocator.allocate(parent)
        # remaining = 800, remaining_slots = 4 - 2 = 2, per child = 400
        assert child_budget.max_tokens == 400

    def test_recovery_reserve(self):
        allocator = EqualShareAllocator(expected_children=2, recovery_pct=0.20)
        parent = _make_node(
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=0,
        )
        child_budget = allocator.allocate(parent)
        # allocatable = 1000 * 0.8 = 800, per child = 800 / 2 = 400
        assert child_budget.max_tokens == 400

    def test_defaults_to_max_sub_agents_total(self):
        allocator = EqualShareAllocator(recovery_pct=0.0)
        parent = _make_node(
            budget=BudgetConfig(max_tokens=900, max_sub_agents_total=3),
            consumed_tokens=0,
        )
        child_budget = allocator.allocate(parent)
        # 900 / 3 = 300
        assert child_budget.max_tokens == 300

    def test_cost_split(self):
        allocator = EqualShareAllocator(expected_children=2, recovery_pct=0.10)
        parent = _make_node(
            budget=BudgetConfig(max_cost=10.0),
            consumed_cost=2.0,
        )
        child_budget = allocator.allocate(parent)
        # remaining = 8.0, allocatable = 8.0 * 0.9 = 7.2, per child = 3.6
        assert child_budget.max_cost == pytest.approx(3.6)

    def test_inherit_remaining_false(self):
        allocator = EqualShareAllocator(expected_children=2)
        parent = _make_node(
            budget=BudgetConfig(max_tokens=1000, inherit_remaining=False),
        )
        child_budget = allocator.allocate(parent)
        assert child_budget.max_tokens is None

    def test_all_children_spawned_gets_minimum_one_slot(self):
        """Even if all expected children are spawned, at least 1 slot is used."""
        allocator = EqualShareAllocator(expected_children=2, recovery_pct=0.0)
        parent = _make_node(
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=0,
            children=["c1", "c2"],
        )
        child_budget = allocator.allocate(parent)
        # remaining_slots = max(1, 2 - 2) = 1, so gets full remaining
        assert child_budget.max_tokens == 1000

    def test_decrements_depth(self):
        allocator = EqualShareAllocator(expected_children=2)
        parent = _make_node(budget=BudgetConfig(max_agent_depth=5))
        child_budget = allocator.allocate(parent)
        assert child_budget.max_agent_depth == 4


# =========================================================================
# 4. FixedAllocator
# =========================================================================

class TestFixedAllocator:

    def test_fixed_tokens(self):
        allocator = FixedAllocator(max_tokens=5000)
        parent = _make_node(
            budget=BudgetConfig(max_tokens=100_000),
            consumed_tokens=90_000,
        )
        child_budget = allocator.allocate(parent)
        assert child_budget.max_tokens == 5000

    def test_fixed_cost(self):
        allocator = FixedAllocator(max_cost=1.50)
        parent = _make_node(budget=BudgetConfig(max_cost=100.0))
        child_budget = allocator.allocate(parent)
        assert child_budget.max_cost == 1.50

    def test_fixed_duration(self):
        allocator = FixedAllocator(max_duration_ms=30_000)
        parent = _make_node(budget=BudgetConfig())
        child_budget = allocator.allocate(parent)
        assert child_budget.max_duration_ms == 30_000

    def test_none_means_no_limit(self):
        allocator = FixedAllocator()  # all None
        parent = _make_node(budget=BudgetConfig())
        child_budget = allocator.allocate(parent)
        assert child_budget.max_tokens is None
        assert child_budget.max_cost is None
        assert child_budget.max_duration_ms is None

    def test_inherit_remaining_is_false(self):
        allocator = FixedAllocator(max_tokens=5000)
        parent = _make_node(budget=BudgetConfig())
        child_budget = allocator.allocate(parent)
        assert child_budget.inherit_remaining is False

    def test_decrements_depth(self):
        allocator = FixedAllocator(max_tokens=5000)
        parent = _make_node(budget=BudgetConfig(max_agent_depth=4))
        child_budget = allocator.allocate(parent)
        assert child_budget.max_agent_depth == 3

    def test_preserves_parallel_and_total(self):
        allocator = FixedAllocator(max_tokens=5000)
        parent = _make_node(
            budget=BudgetConfig(max_parallel_agents=5, max_sub_agents_total=7),
        )
        child_budget = allocator.allocate(parent)
        assert child_budget.max_parallel_agents == 5
        assert child_budget.max_sub_agents_total == 7


# =========================================================================
# 5. BudgetTracker with custom allocator
# =========================================================================

class TestBudgetTrackerWithAllocator:

    def test_default_uses_fraction_allocator(self):
        tracker = BudgetTracker()
        assert isinstance(tracker._child_allocator, FractionAllocator)

    def test_custom_allocator_used(self):
        allocator = FixedAllocator(max_tokens=5000)
        tracker = BudgetTracker(child_allocator=allocator)
        parent = _make_node(
            budget=BudgetConfig(max_tokens=100_000),
            consumed_tokens=90_000,
        )
        child_budget = tracker.calculate_child_budget(parent)
        assert child_budget.max_tokens == 5000

    def test_equal_share_allocator(self):
        allocator = EqualShareAllocator(expected_children=4, recovery_pct=0.0)
        tracker = BudgetTracker(child_allocator=allocator)
        parent = _make_node(
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=0,
        )
        child_budget = tracker.calculate_child_budget(parent)
        assert child_budget.max_tokens == 250

    def test_backward_compat_no_allocator(self):
        """BudgetTracker without allocator param uses FractionAllocator (same as before)."""
        tracker = BudgetTracker()
        parent = _make_node(
            budget=BudgetConfig(max_tokens=1000, child_budget_fraction=0.5),
            consumed_tokens=200,
        )
        child_budget = tracker.calculate_child_budget(parent)
        # Same as old behavior: remaining=800, allocatable=720, child=360
        assert child_budget.max_tokens == 360


# =========================================================================
# 6. BudgetTracker.check_limits is public and overridable
# =========================================================================

class TestCheckLimitsPublic:

    def test_check_limits_raises_on_token_exceeded(self):
        tracker = BudgetTracker()
        consumed = BudgetUsage(tokens=1000)
        budget = BudgetConfig(max_tokens=1000)
        with pytest.raises(BudgetExceededException, match="Token limit"):
            tracker.check_limits(
                consumed, budget,
                scope="test", agent_id="a", workflow_id="w",
            )

    def test_check_limits_raises_on_cost_exceeded(self):
        tracker = BudgetTracker()
        consumed = BudgetUsage(cost=5.0)
        budget = BudgetConfig(max_cost=5.0)
        with pytest.raises(BudgetExceededException, match="Cost limit"):
            tracker.check_limits(
                consumed, budget,
                scope="test", agent_id="a", workflow_id="w",
            )

    def test_check_limits_passes_when_under(self):
        tracker = BudgetTracker()
        consumed = BudgetUsage(tokens=500)
        budget = BudgetConfig(max_tokens=1000)
        tracker.check_limits(
            consumed, budget,
            scope="test", agent_id="a", workflow_id="w",
        )


# =========================================================================
# 7. BudgetTracker subclassing
# =========================================================================

class TestBudgetTrackerSubclass:

    def test_custom_check_budget(self):
        """Subclass can add custom checks to check_budget."""
        class RateLimitedTracker(BudgetTracker):
            def __init__(self, *args, max_calls=10, **kwargs):
                super().__init__(*args, **kwargs)
                self._call_count = 0
                self._max_calls = max_calls

            def check_budget(self, node, workflow):
                self._call_count += 1
                if self._call_count > self._max_calls:
                    raise BudgetExceededException(
                        "Rate limit exceeded",
                        agent_id=node.agent_id,
                        workflow_id=workflow.workflow_id,
                        limit_type="rate_limit",
                    )
                super().check_budget(node, workflow)

        tracker = RateLimitedTracker(max_calls=2)
        node = _make_node(budget=BudgetConfig(max_tokens=100_000))
        wf = _make_workflow()

        tracker.check_budget(node, wf)  # call 1 — ok
        tracker.check_budget(node, wf)  # call 2 — ok
        with pytest.raises(BudgetExceededException, match="Rate limit"):
            tracker.check_budget(node, wf)  # call 3 — exceeds

    def test_custom_check_limits(self):
        """Subclass can override check_limits to add custom resource checks."""
        class GpuTracker(BudgetTracker):
            def check_limits(self, consumed, budget, scope, agent_id, workflow_id):
                super().check_limits(consumed, budget, scope, agent_id, workflow_id)
                # Custom GPU check
                if hasattr(budget, '_gpu_limit') and consumed.tokens > budget._gpu_limit:
                    raise BudgetExceededException(
                        "GPU quota exceeded",
                        agent_id=agent_id,
                        workflow_id=workflow_id,
                        limit_type="gpu",
                    )

        tracker = GpuTracker()
        consumed = BudgetUsage(tokens=500)
        budget = BudgetConfig(max_tokens=1000)
        # Standard check passes
        tracker.check_limits(consumed, budget, "test", "a", "w")

    def test_custom_record_usage(self):
        """Subclass can extend record_usage with side effects."""
        recorded = []

        class LoggingTracker(BudgetTracker):
            def record_usage(self, node, workflow, usage, model):
                cost = super().record_usage(node, workflow, usage, model)
                recorded.append({
                    "agent_id": node.agent_id,
                    "tokens": usage.total_tokens,
                    "cost": cost,
                })
                return cost

        tracker = LoggingTracker()
        node = _make_node(budget=BudgetConfig())
        wf = _make_workflow()
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        tracker.record_usage(node, wf, usage, "test-model")

        assert len(recorded) == 1
        assert recorded[0]["agent_id"] == "agent-1"
        assert recorded[0]["tokens"] == 150

    def test_custom_warning_threshold(self):
        """Subclass can override check_warning_threshold."""
        class StrictTracker(BudgetTracker):
            def check_warning_threshold(self, node, threshold=0.5):
                # Lower threshold — warn earlier
                return super().check_warning_threshold(node, threshold=threshold)

        tracker = StrictTracker()
        node = _make_node(
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=600,
        )
        # Default 0.8 wouldn't trigger, but StrictTracker uses 0.5
        result = tracker.check_warning_threshold(node)
        assert result is not None
        assert "Token usage" in result


# =========================================================================
# 8. Orchestrator accepts child_allocator
# =========================================================================

class TestOrchestratorChildAllocator:

    def test_orchestrator_passes_allocator_to_tracker(self):
        from parr.orchestrator import Orchestrator

        llm = MagicMock()
        allocator = FixedAllocator(max_tokens=5000)
        orch = Orchestrator(llm=llm, child_allocator=allocator)
        assert isinstance(orch._budget_tracker._child_allocator, FixedAllocator)

    def test_orchestrator_default_no_allocator(self):
        from parr.orchestrator import Orchestrator

        llm = MagicMock()
        orch = Orchestrator(llm=llm)
        assert isinstance(orch._budget_tracker._child_allocator, FractionAllocator)


# =========================================================================
# 9. Custom allocator (user-defined)
# =========================================================================

class TestCustomAllocator:

    def test_priority_allocator(self):
        """User-defined allocator works with BudgetTracker."""
        class PriorityAllocator(ChildBudgetAllocator):
            """High-priority children get 80% of remaining budget."""
            def allocate(self, parent_node):
                if not parent_node.budget.max_tokens:
                    return BudgetConfig(
                        max_agent_depth=max(0, parent_node.budget.max_agent_depth - 1),
                    )
                remaining = parent_node.budget.max_tokens - parent_node.budget_consumed.tokens
                return BudgetConfig(
                    max_tokens=max(0, int(remaining * 0.8)),
                    max_agent_depth=max(0, parent_node.budget.max_agent_depth - 1),
                    inherit_remaining=True,
                )

        tracker = BudgetTracker(child_allocator=PriorityAllocator())
        parent = _make_node(
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=200,
        )
        child_budget = tracker.calculate_child_budget(parent)
        # remaining = 800, child gets 80% = 640
        assert child_budget.max_tokens == 640


# =========================================================================
# 10. Edge cases
# =========================================================================

class TestBudgetPolicyEdgeCases:

    def test_fraction_allocator_no_token_limit(self):
        allocator = FractionAllocator()
        parent = _make_node(budget=BudgetConfig(max_tokens=None))
        child_budget = allocator.allocate(parent)
        assert child_budget.max_tokens is None

    def test_fraction_allocator_no_cost_limit(self):
        allocator = FractionAllocator()
        parent = _make_node(budget=BudgetConfig(max_cost=None))
        child_budget = allocator.allocate(parent)
        assert child_budget.max_cost is None

    def test_equal_share_with_zero_remaining(self):
        allocator = EqualShareAllocator(expected_children=3, recovery_pct=0.0)
        parent = _make_node(
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=1000,
        )
        child_budget = allocator.allocate(parent)
        assert child_budget.max_tokens == 0

    def test_fixed_allocator_ignores_parent_state(self):
        allocator = FixedAllocator(max_tokens=5000, max_cost=2.0)
        parent = _make_node(
            budget=BudgetConfig(max_tokens=100),
            consumed_tokens=100,
            consumed_cost=100.0,
        )
        child_budget = allocator.allocate(parent)
        assert child_budget.max_tokens == 5000
        assert child_budget.max_cost == 2.0

    def test_budget_tracker_check_budget_still_works(self):
        """Ensure check_budget works correctly after refactoring."""
        tracker = BudgetTracker()
        node = _make_node(
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=1000,
        )
        wf = _make_workflow()
        with pytest.raises(BudgetExceededException, match="Token limit"):
            tracker.check_budget(node, wf)

    def test_budget_tracker_record_usage_still_works(self):
        """Ensure record_usage works correctly after refactoring."""
        tracker = BudgetTracker()
        node = _make_node(budget=BudgetConfig())
        wf = _make_workflow()
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        tracker.record_usage(node, wf, usage, "test-model")
        assert node.budget_consumed.tokens == 150
        assert wf.budget_consumed.tokens == 150
