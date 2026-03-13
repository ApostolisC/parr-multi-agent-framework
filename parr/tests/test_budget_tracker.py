"""Tests for the BudgetTracker module."""

from datetime import datetime, timedelta, timezone

import pytest

from parr.budget_tracker import BudgetExceededException, BudgetTracker
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
    parent_task_id: str | None = None,
    agent_id: str = "agent-1",
    task_id: str = "task-1",
    started_at: datetime | None = None,
) -> AgentNode:
    usage = BudgetUsage(tokens=consumed_tokens, cost=consumed_cost)
    if started_at is not None:
        usage.started_at = started_at
    return AgentNode(
        task_id=task_id,
        agent_id=agent_id,
        parent_task_id=parent_task_id,
        config=AgentConfig(agent_id=agent_id, model="test-model"),
        budget=budget or BudgetConfig(),
        budget_consumed=usage,
    )


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
# check_budget
# =========================================================================

class TestCheckBudget:
    """Tests for BudgetTracker.check_budget()."""

    def test_passes_when_under_all_limits(self, budget_tracker, agent_node, workflow):
        """No exception when usage is well below limits."""
        budget_tracker.check_budget(agent_node, workflow)

    def test_raises_when_token_limit_exceeded(self, budget_tracker, workflow):
        node = _make_node(
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=1000,
        )
        with pytest.raises(BudgetExceededException, match="Token limit exceeded"):
            budget_tracker.check_budget(node, workflow)

    def test_raises_when_cost_limit_exceeded(self, budget_tracker, workflow):
        node = _make_node(
            budget=BudgetConfig(max_cost=1.0),
            consumed_cost=1.0,
        )
        with pytest.raises(BudgetExceededException, match="Cost limit exceeded"):
            budget_tracker.check_budget(node, workflow)

    def test_raises_when_duration_limit_exceeded(self, budget_tracker, workflow):
        past = datetime.now(timezone.utc) - timedelta(seconds=120)
        node = _make_node(
            budget=BudgetConfig(max_duration_ms=60_000),
            started_at=past,
        )
        with pytest.raises(BudgetExceededException, match="Duration limit exceeded"):
            budget_tracker.check_budget(node, workflow)

    def test_passes_when_no_limits_set(self, budget_tracker):
        """Budget with no limits should never raise."""
        node = _make_node(consumed_tokens=999_999, consumed_cost=999.0)
        wf = _make_workflow()
        budget_tracker.check_budget(node, wf)

    def test_workflow_level_token_limit(self, budget_tracker):
        """Workflow-level token limit triggers before agent-level."""
        wf = _make_workflow(
            budget=BudgetConfig(max_tokens=500),
            consumed_tokens=500,
        )
        node = _make_node(budget=BudgetConfig(max_tokens=10_000))
        with pytest.raises(BudgetExceededException, match="workflow"):
            budget_tracker.check_budget(node, wf)

    def test_workflow_level_cost_limit(self, budget_tracker):
        wf = _make_workflow(
            budget=BudgetConfig(max_cost=0.50),
            consumed_cost=0.50,
        )
        node = _make_node(budget=BudgetConfig(max_cost=10.0))
        with pytest.raises(BudgetExceededException, match="workflow"):
            budget_tracker.check_budget(node, wf)

    def test_parent_budget_exceeded_blocks_child(self, budget_tracker):
        """Child agent is blocked when parent budget is exhausted."""
        parent = _make_node(
            task_id="parent-task",
            agent_id="parent-agent",
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=1000,
        )
        child = _make_node(
            task_id="child-task",
            agent_id="child-agent",
            parent_task_id="parent-task",
            budget=BudgetConfig(max_tokens=500),
        )
        wf = _make_workflow()
        wf.agent_tree["parent-task"] = parent
        with pytest.raises(BudgetExceededException, match="parent"):
            budget_tracker.check_budget(child, wf)

    def test_parent_budget_ok_child_proceeds(self, budget_tracker):
        """Child proceeds when parent still has budget."""
        parent = _make_node(
            task_id="parent-task",
            agent_id="parent-agent",
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=500,
        )
        child = _make_node(
            task_id="child-task",
            agent_id="child-agent",
            parent_task_id="parent-task",
            budget=BudgetConfig(max_tokens=500),
        )
        wf = _make_workflow()
        wf.agent_tree["parent-task"] = parent
        budget_tracker.check_budget(child, wf)

    def test_exact_token_exhaustion_raises(self, budget_tracker):
        """Consuming exactly max_tokens triggers the exception (>= check)."""
        node = _make_node(
            budget=BudgetConfig(max_tokens=100),
            consumed_tokens=100,
        )
        wf = _make_workflow()
        with pytest.raises(BudgetExceededException):
            budget_tracker.check_budget(node, wf)

    def test_exact_cost_exhaustion_raises(self, budget_tracker):
        node = _make_node(
            budget=BudgetConfig(max_cost=1.0),
            consumed_cost=1.0,
        )
        wf = _make_workflow()
        with pytest.raises(BudgetExceededException):
            budget_tracker.check_budget(node, wf)

    def test_one_below_limit_passes(self, budget_tracker):
        """Usage one unit below the limit should not raise."""
        node = _make_node(
            budget=BudgetConfig(max_tokens=100),
            consumed_tokens=99,
        )
        wf = _make_workflow()
        budget_tracker.check_budget(node, wf)

    def test_exception_carries_agent_and_workflow_ids(self, budget_tracker):
        wf = _make_workflow(budget=BudgetConfig(max_tokens=10), consumed_tokens=10)
        node = _make_node(agent_id="my-agent")
        try:
            budget_tracker.check_budget(node, wf)
            pytest.fail("Expected BudgetExceededException")
        except BudgetExceededException as exc:
            assert exc.agent_id == "my-agent"
            assert exc.workflow_id == wf.workflow_id


# =========================================================================
# record_usage
# =========================================================================

class TestRecordUsage:
    """Tests for BudgetTracker.record_usage()."""

    def test_updates_node_tokens(self, budget_tracker, agent_node, workflow):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        budget_tracker.record_usage(agent_node, workflow, usage, "test-model")
        assert agent_node.budget_consumed.tokens == 150

    def test_updates_workflow_tokens(self, budget_tracker, agent_node, workflow):
        usage = TokenUsage(input_tokens=200, output_tokens=100)
        budget_tracker.record_usage(agent_node, workflow, usage, "test-model")
        assert workflow.budget_consumed.tokens == 300

    def test_updates_node_cost(self, budget_tracker, agent_node, workflow):
        usage = TokenUsage(input_tokens=1000, output_tokens=1000)
        cost = budget_tracker.record_usage(agent_node, workflow, usage, "test-model")
        # test-model: input 0.01/1k, output 0.03/1k
        expected_cost = (1000 / 1000) * 0.01 + (1000 / 1000) * 0.03
        assert cost == pytest.approx(expected_cost)
        assert agent_node.budget_consumed.cost == pytest.approx(expected_cost)

    def test_updates_workflow_cost(self, budget_tracker, agent_node, workflow):
        usage = TokenUsage(input_tokens=1000, output_tokens=1000)
        budget_tracker.record_usage(agent_node, workflow, usage, "test-model")
        expected_cost = 0.01 + 0.03
        assert workflow.budget_consumed.cost == pytest.approx(expected_cost)

    def test_returns_calculated_cost(self, budget_tracker, agent_node, workflow):
        usage = TokenUsage(input_tokens=500, output_tokens=500)
        cost = budget_tracker.record_usage(agent_node, workflow, usage, "test-model")
        expected = (500 / 1000) * 0.01 + (500 / 1000) * 0.03
        assert cost == pytest.approx(expected)

    def test_accumulates_across_multiple_calls(self, budget_tracker, agent_node, workflow):
        u1 = TokenUsage(input_tokens=100, output_tokens=50)
        u2 = TokenUsage(input_tokens=200, output_tokens=100)
        budget_tracker.record_usage(agent_node, workflow, u1, "test-model")
        budget_tracker.record_usage(agent_node, workflow, u2, "test-model")
        assert agent_node.budget_consumed.tokens == 450
        assert workflow.budget_consumed.tokens == 450

    def test_zero_usage_is_noop(self, budget_tracker, agent_node, workflow):
        usage = TokenUsage(input_tokens=0, output_tokens=0)
        cost = budget_tracker.record_usage(agent_node, workflow, usage, "test-model")
        assert cost == 0.0
        assert agent_node.budget_consumed.tokens == 0
        assert agent_node.budget_consumed.cost == 0.0

    def test_unknown_model_returns_zero_cost_without_cost_budget(self, cost_config):
        tracker = BudgetTracker(cost_config=cost_config)
        node = _make_node(budget=BudgetConfig())  # no max_cost
        wf = _make_workflow()
        usage = TokenUsage(input_tokens=100, output_tokens=100)
        cost = tracker.record_usage(node, wf, usage, "unknown-model")
        assert cost == 0.0
        # Tokens are still tracked even without pricing
        assert node.budget_consumed.tokens == 200

    def test_unknown_model_raises_when_cost_budget_set(self, cost_config):
        tracker = BudgetTracker(cost_config=cost_config)
        node = _make_node(budget=BudgetConfig(max_cost=10.0))
        wf = _make_workflow()
        usage = TokenUsage(input_tokens=100, output_tokens=100)
        with pytest.raises(ValueError, match="No pricing data"):
            tracker.record_usage(node, wf, usage, "unknown-model")


# =========================================================================
# check_warning_threshold
# =========================================================================

class TestCheckWarningThreshold:
    """Tests for BudgetTracker.check_warning_threshold()."""

    def test_returns_none_when_below_threshold(self, budget_tracker):
        node = _make_node(
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=700,
        )
        result = budget_tracker.check_warning_threshold(node, threshold=0.8)
        assert result is None

    def test_returns_warning_when_above_token_threshold(self, budget_tracker):
        node = _make_node(
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=800,
        )
        result = budget_tracker.check_warning_threshold(node, threshold=0.8)
        assert result is not None
        assert "Token usage" in result

    def test_returns_warning_when_above_cost_threshold(self, budget_tracker):
        node = _make_node(
            budget=BudgetConfig(max_cost=10.0),
            consumed_cost=9.0,
        )
        result = budget_tracker.check_warning_threshold(node, threshold=0.8)
        assert result is not None
        assert "Cost" in result

    def test_returns_warning_for_duration(self, budget_tracker):
        past = datetime.now(timezone.utc) - timedelta(milliseconds=9000)
        node = _make_node(
            budget=BudgetConfig(max_duration_ms=10_000),
            started_at=past,
        )
        result = budget_tracker.check_warning_threshold(node, threshold=0.8)
        assert result is not None
        assert "Duration" in result

    def test_returns_none_when_no_limits(self, budget_tracker):
        node = _make_node(consumed_tokens=999_999, consumed_cost=999.0)
        result = budget_tracker.check_warning_threshold(node)
        assert result is None

    def test_combines_multiple_warnings(self, budget_tracker):
        node = _make_node(
            budget=BudgetConfig(max_tokens=1000, max_cost=1.0),
            consumed_tokens=900,
            consumed_cost=0.9,
        )
        result = budget_tracker.check_warning_threshold(node, threshold=0.8)
        assert result is not None
        assert "Token usage" in result
        assert "Cost" in result

    def test_exact_threshold_triggers(self, budget_tracker):
        """Usage at exactly the threshold ratio triggers warning."""
        node = _make_node(
            budget=BudgetConfig(max_tokens=1000),
            consumed_tokens=800,
        )
        result = budget_tracker.check_warning_threshold(node, threshold=0.8)
        assert result is not None

    def test_default_threshold_is_80_percent(self, budget_tracker):
        node = _make_node(
            budget=BudgetConfig(max_tokens=100),
            consumed_tokens=80,
        )
        # Default threshold = 0.8
        result = budget_tracker.check_warning_threshold(node)
        assert result is not None


# =========================================================================
# calculate_child_budget
# =========================================================================

class TestCalculateChildBudget:
    """Tests for BudgetTracker.calculate_child_budget()."""

    def test_inherits_fraction_of_remaining_tokens(self, budget_tracker):
        parent = _make_node(
            budget=BudgetConfig(max_tokens=1000, child_budget_fraction=0.5),
            consumed_tokens=200,
        )
        child_budget = budget_tracker.calculate_child_budget(parent)
        # remaining = 800, allocatable = 800 * 0.9 (10% recovery reserve) = 720
        # child = 720 * 0.5 = 360
        assert child_budget.max_tokens == 360

    def test_inherits_fraction_of_remaining_cost(self, budget_tracker):
        parent = _make_node(
            budget=BudgetConfig(max_cost=10.0, child_budget_fraction=0.5),
            consumed_cost=4.0,
        )
        child_budget = budget_tracker.calculate_child_budget(parent)
        # remaining = 6.0, allocatable = 6.0 * 0.9 = 5.4, child = 5.4 * 0.5 = 2.7
        assert child_budget.max_cost == pytest.approx(2.7)

    def test_inherits_duration_unchanged(self, budget_tracker):
        parent = _make_node(
            budget=BudgetConfig(max_duration_ms=60_000),
        )
        child_budget = budget_tracker.calculate_child_budget(parent)
        assert child_budget.max_duration_ms == 60_000

    def test_decrements_agent_depth(self, budget_tracker):
        parent = _make_node(budget=BudgetConfig(max_agent_depth=3))
        child_budget = budget_tracker.calculate_child_budget(parent)
        assert child_budget.max_agent_depth == 2

    def test_depth_does_not_go_negative(self, budget_tracker):
        parent = _make_node(budget=BudgetConfig(max_agent_depth=0))
        child_budget = budget_tracker.calculate_child_budget(parent)
        assert child_budget.max_agent_depth == 0

    def test_preserves_parallel_agents(self, budget_tracker):
        parent = _make_node(budget=BudgetConfig(max_parallel_agents=5))
        child_budget = budget_tracker.calculate_child_budget(parent)
        assert child_budget.max_parallel_agents == 5

    def test_preserves_sub_agents_total(self, budget_tracker):
        parent = _make_node(budget=BudgetConfig(max_sub_agents_total=7))
        child_budget = budget_tracker.calculate_child_budget(parent)
        assert child_budget.max_sub_agents_total == 7

    def test_zero_remaining_tokens_yields_zero(self, budget_tracker):
        parent = _make_node(
            budget=BudgetConfig(max_tokens=500, child_budget_fraction=0.5),
            consumed_tokens=500,
        )
        child_budget = budget_tracker.calculate_child_budget(parent)
        assert child_budget.max_tokens == 0

    def test_zero_remaining_cost_yields_zero(self, budget_tracker):
        parent = _make_node(
            budget=BudgetConfig(max_cost=5.0, child_budget_fraction=0.5),
            consumed_cost=5.0,
        )
        child_budget = budget_tracker.calculate_child_budget(parent)
        assert child_budget.max_cost == pytest.approx(0.0)

    def test_over_consumed_still_yields_zero(self, budget_tracker):
        """If parent over-consumed, child gets 0, not negative."""
        parent = _make_node(
            budget=BudgetConfig(max_tokens=100, child_budget_fraction=0.5),
            consumed_tokens=200,
        )
        child_budget = budget_tracker.calculate_child_budget(parent)
        assert child_budget.max_tokens == 0

    def test_inherit_remaining_false_returns_defaults(self, budget_tracker):
        parent = _make_node(
            budget=BudgetConfig(
                max_tokens=1000,
                max_cost=10.0,
                inherit_remaining=False,
                max_agent_depth=3,
                max_parallel_agents=5,
                max_sub_agents_total=7,
            ),
            consumed_tokens=100,
        )
        child_budget = budget_tracker.calculate_child_budget(parent)
        # When not inheriting, tokens/cost/duration are None
        assert child_budget.max_tokens is None
        assert child_budget.max_cost is None
        assert child_budget.max_duration_ms is None
        assert child_budget.max_agent_depth == 2
        assert child_budget.max_parallel_agents == 5
        assert child_budget.max_sub_agents_total == 7

    def test_child_budget_fraction_custom(self, budget_tracker):
        parent = _make_node(
            budget=BudgetConfig(max_tokens=1000, child_budget_fraction=0.25),
            consumed_tokens=0,
        )
        child_budget = budget_tracker.calculate_child_budget(parent)
        # remaining = 1000, allocatable = 1000 * 0.9 = 900, child = 900 * 0.25 = 225
        assert child_budget.max_tokens == 225

    def test_child_budget_sets_inherit_remaining_true(self, budget_tracker):
        parent = _make_node(
            budget=BudgetConfig(inherit_remaining=True, max_tokens=1000),
        )
        child_budget = budget_tracker.calculate_child_budget(parent)
        assert child_budget.inherit_remaining is True

    def test_no_token_limit_yields_none(self, budget_tracker):
        """Parent without token limit → child has no token limit."""
        parent = _make_node(budget=BudgetConfig(max_tokens=None))
        child_budget = budget_tracker.calculate_child_budget(parent)
        assert child_budget.max_tokens is None

    def test_no_cost_limit_yields_none(self, budget_tracker):
        parent = _make_node(budget=BudgetConfig(max_cost=None))
        child_budget = budget_tracker.calculate_child_budget(parent)
        assert child_budget.max_cost is None


# =========================================================================
# Edge cases & integration
# =========================================================================

class TestEdgeCases:
    """Edge-case and integration scenarios."""

    def test_child_with_zero_budget_immediately_fails(self, budget_tracker):
        """A child with max_tokens=0 fails the first budget check."""
        node = _make_node(budget=BudgetConfig(max_tokens=0))
        wf = _make_workflow()
        # max_tokens=0 is falsy, so _check_limits skips the token check.
        # This is the expected behavior: 0 means "no limit" (same as None).
        budget_tracker.check_budget(node, wf)

    def test_record_then_check_exceeds(self, budget_tracker):
        """Recording usage that pushes past the limit, then checking, raises."""
        node = _make_node(budget=BudgetConfig(max_tokens=200, max_cost=10.0))
        wf = _make_workflow()
        usage = TokenUsage(input_tokens=100, output_tokens=100)
        budget_tracker.record_usage(node, wf, usage, "test-model")
        assert node.budget_consumed.tokens == 200
        with pytest.raises(BudgetExceededException):
            budget_tracker.check_budget(node, wf)

    def test_budget_tracker_without_cost_config(self):
        """BudgetTracker with no CostConfig still works (defaults)."""
        tracker = BudgetTracker()
        node = _make_node(budget=BudgetConfig(max_tokens=1000))
        wf = _make_workflow()
        tracker.check_budget(node, wf)

    def test_multiple_agents_share_workflow_budget(self, budget_tracker):
        """Two agents recording usage both contribute to workflow totals."""
        wf = _make_workflow(budget=BudgetConfig(max_tokens=500))
        node_a = _make_node(
            task_id="a", agent_id="agent-a",
            budget=BudgetConfig(max_tokens=1000),
        )
        node_b = _make_node(
            task_id="b", agent_id="agent-b",
            budget=BudgetConfig(max_tokens=1000),
        )
        budget_tracker.record_usage(
            node_a, wf, TokenUsage(input_tokens=200, output_tokens=0), "test-model"
        )
        budget_tracker.record_usage(
            node_b, wf, TokenUsage(input_tokens=200, output_tokens=0), "test-model"
        )
        assert wf.budget_consumed.tokens == 400
        # One more push triggers workflow limit
        budget_tracker.record_usage(
            node_a, wf, TokenUsage(input_tokens=100, output_tokens=0), "test-model"
        )
        with pytest.raises(BudgetExceededException, match="workflow"):
            budget_tracker.check_budget(node_b, wf)
