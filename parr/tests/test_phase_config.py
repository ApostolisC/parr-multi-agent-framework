"""Tests for parr.core_types.PhaseConfig and custom phase definitions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from parr.core_types import (
    AgentConfig,
    AgentInput,
    AgentNode,
    AgentOutput,
    BudgetConfig,
    ExecutionMetadata,
    Message,
    MessageRole,
    Phase,
    PhaseConfig,
    TokenUsage,
    WorkflowExecution,
)
from parr.context_manager import ContextManager, PHASE_PROMPTS


# ===================================================================
# 1. PhaseConfig dataclass
# ===================================================================

class TestPhaseConfigDefaults:

    def test_default_phases(self):
        pc = PhaseConfig()
        assert pc.phases == [Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT]

    def test_default_max_review_cycles(self):
        pc = PhaseConfig()
        assert pc.max_review_cycles == 2

    def test_default_phase_limits_none(self):
        pc = PhaseConfig()
        assert pc.phase_limits is None

    def test_default_phase_prompts_none(self):
        pc = PhaseConfig()
        assert pc.phase_prompts is None

    def test_default_review_phase_none(self):
        pc = PhaseConfig()
        assert pc.review_phase is None
        assert pc.review_retry_phase is None


class TestPhaseConfigCustom:

    def test_custom_phases(self):
        pc = PhaseConfig(phases=[Phase.ACT, Phase.REPORT])
        assert pc.phases == [Phase.ACT, Phase.REPORT]

    def test_custom_phase_limits(self):
        limits = {Phase.PLAN: 3, Phase.ACT: 10}
        pc = PhaseConfig(phase_limits=limits)
        assert pc.phase_limits == limits

    def test_custom_phase_prompts(self):
        prompts = {Phase.ACT: "Custom execution prompt."}
        pc = PhaseConfig(phase_prompts=prompts)
        assert pc.phase_prompts == prompts

    def test_custom_max_review_cycles(self):
        pc = PhaseConfig(max_review_cycles=0)
        assert pc.max_review_cycles == 0


# ===================================================================
# 2. Effective review phase detection
# ===================================================================

class TestEffectiveReviewPhase:

    def test_default_detects_review(self):
        pc = PhaseConfig()
        assert pc.effective_review_phase == Phase.REVIEW

    def test_no_review_in_phases(self):
        pc = PhaseConfig(phases=[Phase.PLAN, Phase.ACT, Phase.REPORT])
        assert pc.effective_review_phase is None

    def test_explicit_review_phase(self):
        pc = PhaseConfig(review_phase=Phase.ACT)
        assert pc.effective_review_phase == Phase.ACT

    def test_explicit_review_phase_not_in_phases(self):
        pc = PhaseConfig(
            phases=[Phase.PLAN, Phase.ACT, Phase.REPORT],
            review_phase=Phase.REVIEW,
        )
        assert pc.effective_review_phase is None

    def test_act_only_no_review(self):
        pc = PhaseConfig(phases=[Phase.ACT])
        assert pc.effective_review_phase is None


class TestEffectiveReviewRetryPhase:

    def test_default_retries_act(self):
        pc = PhaseConfig()
        assert pc.effective_review_retry_phase == Phase.ACT

    def test_no_review_means_no_retry(self):
        pc = PhaseConfig(phases=[Phase.PLAN, Phase.ACT, Phase.REPORT])
        assert pc.effective_review_retry_phase is None

    def test_review_first_in_phases_no_predecessor(self):
        pc = PhaseConfig(phases=[Phase.REVIEW, Phase.REPORT])
        assert pc.effective_review_retry_phase is None

    def test_explicit_retry_phase(self):
        pc = PhaseConfig(review_retry_phase=Phase.PLAN)
        assert pc.effective_review_retry_phase == Phase.PLAN

    def test_explicit_retry_phase_not_in_phases(self):
        pc = PhaseConfig(
            phases=[Phase.ACT, Phase.REVIEW],
            review_retry_phase=Phase.PLAN,
        )
        assert pc.effective_review_retry_phase is None

    def test_custom_sequence_predecessor(self):
        """When REVIEW follows PLAN, PLAN is the retry phase."""
        pc = PhaseConfig(phases=[Phase.PLAN, Phase.REVIEW, Phase.REPORT])
        assert pc.effective_review_retry_phase == Phase.PLAN


# ===================================================================
# 3. ContextManager — custom phase prompts
# ===================================================================

class TestContextManagerCustomPrompts:

    def _build_system(self, phase, phase_prompts=None):
        cm = ContextManager(phase_prompts=phase_prompts)
        config = AgentConfig(system_prompt="Base prompt.")
        input_ = AgentInput(task="Test task")
        msgs = cm.build_phase_messages(phase, config, input_)
        return msgs[0].content  # system message

    def test_default_prompts_used(self):
        system = self._build_system(Phase.ACT)
        assert "Phase: Execution" in system

    def test_custom_prompt_replaces_default(self):
        custom = {Phase.ACT: "Custom execution phase instructions."}
        system = self._build_system(Phase.ACT, phase_prompts=custom)
        assert "Custom execution phase instructions." in system
        assert "Phase: Execution" not in system

    def test_unoverridden_phase_gets_default(self):
        """Only the overridden phase gets custom prompt; others keep defaults."""
        custom = {Phase.ACT: "Custom act."}
        system = self._build_system(Phase.PLAN, phase_prompts=custom)
        assert "Phase: Planning" in system

    def test_empty_custom_prompt(self):
        """Empty string custom prompt overrides default (no phase prompt)."""
        custom = {Phase.ACT: ""}
        system = self._build_system(Phase.ACT, phase_prompts=custom)
        # The empty string should result in no phase prompt being added
        assert "Phase: Execution" not in system


# ===================================================================
# 4. ContextManager — phase sequence cross-phase context
# ===================================================================

class TestContextManagerPhaseSequence:

    def _build_user_msg(self, phase, phase_sequence=None, summaries=None):
        cm = ContextManager(phase_sequence=phase_sequence)
        for p, s in (summaries or {}).items():
            cm.record_phase_summary(p, s)
        config = AgentConfig(system_prompt="Base prompt.")
        input_ = AgentInput(task="Test task")
        msgs = cm.build_phase_messages(phase, config, input_)
        return msgs[1].content  # user message

    def test_legacy_act_gets_plan_summary(self):
        """Without phase_sequence, ACT gets Plan Summary (legacy)."""
        user = self._build_user_msg(
            Phase.ACT,
            summaries={Phase.PLAN: "Plan done."},
        )
        assert "## Plan Summary" in user
        assert "Plan done." in user

    def test_legacy_review_gets_act_summary(self):
        user = self._build_user_msg(
            Phase.REVIEW,
            summaries={Phase.ACT: "Act done."},
        )
        assert "## Execution Summary" in user

    def test_legacy_report_gets_review_summary(self):
        user = self._build_user_msg(
            Phase.REPORT,
            summaries={Phase.REVIEW: "Review done."},
        )
        assert "## Review Result" in user

    def test_sequence_act_gets_plan_summary(self):
        """With default sequence, ACT still gets Plan Summary."""
        seq = [Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT]
        user = self._build_user_msg(
            Phase.ACT,
            phase_sequence=seq,
            summaries={Phase.PLAN: "Plan done."},
        )
        assert "Plan Summary" in user
        assert "Plan done." in user

    def test_sequence_first_phase_no_predecessor(self):
        """First phase in sequence has no predecessor summary."""
        seq = [Phase.PLAN, Phase.ACT]
        user = self._build_user_msg(
            Phase.PLAN,
            phase_sequence=seq,
            summaries={Phase.ACT: "Act done."},
        )
        assert "Summary" not in user or "Act done." not in user

    def test_custom_sequence_report_after_act(self):
        """In [PLAN, ACT, REPORT], REPORT gets ACT summary."""
        seq = [Phase.PLAN, Phase.ACT, Phase.REPORT]
        user = self._build_user_msg(
            Phase.REPORT,
            phase_sequence=seq,
            summaries={Phase.ACT: "Act done.", Phase.REVIEW: "Review done."},
        )
        assert "Act done." in user
        assert "Review done." not in user

    def test_custom_sequence_review_after_plan(self):
        """In [PLAN, REVIEW, ACT, REPORT], REVIEW gets PLAN summary."""
        seq = [Phase.PLAN, Phase.REVIEW, Phase.ACT, Phase.REPORT]
        user = self._build_user_msg(
            Phase.REVIEW,
            phase_sequence=seq,
            summaries={Phase.PLAN: "Plan done."},
        )
        assert "Plan done." in user


# ===================================================================
# 5. ContextManager — backward compatibility
# ===================================================================

class TestContextManagerBackwardCompat:

    def test_no_new_params_same_behavior(self):
        """ContextManager without new params behaves identically to before."""
        cm = ContextManager()
        assert cm._phase_prompts is None
        assert cm._phase_sequence is None

    def test_phase_sequence_none_uses_legacy(self):
        cm = ContextManager(phase_sequence=None)
        cm.record_phase_summary(Phase.PLAN, "Done planning.")
        config = AgentConfig(system_prompt="Base.")
        input_ = AgentInput(task="Test")
        msgs = cm.build_phase_messages(Phase.ACT, config, input_)
        user_msg = msgs[1].content
        assert "## Plan Summary" in user_msg


# ===================================================================
# 6. PhaseConfig integration with AgentRuntime constructor
# ===================================================================

class TestAgentRuntimePhaseConfigIntegration:

    def test_legacy_params_create_default_phase_config(self):
        """Legacy max_review_cycles/phase_limits create a PhaseConfig."""
        from parr.agent_runtime import AgentRuntime

        llm = MagicMock()
        bt = MagicMock()
        eb = MagicMock()

        rt = AgentRuntime(
            llm=llm, budget_tracker=bt, event_bus=eb,
            max_review_cycles=3,
            phase_limits={Phase.ACT: 10},
        )
        assert rt._phase_config.max_review_cycles == 3
        assert rt._phase_config.phase_limits == {Phase.ACT: 10}
        assert rt._phase_config.phases == [Phase.PLAN, Phase.ACT, Phase.REVIEW, Phase.REPORT]

    def test_explicit_phase_config_overrides_legacy(self):
        """Explicit PhaseConfig takes precedence over legacy params."""
        from parr.agent_runtime import AgentRuntime

        llm = MagicMock()
        bt = MagicMock()
        eb = MagicMock()

        pc = PhaseConfig(
            phases=[Phase.ACT, Phase.REPORT],
            max_review_cycles=0,
        )
        rt = AgentRuntime(
            llm=llm, budget_tracker=bt, event_bus=eb,
            max_review_cycles=5,  # should be overridden
            phase_config=pc,
        )
        assert rt._phase_config is pc
        assert rt._max_review_cycles == 0
        assert rt._phase_config.phases == [Phase.ACT, Phase.REPORT]

    def test_backward_compat_fields_derived(self):
        """_max_review_cycles and _phase_limits are derived from PhaseConfig."""
        from parr.agent_runtime import AgentRuntime

        llm = MagicMock()
        bt = MagicMock()
        eb = MagicMock()

        pc = PhaseConfig(max_review_cycles=1, phase_limits={Phase.PLAN: 3})
        rt = AgentRuntime(
            llm=llm, budget_tracker=bt, event_bus=eb,
            phase_config=pc,
        )
        assert rt._max_review_cycles == 1
        assert rt._phase_limits == {Phase.PLAN: 3}


# ===================================================================
# 7. Orchestrator phase_config passthrough
# ===================================================================

class TestOrchestratorPhaseConfig:

    def test_orchestrator_stores_phase_config(self):
        from parr.orchestrator import Orchestrator

        llm = MagicMock()
        pc = PhaseConfig(phases=[Phase.ACT])
        orch = Orchestrator(llm=llm, phase_config=pc)
        assert orch._phase_config is pc

    def test_orchestrator_default_no_phase_config(self):
        from parr.orchestrator import Orchestrator

        llm = MagicMock()
        orch = Orchestrator(llm=llm)
        assert orch._phase_config is None


# ===================================================================
# 8. Edge cases
# ===================================================================

class TestPhaseConfigEdgeCases:

    def test_empty_phases_list(self):
        pc = PhaseConfig(phases=[])
        assert pc.effective_review_phase is None
        assert pc.effective_review_retry_phase is None

    def test_single_phase(self):
        pc = PhaseConfig(phases=[Phase.ACT])
        assert pc.phases == [Phase.ACT]
        assert pc.effective_review_phase is None

    def test_review_only(self):
        pc = PhaseConfig(phases=[Phase.REVIEW])
        assert pc.effective_review_phase == Phase.REVIEW
        assert pc.effective_review_retry_phase is None  # No predecessor

    def test_all_custom_prompts(self):
        prompts = {
            Phase.PLAN: "P",
            Phase.ACT: "A",
            Phase.REVIEW: "R",
            Phase.REPORT: "Rp",
        }
        pc = PhaseConfig(phase_prompts=prompts)
        cm = ContextManager(phase_prompts=pc.phase_prompts)
        config = AgentConfig(system_prompt="Base.")
        input_ = AgentInput(task="Test")
        for phase, prompt_text in prompts.items():
            msgs = cm.build_phase_messages(phase, config, input_)
            assert prompt_text in msgs[0].content

    def test_phase_config_with_zero_review_cycles(self):
        pc = PhaseConfig(max_review_cycles=0)
        assert pc.max_review_cycles == 0
        assert pc.effective_review_phase == Phase.REVIEW
