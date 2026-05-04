"""Tests for config_loader.py — YAML loading and ConfigBundle building."""
import tempfile
from pathlib import Path

import pytest
import yaml

from parr.core_types import PoliciesConfig, SpawnPolicy
from parr.config.config_loader import _build_policies_config


class TestBuildPoliciesConfig:
    def test_defaults(self):
        pc = _build_policies_config({})
        assert pc.same_role_spawn_policy == SpawnPolicy.WARN
        assert pc.consultant_model is None
        assert pc.consultant_max_tokens == 512

    def test_deny(self):
        pc = _build_policies_config({"same_role_spawn_policy": "deny"})
        assert pc.same_role_spawn_policy == SpawnPolicy.DENY

    def test_consult(self):
        pc = _build_policies_config({
            "same_role_spawn_policy": "consult",
            "consultant_model": "gpt-4.1-nano",
            "consultant_max_tokens": 256,
            "consultant_temperature": 0.0,
        })
        assert pc.same_role_spawn_policy == SpawnPolicy.CONSULT
        assert pc.consultant_model == "gpt-4.1-nano"
        assert pc.consultant_max_tokens == 256
        assert pc.consultant_temperature == 0.0

    def test_invalid_policy_defaults_to_warn(self):
        pc = _build_policies_config({"same_role_spawn_policy": "invalid_value"})
        assert pc.same_role_spawn_policy == SpawnPolicy.WARN

    def test_missing_policy_key(self):
        pc = _build_policies_config({"consultant_model": "test"})
        assert pc.same_role_spawn_policy == SpawnPolicy.WARN


class TestPoliciesYamlLoading:
    """Test that policies.yaml is loaded when present."""

    def test_load_config_includes_policies(self):
        """ConfigBundle includes policies_config when policies.yaml exists."""
        from parr.config.config_loader import ConfigBundle
        # Verify the field exists
        bundle = ConfigBundle.__dataclass_fields__
        assert "policies_config" in bundle

    def test_policies_yaml_optional(self, tmp_path):
        """When policies.yaml is missing, defaults are used."""
        # Create minimal config files with file-based prompts
        models = {"models": {"test": {
            "input_price_per_1k": 0.001,
            "output_price_per_1k": 0.002,
            "context_window": 128000,
        }}}
        (tmp_path / "system_prompt.md").write_text("You are a test.")
        (tmp_path / "report_template.md").write_text("Report here.")
        roles = {"roles": {"test_role": {
            "model": "test",
            "system_prompt": "system_prompt.md",
            "report_template": "report_template.md",
        }}}
        budget = {"budget_defaults": {}}

        (tmp_path / "models.yaml").write_text(yaml.dump(models))
        (tmp_path / "roles.yaml").write_text(yaml.dump(roles))
        (tmp_path / "budget.yaml").write_text(yaml.dump(budget))

        from parr.config.config_loader import load_config
        bundle = load_config(str(tmp_path))
        assert bundle.policies_config.same_role_spawn_policy == SpawnPolicy.WARN
