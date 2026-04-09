"""Tests for parr.persistence — file-system persistence layer.

Covers:
- AgentFileStore: folder creation, incremental saves, read helpers, child stores
- WorkflowFileStore: workflow folder, root/child store management
- Integration: orchestrator with persist_dir, agent runtime saves
- Hierarchical structure: nested sub-agents, sub_agents.json only tracks direct children
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from parr.core_types import (
    AgentConfig,
    AgentNode,
    AgentOutput,
    AgentStatus,
    BudgetConfig,
    ErrorEntry,
    ErrorSource,
    ExecutionMetadata,
    Phase,
    TokenUsage,
    ToolCall,
    ToolDef,
    WorkflowExecution,
    generate_id,
)
from parr.framework_tools import AgentWorkingMemory, Finding, ReviewItem, TodoItem
from parr.orchestrator import Orchestrator
from parr.persistence import AgentFileStore, WorkflowFileStore, _read_json
from parr.tests.mock_llm import MockToolCallingLLM, make_text_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory() -> AgentWorkingMemory:
    """Create a populated AgentWorkingMemory for testing."""
    mem = AgentWorkingMemory()
    mem.todo_list = [
        TodoItem(index=0, description="Analyze data", priority="high", completed=True, completion_summary="Done"),
        TodoItem(index=1, description="Write report", priority="medium"),
    ]
    mem.findings = [
        Finding(category="risk", content="High CPU usage", source="metrics", confidence="high"),
    ]
    mem.review_checklist = [
        ReviewItem(criterion="Completeness", rating="pass", justification="All tasks done"),
    ]
    mem.submitted_report = {"summary": "All good", "findings": ["risk1"]}
    return mem


def _make_output(task_id: str = "task-1", agent_id: str = "agent-1") -> AgentOutput:
    """Create a minimal AgentOutput for testing."""
    return AgentOutput(
        task_id=task_id,
        agent_id=agent_id,
        role="analyst",
        status="completed",
        summary="Analysis complete",
        findings={"risks": ["cpu"]},
    )


# =========================================================================
# AgentFileStore — folder creation and structure
# =========================================================================


class TestAgentFileStoreCreation:
    def test_creates_directory(self, tmp_path: Path):
        agent_dir = tmp_path / "agent_1"
        store = AgentFileStore(agent_dir)
        assert agent_dir.exists()
        assert (agent_dir / "memory").exists()

    def test_path_property(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "my_agent")
        assert store.path == tmp_path / "my_agent"


# =========================================================================
# AgentFileStore — agent info
# =========================================================================


class TestAgentFileStoreAgentInfo:
    def test_save_agent_info(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        store.save_agent_info(
            task_id="t1", agent_id="a1", role="analyst",
            sub_role="risk", task="Analyze risks", status="running",
            depth=0, model="gpt-4",
        )
        data = _read_json(tmp_path / "agent" / "agent.json")
        assert data["task_id"] == "t1"
        assert data["role"] == "analyst"
        assert data["sub_role"] == "risk"
        assert data["status"] == "running"
        assert data["model"] == "gpt-4"

    def test_update_agent_status(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        store.save_agent_info(task_id="t1", agent_id="a1", role="analyst")
        store.update_agent_status("completed")
        data = _read_json(tmp_path / "agent" / "agent.json")
        assert data["status"] == "completed"

    def test_read_agent_info(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        store.save_agent_info(task_id="t1", agent_id="a1", role="analyst")
        info = store.read_agent_info()
        assert info["task_id"] == "t1"


# =========================================================================
# AgentFileStore — phase conversations
# =========================================================================


class TestAgentFileStoreConversation:
    def test_save_phase_conversation(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        store.save_phase_conversation(
            "plan", content="I will do X", iterations=2,
            hit_iteration_limit=False, tool_calls_made=[{"name": "create_todo_list"}],
        )
        data = _read_json(tmp_path / "agent" / "conversation.json")
        assert "plan" in data
        assert data["plan"]["content"] == "I will do X"
        assert data["plan"]["iterations"] == 2
        assert data["plan"]["tool_calls_count"] == 1

    def test_multiple_phases_accumulate(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        store.save_phase_conversation("plan", content="plan text")
        store.save_phase_conversation("act", content="act text")
        data = _read_json(tmp_path / "agent" / "conversation.json")
        assert "plan" in data
        assert "act" in data


# =========================================================================
# AgentFileStore — tool calls
# =========================================================================


class TestAgentFileStoreToolCalls:
    def test_append_tool_calls(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        store.append_tool_calls([
            {"name": "search", "success": True},
        ])
        data = _read_json(tmp_path / "agent" / "tool_calls.json")
        assert len(data) == 1
        assert data[0]["name"] == "search"

    def test_incremental_append(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        store.append_tool_calls([{"name": "a"}])
        store.append_tool_calls([{"name": "b"}, {"name": "c"}])
        data = _read_json(tmp_path / "agent" / "tool_calls.json")
        assert len(data) == 3


# =========================================================================
# AgentFileStore — working memory
# =========================================================================


class TestAgentFileStoreMemory:
    def test_save_memory(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        memory = _make_memory()
        store.save_memory(memory)

        todo = _read_json(tmp_path / "agent" / "memory" / "todo_list.json")
        assert len(todo) == 2
        assert todo[0]["description"] == "Analyze data"
        assert todo[0]["completed"] is True

        findings = _read_json(tmp_path / "agent" / "memory" / "findings.json")
        assert len(findings) == 1
        assert findings[0]["category"] == "risk"

        review = _read_json(tmp_path / "agent" / "memory" / "review.json")
        assert len(review) == 1
        assert review[0]["rating"] == "pass"

        report = _read_json(tmp_path / "agent" / "memory" / "report.json")
        assert report["summary"] == "All good"

    def test_save_empty_memory(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        memory = AgentWorkingMemory()
        store.save_memory(memory)
        todo = _read_json(tmp_path / "agent" / "memory" / "todo_list.json")
        assert todo == []

    def test_read_memory(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        store.save_memory(_make_memory())
        mem = store.read_memory()
        assert len(mem["todo_list"]) == 2
        assert len(mem["findings"]) == 1
        assert mem["report"]["summary"] == "All good"


# =========================================================================
# AgentFileStore — output
# =========================================================================


class TestAgentFileStoreOutput:
    def test_save_output(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        output = _make_output()
        store.save_output(output)
        data = _read_json(tmp_path / "agent" / "output.json")
        assert data["status"] == "completed"
        assert data["summary"] == "Analysis complete"
        assert data["findings"] == {"risks": ["cpu"]}

    def test_read_output(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        store.save_output(_make_output())
        data = store.read_output()
        assert data["role"] == "analyst"


# =========================================================================
# AgentFileStore — sub-agents
# =========================================================================


class TestAgentFileStoreSubAgents:
    def test_register_child(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        store.register_child(
            task_id="c1", agent_id="ca1", role="reviewer",
            sub_role="code", task_description="Review code quality",
        )
        data = _read_json(tmp_path / "agent" / "sub_agents.json")
        assert len(data) == 1
        assert data[0]["task_id"] == "c1"
        assert data[0]["role"] == "reviewer"
        assert data[0]["status"] == "spawned"

    def test_multiple_children(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        store.register_child(task_id="c1", agent_id="a1", role="r1")
        store.register_child(task_id="c2", agent_id="a2", role="r2")
        data = store.read_sub_agents()
        assert len(data) == 2

    def test_update_child_status(self, tmp_path: Path):
        store = AgentFileStore(tmp_path / "agent")
        store.register_child(task_id="c1", agent_id="a1", role="r1")
        store.update_child_status("c1", "completed")
        data = store.read_sub_agents()
        assert data[0]["status"] == "completed"

    def test_create_child_store(self, tmp_path: Path):
        parent = AgentFileStore(tmp_path / "parent")
        child = parent.create_child_store("reviewer", "abc12345-1234")
        assert child.path.parent.name == "sub_agents"
        assert child.path.name == "reviewer_abc12345"
        assert child.path.exists()

    def test_nested_child_stores(self, tmp_path: Path):
        """Verify recursive nesting: parent → child → grandchild."""
        root = AgentFileStore(tmp_path / "root")
        child = root.create_child_store("analyst", "child-id")
        grandchild = child.create_child_store("researcher", "grand-id")

        # Grandchild should be nested under child's sub_agents/
        assert "sub_agents" in str(grandchild.path)
        assert grandchild.path.exists()

        # Each level has its own memory directory
        assert (root.path / "memory").exists()
        assert (child.path / "memory").exists()
        assert (grandchild.path / "memory").exists()


# =========================================================================
# AgentFileStore — sub_agents.json only tracks direct children
# =========================================================================


class TestHierarchicalSubAgentsJson:
    def test_parent_does_not_see_grandchildren(self, tmp_path: Path):
        """Parent's sub_agents.json should only list direct children."""
        root = AgentFileStore(tmp_path / "root")
        child = root.create_child_store("analyst", "child-1")

        # Register child in root
        root.register_child(task_id="child-1", agent_id="ca", role="analyst")

        # Create grandchild under child
        grandchild = child.create_child_store("researcher", "grand-1")
        child.register_child(task_id="grand-1", agent_id="ga", role="researcher")

        # Root should only know about its direct child
        root_subs = root.read_sub_agents()
        assert len(root_subs) == 1
        assert root_subs[0]["task_id"] == "child-1"

        # Child should know about its child (the grandchild)
        child_subs = child.read_sub_agents()
        assert len(child_subs) == 1
        assert child_subs[0]["task_id"] == "grand-1"


# =========================================================================
# WorkflowFileStore
# =========================================================================


class TestWorkflowFileStore:
    def test_creates_workflow_directory(self, tmp_path: Path):
        store = WorkflowFileStore(tmp_path, "wf-1")
        assert (tmp_path / "wf-1").exists()

    def test_workflow_dir_property(self, tmp_path: Path):
        store = WorkflowFileStore(tmp_path, "wf-1")
        assert store.workflow_dir == tmp_path / "wf-1"

    def test_save_workflow_info(self, tmp_path: Path):
        store = WorkflowFileStore(tmp_path, "wf-1")
        store.save_workflow_info(
            workflow_id="wf-1", status="running",
            root_task_id="root-1",
            budget={"max_tokens": 5000},
        )
        data = store.read_workflow_info()
        assert data["workflow_id"] == "wf-1"
        assert data["status"] == "running"
        assert data["root_task_id"] == "root-1"

    def test_update_workflow_status(self, tmp_path: Path):
        store = WorkflowFileStore(tmp_path, "wf-1")
        store.save_workflow_info(workflow_id="wf-1", status="running")
        store.update_workflow_status("completed")
        data = store.read_workflow_info()
        assert data["status"] == "completed"

    def test_create_root_store(self, tmp_path: Path):
        store = WorkflowFileStore(tmp_path, "wf-1")
        root = store.create_root_store("root-task-1")
        # Root store writes directly into workflow directory
        assert root.path == tmp_path / "wf-1"

    def test_create_child_store(self, tmp_path: Path):
        store = WorkflowFileStore(tmp_path, "wf-1")
        root = store.create_root_store("root-1")
        child = store.create_child_store("root-1", "child-1", "analyst")
        assert child.path.exists()
        assert "sub_agents" in str(child.path)

    def test_get_store(self, tmp_path: Path):
        store = WorkflowFileStore(tmp_path, "wf-1")
        root = store.create_root_store("root-1")
        assert store.get_store("root-1") is root
        assert store.get_store("nonexistent") is None

    def test_create_child_store_unknown_parent_raises(self, tmp_path: Path):
        store = WorkflowFileStore(tmp_path, "wf-1")
        with pytest.raises(ValueError, match="parent task_id"):
            store.create_child_store("unknown", "child-1", "analyst")


# =========================================================================
# Integration: Orchestrator with persist_dir
# =========================================================================


class TestOrchestratorPersistence:
    """Test that the orchestrator creates the expected folder structure."""

    @pytest.mark.asyncio
    async def test_workflow_creates_folder_structure(self, tmp_path: Path):
        """A simple workflow should produce workflow.json, agent.json, etc."""
        llm = MockToolCallingLLM([
            make_text_response("Plan: do task"),          # plan
            make_text_response("Done with act"),          # act
            make_text_response("REVIEW_PASSED"),          # review
            make_text_response("Final report"),           # report
        ])
        orch = Orchestrator(
            llm=llm,
            persist_dir=str(tmp_path),
        )
        output = await orch.start_workflow(
            task="Test task",
            role="analyst",
            system_prompt="You are a test analyst.",
            model="test-model",
        )

        # Find the workflow directory (name is the workflow_id)
        wf_dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
        assert len(wf_dirs) == 1
        wf_dir = wf_dirs[0]

        # Check workflow.json
        wf_info = json.loads((wf_dir / "workflow.json").read_text())
        assert wf_info["status"] in ("completed", "failed")

        # Check agent.json
        agent_info = json.loads((wf_dir / "agent.json").read_text())
        assert agent_info["role"] == "analyst"

        # Check conversation.json exists and has phase entries
        conv = json.loads((wf_dir / "conversation.json").read_text())
        assert "plan" in conv

        # Check memory directory
        assert (wf_dir / "memory").exists()
        assert (wf_dir / "memory" / "todo_list.json").exists()
        assert (wf_dir / "memory" / "findings.json").exists()

    @pytest.mark.asyncio
    async def test_no_persist_dir_produces_no_files(self, tmp_path: Path):
        """When persist_dir is not set, no files should be created."""
        llm = MockToolCallingLLM([
            make_text_response("Plan"),
            make_text_response("Act"),
            make_text_response("REVIEW_PASSED"),
            make_text_response("Report"),
        ])
        orch = Orchestrator(llm=llm)  # No persist_dir
        output = await orch.start_workflow(
            task="Test task",
            role="analyst",
            system_prompt="You are a test analyst.",
            model="test-model",
        )
        # tmp_path should have nothing in it
        assert list(tmp_path.iterdir()) == []


# =========================================================================
# Full folder structure verification
# =========================================================================


class TestFullFolderStructure:
    """Verify the complete folder layout for a multi-level agent hierarchy."""

    def test_three_level_hierarchy(self, tmp_path: Path):
        """Root → child → grandchild: each with full file set."""
        # Build the hierarchy
        wf_store = WorkflowFileStore(tmp_path, "wf-1")
        wf_store.save_workflow_info(workflow_id="wf-1", status="running")

        root = wf_store.create_root_store("root-1")
        root.save_agent_info(task_id="root-1", agent_id="ra", role="lead")
        root.save_memory(_make_memory())
        root.save_output(_make_output("root-1", "ra"))
        root.register_child(task_id="child-1", agent_id="ca", role="analyst")

        child = wf_store.create_child_store("root-1", "child-1", "analyst")
        child.save_agent_info(task_id="child-1", agent_id="ca", role="analyst", parent_task_id="root-1", depth=1)
        child.save_memory(AgentWorkingMemory())
        child.save_output(_make_output("child-1", "ca"))
        child.register_child(task_id="grand-1", agent_id="ga", role="researcher")

        grand = wf_store.create_child_store("child-1", "grand-1", "researcher")
        grand.save_agent_info(task_id="grand-1", agent_id="ga", role="researcher", parent_task_id="child-1", depth=2)
        grand.save_memory(AgentWorkingMemory())
        grand.save_output(_make_output("grand-1", "ga"))

        wf_dir = tmp_path / "wf-1"

        # Root level
        assert (wf_dir / "workflow.json").exists()
        assert (wf_dir / "agent.json").exists()
        assert (wf_dir / "output.json").exists()
        assert (wf_dir / "memory" / "todo_list.json").exists()
        assert (wf_dir / "sub_agents.json").exists()

        # Child level
        child_dirs = list((wf_dir / "sub_agents").iterdir())
        assert len(child_dirs) == 1
        child_dir = child_dirs[0]
        assert (child_dir / "agent.json").exists()
        assert (child_dir / "output.json").exists()
        assert (child_dir / "memory" / "todo_list.json").exists()
        assert (child_dir / "sub_agents.json").exists()

        # Grandchild level
        grand_dirs = list((child_dir / "sub_agents").iterdir())
        assert len(grand_dirs) == 1
        grand_dir = grand_dirs[0]
        assert (grand_dir / "agent.json").exists()
        assert (grand_dir / "output.json").exists()
        assert (grand_dir / "memory" / "todo_list.json").exists()

        # Verify hierarchy isolation: root sub_agents.json only has child
        root_subs = json.loads((wf_dir / "sub_agents.json").read_text())
        assert len(root_subs) == 1
        assert root_subs[0]["role"] == "analyst"

        # Child sub_agents.json only has grandchild
        child_subs = json.loads((child_dir / "sub_agents.json").read_text())
        assert len(child_subs) == 1
        assert child_subs[0]["role"] == "researcher"


# =========================================================================
# Edge cases
# =========================================================================


class TestPersistenceEdgeCases:
    def test_special_characters_in_role(self, tmp_path: Path):
        """Role names with special characters should be sanitized."""
        store = AgentFileStore(tmp_path / "agent")
        child = store.create_child_store("risk/analysis\\expert", "abc12345")
        assert child.path.exists()
        # Slashes should be replaced
        assert "/" not in child.path.name
        assert "\\" not in child.path.name

    def test_read_missing_files(self, tmp_path: Path):
        """Reading from a fresh store should return None/empty."""
        store = AgentFileStore(tmp_path / "agent")
        assert store.read_agent_info() is None
        assert store.read_output() is None
        assert store.read_sub_agents() is None
        mem = store.read_memory()
        assert mem["todo_list"] is None
        assert mem["findings"] is None

    def test_overwrite_agent_info(self, tmp_path: Path):
        """Saving agent info twice should overwrite cleanly."""
        store = AgentFileStore(tmp_path / "agent")
        store.save_agent_info(task_id="t1", agent_id="a1", role="analyst", status="running")
        store.save_agent_info(task_id="t1", agent_id="a1", role="analyst", status="completed")
        data = store.read_agent_info()
        assert data["status"] == "completed"


# =========================================================================
# save_memory — None-safe guards
# =========================================================================


class TestSaveMemoryNoneGuards:
    """Ensure save_memory handles None attributes without crashing."""

    def test_none_review_checklist(self, tmp_path: Path):
        """save_memory must not crash when review_checklist is None."""
        store = AgentFileStore(tmp_path / "agent")
        mem = AgentWorkingMemory()
        mem.todo_list = [TodoItem(index=0, description="Do something")]
        mem.findings = [Finding(category="test", content="data", source="src")]
        mem.review_checklist = None  # Explicitly None (REVIEW skipped)
        mem.submitted_report = None
        store.save_memory(mem)  # Must not raise TypeError

        review_data = _read_json(tmp_path / "agent" / "memory" / "review.json")
        assert review_data == []

    def test_none_todo_list(self, tmp_path: Path):
        """save_memory must not crash when todo_list is None."""
        store = AgentFileStore(tmp_path / "agent")
        mem = AgentWorkingMemory()
        mem.todo_list = None
        mem.findings = []
        mem.review_checklist = None
        mem.submitted_report = None
        store.save_memory(mem)  # Must not raise

        todo_data = _read_json(tmp_path / "agent" / "memory" / "todo_list.json")
        assert todo_data == []

    def test_none_findings(self, tmp_path: Path):
        """save_memory must not crash when findings is None."""
        store = AgentFileStore(tmp_path / "agent")
        mem = AgentWorkingMemory()
        mem.todo_list = []
        mem.findings = None
        mem.review_checklist = None
        mem.submitted_report = None
        store.save_memory(mem)

        findings_data = _read_json(tmp_path / "agent" / "memory" / "findings.json")
        assert findings_data == []

    def test_all_none(self, tmp_path: Path):
        """save_memory with all memory fields None must not crash."""
        store = AgentFileStore(tmp_path / "agent")
        mem = AgentWorkingMemory()
        mem.todo_list = None
        mem.findings = None
        mem.review_checklist = None
        mem.submitted_report = None
        store.save_memory(mem)

        # All files should exist with empty data
        assert _read_json(tmp_path / "agent" / "memory" / "todo_list.json") == []
        assert _read_json(tmp_path / "agent" / "memory" / "findings.json") == []
        assert _read_json(tmp_path / "agent" / "memory" / "review.json") == []
        assert _read_json(tmp_path / "agent" / "memory" / "report.json") is None

    def test_populated_memory_still_works(self, tmp_path: Path):
        """save_memory with fully populated memory still works as before."""
        store = AgentFileStore(tmp_path / "agent")
        mem = _make_memory()
        store.save_memory(mem)

        todo_data = _read_json(tmp_path / "agent" / "memory" / "todo_list.json")
        assert len(todo_data) == 2
        findings_data = _read_json(tmp_path / "agent" / "memory" / "findings.json")
        assert len(findings_data) == 1
        review_data = _read_json(tmp_path / "agent" / "memory" / "review.json")
        assert len(review_data) == 1
