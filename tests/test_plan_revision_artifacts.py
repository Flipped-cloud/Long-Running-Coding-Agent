import json
from pathlib import Path

from longrun_agent.model.fake import FakeModelProvider
from longrun_agent.orchestration.orchestrator import ProjectOrchestrator
from longrun_agent.protocol import ModelResponse, ToolCall
from longrun_agent.state.store import ProjectStateStore
from tests.test_project_orchestrator import completion, config, submit_plan


def test_initial_plan_revision_artifact_matches_state(tmp_path: Path):
    cfg = config(tmp_path, mode="static", max_sessions=1)
    ProjectOrchestrator(cfg, FakeModelProvider([submit_plan(), completion("c1")]), project_id="rev-1").start("ship")
    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    state = store.load("rev-1")
    assert len(state.revisions) == 1
    artifact = json.loads(store.revision_path("rev-1", state.revisions[0].revision_id).read_text(encoding="utf-8"))
    assert artifact == state.revisions[0].model_dump(mode="json")


def test_decomposition_revision_artifact_matches_state(tmp_path: Path):
    cfg = config(tmp_path, mode="adaptive", max_sessions=2)
    responses = [
        submit_plan(),
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="b1",
                    name="report_blocker",
                    arguments={"reason": "split", "attempted_actions": [], "decomposition_recommended": True},
                )
            ]
        ),
        ModelResponse(
            tool_calls=[
                ToolCall(
                    id="d1",
                    name="submit_decomposition",
                    arguments={
                        "parent_task_id": "T1",
                        "reason": "split",
                        "children": [
                            {"key": "A", "title": "A", "objective": "a", "acceptance_criteria": ["done"], "depends_on_child_keys": []},
                            {
                                "key": "B",
                                "title": "B",
                                "objective": "b",
                                "acceptance_criteria": ["done"],
                                "depends_on_child_keys": ["A"],
                            },
                        ],
                    },
                )
            ]
        ),
        completion("ca"),
    ]
    ProjectOrchestrator(cfg, FakeModelProvider(responses), project_id="rev-2").start("ship")
    store = ProjectStateStore(cfg.state.root, workspace_root=cfg.workspace.root)
    state = store.load("rev-2")
    assert len(state.revisions) == 2
    for revision in state.revisions:
        artifact = json.loads(store.revision_path("rev-2", revision.revision_id).read_text(encoding="utf-8"))
        assert artifact == revision.model_dump(mode="json")
