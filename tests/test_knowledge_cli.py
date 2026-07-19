from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from longrun_agent.cli import app
from longrun_agent.config import load_config
from longrun_agent.knowledge.schema import MemoryKind, MemoryRecord, MemoryScope, MemoryStatus, SkillRecord, SkillStatus, SkillStep
from longrun_agent.knowledge.store import KnowledgeStore


def _write_test_config(tmp_path: Path, *, mode: str) -> Path:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['tests']\n", encoding="utf-8")
    knowledge_root = tmp_path / "knowledge"
    state_root = tmp_path / "projects"
    config_path = tmp_path / f"{mode}.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  provider: fake",
                "  model_name: fake-scripted",
                "workspace:",
                f"  root: {json.dumps(workspace_root.as_posix())}",
                "state:",
                f"  root: {json.dumps(state_root.as_posix())}",
                "knowledge:",
                f"  mode: {mode}",
                f"  root: {json.dumps(knowledge_root.as_posix())}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _store_for_config(config_path: Path) -> KnowledgeStore:
    app_config = load_config(config_path)
    return KnowledgeStore(app_config.knowledge.root, workspace_root=app_config.workspace.root, atomic_write=app_config.state.atomic_write)


def _memory(*, status: MemoryStatus) -> MemoryRecord:
    return MemoryRecord(
        kind=MemoryKind.FAILURE_REFLECTION,
        scope=MemoryScope.REPOSITORY,
        status=status,
        title=f"{status.value} memory",
        lesson="Rerun pytest before reporting completion.",
        recommended_actions=["rerun pytest"],
        avoid_actions=["avoid stale output"],
        applicability_conditions=["pytest validation task"],
        anti_conditions=["documentation only"],
        tags=["pytest"],
        repository_fingerprint="repo-a",
        source_episode_ids=[f"episode-{status.value}"],
        source_task_ids=["task-1"],
        source_session_ids=["session-1"],
        evidence_ids=["e1"],
        confidence=0.8,
    )


def _skill(*, status: SkillStatus) -> SkillRecord:
    return SkillRecord(
        status=status,
        title=f"{status.value} skill",
        summary="Inspect files and rerun pytest.",
        preconditions=["pytest validation task"],
        anti_conditions=["documentation only"],
        procedure=[
            SkillStep(description="Inspect the target file.", tool_name="read_file"),
            SkillStep(description="Run pytest.", tool_name="bash", command="pytest -q"),
        ],
        verification=["pytest -q"],
        tags=["pytest"],
        repository_fingerprint="repo-a",
        source_episode_ids=[f"episode-{status.value}"],
        source_task_ids=["task-1"],
        evidence_ids=["e1"],
        success_count=1,
        distinct_task_ids=["task-1"],
        distinct_repository_fingerprints=["repo-a"],
    )


def _json_output(result) -> object:
    return json.loads(result.output)


def _events(store: KnowledgeStore) -> list[dict]:
    if not store.events_path.exists():
        return []
    return [json.loads(line) for line in store.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_knowledge_memories_list_empty_store(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="verified_memory")
    result = CliRunner().invoke(app, ["knowledge", "memories", "list", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert _json_output(result) == []


def test_knowledge_memories_list_status_filter(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="verified_memory")
    store = _store_for_config(config)
    active = _memory(status=MemoryStatus.ACTIVE)
    quarantined = _memory(status=MemoryStatus.QUARANTINED)
    store.save_memory(active)
    store.save_memory(quarantined)

    result = CliRunner().invoke(app, ["knowledge", "memories", "list", "--config", str(config), "--status", "active"])

    assert result.exit_code == 0, result.output
    payload = _json_output(result)
    assert [item["memory_id"] for item in payload] == [active.memory_id]
    assert payload[0]["status"] == "active"


def test_knowledge_memories_list_invalid_status_is_friendly(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="verified_memory")
    result = CliRunner().invoke(app, ["knowledge", "memories", "list", "--config", str(config), "--status", "does-not-exist"])
    assert result.exit_code != 0
    assert "expected one of" in result.output
    assert "active" in result.output
    assert "Traceback" not in result.output


def test_knowledge_skills_list_empty_store(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="memory_skill")
    result = CliRunner().invoke(app, ["knowledge", "skills", "list", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert _json_output(result) == []


def test_knowledge_skills_list_status_filter(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="memory_skill")
    store = _store_for_config(config)
    validated = _skill(status=SkillStatus.VALIDATED)
    active = _skill(status=SkillStatus.ACTIVE)
    store.save_skill(validated)
    store.save_skill(active)

    result = CliRunner().invoke(app, ["knowledge", "skills", "list", "--config", str(config), "--status", "validated"])

    assert result.exit_code == 0, result.output
    payload = _json_output(result)
    assert [item["skill_id"] for item in payload] == [validated.skill_id]
    assert payload[0]["status"] == "validated"
    assert payload[0]["version"] == 1


def test_knowledge_skills_list_invalid_status_is_friendly(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="memory_skill")
    result = CliRunner().invoke(app, ["knowledge", "skills", "list", "--config", str(config), "--status", "does-not-exist"])
    assert result.exit_code != 0
    assert "expected one of" in result.output
    assert "validated" in result.output
    assert "Traceback" not in result.output


def test_knowledge_memories_show_existing(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="verified_memory")
    store = _store_for_config(config)
    memory = _memory(status=MemoryStatus.ACTIVE)
    store.save_memory(memory)

    result = CliRunner().invoke(app, ["knowledge", "memories", "show", memory.memory_id, "--config", str(config)])

    assert result.exit_code == 0, result.output
    payload = _json_output(result)
    assert payload["memory_id"] == memory.memory_id
    assert payload["status"] == "active"


def test_knowledge_memories_show_missing_is_friendly(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="verified_memory")
    result = CliRunner().invoke(app, ["knowledge", "memories", "show", "missing-memory", "--config", str(config)])
    assert result.exit_code != 0
    assert "memory not found: missing-memory" in result.output
    assert "Traceback" not in result.output


def test_knowledge_memories_invalidate_updates_status_and_writes_one_status_event(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="verified_memory")
    store = _store_for_config(config)
    memory = _memory(status=MemoryStatus.ACTIVE)
    store.save_memory(memory)

    result = CliRunner().invoke(app, ["knowledge", "memories", "invalidate", memory.memory_id, "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert store.load_memory(memory.memory_id).status == MemoryStatus.INVALIDATED
    matching = [event for event in _events(store) if event["event_type"] == "memory_invalidated"]
    assert len(matching) == 1
    assert matching[0]["reason"] == "manually updated through CLI"


def test_knowledge_skills_show_existing(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="memory_skill")
    store = _store_for_config(config)
    skill = _skill(status=SkillStatus.VALIDATED)
    store.save_skill(skill)

    result = CliRunner().invoke(app, ["knowledge", "skills", "show", skill.skill_id, "--config", str(config)])

    assert result.exit_code == 0, result.output
    payload = _json_output(result)
    assert payload["skill_id"] == skill.skill_id
    assert payload["status"] == "validated"


def test_knowledge_skills_show_missing_is_friendly(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="memory_skill")
    result = CliRunner().invoke(app, ["knowledge", "skills", "show", "missing-skill", "--config", str(config)])
    assert result.exit_code != 0
    assert "skill not found: missing-skill" in result.output
    assert "Traceback" not in result.output


def test_knowledge_skills_deprecate_updates_status_and_writes_one_status_event(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="memory_skill")
    store = _store_for_config(config)
    skill = _skill(status=SkillStatus.ACTIVE)
    store.save_skill(skill)

    result = CliRunner().invoke(app, ["knowledge", "skills", "deprecate", skill.skill_id, "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert store.load_skill(skill.skill_id).status == SkillStatus.DEPRECATED
    matching = [event for event in _events(store) if event["event_type"] == "skill_deprecated"]
    assert len(matching) == 1
    assert matching[0]["reason"] == "manually updated through CLI"


def test_knowledge_retrieval_explain_empty_store(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="memory_skill")
    result = CliRunner().invoke(
        app,
        ["knowledge", "retrieval", "explain", "--config", str(config), "--task", "diagnose pytest import mismatch"],
    )
    assert result.exit_code == 0, result.output
    payload = _json_output(result)
    assert payload["memory_ids"] == []
    assert payload["skill_ids"] == []


def test_knowledge_retrieval_explain_rejects_empty_task(tmp_path: Path) -> None:
    config = _write_test_config(tmp_path, mode="memory_skill")
    result = CliRunner().invoke(app, ["knowledge", "retrieval", "explain", "--config", str(config), "--task", "   "])
    assert result.exit_code != 0
    assert "task must not be empty" in result.output
    assert "Traceback" not in result.output
