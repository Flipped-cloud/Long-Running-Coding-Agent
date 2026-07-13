from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["static", "adaptive", "adaptive_search"], required=True)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()

    state = read_json(project_dir / "project_state.json")
    events = read_jsonl(project_dir / "project_events.jsonl")
    sessions = read_jsonl(project_dir / "sessions.jsonl")
    metrics = read_json(project_dir / "project_metrics.json")
    revisions = [read_json(path) for path in sorted((project_dir / "plan_revisions").glob("*.json"))]
    final_verification_path = project_dir / "final_verification.txt"

    if state["status"] != "candidate_complete":
        raise AssertionError(f"project is not candidate_complete: {state['status']}")
    if len(revisions) != len(state["revisions"]):
        raise AssertionError("revision artifact count does not match project_state")
    revisions_by_id = {revision["revision_id"]: revision for revision in revisions}
    state_revisions_by_id = {revision["revision_id"]: revision for revision in state["revisions"]}
    if revisions_by_id != state_revisions_by_id:
        raise AssertionError("revision artifacts are not consistent with project_state")
    if not sessions or not all(session.get("tool_call_count", 0) > 0 for session in sessions):
        raise AssertionError("sessions did not record tool calls")
    if not any(session.get("terminal_signal") == "completion_request" for session in sessions):
        raise AssertionError("no completion signal recorded in sessions")
    if not any(session.get("run_status") == "completed" for session in sessions):
        raise AssertionError("no completed agent run recorded")
    if metrics["sessions_without_terminal_signal"] != sum(1 for session in sessions if not session.get("terminal_signal")):
        raise AssertionError("sessions_without_terminal_signal is not derived from sessions")
    if metrics["total_tool_calls"] != sum(session["tool_call_count"] for session in sessions):
        raise AssertionError("total_tool_calls is not derived from sessions")
    if not any(event["event_type"] == "initial_plan_generated" for event in events):
        raise AssertionError("initial plan event missing")

    if args.mode == "adaptive":
        required = {"task_blocked", "decomposition_generated", "parent_task_aggregated"}
        if not required.issubset({event["event_type"] for event in events}):
            raise AssertionError("adaptive trajectory events missing")
    if args.mode == "adaptive_search":
        required = {"recovery_candidates_generated", "recovery_candidate_rejected", "recovery_candidate_selected"}
        if not required.issubset({event["event_type"] for event in events}):
            raise AssertionError("adaptive_search recovery events missing")
        task_keys = {task["key"] for task in state["tasks"]}
        if "UNSELECTED" in task_keys or "BAD" in task_keys:
            raise AssertionError("unselected recovery candidates leaked into ProjectState")

    if state["status"] == "candidate_complete":
        touched = {path for session in sessions for path in session.get("files_touched", [])}
        if not touched:
            raise AssertionError("candidate_complete project has no touched files")
        if not any(event["event_type"] == "task_completion_requested" for event in events):
            raise AssertionError("candidate_complete project lacks completion artifact")
        if not final_verification_path.exists():
            raise AssertionError("candidate_complete project lacks final_verification.txt")
        if "exit_code: 0" not in final_verification_path.read_text(encoding="utf-8"):
            raise AssertionError("candidate_complete project final verification did not pass")
        if metrics.get("final_verification_passed") is not True:
            raise AssertionError("candidate_complete metrics do not record successful final verification")

    print(json.dumps({"status": "ok", "project_id": state["project_id"], "metrics": metrics}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
