from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from longrun_agent.verification.contract import private_marker_registry
from longrun_agent.verification.schema import OraclePrivateContract


def check_evaluation_leakage(run_root: Path) -> dict[str, Any]:
    root = run_root.resolve()
    findings = []
    affected = set()
    trials_root = root / "trials"
    for trial in sorted(path for path in trials_root.iterdir() if path.is_dir()) if trials_root.exists() else []:
        private_path = trial / "oracle" / "private" / "contract.json"
        if not private_path.exists():
            continue
        private = OraclePrivateContract.model_validate_json(private_path.read_text(encoding="utf-8"))
        markers = private_marker_registry(private)
        for path in _agent_visible_paths(trial):
            for pointer, text in _text_values(path):
                normalized = text.replace("\\", "/").casefold()
                for marker in markers:
                    if marker.replace("\\", "/").casefold() not in normalized:
                        continue
                    findings.append(
                        {
                            "trial_id": trial.name,
                            "artifact_path": str(path.relative_to(root)),
                            "json_pointer": pointer,
                            "marker_category": _marker_category(marker),
                        }
                    )
                    affected.add(trial.name)
                    break
    report = {
        "leak_count": len(findings),
        "affected_trials": sorted(affected),
        "findings": findings,
        "status": "GO" if not findings else "NO_GO",
    }
    output = root / "leakage_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report["report_path"] = str(output)
    return report


def _agent_visible_paths(trial: Path) -> list[Path]:
    paths = []
    for root in (trial / "telemetry", trial / "state", trial / "artifacts", trial / "knowledge", trial / "workspace"):
        if not root.exists():
            continue
        paths.extend(path for path in root.rglob("*") if path.is_file())
    return [path for path in paths if "private" not in {part.casefold() for part in path.parts}]


def _text_values(path: Path) -> list[tuple[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    values = []
    if path.suffix == ".json":
        try:
            return _walk_json(json.loads(text), "")
        except json.JSONDecodeError:
            return [("/", text)]
    if path.suffix == ".jsonl":
        for line_number, line in enumerate(text.splitlines(), start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                values.append((f"/line/{line_number}", line))
                continue
            values.extend(_walk_json(payload, f"/line/{line_number}" if path.suffix == ".jsonl" else ""))
        return values
    return [("/", text)]


def _walk_json(value: Any, pointer: str) -> list[tuple[str, str]]:
    if isinstance(value, dict):
        return [item for key, child in value.items() for item in _walk_json(child, f"{pointer}/{_escape(key)}")]
    if isinstance(value, list):
        return [item for index, child in enumerate(value) for item in _walk_json(child, f"{pointer}/{index}")]
    return [(pointer or "/", value)] if isinstance(value, str) else []


def _escape(value: Any) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _marker_category(marker: str) -> str:
    normalized = marker.casefold()
    if "hidden" in normalized:
        return "hidden_contract_marker"
    if "oracle" in normalized or "private" in normalized:
        return "private_store_marker"
    return "private_requirement_marker"
