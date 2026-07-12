from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from longrun_agent.config import ToolsConfig
from longrun_agent.protocol import ToolResult


class ToolContext:
    def __init__(
        self,
        workspace: Path,
        tool_outputs_dir: Path | None = None,
        diffs_dir: Path | None = None,
        config: ToolsConfig | None = None,
        artifacts_dir: Path | None = None,
    ):
        artifacts = artifacts_dir or tool_outputs_dir or workspace / ".runs" / "tool_outputs"
        self.workspace = workspace
        self.tool_outputs_dir = artifacts
        self.diffs_dir = diffs_dir or artifacts
        self.config = config or ToolsConfig()
        self.tool_outputs_dir.mkdir(parents=True, exist_ok=True)
        self.diffs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def artifacts_dir(self) -> Path:
        return self.tool_outputs_dir


class BaseTool:
    name: ClassVar[str]
    description: ClassVar[str]
    args_model: ClassVar[type[BaseModel]]

    def input_schema(self) -> dict[str, Any]:
        return self.args_model.model_json_schema()

    def json_schema(self) -> dict[str, Any]:
        return self.input_schema()

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema(),
            },
        }

    def execute(self, call_id: str, arguments: BaseModel, context: ToolContext) -> ToolResult:
        raise NotImplementedError
