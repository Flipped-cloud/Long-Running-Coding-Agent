from __future__ import annotations

from pydantic import BaseModel, Field

from longrun_agent.knowledge.schema import KnowledgeUseType
from longrun_agent.protocol import ErrorType, ToolResult
from longrun_agent.tools.base import BaseTool, ToolContext


class ReportKnowledgeUseArgs(BaseModel):
    memory_ids: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)
    use_type: KnowledgeUseType = KnowledgeUseType.REFERENCED
    reason: str = Field(min_length=1)


class ReportKnowledgeUseTool(BaseTool):
    name = "report_knowledge_use"
    description = (
        "Record the required Knowledge Decision for retrieved knowledge. If retrieved Memory or Skill IDs materially affected "
        "diagnosis, edits, verification, or completion, report those exposed IDs with use_type='referenced'. If none were used, "
        "report empty memory_ids and skill_ids with a non-empty reason. Helpful or harmful attribution is determined by the harness."
    )
    args_model = ReportKnowledgeUseArgs

    def execute(self, call_id: str, arguments: ReportKnowledgeUseArgs, context: ToolContext) -> ToolResult:
        channel = getattr(context, "knowledge_channel", None)
        if channel is None:
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=False,
                summary="knowledge channel is not configured",
                error_type=ErrorType.TOOL,
                error_message="knowledge channel is not configured",
            )
        if arguments.use_type != KnowledgeUseType.REFERENCED:
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=False,
                summary="knowledge use rejected: only referenced is allowed",
                output="Only use_type='referenced' is accepted; helpful or harmful attribution is assigned by verification evidence.",
                error_type=ErrorType.TOOL,
                error_message="only use_type='referenced' is accepted",
            )
        try:
            channel.record(arguments)
        except ValueError as exc:
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self.name,
                success=False,
                summary=f"knowledge use rejected: {exc}",
                output=str(exc),
                error_type=ErrorType.TOOL,
                error_message=str(exc),
            )
        return ToolResult(
            tool_call_id=call_id,
            tool_name=self.name,
            success=True,
            summary="knowledge decision recorded",
            output="Knowledge decision recorded.",
            metadata={"memory_ids": arguments.memory_ids, "skill_ids": arguments.skill_ids, "use_type": arguments.use_type.value},
        )


class KnowledgeUseChannel:
    def __init__(self, *, exposed_memory_ids: list[str], exposed_skill_ids: list[str]):
        self.exposed_memory_ids = set(exposed_memory_ids)
        self.exposed_skill_ids = set(exposed_skill_ids)
        self.decision_recorded = False
        self.referenced_memory_ids: set[str] = set()
        self.referenced_skill_ids: set[str] = set()
        self.not_used_reason: str | None = None
        self.records: list[ReportKnowledgeUseArgs] = []

    def record(self, args: ReportKnowledgeUseArgs) -> None:
        unknown_memory = [item for item in args.memory_ids if item not in self.exposed_memory_ids]
        unknown_skill = [item for item in args.skill_ids if item not in self.exposed_skill_ids]
        if unknown_memory or unknown_skill:
            raise ValueError(f"only exposed knowledge IDs can be reported; unknown={unknown_memory + unknown_skill}")
        self.decision_recorded = True
        if not args.memory_ids and not args.skill_ids:
            self.not_used_reason = args.reason
            return
        self.referenced_memory_ids.update(args.memory_ids)
        self.referenced_skill_ids.update(args.skill_ids)
        self.records.append(args)

    def has_exposed_knowledge(self) -> bool:
        return bool(self.exposed_memory_ids or self.exposed_skill_ids)
