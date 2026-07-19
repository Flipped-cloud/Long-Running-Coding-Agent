from __future__ import annotations


class LongRunAgentError(Exception):
    """Base exception for runtime failures."""


class ConfigurationError(LongRunAgentError):
    """Raised when configuration is invalid or incomplete."""


class KnowledgeMutationViolation(LongRunAgentError):
    """Raised when frozen Memory or Skill records are mutated."""


class ProviderError(LongRunAgentError):
    """Raised when the model provider cannot produce a response."""


class ProtocolError(LongRunAgentError):
    """Raised when a provider response cannot be normalized."""


class ToolArgumentsProtocolError(ProtocolError):
    """Raised when a model emits non-JSON tool arguments."""

    def __init__(self, tool_name: str, parse_error: str, raw_arguments: str):
        super().__init__(f"invalid tool call JSON for {tool_name}: {parse_error}")
        self.tool_name = tool_name
        self.parse_error = parse_error
        self.raw_arguments = raw_arguments


class ToolExecutionError(LongRunAgentError):
    """Raised for internal tool execution failures."""


class StateStoreError(LongRunAgentError):
    """Raised when project state artifacts cannot be read or written safely."""


class WorkspaceSecurityError(LongRunAgentError, ValueError):
    """Raised when a requested path escapes the configured workspace."""
