from __future__ import annotations


class LongRunAgentError(Exception):
    """Base exception for runtime failures."""


class ConfigurationError(LongRunAgentError):
    """Raised when configuration is invalid or incomplete."""


class ProviderError(LongRunAgentError):
    """Raised when the model provider cannot produce a response."""


class ProtocolError(LongRunAgentError):
    """Raised when a provider response cannot be normalized."""


class ToolExecutionError(LongRunAgentError):
    """Raised for internal tool execution failures."""


class WorkspaceSecurityError(LongRunAgentError, ValueError):
    """Raised when a requested path escapes the configured workspace."""
