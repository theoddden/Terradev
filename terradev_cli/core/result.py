#!/usr/bin/env python3
"""Terradev structured result and error types.

These dataclasses are the long-term contract between the Terradev execution
kernel, the CLI, the MCP server, CI runners, and any future agent protocol.
The schema is intentionally minimal and stable: a result is either a success
with a payload, or a failure with one or more machine-readable errors.

Versioning note: the top-level envelope (``version``) follows a
``YYYY.N`` scheme and is bumped only when a backwards-incompatible change
is made (field removed, renamed, or retyped).  New fields may be added
without a version bump.  Last bumped: 2026.1 (schema stabilised post-v6).
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ErrorCategory(str, Enum):
    """Stable error category taxonomy for agent reasoning."""

    PROVISION = "provision"
    AUTH = "auth"
    VALIDATION = "validation"
    NETWORK = "network"
    RESOURCE = "resource"
    USER = "user"
    INTERNAL = "internal"
    POLICY = "policy"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Stable severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class ErrorCode(str, Enum):
    """Common stable error codes. New codes can be added; existing ones do not change meaning."""

    UNKNOWN = "UNKNOWN"
    PROVISION_NO_QUOTES = "PROVISION_NO_QUOTES"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    AUTH_MISSING_CREDENTIALS = "AUTH_MISSING_CREDENTIALS"
    AUTH_INVALID_CREDENTIALS = "AUTH_INVALID_CREDENTIALS"
    VALIDATION_INVALID_INPUT = "VALIDATION_INVALID_INPUT"
    VALIDATION_BUDGET_EXCEEDED = "VALIDATION_BUDGET_EXCEEDED"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    NETWORK_UNREACHABLE = "NETWORK_UNREACHABLE"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    POLICY_DENIED = "POLICY_DENIED"
    DAG_NODE_FAILED = "DAG_NODE_FAILED"
    COMMAND_NOT_FOUND = "COMMAND_NOT_FOUND"
    CLI_PARSE_ERROR = "CLI_PARSE_ERROR"


@dataclass
class TerradevError:
    """A single, machine-readable error or warning.

    This is the only way the execution kernel communicates failure to agents.
    Free-form exception strings are converted to ``TerradevError`` at the
    boundary so that callers can act on ``code``, ``recoverable``,
    ``retryable``, and ``suggested_action``.
    """

    code: ErrorCode = ErrorCode.UNKNOWN
    message: str = ""
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: Severity = Severity.ERROR
    recoverable: bool = False
    retryable: bool = False
    suggested_action: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    span_id: Optional[str] = None
    trace_id: Optional[str] = None
    exception_type: Optional[str] = None
    exception_traceback: Optional[str] = None

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        code: ErrorCode = ErrorCode.UNKNOWN,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: Severity = Severity.ERROR,
        recoverable: bool = False,
        retryable: bool = False,
        suggested_action: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> "TerradevError":
        return cls(
            code=code,
            message=str(exc),
            category=category,
            severity=severity,
            recoverable=recoverable,
            retryable=retryable,
            suggested_action=suggested_action,
            trace_id=trace_id,
            span_id=span_id,
            exception_type=type(exc).__name__,
            exception_traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class TerradevResult:
    """Stable result envelope for every Terradev operation.

    The ``result`` field holds the operation-specific payload. All other fields
    are generic metadata. CLI, MCP, and future adapters are responsible for
    translating this envelope into their wire format (JSON, OTel, MCP text,
    etc.).
    """

    version: str = "2026.1"
    success: bool = True
    result: Dict[str, Any] = field(default_factory=dict)
    errors: List[TerradevError] = field(default_factory=list)
    warnings: List[TerradevError] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    trace_id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: Optional[str] = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: Optional[str] = None
    command: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float = 0.0
    dry_run: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        code: ErrorCode = ErrorCode.UNKNOWN,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        recoverable: bool = False,
        retryable: bool = False,
        suggested_action: Optional[str] = None,
        command: Optional[str] = None,
    ) -> "TerradevResult":
        error = TerradevError.from_exception(
            exc,
            code=code,
            category=category,
            recoverable=recoverable,
            retryable=retryable,
            suggested_action=suggested_action,
        )
        return cls(
            success=False,
            errors=[error],
            command=command,
        )

    def add_error(
        self,
        code: ErrorCode = ErrorCode.UNKNOWN,
        message: str = "",
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: Severity = Severity.ERROR,
        recoverable: bool = False,
        retryable: bool = False,
        suggested_action: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        exception_type: Optional[str] = None,
        exception_traceback: Optional[str] = None,
    ) -> "TerradevResult":
        self.errors.append(
            TerradevError(
                code=code,
                message=message,
                category=category,
                severity=severity,
                recoverable=recoverable,
                retryable=retryable,
                suggested_action=suggested_action,
                context=context or {},
                exception_type=exception_type,
                exception_traceback=exception_traceback,
            )
        )
        self.success = False
        return self

    def add_warning(
        self,
        code: ErrorCode = ErrorCode.UNKNOWN,
        message: str = "",
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        suggested_action: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> "TerradevResult":
        self.warnings.append(
            TerradevError(
                code=code,
                message=message,
                category=category,
                severity=Severity.WARNING,
                recoverable=True,
                retryable=False,
                suggested_action=suggested_action,
                context=context or {},
            )
        )
        return self

    def add_message(self, message: str) -> "TerradevResult":
        self.messages.append(message)
        return self

    def set_result(self, key: str, value: Any) -> "TerradevResult":
        self.result[key] = value
        return self

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["errors"] = [e.to_dict() for e in self.errors]
        data["warnings"] = [e.to_dict() for e in self.warnings]
        return data

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str, sort_keys=True)

    def human_str(self) -> str:
        lines = []
        status = "OK" if self.success else "ERROR"
        lines.append(f"[{status}] {self.command or 'terradev'}")
        for msg in self.messages:
            lines.append(f"  {msg}")
        if self.result:
            for key, value in sorted(self.result.items()):
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, default=str)
                lines.append(f"  {key}: {value}")
        for err in self.errors:
            lines.append(f"  [{err.severity.upper()}] {err.code}: {err.message}")
            if err.suggested_action:
                lines.append(f"      Action: {err.suggested_action}")
        return "\n".join(lines)
