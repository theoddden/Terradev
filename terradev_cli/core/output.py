#!/usr/bin/env python3
"""Terradev output contract for humans, CI, Docker, and agents.

This module provides a single Click-context-aware output object. It collects
messages, warnings, errors, and the final result payload, then emits either
human-readable text or a stable JSON ``TerradevResult`` when the command ends.

No new CLI commands are required for this to work. The ``--format`` option is
added to the existing root group, and the existing ``print`` calls in command
functions are redirected to the output collector while the command is running.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from contextlib import contextmanager
from io import StringIO
from typing import Any, Dict, List, Optional, TextIO

from .result import ErrorCategory, ErrorCode, Severity, TerradevError, TerradevResult


# Sentinel for messages that are not explicitly typed.
class _Message:
    def __init__(self, text: str, level: str = "info"):
        self.text = text
        self.level = level


class TerradevOutput:
    """Context-aware output collector.

    - In human mode (default): messages are written to ``stdout`` as they happen
      and the final result is printed as a short summary.
    - In JSON mode: all messages/results are collected and a single
      ``TerradevResult`` JSON object is emitted at the end. This is the right
      mode for Docker containers and CI/CD pipelines.
    - In JSONL mode: each message/result is emitted as a single JSON line
      (useful for long-running streaming commands).

    The output format is chosen via, in order of precedence:
      1. The ``--format`` CLI option.
      2. The ``TERRADEV_OUTPUT`` environment variable (``human``/``json``/``jsonl``).
      3. Isatty detection (non-TTY defaults to ``json`` when possible).
    """

    FORMATS = {"human", "json", "jsonl"}

    def __init__(
        self,
        format: Optional[str] = None,
        command: Optional[str] = None,
        trace_id: Optional[str] = None,
        request_id: Optional[str] = None,
        stream: Optional[TextIO] = None,
    ):
        self._format = self._resolve_format(format)
        self._command = command
        self._result = TerradevResult(
            trace_id=trace_id,
            request_id=request_id,
            command=command,
        )
        self._messages: List[_Message] = []
        self._lock = threading.RLock()
        self._stream = stream or sys.stdout
        self._original_stdout: Optional[TextIO] = None
        self._redirect_active = False
        self._closed = False

    @staticmethod
    def _resolve_format(explicit: Optional[str]) -> str:
        if explicit and explicit.lower() in TerradevOutput.FORMATS:
            return explicit.lower()
        env = os.environ.get("TERRADEV_OUTPUT", "").strip().lower()
        if env in TerradevOutput.FORMATS:
            return env
        # In a non-TTY environment (Docker/CI) default to JSON for composability.
        if not sys.stdout.isatty():
            return "json"
        return "human"

    @property
    def format(self) -> str:
        return self._format

    @property
    def result(self) -> TerradevResult:
        return self._result

    def set_command(self, command: str) -> "TerradevOutput":
        self._command = command
        self._result.command = command
        return self

    def set_trace_id(self, trace_id: str) -> "TerradevOutput":
        self._result.trace_id = trace_id
        return self

    def set_request_id(self, request_id: str) -> "TerradevOutput":
        self._result.request_id = request_id
        return self

    def set_dry_run(self, dry_run: bool) -> "TerradevOutput":
        self._result.dry_run = dry_run
        return self

    def set_result(self, data: Dict[str, Any]) -> "TerradevOutput":
        """Replace the entire result payload."""
        self._result.result = data
        return self

    def add_result(self, key: str, value: Any) -> "TerradevOutput":
        """Add a single key to the result payload."""
        self._result.result[key] = value
        return self

    def info(self, text: str) -> "TerradevOutput":
        self._emit(_Message(text, "info"))
        return self

    def success(self, text: str) -> "TerradevOutput":
        self._emit(_Message(text, "success"))
        return self

    def warning(self, text: str) -> "TerradevOutput":
        # Log as a structured warning but emit a single user-facing message.
        self.add_warning_result(message=text)
        return self

    def error(self, text: str) -> "TerradevOutput":
        # Log as a structured error and mark the result as failed.
        self.add_error(message=text)
        return self

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
    ) -> "TerradevOutput":
        self._result.add_error(
            code=code,
            message=message,
            category=category,
            severity=severity,
            recoverable=recoverable,
            retryable=retryable,
            suggested_action=suggested_action,
            context=context,
        )
        self._emit(_Message(message, "error"))
        return self

    def add_warning_result(
        self,
        code: ErrorCode = ErrorCode.UNKNOWN,
        message: str = "",
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        suggested_action: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> "TerradevOutput":
        self._result.add_warning(
            code=code,
            message=message,
            category=category,
            suggested_action=suggested_action,
            context=context,
        )
        self._emit(_Message(message, "warning"))
        return self

    def _emit(self, message: _Message) -> None:
        with self._lock:
            self._messages.append(message)
            if self._format == "human":
                if message.level == "error":
                    self._write_raw(f"ERROR: {message.text}\n")
                elif message.level == "warning":
                    self._write_raw(f"WARN: {message.text}\n")
                elif message.level == "success":
                    self._write_raw(f"OK: {message.text}\n")
                else:
                    self._write_raw(f"{message.text}\n")
            elif self._format == "jsonl":
                line = json.dumps({
                    "type": "message",
                    "level": message.level,
                    "text": message.text,
                    "trace_id": self._result.trace_id,
                    "command": self._command,
                }, default=str)
                self._write_raw(line + "\n")
            # In JSON mode, messages are buffered until close().

    def _write_raw(self, text: str) -> None:
        try:
            self._stream.write(text)
            self._stream.flush()
        except Exception:
            pass

    @contextmanager
    def capture_print(self):
        """Temporarily redirect ``print`` to this output collector.

        This lets existing command functions that call ``print()`` participate
        in the composable output contract without an immediate rewrite of every
        command. ``print`` calls become messages; if they start with a known
        prefix (``OK:``, ``ERROR:``, ``WARN:``, etc.) they are typed accordingly.
        """
        original = __builtins__["print"] if isinstance(__builtins__, dict) else __builtins__.print

        def _print(*args, sep=" ", end="\n", file=None, flush=False):
            text = sep.join(str(a) for a in args)
            # Strip known severity prefixes and route through output.
            stripped = text
            for prefix in ["OK:", "OK", "ERROR:", "Error:", "WARN:", "Warning:", "Warning"]:
                if stripped.lstrip().startswith(prefix):
                    stripped = stripped.lstrip()[len(prefix) :].lstrip()
                    break

            if text.lstrip().startswith(("ERROR:", "Error:")):
                self.error(stripped)
            elif text.lstrip().startswith(("WARN:", "Warning:", "Warning")):
                self.warning(stripped)
            elif text.lstrip().startswith(("OK:", "OK")):
                self.success(stripped)
            else:
                self.info(text)

        try:
            if isinstance(__builtins__, dict):
                __builtins__["print"] = _print
            else:
                __builtins__.print = _print
            self._redirect_active = True
            yield
        finally:
            if isinstance(__builtins__, dict):
                __builtins__["print"] = original
            else:
                __builtins__.print = original
            self._redirect_active = False

    def close(self) -> None:
        """Emit the final result and close the output."""
        if self._closed:
            return
        self._closed = True

        with self._lock:
            # Add collected messages to the result envelope.
            self._result.messages = [m.text for m in self._messages]
            # If any error-level message was printed, reflect that in success.
            if any(m.level == "error" for m in self._messages):
                self._result.success = False

            if self._format == "json":
                self._write_raw(self._result.to_json(indent=2) + "\n")
            elif self._format == "jsonl":
                self._write_raw(
                    json.dumps({
                        "type": "result",
                        "success": self._result.success,
                        "result": self._result.result,
                        "errors": [e.to_dict() for e in self._result.errors],
                        "warnings": [e.to_dict() for e in self._result.warnings],
                        "trace_id": self._result.trace_id,
                        "command": self._command,
                    }, default=str)
                    + "\n"
                )
            else:
                # Human mode: final summary already printed inline; nothing extra.
                pass

    def __enter__(self) -> "TerradevOutput":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None:
            if not self._result.errors:
                self._result.add_error(
                    code=ErrorCode.UNKNOWN,
                    message=str(exc),
                    category=ErrorCategory.INTERNAL,
                    severity=Severity.FATAL,
                    exception_type=type(exc).__name__,
                    exception_traceback="".join(traceback.format_exception(exc_type, exc, tb)) if tb else None,
                )
        self.close()


def get_output(ctx: Optional[Any] = None) -> TerradevOutput:
    """Get the current ``TerradevOutput`` from a Click context or create one."""
    if ctx is None:
        try:
            import click
            ctx = click.get_current_context(silent=True)
        except Exception:
            ctx = None
    if ctx is not None and hasattr(ctx, "obj") and ctx.obj and isinstance(ctx.obj, dict):
        if "terradev_output" in ctx.obj and ctx.obj["terradev_output"] is not None:
            return ctx.obj["terradev_output"]
    return TerradevOutput()
