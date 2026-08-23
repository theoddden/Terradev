"""Telinea connector that attaches to a NodeSpanStream.

The connector receives every span and command record emitted by the local
TerradevTelemetry system, packages them, and enqueues them for asynchronous
delivery to api.telinea.cloud. It is only active when an API key is present.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..telemetry import get_telemetry
from .client import TelineaClient, get_telinea_client
from .config import TelineaConfig
from .payload import build_stream_payload

logger = logging.getLogger(__name__)


class TelineaConnector:
    """A telemetry stream consumer that forwards spans to Telinea.

    This object is intentionally lightweight: it collects span dicts in memory
    and periodically flushes them via the async ``TelineaClient``. It is safe
    to attach to ``TerradevTelemetry`` because it never blocks and swallows
    all exceptions.
    """

    def __init__(
        self,
        stream: Any,
        config: Optional[TelineaConfig] = None,
        client: Optional[TelineaClient] = None,
    ):
        self.stream = stream
        self.config = config or TelineaConfig()
        self.client = client or get_telinea_client(self.config)
        self._spans: List[Dict[str, Any]] = []
        self._enabled = self.client.is_configured

    @property
    def enabled(self) -> bool:
        return self._enabled and self.client.is_configured

    def append_span(self, span: Any) -> bool:
        """Called by ``TerradevTelemetry._emit_to_streams``."""
        if not self.enabled:
            return False
        try:
            span_dict = span.to_dict() if hasattr(span, "to_dict") else span
            self._spans.append(span_dict)
            if len(self._spans) >= self.config.batch_size:
                self.flush()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("TelineaConnector.append_span failed: %s", exc)
            return False

    def record_command(
        self,
        command: str,
        args: Optional[List[str]] = None,
        success: bool = True,
        returncode: int = 0,
        duration_ms: float = 0.0,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called by ``TerradevTelemetry.record_command_to_active_streams``."""
        if not self.enabled:
            return
        try:
            payload = {
                "type": "command_record",
                "command": command,
                "args": args or [],
                "success": success,
                "returncode": returncode,
                "duration_ms": duration_ms,
                **(attributes or {}),
            }
            self.client.enqueue(payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("TelineaConnector.record_command failed: %s", exc)

    def flush(self, extra: Optional[Dict[str, Any]] = None) -> None:
        """Flush collected spans as a single payload."""
        if not self.enabled or not self._spans:
            return
        try:
            payload = build_stream_payload(
                self.stream,
                list(self._spans),
                config=self.config,
                extra=extra,
            )
            self.client.enqueue(payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("TelineaConnector.flush failed: %s", exc)
        finally:
            self._spans = []

    def close(self, extra: Optional[Dict[str, Any]] = None) -> None:
        """Flush any remaining data and stop the client worker."""
        self.flush(extra=extra)
        try:
            self.client.flush()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Telinea client flush failed: %s", exc)


def maybe_attach_telinea(stream: Any) -> Optional[TelineaConnector]:
    """Attach a TelineaConnector to the telemetry system if configured.

    This is the integration point called by ``NodeSpanStream.start()``. It is
    a no-op when no API key is configured, so the Redis stream stays air-gapped
    by default.
    """
    try:
        config = TelineaConfig()
        if not config.is_configured:
            return None
        connector = TelineaConnector(stream=stream, config=config)
        telemetry = get_telemetry()
        telemetry.attach_stream(connector)
        logger.debug("Telinea connector attached to stream %s", getattr(stream, "stream_key", "?"))
        return connector
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to attach Telinea connector: %s", exc)
        return None
