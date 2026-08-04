#!/usr/bin/env python3
"""Redis span stream exporter for Terradev.

This module is entirely optional. If ``redis`` is not installed or ``REDIS_URL``
is not set, the exporter silently falls back to the local JSONL trace file.
The exporter writes OpenTelemetry-compatible spans as JSON into Redis Streams
using a single ``span`` field.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional

from .result import TerradevError

logger = logging.getLogger(__name__)


def _get_redis_url() -> Optional[str]:
    return os.environ.get("REDIS_URL") or os.environ.get("TERRADEV_REDIS_URL")


class RedisSpanExporter:
    """Fire-and-forget Redis stream exporter.

    All operations are best-effort: a connection or timeout failure is logged
    as a Terradev error but never raises, so provisioning never blocks on the
    observability sidecar.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        key_prefix: str = "terradev:span",
        connect_timeout: float = 2.0,
        socket_timeout: float = 2.0,
    ):
        self.redis_url = redis_url or _get_redis_url()
        self.key_prefix = key_prefix
        self.connect_timeout = connect_timeout
        self.socket_timeout = socket_timeout
        self._redis = None
        self._client_class = None
        self._available: Optional[bool] = None

    def _load_client(self) -> bool:
        """Lazy-load redis client class; return True if import succeeded."""
        if self._client_class is not None:
            return self._client_class is not False
        try:
            from redis import Redis

            self._client_class = Redis
            return True
        except ImportError:
            self._client_class = False
            logger.debug("redis package not installed; Redis span streams disabled")
            return False

    def _connect(self) -> bool:
        """Create and validate Redis connection."""
        if not self.redis_url:
            self._available = False
            return False
        if not self._load_client():
            self._available = False
            return False
        if self._redis is not None:
            return self._available is not False

        try:
            self._redis = self._client_class.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=self.connect_timeout,
                socket_timeout=self.socket_timeout,
                health_check_interval=30,
            )
            self._redis.ping()
            self._available = True
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Redis span stream unavailable: {e}")
            self._available = False
            self._redis = None
            return False

    def is_available(self) -> bool:
        if self._available is None:
            self._connect()
        return bool(self._available)

    @staticmethod
    def _sanitize_value(value: Any) -> Any:
        """Recursively redact values that look like secrets."""
        if isinstance(value, dict):
            return {k: RedisSpanExporter._sanitize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [RedisSpanExporter._sanitize_value(v) for v in value]
        if isinstance(value, str):
            # Mask common credential patterns and long tokens.
            if re.search(r"(?i)(token|secret|password|key|credential)", value):
                return "[REDACTED]"
            if len(value) > 64 and re.match(r"^[A-Za-z0-9_\-./=]+$", value):
                return value[:4] + "..." + value[-4:]
            return value
        return value

    @staticmethod
    def sanitize_span(span_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of the span with sensitive fields redacted."""
        clean = dict(span_dict)
        if "attributes" in clean:
            clean["attributes"] = RedisSpanExporter._sanitize_value(clean["attributes"])
        if "events" in clean:
            clean["events"] = [
                {
                    **evt,
                    "attributes": RedisSpanExporter._sanitize_value(
                        evt.get("attributes", {})
                    ),
                }
                for evt in clean.get("events", [])
            ]
        return clean

    def _make_stream_key(self, *parts: str) -> str:
        return ":".join([self.key_prefix, *parts])

    def stream_key(self, job: str, version: str, instance_id: str) -> str:
        """Deterministic stream key for an instance."""
        safe_instance = re.sub(r"[^a-zA-Z0-9_.-]", "_", instance_id)
        return self._make_stream_key(job, version, safe_instance)

    def _write(self, stream_key: str, span: Any, operation: str = "append") -> bool:
        if not self.is_available():
            return False

        from .telemetry import get_telemetry

        telemetry = get_telemetry()

        try:
            span_dict = span.to_dict() if hasattr(span, "to_dict") else span
            span_dict = self.sanitize_span(span_dict)
            payload = json.dumps(span_dict, default=str, sort_keys=True)

            if operation == "start":
                self._redis.xadd(stream_key, {"span": payload}, id="*")
                ttl = int(os.environ.get("TERRADEV_REDIS_STREAM_TTL", "86400"))
                self._redis.expire(stream_key, ttl)
            elif operation == "end":
                self._redis.xadd(stream_key, {"span": payload}, id="*")
            elif operation == "append":
                self._redis.xadd(stream_key, {"span": payload}, id="*")
            else:
                self._redis.xadd(stream_key, {"span": payload}, id="*")

            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Redis span stream write failed for {stream_key}: {e}")
            telemetry.log_error(
                TerradevError(
                    code="REDIS_STREAM_WRITE_FAILED",
                    message=f"Failed to write span to {stream_key}: {e}",
                    category="network",
                    severity="warning",
                    recoverable=True,
                    retryable=True,
                    context={"stream_key": stream_key, "operation": operation},
                )
            )
            return False

    def start_stream(self, stream_key: str, span: Any) -> bool:
        """Create stream and write the first span; set TTL."""
        return self._write(stream_key, span, operation="start")

    def append_span(self, stream_key: str, span: Any) -> bool:
        """Append a span to an existing stream."""
        return self._write(stream_key, span, operation="append")

    def end_stream(self, stream_key: str, span: Any, tombstone_ttl: int = 60) -> bool:
        """Append a final span and leave a short-lived tombstone."""
        ok = self._write(stream_key, span, operation="end")
        try:
            if ok and self._redis:
                self._redis.expire(stream_key, max(1, tombstone_ttl))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Redis stream expire failed for {stream_key}: {e}")
        return ok

    def delete_stream(self, stream_key: str) -> bool:
        """Remove a stream. Used on explicit destroy."""
        if not self.is_available():
            return False
        try:
            self._redis.delete(stream_key)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Redis stream delete failed for {stream_key}: {e}")
            return False

    def xread_latest(self, stream_key: str, count: int = 10, block_ms: int = 100) -> list:
        """Read latest entries from a stream (for diagnostics/tests)."""
        if not self.is_available():
            return []
        try:
            return self._redis.xrevrange(stream_key, count=count)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Redis xrevrange failed for {stream_key}: {e}")
            return []

    def refresh_ttl(self, stream_key: str, ttl: int = 86400) -> bool:
        """Refresh the TTL on a stream. Used by heartbeats."""
        if not self.is_available():
            return False
        try:
            self._redis.expire(stream_key, max(1, ttl))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Redis refresh TTL failed for {stream_key}: {e}")
            return False

    def ping_stream(self, stream_key: str) -> bool:
        """Lightweight Redis-side existence check for the stream."""
        if not self.is_available():
            return False
        try:
            return self._redis.exists(stream_key) > 0
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Redis ping_stream failed for {stream_key}: {e}")
            return False

    def is_stream_alive(self, stream_key: str, max_age_seconds: int = 120) -> tuple:
        """Check whether the pod/sequence linked to this stream is still active.

        Uses both Redis TTL and the latest span timestamp. Returns
        (is_alive, seconds_since_last_span, last_span_name).
        """
        if not self.is_available():
            return False, None, None

        try:
            ttl = self._redis.ttl(stream_key)
            latest = self._redis.xrevrange(stream_key, count=1)
            if not latest:
                return ttl > 0, None, None

            _, fields = latest[0]
            span_payload = fields.get("span") or fields.get(b"span")
            if not span_payload:
                return ttl > 0, None, None

            span = json.loads(span_payload)
            end_time_str = span.get("end_time") or span.get("start_time")
            last_name = span.get("name", "unknown")
            if not end_time_str:
                return ttl > 0, None, last_name

            try:
                last_dt = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
                age = (datetime.now(timezone.utc) - last_dt).total_seconds()
                is_alive = (ttl > 0) and (age < max_age_seconds)
                return is_alive, age, last_name
            except (ValueError, TypeError):
                return ttl > 0, None, last_name
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Redis is_stream_alive failed for {stream_key}: {e}")
            return False, None, None
