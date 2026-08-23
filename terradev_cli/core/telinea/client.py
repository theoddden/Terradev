"""Telinea telemetry client — non-blocking, fail-safe push.

All network calls are wrapped so that a missing key, timeout, or cloud outage
can never crash a Terradev deployment. Telemetry is best-effort only.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from .config import TelineaConfig

logger = logging.getLogger(__name__)


class TelineaClient:
    """Fire-and-forget telemetry client.

    Spans and events are queued in memory and flushed by a background daemon
    thread. If the queue fills up, oldest entries are dropped so that a long
    running CLI process cannot grow memory unbounded.
    """

    def __init__(self, config: Optional[TelineaConfig] = None):
        self.config = config or TelineaConfig()
        self._queue: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue(
            maxsize=self.config.max_queue_size
        )
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._dropped = 0

    @property
    def is_configured(self) -> bool:
        return self.config.is_configured

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        if not self.is_configured:
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        """Background thread that batches and flushes queued payloads."""
        while not self._stop.is_set():
            batch: List[Dict[str, Any]] = []
            deadline = time.monotonic() + self.config.flush_interval_seconds
            while len(batch) < self.config.batch_size and time.monotonic() < deadline:
                try:
                    item = self._queue.get(timeout=0.1)
                    if item is None:
                        # Poison pill — flush what we have and exit.
                        break
                    batch.append(item)
                except queue.Empty:
                    continue
            if not batch:
                continue
            self._flush_batch(batch)

    def _flush_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Push a batch to the Telinea ingest endpoint.

        All errors are swallowed. This is the fail-safe rule.
        """
        if not self.config.api_key:
            return

        try:
            payload = {
                "workspace_id": self.config.workspace_id,
                "project_id": self.config.project_id,
                "events": batch,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": self.config.auth_header,
                "X-Telinea-Source": "terradev-cli",
            }
            response = requests.post(
                self.config.ingest_url,
                data=json.dumps(payload, default=str),
                headers=headers,
                timeout=(self.config.connect_timeout, self.config.read_timeout),
            )
            if response.status_code >= 400:
                logger.debug(
                    "Telinea ingest returned %s: %s",
                    response.status_code,
                    response.text[:200],
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Telinea ingest failed: %s", exc)

    def enqueue(self, payload: Dict[str, Any]) -> bool:
        """Enqueue a telemetry payload for asynchronous delivery.

        Returns ``True`` if accepted, ``False`` if dropped because the queue
        is full.
        """
        if not self.is_configured:
            return False
        self._ensure_worker()
        try:
            self._queue.put_nowait(payload)
            return True
        except queue.Full:
            self._dropped += 1
            logger.debug("Telinea queue full; dropped telemetry payload")
            return False

    def flush(self) -> None:
        """Flush the queue synchronously and stop the worker."""
        if self._worker is None:
            return
        self._stop.set()
        # Drain remaining items
        batch: List[Dict[str, Any]] = []
        while not self._queue.empty() and len(batch) < self.config.batch_size:
            try:
                item = self._queue.get_nowait()
                if item is not None:
                    batch.append(item)
            except queue.Empty:
                break
        if batch:
            self._flush_batch(batch)
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self._worker = None

    def close(self) -> None:
        """Stop the worker and release resources."""
        self.flush()


_global_client: Optional[TelineaClient] = None
_global_lock = threading.Lock()


def get_telinea_client(config: Optional[TelineaConfig] = None) -> TelineaClient:
    """Return a shared TelineaClient instance."""
    global _global_client
    with _global_lock:
        if _global_client is None or config is not None:
            _global_client = TelineaClient(config)
        return _global_client
