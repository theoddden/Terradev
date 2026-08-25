"""
Minimal MetricsCollector stub so optimization modules can import without error.
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Lightweight metrics collector used by the optimization layer."""

    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self._metrics: List[Dict[str, Any]] = []

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        entry = {"name": name, "value": value, "ts": time.time(), "tags": tags or {}}
        self._metrics.append(entry)
        logger.debug("metric %s=%.4f tags=%s", name, value, entry["tags"])

    def increment(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        self.record(name, 1.0, tags)

    def get_metrics(self) -> List[Dict[str, Any]]:
        return list(self._metrics)

    def reset(self) -> None:
        self._metrics.clear()
