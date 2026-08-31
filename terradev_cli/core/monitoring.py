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


    def record_kv_cache(
        self,
        endpoint_id: str,
        engine: str,
        block_size: int,
        total_prompt_tokens: int,
        cached_prompt_tokens: int,
    ) -> None:
        """Record KV cache avoided-token metrics.

        Avoided tokens = total - cached. Unlike hit rate, this cannot be
        gamed by an optimizer that re-requests hot prefixes to inflate hits.
        """
        uncached = max(0, total_prompt_tokens - cached_prompt_tokens)
        tags = {
            "endpoint_id": endpoint_id,
            "engine": engine,
            "block_size": str(block_size),
        }
        self.record("kv.prompt_tokens.total", float(total_prompt_tokens), tags)
        self.record("kv.prompt_tokens.cached", float(cached_prompt_tokens), tags)
        self.record("kv.prompt_tokens.uncached", float(uncached), tags)
        if total_prompt_tokens > 0:
            self.record(
                "kv.cache.uncached_ratio",
                uncached / total_prompt_tokens,
                tags,
            )

    def get_metrics(self) -> List[Dict[str, Any]]:
        return list(self._metrics)

    def reset(self) -> None:
        self._metrics.clear()
