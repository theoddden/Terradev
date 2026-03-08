#!/usr/bin/env python3
"""
Base signal interface for the Terradev semantic signal extraction layer.

Each signal extracts routing-relevant information from a raw query,
reducing H(M | signals) toward zero per the entropy-collapse model.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal categories for semantic routing signal extraction"""
    KEYWORD = "keyword"
    MODALITY = "modality"
    COMPLEXITY = "complexity"
    DOMAIN = "domain"
    LANGUAGE = "language"
    SAFETY = "safety"
    EMBEDDING = "embedding"
    LENGTH = "length"
    COST_HINT = "cost_hint"
    USER_TIER = "user_tier"
    LATENCY_BUDGET = "latency_budget"
    CONTEXT_WINDOW = "context_window"
    CUSTOM = "custom"


@dataclass
class SignalResult:
    """Result of a single signal extraction"""
    signal_type: SignalType
    name: str
    value: Any                          # bool, str, float, dict — depends on signal
    confidence: float = 1.0             # 0.0–1.0
    latency_ms: float = 0.0            # extraction wall-clock time
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_bool(self) -> bool:
        """Coerce value to boolean for decision engine"""
        if isinstance(self.value, bool):
            return self.value
        if isinstance(self.value, (int, float)):
            return self.value > 0.5
        if isinstance(self.value, str):
            return bool(self.value)
        return bool(self.value)

    def as_float(self) -> float:
        """Coerce value to float for scoring"""
        if isinstance(self.value, (int, float)):
            return float(self.value)
        if isinstance(self.value, bool):
            return 1.0 if self.value else 0.0
        return 0.0

    def as_str(self) -> str:
        """Coerce value to string for matching"""
        return str(self.value)


class BaseSignal(ABC):
    """
    Abstract base class for all signal extractors.

    Subclasses implement `extract()` which takes a query dict and returns
    a SignalResult. The base class handles timing and error wrapping.
    """

    def __init__(self, name: str, signal_type: SignalType, enabled: bool = True):
        self.name = name
        self.signal_type = signal_type
        self.enabled = enabled

    def run(self, query: Dict[str, Any]) -> Optional[SignalResult]:
        """Run signal extraction with timing and error handling"""
        if not self.enabled:
            return None

        start = time.perf_counter()
        try:
            result = self.extract(query)
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.latency_ms = elapsed_ms
            return result
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning(f"Signal {self.name} failed in {elapsed_ms:.2f}ms: {e}")
            return SignalResult(
                signal_type=self.signal_type,
                name=self.name,
                value=None,
                confidence=0.0,
                latency_ms=elapsed_ms,
                metadata={"error": str(e)},
            )

    @abstractmethod
    def extract(self, query: Dict[str, Any]) -> SignalResult:
        """
        Extract signal from query.

        Args:
            query: Dict with at minimum:
                - "content": str — the raw query text
                - "messages": list[dict] — optional chat messages
                - "model": str — optional requested model
                - "images": list — optional image attachments
                - "metadata": dict — optional user/session metadata

        Returns:
            SignalResult with extracted signal value
        """
        ...
