#!/usr/bin/env python3
"""
Terradev Semantic Signal Extraction Layer

Implements a signal extraction layer for query-level routing decisions:
- Sub-millisecond heuristic signals (keyword, modality, language)
- Lightweight classifier signals (complexity, domain, safety)
- Composable signal orchestration via SignalOrchestrator

Inspired by signal-driven routing concepts from the academic literature.
This is an independent, clean-room implementation.
"""

from .base_signal import BaseSignal, SignalResult, SignalType
from .keyword_signal import KeywordSignal
from .modality_signal import ModalitySignal
from .complexity_signal import ComplexitySignal
from .domain_signal import DomainSignal
from .language_signal import LanguageSignal
from .safety_signal import SafetySignal
from .orchestrator import SignalOrchestrator, SignalVector

__all__ = [
    'BaseSignal',
    'SignalResult',
    'SignalType',
    'KeywordSignal',
    'ModalitySignal',
    'ComplexitySignal',
    'DomainSignal',
    'LanguageSignal',
    'SafetySignal',
    'SignalOrchestrator',
    'SignalVector',
]
