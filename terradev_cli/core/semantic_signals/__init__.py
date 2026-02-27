#!/usr/bin/env python3
"""
Semantic Signal Extraction for vLLM Semantic Router Integration

Implements the signal extraction layer from the VSR paper:
- Sub-millisecond heuristic signals (keyword, modality, language)
- Lightweight neural classifier signals (complexity, domain, safety)
- Composable signal orchestration via SignalOrchestrator

Reference: vLLM Semantic Router: Signal Driven Decision Routing
           for Mixture-of-Modality Models (Feb 2026)
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
