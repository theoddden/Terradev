#!/usr/bin/env python3
"""
Safety Signal — Detects potentially unsafe, toxic, or policy-violating content.

Implements a lightweight version of the VSR paper's three-stage safety pipeline:
  1. Keyword blocklist scan (sub-ms)
  2. Pattern-based PII detection
  3. Jailbreak attempt heuristics

No neural inference — pure heuristic for speed. Can be extended with
a neural classifier for higher accuracy.
"""

import re
from typing import Any, Dict, List, Set
from .base_signal import BaseSignal, SignalResult, SignalType


# Stage 1: Toxicity keyword patterns
_TOXICITY_PATTERNS = [
    re.compile(r'\b(?:kill|murder|attack|bomb|weapon|explosive|poison)\b', re.I),
    re.compile(r'\b(?:hack|exploit|breach|bypass|crack|phish)\b', re.I),
    re.compile(r'\b(?:illegal|illicit|drugs|narcotic|trafficking)\b', re.I),
]

# Stage 2: PII detection patterns
_PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    "api_key": re.compile(r'\b(?:sk-|pk-|api[_-]?key[=:]\s*)[A-Za-z0-9_-]{20,}\b', re.I),
    "aws_key": re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
}

# Stage 3: Jailbreak attempt patterns
_JAILBREAK_PATTERNS = [
    re.compile(r'\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?)\b', re.I),
    re.compile(r'\bpretend\s+(?:you\s+are|to\s+be)\s+(?:a\s+)?(?:different|new|evil|unrestricted)\b', re.I),
    re.compile(r'\bDAN\s+mode\b|\bdo\s+anything\s+now\b', re.I),
    re.compile(r'\byou\s+(?:are|will)\s+(?:now\s+)?(?:free|unrestricted|unfiltered)\b', re.I),
    re.compile(r'\b(?:bypass|override|disable)\s+(?:your\s+)?(?:safety|filter|restriction|guideline)\b', re.I),
    re.compile(r'\bacting\s+as\s+(?:a\s+)?(?:system|admin|root|developer)\b', re.I),
    re.compile(r'\brole[\s-]?play\s+(?:as\s+)?(?:a\s+)?(?:hacker|criminal|villain)\b', re.I),
]

# Prompt injection patterns
_INJECTION_PATTERNS = [
    re.compile(r'\bsystem:\s', re.I),
    re.compile(r'\b\[INST\]|\b\[/INST\]|\b<<SYS>>|\b<</SYS>>', re.I),
    re.compile(r'\b<\|im_start\|>|\b<\|im_end\|>', re.I),
    re.compile(r'\bHuman:\s|\bAssistant:\s', re.I),
]


class SafetyFlag:
    CLEAN = "clean"
    PII_DETECTED = "pii_detected"
    TOXICITY_DETECTED = "toxicity_detected"
    JAILBREAK_ATTEMPT = "jailbreak_attempt"
    PROMPT_INJECTION = "prompt_injection"


class SafetySignal(BaseSignal):
    """
    Multi-stage safety signal extractor.

    Returns:
        value: dict with:
          - "flagged": bool — whether any safety concern was detected
          - "flags": list of SafetyFlag values
          - "pii_types": list of detected PII types
          - "severity": float 0.0–1.0
    """

    def __init__(self, pii_scan: bool = True, jailbreak_scan: bool = True, enabled: bool = True):
        super().__init__(name="safety", signal_type=SignalType.SAFETY, enabled=enabled)
        self.pii_scan = pii_scan
        self.jailbreak_scan = jailbreak_scan

    def extract(self, query: Dict[str, Any]) -> SignalResult:
        content = self._get_content(query)
        flags: List[str] = []
        pii_types: List[str] = []
        severity = 0.0

        # Stage 1: Toxicity keyword scan
        for pattern in _TOXICITY_PATTERNS:
            if pattern.search(content):
                flags.append(SafetyFlag.TOXICITY_DETECTED)
                severity = max(severity, 0.6)
                break

        # Stage 2: PII detection
        if self.pii_scan:
            for pii_type, pattern in _PII_PATTERNS.items():
                if pattern.search(content):
                    pii_types.append(pii_type)
            if pii_types:
                flags.append(SafetyFlag.PII_DETECTED)
                # API keys and SSNs are more severe
                if any(t in pii_types for t in ("ssn", "credit_card", "api_key", "aws_key")):
                    severity = max(severity, 0.8)
                else:
                    severity = max(severity, 0.4)

        # Stage 3: Jailbreak detection
        if self.jailbreak_scan:
            for pattern in _JAILBREAK_PATTERNS:
                if pattern.search(content):
                    flags.append(SafetyFlag.JAILBREAK_ATTEMPT)
                    severity = max(severity, 0.9)
                    break

            for pattern in _INJECTION_PATTERNS:
                if pattern.search(content):
                    flags.append(SafetyFlag.PROMPT_INJECTION)
                    severity = max(severity, 0.7)
                    break

        flagged = len(flags) > 0
        if not flagged:
            flags.append(SafetyFlag.CLEAN)

        return SignalResult(
            signal_type=SignalType.SAFETY,
            name=self.name,
            value={
                "flagged": flagged,
                "flags": list(set(flags)),
                "pii_types": pii_types,
                "severity": round(severity, 2),
            },
            confidence=0.85 if flagged else 0.95,
            metadata={
                "stages_run": ["toxicity", "pii", "jailbreak"],
                "pii_scan": self.pii_scan,
                "jailbreak_scan": self.jailbreak_scan,
            },
        )

    @staticmethod
    def _get_content(query: Dict[str, Any]) -> str:
        if "content" in query:
            return query["content"]
        if "messages" in query:
            parts = []
            for msg in query.get("messages", []):
                if isinstance(msg, dict):
                    c = msg.get("content", "")
                    if isinstance(c, str):
                        parts.append(c)
                    elif isinstance(c, list):
                        for item in c:
                            if isinstance(item, dict) and item.get("type") == "text":
                                parts.append(item.get("text", ""))
            return " ".join(parts)
        if "prompt" in query:
            return query["prompt"]
        return ""
