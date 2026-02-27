#!/usr/bin/env python3
"""
Complexity Signal — Estimates query difficulty to route simple queries
to smaller/cheaper models and complex queries to larger/more capable ones.

Uses heuristic features: length, vocabulary richness, structural indicators,
reasoning markers, and domain-specific complexity cues.
"""

import re
import math
from typing import Any, Dict, Set
from .base_signal import BaseSignal, SignalResult, SignalType


# Reasoning complexity markers
_MULTI_STEP_RE = re.compile(
    r'\b(?:step\s*by\s*step|first.*then.*finally|chain\s*of\s*thought|'
    r'let\'?s\s+think|walk\s+(?:me\s+)?through|break\s+(?:it\s+)?down|'
    r'systematically|comprehensive|in\s+depth|detailed\s+analysis)\b', re.I
)

_CONSTRAINT_RE = re.compile(
    r'\b(?:must|shall|require|constraint|condition|ensure|guarantee|'
    r'at\s+least|at\s+most|no\s+more\s+than|exactly|between\s+\d+\s+and)\b', re.I
)

_COMPARISON_RE = re.compile(
    r'\b(?:compare|contrast|difference|versus|vs\.?|trade-?off|'
    r'pros?\s+and\s+cons?|advantages?\s+and\s+disadvantages?)\b', re.I
)

_NESTED_LOGIC_RE = re.compile(
    r'\b(?:if.*then.*else|while.*do|for\s+each|nested|recursive|'
    r'given\s+that.*and.*then|assuming|suppose)\b', re.I
)

# Simple query indicators
_SIMPLE_PATTERNS = [
    re.compile(r'^(?:what|who|when|where)\s+(?:is|are|was|were)\s+', re.I),
    re.compile(r'^(?:define|translate|summarize|list)\s+', re.I),
    re.compile(r'^(?:hi|hello|hey|thanks|thank you|ok|okay)\b', re.I),
]


class ComplexitySignal(BaseSignal):
    """
    Estimates query complexity on a 0.0–1.0 scale.

    Features used:
      - Token count / length
      - Vocabulary richness (type-token ratio)
      - Reasoning markers count
      - Constraint density
      - Code block presence
      - Multi-part question detection
      - Nested logic indicators

    Returns:
        value: float — complexity score 0.0 (trivial) to 1.0 (very complex)
        metadata: breakdown of feature scores
    """

    def __init__(self, enabled: bool = True):
        super().__init__(name="complexity", signal_type=SignalType.COMPLEXITY, enabled=enabled)

    def extract(self, query: Dict[str, Any]) -> SignalResult:
        content = self._get_content(query)

        if not content.strip():
            return SignalResult(
                signal_type=SignalType.COMPLEXITY,
                name=self.name,
                value=0.0,
                confidence=1.0,
                metadata={"reason": "empty_query"},
            )

        features = {}

        # 1. Length score (logarithmic scaling)
        char_count = len(content)
        words = content.split()
        word_count = len(words)
        features["length"] = min(1.0, math.log1p(word_count) / math.log1p(500))

        # 2. Vocabulary richness (type-token ratio)
        unique_words = set(w.lower() for w in words)
        features["vocabulary"] = len(unique_words) / max(word_count, 1)

        # 3. Reasoning markers
        reasoning_hits = len(_MULTI_STEP_RE.findall(content))
        features["reasoning"] = min(1.0, reasoning_hits / 2.0)

        # 4. Constraints
        constraint_hits = len(_CONSTRAINT_RE.findall(content))
        features["constraints"] = min(1.0, constraint_hits / 3.0)

        # 5. Comparison requests
        features["comparison"] = 1.0 if _COMPARISON_RE.search(content) else 0.0

        # 6. Nested logic
        features["nested_logic"] = 1.0 if _NESTED_LOGIC_RE.search(content) else 0.0

        # 7. Code blocks present
        code_blocks = len(re.findall(r'```', content)) // 2
        features["code_blocks"] = min(1.0, code_blocks / 2.0)

        # 8. Multi-part question (question marks, numbered lists)
        question_marks = content.count('?')
        numbered_items = len(re.findall(r'(?:^|\n)\s*\d+[\.\)]\s', content))
        features["multi_part"] = min(1.0, (question_marks + numbered_items) / 4.0)

        # 9. Simple query penalty
        is_simple = any(p.search(content) for p in _SIMPLE_PATTERNS)
        features["simple_penalty"] = -0.3 if is_simple and word_count < 15 else 0.0

        # Weighted combination
        weights = {
            "length": 0.15,
            "vocabulary": 0.05,
            "reasoning": 0.25,
            "constraints": 0.15,
            "comparison": 0.10,
            "nested_logic": 0.10,
            "code_blocks": 0.05,
            "multi_part": 0.10,
            "simple_penalty": 0.05,  # negative contribution
        }

        raw_score = sum(features[k] * weights[k] for k in weights)
        complexity = max(0.0, min(1.0, raw_score))

        # Confidence: higher when features are clearly present or absent
        feature_variance = sum((v - complexity) ** 2 for k, v in features.items() if k != "simple_penalty") / max(len(features) - 1, 1)
        confidence = max(0.5, 1.0 - feature_variance)

        return SignalResult(
            signal_type=SignalType.COMPLEXITY,
            name=self.name,
            value=round(complexity, 3),
            confidence=round(confidence, 3),
            metadata={
                "features": {k: round(v, 3) for k, v in features.items()},
                "word_count": word_count,
                "char_count": char_count,
                "level": self._level(complexity),
            },
        )

    @staticmethod
    def _level(score: float) -> str:
        if score < 0.2:
            return "trivial"
        elif score < 0.4:
            return "simple"
        elif score < 0.6:
            return "moderate"
        elif score < 0.8:
            return "complex"
        else:
            return "expert"

    @staticmethod
    def _get_content(query: Dict[str, Any]) -> str:
        if "__content__" in query:
            return query["__content__"]
        if "content" in query:
            return query["content"]
        if "messages" in query:
            parts = []
            for msg in query.get("messages", []):
                if isinstance(msg, dict) and msg.get("role") != "system":
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
