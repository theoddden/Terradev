#!/usr/bin/env python3
"""
Language Signal — Detects the natural language of a query.

Uses character-set heuristics and common word frequency analysis.
No external dependencies — pure Python implementation for sub-ms latency.
"""

import re
from typing import Any, Dict, Set, Tuple
from .base_signal import BaseSignal, SignalResult, SignalType


# High-frequency function words for language detection
_LANG_MARKERS: Dict[str, Set[str]] = {
    "en": {"the", "is", "are", "was", "were", "have", "has", "been", "will",
            "would", "could", "should", "with", "from", "that", "this", "which",
            "they", "their", "there", "about", "when", "where", "what", "how"},
    "de": {"der", "die", "das", "und", "ist", "ein", "eine", "nicht", "ich",
            "auf", "mit", "sich", "den", "von", "auch", "nach", "wie", "aus",
            "noch", "aber", "bei", "nur", "sind", "kann", "wird", "haben"},
    "fr": {"les", "des", "une", "est", "pas", "que", "pour", "dans", "sur",
            "avec", "son", "qui", "sont", "mais", "tout", "elle", "nous",
            "vous", "ont", "cette", "plus", "bien", "comme", "peut"},
    "es": {"los", "las", "una", "que", "por", "con", "para", "del", "son",
            "pero", "como", "esta", "todo", "tiene", "puede", "cuando",
            "desde", "algo", "muy", "hay", "entre", "sobre", "ella", "ser"},
    "pt": {"que", "para", "com", "uma", "dos", "por", "como", "mais",
            "mas", "tem", "foi", "quando", "muito", "pode", "esta", "nos",
            "isso", "bem", "aqui", "onde", "ainda", "mesmo", "fazer"},
    "it": {"che", "per", "con", "una", "sono", "come", "anche", "questo",
            "dal", "della", "nella", "suo", "tutto", "molto", "essere",
            "stato", "fatto", "hanno", "quando", "dopo", "ancora", "dove"},
    "zh": set(),  # detected via character ranges
    "ja": set(),  # detected via character ranges
    "ko": set(),  # detected via character ranges
    "ar": set(),  # detected via character ranges
    "ru": set(),  # detected via character ranges
}

# Unicode character range patterns
_CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
_HIRAGANA_RE = re.compile(r'[\u3040-\u309f]')
_KATAKANA_RE = re.compile(r'[\u30a0-\u30ff]')
_HANGUL_RE = re.compile(r'[\uac00-\ud7af\u1100-\u11ff]')
_ARABIC_RE = re.compile(r'[\u0600-\u06ff\u0750-\u077f]')
_CYRILLIC_RE = re.compile(r'[\u0400-\u04ff]')
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097f]')


class LanguageSignal(BaseSignal):
    """
    Detects the primary language of a query.

    Returns:
        value: str — ISO 639-1 language code (e.g., "en", "de", "zh")
        metadata: confidence scores per detected language
    """

    def __init__(self, enabled: bool = True):
        super().__init__(name="language", signal_type=SignalType.LANGUAGE, enabled=enabled)

    def extract(self, query: Dict[str, Any]) -> SignalResult:
        content = self._get_content(query)

        if not content.strip():
            return SignalResult(
                signal_type=SignalType.LANGUAGE,
                name=self.name,
                value="en",
                confidence=0.1,
                metadata={"reason": "empty_query"},
            )

        # Phase 1: Character-set detection (CJK, Arabic, Cyrillic, etc.)
        script_lang = self._detect_by_script(content)
        if script_lang:
            return SignalResult(
                signal_type=SignalType.LANGUAGE,
                name=self.name,
                value=script_lang[0],
                confidence=script_lang[1],
                metadata={"method": "script_detection", "script": script_lang[0]},
            )

        # Phase 2: Function-word frequency for Latin-script languages
        words = set(re.findall(r'\b\w+\b', content.lower()))
        scores: Dict[str, float] = {}

        for lang, markers in _LANG_MARKERS.items():
            if not markers:
                continue
            hits = words & markers
            if hits:
                scores[lang] = len(hits) / len(markers)

        if not scores:
            return SignalResult(
                signal_type=SignalType.LANGUAGE,
                name=self.name,
                value="en",
                confidence=0.3,
                metadata={"method": "default", "reason": "no_markers_matched"},
            )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = ranked[0][0]
        confidence = min(1.0, ranked[0][1] * 5)  # scale up since marker overlap is small

        # Disambiguate close scores
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.05:
            confidence *= 0.7  # lower confidence when ambiguous

        return SignalResult(
            signal_type=SignalType.LANGUAGE,
            name=self.name,
            value=primary,
            confidence=round(confidence, 3),
            metadata={
                "method": "function_word",
                "scores": {k: round(v, 3) for k, v in ranked[:5]},
            },
        )

    def _detect_by_script(self, text: str) -> Tuple[str, float]:
        """Detect language from non-Latin character scripts"""
        checks = [
            (_HIRAGANA_RE, "ja", 0.95),   # Hiragana is uniquely Japanese
            (_KATAKANA_RE, "ja", 0.90),
            (_HANGUL_RE, "ko", 0.95),
            (_ARABIC_RE, "ar", 0.90),
            (_CYRILLIC_RE, "ru", 0.85),    # Could be Ukrainian, etc.
            (_DEVANAGARI_RE, "hi", 0.85),
            (_CJK_RE, "zh", 0.80),         # CJK could be Chinese or Japanese kanji
        ]

        for pattern, lang, conf in checks:
            matches = pattern.findall(text)
            if len(matches) >= 3:
                return (lang, conf)

        return None

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
