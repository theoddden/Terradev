#!/usr/bin/env python3
"""
Keyword Signal — Sub-millisecond regex/keyword heuristic signal.

Scans query text for keyword patterns that indicate routing-relevant intent.
This is the fastest signal type (<0.1ms) and runs first in the pipeline.
"""

import re
from typing import Any, Dict, List, Set
from .base_signal import BaseSignal, SignalResult, SignalType


# Precompiled keyword sets for O(1) lookup
_CODE_KEYWORDS: Set[str] = {
    "code", "debug", "function", "class", "implement", "refactor",
    "bug", "error", "compile", "syntax", "algorithm", "api",
    "endpoint", "database", "sql", "query", "script", "test",
    "unittest", "deploy", "docker", "kubernetes", "terraform",
    "git", "commit", "merge", "pull request", "ci/cd", "pipeline",
    "python", "javascript", "typescript", "rust", "golang", "java",
    "c++", "html", "css", "react", "node", "fastapi", "flask",
    "django", "pytorch", "tensorflow", "numpy", "pandas",
}

_MATH_KEYWORDS: Set[str] = {
    "calculate", "equation", "integral", "derivative", "matrix",
    "probability", "statistics", "regression", "optimization",
    "proof", "theorem", "formula", "algebra", "calculus",
    "linear", "polynomial", "eigenvalue", "gradient",
}

_CREATIVE_KEYWORDS: Set[str] = {
    "write", "story", "poem", "creative", "fiction", "essay",
    "blog", "article", "narrative", "character", "dialogue",
    "metaphor", "brainstorm", "imagine", "design",
}

_REASONING_KEYWORDS: Set[str] = {
    "explain", "analyze", "compare", "evaluate", "reason",
    "think step by step", "chain of thought", "let's think",
    "why", "how does", "what if", "trade-off", "pros and cons",
}

_VISION_KEYWORDS: Set[str] = {
    "image", "picture", "photo", "screenshot", "diagram",
    "chart", "graph", "visual", "look at", "describe this image",
    "what's in this", "ocr", "scan",
}

# Patterns that indicate specific routing needs
_CODE_BLOCK_PATTERN = re.compile(r'```[\s\S]*?```')
_URL_PATTERN = re.compile(r'https?://\S+')
_FILE_PATH_PATTERN = re.compile(r'(?:/[\w.-]+)+(?:\.\w+)?|[\w.-]+\.(?:py|js|ts|go|rs|java|cpp|h|yaml|json|toml|sql)')


class KeywordSignal(BaseSignal):
    """
    Extracts keyword-based routing signals from query text.

    Returns a dict with:
      - "tags": set of matched keyword categories
      - "has_code_block": bool
      - "has_url": bool
      - "has_file_path": bool
      - "dominant_category": str — the strongest keyword match
    """

    def __init__(self, custom_keywords: Dict[str, Set[str]] = None, enabled: bool = True):
        super().__init__(name="keyword", signal_type=SignalType.KEYWORD, enabled=enabled)
        self.keyword_sets = {
            "code": _CODE_KEYWORDS,
            "math": _MATH_KEYWORDS,
            "creative": _CREATIVE_KEYWORDS,
            "reasoning": _REASONING_KEYWORDS,
            "vision": _VISION_KEYWORDS,
        }
        if custom_keywords:
            self.keyword_sets.update(custom_keywords)

    def extract(self, query: Dict[str, Any]) -> SignalResult:
        content = self._get_content(query).lower()
        words = set(re.findall(r'\b\w+(?:[/+.-]\w+)*\b', content))

        # Count matches per category
        scores: Dict[str, int] = {}
        for category, kw_set in self.keyword_sets.items():
            hits = words & kw_set
            if hits:
                scores[category] = len(hits)

        # Structural pattern checks
        has_code_block = bool(_CODE_BLOCK_PATTERN.search(content))
        has_url = bool(_URL_PATTERN.search(content))
        has_file_path = bool(_FILE_PATH_PATTERN.search(content))

        if has_code_block:
            scores["code"] = scores.get("code", 0) + 3

        tags = set(scores.keys())
        dominant = max(scores, key=scores.get) if scores else "general"

        return SignalResult(
            signal_type=SignalType.KEYWORD,
            name=self.name,
            value={
                "tags": tags,
                "has_code_block": has_code_block,
                "has_url": has_url,
                "has_file_path": has_file_path,
                "dominant_category": dominant,
                "scores": scores,
            },
            confidence=min(1.0, max(scores.values()) / 3.0) if scores else 0.1,
            metadata={"word_count": len(words)},
        )

    @staticmethod
    def _get_content(query: Dict[str, Any]) -> str:
        """Extract text content from query dict"""
        if "content" in query:
            return query["content"]
        if "messages" in query:
            parts = []
            for msg in query["messages"]:
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
