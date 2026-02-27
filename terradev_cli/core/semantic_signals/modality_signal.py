#!/usr/bin/env python3
"""
Modality Signal — Detects the primary modality of a request.

Classifies queries into: text, code, vision, multimodal, embedding, diffusion.
Sub-millisecond heuristic — no neural inference required.
"""

import re
from typing import Any, Dict
from .base_signal import BaseSignal, SignalResult, SignalType


class Modality:
    TEXT = "text"
    CODE = "code"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    EMBEDDING = "embedding"
    DIFFUSION = "diffusion"


# Heuristic patterns
_CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')
_INLINE_CODE_RE = re.compile(r'`[^`]+`')
_FUNCTION_DEF_RE = re.compile(r'\b(?:def|function|fn|func|class|struct|impl|module)\s+\w+')
_IMPORT_RE = re.compile(r'\b(?:import|from|require|include|using)\s+\w+')
_IMAGE_URL_RE = re.compile(r'https?://\S+\.(?:png|jpg|jpeg|gif|webp|svg|bmp|tiff)', re.I)
_DIFFUSION_RE = re.compile(r'\b(?:generate|create|draw|paint|render|make)\s+(?:an?\s+)?(?:image|picture|photo|illustration|art)', re.I)
_EMBEDDING_RE = re.compile(r'\b(?:embed|embedding|vectorize|encode|similarity|cosine)\b', re.I)


class ModalitySignal(BaseSignal):
    """
    Detects the primary modality of an incoming request.

    Returns:
        value: str — one of Modality.{TEXT, CODE, VISION, MULTIMODAL, EMBEDDING, DIFFUSION}
        metadata.modalities: list of all detected modalities
    """

    def __init__(self, enabled: bool = True):
        super().__init__(name="modality", signal_type=SignalType.MODALITY, enabled=enabled)

    def extract(self, query: Dict[str, Any]) -> SignalResult:
        content = self._get_content(query)
        detected = []
        scores: Dict[str, float] = {}

        # Vision: explicit image attachments
        has_images = bool(query.get("images")) or self._has_image_content(query)
        if has_images:
            detected.append(Modality.VISION)
            scores[Modality.VISION] = 1.0

        # Vision: image URLs in text
        if _IMAGE_URL_RE.search(content):
            detected.append(Modality.VISION)
            scores[Modality.VISION] = max(scores.get(Modality.VISION, 0), 0.8)

        # Diffusion: image generation requests
        if _DIFFUSION_RE.search(content):
            detected.append(Modality.DIFFUSION)
            scores[Modality.DIFFUSION] = 0.9

        # Code: code blocks, function defs, imports
        code_signals = sum([
            bool(_CODE_BLOCK_RE.search(content)) * 3,
            bool(_FUNCTION_DEF_RE.search(content)) * 2,
            bool(_IMPORT_RE.search(content)) * 2,
            len(_INLINE_CODE_RE.findall(content)),
        ])
        if code_signals >= 2:
            detected.append(Modality.CODE)
            scores[Modality.CODE] = min(1.0, code_signals / 5.0)

        # Embedding: embedding/similarity requests
        if _EMBEDDING_RE.search(content):
            detected.append(Modality.EMBEDDING)
            scores[Modality.EMBEDDING] = 0.8

        # Default: text
        if not detected:
            detected.append(Modality.TEXT)
            scores[Modality.TEXT] = 1.0

        # If multiple modalities detected, it's multimodal
        primary_modalities = set(detected)
        if len(primary_modalities) > 1:
            primary = Modality.MULTIMODAL
        else:
            primary = detected[0]

        # If model was explicitly requested, respect it as a hint
        model_hint = query.get("model", "")
        if model_hint:
            if any(k in model_hint.lower() for k in ("vision", "4o", "gemini-pro-vision")):
                primary = Modality.VISION if Modality.VISION in detected else primary
            elif any(k in model_hint.lower() for k in ("code", "deepseek-coder", "codellama")):
                primary = Modality.CODE
            elif any(k in model_hint.lower() for k in ("embed", "e5", "bge")):
                primary = Modality.EMBEDDING
            elif any(k in model_hint.lower() for k in ("dall-e", "sdxl", "flux", "midjourney")):
                primary = Modality.DIFFUSION

        return SignalResult(
            signal_type=SignalType.MODALITY,
            name=self.name,
            value=primary,
            confidence=max(scores.values()) if scores else 0.5,
            metadata={
                "modalities": list(primary_modalities),
                "scores": scores,
                "has_images": has_images,
            },
        )

    @staticmethod
    def _get_content(query: Dict[str, Any]) -> str:
        if "__content__" in query:
            return query["__content__"]
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

    @staticmethod
    def _has_image_content(query: Dict[str, Any]) -> bool:
        """Check if messages contain image_url content parts"""
        for msg in query.get("messages", []):
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            return True
        return False
