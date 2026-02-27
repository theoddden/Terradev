#!/usr/bin/env python3
"""
Domain Signal — Classifies query into domain categories for
domain-specific model routing.

Uses keyword-based heuristic classification across common domains.
"""

import re
from typing import Any, Dict, List, Set, Tuple
from .base_signal import BaseSignal, SignalResult, SignalType


_DOMAINS: Dict[str, Set[str]] = {
    "medical": {
        "diagnosis", "symptom", "treatment", "patient", "clinical",
        "medication", "dosage", "disease", "pathology", "radiology",
        "surgery", "therapy", "prognosis", "anesthesia", "oncology",
        "cardiology", "neurology", "pediatric", "pharmaceutical",
        "prescription", "chronic", "acute", "biopsy", "mri", "ct scan",
    },
    "legal": {
        "contract", "clause", "liability", "plaintiff", "defendant",
        "statute", "regulation", "compliance", "tort", "litigation",
        "arbitration", "jurisdiction", "precedent", "deposition",
        "affidavit", "injunction", "subpoena", "indemnify", "warrant",
        "copyright", "patent", "trademark", "gdpr", "hipaa", "soc2",
    },
    "finance": {
        "portfolio", "investment", "stock", "bond", "derivative",
        "hedge", "equity", "dividend", "yield", "volatility",
        "valuation", "balance sheet", "income statement", "cash flow",
        "roi", "irr", "npv", "ebitda", "leverage", "amortization",
        "forex", "cryptocurrency", "bitcoin", "defi", "tokenomics",
    },
    "science": {
        "hypothesis", "experiment", "peer review", "methodology",
        "statistical significance", "control group", "variable",
        "correlation", "causation", "replication", "abstract",
        "journal", "citation", "meta-analysis", "p-value",
        "genome", "protein", "molecular", "quantum", "photon",
    },
    "engineering": {
        "architecture", "infrastructure", "scalability", "load balancer",
        "microservice", "api gateway", "database", "cache", "queue",
        "kubernetes", "docker", "terraform", "ci/cd", "monitoring",
        "latency", "throughput", "availability", "fault tolerance",
        "distributed", "consensus", "sharding", "replication",
    },
    "ml_ai": {
        "model", "training", "inference", "fine-tune", "finetuning",
        "dataset", "feature", "embedding", "transformer", "attention",
        "gradient", "backpropagation", "loss function", "optimizer",
        "batch size", "learning rate", "epoch", "overfitting",
        "regularization", "dropout", "convolution", "recurrent",
        "diffusion", "rlhf", "dpo", "lora", "qlora", "quantization",
        "vllm", "gpu", "tensor", "cuda", "onnx", "triton",
    },
    "education": {
        "teach", "learn", "student", "curriculum", "course",
        "assignment", "grade", "exam", "lecture", "tutorial",
        "explain like", "eli5", "beginner", "intermediate", "advanced",
    },
    "general": set(),  # fallback
}


class DomainSignal(BaseSignal):
    """
    Classifies query into domain categories.

    Returns:
        value: str — primary domain (e.g., "code", "medical", "finance")
        metadata.domains: list of (domain, score) tuples ranked by relevance
    """

    def __init__(self, custom_domains: Dict[str, Set[str]] = None, enabled: bool = True):
        super().__init__(name="domain", signal_type=SignalType.DOMAIN, enabled=enabled)
        self.domains = dict(_DOMAINS)
        if custom_domains:
            for k, v in custom_domains.items():
                if k in self.domains:
                    self.domains[k] = self.domains[k] | v
                else:
                    self.domains[k] = v

    def extract(self, query: Dict[str, Any]) -> SignalResult:
        content = self._get_content(query).lower()
        words = set(re.findall(r'\b\w+(?:[/+.-]\w+)*\b', content))

        scores: Dict[str, int] = {}
        for domain, keywords in self.domains.items():
            if not keywords:
                continue
            hits = words & keywords
            if hits:
                scores[domain] = len(hits)

        if not scores:
            return SignalResult(
                signal_type=SignalType.DOMAIN,
                name=self.name,
                value="general",
                confidence=0.3,
                metadata={"domains": [("general", 0)], "scores": {}},
            )

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = ranked[0][0]
        top_score = ranked[0][1]
        confidence = min(1.0, top_score / 4.0)

        return SignalResult(
            signal_type=SignalType.DOMAIN,
            name=self.name,
            value=primary,
            confidence=round(confidence, 3),
            metadata={
                "domains": ranked[:5],
                "scores": scores,
            },
        )

    @staticmethod
    def _get_content(query: Dict[str, Any]) -> str:
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
