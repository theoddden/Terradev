#!/usr/bin/env python3
"""
Signal Orchestrator — Runs all signal extractors in parallel and produces
a unified SignalVector for the decision engine.

This is the interface between the information-theoretic regime (signal extraction)
and the Boolean-algebraic regime (decision engine / policy evaluation).
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base_signal import BaseSignal, SignalResult, SignalType
from .keyword_signal import KeywordSignal
from .modality_signal import ModalitySignal
from .complexity_signal import ComplexitySignal
from .domain_signal import DomainSignal
from .language_signal import LanguageSignal
from .safety_signal import SafetySignal

try:
    from terradev_cli.core.dag_executor import DAGExecutor
    _HAS_DAG = True
except ImportError:
    _HAS_DAG = False

logger = logging.getLogger(__name__)


@dataclass
class SignalVector:
    """
    The complete signal vector s — the precise interface between
    signal extraction and the decision engine.

    Provides dict-like access for the policy engine's expression evaluator.
    """
    signals: Dict[str, SignalResult] = field(default_factory=dict)
    total_latency_ms: float = 0.0
    timestamp: float = 0.0
    _flat_cache: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __getitem__(self, key: str) -> Any:
        """Access signal values by name: sv['modality'] -> 'code'"""
        if key in self.signals:
            return self.signals[key].value
        raise KeyError(f"Signal '{key}' not found in vector")

    def get(self, key: str, default: Any = None) -> Any:
        """Safe access to signal values"""
        if key in self.signals:
            return self.signals[key].value
        return default

    def get_result(self, key: str) -> Optional[SignalResult]:
        """Get full SignalResult object"""
        return self.signals.get(key)

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Flatten signal vector into a dict suitable for policy expression evaluation.

        Cached after first call — zero cost on repeated access.

        Produces keys like:
          - modality -> "code"
          - complexity -> 0.72
          - safety.flagged -> True
          - safety.severity -> 0.8
          - keyword.dominant_category -> "code"
          - keyword.has_code_block -> True
          - domain -> "engineering"
          - language -> "en"
        """
        if self._flat_cache is not None:
            return self._flat_cache

        flat: Dict[str, Any] = {}

        for name, result in self.signals.items():
            val = result.value

            if isinstance(val, dict):
                # Promote dict values to dotted keys
                for k, v in val.items():
                    if isinstance(v, (str, int, float, bool)):
                        flat[f"{name}.{k}"] = v
                    elif isinstance(v, set):
                        flat[f"{name}.{k}"] = v
                # Also store the dict itself
                flat[name] = val
            else:
                flat[name] = val

            # Always store confidence
            flat[f"{name}.confidence"] = result.confidence

        self._flat_cache = flat
        return flat

    def summary(self) -> Dict[str, Any]:
        """Compact summary for logging / debugging"""
        s = {}
        for name, result in self.signals.items():
            val = result.value
            if isinstance(val, dict):
                # Show the most informative sub-key
                if "flagged" in val:
                    s[name] = f"flagged={val['flagged']}"
                elif "dominant_category" in val:
                    s[name] = val["dominant_category"]
                else:
                    s[name] = str(val)[:60]
            else:
                s[name] = val
            s[f"{name}_ms"] = round(result.latency_ms, 2)
        s["total_ms"] = round(self.total_latency_ms, 2)
        return s


class SignalOrchestrator:
    """
    Orchestrates signal extraction across all registered extractors.

    Default configuration runs the six core signals. Custom signals
    can be added via `add_signal()`.

    Uses Terraform-style DAG parallel execution when available:
    all signals are independent nodes at depth 0, so they execute
    in a single wave with full thread-level parallelism.
    """

    def __init__(self, config: Dict[str, Any] = None, parallel: bool = True):
        self.signals: Dict[str, BaseSignal] = {}
        self._config = config or {}
        self._parallel = parallel and _HAS_DAG
        self._warm_dag: Optional[DAGExecutor] = None  # persistent warm pool
        self._dag_dirty = True  # rebuild DAG when signals change
        self._init_default_signals()

    def _init_default_signals(self):
        """Initialize the six core signal extractors"""
        cfg = self._config

        self.signals["keyword"] = KeywordSignal(
            custom_keywords=cfg.get("keyword_custom"),
            enabled=cfg.get("keyword_enabled", True),
        )
        self.signals["modality"] = ModalitySignal(
            enabled=cfg.get("modality_enabled", True),
        )
        self.signals["complexity"] = ComplexitySignal(
            enabled=cfg.get("complexity_enabled", True),
        )
        self.signals["domain"] = DomainSignal(
            custom_domains=cfg.get("domain_custom"),
            enabled=cfg.get("domain_enabled", True),
        )
        self.signals["language"] = LanguageSignal(
            enabled=cfg.get("language_enabled", True),
        )
        self.signals["safety"] = SafetySignal(
            pii_scan=cfg.get("safety_pii_scan", True),
            jailbreak_scan=cfg.get("safety_jailbreak_scan", True),
            enabled=cfg.get("safety_enabled", True),
        )

    def add_signal(self, name: str, signal: BaseSignal):
        """Register a custom signal extractor"""
        self.signals[name] = signal
        self._dag_dirty = True

    def remove_signal(self, name: str):
        """Remove a signal extractor"""
        self.signals.pop(name, None)
        self._dag_dirty = True

    def _get_warm_dag(self) -> 'DAGExecutor':
        """Get or rebuild the warm DAGExecutor with persistent thread pool."""
        if self._warm_dag is None or self._dag_dirty:
            if self._warm_dag is not None:
                self._warm_dag.shutdown()
            n_signals = sum(1 for s in self.signals.values() if s.enabled)
            dag = DAGExecutor(
                max_workers=max(n_signals, 1),
                name="signals",
                reuse_pool=True,
            )
            for name, signal in self.signals.items():
                if not signal.enabled:
                    continue
                def make_fn(sig):
                    def fn(ctx):
                        return sig.run(ctx.get("__query__", {}))
                    return fn
                dag.add_node(name, make_fn(signal))
            self._warm_dag = dag
            self._dag_dirty = False
        return self._warm_dag

    def shutdown(self):
        """Release the persistent thread pool."""
        if self._warm_dag:
            self._warm_dag.shutdown()
            self._warm_dag = None

    @staticmethod
    def _preprocess_query(query: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text content ONCE and inject into query for all signals.

        Avoids 6 redundant _get_content() calls (one per signal).
        """
        if "__content__" in query:
            return query
        content = ""
        if "content" in query:
            content = query["content"]
        elif "messages" in query:
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
            content = " ".join(parts)
        elif "prompt" in query:
            content = query["prompt"]
        return {**query, "__content__": content}

    def extract(self, query: Dict[str, Any]) -> SignalVector:
        """
        Run all enabled signal extractors on the query.

        Execution modes:
          - parallel (default): Terraform-style DAG — all signals at depth 0,
            executed in a single wave via ThreadPoolExecutor.
          - sequential: fallback when DAG executor is unavailable.

        Total pipeline target: <3ms wall-clock with parallel, <5ms sequential.
        """
        query = self._preprocess_query(query)
        if self._parallel:
            return self._extract_parallel(query)
        return self._extract_sequential(query)

    def _extract_parallel(self, query: Dict[str, Any]) -> SignalVector:
        """DAG-parallel signal extraction using warm persistent pool."""
        dag = self._get_warm_dag()
        result = dag.apply(initial_context={"__query__": query})

        results: Dict[str, SignalResult] = {}
        for name, output in result.outputs.items():
            if output is not None:
                results[name] = output

        vector = SignalVector(
            signals=results,
            total_latency_ms=result.wall_clock_ms,
            timestamp=time.time(),
        )

        logger.debug(
            f"Signal extraction (parallel) in {result.wall_clock_ms:.2f}ms "
            f"(sum={result.total_latency_ms:.2f}ms, "
            f"parallelism={result.parallelism_achieved:.1f}x): "
            f"{len(results)}/{len(self.signals)} signals"
        )

        return vector

    def batch_extract(self, queries: List[Dict[str, Any]]) -> List[SignalVector]:
        """
        Extract signals for N queries in a single batched pass.

        Uses the warm DAG's batch_apply() to share a single thread pool
        across all queries — maximum throughput for MCP batch tool calls.
        """
        if not queries:
            return []

        preprocessed = [self._preprocess_query(q) for q in queries]

        if self._parallel:
            dag = self._get_warm_dag()
            contexts = [{"__query__": q} for q in preprocessed]
            batch_results = dag.batch_apply(contexts)

            vectors = []
            for br in batch_results:
                results: Dict[str, SignalResult] = {}
                for name, output in br.outputs.items():
                    if output is not None:
                        results[name] = output
                vectors.append(SignalVector(
                    signals=results,
                    total_latency_ms=br.wall_clock_ms,
                    timestamp=time.time(),
                ))
            return vectors

        # Sequential fallback
        return [self.extract(q) for q in queries]

    def _extract_sequential(self, query: Dict[str, Any]) -> SignalVector:
        """Sequential fallback when DAG executor is unavailable."""
        start = time.perf_counter()
        results: Dict[str, SignalResult] = {}

        for name, signal in self.signals.items():
            result = signal.run(query)
            if result is not None:
                results[name] = result

        total_ms = (time.perf_counter() - start) * 1000

        vector = SignalVector(
            signals=results,
            total_latency_ms=total_ms,
            timestamp=time.time(),
        )

        logger.debug(
            f"Signal extraction (sequential) in {total_ms:.2f}ms: "
            f"{len(results)}/{len(self.signals)} signals"
        )

        return vector

    def get_enabled_signals(self) -> List[str]:
        """List enabled signal names"""
        return [name for name, sig in self.signals.items() if sig.enabled]

    def __del__(self):
        """Cleanup persistent pool on garbage collection."""
        self.shutdown()
