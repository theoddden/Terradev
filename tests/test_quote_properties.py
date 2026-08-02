"""Property-style / randomized tests for TerradevEngine quote logic."""

import random
from unittest.mock import MagicMock, patch

import pytest

from terradev_cli.core.terradev_engine import InstanceQuote, TerradevEngine


@pytest.fixture
def engine(tmp_path):
    """Engine with a temp dataset stager and no live providers."""
    cfg = MagicMock()
    cfg.get_enabled_providers.return_value = []
    cfg.get_provider_reliability.return_value = 0.8
    auth = MagicMock()

    with patch("terradev_cli.core.terradev_engine.ProviderFactory"), patch(
        "terradev_cli.core.terradev_engine.ProviderRegistry"
    ):
        engine = TerradevEngine(config=cfg, auth=auth)
        engine.providers = {}
        return engine


class TestOptimizationScoreInvariants:
    def test_score_between_0_and_1(self, engine):
        for _ in range(100):
            price = random.uniform(0.1, 20.0)
            latency = random.uniform(1, 2000)
            quote = {
                "price_per_hour": price,
                "available": random.choice([True, False]),
                "latency_ms": latency,
                "provider": "runpod",
            }
            score = engine._calculate_optimization_score(quote)
            assert 0.0 <= score <= 1.0

    def test_cheaper_is_better(self, engine):
        cheap = {"price_per_hour": 1.0, "available": True, "latency_ms": 50, "provider": "runpod"}
        expensive = {"price_per_hour": 5.0, "available": True, "latency_ms": 50, "provider": "runpod"}
        assert engine._calculate_optimization_score(cheap) > engine._calculate_optimization_score(expensive)

    def test_faster_is_better(self, engine):
        fast = {"price_per_hour": 1.0, "available": True, "latency_ms": 10, "provider": "runpod"}
        slow = {"price_per_hour": 1.0, "available": True, "latency_ms": 500, "provider": "runpod"}
        assert engine._calculate_optimization_score(fast) > engine._calculate_optimization_score(slow)

    def test_available_is_better(self, engine):
        yes = {"price_per_hour": 1.0, "available": True, "latency_ms": 50, "provider": "runpod"}
        no = {"price_per_hour": 1.0, "available": False, "latency_ms": 50, "provider": "runpod"}
        assert engine._calculate_optimization_score(yes) > engine._calculate_optimization_score(no)


class TestFilterAndRankInvariants:
    def _make_quote(self, price=1.0, available=True, score=0.5):
        return InstanceQuote(
            provider="runpod",
            instance_type="h100",
            gpu_type="H100",
            price_per_hour=price,
            region="us-east-1",
            available=available,
            latency_ms=50,
            optimization_score=score,
            metadata={},
        )

    def test_filter_respects_max_price(self, engine):
        quotes = [self._make_quote(price=1.0), self._make_quote(price=5.0), self._make_quote(price=10.0)]
        filtered = engine._filter_quotes(quotes, max_price=4.9)
        assert all(q.price_per_hour <= 4.9 for q in filtered)

    def test_filter_excludes_unavailable(self, engine):
        quotes = [self._make_quote(available=True), self._make_quote(available=False)]
        filtered = engine._filter_quotes(quotes, max_price=None)
        assert all(q.available for q in filtered)

    def test_rank_is_descending(self, engine):
        random.seed(42)
        quotes = [self._make_quote(score=random.random()) for _ in range(50)]
        ranked = engine._rank_quotes(quotes)
        scores = [q.optimization_score for q in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_ranked_is_best(self, engine):
        quotes = [
            self._make_quote(price=10.0, score=0.1),
            self._make_quote(price=1.0, score=0.9),
            self._make_quote(price=5.0, score=0.5),
        ]
        ranked = engine._rank_quotes(quotes)
        assert ranked[0].optimization_score == 0.9


class TestCostAnalysisInvariants:
    def _make_instance(self, price):
        from terradev_cli.core.terradev_engine import ProvisionedInstance, ProvisioningStatus
        from datetime import datetime
        return ProvisionedInstance(
            instance_id=f"mock-{price}",
            provider="runpod",
            instance_type="h100",
            gpu_type="H100",
            price_per_hour=price,
            region="us-east-1",
            status=ProvisioningStatus.RUNNING,
            created_at=datetime.now(),
            metadata={},
        )

    def test_total_cost_is_sum(self, engine):
        instances = [self._make_instance(1.5), self._make_instance(2.5)]
        result = engine._analyze_costs(instances)
        assert result["total_cost_per_hour"] == 4.0
        assert result["instance_count"] == 2

    def test_savings_never_negative(self, engine):
        for _ in range(50):
            prices = [random.uniform(0.1, 5.0) for _ in range(random.randint(1, 8))]
            instances = [self._make_instance(p) for p in prices]
            result = engine._analyze_costs(instances)
            assert result["estimated_savings"] >= 0
            assert 0 <= result["estimated_savings_percent"] <= 100
