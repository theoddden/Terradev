"""Tests for TerradevEngine — pure logic mocked, no live providers."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_config(providers=None, reliability=0.8):
    """Return a minimal TerradevConfig mock."""
    cfg = MagicMock()
    cfg.get_enabled_providers.return_value = list(providers or [])
    cfg.get_provider_reliability.return_value = reliability
    return cfg


def _make_auth(creds=None):
    auth = MagicMock()
    auth.get_credentials.return_value = creds or {}
    return auth


def _make_engine(providers=None, reliability=0.8):
    """Construct a TerradevEngine with no real providers initialised."""
    from terradev_cli.core.terradev_engine import TerradevEngine

    cfg = _make_config(providers=[], reliability=reliability)
    auth = _make_auth()

    with patch("terradev_cli.core.terradev_engine.ProviderFactory") as MockFactory, \
         patch("terradev_cli.core.terradev_engine.ProviderRegistry"):
        engine = TerradevEngine(config=cfg, auth=auth)
        engine.providers = providers or {}
        engine.config = cfg
    return engine


# ── Enums / dataclasses ─────────────────────────────────────────────────────

class TestEnumsAndDataclasses:
    def test_provisioning_status_values(self):
        from terradev_cli.core.terradev_engine import ProvisioningStatus
        assert ProvisioningStatus.PENDING.value == "pending"
        assert ProvisioningStatus.RUNNING.value == "running"
        assert ProvisioningStatus.FAILED.value == "failed"
        assert ProvisioningStatus.COMPLETED.value == "completed"

    def test_instance_request_fields(self):
        from terradev_cli.core.terradev_engine import InstanceRequest
        req = InstanceRequest(
            gpu_type="H100", count=2, max_price=5.0,
            region="us-east-1", providers=["runpod"],
            requirements={"memory": 80}
        )
        assert req.gpu_type == "H100"
        assert req.count == 2

    def test_instance_quote_fields(self):
        from terradev_cli.core.terradev_engine import InstanceQuote
        q = InstanceQuote(
            provider="runpod", instance_type="H100_80GB",
            gpu_type="H100", price_per_hour=2.5,
            region="us-east-1", available=True,
            latency_ms=50.0, optimization_score=0.9,
            metadata={}
        )
        assert q.provider == "runpod"
        assert q.price_per_hour == 2.5

    def test_provisioned_instance_fields(self):
        from terradev_cli.core.terradev_engine import ProvisionedInstance, ProvisioningStatus
        inst = ProvisionedInstance(
            instance_id="i-abc",
            provider="vastai",
            instance_type="A100x8",
            gpu_type="A100",
            price_per_hour=1.5,
            region="eu-west-1",
            status=ProvisioningStatus.RUNNING,
            created_at=datetime.now(),
            metadata={}
        )
        assert inst.status == ProvisioningStatus.RUNNING

    def test_provisioning_result_fields(self):
        from terradev_cli.core.terradev_engine import ProvisioningResult
        res = ProvisioningResult(
            success=True, instances=[], cost_analysis={},
            total_time=1.23, errors=[]
        )
        assert res.success is True
        assert res.total_time == 1.23


# ── _calculate_optimization_score ──────────────────────────────────────────

class TestCalculateOptimizationScore:
    def test_available_instance_higher_score(self):
        engine = _make_engine()
        available = {"price_per_hour": 1.0, "available": True, "latency_ms": 50}
        unavailable = {"price_per_hour": 1.0, "available": False, "latency_ms": 50}
        assert engine._calculate_optimization_score(available) > engine._calculate_optimization_score(unavailable)

    def test_cheaper_instance_higher_score(self):
        engine = _make_engine()
        cheap = {"price_per_hour": 1.0, "available": True, "latency_ms": 50}
        expensive = {"price_per_hour": 8.0, "available": True, "latency_ms": 50}
        assert engine._calculate_optimization_score(cheap) > engine._calculate_optimization_score(expensive)

    def test_lower_latency_higher_score(self):
        engine = _make_engine()
        fast = {"price_per_hour": 1.0, "available": True, "latency_ms": 10}
        slow = {"price_per_hour": 1.0, "available": True, "latency_ms": 900}
        assert engine._calculate_optimization_score(fast) > engine._calculate_optimization_score(slow)

    def test_score_between_0_and_1(self):
        engine = _make_engine()
        quote = {"price_per_hour": 5.0, "available": True, "latency_ms": 200}
        score = engine._calculate_optimization_score(quote)
        assert 0.0 <= score <= 1.0

    def test_reliability_factored_in(self):
        engine_high = _make_engine(reliability=1.0)
        engine_low = _make_engine(reliability=0.0)
        quote = {"price_per_hour": 1.0, "available": True, "latency_ms": 50, "provider": "runpod"}
        assert engine_high._calculate_optimization_score(quote) > engine_low._calculate_optimization_score(quote)


# ── _filter_quotes ──────────────────────────────────────────────────────────

class TestFilterQuotes:
    def _make_quote(self, price, available=True):
        from terradev_cli.core.terradev_engine import InstanceQuote
        return InstanceQuote(
            provider="runpod", instance_type="H100", gpu_type="H100",
            price_per_hour=price, region="us-east-1", available=available,
            latency_ms=50, optimization_score=0.8, metadata={}
        )

    def test_filter_by_max_price(self):
        engine = _make_engine()
        quotes = [self._make_quote(1.0), self._make_quote(5.0), self._make_quote(10.0)]
        filtered = engine._filter_quotes(quotes, max_price=4.9)
        assert len(filtered) == 1
        assert filtered[0].price_per_hour == 1.0

    def test_filter_unavailable(self):
        engine = _make_engine()
        quotes = [self._make_quote(1.0, available=True), self._make_quote(1.0, available=False)]
        filtered = engine._filter_quotes(quotes, max_price=None)
        assert len(filtered) == 1
        assert filtered[0].available is True

    def test_no_max_price_passes_all_available(self):
        engine = _make_engine()
        quotes = [self._make_quote(1.0), self._make_quote(9.0)]
        filtered = engine._filter_quotes(quotes, max_price=None)
        assert len(filtered) == 2

    def test_empty_quotes(self):
        engine = _make_engine()
        assert engine._filter_quotes([], max_price=5.0) == []


# ── _rank_quotes ────────────────────────────────────────────────────────────

class TestRankQuotes:
    def _make_quote(self, score):
        from terradev_cli.core.terradev_engine import InstanceQuote
        return InstanceQuote(
            provider="p", instance_type="t", gpu_type="H100",
            price_per_hour=1.0, region="r", available=True,
            latency_ms=50, optimization_score=score, metadata={}
        )

    def test_sorted_descending(self):
        engine = _make_engine()
        quotes = [self._make_quote(0.3), self._make_quote(0.9), self._make_quote(0.6)]
        ranked = engine._rank_quotes(quotes)
        scores = [q.optimization_score for q in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_returns_empty(self):
        engine = _make_engine()
        assert engine._rank_quotes([]) == []


# ── _create_mock_instances ──────────────────────────────────────────────────

class TestCreateMockInstances:
    def _make_quote(self, provider="runpod", price=2.5):
        from terradev_cli.core.terradev_engine import InstanceQuote
        return InstanceQuote(
            provider=provider, instance_type="H100x8", gpu_type="H100",
            price_per_hour=price, region="us-east-1", available=True,
            latency_ms=50, optimization_score=0.8, metadata={}
        )

    def test_returns_one_instance_per_quote(self):
        engine = _make_engine()
        quotes = [self._make_quote(), self._make_quote(provider="vastai")]
        instances = engine._create_mock_instances(quotes)
        assert len(instances) == 2

    def test_instance_has_dry_run_flag(self):
        engine = _make_engine()
        instances = engine._create_mock_instances([self._make_quote()])
        assert instances[0].metadata["dry_run"] is True

    def test_instance_id_starts_with_mock(self):
        engine = _make_engine()
        instances = engine._create_mock_instances([self._make_quote()])
        assert instances[0].instance_id.startswith("mock_")

    def test_instance_id_unique(self):
        engine = _make_engine()
        quotes = [self._make_quote(), self._make_quote()]
        instances = engine._create_mock_instances(quotes)
        ids = {i.instance_id for i in instances}
        assert len(ids) == 2


# ── _analyze_costs ──────────────────────────────────────────────────────────

class TestAnalyzeCosts:
    def _make_instance(self, price):
        from terradev_cli.core.terradev_engine import ProvisionedInstance, ProvisioningStatus
        return ProvisionedInstance(
            instance_id="i-1", provider="runpod", instance_type="H100",
            gpu_type="H100", price_per_hour=price, region="us-east-1",
            status=ProvisioningStatus.RUNNING, created_at=datetime.now(), metadata={}
        )

    def test_total_cost_summed(self):
        engine = _make_engine()
        instances = [self._make_instance(1.0), self._make_instance(2.0)]
        analysis = engine._analyze_costs(instances)
        assert analysis["total_cost_per_hour"] == pytest.approx(3.0)

    def test_instance_count_correct(self):
        engine = _make_engine()
        instances = [self._make_instance(1.0), self._make_instance(1.0)]
        analysis = engine._analyze_costs(instances)
        assert analysis["instance_count"] == 2

    def test_empty_instances(self):
        engine = _make_engine()
        analysis = engine._analyze_costs([])
        assert analysis["total_cost_per_hour"] == 0.0
        assert analysis["instance_count"] == 0

    def test_savings_calculated(self):
        engine = _make_engine()
        instances = [self._make_instance(1.0)]
        analysis = engine._analyze_costs(instances)
        assert "estimated_savings" in analysis
        assert "monthly_savings" in analysis


# ── provision_instances (async, mocked provider) ────────────────────────────

class TestProvisionInstances:
    def _make_quote(self):
        from terradev_cli.core.terradev_engine import InstanceQuote
        return InstanceQuote(
            provider="mockprov", instance_type="H100x1", gpu_type="H100",
            price_per_hour=2.0, region="us-east-1", available=True,
            latency_ms=40, optimization_score=0.85, metadata={}
        )

    async def test_dry_run_returns_mock_instances(self):
        engine = _make_engine()

        with patch.object(engine, "_get_parallel_quotes", new=AsyncMock(return_value=[self._make_quote()])):
            result = await engine.provision_instances("H100", count=1, dry_run=True)

        assert result.success is True
        assert len(result.instances) == 1
        assert result.instances[0].metadata["dry_run"] is True

    async def test_no_quotes_returns_failure(self):
        engine = _make_engine()

        with patch.object(engine, "_get_parallel_quotes", new=AsyncMock(return_value=[])):
            result = await engine.provision_instances("H100", count=1)

        assert result.success is False
        assert "No suitable instances found" in result.errors

    async def test_exception_returns_failure(self):
        engine = _make_engine()

        with patch.object(engine, "_get_parallel_quotes", new=AsyncMock(side_effect=RuntimeError("boom"))):
            result = await engine.provision_instances("H100", count=1)

        assert result.success is False
        assert "boom" in result.errors[0]

    async def test_max_price_filters_results(self):
        engine = _make_engine()
        expensive = InstanceQuote = self._make_quote()

        with patch.object(engine, "_get_parallel_quotes", new=AsyncMock(return_value=[expensive])):
            result = await engine.provision_instances("H100", count=1, max_price=0.01, dry_run=True)

        assert result.success is False  # filtered out by max_price

    async def test_total_time_recorded(self):
        engine = _make_engine()

        with patch.object(engine, "_get_parallel_quotes", new=AsyncMock(return_value=[self._make_quote()])):
            result = await engine.provision_instances("H100", count=1, dry_run=True)

        assert result.total_time >= 0


# ── get_quotes ──────────────────────────────────────────────────────────────

class TestGetQuotes:
    async def test_delegates_to_parallel(self):
        engine = _make_engine()
        expected = []

        with patch.object(engine, "_get_parallel_quotes", new=AsyncMock(return_value=expected)) as mock:
            result = await engine.get_quotes(providers=["runpod"], gpu_type="H100", region="us-east-1")

        mock.assert_awaited_once_with(
            gpu_type="H100", region="us-east-1", providers=["runpod"], parallel_queries=6
        )
        assert result is expected


# ── manage_instance ─────────────────────────────────────────────────────────

class TestManageInstance:
    def _mock_provider(self):
        p = MagicMock()
        p.get_instance_status = AsyncMock(return_value={"status": "running"})
        p.stop_instance = AsyncMock(return_value={"stopped": True})
        p.start_instance = AsyncMock(return_value={"started": True})
        p.terminate_instance = AsyncMock(return_value={"terminated": True})
        return p

    async def test_status_action(self):
        engine = _make_engine()
        provider = self._mock_provider()
        engine.providers["runpod"] = provider

        result = await engine.manage_instance("runpod_inst-123", "status")
        provider.get_instance_status.assert_awaited_once()
        assert result["status"] == "running"

    async def test_stop_action(self):
        engine = _make_engine()
        engine.providers["runpod"] = self._mock_provider()
        result = await engine.manage_instance("runpod_inst-123", "stop")
        assert result["stopped"] is True

    async def test_unknown_action_raises(self):
        engine = _make_engine()
        engine.providers["runpod"] = self._mock_provider()
        with pytest.raises(ValueError, match="Unknown action"):
            await engine.manage_instance("runpod_inst-123", "explode")

    async def test_unknown_provider_raises(self):
        engine = _make_engine()
        with pytest.raises(ValueError, match="Unknown provider"):
            await engine.manage_instance("noprovider_inst-123", "status")

    async def test_no_underscore_raises(self):
        engine = _make_engine()
        with pytest.raises(ValueError):
            await engine.manage_instance("nounderscore", "status")
