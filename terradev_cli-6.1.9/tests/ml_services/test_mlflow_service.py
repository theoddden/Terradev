"""Unit tests for the MLflow ML service."""
import asyncio

import pytest

from terradev_cli.ml_services.mlflow_service import (
    MLflowConfig,
    MLflowService,
    create_mlflow_service_from_credentials,
    get_mlflow_setup_instructions,
)


@pytest.fixture
def svc():
    return MLflowService(MLflowConfig(tracking_uri="http://mlflow.local:5000"))


@pytest.fixture
def auth_svc():
    return MLflowService(
        MLflowConfig(
            tracking_uri="http://mlflow.local:5000",
            username="admin",
            password="secret",
        )
    )


def test_get_setup_instructions():
    instructions = get_mlflow_setup_instructions()
    assert "MLflow" in instructions
    assert "pip install mlflow" in instructions




def test_create_service_from_credentials():
    svc = create_mlflow_service_from_credentials(
        {
            "tracking_uri": "http://mlflow:5000",
            "username": "u",
            "password": "p",
            "experiment_name": "exp",
            "registry_uri": "http://mlflow:5001",
        }
    )
    assert isinstance(svc, MLflowService)
    assert svc.config.tracking_uri == "http://mlflow:5000"
    assert svc.config.username == "u"
    assert svc.config.password == "p"
    assert svc.config.experiment_name == "exp"
    assert svc.config.registry_uri == "http://mlflow:5001"


def test_ensure_session_creates_session(svc, fake_aiohttp):
    session = svc._ensure_session()
    assert session is not None


def test_ensure_session_uses_basic_auth(auth_svc, fake_aiohttp):
    session = auth_svc._ensure_session()
    assert session is not None


@pytest.mark.asyncio
async def test_test_connection_success(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"experiments": [{"name": "exp1"}, {"name": "exp2"}]}, ""),
    ]
    result = await svc.test_connection()
    assert result["status"] == "connected"
    assert result["experiments_count"] == 2
    assert result["tracking_uri"] == "http://mlflow.local:5000"


@pytest.mark.asyncio
async def test_test_connection_failure(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (500, {}, "boom"),
        (500, {}, "boom"),
        (500, {}, "boom"),
    ]
    result = await svc.test_connection()
    assert result["status"] == "failed"
    assert "boom" in result["error"]


@pytest.mark.asyncio
async def test_list_experiments(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"experiments": [{"experiment_id": "1", "name": "a"}]}, ""),
    ]
    assert await svc.list_experiments() == [{"experiment_id": "1", "name": "a"}]


@pytest.mark.asyncio
async def test_create_experiment(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"experiment_id": "123", "name": "new-exp"}, ""),
    ]
    result = await svc.create_experiment("new-exp")
    assert result["experiment_id"] == "123"


@pytest.mark.asyncio
async def test_get_experiment(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"experiment": {"experiment_id": "123", "name": "new-exp"}}, ""),
    ]
    result = await svc.get_experiment("123")
    assert result["experiment"]["name"] == "new-exp"


@pytest.mark.asyncio
async def test_list_runs(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"runs": [{"info": {"run_id": "r1"}}]}, ""),
    ]
    assert await svc.list_runs(["123"]) == [{"info": {"run_id": "r1"}}]


@pytest.mark.asyncio
async def test_get_run(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"run": {"info": {"run_id": "r1"}}}, ""),
    ]
    result = await svc.get_run("r1")
    assert result["run"]["info"]["run_id"] == "r1"


@pytest.mark.asyncio
async def test_log_run(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {}, ""),
        (200, {}, ""),
        (200, {}, ""),
    ]
    result = await svc.log_run(
        "r1",
        metrics={"accuracy": 0.95},
        params={"lr": 0.01},
        tags={"env": "test"},
    )
    assert result["status"] == "logged"
    assert result["run_id"] == "r1"


@pytest.mark.asyncio
async def test_list_registered_models(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"registered_models": [{"name": "m1"}]}, ""),
    ]
    assert await svc.list_registered_models() == [{"name": "m1"}]


@pytest.mark.asyncio
async def test_create_model_version(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"model_version": {"version": 1, "name": "m1"}}, ""),
    ]
    result = await svc.create_model_version(
        name="m1", source="s3://bucket/model", run_id="r1"
    )
    assert result["model_version"]["version"] == 1


@pytest.mark.asyncio
async def test_log_terradev_run(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"experiments": []}, ""),
        (200, {"experiment_id": "e1"}, ""),
        (
            200,
            {"run": {"info": {"run_id": "r1"}}},
            "",
        ),
        (200, {}, ""),
        (200, {}, ""),
        (200, {}, ""),
    ]
    result = await svc.log_terradev_run(
        experiment_name="exp",
        gpu_type="A100",
        provider="runpod",
        region="us-east-1",
        price_hr=2.5,
        duration_hrs=4.0,
        instance_id="pod-123",
        spot=True,
    )
    assert result["run_id"] == "r1"
    assert result["experiment_id"] == "e1"
    assert result["total_cost"] == 10.0


@pytest.mark.asyncio
async def test_register_terradev_model(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"registered_models": []}, ""),
        (200, {}, ""),
        (200, {"model_version": {"version": "1"}}, ""),
        (200, {}, ""),
    ]
    result = await svc.register_terradev_model(
        model_name="m1",
        source="s3://bucket/model",
        run_id="r1",
        checkpoint_id="ckpt-1",
        training_job_id="job-1",
        gpu_type="A100",
        provider="runpod",
    )
    assert result["version"] == "1"


@pytest.mark.asyncio
async def test_export_experiment_data_json(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (200, {"runs": [{"info": {"run_id": "r1"}}]}, ""),
    ]
    output = await svc.export_experiment_data("e1", format="json")
    assert '"run_id": "r1"' in output


@pytest.mark.asyncio
async def test_export_experiment_data_csv(svc, fake_aiohttp):
    fake_aiohttp.responses = [
        (
            200,
            {
                "runs": [
                    {
                        "info": {
                            "run_id": "r1",
                            "experiment_id": "e1",
                            "status": "FINISHED",
                            "start_time": 1,
                            "end_time": 2,
                            "artifact_uri": "s3://a",
                        }
                    }
                ]
            },
            "",
        ),
    ]
    output = await svc.export_experiment_data("e1", format="csv")
    assert "run_id" in output
    assert "r1" in output


def test_get_tracking_config(svc):
    cfg = svc.get_tracking_config()
    assert cfg["MLFLOW_TRACKING_URI"] == "http://mlflow.local:5000"


def test_context_manager(fake_aiohttp):
    svc = MLflowService(MLflowConfig(tracking_uri="http://localhost"))
    session = None

    async def run():
        nonlocal session
        async with svc as s:
            session = s.session
            assert session is not None
        assert session.closed

    asyncio.run(run())
