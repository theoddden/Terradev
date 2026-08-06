"""Comprehensive structural and functional tests for ML integration commands."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from terradev_cli.commands import cli

def _help_test(runner, mock_api, path):
    result = runner.invoke(cli, path.split() + ["--help"], obj={"api": mock_api})
    assert result.exit_code == 0

def _missing_arg_test(runner, mock_api, path):
    result = runner.invoke(cli, path.split(), obj={"api": mock_api})
    assert result.exit_code != 0

def test_ml_wandb_test_help(runner, mock_api):
    _help_test(runner, mock_api, "ml wandb test")

def test_ml_wandb_list_projects_help(runner, mock_api):
    _help_test(runner, mock_api, "ml wandb list-projects")

def test_ml_wandb_create_project_help(runner, mock_api):
    _help_test(runner, mock_api, "ml wandb create-project")

def test_ml_wandb_list_runs_help(runner, mock_api):
    _help_test(runner, mock_api, "ml wandb list-runs")

def test_ml_wandb_create_dashboard_help(runner, mock_api):
    _help_test(runner, mock_api, "ml wandb create-dashboard")

def test_ml_wandb_create_report_help(runner, mock_api):
    _help_test(runner, mock_api, "ml wandb create-report")

def test_ml_wandb_setup_alerts_help(runner, mock_api):
    _help_test(runner, mock_api, "ml wandb setup-alerts")

def test_ml_wandb_dashboard_status_help(runner, mock_api):
    _help_test(runner, mock_api, "ml wandb dashboard-status")

def test_ml_langchain_test_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langchain test")

def test_ml_langchain_create_workflow_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langchain create-workflow")

def test_ml_langchain_create_langgraph_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langchain create-langgraph")

def test_ml_langchain_create_pipeline_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langchain create-pipeline")

def test_ml_langgraph_test_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langgraph test")

def test_ml_langgraph_create_workflow_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langgraph create-workflow")

def test_ml_langgraph_status_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langgraph status")

def test_ml_langgraph_deploy_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langgraph deploy")

def test_ml_kserve_test_help(runner, mock_api):
    _help_test(runner, mock_api, "ml kserve test")

def test_ml_dvc_test_help(runner, mock_api):
    _help_test(runner, mock_api, "ml dvc test")

def test_ml_dvc_init_help(runner, mock_api):
    _help_test(runner, mock_api, "ml dvc init")

def test_ml_dvc_add_remote_help(runner, mock_api):
    _help_test(runner, mock_api, "ml dvc add-remote")

def test_ml_dvc_add_data_help(runner, mock_api):
    _help_test(runner, mock_api, "ml dvc add-data")

def test_ml_dvc_push_help(runner, mock_api):
    _help_test(runner, mock_api, "ml dvc push")

def test_ml_dvc_pull_help(runner, mock_api):
    _help_test(runner, mock_api, "ml dvc pull")

def test_ml_dvc_status_help(runner, mock_api):
    _help_test(runner, mock_api, "ml dvc status")

def test_ml_mlflow_legacy_test_help(runner, mock_api):
    _help_test(runner, mock_api, "ml mlflow-legacy test")

def test_ml_mlflow_legacy_list_experiments_help(runner, mock_api):
    _help_test(runner, mock_api, "ml mlflow-legacy list-experiments")

def test_ml_mlflow_legacy_create_experiment_help(runner, mock_api):
    _help_test(runner, mock_api, "ml mlflow-legacy create-experiment")

def test_ml_mlflow_legacy_list_runs_help(runner, mock_api):
    _help_test(runner, mock_api, "ml mlflow-legacy list-runs")

def test_ml_mlflow_legacy_export_help(runner, mock_api):
    _help_test(runner, mock_api, "ml mlflow-legacy export")

def test_ml_ray_test_help(runner, mock_api):
    _help_test(runner, mock_api, "ml ray test")

def test_ml_ray_install_help(runner, mock_api):
    _help_test(runner, mock_api, "ml ray install")

def test_ml_ray_install_monitoring_help(runner, mock_api):
    _help_test(runner, mock_api, "ml ray install-monitoring")

def test_ml_ray_metrics_summary_help(runner, mock_api):
    _help_test(runner, mock_api, "ml ray metrics-summary")

def test_ml_ray_grafana_help(runner, mock_api):
    _help_test(runner, mock_api, "ml ray grafana")

def test_ml_ray_prometheus_help(runner, mock_api):
    _help_test(runner, mock_api, "ml ray prometheus")

def test_ml_ray_status_help(runner, mock_api):
    _help_test(runner, mock_api, "ml ray status")

def test_ml_ray_list_nodes_help(runner, mock_api):
    _help_test(runner, mock_api, "ml ray list-nodes")

def test_ml_ray_start_help(runner, mock_api):
    _help_test(runner, mock_api, "ml ray start")

def test_ml_ray_stop_help(runner, mock_api):
    _help_test(runner, mock_api, "ml ray stop")

def test_ml_ray_dashboard_help(runner, mock_api):
    _help_test(runner, mock_api, "ml ray dashboard")

def test_ml_vllm_optimize_help(runner, mock_api):
    _help_test(runner, mock_api, "ml vllm optimize")

def test_ml_vllm_auto_optimize_help(runner, mock_api):
    _help_test(runner, mock_api, "ml vllm auto-optimize")

def test_ml_vllm_analyze_help(runner, mock_api):
    _help_test(runner, mock_api, "ml vllm analyze")

def test_ml_vllm_benchmark_help(runner, mock_api):
    _help_test(runner, mock_api, "ml vllm benchmark")

def test_ml_phoenix_test_help(runner, mock_api):
    _help_test(runner, mock_api, "ml phoenix test")

def test_ml_phoenix_projects_help(runner, mock_api):
    _help_test(runner, mock_api, "ml phoenix projects")

def test_ml_phoenix_spans_help(runner, mock_api):
    _help_test(runner, mock_api, "ml phoenix spans")

def test_ml_phoenix_trace_help(runner, mock_api):
    _help_test(runner, mock_api, "ml phoenix trace")

def test_ml_phoenix_otel_env_help(runner, mock_api):
    _help_test(runner, mock_api, "ml phoenix otel-env")

def test_ml_phoenix_snippet_help(runner, mock_api):
    _help_test(runner, mock_api, "ml phoenix snippet")

def test_ml_phoenix_k8s_help(runner, mock_api):
    _help_test(runner, mock_api, "ml phoenix k8s")

def test_ml_guardrails_test_help(runner, mock_api):
    _help_test(runner, mock_api, "ml guardrails test")

def test_ml_guardrails_chat_help(runner, mock_api):
    _help_test(runner, mock_api, "ml guardrails chat")

def test_ml_guardrails_generate_config_help(runner, mock_api):
    _help_test(runner, mock_api, "ml guardrails generate-config")

def test_ml_guardrails_k8s_help(runner, mock_api):
    _help_test(runner, mock_api, "ml guardrails k8s")

def test_ml_qdrant_test_help(runner, mock_api):
    _help_test(runner, mock_api, "ml qdrant test")

def test_ml_qdrant_collections_help(runner, mock_api):
    _help_test(runner, mock_api, "ml qdrant collections")

def test_ml_qdrant_create_collection_help(runner, mock_api):
    _help_test(runner, mock_api, "ml qdrant create-collection")

def test_ml_qdrant_info_help(runner, mock_api):
    _help_test(runner, mock_api, "ml qdrant info")

def test_ml_qdrant_count_help(runner, mock_api):
    _help_test(runner, mock_api, "ml qdrant count")

def test_ml_qdrant_k8s_help(runner, mock_api):
    _help_test(runner, mock_api, "ml qdrant k8s")

def test_ml_sglang_sglang_optimize_help(runner, mock_api):
    _help_test(runner, mock_api, "ml sglang sglang-optimize")

def test_ml_sglang_router_help(runner, mock_api):
    _help_test(runner, mock_api, "ml sglang router")

def test_ml_sglang_detect_help(runner, mock_api):
    _help_test(runner, mock_api, "ml sglang detect")

def test_ml_sglang_install_help(runner, mock_api):
    _help_test(runner, mock_api, "ml sglang install")

def test_ml_sglang_start_help(runner, mock_api):
    _help_test(runner, mock_api, "ml sglang start")

def test_ml_sglang_test_help(runner, mock_api):
    _help_test(runner, mock_api, "ml sglang test")

def test_ml_langfuse_configure_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langfuse configure")

def test_ml_langfuse_test_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langfuse test")

def test_ml_langfuse_traces_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langfuse traces")

def test_ml_langfuse_trace_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langfuse trace")

def test_ml_langfuse_scores_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langfuse scores")

def test_ml_langfuse_score_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langfuse score")

def test_ml_langfuse_datasets_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langfuse datasets")

def test_ml_langfuse_export_training_data_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langfuse export-training-data")

def test_ml_langfuse_quality_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langfuse quality")

def test_ml_langfuse_otel_env_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langfuse otel-env")

def test_ml_langfuse_k8s_help(runner, mock_api):
    _help_test(runner, mock_api, "ml langfuse k8s")

class TestFunctionalMl:
    """Functional tests for the heavy ML service test commands."""

    @pytest.mark.parametrize(
        "path, provider, module_name, create_name, setup_name, creds",
        [
            (
                ["ml", "wandb", "test"],
                "wandb",
                "terradev_cli.ml_services.wandb_enhanced",
                "create_enhanced_wandb_service_from_credentials",
                "get_enhanced_wandb_setup_instructions",
                {"api_key": "test-key"},
            ),
            (
                ["ml", "langchain", "test"],
                "langchain",
                "terradev_cli.ml_services.langchain_service",
                "create_langchain_service_from_credentials",
                "get_langchain_setup_instructions",
                {"api_key": "test-key"},
            ),
            (
                ["ml", "langgraph", "test"],
                "langchain",
                "terradev_cli.ml_services.langgraph_service",
                "create_langgraph_service_from_credentials",
                "get_langgraph_setup_instructions",
                {"api_key": "test-key"},
            ),
            (
                ["ml", "kserve", "test"],
                "kserve",
                "terradev_cli.ml_services.kserve_service",
                "create_kserve_service_from_credentials",
                "get_kserve_setup_instructions",
                {"api_key": "test-key"},
            ),
            (
                ["ml", "dvc", "test"],
                "dvc",
                "terradev_cli.ml_services.dvc_service",
                "create_dvc_service_from_credentials",
                "get_dvc_setup_instructions",
                {"repo_path": "."},
            ),
            (
                ["ml", "mlflow-legacy", "test"],
                "mlflow",
                "terradev_cli.ml_services.mlflow_service",
                "create_mlflow_service_from_credentials",
                "get_mlflow_setup_instructions",
                {"tracking_uri": "http://mlflow.local:5000"},
            ),
            (
                ["ml", "ray", "test"],
                "ray",
                "terradev_cli.ml_services.ray_enhanced",
                "create_enhanced_ray_service_from_credentials",
                None,
                {"api_key": "test-key"},
            ),
            (
                ["ml", "phoenix", "test"],
                "phoenix",
                "terradev_cli.ml_services.phoenix_service",
                "create_phoenix_service_from_credentials",
                "get_phoenix_setup_instructions",
                {"api_key": "test-key"},
            ),
            (
                ["ml", "guardrails", "test"],
                "guardrails",
                "terradev_cli.ml_services.guardrails_service",
                "create_guardrails_service_from_credentials",
                "get_guardrails_setup_instructions",
                {"api_key": "test-key"},
            ),
            (
                ["ml", "qdrant", "test"],
                "qdrant",
                "terradev_cli.ml_services.qdrant_service",
                "create_qdrant_service_from_credentials",
                "get_qdrant_setup_instructions",
                {"api_key": "test-key"},
            ),
            (
                ["ml", "langfuse", "test"],
                "langfuse",
                "terradev_cli.ml_services.langfuse_service",
                "create_langfuse_service_from_credentials",
                None,
                {"api_key": "test-key"},
            ),
        ],
    )
    def test_ml_service_test_command(
        self, runner, mock_api, path, provider, module_name, create_name, setup_name, creds
    ):
        fake_mod = MagicMock()
        ok_result = {
            "status": "connected",
            "entity": "terradev",
            "project": "test",
            "base_url": "http://localhost",
            "collector_endpoint": "http://localhost",
            "projects_found": 1,
            "url": "http://localhost",
            "collections": [],
            "server_url": "http://localhost",
            "host": "http://localhost",
            "clusters": 0,
            "ray_version": "2.0",
            "dashboard_uri": "http://localhost",
            "project_names": [],
            "langsmith": "ready",
            "environment": "test",
            "namespace": "default",
            "workspace_id": "ws-1",
            "endpoint": "http://localhost:8000",
            "repo_path": ".",
            "tracking_uri": "http://mlflow.local:5000",
            "experiments_count": 0,
            "projects": [],
            "error": "",
        }

        if create_name:
            fake_create = MagicMock(return_value=MagicMock())
            fake_create.return_value.test_connection = AsyncMock(
                return_value=ok_result
            )
            setattr(fake_mod, create_name, fake_create)

        if setup_name:
            setattr(fake_mod, setup_name, MagicMock(return_value="setup instructions"))

        if not create_name and not setup_name:
            # databricks test imports test_connection directly
            fake_mod.test_connection = AsyncMock(return_value=ok_result)

        with patch.dict(sys.modules, {module_name: fake_mod}):
            mock_api._provider_creds.return_value = creds
            result = runner.invoke(cli, path, obj={"api": mock_api})
            assert result.exit_code == 0, result.output
            assert "OK" in result.output or "Connected" in result.output or "\u2705" in result.output
            mock_api._provider_creds.assert_called_with(provider)
