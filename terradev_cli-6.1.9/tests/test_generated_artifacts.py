"""Schema and structural validation for generated deployment artifacts."""

import json
import re

import pytest

from terradev_cli.ml_services.agentic_serving import (
    AgenticServingConfig,
    generate_helm_values,
    generate_k8s_deployment,
)
from terradev_cli.ml_services.vllm_service import VLLMConfig, VLLMService
from terradev_cli.ml_services.sglang_service import SGLangConfig, SGLangService, WorkloadType


class TestVLLMDeploymentScript:
    def test_deployment_script_is_bash(self):
        svc = VLLMService(VLLMConfig(model_name="meta-llama/Llama-3.1-8B-Instruct"))
        script = svc.get_deployment_script("1.2.3.4")
        assert "#!/bin/bash" in script
        assert "vllm" in script
        assert "meta-llama/Llama-3.1-8B-Instruct" in script

    def test_deployment_script_contains_required_flags(self):
        svc = VLLMService(VLLMConfig(model_name="model"))
        script = svc.get_deployment_script("1.2.3.4")
        assert "serve" in script
        assert "model" in script
        for flag in ["--host", "--port", "--tensor-parallel-size"]:
            assert flag in script, f"missing {flag}"


class TestSGLangDeploymentScript:
    def test_deployment_script_is_bash(self):
        svc = SGLangService(SGLangConfig(model_path="meta-llama/Llama-2-7b-hf", workload_type=WorkloadType.AGENTIC_CHAT))
        script = svc.get_deployment_script("1.2.3.4")
        assert "#!/bin/bash" in script
        assert "sglang" in script
        assert "meta-llama/Llama-2-7b-hf" in script

    def test_deployment_script_contains_required_flags(self):
        svc = SGLangService(SGLangConfig(model_path="model", workload_type=WorkloadType.AGENTIC_CHAT))
        script = svc.get_deployment_script("1.2.3.4")
        assert "--model-path" in script
        assert "--tp-size" in script
        assert "--dp-size" in script


class TestAgenticServingArtifacts:
    def test_helm_values_schema(self):
        cfg = AgenticServingConfig()
        values = generate_helm_values(cfg)
        assert isinstance(values, dict)
        assert "agenticInference" in values
        assert values["agenticInference"]["image"]
        assert "model" in values["agenticInference"]
        assert "args" in values["agenticInference"]
        assert "env" in values["agenticInference"]

    def test_helm_values_engine_override(self):
        cfg = AgenticServingConfig(engine="sglang", model="meta-llama/Llama-2-7b-hf")
        values = generate_helm_values(cfg)
        inner = values.get("agenticInference", {})
        assert "sglang" in inner.get("image", "") or "sglang" in json.dumps(values).lower()

    def test_k8s_deployment_contains_apiversion(self):
        cfg = AgenticServingConfig()
        manifest = generate_k8s_deployment(cfg, namespace="test-ns")
        assert "apiVersion" in manifest
        assert "kind" in manifest
        assert "test-ns" in manifest

    def test_k8s_deployment_is_yaml(self):
        cfg = AgenticServingConfig()
        manifest = generate_k8s_deployment(cfg)
        # Top-level YAML should have at least one document marker or key
        assert "---" in manifest or "apiVersion" in manifest


class TestGeneratedArtifactValidity:
    def test_vllm_script_has_systemd_service(self):
        svc = VLLMService(VLLMConfig(model_name="m"))
        script = svc.get_deployment_script("1.2.3.4")
        assert "systemctl" in script or "systemd" in script

    def test_sglang_script_uses_model_path_not_model_name(self):
        svc = SGLangService(SGLangConfig(model_path="the-model", workload_type=WorkloadType.AGENTIC_CHAT))
        script = svc.get_deployment_script("1.2.3.4")
        # Ensure the flag is --model-path
        assert re.search(r"--model-path\s+the-model", script)
