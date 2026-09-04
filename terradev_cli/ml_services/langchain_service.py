#!/usr/bin/env python3
"""
LangChain Service Integration for Terradev
Enhanced LangChain integration with workflow orchestration and monitoring
"""

import aiohttp
import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LangChainConfig:
    """LangChain configuration"""

    api_key: str
    langsmith_api_key: Optional[str] = None
    langsmith_endpoint: Optional[str] = None
    workspace_id: Optional[str] = None
    project_name: Optional[str] = None
    environment: str = "development"
    dashboard_enabled: bool = False
    tracing_enabled: bool = False
    evaluation_enabled: bool = False
    workflow_enabled: bool = False
    openai_api_key: Optional[str] = None


class LangChainService:
    """LangChain integration service for LLM workflows and chains"""

    def __init__(self, config: LangChainConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.langsmith_api_base = (
            config.langsmith_endpoint or "https://api.smith.langchain.com"
        )

    async def __aenter__(self):
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def test_connection(self) -> Dict[str, Any]:
        """Test LangChain and LangSmith connection"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)

            # Test LangSmith connection
            if self.config.langsmith_api_key:
                langsmith_headers = {
                    "Authorization": f"Bearer {self.config.langsmith_api_key}"
                }
                langsmith_session = aiohttp.ClientSession(headers=langsmith_headers)

                url = f"{self.langsmith_api_base}/v1/organizations"
                async with langsmith_session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        langsmith_data = await response.json()
                        langsmith_status = "connected"
                    else:
                        langsmith_status = "failed"
                        langsmith_data = {
                            "error": f"LangSmith API request failed: {response.status}"
                        }

                await langsmith_session.close()
            else:
                langsmith_status = "not_configured"
                langsmith_data = {"message": "LangSmith API key not provided"}

            return {
                "status": langsmith_status,
                "langsmith": langsmith_data,
                "environment": self.config.environment,
                "dashboard_enabled": self.config.dashboard_enabled,
                "tracing_enabled": self.config.tracing_enabled,
                "evaluation_enabled": self.config.evaluation_enabled,
                "workflow_enabled": self.config.workflow_enabled,
            }

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def create_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a LangChain workflow.

        Builds a runnable LLM chain from the supplied prompt. If an
        OpenAI-compatible API key is configured and a prompt is provided, the
        chain is executed and the output is returned; otherwise the workflow
        definition is returned without invoking a model.
        """
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)

            workflow_id = (
                f"terradev-workflow-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

            name = workflow_config.get("name", "Terradev Workflow")
            prompt = workflow_config.get(
                "prompt",
                workflow_config.get("description", "Answer the user's question."),
            )
            model = workflow_config.get("model", "openai/gpt-4")

            chain_config = {
                "name": name,
                "prompt": prompt,
                "model": model,
                "workflow_id": workflow_id,
            }

            # Try to build and optionally run a real LangChain chain
            result = {
                "status": "created",
                "workflow_id": workflow_id,
                "config": workflow_config,
                "name": name,
                "description": workflow_config.get(
                    "description", "LangChain workflow created via Terradev CLI"
                ),
                "chain_config": chain_config,
                "output": None,
            }

            try:
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_openai import ChatOpenAI

                openai_key = self.config.openai_api_key or os.environ.get(
                    "OPENAI_API_KEY"
                )
                model_name = model.split("/", 1)[-1]
                llm = ChatOpenAI(
                    model=model_name,
                    temperature=workflow_config.get("temperature", 0.7),
                    api_key=openai_key or None,
                )
                chat_prompt = ChatPromptTemplate.from_messages(
                    [("system", "You are a helpful assistant."), ("human", prompt)]
                )
                chain = chat_prompt | llm

                if openai_key:
                    invocation = await chain.ainvoke({})
                    result["output"] = invocation.content
                else:
                    result["message"] = (
                        "Workflow built but not executed; set OPENAI_API_KEY or "
                        "openai_api_key to run the chain."
                    )
            except ImportError:
                result["message"] = (
                    "langchain-core and langchain-openai are not installed. "
                    "Install them to build a real chain."
                )
            except Exception as llm_error:  # noqa: BLE001
                result["llm_error"] = str(llm_error)

            return result

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def create_langgraph_workflow(
        self, graph_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a LangGraph workflow with monitoring.

        Delegates to the LangGraph service so the graph is built with the same
        engine used by `terradev ml langgraph`. If langgraph is not installed,
        a descriptive message is returned alongside the workflow metadata.
        """
        try:
            from terradev_cli.ml_services.langgraph_service import (
                LangGraphService,
                LangGraphConfig,
            )

            langgraph_creds = {
                "api_key": self.config.api_key,
                "langsmith_api_key": self.config.langsmith_api_key,
                "langsmith_endpoint": self.config.langsmith_endpoint,
                "workspace_id": self.config.workspace_id,
                "project_name": self.config.project_name,
                "environment": self.config.environment,
                "openai_api_key": self.config.openai_api_key,
                "dashboard_enabled": "true" if self.config.dashboard_enabled else "false",
                "tracing_enabled": "true" if self.config.tracing_enabled else "false",
                "evaluation_enabled": "true" if self.config.evaluation_enabled else "false",
                "deployment_enabled": "false",
                "observability_enabled": "false",
            }

            langgraph_config = LangGraphConfig(
                api_key=langgraph_creds["api_key"],
                langsmith_api_key=langgraph_creds.get("langsmith_api_key") or None,
                langsmith_endpoint=langgraph_creds.get("langsmith_endpoint") or None,
                workspace_id=langgraph_creds.get("workspace_id") or None,
                project_name=langgraph_creds.get("project_name") or None,
                environment=langgraph_creds.get("environment", "development"),
                dashboard_enabled=langgraph_creds.get("dashboard_enabled") == "true",
                tracing_enabled=langgraph_creds.get("tracing_enabled") == "true",
                evaluation_enabled=langgraph_creds.get("evaluation_enabled") == "true",
                deployment_enabled=False,
                observability_enabled=False,
                openai_api_key=langgraph_creds.get("openai_api_key") or None,
            )

            service = LangGraphService(langgraph_config)
            return await service.create_workflow(graph_config)

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def create_sglang_pipeline(
        self, pipeline_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an SGLang pipeline for model serving.

        Generates a workload-optimized SGLang serving configuration and a
        launch command. No live SGLang server is started by this method.
        """
        try:
            from terradev_cli.ml_services.sglang_service import (
                SGLangService,
                WorkloadType,
            )

            pipeline_id = f"terradev-sglang-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            model_path = pipeline_config.get(
                "model_path", "meta-llama/Llama-3.1-8B-Instruct"
            )
            workload_type_value = pipeline_config.get("workload_type")
            workload_type = None
            if workload_type_value:
                try:
                    workload_type = WorkloadType(workload_type_value)
                except ValueError:
                    workload_type = None

            optimizer = SGLangService()
            optimized = optimizer.create_optimized_config(
                model_path=model_path,
                workload_type=workload_type,
                user_description=pipeline_config.get("description", ""),
            )

            launch_command = self._build_sglang_launch_command(optimized)

            return {
                "status": "created",
                "pipeline_id": pipeline_id,
                "config": pipeline_config,
                "name": pipeline_config.get("name", "Terradev SGLang Pipeline"),
                "description": pipeline_config.get(
                    "description", "SGLang pipeline created via Terradev CLI"
                ),
                "model_path": model_path,
                "optimized_config": self._sglang_config_to_dict(optimized),
                "launch_command": launch_command,
            }

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    def _build_sglang_launch_command(self, config) -> str:
        """Build an SGLang launch command from an optimized config."""
        flags = [
            f"--model-path {config.model_path}",
            f"--tp-size {config.tp}",
            f"--dp-size {config.dp_size}",
            f"--mem-fraction-static {config.mem_fraction_static}",
            f"--max-running-requests {config.max_running_requests}",
            f"--chunked-prefill-size {config.chunked_prefill_size}",
        ]
        if config.schedule_policy:
            flags.append(f"--schedule-policy {config.schedule_policy.value}")
        if config.attention_backend:
            flags.append(f"--attention-backend {config.attention_backend.value}")
        if config.kv_cache_dtype:
            flags.append(f"--kv-cache-dtype {config.kv_cache_dtype}")
        if config.quantization:
            flags.append(f"--quantization {config.quantization}")
        if config.enable_xgrammar:
            flags.append("--enable-xgrammar")
        if config.disaggregation_mode:
            flags.append(f"--disaggregation-mode {config.disaggregation_mode}")
        if config.nnodes > 1:
            flags.append(f"--nnodes {config.nnodes}")
        if config.enable_expert_parallel:
            flags.append("--enable-expert-parallel")
        if config.enable_eplb:
            flags.append("--enable-eplb")
        if config.enable_dp_attention:
            flags.append("--enable-dp-attention")

        return "python3 -m sglang.launch_server " + " ".join(flags)

    def _sglang_config_to_dict(self, config) -> Dict[str, Any]:
        """Convert an SGLangConfig dataclass to a JSON-safe dict."""

        def _convert(value: Any) -> Any:
            from enum import Enum

            if isinstance(value, Enum):
                return value.value
            if isinstance(value, list):
                return [_convert(v) for v in value]
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            return str(value)

        return _convert(asdict(config))

    async def get_langsmith_projects(self) -> List[Dict[str, Any]]:
        """Get LangSmith projects"""
        try:
            if not self.config.langsmith_api_key:
                return []

            headers = {"Authorization": f"Bearer {self.config.langsmith_api_key}"}
            langsmith_session = aiohttp.ClientSession(headers=headers)

            url = f"{self.langsmith_api_base}/v1/organizations"
            async with langsmith_session.get(
                url, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("organizations", [])
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to get LangSmith projects: {response.status} - {error_text}"
                    )

        except Exception as e:  # noqa: BLE001
            raise Exception(f"Failed to get LangSmith projects: {e}")

    async def get_langsmith_workspaces(self) -> List[Dict[str, Any]]:
        """Get LangSmith workspaces"""
        try:
            if not self.config.langsmith_api_key:
                return []

            headers = {"Authorization": f"Bearer {self.config.langsmith_api_key}"}
            langsmith_session = aiohttp.ClientSession(headers=headers)

            url = f"{self.langsmith_api_base}/v1/workspaces"
            async with langsmith_session.get(
                url, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("workspaces", [])
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to get LangSmith workspaces: {response.status} - {error_text}"
                    )

        except Exception as e:  # noqa: BLE001
            raise Exception(f"Failed to get LangSmith workspaces: {e}")

    async def create_langsmith_project(
        self, name: str, description: str = ""
    ) -> Dict[str, Any]:
        """Create a LangSmith project"""
        try:
            if not self.config.langsmith_api_key:
                return {"status": "failed", "error": "LangSmith API key not configured"}

            headers = {"Authorization": f"Bearer {self.config.langsmith_api_key}"}
            langsmith_session = aiohttp.ClientSession(headers=headers)

            # Find workspace ID (use first available if not specified)
            workspaces = await self.get_langsmith_workspaces()
            workspace_id = self.config.workspace_id or (
                workspaces[0]["id"] if workspaces else None
            )

            if not workspace_id:
                return {"status": "failed", "error": "No workspace found"}

            url = f"{self.langsmith_api_base}/v1/organizations/{workspace_id}/projects"
            payload = {"name": name, "description": description}

            async with langsmith_session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200 or response.status == 201:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to create LangSmith project: {response.status} - {error_text}"
                    )

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def get_langsmith_runs(
        self, project_name: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get LangSmith runs from a project"""
        try:
            if not self.config.langsmith_api_key:
                return []

            headers = {"Authorization": f"Bearer {self.config.langsmith_api_key}"}
            langsmith_session = aiohttp.ClientSession(headers=headers)

            # Find project ID
            projects = await self.get_langsmith_projects()
            project_id = None
            for project in projects:
                if project.get("name") == project_name:
                    project_id = project["id"]
                    break

            if not project_id:
                return []

            url = f"{self.langsmith_api_base}/v1/projects/{project_id}/runs"
            params = {"limit": limit}

            async with langsmith_session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("runs", [])
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to get LangSmith runs: {response.status} - {error_text}"
                    )

        except Exception as e:  # noqa: BLE001
            raise Exception(f"Failed to get LangSmith runs: {e}")

    async def create_trace(
        self, run_id: str, trace_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a trace in LangSmith"""
        try:
            if not self.config.langsmith_api_key:
                return {"status": "failed", "error": "LangSmith API key not configured"}

            headers = {"Authorization": f"Bearer {self.config.langsmith_api_key}"}
            langsmith_session = aiohttp.ClientSession(headers=headers)

            url = f"{self.langsmith_api_base}/v1/traces"
            payload = {"id": run_id, "data": trace_data}

            async with langsmith_session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(
                        f"Failed to create trace: {response.status} - {error_text}"
                    )

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    def get_langchain_config(self) -> Dict[str, str]:
        """Get LangChain configuration for environment variables"""
        config = {"LANGCHAIN_API_KEY": self.config.api_key}

        if self.config.langsmith_api_key:
            config["LANGSMITH_API_KEY"] = self.config.langsmith_api_key
            config["LANGSMITH_TRACING"] = "true"

        if self.config.langsmith_endpoint:
            config["LANGSMITH_ENDPOINT"] = self.config.langsmith_endpoint

        if self.config.workspace_id:
            config["LANGSMITH_WORKSPACE_ID"] = self.config.workspace_id

        if self.config.project_name:
            config["LANGSMITH_PROJECT"] = self.config.project_name
        else:
            config["LANGSMITH_PROJECT"] = "terradev"

        if self.config.environment:
            config["LANGCHAIN_ENVIRONMENT"] = self.config.environment

        if self.config.dashboard_enabled:
            config["LANGCHAIN_DASHBOARD_ENABLED"] = "true"

        if self.config.tracing_enabled:
            config["LANGCHAIN_TRACING"] = "true"

        if self.config.evaluation_enabled:
            config["LANGCHAIN_EVALUATION"] = "true"

        if self.config.workflow_enabled:
            config["LANGCHAIN_WORKFLOW_ENABLED"] = "true"

        return config

    def generate_integration_script(self) -> str:
        """Generate LangChain integration script"""
        script_lines = [
            "# LangChain Integration Script (generated by Terradev)",
            "",
            "# Set up LangChain environment variables",
            f"export LANGCHAIN_API_KEY='{self.config.api_key}'",
            "",
            f"export LANGSMITH_API_KEY='{self.config.langsmith_api_key or ''}'",
            f"export LANGSMITH_ENDPOINT='{self.config.langsmith_endpoint or 'https://api.smith.langchain.com'}'",
            f"export LANGSMITH_WORKSPACE_ID='{self.config.workspace_id or ''}'",
            f"export LANGSMITH_PROJECT='{self.config.project_name or 'terradev'}'",
            f"export LANGCHAIN_ENVIRONMENT='{self.config.environment}'",
            "",
            "# Enhanced features",
            f"export LANGCHAIN_DASHBOARD_ENABLED={'true' if self.config.dashboard_enabled else 'false'}",
            f"export LANGCHAIN_TRACING={'true' if self.config.tracing_enabled else 'false'}",
            f"export LANGCHAIN_EVALUATION={'true' if self.config.evaluation_enabled else 'false'}",
            f"export LANGCHAIN_WORKFLOW_ENABLED={'true' if self.config.workflow_enabled else 'false'}",
            "",
            "# Test LangChain connection",
            "python -c \"import langchain; print('LangChain configured successfully')\"",
            "",
            "# Example usage in training script:",
            "from langchain.chains import LLMChain",
            "from langchain.schema import BasePromptTemplate",
            "",
            "# Initialize with Terradev metadata",
            "chain = LLMChain(llm='openai/gpt-4', temperature=0.7)",
            "chain.invoke('What is the meaning of life?')",
            "",
            "# Log to LangSmith",
            "from langsmith import Client",
            "client = Client(api_key=os.environ.get('LANGSMITH_API_KEY'))",
            "client.create_run(project='terradev')",
            "",
            "# Create workflow",
            "from langgraph.graph import StateGraph, START, END",
            "def orchestrator(state):",
            "    # Your orchestrator logic here",
            "    return {'next': 'worker'}",
            "",
            "def worker(state):",
            "    # Your worker logic here",
            "    return {'result': 'completed'}",
            "",
            "# Build workflow",
            "workflow = StateGraph(State)",
            "workflow.add_node('orchestrator', orchestrator)",
            "workflow.add_node('worker', worker)",
            "workflow.add_edge('orchestrator', 'worker')",
            "workflow.add_edge('worker', END)",
            "",
            "# Compile and run",
            "workflow.invoke({})",
            "",
            "print('LangChain integration complete! Check your LangSmith dashboard at: https://smith.langchain.com/' + os.environ.get('LANGSMITH_WORKSPACE_ID', 'default') + '/' + os.environ.get('LANGSMITH_PROJECT', 'terradev'))",
        ]

        return "\n".join(script_lines)


def create_langchain_service_from_credentials(
    credentials: Dict[str, str]
) -> LangChainService:
    """Create LangChainService from credential dictionary"""
    config = LangChainConfig(
        api_key=credentials["api_key"],
        langsmith_api_key=credentials.get("langsmith_api_key"),
        langsmith_endpoint=credentials.get("langsmith_endpoint"),
        workspace_id=credentials.get("workspace_id"),
        project_name=credentials.get("project_name"),
        environment=credentials.get("environment", "development"),
        dashboard_enabled=credentials.get("dashboard_enabled", "false").lower()
        == "true",
        tracing_enabled=credentials.get("tracing_enabled", "false").lower() == "true",
        evaluation_enabled=credentials.get("evaluation_enabled", "false").lower()
        == "true",
        workflow_enabled=credentials.get("workflow_enabled", "false").lower() == "true",
        openai_api_key=credentials.get("openai_api_key")
        or os.environ.get("OPENAI_API_KEY"),
    )

    return LangChainService(config)


def get_langchain_setup_instructions() -> str:
    """Get setup instructions for LangChain"""
    return """
LangChain Setup Instructions:

1. Install optional dependencies:
   pip install langchain langchain-openai langgraph sglang

2. Configure credentials:
   terradev configure
   # Answer "y" when asked to configure LangChain and provide:
   # - LangChain / LangSmith API key
   # - Feature flags (dashboard, tracing, evaluation, workflow)
   # - (Optional) openai_api_key to execute LLM nodes without OPENAI_API_KEY

   Credentials are stored in ~/.terradev/credentials.json as flat langchain_* keys.

3. Test the integration:
   terradev ml langchain test
   terradev ml langchain create-workflow my-workflow
   terradev ml langchain create-workflow my-workflow --prompt "What is GPU oversubscription?" --model openai/gpt-4o
   terradev ml langchain create-langgraph my-graph
   terradev ml langchain create-pipeline my-pipeline --model-path meta-llama/Llama-3.1-8B-Instruct

4. Set OPENAI_API_KEY or add openai_api_key to credentials to run the LLM chain.
"""
