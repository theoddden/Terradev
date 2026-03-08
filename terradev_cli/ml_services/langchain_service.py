#!/usr/bin/env python3
"""
LangChain Service Integration for Terradev
Enhanced LangChain integration with workflow orchestration and monitoring
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import base64


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


class LangChainService:
    """LangChain integration service for LLM workflows and chains"""
    
    def __init__(self, config: LangChainConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.langsmith_api_base = config.langsmith_endpoint or "https://api.smith.langchain.com"
        
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
                langsmith_headers = {"Authorization": f"Bearer {self.config.langsmith_api_key}"}
                langsmith_session = aiohttp.ClientSession(headers=langsmith_headers)
                
                url = f"{self.langsmith_api_base}/v1/organizations"
                async with langsmith_session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        langsmith_data = await response.json()
                        langsmith_status = "connected"
                    else:
                        langsmith_status = "failed"
                        langsmith_data = {"error": f"LangSmith API request failed: {response.status}"}
                
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
                "workflow_enabled": self.config.workflow_enabled
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def create_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a LangChain workflow"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            # This would integrate with LangChain's workflow APIs
            # For now, we'll create a mock workflow configuration
            workflow_id = f"terradev-workflow-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            return {
                "status": "created",
                "workflow_id": workflow_id,
                "config": workflow_config,
                "name": workflow_config.get("name", "Terradev Workflow"),
                "description": workflow_config.get("description", "Workflow created via Terradev CLI")
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def create_langgraph_workflow(self, graph_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a LangGraph workflow with monitoring"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            # This would integrate with LangGraph's workflow APIs
            # For now, we'll create a mock LangGraph configuration
            workflow_id = f"terradev-langgraph-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            return {
                "status": "created",
                "workflow_id": workflow_id,
                "graph_config": graph_config,
                "name": graph_config.get("name", "Terradev LangGraph"),
                "description": graph_config.get("description", "LangGraph created via Terradev CLI"),
                "nodes": graph_config.get("nodes", []),
                "edges": graph_config.get("edges", [])
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def create_sglang_pipeline(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create an SGLang pipeline for model serving"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            # This would integrate with SGLang's serving APIs
            # For now, we'll create a mock SGLang configuration
            pipeline_id = f"terradev-sglang-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            return {
                "status": "created",
                "pipeline_id": pipeline_id,
                "config": pipeline_config,
                "name": pipeline_config.get("name", "Terradev SGLang Pipeline"),
                "description": pipeline_config.get("description", "SGLang pipeline created via Terradev CLI"),
                "model_path": pipeline_config.get("model_path", ""),
                "serving_config": pipeline_config.get("serving_config", {})
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def get_langsmith_projects(self) -> List[Dict[str, Any]]:
        """Get LangSmith projects"""
        try:
            if not self.config.langsmith_api_key:
                return []
            
            headers = {"Authorization": f"Bearer {self.config.langsmith_api_key}"}
            langsmith_session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.langsmith_api_base}/v1/organizations"
            async with langsmith_session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("organizations", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get LangSmith projects: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get LangSmith projects: {e}")
    
    async def get_langsmith_workspaces(self) -> List[Dict[str, Any]]:
        """Get LangSmith workspaces"""
        try:
            if not self.config.langsmith_api_key:
                return []
            
            headers = {"Authorization": f"Bearer {self.config.langsmith_api_key}"}
            langsmith_session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.langsmith_api_base}/v1/workspaces"
            async with langsmith_session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("workspaces", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get LangSmith workspaces: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get LangSmith workspaces: {e}")
    
    async def create_langsmith_project(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a LangSmith project"""
        try:
            if not self.config.langsmith_api_key:
                return {
                    "status": "failed",
                    "error": "LangSmith API key not configured"
                }
            
            headers = {"Authorization": f"Bearer {self.config.langsmith_api_key}"}
            langsmith_session = aiohttp.ClientSession(headers=headers)
            
            # Find workspace ID (use first available if not specified)
            workspaces = await self.get_langsmith_workspaces()
            workspace_id = self.config.workspace_id or (workspaces[0]["id"] if workspaces else None)
            
            if not workspace_id:
                return {
                    "status": "failed",
                    "error": "No workspace found"
                }
            
            url = f"{self.langsmith_api_base}/v1/organizations/{workspace_id}/projects"
            payload = {
                "name": name,
                "description": description
            }
            
            async with langsmith_session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200 or response.status == 201:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create LangSmith project: {response.status} - {error_text}")
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def get_langsmith_runs(self, project_name: str, limit: int = 100) -> List[Dict[str, Any]]:
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
            
            async with langsmith_session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("runs", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to get LangSmith runs: {response.status} - {error_text}")
                    
        except Exception as e:
            raise Exception(f"Failed to get LangSmith runs: {e}")
    
    async def create_trace(self, run_id: str, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a trace in LangSmith"""
        try:
            if not self.config.langsmith_api_key:
                return {
                    "status": "failed",
                    "error": "LangSmith API key not configured"
                }
            
            headers = {"Authorization": f"Bearer {self.config.langsmith_api_key}"}
            langsmith_session = aiohttp.ClientSession(headers=headers)
            
            url = f"{self.langsmith_api_base}/v1/traces"
            payload = {
                "id": run_id,
                "data": trace_data
            }
            
            async with langsmith_session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create trace: {response.status} - {error_text}")
                    
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def get_langchain_config(self) -> Dict[str, str]:
        """Get LangChain configuration for environment variables"""
        config = {
            "LANGCHAIN_API_KEY": self.config.api_key
        }
        
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
            "print('LangChain integration complete! Check your LangSmith dashboard at: https://smith.langchain.com/' + os.environ.get('LANGSMITH_WORKSPACE_ID', 'default') + '/' + os.environ.get('LANGSMITH_PROJECT', 'terradev'))"
        ]
        
        return "\n".join(script_lines)


def create_langchain_service_from_credentials(credentials: Dict[str, str]) -> LangChainService:
    """Create LangChainService from credential dictionary"""
    config = LangChainConfig(
        api_key=credentials["api_key"],
        langsmith_api_key=credentials.get("langsmith_api_key"),
        langsmith_endpoint=credentials.get("langsmith_endpoint"),
        workspace_id=credentials.get("workspace_id"),
        project_name=credentials.get("project_name"),
        environment=credentials.get("environment", "development"),
        dashboard_enabled=credentials.get("dashboard_enabled", "false").lower() == "true",
        tracing_enabled=credentials.get("tracing_enabled", "false").lower() == "true",
        evaluation_enabled=credentials.get("evaluation_enabled", "false").lower() == "true",
        workflow_enabled=credentials.get("workflow_enabled", "false").lower() == "true"
    )
    
    return LangChainService(config)


def get_langchain_setup_instructions() -> str:
    """Get setup instructions for LangChain"""
    return """
🚀 LangChain Setup Instructions:

1. Install LangChain:
   # Basic installation
   pip install langchain
   
   # With LangSmith support
   pip install langsmith
   
   # With all integrations
   pip install langchain[all]

2. Create LangSmith account:
   - Go to https://smith.langchain.com
   - Sign up for a free account
   - Create an API key

3. Create a workspace (optional):
   - Navigate to your workspace settings
   - Copy the workspace ID

4. Configure Terradev with LangChain:
   terradev configure --provider langchain \
     --api-key YOUR_KEY \
     --langsmith-api-key YOUR_LANGSMITH_KEY \
     --workspace-id YOUR_WORKSPACE_ID \
     --project-name terradev \
     --environment development \
     --dashboard-enabled true \
     --tracing-enabled true \
     --evaluation-enabled true \
     --workflow-enabled true

📋 Required Credentials:
- api_key: LangChain API key (required)
- langsmith_api_key: LangSmith API key (optional, for tracing)
- langsmith_endpoint: LangSmith endpoint (optional, default: https://api.smith.langchain.com)
- workspace_id: LangSmith workspace ID (optional)
- project_name: Default project name (optional, default: "terradev")
- environment: Environment (default: "development")
- dashboard_enabled: Enable dashboard features (default: "false")
- tracing_enabled: Enable tracing (default: "false")
- evaluation_enabled: Enable evaluation (default: "false")
- workflow_enabled: Enable workflow features (default: "false")

💡 Usage Examples:
# Test connection
terradev ml langchain --test

# Create LangChain workflow
terradev ml langchain --create-workflow --name my-workflow

# Create LangGraph workflow
terradev ml langchain --create-langgraph --name my-graph

# Create SGLang pipeline
terradev ml langchain --create-pipeline --name my-pipeline

# Get LangSmith projects
terradev ml langchain --list-projects

# Get LangSmith runs
terradev ml langchain --list-runs --project my-project

# Create trace
terradev ml langchain --create-trace --run-id RUN_ID --data '{"key": "value"}'

🔗 Environment Variables for Training:
Add these to your ML training scripts:
export LANGCHAIN_API_KEY="your-key"
export LANGSMITH_API_KEY="your-langsmith-key"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_WORKSPACE_ID="your-workspace-id"
export LANGSMITH_PROJECT="terradev"

🎯 Integration with Terradev:
LangChain can be used alongside Terradev's provisioning:
- Provision GPU instances with Terradev
- Run LLM chains on provisioned instances
- Trace workflows with LangSmith
- Evaluate models with LangSmith
- Deploy workflows with LangGraph
- Serve models with SGLang

📊 Dashboard Integration:
- **LangSmith Dashboard**: https://smith.langchain.com
- **Terradev Integration**: Custom dashboards for infrastructure metrics
- **Workflow Visualization**: LangGraph Studio visualizations
- **Performance Metrics**: Chain performance and latency tracking

📝 Example Training Script:
import langchain
from langsmith import Client
from terradev_cli.ml_services.langchain_service import create_langchain_service_from_credentials

# Initialize LangChain with Terradev metadata
chain = langchain.LLMChain(llm='openai/gpt-4', temperature=0.7)

# Log to LangSmith
client = Client(api_key=os.environ.get('LANGSMITH_API_KEY'))
client.create_run(project='terradev')

# Create workflow
from langgraph.graph import StateGraph, START, END

def orchestrator(state):
    return {'next': 'worker'}

def worker(state):
    return {'result': 'completed'}

# Build workflow
workflow = StateGraph(State)
workflow.add_node('orchestrator', orchestrator)
workflow.add_node('worker', worker)
workflow.add_edge('orchestrator', 'worker')
workflow.add_edge('worker', END')

# Compile and run
result = workflow.invoke({})
print("Workflow completed! Check LangSmith dashboard for details.")
"""
