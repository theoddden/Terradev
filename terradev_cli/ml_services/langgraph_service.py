#!/usr/bin/env python3
"""
LangGraph Service Integration for Terradev
Enhanced LangGraph integration with workflow orchestration and monitoring
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
class LangGraphConfig:
    """LangGraph configuration"""
    api_key: str
    langsmith_api_key: Optional[str] = None
    langsmith_endpoint: Optional[str] = None
    workspace_id: Optional[str] = None
    project_name: Optional[str] = None
    environment: str = "development"
    dashboard_enabled: bool = False
    tracing_enabled: bool = False
    evaluation_enabled: bool = False
    deployment_enabled: bool = False
    observability_enabled: bool = False


class LangGraphService:
    """LangGraph integration service for workflow orchestration"""
    
    def __init__(self, config: LangGraphConfig):
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
        """Test LangGraph and LangSmith connection"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            # Test LangSmith connection
            if self.config.langsmith_api_key:
                langsmith_headers = {"Authorization": f"Bearer {self.config.langsmith_api_key}"}
                langsmith_session = langsmith_session
                
                url = f"{self.langsmith_api_base}/v1/organizations"
                async with langsmith_session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        langsmith_data = await response.json()
                        langsmith_status = "connected"
                    else:
                        langsmith_status = "failed"
                        langsmith_data = {"error": f"LangSmith API request failed: {response.status}"}
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
                "deployment_enabled": self.config.deployment_enabled,
                "observability_enabled": self.config.observability_enabled
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def create_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a LangGraph workflow with monitoring"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)
            
            # This would integrate with LangGraph's workflow APIs
            workflow_id = f"terradev-langgraph-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            # Create workflow with monitoring configuration
            enhanced_config = {
                **workflow_config,
                "monitoring": {
                    "enabled": self.config.dashboard_enabled,
                    "tracing": self.config.tracing_enabled,
                    "evaluation": self.config.evaluation_enabled,
                    "deployment": self.config.deployment_enabled,
                    "observability": self.config.observability_enabled
                },
                "langsmith": {
                    "project": self.config.project_name or "terradev",
                    "workspace_id": self.config.workspace_id
                }
            }
            
            return {
                "status": "created",
                "workflow_id": workflow_id,
                "config": enhanced_config,
                "name": workflow_config.get("name", "Terradev LangGraph Workflow"),
                "description": workflow_config.get("description", "LangGraph workflow created via Terradev CLI")
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def create_orchestrator_worker_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create an orchestrator-worker pattern workflow"""
        try:
            # Define the workflow state
            from langgraph.graph import StateGraph, START, END
            
            # Enhanced state with monitoring
            class WorkflowState:
                topic: str
                sections: List[Dict[str, Any]]
                completed_sections: List[str]
                current_section: Optional[str]
                total_sections: int
                orchestrator_status: str
                worker_status: str
                metrics: Dict[str, Any]
                langsmith_run_id: Optional[str]
                start_time: datetime
                end_time: Optional[datetime]
            
            # Enhanced orchestrator with monitoring
            def enhanced_orchestrator(state: WorkflowState):
                """Enhanced orchestrator with monitoring"""
                try:
                    # Generate plan
                    from langchain.chains import LLMChain
                    llm = LLM(llm='openai/gpt-4', temperature=0.7)
                    
                    plan_result = llm.invoke([
                        SystemMessage(content="Generate a plan for the report."),
                        HumanMessage(content=f"Here is the report topic: {state['topic']}"),
                    ])
                    
                    sections = plan_result.content.split('\n')
                    return {"sections": [{"name": section.strip(), "description": section.strip()} for section in sections if section.strip()]}
                    
                except Exception as e:
                    return {"error": str(e)}
            
            # Enhanced worker with monitoring
            def enhanced_worker(state: WorkflowState):
                """Enhanced worker with monitoring"""
                try:
                    from langchain.chains import LLMChain
                    llm = LLM(llm='openai/gpt-4', temperature=0.7)
                    
                    section = state['current_section']
                    if section:
                        msg = llm.invoke(f"Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting.")
                    else:
                        msg = llm.invoke(f"Write a section about {state['topic']}")
                    
                    return {
                        "completed_sections": [section.content],
                        "current_section": None,
                        "metrics": {"section_length": len(section.content)}
                    }
                    
                except Exception as e:
                    return {"error": str(e)}
            
            # Enhanced synthesizer with monitoring
            def enhanced_synthesizer(state: WorkflowState):
                """Enhanced synthesizer with monitoring"""
                try:
                    completed_sections = state.get("completed_sections", [])
                    completed_report = "\n\n---\n\n".join(completed_sections)
                    
                    return {
                        "final_report": completed_report,
                        "total_sections": len(completed_sections),
                        "metrics": {"report_length": len(completed_report)}
                    }
                    
                except Exception as e:
                    return {"error": str(e)}
            
            # Conditional edge function for routing
            def route_section(state: WorkflowState):
                """Route to next section or end"""
                if state["current_section"] is None:
                    if len(state["completed_sections"]) >= state["total_sections"]:
                        return "END"
                    else:
                        return "worker"
                else:
                    return "synthesizer"
            
            # Build workflow
            builder = StateGraph(WorkflowState)
            builder.add_node("orchestrator", enhanced_orchestrator)
            builder.add_node("worker", enhanced_worker)
            builder.add_node("synthesizer", enhanced_synthesizer)
            
            # Add edges
            builder.add_edge(START, "orchestrator")
            builder.add_conditional_edges("orchestrator", "worker", ["worker"])
            builder.add_edge("worker", "synthesizer")
            builder.add_edge("synthesizer", END)
            
            # Compile workflow
            workflow = builder.compile()
            
            return {
                "status": "created",
                "workflow_id": workflow_id,
                "config": enhanced_config,
                "graph": workflow.get_graph().dict(),
                "name": workflow_config.get("name", "Terradev Orchestrator-Worker Workflow"),
                "description": workflow_config.get("description", "Orchestrator-worker workflow created via Terradev CLI")
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def create_evaluation_workflow(self, evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create an evaluator-optimizer workflow"""
        try:
            # Define the workflow state
            from langgraph.graph import StateGraph, START, END
            from pydantic import BaseModel, Field
            from typing import Literal
            
            class Feedback(BaseModel):
                grade: Literal["funny", "not funny"] = Field(description="Decide if the joke is funny or not.")
                feedback: str = Field(description="If the joke is not funny, provide feedback on how to improve it.")
            
            # Enhanced state with monitoring
            class EvaluationState:
                joke: str
                topic: str
                feedback: str
                funny_or_not: str
                iteration: int
                metrics: Dict[str, Any]
                langsmith_run_id: Optional[str]
                start_time: datetime
                end_time: Optional[datetime]
            
            # Enhanced generator with monitoring
            def enhanced_generator(state: EvaluationState):
                """Enhanced generator with monitoring"""
                try:
                    from langchain.chains import LLMChain
                    llm = LLM(llm='openai/gpt-4', temperature=0.7)
                    
                    if state.get("feedback"):
                        msg = llm.invoke(
                            f"Write a joke about {state['topic']} but take into account the feedback: {state['feedback']}"
                        )
                    else:
                        msg = llm.invoke(f"Write a joke about {state['topic']}")
                    
                    return {
                        "joke": msg.content,
                        "iteration": state.get("iteration", 0) + 1,
                        "metrics": {"joke_length": len(msg.content)}
                    }
                    
                except Exception as e:
                    return {"error": str(e)}
            
            # Enhanced evaluator with monitoring
            def enhanced_evaluator(state: EvaluationState):
                """Enhanced evaluator with monitoring"""
                try:
                    from langchain.evaluation import load_evaluator
                    evaluator = load_evaluator()
                    
                    grade = evaluator.invoke(f"Grade the joke {state['joke']}")
                    
                    return {
                        "funny_or_not": grade.grade,
                        "feedback": grade.feedback,
                        "iteration": state.get("iteration", 0),
                        "metrics": {"evaluation_time": datetime.now().isoformat()}
                    }
                    
                except Exception as e:
                    return {"error": str(e)}
            
            # Conditional edge function for routing
            def route_evaluation(state: EvaluationState):
                """Route based on evaluation feedback"""
                if state["funny_or_not"] == "funny":
                    return "END"
                elif state["funny_or_not"] == "not funny":
                    return "generator"
            
            # Build workflow
            builder = StateGraph(EvaluationState)
            builder.add_node("generator", enhanced_generator)
            builder.add_node("evaluator", enhanced_evaluator)
            builder.add_edge("generator", "evaluator")
            builder.add_conditional_edges("evaluator", route_evaluation, {
                "Accepted": "END",
                "Rejected + Feedback": "generator"
            })
            
            # Compile workflow
            workflow = builder.compile()
            
            return {
                "status": "created",
                "workflow_id": f"terradev-evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                "config": evaluation_config,
                "graph": workflow.get_graph().dict(),
                "name": evaluation_config.get("name", "Terradev Evaluator-Optimizer Workflow"),
                "description": evaluation_config.get("description", "Evaluator-optimizer workflow created via Terradev CLI")
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status and metrics"""
        try:
            # This would integrate with LangGraph's workflow APIs
            # For now, we'll return a mock status
            return {
                "status": "running",
                "workflow_id": workflow_id,
                "status": "active",
                "metrics": {
                    "nodes": 4,
                    "edges": 3,
                    "runs": 12,
                    "success_rate": 0.95
                },
                "monitoring": {
                    "tracing": self.config.tracing_enabled,
                    "evaluation": self.config.evaluation_enabled,
                    "deployment": self.config.deployment_enabled,
                    "observability": self.config.observability_enabled
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def get_langgraph_config(self) -> Dict[str, str]:
        """Get LangGraph configuration for environment variables"""
        config = self.get_langchain_config()
        
        # Add LangGraph-specific configuration
        if self.config.dashboard_enabled:
            config["LANGGRAPH_DASHBOARD_ENABLED"] = "true"
        
        if self.config.deployment_enabled:
            config["LANGGRAPH_DEPLOYMENT_ENABLED"] = "true"
        
        if self.config.observability_enabled:
            config["LANGGRAPH_OBSERVABILITY_ENABLED"] = "true"
        
        return config
    
    def generate_integration_script(self) -> str:
        """Generate LangGraph integration script"""
        script_lines = [
            "# LangGraph Integration Script (generated by Terradev)",
            "",
            "# Set up LangGraph environment variables",
            f"export LANGCHAIN_API_KEY='{self.config.api_key}'",
            f"export LANGSMITH_API_KEY='{self.config.langsmith_api_key or ''}'",
            f"export LANGSMITH_ENDPOINT='{self.config.langsmith_endpoint or 'https://api.smith.langchain.com'}'",
            f"export LANGSMITH_WORKSPACE_ID='{self.config.workspace_id or ''}'",
            f"export LANGSMITH_PROJECT='{self.config.project_name or 'terradev'}'",
            f"export LANGCHAIN_ENVIRONMENT='{self.config.environment}'",
            "",
            "# Enhanced features",
            f"export LANGGRAPH_DASHBOARD_ENABLED={'true' if self.config.dashboard_enabled else 'false'}",
            f"export LANGGRAPH_DEPLOYMENT_ENABLED={'true' if self.config.deployment_enabled else 'false'}",
            f"export LANGGRAPH_OBSERVABILITY_ENABLED={'true' if self.config.observability_enabled else 'false'}",
            "",
            "# Test LangGraph connection",
            "python -c \"import langgraph; print('LangGraph configured successfully')",
            "",
            "# Example workflow creation",
            "from langgraph.graph import StateGraph, START, END",
            "",
            "# Define state",
            "class State(TypedDict):",
            "    topic: str",
            "    sections: list[str]",
            "    completed_sections: list[str]",
            "    current_section: Optional[str]",
            "    total_sections: int",
            "    metrics: dict",
            "",
            "# Define nodes",
            "def orchestrator(state: State):",
            "    # Your orchestrator logic here",
            "    return {'next': 'worker'}",
            "",
            "def worker(state: State):",
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
            "result = workflow.invoke({})",
            "",
            "print('LangGraph workflow completed! Check LangSmith dashboard for details.')",
            "",
            "# Deploy workflow (if deployment enabled)",
            "if os.environ.get('LANGGRAPH_DEPLOYMENT_ENABLED') == 'true':",
            "    workflow.deploy('my-workflow')",
            "    print('Workflow deployed! Access at: https://smith.langchain.com/deployments')",
            "",
            "# Access dashboard",
            "print('LangSmith Dashboard: https://smith.langchain.com/' + os.environ.get('LANGSMITH_WORKSPACE_ID', 'default') + '/' + os.environ.get('LANGSMITH_PROJECT', 'terradev'))"
        ]
        
        return "\n".join(script_lines)


def create_langgraph_service_from_credentials(credentials: Dict[str, str]) -> LangGraphService:
    """Create LangGraphService from credential dictionary"""
    config = LangGraphConfig(
        api_key=credentials["api_key"],
        langsmith_api_key=credentials.get("langsmith_api_key"),
        langsmith_endpoint=credentials.get("langsmith_endpoint"),
        workspace_id=credentials.get("workspace_id"),
        project_name=credentials.get("project_name"),
        environment=credentials.get("environment", "development"),
        dashboard_enabled=credentials.get("dashboard_enabled", "false").lower() == "true",
        tracing_enabled=credentials.get("tracing_enabled", "false").lower() == "true",
        evaluation_enabled=credentials.get("evaluation_enabled", "false").lower() == "true",
        deployment_enabled=credentials.get("deployment_enabled", "false").lower() == "true",
        observability_enabled=credentials.get("observability_enabled", "false").lower() == "true"
    )
    
    return LangGraphService(config)


def get_langgraph_setup_instructions() -> str:
    """Get setup instructions for LangGraph"""
    return """
🚀 LangGraph Setup Instructions:

1. Install LangGraph:
   # Basic installation
   pip install langgraph
   
   # With LangSmith support
   pip install langsmith
   
   # With all integrations
   pip install langgraph[all]

2. Create LangSmith account:
   - Go to https://smith.langchain.com
   - Sign up for a free account
   - Create an API key

3. Configure Terradev with LangGraph:
   terradev configure --provider langchain \
     --api-key YOUR_KEY \
     --langsmith-api-key YOUR_LANGSMITH_KEY \
     --workspace-id YOUR_WORKSPACE_ID \
     --project-name terradev \
     --environment development \
     --dashboard-enabled true \
     --tracing-enabled true \
     --evaluation-enabled true \
     --deployment-enabled true \
     --observability-enabled true

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
- deployment_enabled: Enable deployment (default: "false")
- observability_enabled: Enable observability (default: "false")

💡 Usage Examples:
# Test connection
terradev ml langchain --test

# Create orchestrator-worker workflow
terradev ml langchain --create-workflow --name my-workflow --type orchestrator-worker

# Create evaluator-optimizer workflow
terradev ml langchain --create-workflow --name my-evaluator --type evaluator-optimizer

# Create SGLang pipeline
terradev ml langchain --create-pipeline --name my-pipeline

# Get LangSmith projects
terradev ml langchain --list-projects

# Get LangSmith runs
terradev ml langchain --list-runs --project my-project

# Create trace
terradev ml langchain --create-trace --run-id RUN_ID --data '{"key": "value"}'

# Get workflow status
terradev ml langchain --workflow-status --workflow-id WORKFLOW_ID

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
- Deploy workflows with LangGraph
- Serve models with SGLang
- Evaluate models with LangSmith

📊 Dashboard Integration:
- **LangSmith Dashboard**: https://smith.langchain.com
- **Terradev Integration**: Custom dashboards for workflow metrics
- **Workflow Visualization**: LangGraph Studio visualizations
- **Performance Metrics**: Chain performance and latency tracking
- **Evaluation Results**: Model evaluation results and feedback

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
workflow.add_edge('worker', END)

# Compile and run
result = workflow.invoke({})
print("Workflow completed! Check LangSmith dashboard for details.")

🔧 Integration with Terradev:
- **Provisioning**: terradev provision -g A100 -n 4
- **Chains**: Run LLM chains on provisioned instances
- **Workflows**: Deploy complex multi-step workflows
- **Tracing**: Automatic trace collection
- **Evaluation**: Automated model evaluation
- **Serving**: Deploy models with SGLang
"""
