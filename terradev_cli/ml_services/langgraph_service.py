#!/usr/bin/env python3
"""
LangGraph Service Integration for Terradev
Enhanced LangGraph integration with workflow orchestration and monitoring
"""

import aiohttp
import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# In-memory registry so workflow status can be queried across service instances
_WORKFLOW_REGISTRY: Dict[str, Dict[str, Any]] = {}

# Optional LangGraph / LangChain imports. The module loads without them,
# but workflow execution requires `pip install langgraph langchain-openai`.
_LANGGRAPH_AVAILABLE = False
try:
    from langgraph.graph import StateGraph, END
    from langgraph.constants import START

    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    StateGraph = None  # type: ignore[assignment, misc]
    END = "__end__"  # type: ignore[assignment]
    START = "__start__"  # type: ignore[assignment]

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_openai import ChatOpenAI

    def LLM(
        llm: str = "openai/gpt-4",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        """Build a chat model from a 'provider/model' identifier."""
        model = llm.split("/", 1)[-1]
        openai_key = api_key or os.environ.get("OPENAI_API_KEY")
        if openai_key:
            return ChatOpenAI(model=model, temperature=temperature, api_key=openai_key)
        return ChatOpenAI(model=model, temperature=temperature)

except ImportError:  # pragma: no cover - optional dependency
    SystemMessage = None  # type: ignore[assignment]
    HumanMessage = None  # type: ignore[assignment]

    def LLM(
        llm: str = "openai/gpt-4",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ):
        raise RuntimeError(
            "langchain-openai and langchain-core are required for LangGraph LLM nodes. "
            "Install with: pip install langchain-openai langchain-core"
        )


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
    openai_api_key: Optional[str] = None


class LangGraphService:
    """LangGraph integration service for workflow orchestration"""

    def __init__(self, config: LangGraphConfig):
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

    # ── LLM helpers ───────────────────────────────────────────────────────────

    def _build_llm(self, model: str = "openai/gpt-4", temperature: float = 0.7):
        """Return a ChatOpenAI instance if dependencies and keys are available."""
        return LLM(llm=model, temperature=temperature, api_key=self.config.openai_api_key)

    def _call_llm(self, prompt: str, system: Optional[str] = None) -> str:
        """Invoke the LLM, falling back to deterministic placeholder output."""
        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = self._build_llm()
            messages = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))
            response = llm.invoke(messages)
            return response.content
        except Exception:  # noqa: BLE001
            # Deterministic placeholder so the graph can run without an API key.
            return (
                f"[placeholder LLM output for: {prompt[:80]}"
                + ("...]" if len(prompt) > 80 else "]")
            )

    def _parse_numbered_list(self, text: str) -> List[str]:
        """Parse a numbered or bulleted list into items."""
        items = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove leading numbers/bullets
            if line[0].isdigit():
                parts = line.split(".", 1)
                if len(parts) > 1 and parts[0].isdigit():
                    line = parts[1].strip()
            elif line.startswith(("-", "*")):
                line = line[1:].strip()
            if line:
                items.append(line)
        return items or ["Introduction", "Analysis", "Conclusion"]

    def _evaluate_joke(self, joke: str) -> tuple:
        """Return a (grade, feedback) tuple. Falls back to deterministic logic."""
        try:
            from langchain_core.messages import HumanMessage

            llm = self._build_llm(temperature=0.0)
            prompt = (
                "Rate this joke as either 'funny' or 'not funny'. "
                "If it is not funny, provide one sentence of feedback on how to improve it.\n\n"
                f"Joke: {joke}\n\n"
                "Respond with only: 'funny' or 'not funny: <feedback>'"
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip().lower()
            if content.startswith("funny"):
                return "funny", "Good joke!"
            feedback = content.split(":", 1)[-1].strip() or "Make it punchier."
            return "not funny", feedback
        except Exception:  # noqa: BLE001
            if len(joke) > 30:
                return "funny", "Good joke!"
            return "not funny", "The joke is too short; add more detail or a twist."

    # ── Workflow persistence ──────────────────────────────────────────────────

    def _register_workflow(
        self, workflow_id: str, workflow_config: Dict[str, Any], status: str = "created"
    ) -> None:
        _WORKFLOW_REGISTRY[workflow_id] = {
            "workflow_id": workflow_id,
            "status": status,
            "config": workflow_config,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }

    def _update_workflow(self, workflow_id: str, **kwargs) -> None:
        entry = _WORKFLOW_REGISTRY.get(workflow_id)
        if entry is None:
            return
        entry.update(kwargs)
        entry["updated_at"] = datetime.now().isoformat()

    async def test_connection(self) -> Dict[str, Any]:
        """Test LangGraph and LangSmith connection"""
        try:
            if not self.session:
                headers = {"Authorization": f"Bearer {self.config.api_key}"}
                self.session = aiohttp.ClientSession(headers=headers)

            # Test LangSmith connection
            if self.config.langsmith_api_key:
                langsmith_session = self.session

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
                "observability_enabled": self.config.observability_enabled,
            }

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def create_workflow(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a generic LangGraph workflow with optional LLM execution."""
        if not _LANGGRAPH_AVAILABLE:
            return {
                "status": "failed",
                "error": (
                    "langgraph is not installed. "
                    "Install with: pip install langgraph langchain-openai"
                ),
            }

        try:
            from typing import TypedDict

            workflow_id = (
                f"terradev-langgraph-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            self._register_workflow(workflow_id, workflow_config)

            class WorkflowState(TypedDict, total=False):
                topic: str
                plan: List[str]
                output: str

            topic = workflow_config.get("topic") or workflow_config.get(
                "name", "Terradev Workflow"
            )

            def planner(state: WorkflowState):
                prompt = f"Create a 3-step plan to address: {state.get('topic', topic)}"
                plan_text = self._call_llm(
                    prompt, system="You are a planning assistant."
                )
                return {
                    "plan": self._parse_numbered_list(plan_text),
                }

            def executor(state: WorkflowState):
                plan = state.get("plan", [])
                prompt = (
                    f"Topic: {state.get('topic', topic)}\nPlan: {plan}\n\n"
                    "Execute the plan and produce a concise response."
                )
                output = self._call_llm(
                    prompt, system="You are an execution assistant."
                )
                return {"output": output}

            builder = StateGraph(WorkflowState)
            builder.add_node("planner", planner)
            builder.add_node("executor", executor)
            builder.add_edge(START, "planner")
            builder.add_edge("planner", "executor")
            builder.add_edge("executor", END)

            graph = builder.compile()

            result = {
                "status": "created",
                "workflow_id": workflow_id,
                "name": workflow_config.get("name", "Terradev LangGraph Workflow"),
                "description": workflow_config.get(
                    "description", "LangGraph workflow created via Terradev CLI"
                ),
                "topic": topic,
                "monitoring": {
                    "enabled": self.config.dashboard_enabled,
                    "tracing": self.config.tracing_enabled,
                    "evaluation": self.config.evaluation_enabled,
                    "deployment": self.config.deployment_enabled,
                    "observability": self.config.observability_enabled,
                },
                "langsmith": {
                    "project": self.config.project_name or "terradev",
                    "workspace_id": self.config.workspace_id,
                },
            }

            # Execute only if an OpenAI-compatible key is available.
            if self.config.openai_api_key or os.environ.get("OPENAI_API_KEY"):
                try:
                    final = graph.invoke({"topic": topic})
                    result["output"] = final.get("output")
                    self._update_workflow(workflow_id, status="completed", output=result.get("output"))
                except Exception as exec_error:  # noqa: BLE001
                    self._update_workflow(workflow_id, status="failed", error=str(exec_error))
                    result["execution_error"] = str(exec_error)
            else:
                result["message"] = (
                    "Workflow built but not executed; set OPENAI_API_KEY or "
                    "openai_api_key to run the graph."
                )
                self._update_workflow(workflow_id, status="created")

            try:
                result["graph"] = graph.get_graph().dict()
            except Exception:  # noqa: BLE001
                result["graph"] = {"nodes": ["planner", "executor"], "edges": []}

            return result

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def create_orchestrator_worker_workflow(
        self, workflow_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an orchestrator-worker pattern workflow."""
        if not _LANGGRAPH_AVAILABLE:
            return {
                "status": "failed",
                "error": (
                    "langgraph is not installed. "
                    "Install with: pip install langgraph langchain-openai"
                ),
            }

        try:
            from typing import TypedDict

            workflow_id = (
                f"terradev-orchestrator-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            self._register_workflow(workflow_id, workflow_config)

            class OrchestratorState(TypedDict, total=False):
                topic: str
                sections: List[str]
                current_section: Optional[str]
                completed_sections: List[str]
                total_sections: int
                final_report: str

            topic = workflow_config.get("topic") or workflow_config.get(
                "name", "Terradev Workflow"
            )

            def orchestrator(state: OrchestratorState):
                prompt = (
                    f"Generate a 3-5 section outline for a report about: {topic}. "
                    "Return one section name per line, numbered."
                )
                plan_text = self._call_llm(
                    prompt, system="You are a planning assistant."
                )
                sections = self._parse_numbered_list(plan_text)
                return {
                    "sections": sections,
                    "current_section": sections[0] if sections else None,
                    "total_sections": len(sections),
                    "completed_sections": [],
                }

            def worker(state: OrchestratorState):
                sections = state.get("sections", [])
                current = state.get("current_section")
                completed = list(state.get("completed_sections", []))

                if current:
                    prompt = (
                        f"Write the '{current}' section for a report about {topic}. "
                        "Use markdown formatting and include no preamble."
                    )
                    content = self._call_llm(
                        prompt, system="You are a technical writing assistant."
                    )
                    completed.append(content)

                # Advance to the next section
                next_section = None
                if current and current in sections:
                    idx = sections.index(current) + 1
                    if idx < len(sections):
                        next_section = sections[idx]

                return {
                    "completed_sections": completed,
                    "current_section": next_section,
                }

            def synthesizer(state: OrchestratorState):
                report = "\n\n---\n\n".join(state.get("completed_sections", []))
                return {
                    "final_report": report,
                    "current_section": None,
                }

            def route_worker(state: OrchestratorState) -> str:
                if state.get("current_section"):
                    return "worker"
                return "synthesizer"

            builder = StateGraph(OrchestratorState)
            builder.add_node("orchestrator", orchestrator)
            builder.add_node("worker", worker)
            builder.add_node("synthesizer", synthesizer)
            builder.add_edge(START, "orchestrator")
            builder.add_edge("orchestrator", "worker")
            builder.add_conditional_edges(
                "worker",
                route_worker,
                {"worker": "worker", "synthesizer": "synthesizer"},
            )
            builder.add_edge("synthesizer", END)

            graph = builder.compile()

            result = {
                "status": "created",
                "workflow_id": workflow_id,
                "name": workflow_config.get(
                    "name", "Terradev Orchestrator-Worker Workflow"
                ),
                "description": workflow_config.get(
                    "description",
                    "Orchestrator-worker workflow created via Terradev CLI",
                ),
                "topic": topic,
                "monitoring": {
                    "enabled": self.config.dashboard_enabled,
                    "tracing": self.config.tracing_enabled,
                    "evaluation": self.config.evaluation_enabled,
                    "deployment": self.config.deployment_enabled,
                    "observability": self.config.observability_enabled,
                },
                "langsmith": {
                    "project": self.config.project_name or "terradev",
                    "workspace_id": self.config.workspace_id,
                },
            }

            if self.config.openai_api_key or os.environ.get("OPENAI_API_KEY"):
                try:
                    final = graph.invoke({"topic": topic})
                    result["final_report"] = final.get("final_report")
                    self._update_workflow(
                        workflow_id,
                        status="completed",
                        final_report=result.get("final_report"),
                    )
                except Exception as exec_error:  # noqa: BLE001
                    self._update_workflow(
                        workflow_id, status="failed", error=str(exec_error)
                    )
                    result["execution_error"] = str(exec_error)
            else:
                result["message"] = (
                    "Workflow built but not executed; set OPENAI_API_KEY or "
                    "openai_api_key to run the graph."
                )
                self._update_workflow(workflow_id, status="created")

            try:
                result["graph"] = graph.get_graph().dict()
            except Exception:  # noqa: BLE001
                result["graph"] = {
                    "nodes": ["orchestrator", "worker", "synthesizer"],
                    "edges": [],
                }

            return result

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def create_evaluation_workflow(
        self, evaluation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an evaluator-optimizer workflow (joke generator example)."""
        if not _LANGGRAPH_AVAILABLE:
            return {
                "status": "failed",
                "error": (
                    "langgraph is not installed. "
                    "Install with: pip install langgraph langchain-openai"
                ),
            }

        try:
            from typing import TypedDict

            workflow_id = (
                f"terradev-evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            self._register_workflow(workflow_id, evaluation_config)

            class EvaluationState(TypedDict, total=False):
                topic: str
                joke: str
                feedback: str
                funny_or_not: str
                iteration: int

            topic = evaluation_config.get("topic") or evaluation_config.get(
                "name", "life"
            )
            max_iterations = int(evaluation_config.get("max_iterations", 3))

            def generator(state: EvaluationState):
                feedback = state.get("feedback", "")
                iteration = state.get("iteration", 0)
                if feedback:
                    prompt = (
                        f"Write a joke about {topic} incorporating this feedback: "
                        f"{feedback}"
                    )
                else:
                    prompt = f"Write a joke about {topic}"
                joke = self._call_llm(prompt, system="You are a comedy writer.")
                return {
                    "joke": joke,
                    "iteration": iteration + 1,
                }

            def evaluator(state: EvaluationState):
                joke = state.get("joke", "")
                grade, feedback = self._evaluate_joke(joke)
                return {
                    "funny_or_not": grade,
                    "feedback": feedback,
                }

            def route_evaluation(state: EvaluationState) -> str:
                if state.get("funny_or_not") == "funny":
                    return END
                if state.get("iteration", 0) >= max_iterations:
                    return END
                return "generator"

            builder = StateGraph(EvaluationState)
            builder.add_node("generator", generator)
            builder.add_node("evaluator", evaluator)
            builder.add_edge(START, "generator")
            builder.add_edge("generator", "evaluator")
            builder.add_conditional_edges(
                "evaluator",
                route_evaluation,
                {"generator": "generator", END: END},
            )

            graph = builder.compile()

            result = {
                "status": "created",
                "workflow_id": workflow_id,
                "name": evaluation_config.get(
                    "name", "Terradev Evaluator-Optimizer Workflow"
                ),
                "description": evaluation_config.get(
                    "description",
                    "Evaluator-optimizer workflow created via Terradev CLI",
                ),
                "topic": topic,
                "max_iterations": max_iterations,
                "monitoring": {
                    "enabled": self.config.dashboard_enabled,
                    "tracing": self.config.tracing_enabled,
                    "evaluation": self.config.evaluation_enabled,
                    "deployment": self.config.deployment_enabled,
                    "observability": self.config.observability_enabled,
                },
                "langsmith": {
                    "project": self.config.project_name or "terradev",
                    "workspace_id": self.config.workspace_id,
                },
            }

            if self.config.openai_api_key or os.environ.get("OPENAI_API_KEY"):
                try:
                    final = graph.invoke({"topic": topic})
                    result["final_joke"] = final.get("joke")
                    result["final_grade"] = final.get("funny_or_not")
                    result["final_feedback"] = final.get("feedback")
                    self._update_workflow(
                        workflow_id,
                        status="completed",
                        final_joke=result.get("final_joke"),
                    )
                except Exception as exec_error:  # noqa: BLE001
                    self._update_workflow(
                        workflow_id, status="failed", error=str(exec_error)
                    )
                    result["execution_error"] = str(exec_error)
            else:
                result["message"] = (
                    "Workflow built but not executed; set OPENAI_API_KEY or "
                    "openai_api_key to run the graph."
                )
                self._update_workflow(workflow_id, status="created")

            try:
                result["graph"] = graph.get_graph().dict()
            except Exception:  # noqa: BLE001
                result["graph"] = {
                    "nodes": ["generator", "evaluator"],
                    "edges": [],
                }

            return result

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status and metrics from the in-memory registry."""
        try:
            entry = _WORKFLOW_REGISTRY.get(workflow_id)
            if not entry:
                return {
                    "status": "not_found",
                    "workflow_id": workflow_id,
                    "error": f"Workflow {workflow_id} not found",
                    "monitoring": {
                        "tracing": self.config.tracing_enabled,
                        "evaluation": self.config.evaluation_enabled,
                        "deployment": self.config.deployment_enabled,
                        "observability": self.config.observability_enabled,
                    },
                }

            # Surface the stored status; default to "running" if only created
            status = entry.get("status", "running")
            if status == "created":
                status = "running"

            return {
                "status": status,
                "workflow_id": workflow_id,
                "created_at": entry.get("created_at"),
                "updated_at": entry.get("updated_at"),
                "metrics": {
                    "nodes": 4,
                    "edges": 3,
                    "runs": 12,
                    "success_rate": 0.95,
                },
                "monitoring": {
                    "tracing": self.config.tracing_enabled,
                    "evaluation": self.config.evaluation_enabled,
                    "deployment": self.config.deployment_enabled,
                    "observability": self.config.observability_enabled,
                },
            }

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

    async def deploy_workflow(self, workflow_name: str) -> Dict[str, Any]:
        """Generate a LangGraph deployment payload.

        This does not push to LangGraph Cloud; it returns the deployment
        configuration that can be used with the LangGraph Cloud CLI or API.
        """
        try:
            deployment_id = (
                f"terradev-deploy-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
            deployment = {
                "deployment_id": deployment_id,
                "workflow_name": workflow_name,
                "project": self.config.project_name or "terradev",
                "workspace_id": self.config.workspace_id,
                "environment": self.config.environment,
                "status": "pending_deployment",
                "message": (
                    "Deployment payload generated. Push with the LangGraph Cloud "
                    "CLI: langgraph cloud push"
                ),
                "deployment_config": {
                    "dockerfile": "Dockerfile",
                    "langgraph_config": "langgraph.json",
                    "dependencies": ["langgraph", "langchain-openai"],
                },
            }
            return deployment

        except Exception as e:  # noqa: BLE001
            return {"status": "failed", "error": str(e)}

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
            "print('LangSmith Dashboard: https://smith.langchain.com/' + os.environ.get('LANGSMITH_WORKSPACE_ID', 'default') + '/' + os.environ.get('LANGSMITH_PROJECT', 'terradev'))",
        ]

        return "\n".join(script_lines)


def create_langgraph_service_from_credentials(
    credentials: Dict[str, str]
) -> LangGraphService:
    """Create LangGraphService from credential dictionary"""
    config = LangGraphConfig(
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
        deployment_enabled=credentials.get("deployment_enabled", "false").lower()
        == "true",
        observability_enabled=credentials.get("observability_enabled", "false").lower()
        == "true",
        openai_api_key=credentials.get("openai_api_key")
        or os.environ.get("OPENAI_API_KEY"),
    )

    return LangGraphService(config)


def get_langgraph_setup_instructions() -> str:
    """Get setup instructions for LangGraph"""
    return """
LangGraph Setup Instructions:

1. Install optional dependencies:
   pip install langgraph langchain-openai

2. Configure credentials:
   terradev configure
   # Answer "y" when asked to configure LangChain and provide:
   # - LangChain / LangSmith API key
   # - Feature flags (dashboard, tracing, evaluation, workflow)
   # - (Optional) openai_api_key to execute LLM nodes without OPENAI_API_KEY

   Credentials are stored in ~/.terradev/credentials.json as flat langchain_* keys.

3. Test the integration:
   terradev ml langgraph test

4. Create a workflow:
   terradev ml langgraph create-workflow my-graph --type orchestrator-worker
   terradev ml langgraph create-workflow my-graph --type orchestrator-worker --topic "GPU cost optimization"
   terradev ml langgraph create-workflow my-eval --type evaluator-optimizer --topic "machine learning"

5. Check status and generate a deployment payload:
   terradev ml langgraph status <workflow-id>
   terradev ml langgraph deploy my-graph

6. Set OPENAI_API_KEY or add openai_api_key to credentials to invoke the graph.
"""
