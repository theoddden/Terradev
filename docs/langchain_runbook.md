# Terradev LangChain Integration — Analysis & Runbook

## 1. Executive summary

Terradev’s LangChain integration is a lightweight, credential-aware wrapper around two related stacks:

* **LangChain / LangSmith** (`terradev_cli/ml_services/langchain_service.py`) — configuration, tracing, and observability.
* **LangGraph** (`terradev_cli/ml_services/langgraph_service.py`) — state-machine workflow orchestration (`StateGraph`).

The integration is surfaced under both the ML and agent command trees:

* `terradev ml langchain` and `terradev agent langchain`
* `terradev ml langgraph` and `terradev agent langgraph`

It is also exposed to MCP clients through the `agentic` MCP tool batch (`agent_langchain_*`, `agent_langgraph_*`).

**Important context:** `terradev ml langsmith` was removed from the CLI (per `README.md`), but several LangSmith helper methods still live inside `LangChainService` and are exercised by canary tests. They are not currently exposed as CLI or MCP tools.

## 2. What is integrated

| Capability | Status | Notes |
|---|---|---|
| `terradev ml langchain test` | Works | Calls LangSmith `/v1/organizations` if a LangSmith API key is configured. |
| `terradev ml langchain create-workflow` | Real (optional execution) | Builds a `langchain` prompt + `ChatOpenAI` chain. If `openai_api_key` or `OPENAI_API_KEY` is set, the chain is executed and the output is returned. |
| `terradev ml langchain create-langgraph` | Real (builds graph) | Delegates to `LangGraphService.create_workflow`; returns a compiled `langgraph.StateGraph`. |
| `terradev ml langchain create-pipeline` | Real (generates config) | Generates a workload-optimized SGLang serving config and a launch command. |
| `terradev ml langgraph test` | Works | Tests the LangSmith connection using the `langchain` credential bucket. |
| `terradev ml langgraph create-workflow` | Real | Builds and compiles a `langgraph.StateGraph` for `orchestrator-worker` and `evaluator-optimizer` patterns. Requires `langgraph`, `langchain-openai`, and either `openai_api_key` in credentials or `OPENAI_API_KEY`. |
| `terradev ml langgraph status <id>` | Real | Reads from an in-memory workflow registry created by the service. |
| `terradev ml langgraph deploy <name>` | Real (generates payload) | Generates a LangGraph Cloud deployment payload and instructions; does not push to LangGraph Cloud. |

## 3. Architecture & key files

| File | Purpose |
|---|---|
| `terradev_cli/ml_services/langchain_service.py` | `LangChainConfig`, `LangChainService` — handles LangSmith connection, project/run helpers, env-var generation, and setup instructions. |
| `terradev_cli/ml_services/langgraph_service.py` | `LangGraphConfig`, `LangGraphService` — real `StateGraph` builders; optional `langchain-core`/`langchain-openai` imports. |
| `terradev_cli/commands/ml.py` | Defines `ml langchain` and `ml langgraph` command groups and subcommands. |
| `terradev_cli/commands/platform.py` | Defines the `agent` group; `commands/__init__.py` re-attaches `ml.langchain` and `ml.langgraph` under `agent`. |
| `terradev_cli/commands/_api.py` | `TerradevAPI._provider_creds()` — maps flat `langchain_*` wizard keys to `LangChainConfig` / `LangGraphConfig` fields (implemented). |
| `terradev_cli/commands/providers.py` | Interactive `terradev configure` wizard that stores flat `langchain_api_key` and feature flags. |
| `terradev_cli/mcp/schemas/agentic.py` | MCP tool definitions: `agent_langchain_test`, `agent_langchain_create_workflow`, `agent_langchain_create_langgraph`, `agent_langchain_create_pipeline`, `agent_langgraph_test`, `agent_langgraph_create_workflow`, `agent_langgraph_status`, `agent_langgraph_deploy`. |
| `terradev_cli/mcp/handlers/agentic.py` | Argument builders for the MCP tools above. |
| `tests/commands/test_commands_ml.py` | CLI help/functional tests (mocks `_provider_creds`). |
| `tests/test_low_coverage_canary.py` | Canary tests for `LangChainService.get_langchain_config()` and `create_trace()`. |
| `pyproject.toml` | Optional dependencies: `langchain>=0.1.0`, `langgraph>=0.0.20` under `[project.optional-dependencies] all`. |

## 4. Installation

Install the optional dependencies needed for LangChain and LangGraph workflows:

```bash
pip install -e ".[all]"   # includes langchain, langgraph, sglang, etc.
```

Or install only what you need:

```bash
pip install langchain langgraph langchain-openai langchain-core langsmith
```

Verify imports:

```bash
python -c "import langchain, langgraph, langchain_openai; print('ok')"
```

## 5. Configuration

### 5.1 Interactive wizard (incomplete)

The easiest on-ramp is the generic credential wizard, but it only stores a subset of fields:

```bash
terradev configure
# When prompted:
# - Configure LangChain? y
# - LangChain API Key (optional)
# - Enable enhanced LangChain features? y  -> sets dashboard/tracing/evaluation/workflow flags
```

This writes flat keys to `~/.terradev/credentials.json`:

```json
{
  "langchain_api_key": "...",
  "langchain_dashboard_enabled": "true",
  "langchain_tracing_enabled": "true",
  "langchain_evaluation_enabled": "true",
  "langchain_workflow_enabled": "true"
}
```

### 5.2 Manual configuration (recommended)

`_provider_creds()` now has an explicit `langchain` branch that converts the flat `langchain_*` keys written by `terradev configure` into the `api_key`, `langsmith_api_key`, `dashboard_enabled`, etc. keys that the services expect. You can store credentials either way, but the flat wizard format is fully supported. You can also set an optional `openai_api_key` for executing LLM nodes without relying on `OPENAI_API_KEY`:

```json
{
  "langchain_api_key": "your-langchain-or-langsmith-api-key",
  "langsmith_api_key": "your-langsmith-api-key",
  "openai_api_key": "your-openai-api-key",
  "workspace_id": "your-workspace-id",
  "project_name": "terradev",
  "environment": "development",
  "langchain_dashboard_enabled": "true",
  "langchain_tracing_enabled": "true",
  "langchain_evaluation_enabled": "true",
  "langchain_workflow_enabled": "true"
}
```

If you prefer the nested format, you can still use it:

```bash
mkdir -p ~/.terradev
```

```json
{
  "langchain": {
    "api_key": "your-langchain-or-langsmith-api-key",
    "langsmith_api_key": "your-langsmith-api-key",
    "langsmith_endpoint": "https://api.smith.langchain.com",
    "workspace_id": "your-workspace-id",
    "project_name": "terradev",
    "environment": "development",
    "dashboard_enabled": "true",
    "tracing_enabled": "true",
    "evaluation_enabled": "true",
    "workflow_enabled": "true"
  }
}
```

Store this as `~/.terradev/credentials.json`. The nested format is still returned as-is when it contains real credentials.

## 6. CLI runbook

### 6.1 Test the connection

```bash
terradev ml langchain test
# equivalent:
terradev agent langchain test
```

Expected success output:

```text
OK: LangChain connected successfully
   Environment: development
   Dashboard: Enabled
   Tracing: Enabled
   Evaluation: Enabled
   Workflow: Enabled
```

If credentials are missing, it prints the setup instructions string.

### 6.2 Create a LangChain workflow

```bash
terradev ml langchain create-workflow my-workflow
terradev ml langchain create-workflow my-workflow --prompt "What is GPU oversubscription?"
terradev ml langchain create-workflow my-workflow --prompt "Explain MoE" --model openai/gpt-4o
```

Builds a `langchain` chain (prompt + `ChatOpenAI`). If `openai_api_key` is configured or `OPENAI_API_KEY` is set, the chain is executed and the output is returned. Otherwise the command reports that the workflow was built but not executed.

### 6.3 Create a LangGraph workflow

```bash
terradev ml langgraph create-workflow my-graph --type orchestrator-worker
terradev ml langgraph create-workflow my-graph --type orchestrator-worker --topic "GPU cost optimization"
terradev ml langgraph create-workflow my-eval --type evaluator-optimizer --topic "machine learning"
```

These build and compile real `langgraph.StateGraph` objects. If an OpenAI API key is available, the graph is invoked and the final report / joke is produced. If the key is missing, the graph is still compiled and the workflow metadata is returned.

### 6.4 Check workflow status

```bash
terradev ml langgraph status wf-abc
```

Reads from the in-memory workflow registry. If the workflow was created in the same process or another process in the registry, it returns the stored status (`running`/`completed`/`created`) plus monitoring flags.

### 6.5 Deploy a workflow

```bash
terradev ml langgraph deploy my-graph
```

Generates a LangGraph Cloud deployment payload (`deployment_id`, project, environment, config) and instructions. It does **not** push to LangGraph Cloud; use the output with the LangGraph Cloud CLI.

### 6.6 Create a pipeline

```bash
terradev ml langchain create-pipeline my-pipeline
terradev ml langchain create-pipeline my-pipeline --model-path meta-llama/Llama-3.1-8B-Instruct --workload-type agentic_chat
```

Generates a workload-optimized `SGLangConfig`, converts it to JSON, and prints a launch command. No live SGLang server is started.

## 7. MCP tool reference

When Terradev is run as an MCP server, the following LangChain/LangGraph tools are advertised in the `agentic` batch:

| Tool | Args | Maps to CLI |
|---|---|---|
| `agent_langchain_test` | — | `agent langchain test` |
| `agent_langchain_create_workflow` | `workflow_name` | `agent langchain create-workflow <name>` |
| `agent_langchain_create_langgraph` | `graph_name` | `agent langchain create-langgraph <name>` |
| `agent_langchain_create_pipeline` | `pipeline_name` | `agent langchain create-pipeline <name>` |
| `agent_langgraph_test` | — | `agent langgraph test` |
| `agent_langgraph_create_workflow` | `workflow_name`, `type` | `agent langgraph create-workflow <name> --type <type>` |
| `agent_langgraph_status` | `workflow_id` | `agent langgraph status <id>` |
| `agent_langgraph_deploy` | `workflow_name` | `agent langgraph deploy <name>` |

The `agentic.py` handler does not implement MCP-native logic; it converts the tool call to CLI arguments and invokes the same command body.

## 8. Programmatic use

Both services can be imported directly:

```python
from terradev_cli.ml_services.langchain_service import (
    LangChainConfig,
    LangChainService,
    create_langchain_service_from_credentials,
)

creds = {
    "api_key": "...",
    "langsmith_api_key": "...",
    "project_name": "terradev",
    "tracing_enabled": "true",
}

svc = create_langchain_service_from_credentials(creds)
result = await svc.test_connection()
```

For LangGraph:

```python
from terradev_cli.ml_services.langgraph_service import (
    LangGraphConfig,
    LangGraphService,
    create_langgraph_service_from_credentials,
)

svc = create_langgraph_service_from_credentials(creds)
result = await svc.create_orchestrator_worker_workflow({"name": "my-plan", "topic": "GPU cost optimization"})
```

## 9. Current limitations & known issues

1. **No `list-projects`, `list-runs`, or `create-trace` CLI commands.** The service methods `get_langsmith_projects`, `get_langsmith_runs`, `create_langsmith_project`, and `create_trace` exist, but no CLI or MCP tool exposes them.
2. **`LangGraphService` execution requires an OpenAI-compatible key.** The graph code now prefers the `openai_api_key` credential; if not set, it falls back to the `OPENAI_API_KEY` environment variable. Without either, graphs compile but are not executed.
3. **Workflow status is process-local.** `get_workflow_status` reads from an in-memory module registry, so status is only available to workflows created in the same Python process or until the process exits.
4. **`langgraph deploy` does not push to LangGraph Cloud.** It only generates a deployment payload and instructions. Real deployment still requires the LangGraph Cloud CLI or LangSmith UI.
5. **The `langchain` API key is mostly unused for LangSmith calls.** `LangChainService` uses `langsmith_api_key` for LangSmith calls and the base session uses `api_key` for its `Authorization: Bearer` header, but no other LangChain API call is made.
6. **The `ml` extra does not include `langchain`.** Only the `all` extra lists `langchain>=0.1.0` and `langgraph>=0.0.20`. If a user installs `pip install -e ".[ml]"` they will not get LangChain.

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: LangChain not configured. Run ... first.` | `_provider_creds("langchain")` returned empty or no `api_key`. | Use nested `~/.terradev/credentials.json` (§5.2). |
| `ERROR: Enhanced LangChain service not available.` | `terradev_cli.ml_services.langchain_service` could not be imported. | Install optional deps: `pip install langchain langgraph langchain-openai`. |
| `ModuleNotFoundError: No module named 'langgraph'` | `langgraph` not installed. | Install `langgraph` and `langchain-openai`. |
| `Workflow creation failed: langchain-openai and langchain-core are required ...` | Optional LangChain packages missing. | `pip install langchain-openai langchain-core`. |
| LangGraph workflow returns `error` with OpenAI message | `OPENAI_API_KEY` or `openai_api_key` missing. | Add `openai_api_key` to `~/.terradev/credentials.json` or export `OPENAI_API_KEY=...`. |
| `--create-workflow` or `--test` flags rejected | Those flags do not exist; the CLI uses subcommands. | Use `terradev ml langchain test` and `terradev ml langchain create-workflow <name>`. |
| `langgraph is not installed` | The `langgraph` package is missing. | Install `pip install langgraph langchain-openai`. |

## 11. Quick validation checklist

```bash
# 1. Install deps
pip install -e ".[all]"

# 2. Verify imports
python -c "import langchain, langgraph, langchain_openai; print('imports ok')"

# 3. Configure nested credentials (edit ~/.terradev/credentials.json)
#    See §5.2 for the exact JSON shape.

# 4. Test connection
terradev ml langchain test
terradev agent langgraph test

# 5. Create a LangChain workflow
terradev ml langchain create-workflow smoke-test
terradev ml langchain create-workflow explain-moe --prompt "Explain Mixture-of-Experts"

# 6. Create a real LangGraph workflow
export OPENAI_API_KEY=...
terradev ml langgraph create-workflow real-plan --type orchestrator-worker --topic "GPU cost optimization"

# 7. Check workflow status
terradev ml langgraph status <workflow-id>

# 8. Check it appears in LangSmith at https://smith.langchain.com
```

## 12. Recommendations

If you intend to make this integration production-grade, the highest-value next steps are:

1. **Persist the workflow registry.** Replace the in-memory `_WORKFLOW_REGISTRY` with a small on-disk store (e.g. `~/.terradev/langgraph_workflows.json`) so `status` works across CLI invocations.
2. **Expose LangSmith helper methods.** Add CLI/MCP subcommands for `list-projects`, `list-runs`, and `create-trace` so the existing service methods are reachable.
3. **Add real LangGraph Cloud deploy.** Change `LangGraphService.deploy_workflow` to call the LangGraph Cloud API or write a deployable `langgraph.json` and `Dockerfile` to a project directory.
4. **Add tracing by default.** When `tracing_enabled` is true, wrap graph and chain invocations with `langsmith.run_trees` or use `LangChainTracer` so runs show up in LangSmith automatically.
5. **Add support for non-OpenAI providers.** Extend the `LLM` helper to build `ChatOllama`, `ChatAnthropic`, or other chat models based on a `llm` string like `ollama/llama3`.
