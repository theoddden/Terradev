# CLI Modularisation Work Plan

**Goal:** Split `terradev_cli/cli.py` (16,270 lines / 612 KB, ~295 commands) into a composable
`commands/` package so that individual domains can be imported, tested, and maintained in isolation.

**Non-goals for this effort:**
- Rewriting business logic (that already lives in `core/`, `providers/`, `ml_services/`).
- Changing any user-facing command names or flags.
- Touching the `mcp/server.py` monolith (separate effort).

---

## Inventory

| Group / Domain | Lines (approx) | Current location |
|---|---|---|
| `TerradevAPI` class + helpers | 1 – 957 | `cli.py` |
| `cli` root + onboarding, configure, setup, quote | 958 – 2 075 | `cli.py` |
| `providers` group | 2 076 – 2 360 | `cli.py` |
| `provision`, manage, status, stage, execute, analytics, optimize, integrations, cleanup, job, run | 2 361 – 4 554 | `cli.py` |
| `infer` group (deploy, endpoint, status, failover, route) | 4 555 – 5 268 | `cli.py` |
| `k8s` group (create, destroy, list, info) | 5 715 – 5 935 | `cli.py` |
| smart_deploy, price_discovery, budget_optimize, helm_generate, percentiles, availability, reliability | 5 936 – 6 532 | `cli.py` |
| `ml` group (wandb, langchain, langgraph, kserve, langsmith, dvc, mlflow_legacy, ray, sglang, langfuse, databricks sub-groups) | 6 533 – 8 331 | `cli.py` |
| `orchestrator` group | 8 332 – 8 528 | `cli.py` |
| `warm-pool` group | 8 529 – 8 630 | `cli.py` |
| `cost_scaler` group | 8 631 – 8 749 | `cli.py` |
| `gitops` group | 8 750 – 8 946 | `cli.py` |
| `inferx` group | 8 947 – 9 375 | `cli.py` |
| preflight, flashoptim + related | 9 376 – 9 808 | `cli.py` |
| `train` group (start, status, stop, resume) | 9 576 – 10 504 | `cli.py` |
| `lora` group + sub-groups | 10 505 – 11 657 | `cli.py` |
| `sso` group | 11 658 – 12 174 | `cli.py` |
| `retrain` group | 12 175 – 12 510 | `cli.py` |
| `agentic-serving` group | 13 131 – 13 355 | `cli.py` |
| `model-router` group | 13 356 – 13 519 | `cli.py` |
| `migrate` group | 13 520 – 13 653 | `cli.py` |
| `eval` group | 13 654 – 13 850 | `cli.py` |
| export, import_cmd (standalone) | 13 851 – 14 133 | `cli.py` |
| `record` group | 14 134 – 14 263 | `cli.py` |
| `triggers` group | 14 264 – 14 445 | `cli.py` |
| `environments` group | 14 446 – 14 598 | `cli.py` |
| `lineage` group | 14 599 – 15 131 | `cli.py` |
| `mcp` command | 15 132 – 15 173 | `cli.py` |
| `local` group | 15 174 – 15 570 | `cli.py` |
| `agent` group | 15 571 – 15 980 | `cli.py` |
| `gateway` command | 15 981 – 16 105 | `cli.py` |
| `observe` group | 16 106 – 16 188 | `cli.py` |
| `schedule` group | 16 189 – 16 270 | `cli.py` |

---

## Progress

| Work Package | Status | Module |
|---|---|---|
| WP-0 Foundation | Done | `terradev_cli/commands/__init__.py`, `terradev_cli/commands/_api.py` |
| WP-1 Providers | Done | `terradev_cli/commands/providers.py` |
| WP-2 Compute / Provisioning | Done | `terradev_cli/commands/compute.py` |
| WP-3 Inference Serving | Done | `terradev_cli/commands/inference.py` |
| WP-4 Kubernetes / GitOps | Done | `terradev_cli/commands/k8s.py` (houses `gitops` for now) |
| WP-5 ML Integrations | Done | `terradev_cli/commands/ml.py` (15 sub-groups) |
| WP-6 Training / LoRA | Pending | — |
| WP-7 Infrastructure Intelligence | Pending | — |
| WP-8 MLOps / Governance | Pending | — |
| WP-9 Platform Services | Pending | — |

---

## Target Package Layout

```
terradev_cli/
  commands/
    __init__.py          # Sole owner of the root `cli` group; calls add_command() for every module
    _api.py              # TerradevAPI class, validate_credentials, run_interactive_onboarding
    providers.py         # WP-1 — providers group + configure, setup, quote
    compute.py           # WP-2 — provision, manage, status, stage, execute, analytics,
                         #         optimize, integrations, cleanup, job, run
    inference.py         # WP-3 — infer group, inferx group, smart_deploy, orchestrator,
                         #         warm-pool, cost_scaler
    k8s.py               # WP-4 — k8s group, gitops group, up, rollback, manifests, hf-space
                         #         (rename to k8s_gitops.py if desired)
    ml.py                # WP-5 — ml group and all sub-groups (wandb, langchain, langgraph,
                         #         kserve, langsmith, dvc, mlflow_legacy, ray, sglang, langfuse,
                         #         qdrant, guardrails, phoenix, databricks)
    training.py          # WP-6 — train group, lora group, flashoptim, preflight
    infrastructure.py    # WP-7 — price_discovery, budget_optimize, helm_generate,
                         #         percentiles, availability, reliability
    mlops.py             # WP-8 — lineage, environments, triggers, record, eval, migrate,
                         #         export, import_cmd, agentic-serving, model-router
    platform.py          # WP-9 — agent, observe, schedule, local, sso, mcp, gateway
  cli.py                 # Shrinks to: `from .commands import cli` — nothing else
```

The entry point in `pyproject.toml` (`terradev_cli.cli:cli`) stays **unchanged**.

---

## Cross-Cutting Engineering Decisions (agree before WP-1 starts)

### 1. Dependency Injection via `ctx.obj`

Every group module must accept its `TerradevAPI` instance through Click context rather than
constructing one internally. This is the single most important change for testability.

**Pattern to apply in every module:**

```python
# commands/infer.py
import click

@click.group()
@click.pass_context
def infer(ctx: click.Context) -> None:
    """Inference deployment and routing"""
    ctx.ensure_object(dict)
    if "api" not in ctx.obj:
        from terradev_cli.commands._api import TerradevAPI
        ctx.obj["api"] = TerradevAPI()


@infer.command("deploy")
@click.option("--model", "-m", required=True)
@click.pass_obj
def infer_deploy(obj: dict, model: str, ...) -> None:
    api = obj["api"]   # <-- injected; mockable in tests with no `patch`
    ...
```

**Test side:**

```python
def test_infer_deploy_routes_to_cheapest(runner, mock_api):
    result = runner.invoke(infer, ["deploy", "--model", "llama3"],
                           obj={"api": mock_api})
    assert result.exit_code == 0
    mock_api.get_inference_quotes.assert_called_once()
```

### 2. Extract Logic from Handlers into Testable Functions

Command handlers should be thin. Move any non-trivial logic into a `_impl` function in the same
module (or into `core/` if it's reusable). Pure functions are testable without Click at all.

```python
# Untestable (current)
def manage(instance_id, action):
    api = TerradevAPI()
    for inst in api.usage["instances_created"]:
        ...  # 150 lines mixed with print()

# Testable target
def _find_instance(instances: list, instance_id: str) -> dict | None:
    return next((i for i in instances if i["id"] == instance_id), None)

def _format_manage_result(instance: dict, action: str) -> str:
    ...  # pure, returns string

@cli.command()
@click.pass_obj
def manage(obj, instance_id, action):
    instance = _find_instance(obj["api"].usage["instances_created"], instance_id)
    click.echo(_format_manage_result(instance, action))
```

### 3. Module Registration in `commands/__init__.py`

```python
# commands/__init__.py
import click
from .providers   import providers, configure, setup, quote
from .compute     import provision, manage, status, stage, execute, \
                         analytics, optimize, integrations, cleanup, job, run
from .infer       import infer, inferx, smart_deploy, orchestrator, warm_pool, cost_scaler
from .k8s_gitops  import k8s, gitops, up, rollback, manifests
from .ml          import ml
from .training    import train, lora, flashoptim, preflight
from .infrastructure import price_discovery, budget_optimize, helm_generate, \
                          percentiles, availability, reliability
from .mlops       import lineage, environments, triggers, record, eval, migrate, \
                         export_cmd, import_cmd, agentic_serving, model_router
from .platform    import agent, observe, schedule, local, sso, mcp, gateway

@click.group()
@click.version_option(version="5.6.5", prog_name="Terradev CLI")
@click.option("--config", "-c", help="Configuration file path")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--skip-onboarding", is_flag=True)
def cli(config, verbose, skip_onboarding):
    """Terradev CLI — Cross-Cloud Compute Optimization Platform"""
    ...  # onboarding check only

# Declare the registry once
_GROUPS = [
    providers, infer, k8s, ml, orchestrator, warm_pool, cost_scaler, gitops, inferx,
    train, lora, agentic_serving, model_router, migrate, eval, record, triggers,
    environments, lineage, local, agent, observe, schedule,
]
_COMMANDS = [
    configure, setup, quote, provision, manage, status, stage, execute, analytics,
    optimize, integrations, cleanup, job, run, smart_deploy, price_discovery,
    budget_optimize, helm_generate, percentiles, availability, reliability,
    up, rollback, manifests, flashoptim, preflight, export_cmd, import_cmd,
    mcp, gateway,
]

for _g in _GROUPS:
    cli.add_command(_g)
for _c in _COMMANDS:
    cli.add_command(_c)
```

---

## Work Packages

Each work package is self-contained and can be executed in parallel by different developers once
WP-0 is merged. Estimated sizes are in half-days for an engineer already familiar with the codebase.

---

### WP-0 — Foundation (prerequisite for all) · ~3 days · owner: lead

**Creates the `commands/` scaffold and shared infrastructure. Nothing moves yet.**

Tasks:
1. Create `terradev_cli/commands/__init__.py` with the root `cli` group (copy from `cli.py` lines 958–979) and stub `add_command()` calls.
2. Create `terradev_cli/commands/_api.py` — move `TerradevAPI`, `validate_credentials`, `run_interactive_onboarding` out of `cli.py`. Update `cli.py` to import from there.
3. Update `cli.py` entry point to: `from terradev_cli.commands import cli`. The file becomes ~5 lines.
4. Verify `pyproject.toml` entry point `terradev_cli.cli:cli` still resolves.
5. Add `pytest.ini` markers: `unit`, `integration`, `slow` — update `pyproject.toml`:
   ```toml
   [tool.pytest.ini_options]
   markers = [
     "unit: pure-function tests, no I/O",
     "integration: requires mocked external APIs",
     "slow: full CLI CliRunner tests",
   ]
   ```
6. Move `tests/conftest.py` fixture `mock_api` to import from `commands._api` not `cli`.
7. Add a `tests/commands/` directory with a shared `conftest.py` that provides `runner` and
   `mock_api` fixtures at the command-module level.
8. Write smoke test: `pytest -m "not slow" tests/commands/` passes (empty suite, exit 0).
9. Run full existing test suite — must pass before any WP-1+ work starts.

**Definition of done:** `from terradev_cli.commands import cli` works, all existing tests pass,
`cli.py` is ≤20 lines.

---

### WP-1 — Providers · ~2 days · owner: A

**Extracts:** `providers` group (list_profiles, load_profiles, show_profile, remove_profile, export_example),
`configure`, `setup`, `quote` commands.
**Source lines:** 958 – 2 360

**Tests to write in `tests/commands/test_providers.py`:**
- `test_providers_list_empty` — no profiles, exit 0, expected output
- `test_providers_load_valid_yaml` — load a fixture YAML, verify profile count
- `test_providers_load_invalid_yaml` — bad file, exit non-zero, error message present
- `test_configure_provider_saves_credentials` — mock `AuthManager.save`, assert called
- `test_configure_unknown_provider` — exit 1
- `test_quote_returns_sorted_by_price` — mock all provider quote methods, assert ordering
- `test_quote_filters_by_provider` — `--providers runpod`, assert other providers not called
- `test_setup_quick_mode` — `--quick` flag, mock `TerradevAPI`, assert wizard skipped

**Key `_impl` extractions:**
- `_sort_quotes(quotes: list[dict]) -> list[dict]` — pure sort + filter
- `_validate_yaml_profile(data: dict) -> list[str]` — returns list of validation errors

---

### WP-2 — Compute / Provisioning · ~4 days · owner: B

**Extracts:** `provision`, `manage`, `status`, `stage`, `execute`, `analytics`, `optimize`,
`integrations`, `cleanup`, `job`, `run`.
**Source lines:** 2 361 – 4 554 (includes the ~1 000-line `provision` handler)

**This is the largest and most complex WP.** `provision` contains inline quota checks, retry
loops, provider selection, and output formatting — all of which need `_impl` extraction.

**Tests to write in `tests/commands/test_compute.py`:**
- `test_provision_selects_cheapest_provider` — mock quotes, assert chosen provider
- `test_provision_respects_max_price_filter` — quotes above `--max-price` excluded
- `test_provision_dry_run_does_not_call_provision_instance`
- `test_manage_status_known_instance`
- `test_manage_status_unknown_instance_exit_1`
- `test_manage_terminate_calls_terminate_instance`
- `test_status_live_calls_get_instance_status`
- `test_cleanup_removes_old_instances` — mock `datetime.now`, assert `save_usage` called
- `test_analytics_formats_json` — `--format json`, assert valid JSON in output
- `test_execute_async_flag`

**Key `_impl` extractions:**
- `_select_best_provider(quotes: list, max_price: float, gpu_type: str) -> dict`
- `_filter_quotes(quotes: list, *, max_price, provider_filter, region) -> list`
- `_format_instances_table(instances: list) -> str`
- `_find_instance(instances: list, instance_id: str) -> dict | None`

---

### WP-3 — Inference Serving · ~3 days · owner: C

**Extracts:** `infer` group (deploy, endpoint, status, failover, route), `inferx` group,
`smart_deploy`, `orchestrator` group, `warm-pool` group, `cost_scaler` group.
**Source lines:** 4 555 – 5 268 + 8 332 – 8 946

**Tests to write in `tests/commands/test_infer.py`:**
- `test_infer_deploy_model_required` — missing `--model`, exit 2
- `test_infer_deploy_routes_to_cheapest_provider`
- `test_infer_endpoint_dry_run_no_provision`
- `test_infer_status_all_healthy`
- `test_infer_failover_dry_run_lists_candidates`
- `test_infer_route_strategy_choices` — invalid strategy, exit 2
- `test_smart_deploy_recommends_provider` — mock quotes, assert recommendation present
- `test_inferx_configure_saves_api_key`
- `test_orchestrator_start_invalid_policy` — exit 2
- `test_warm_pool_start_invalid_strategy` — exit 2

**Key `_impl` extractions:**
- `_build_endpoint_spec(model, gpu_type, min_workers, max_workers) -> dict`
- `_select_inference_provider(quotes, constraints) -> dict`

---

### WP-4 — Kubernetes / GitOps · ~2 days · owner: D

**Extracts:** `k8s` group (create, destroy, list, info), `gitops` group, `up`, `rollback`,
`manifests`, `hf-space`.
**Source lines:** 5 715 – 5 935 + 8 750 – 8 946 + 7 872 – 8 331

**Note:** `k8s` is already declared as `@click.group()` (not `@cli.group()`) — it is the only
group in `cli.py` that already follows the target pattern. Move it first as a template.

**Tests to write in `tests/commands/test_k8s_gitops.py`:**
- `test_build_k8s_job_manifest_training_workload` — pure function, no mock needed
- `test_build_k8s_job_manifest_budget_under_2_forces_spot`
- `test_build_k8s_job_manifest_env_vars_parsed`
- `test_k8s_create_calls_terraform_wrapper` — mock `TerraformWrapper`
- `test_k8s_destroy_prompts_confirmation`
- `test_k8s_list_empty`
- `test_gitops_group_has_expected_subcommands`
- `test_up_dry_run_no_apply`
- `test_rollback_requires_version`

**Key `_impl` extractions:**
- `_build_k8s_job_manifest(...)` — already pure, just move it out of `cli.py`

---

### WP-5 — ML Integrations · ~4 days · owner: E

**Extracts:** `ml` group and all 13 sub-groups: wandb, langchain, langgraph, kserve, langsmith,
dvc, mlflow_legacy, ray, sglang, langfuse, databricks (+ mlflow sub-group), retrain.
**Source lines:** 6 533 – 8 331 + 12 175 – 12 510

This is the widest WP. All handlers are thin wrappers over `ml_services/` — the extraction is
mostly mechanical. Sub-group ownership can be split further within E's team if needed.

**Tests to write in `tests/commands/test_ml.py`:**
- `test_wandb_test_requires_api_key` — mock `WandbService`, assert configured check
- `test_wandb_list_projects_formats_table`
- `test_langchain_create_workflow_calls_service`
- `test_ray_start_calls_ray_service`
- `test_sglang_deploy_missing_model_exit_2`
- `test_langfuse_configure_saves_keys`
- `test_databricks_configure_saves_host_token`
- `test_retrain_drift_calls_drift_service`
- `test_ml_group_has_all_expected_subgroups` — structural test, no I/O

---

### WP-6 — Training / LoRA · ~3 days · owner: F

**Extracts:** `train` group (start, status, stop, resume), `lora` group (+ MIG sub-group,
adapter sub-group), `flashoptim`, `preflight`.
**Source lines:** 9 376 – 11 657

**Tests to write in `tests/commands/test_training.py`:**
- `test_train_start_requires_gpu`
- `test_train_start_dry_run_no_provision`
- `test_train_status_job_not_found`
- `test_train_stop_calls_terminate`
- `test_train_resume_checkpoint_required`
- `test_lora_register_saves_adapter`
- `test_lora_list_empty`
- `test_lora_activate_unknown_adapter_exit_1`
- `test_lora_drift_check_calls_lora_consistency`
- `test_lora_cost_report_formats_json`
- `test_flashoptim_invalid_precision` — exit 2
- `test_preflight_quick_mode_skips_benchmarks`

---

### WP-7 — Infrastructure Intelligence · ~1.5 days · owner: G

**Extracts:** `price_discovery`, `budget_optimize`, `helm_generate`, `percentiles`,
`availability`, `reliability`.
**Source lines:** 5 936 – 6 532

These are mostly read-only query commands — easiest WP to test because they don't mutate state.

**Tests to write in `tests/commands/test_infrastructure.py`:**
- `test_price_discovery_filters_by_region`
- `test_price_discovery_shows_trends_flag`
- `test_budget_optimize_returns_within_budget` — mock quotes, assert all results ≤ budget
- `test_budget_optimize_no_results_message`
- `test_helm_generate_produces_valid_yaml` — assert output parseable by `yaml.safe_load`
- `test_percentiles_requires_gpu_type`
- `test_availability_window_default`
- `test_reliability_ranking_flag`

---

### WP-8 — MLOps / Governance · ~3 days · owner: H

**Extracts:** `lineage`, `environments`, `triggers`, `record`, `eval`, `migrate`, `export`,
`import_cmd`, `agentic-serving`, `model-router`.
**Source lines:** 13 131 – 15 131

**Tests to write in `tests/commands/test_mlops.py`:**
- `test_lineage_register_artifact_saves`
- `test_lineage_graph_direction_choices`
- `test_lineage_export_formats` — json / yaml / dot
- `test_environments_list_all`
- `test_environments_promote_requires_approval`
- `test_triggers_create_persists`
- `test_triggers_fire_unknown_type_exit_1`
- `test_eval_compare_requires_dataset`
- `test_migrate_dry_run_no_changes`
- `test_agentic_serving_configure_saves`
- `test_model_router_classify_returns_route`
- `test_export_produces_valid_yaml`
- `test_import_cmd_validate_only_no_write`

---

### WP-9 — Platform Services · ~2 days · owner: I

**Extracts:** `agent`, `observe`, `schedule`, `local`, `sso`, `mcp`, `gateway`.
**Source lines:** 11 658 – 12 174 + 15 132 – 16 270

**Tests to write in `tests/commands/test_platform.py`:**
- `test_agent_plan_formats_topology`
- `test_agent_deploy_dry_run`
- `test_agent_teardown_requires_confirmation`
- `test_observe_gateway_registers_routes`
- `test_observe_status_no_trace`
- `test_schedule_job_invalid_cron` — exit 1
- `test_schedule_list_empty`
- `test_local_scan_no_hosts_exit_1`
- `test_sso_group_has_expected_subcommands`
- `test_mcp_start_action`
- `test_gateway_start_binds_port`

---

## Shared Test Utilities to Add in `tests/commands/conftest.py`

```python
import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock, AsyncMock
from terradev_cli.commands._api import TerradevAPI

@pytest.fixture
def runner():
    return CliRunner(mix_stderr=False)

@pytest.fixture
def mock_api(tmp_path):
    api = MagicMock(spec=TerradevAPI)
    api.credentials = {"runpod": {"api_key": "test-key"}}
    api.config_dir = tmp_path / ".terradev"
    api.usage = {"instances_created": [], "inference_endpoints": []}
    api.is_first_time_user.return_value = False
    api.get_runpod_quotes = AsyncMock(return_value=[])
    # ... (full mock matching existing conftest.py pattern)
    return api
```

---

## Execution Order & Dependencies

```
WP-0 (Foundation)
    └── All WPs can start in parallel after WP-0 merges
            ├── WP-1 (Providers)            ← quickest, go first for morale
            ├── WP-4 (K8s/GitOps)          ← k8s group already isolated
            ├── WP-7 (Infrastructure)       ← read-only, easiest tests
            ├── WP-2 (Compute)              ← biggest, start early
            ├── WP-3 (Inference)            ← start with WP-2 owner if same team
            ├── WP-5 (ML Integrations)      ← wide but mechanical
            ├── WP-6 (Training/LoRA)        ← depends on WP-5 fixtures
            ├── WP-8 (MLOps/Governance)     ← can start immediately
            └── WP-9 (Platform Services)    ← can start immediately
```

---

## CI Integration

Add to `.github/workflows/` (or `.gitlab-ci.yml`):

```yaml
unit-tests:
  script:
    - pytest -m unit tests/commands/ -x --tb=short -q
  # Runs in <30s — suitable for every commit and PR

integration-tests:
  script:
    - pytest -m integration tests/commands/ tests/ -x --tb=short
  # Runs on PR merge to main

slow-tests:
  script:
    - pytest tests/ --tb=short
  # Full suite — nightly or pre-release
```

---

## Success Metrics

| Metric | Before | Target after all WPs |
|---|---|---|
| `cli.py` size | 16,270 lines | ≤ 20 lines |
| Import time of any single command module | ~2-3 s (full cli.py) | < 200 ms |
| Test files that import from `cli.py` directly | ~15 | 0 |
| Unit-marked tests (pure function, no I/O) | 0 | ≥ 80 |
| Command-level test coverage | ~30% (CliRunner only) | ≥ 70% |
| Commands testable without `patch('terradev_cli.cli.TerradevAPI')` | 0 | 100% |
