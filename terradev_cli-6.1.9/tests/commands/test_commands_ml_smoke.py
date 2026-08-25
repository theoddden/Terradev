"""
Brute-force smoke test for the `ml` command tree.

Each leaf subcommand is invoked with auto-generated arguments and mocked
async/subprocess/os I/O. Service modules are replaced with thin fakes so the
command bodies exercise dispatch/argument handling without real network calls.
"""

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
from click.testing import CliRunner

from terradev_cli.commands import cli


# ── Fake ml_services modules ──────────────────────────────────────────────────


def _result_for(name: str):
    """Return a sensible result for an ml_services method/function name."""
    lowered = name.lower()

    if "test_connection" in lowered or "dashboard_status" in lowered or "monitoring_status" in lowered:
        return _SafeDict(
            {
                "status": "connected",
                "repo_path": "/tmp",
                "tracking_uri": "http://localhost:5000",
                "namespace": "default",
                "collector_endpoint": "http://localhost:8080",
                "server_url": "http://localhost:8000",
                "langsmith": "http://localhost",
                "base_url": "http://localhost",
                "host": "http://localhost",
                "clusters": 1,
                "project": "test-project",
                "collections": ["test"],
                "projects": ["test"],
                "project_names": ["test"],
                "dashboard": {"id": "1"},
                "report": {"id": "1"},
                "alerts": [],
                "workflow_id": "wf-1",
                "pipeline_id": "p-1",
                "collection": {"name": "test"},
                "input": "hello",
                "output": "world",
                "name": "test",
                "id": "1",
                "entity": "test",
                "url": "http://localhost",
                "error": "",
            }
        )

    if "test_rail" in lowered:
        return _SafeDict({"input": "hello", "config_id": "cfg-1", "output": {"text": "hello"}})

    if "get_collection_info" in lowered:
        return _SafeDict({"name": "test", "vectors_count": 0, "status": "green"})

    if "export_training_data" in lowered:
        return [{"instruction": "hi", "response": "hello", "score": 0.9}]

    return MagicMock()


class FakeService:
    """Generic service stand-in whose methods return MagicMock or JSON-safe dicts."""

    def __init__(self, name=""):
        self._name = name

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def _method(*args, **kwargs):
            return _result_for(name)

        return _method


class FakeVLLMConfig:
    """Stand-in for the vLLMConfig class used by the vllm commands."""

    def __init__(self, model_name=None, tensor_parallel_size=1, **kwargs):
        self.model_name = model_name or kwargs.get("model_name") or ""
        self.tensor_parallel_size = tensor_parallel_size
        self.host = kwargs.get("host", "127.0.0.1")
        self.port = kwargs.get("port", 8000)
        self.api_key = kwargs.get("api_key")
        self.gpu_memory_utilization = 0.9
        self.max_num_batched_tokens = 4096
        self.max_num_seqs = 256
        self.enable_prefix_caching = True
        self.enable_chunked_prefill = True
        self.cpu_cores = 2

    @classmethod
    def create_throughput_optimized(cls, model_name, tensor_parallel_size=1):
        return cls(model_name, tensor_parallel_size)

    @classmethod
    def create_latency_optimized(cls, model_name, tensor_parallel_size=1):
        return cls(model_name, tensor_parallel_size)


class FakeVLLMService:
    """Stand-in for the VLLMService class used by the vllm optimize command."""

    def __init__(self, config):
        self.config = config

    def _build_server_args(self):
        return ["--model", self.config.model_name]


def _make_ml_services_fakes() -> dict:
    """Return a mapping of terradev_cli.ml_services.* module names to fake modules."""
    repo_root = Path(__file__).resolve().parents[2]
    ml_services_dir = repo_root / "terradev_cli" / "ml_services"
    fakes = {}

    for py_file in ml_services_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        full_name = f"terradev_cli.ml_services.{py_file.stem}"
        mod = types.ModuleType(full_name)
        mod.__file__ = str(py_file)

        def _make_getattr(mod_name=full_name):
            def _fake_getattr(name):
                if mod_name == "terradev_cli.ml_services.vllm_service":
                    if name == "VLLMConfig":
                        return FakeVLLMConfig
                    if name == "VLLMService":
                        return FakeVLLMService

                if name.startswith("create_") and "service" in name:

                    def _factory(*args, **kwargs):
                        return FakeService(name)

                    return _factory

                def _func(*args, **kwargs):
                    return _result_for(name)

                return _func

            return _fake_getattr

        mod.__getattr__ = _make_getattr()
        fakes[full_name] = mod
    return fakes


FAKE_ML_SERVICES = _make_ml_services_fakes()


# ── Smart asyncio.run replacement ─────────────────────────────────────────────


class _SafeDict(dict):
    """dict that returns a MagicMock for missing keys instead of raising KeyError."""

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return MagicMock()


def _co_name(coro) -> str:
    code = getattr(coro, "cr_code", None)
    if code:
        return code.co_name
    return ""


def _fake_asyncio_run(coro, *args, **kwargs):
    """Intercept asyncio.run inside ml.py so real coroutines never execute."""
    if asyncio.iscoroutine(coro):
        coro.close()
        return _result_for(_co_name(coro))
    return coro


# ── Argument generation from Click introspection ──────────────────────────────


def _value_for_param(param, index: int = 0) -> str:
    """Return a concrete argument/option value for a Click parameter."""
    param_name = getattr(param, "name", "arg")
    hint = param_name.lower()

    choices = getattr(param, "type", None)
    if isinstance(choices, click.Choice):
        return choices.choices[0]

    if isinstance(choices, click.Path):
        return "/tmp"

    if isinstance(choices, click.IntRange) or isinstance(choices, click.types.IntParamType):
        return "1"
    if isinstance(choices, click.FloatRange) or isinstance(choices, click.types.FloatParamType):
        return "1.0"

    if "path" in hint or "dir" in hint or "file" in hint:
        return "/tmp"
    if "url" in hint or "endpoint" in hint:
        return "http://localhost"
    if "id" in hint:
        return f"test-{index}"
    if "name" in hint or "project" in hint:
        return "test-project"
    if "command" in hint or "script" in hint:
        return "echo hello"
    if "query" in hint or "sql" in hint:
        return "SELECT 1"
    if "format" in hint:
        return "json"
    if "tag" in hint:
        return "tag1,tag2"
    return "test"


def _build_args_for_command(command: click.Command) -> list:
    """Build a minimal argv for a Click command."""
    args: list = []
    arg_index = 0

    for param in command.params:
        if isinstance(param, click.Argument):
            nargs = getattr(param, "nargs", 1)
            if nargs == -1:
                nargs = 2
            for _ in range(nargs):
                args.append(_value_for_param(param, arg_index))
                arg_index += 1

        elif isinstance(param, click.Option):
            long_flag = next((o for o in param.opts if o.startswith("--")), param.opts[0] if param.opts else f"--{param.name}")
            if param.is_flag:
                if param.default is True:
                    args.append(long_flag)
                continue
            if param.required or param.default is None:
                args.append(long_flag)
                args.append(_value_for_param(param, arg_index))
                arg_index += 1

    return args


def _iter_leaf_commands(group: click.Group, prefix: list = None):
    """Yield (path, command) for every leaf subcommand under `group`."""
    prefix = prefix or []
    for name, cmd in group.commands.items():
        path = prefix + [name]
        if isinstance(cmd, click.Group):
            yield from _iter_leaf_commands(cmd, path)
        else:
            yield path, cmd


# ── Test ──────────────────────────────────────────────────────────────────────


def test_all_ml_commands_invoke_without_unhandled_exception(mock_api):
    """Run every leaf `ml` subcommand with fake I/O and surface failures."""
    runner = CliRunner()
    ml_group = cli.commands["ml"]

    mock_api._provider_creds.return_value = {
        "api_key": "test",
        "api_endpoint": "http://localhost",
        "tracking_uri": "http://localhost:5000",
        "repo_path": "/tmp",
    }
    mock_api._save_provider_creds = MagicMock()

    fake_proc = MagicMock()
    fake_proc.communicate.return_value = (b"", b"")
    fake_proc.poll.return_value = 0
    fake_proc.wait.return_value = 0
    fake_proc.returncode = 0

    failures = []
    with patch.dict(sys.modules, FAKE_ML_SERVICES):
        with patch("asyncio.run", _fake_asyncio_run):
            with patch("terradev_cli.commands.ml.subprocess.check_output", return_value=b""):
                with patch("terradev_cli.commands.ml.subprocess.run", return_value=fake_proc):
                    with patch("terradev_cli.commands.ml.subprocess.Popen", return_value=fake_proc):
                        with patch("terradev_cli.commands.ml.os.system", return_value=0):
                            with patch("terradev_cli.commands.ml.time.sleep", return_value=None):
                                for path, cmd in _iter_leaf_commands(ml_group, ["ml"]):
                                    full_path = path
                                    argv = full_path + _build_args_for_command(cmd)
                                    try:
                                        result = runner.invoke(
                                            cli, argv, obj={"api": mock_api}, input="test\n" * 10
                                        )
                                    except Exception as exc:  # noqa: BLE001
                                        failures.append(f"{'.'.join(full_path)}: runner raised {exc}")
                                        continue

                                    if result.exception and not isinstance(
                                        result.exception, (click.ClickException, click.exceptions.Exit, SystemExit)
                                    ):
                                        failures.append(
                                            f"{'.'.join(full_path)}: {type(result.exception).__name__}: {result.exception}"
                                        )

    assert not failures, "ml commands raised unhandled errors:\n" + "\n".join(failures[:30])
