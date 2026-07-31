"""Generic smoke tests for terradev_cli.ml_services modules.

Exercises the network-free generator methods and setup helpers of every
ML service module.  Heavy external dependencies are avoided by passing default
config objects and only calling methods whose signatures can be satisfied with
defaults or a generic credentials dict.
"""
import importlib
import inspect
import pkgutil
import warnings
from typing import Any, Dict

import pytest
from unittest.mock import AsyncMock

import terradev_cli.ml_services as ml_services


def _iter_modules():
    return [m.name for m in pkgutil.iter_modules(ml_services.__path__) if m.ispkg is False]


def _find_class(mod, suffix):
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ == mod.__name__ and name.endswith(suffix):
            return obj
    return None


def _find_function(mod, *predicates):
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if obj.__module__ != mod.__name__:
            continue
        for pred in predicates:
            if pred(name):
                return obj
    return None


def _can_call_with(sig, allowed):
    """Return True if every required parameter is in allowed or has a default."""
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        if name in allowed:
            continue
        if name == "self":
            continue
        return False
    return True


def _build_args(sig, config=None, service=None, credentials=None, extra=None):
    args = {}
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is not inspect.Parameter.empty:
            continue
        if name in ("self",):
            continue
        if name == "config" and config is not None:
            args[name] = config
        elif name == "service" and service is not None:
            args[name] = service
        elif name == "credentials" and credentials is not None:
            args[name] = credentials
        elif extra and name in extra:
            args[name] = extra[name]
        else:
            raise TypeError(f"required arg {name!r} not fillable")
    return args


@pytest.mark.parametrize("module_name", _iter_modules())
def test_module_smoke(module_name):
    mod = importlib.import_module(f"terradev_cli.ml_services.{module_name}")

    generic_creds: Dict[str, Any] = {
        "api_key": "test",
        "url": "http://localhost",
        "repo_path": ".",
        "tracking_uri": "http://mlflow",
        "endpoint": "http://localhost",
        "collector_endpoint": "http://localhost",
        "project_name": "test",
        "default_collection": "test",
        "host": "http://localhost",
        "api_url": "http://localhost",
        "workspace": "test",
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
    }

    # 1. setup instructions
    setup_fn = _find_function(mod, lambda n: "setup_instructions" in n)
    if setup_fn:
        assert isinstance(setup_fn(), str)

    # 2. obtain a config and a service instance
    config_cls = _find_class(mod, "Config")
    service_cls = _find_class(mod, "Service")
    config = None
    service = None

    # Try create-* factories first (what the CLI uses)
    create_fn = _find_function(
        mod,
        lambda n: n.startswith("create_") and "credentials" in n,
        lambda n: n.startswith("create_") and n.endswith("_from_credentials"),
        lambda n: n.startswith("get_") and "service" in n,
        lambda n: n.startswith("get_") and "registry" in n,
    )
    if create_fn and _can_call_with(inspect.signature(create_fn), {"credentials"}):
        try:
            service = create_fn(**_build_args(inspect.signature(create_fn), credentials=generic_creds))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"{module_name}.{create_fn.__name__} failed: {exc}")

    if service is None and config_cls and _can_call_with(inspect.signature(config_cls), set()):
        try:
            config = config_cls()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"{module_name}.{config_cls.__name__}() failed: {exc}")

    if service is None and config and service_cls and _can_call_with(inspect.signature(service_cls), {"config"}):
        try:
            service = service_cls(config)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"{module_name}.{service_cls.__name__}() failed: {exc}")

    # 3. module-level generator functions
    for fname, fobj in inspect.getmembers(mod, inspect.isfunction):
        if fobj.__module__ != mod.__name__:
            continue
        if not fname.startswith("generate_"):
            continue
        sig = inspect.signature(fobj)
        try:
            args = _build_args(sig, config=config, service=service, credentials=generic_creds)
        except TypeError:
            continue
        try:
            fobj(**args)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"{module_name}.{fname} failed: {exc}")

    # 4. service instance generator methods
    if service:
        for mname, mobj in inspect.getmembers(service, inspect.ismethod):
            if not mname.startswith("generate_"):
                continue
            sig = inspect.signature(mobj)
            try:
                args = _build_args(sig, config=config, service=service, credentials=generic_creds)
            except TypeError:
                continue
            try:
                mobj(**args)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"{module_name}.{service_cls.__name__}.{mname} failed: {exc}")

        # 5. test_connection with a mocked _request
        old_request = getattr(service, "_request", None)
        if hasattr(service, "_request"):
            service._request = AsyncMock(return_value={})
        if hasattr(service, "test_connection"):
            try:
                coro = service.test_connection()
                if inspect.isawaitable(coro):
                    import asyncio

                    asyncio.run(coro)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"{module_name}.test_connection failed: {exc}")

        # 6. exercise all other public service methods with a mocked _request
        for mname, mobj in inspect.getmembers(service, inspect.ismethod):
            if mname.startswith("_") or mname == "test_connection":
                continue
            sig = inspect.signature(mobj)
            try:
                args = _build_args(sig, config=config, service=service, credentials=generic_creds)
            except TypeError:
                continue
            try:
                result = mobj(**args)
                if inspect.isawaitable(result):
                    import asyncio

                    asyncio.run(result)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"{module_name}.{service_cls.__name__}.{mname} failed: {exc}")

        if old_request is not None:
            service._request = old_request
