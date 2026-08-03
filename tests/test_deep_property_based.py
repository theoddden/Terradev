#!/usr/bin/env python3
"""Property-based / fuzz tests for parsers, schemas, and MCP tool dispatch.

These tests use Hypothesis to generate malformed, unexpected, or adversarial
inputs and verify that the CLI never raises unhandled exceptions.  It must
either return a valid result or raise a clean, expected error type.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from unittest.mock import AsyncMock, patch

from terradev_cli.core.config_validator import ConfigValidator
from terradev_cli.providers import gpu_catalog


def _json_object() -> st.SearchStrategy[Dict[str, Any]]:
    """Generate flat JSON-serializable objects."""
    return st.dictionaries(
        st.text(min_size=0, max_size=30),
        st.one_of(
            st.text(min_size=0, max_size=50),
            st.integers(min_value=-(10**6), max_value=10**6),
            st.booleans(),
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9),
            st.none(),
        ),
        min_size=0,
        max_size=10,
    )


class TestConfigValidatorFuzz:
    """Property: validate() is total on well-formed JSON inputs."""

    @given(schema=_json_object(), config=_json_object())
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_validate_never_crashes_on_json_objects(self, schema: Dict[str, Any], config: Dict[str, Any]):
        schema_json = json.dumps(schema)
        config_json = json.dumps(config)

        validator = ConfigValidator(schema_json)
        result = validator.validate(config_json)

        assert isinstance(result, dict)
        assert "is_valid" in result
        assert "errors" in result
        assert isinstance(result["is_valid"], bool)
        assert isinstance(result["errors"], list)


class TestGPUCatalogFuzz:
    """Property: normalize() is total on all strings and never crashes."""

    @given(name=st.text(min_size=0, max_size=100))
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_normalize_never_crashes(self, name: str):
        result = gpu_catalog.normalize(name)

        assert result is None or isinstance(result, gpu_catalog.GPUDescriptor)


class TestMCPHandleCallToolFuzz:
    """Property: handle_call_tool() never raises an unhandled exception.

    No matter what tool name or arguments the MCP client sends, the server
    must return a CallToolResult (success or isError=True).
    """

    @pytest.fixture(autouse=True)
    def _patch_mcp_side_effects(self, monkeypatch):
        """Prevent real commands, Terraform, or GPU discovery from running."""
        import terradev_cli.mcp.server as server

        monkeypatch.setattr(server, "_ensure_tools_loaded", AsyncMock(return_value=None))
        monkeypatch.setattr(server, "execute_terradev_command", AsyncMock(return_value={"success": True, "stdout": "mock"}))
        monkeypatch.setattr(server, "execute_terraform_command", AsyncMock(return_value={"success": True, "stdout": "mock"}))
        monkeypatch.setattr(server, "execute_terraform_parallel", AsyncMock(return_value={"success": True, "stdout": "mock"}))
        monkeypatch.setattr(server, "discover_local_gpus", AsyncMock(return_value={"has_local_gpu": False, "local_devices": []}))
        monkeypatch.setattr(server, "NEW_COMMAND_MAP", {
            "create_sqlite_connection": AsyncMock(return_value=[{"text": "ok"}]),
            "create_postgresql_connection": AsyncMock(return_value=[{"text": "ok"}]),
            "query_database": AsyncMock(return_value=[{"text": "ok"}]),
            "upsert_database": AsyncMock(return_value=[{"text": "ok"}]),
            "get_database_connection": AsyncMock(return_value=[{"text": "ok"}]),
        })

    @pytest.mark.asyncio
    @given(
        tool_name=st.one_of(
            st.sampled_from([
                "quote_gpu",
                "provision_gpu",
                "preflight_provision",
                "local_scan",
                "terraform_plan",
                "k8s_create",
                "create_sqlite_connection",
                "",
            ]),
            st.text(min_size=0, max_size=50),
        ),
        arguments=st.one_of(
            st.none(),
            st.dictionaries(
                st.text(min_size=0, max_size=30),
                st.one_of(
                    st.text(min_size=0, max_size=50),
                    st.integers(min_value=-10, max_value=10),
                    st.booleans(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.lists(st.integers()),
                    st.none(),
                ),
                min_size=0,
                max_size=8,
            ),
        ),
    )
    @settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow], deadline=5000)
    async def test_handle_call_tool_never_crashes(self, tool_name, arguments):
        from mcp.types import CallToolResult
        from terradev_cli.mcp.server import handle_call_tool

        args = arguments if arguments is not None else {}

        result = await handle_call_tool(tool_name, args)

        assert isinstance(result, CallToolResult)
        assert result.content is not None
