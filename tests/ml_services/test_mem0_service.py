"""Tests for the Mem0Service integration."""

from unittest.mock import MagicMock, patch

import pytest

from terradev_cli.ml_services.mem0_service import (
    Mem0Config,
    Mem0Service,
    create_mem0_service_from_credentials,
)


class TestMem0Config:
    def test_default_config_is_hosted(self):
        cfg = Mem0Config()
        assert cfg.mode == "hosted"
        assert cfg.api_key is None

    def test_self_hosted_config_parses_json(self):
        creds = {
            "mode": "self_hosted",
            "vector_store": '{"provider": "qdrant", "config": {"host": "localhost"}}',
            "llm": '{"provider": "openai", "config": {"model": "gpt-4.1-nano"}}',
            "embedder": '{"provider": "openai"}',
        }
        with patch("terradev_cli.ml_services.mem0_service.MEM0_AVAILABLE", True):
            service, cfg = create_mem0_service_from_credentials(creds)
        assert cfg.mode == "self_hosted"
        assert cfg.vector_store == {"provider": "qdrant", "config": {"host": "localhost"}}
        assert cfg.llm == {"provider": "openai", "config": {"model": "gpt-4.1-nano"}}
        assert cfg.embedder == {"provider": "openai"}


class TestMem0ServiceUnit:
    def test_missing_mem0ai_raises_import_error(self):
        with patch(
            "terradev_cli.ml_services.mem0_service.MEM0_AVAILABLE",
            False,
        ):
            with pytest.raises(ImportError):
                Mem0Service(Mem0Config())

    def test_entity_scope_fills_defaults(self):
        cfg = Mem0Config(
            default_user_id="alice",
            default_agent_id="agent-1",
        )
        with patch("terradev_cli.ml_services.mem0_service.MEM0_AVAILABLE", True):
            service = Mem0Service(cfg)
        kwargs = service._entity_scope({"user_id": None, "agent_id": None})
        assert kwargs["user_id"] == "alice"
        assert kwargs["agent_id"] == "agent-1"

    def test_resolve_api_key_prefers_explicit(self, monkeypatch):
        monkeypatch.setenv("MEM0_API_KEY", "env-key")
        cfg = Mem0Config(api_key="explicit-key")
        with patch("terradev_cli.ml_services.mem0_service.MEM0_AVAILABLE", True):
            service = Mem0Service(cfg)
        assert service._resolve_api_key() == "explicit-key"

    def test_resolve_api_key_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("MEM0_API_KEY", "env-key")
        cfg = Mem0Config()
        with patch("terradev_cli.ml_services.mem0_service.MEM0_AVAILABLE", True):
            service = Mem0Service(cfg)
        assert service._resolve_api_key() == "env-key"


class TestMem0ClientWrappers:
    def test_add_calls_client(self):
        cfg = Mem0Config(api_key="test")
        fake_client = MagicMock()
        fake_client.add.return_value = {"id": "mem-1"}

        with patch("terradev_cli.ml_services.mem0_service.MEM0_AVAILABLE", True):
            service = Mem0Service(cfg)
            service._client = fake_client

        result = service.add([{"role": "user", "content": "hello"}], user_id="u1")
        assert result == {"id": "mem-1"}
        fake_client.add.assert_called_once()
        call_kwargs = fake_client.add.call_args.kwargs
        assert call_kwargs["user_id"] == "u1"

    def test_search_calls_client(self):
        cfg = Mem0Config(api_key="test", default_user_id="u1")
        fake_client = MagicMock()
        fake_client.search.return_value = {
            "results": [{"id": "mem-1", "memory": "hello"}]
        }

        with patch("terradev_cli.ml_services.mem0_service.MEM0_AVAILABLE", True):
            service = Mem0Service(cfg)
            service._client = fake_client

        result = service.search("hello")
        assert result["results"][0]["memory"] == "hello"
        fake_client.search.assert_called_once()
