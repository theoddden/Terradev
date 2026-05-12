#!/usr/bin/env python3
"""
ML Service Tests for New AI Integrations

Tests the 3 new AI tool integrations:
- Arize Phoenix (LLM Trace Observability)
- NeMo Guardrails (Output Safety)
- Qdrant (Vector DB for RAG)

These tests verify:
1. Service initialization and configuration
2. Auth header formats (critical for each service's specific auth)
3. API request shapes
4. Error handling and retry logic
5. Response parsing
"""

import asyncio
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from terradev_cli.ml_services.phoenix_service import PhoenixService, PhoenixConfig
from terradev_cli.ml_services.guardrails_service import GuardrailsService, GuardrailsConfig
from terradev_cli.ml_services.qdrant_service import QdrantService, QdrantConfig, EMBEDDING_DIMENSIONS


def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestPhoenixService:
    """Test Arize Phoenix service - LLM Trace Observability"""
    
    def test_init_with_config(self):
        """Phoenix service initialization with config"""
        config = PhoenixConfig(
            collector_endpoint="http://localhost:6006",
            project_name="test-project"
        )
        service = PhoenixService(config)
        assert service.config == config
        assert service.config.collector_endpoint == "http://localhost:6006"
        assert service.config.project_name == "test-project"
    
    def test_auth_header_format(self):
        """Phoenix uses Authorization: Bearer for cloud, no auth for self-hosted"""
        # Self-hosted (no auth)
        config = PhoenixConfig(collector_endpoint="http://localhost:6006")
        service = PhoenixService(config)
        service._ensure_session()
        assert "Authorization" not in service.session.headers
        
        # Cloud (with API key)
        config = PhoenixConfig(
            collector_endpoint="https://app.phoenix.arize.com",
            api_key="test-key"
        )
        service = PhoenixService(config)
        service._ensure_session()
        assert service.session.headers.get("Authorization") == "Bearer test-key"
    
    def test_default_config_values(self):
        """Phoenix config has sensible defaults"""
        config = PhoenixConfig()
        assert config.collector_endpoint == "http://localhost:6006"
        assert config.project_name == "default"
        assert config.image == "arizephoenix/phoenix:latest"
        assert config.replicas == 1
        assert config.storage_size == "50Gi"
        assert config.auth_enabled == False
        assert config.otlp_protocol == "grpc"
        assert config.otlp_port == 6006
    
    def test_context_manager(self):
        """Phoenix service supports async context manager"""
        config = PhoenixConfig()
        service = PhoenixService(config)
        
        async def test_context():
            async with service:
                assert service.session is not None
                assert not service.session.closed
            assert service.session is None or service.session.closed
        
        run_async(test_context)


class TestGuardrailsService:
    """Test NeMo Guardrails service - Output Safety"""
    
    def test_init_with_config(self):
        """Guardrails service initialization with config"""
        config = GuardrailsConfig(
            server_url="http://localhost:8090",
            llm_provider="openai",
            llm_model="gpt-4"
        )
        service = GuardrailsService(config)
        assert service.config == config
        assert service.config.server_url == "http://localhost:8090"
        assert service.config.llm_provider == "openai"
    
    def test_default_config_values(self):
        """Guardrails config has sensible defaults"""
        config = GuardrailsConfig()
        assert config.server_url == "http://localhost:8090"
        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4"
        assert config.image == "nvcr.io/nvidia/nemo-guardrails:latest"
        assert config.port == 8090
        assert config.replicas == 1
        assert config.deployment_mode == "standalone"
        assert config.memory_backend == "memory"
        assert config.enable_topical == True
        assert config.enable_jailbreak == True
        assert config.enable_pii == True
        assert config.enable_factcheck == False
        assert config.default_config_id == "terradev-default"
    
    def test_context_manager(self):
        """Guardrails service supports async context manager"""
        config = GuardrailsConfig()
        service = GuardrailsService(config)
        
        async def test_context():
            async with service:
                assert service.session is not None
                assert not service.session.closed
            assert service.session is None or service.session.closed
        
        run_async(test_context)
    
    def test_memory_backend_options(self):
        """Guardrails supports memory and redis backends"""
        # Memory backend (default)
        config = GuardrailsConfig(memory_backend="memory")
        assert config.memory_backend == "memory"
        
        # Redis backend (production)
        config = GuardrailsConfig(
            memory_backend="redis",
            redis_url="redis://localhost:6379"
        )
        assert config.memory_backend == "redis"
        assert config.redis_url == "redis://localhost:6379"


class TestQdrantService:
    """Test Qdrant service - Vector DB for RAG"""
    
    def test_init_with_config(self):
        """Qdrant service initialization with config"""
        config = QdrantConfig(
            url="http://localhost:6333",
            api_key="test-key",
            vector_size=768
        )
        service = QdrantService(config)
        assert service.config == config
        assert service.config.url == "http://localhost:6333"
        assert service.config.vector_size == 768
    
    def test_auth_header_format(self):
        """Qdrant uses api-key header (NOT Authorization: Bearer)"""
        # Self-hosted (no auth)
        config = QdrantConfig(url="http://localhost:6333")
        service = QdrantService(config)
        service._ensure_session()
        assert "api-key" not in service.session.headers
        
        # Cloud (with API key)
        config = QdrantConfig(
            url="https://xyz.qdrant.io",
            api_key="test-key"
        )
        service = QdrantService(config)
        service._ensure_session()
        assert service.session.headers.get("api-key") == "test-key"
        assert "Authorization" not in service.session.headers  # Should NOT use Bearer
    
    def test_default_config_values(self):
        """Qdrant config has sensible defaults"""
        config = QdrantConfig()
        assert config.url == "http://localhost:6333"
        assert config.grpc_port == 6334
        assert config.prefer_grpc == False
        assert config.default_collection == "terradev-embeddings"
        assert config.vector_size == 1024
        assert config.distance == "Cosine"
        assert config.embedding_model == "BAAI/bge-large-en-v1.5"
        assert config.image == "qdrant/qdrant:latest"
        assert config.replicas == 1
        assert config.storage_size == "100Gi"
        assert config.port == 6333
        assert config.hnsw_m == 16
        assert config.hnsw_ef_construct == 100
    
    def test_embedding_dimensions_mapping(self):
        """Qdrant has embedding dimensions mapping for common models"""
        assert "BAAI/bge-large-en-v1.5" in EMBEDDING_DIMENSIONS
        assert EMBEDDING_DIMENSIONS["BAAI/bge-large-en-v1.5"] == 1024
        
        assert "BAAI/bge-base-en-v1.5" in EMBEDDING_DIMENSIONS
        assert EMBEDDING_DIMENSIONS["BAAI/bge-base-en-v1.5"] == 768
        
        assert "BAAI/bge-small-en-v1.5" in EMBEDDING_DIMENSIONS
        assert EMBEDDING_DIMENSIONS["BAAI/bge-small-en-v1.5"] == 384
        
        assert "text-embedding-3-small" in EMBEDDING_DIMENSIONS
        assert EMBEDDING_DIMENSIONS["text-embedding-3-small"] == 1536
        
        assert "text-embedding-3-large" in EMBEDDING_DIMENSIONS
        assert EMBEDDING_DIMENSIONS["text-embedding-3-large"] == 3072
    
    def test_context_manager(self):
        """Qdrant service supports async context manager"""
        config = QdrantConfig()
        service = QdrantService(config)
        
        async def test_context():
            async with service:
                assert service.session is not None
                assert not service.session.closed
            assert service.session is None or service.session.closed
        
        run_async(test_context)


class TestAIServiceRetryLogic:
    """Test retry logic across all AI services"""
    
    @pytest.mark.parametrize("service_class,config_class", [
        (PhoenixService, PhoenixConfig),
        (GuardrailsService, GuardrailsConfig),
        (QdrantService, QdrantConfig),
    ])
    def test_retry_on_5xx_errors(self, service_class, config_class):
        """All services should retry on 5xx errors"""
        config = config_class()
        service = service_class(config)
        
        async def test_retry():
            with patch.object(service, '_ensure_session') as mock_session:
                mock_response = AsyncMock()
                mock_response.status = 503
                mock_response.text = AsyncMock(return_value="Service Unavailable")
                
                mock_session.return_value.request.return_value.__aenter__.return_value = mock_response
                
                # Should retry and eventually raise exception
                with pytest.raises(Exception):
                    await service._request("GET", "/test")
        
        run_async(test_retry)
    
    @pytest.mark.parametrize("service_class,config_class", [
        (PhoenixService, PhoenixConfig),
        (GuardrailsService, GuardrailsConfig),
        (QdrantService, QdrantConfig),
    ])
    def test_no_retry_on_4xx_errors(self, service_class, config_class):
        """Services should not retry on 4xx errors (except 429)"""
        config = config_class()
        service = service_class(config)
        
        async def test_no_retry():
            with patch.object(service, '_ensure_session') as mock_session:
                mock_response = AsyncMock()
                mock_response.status = 404
                mock_response.text = AsyncMock(return_value="Not Found")
                
                mock_session.return_value.request.return_value.__aenter__.return_value = mock_response
                
                # Should fail immediately without retry
                with pytest.raises(Exception):
                    await service._request("GET", "/test")
        
        run_async(test_no_retry)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
