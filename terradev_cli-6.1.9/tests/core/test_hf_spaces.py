"""Tests for terradev_cli.core.hf_spaces.

HFSpacesDeployer creates and configures HuggingFace Spaces deployments.
Network calls are mocked for isolation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from terradev_cli.core.hf_spaces import HFSpaceConfig, HFSpacesDeployer, HFSpaceTemplates


@pytest.fixture
def deployer():
    return HFSpacesDeployer("fake-token")


@pytest.fixture
def config():
    return HFSpaceConfig(
        name="test-space",
        model_id="meta-llama/Llama-2-7b",
        hardware="a10g-large",
        sdk="gradio",
        secrets={"API_KEY": "secret"},
        env_vars={"MODEL_ID": "meta-llama/Llama-2-7b"},
    )


def test_hf_space_config_defaults():
    """HFSpaceConfig has safe defaults."""
    cfg = HFSpaceConfig(name="x", model_id="y")
    assert cfg.hardware == "cpu-basic"
    assert cfg.sdk == "gradio"
    assert cfg.private is False
    assert cfg.secrets is None
    assert cfg.env_vars is None



@pytest.mark.asyncio
async def test_create_space_success(deployer, config, monkeypatch):
    """create_space returns created status on a successful API response."""
    deployer.aiohttp = MagicMock()

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {"name": "test-space"}

        async def text(self):
            return ""

    class FakeClientSession:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *args):
            return False

    class FakeSession:
        def post(self, *args, **kwargs):
            return FakeResponse()

    deployer.aiohttp.ClientSession = FakeClientSession
    deployer.aiohttp.FormData = MagicMock(return_value=MagicMock())

    result = await deployer.create_space(config)
    assert result["status"] == "created"
    assert result["space_name"] == "test-space"
    assert "huggingface.co/spaces" in result["space_url"]


@pytest.mark.asyncio
async def test_create_space_error(deployer, config, monkeypatch):
    """create_space surfaces API error text."""
    deployer.aiohttp = MagicMock()

    class FakeResponse:
        status = 400

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def text(self):
            return "Bad request"

    class FakeClientSession:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *args):
            return False

    class FakeSession:
        def post(self, *args, **kwargs):
            return FakeResponse()

    deployer.aiohttp.ClientSession = FakeClientSession

    result = await deployer.create_space(config)
    assert result["status"] == "error"
    assert "400" in result["error"]


def test_generate_gradio_app(deployer, config):
    """Gradio app content includes the model id."""
    content = deployer._generate_gradio_app(config)
    assert config.model_id in content
    assert "gradio" in content.lower()


def test_generate_streamlit_app(deployer, config):
    """Streamlit app content includes the model id."""
    config.sdk = "streamlit"
    content = deployer._generate_streamlit_app(config)
    assert config.model_id in content
    assert "st.title" in content


def test_generate_docker_app(deployer, config):
    """Docker app content is a placeholder."""
    config.sdk = "docker"
    content = deployer._generate_docker_app(config)
    assert config.model_id in content
    assert "Docker" in content


def test_generate_app_content_default(deployer, config):
    """Unknown SDK falls back to Gradio."""
    config.sdk = "unknown"
    content = deployer._generate_app_content(config)
    assert "gradio" in content.lower()


def test_templates():
    """HFSpaceTemplates return configs with appropriate hardware and SDK."""
    llm = HFSpaceTemplates.get_llm_template("m-1", "llm-space")
    assert llm.hardware == "a10g-large"
    assert llm.sdk == "gradio"

    emb = HFSpaceTemplates.get_embedding_template("m-2", "emb-space")
    assert emb.hardware == "cpu-upgrade"
    assert emb.sdk == "streamlit"

    img = HFSpaceTemplates.get_image_model_template("m-3", "img-space")
    assert img.hardware == "t4-medium"
    assert img.sdk == "gradio"
