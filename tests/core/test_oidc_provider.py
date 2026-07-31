"""Tests for terradev_cli.core.oidc_provider.

OIDCProvider handles discovery, PKCE, authorization URL generation, token
exchange, and user info retrieval. Network calls are mocked.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from terradev_cli.core.oidc_provider import OIDCProvider


def test_init():
    """OIDCProvider stores configuration and defaults."""
    provider = OIDCProvider(
        {
            "client_id": "cid",
            "client_secret": "secret",
            "domain": "example.auth0.com",
            "discovery_url": "https://example/.well-known/openid-configuration",
            "redirect_uri": "https://api.example/callback",
        }
    )
    assert provider.client_id == "cid"
    assert provider.client_secret == "secret"
    assert provider.redirect_uri == "https://api.example/callback"
    assert provider._expected_state is None


def test_generate_pkce_challenge():
    """PKCE challenge and verifier are generated together."""
    provider = OIDCProvider({})
    challenge = provider.generate_pkce_challenge()
    assert challenge
    assert provider.code_verifier
    assert provider.code_challenge == challenge
    assert "=" not in challenge
    assert "+" not in challenge
    assert "/" not in challenge


def test_generate_auth_url():
    """generate_auth_url produces a valid authorization URL and state."""
    provider = OIDCProvider({"client_id": "cid"})
    provider.authorization_endpoint = "https://example/auth"

    url, state = provider.generate_auth_url(scopes=["openid", "email"], state="s-1")
    assert "https://example/auth" in url
    assert "client_id=cid" in url
    assert "state=s-1" in url
    assert "code_challenge=" in url
    assert "code_challenge_method=S256" in url
    assert provider._expected_state == "s-1"


def test_generate_auth_url_generates_state():
    """When no state is supplied, a random one is generated."""
    provider = OIDCProvider({"client_id": "cid"})
    provider.authorization_endpoint = "https://example/auth"
    url, state = provider.generate_auth_url()
    assert state
    assert f"state={state}" in url
    assert provider._expected_state == state


def test_configure_google_workspace():
    """Google Workspace config returns discovery URL and scopes."""
    provider = OIDCProvider({})
    config = provider.configure_google_workspace("cid", "secret")
    assert config["client_id"] == "cid"
    assert config["client_secret"] == "secret"
    assert "accounts.google.com" in config["discovery_url"]
    assert any("userinfo.email" in scope for scope in config["scopes"])


def test_configure_auth0():
    """Auth0 config uses the tenant domain for discovery."""
    provider = OIDCProvider({})
    config = provider.configure_auth0("dev-abc.us", "cid", "secret")
    assert config["domain"] == "dev-abc.us"
    assert config["discovery_url"] == "https://dev-abc.us/.well-known/openid-configuration"


@pytest.mark.asyncio
async def test_discover_endpoints_success(monkeypatch):
    """discover_endpoints parses endpoints from the discovery document."""
    provider = OIDCProvider({"discovery_url": "https://example/.well-known"})

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {
                "authorization_endpoint": "https://example/auth",
                "token_endpoint": "https://example/token",
                "userinfo_endpoint": "https://example/userinfo",
                "jwks_uri": "https://example/jwks",
            }

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            return FakeResponse()

    fake_aiohttp = MagicMock()
    fake_aiohttp.ClientSession = MagicMock(return_value=FakeSession())
    monkeypatch.setattr("terradev_cli.core.oidc_provider.aiohttp", fake_aiohttp)

    result = await provider.discover_endpoints()
    assert result is True
    assert provider.authorization_endpoint == "https://example/auth"
    assert provider.token_endpoint == "https://example/token"


@pytest.mark.asyncio
async def test_discover_endpoints_missing_required(monkeypatch):
    """Discovery fails if required endpoints are missing."""
    provider = OIDCProvider({"discovery_url": "https://example/.well-known"})

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {"authorization_endpoint": "https://example/auth"}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            return FakeResponse()

    fake_aiohttp = MagicMock()
    fake_aiohttp.ClientSession = MagicMock(return_value=FakeSession())
    monkeypatch.setattr("terradev_cli.core.oidc_provider.aiohttp", fake_aiohttp)

    result = await provider.discover_endpoints()
    assert result is False


@pytest.mark.asyncio
async def test_exchange_code_for_tokens(monkeypatch):
    """Token exchange validates state and returns token data."""
    provider = OIDCProvider({"client_id": "cid", "client_secret": "secret"})
    provider.token_endpoint = "https://example/token"
    provider._expected_state = "s-1"
    provider.code_verifier = "verifier-1"

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {"access_token": "tok", "id_token": "id"}

        async def text(self):
            return ""

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def post(self, *args, **kwargs):
            return FakeResponse()

    fake_aiohttp = MagicMock()
    fake_aiohttp.ClientSession = MagicMock(return_value=FakeSession())
    monkeypatch.setattr("terradev_cli.core.oidc_provider.aiohttp", fake_aiohttp)

    tokens = await provider.exchange_code_for_tokens("code-1", "s-1")
    assert tokens["access_token"] == "tok"
    assert provider._expected_state is None


@pytest.mark.asyncio
async def test_exchange_code_for_tokens_bad_state():
    """Mismatched state raises a CSRF error."""
    provider = OIDCProvider({})
    provider.token_endpoint = "https://example/token"
    provider._expected_state = "s-1"

    with pytest.raises(ValueError, match="CSRF"):
        await provider.exchange_code_for_tokens("code-1", "s-2")


@pytest.mark.asyncio
async def test_get_user_info(monkeypatch):
    """get_user_info extracts normalized user data."""
    provider = OIDCProvider({})
    provider.userinfo_endpoint = "https://example/userinfo"

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {
                "sub": "u-1",
                "email": "u@example.com",
                "name": "User",
                "given_name": "U",
                "family_name": "User",
                "picture": "pic.jpg",
                "locale": "en",
                "email_verified": True,
                "groups": ["admins"],
            }

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            return FakeResponse()

    fake_aiohttp = MagicMock()
    fake_aiohttp.ClientSession = MagicMock(return_value=FakeSession())
    monkeypatch.setattr("terradev_cli.core.oidc_provider.aiohttp", fake_aiohttp)

    user = await provider.get_user_info("tok-1")
    assert user["provider_user_id"] == "u-1"
    assert user["email"] == "u@example.com"
    assert user["verified"] is True
    assert user["groups"] == ["admins"]


@pytest.mark.asyncio
async def test_fetch_jwks(monkeypatch):
    """fetch_jwks returns the decoded key set."""
    provider = OIDCProvider({})
    provider.jwks_uri = "https://example/jwks"

    class FakeResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def json(self):
            return {"keys": [{"kty": "RSA"}]}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            return FakeResponse()

    fake_aiohttp = MagicMock()
    fake_aiohttp.ClientSession = MagicMock(return_value=FakeSession())
    monkeypatch.setattr("terradev_cli.core.oidc_provider.aiohttp", fake_aiohttp)

    jwks = await provider.fetch_jwks()
    assert jwks["keys"]


@pytest.mark.asyncio
async def test_validate_id_token_rejects_without_jwt(monkeypatch):
    """validate_id_token raises if PyJWT is not installed."""
    provider = OIDCProvider({})
    monkeypatch.setattr("terradev_cli.core.oidc_provider.jwt", None)

    with pytest.raises(RuntimeError, match="PyJWT"):
        await provider.validate_id_token("header.payload.sig")
