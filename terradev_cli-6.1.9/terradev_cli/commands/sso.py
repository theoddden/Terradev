"""terradev sso — enterprise identity and workspace role mapping.

Configure, test, and manage SSO providers (Okta, Google Workspace, Azure AD,
SAML) so that enterprise logins map to Telinea workspace permissions.
"""

from __future__ import annotations

import click

from . import cli
from terradev_cli.core.enterprise_auth import (
    AuthProvider,
    EnterpriseAuthManager,
    UserRole,
)
from terradev_cli.core.output import get_output


@cli.group("sso")
def sso():
    """Enterprise SSO and workspace identity configuration."""
    pass


@sso.command("configure")
@click.option("--provider", required=True, type=click.Choice([p.value for p in AuthProvider if p != AuthProvider.LOCAL]), help="SSO provider")
@click.option("--client-id", help="OIDC client ID")
@click.option("--client-secret", help="OIDC client secret")
@click.option("--tenant-id", help="Azure AD tenant ID")
@click.option("--domain", help="Auth0 / custom OIDC domain")
@click.option("--entity-id", help="SAML entity ID")
@click.option("--sso-url", help="SAML SSO URL")
@click.option("--certificate", help="SAML certificate file path")
@click.option("--workspace-id", help="Telinea workspace to map users to")
@click.option("--default-role", default="viewer", type=click.Choice([r.value for r in UserRole]), help="Default role for new users")
def configure(
    provider: str,
    client_id: str,
    client_secret: str,
    tenant_id: str,
    domain: str,
    entity_id: str,
    sso_url: str,
    certificate: str,
    workspace_id: str,
    default_role: str,
):
    """Enable and configure an SSO provider for Telinea workspace auth."""
    output = get_output()
    mgr = EnterpriseAuthManager()

    config = {
        "client_id": client_id or "",
        "client_secret": client_secret or "",
        "tenant_id": tenant_id or "",
        "domain": domain or "",
        "entity_id": entity_id or "",
        "sso_url": sso_url or "",
        "certificate": certificate or "",
        "workspace_id": workspace_id or "",
        "default_role": default_role,
    }
    # Strip empty values
    config = {k: v for k, v in config.items() if v}

    mgr.enable_sso_provider(provider, config)
    output.success(f"SSO provider '{provider}' configured")
    output.set_result({"provider": provider, "workspace_id": workspace_id, "default_role": default_role})


@sso.command("list")
def list_providers():
    """List enabled SSO providers and workspace mappings."""
    output = get_output()
    mgr = EnterpriseAuthManager()
    providers = mgr.list_enabled_providers()
    if not providers:
        output.info("No SSO providers configured")
    else:
        for name in providers:
            cfg = mgr.get_sso_provider_config(name)
            workspace = cfg.get("workspace_id", "(none)")
            default_role = cfg.get("default_role", "viewer")
            output.info(f"{name}: workspace={workspace}, default_role={default_role}")
    output.set_result({"providers": providers})


@sso.command("disable")
@click.argument("provider")
def disable_provider(provider: str):
    """Disable an SSO provider without deleting its configuration."""
    output = get_output()
    mgr = EnterpriseAuthManager()
    mgr.disable_sso_provider(provider)
    output.success(f"SSO provider '{provider}' disabled")
    output.set_result({"provider": provider, "enabled": False})


@sso.command("test")
@click.argument("provider")
def test_provider(provider: str):
    """Test connectivity for a configured SSO provider."""
    output = get_output()
    mgr = EnterpriseAuthManager()
    ok = mgr.test_sso_provider(provider)
    if ok:
        output.success(f"SSO provider '{provider}' connection OK")
    else:
        output.error(f"SSO provider '{provider}' test failed")
    output.set_result({"provider": provider, "ok": ok})
