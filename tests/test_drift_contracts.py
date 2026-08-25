"""Schema and sanity tests for provider API drift contract YAMLs."""

import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "terradev_cli" / "providers" / "contracts"

VALID_METHODS = {"GET", "POST", "PUT", "DELETE"}


@pytest.mark.unit
class TestDriftContracts:
    """Validate that every drift contract YAML is well-formed and testable."""

    @pytest.fixture(params=sorted(CONTRACTS_DIR.glob("*.yaml")))
    def contract_file(self, request: pytest.FixtureRequest) -> Path:
        return request.param

    def test_contract_is_valid_yaml(self, contract_file: Path) -> None:
        data = yaml.safe_load(contract_file.read_text())
        assert data is not None, f"{contract_file.name} is empty"
        assert "provider" in data, f"{contract_file.name} missing 'provider'"
        assert "base_url" in data, f"{contract_file.name} missing 'base_url'"
        assert data["provider"] == contract_file.stem, (
            f"{contract_file.name}: provider field {data['provider']!r} "
            f"does not match filename {contract_file.stem!r}"
        )

    def test_base_url_is_valid(self, contract_file: Path) -> None:
        data = yaml.safe_load(contract_file.read_text())
        url = data.get("base_url", "")

        # Some contracts use a credential placeholder (e.g. {api_endpoint}) as
        # the base URL so it can be overridden at runtime. Substitute a dummy
        # valid URL when validating the scheme/host.
        placeholder_defaults = {
            "api_endpoint": "https://model.inferx.net/endpoints/v1",
            "aws_region": "us-east-1",
            "oci_region": "us-ashburn-1",
        }
        for placeholder, default in placeholder_defaults.items():
            url = url.replace(f"{{{placeholder}}}", default)

        parsed = urlparse(url)
        assert parsed.scheme in {"http", "https"}, (
            f"{contract_file.name}: base_url must be http(s), got {data['base_url']!r}"
        )
        assert parsed.netloc, (
            f"{contract_file.name}: base_url has no host: {data['base_url']!r}"
        )

    def test_auth_fields_are_consistent(self, contract_file: Path) -> None:
        data = yaml.safe_load(contract_file.read_text())
        assert isinstance(data.get("auth_required", True), bool), (
            f"{contract_file.name}: auth_required must be a boolean"
        )

        if data.get("auth_in") == "query":
            assert "auth_query_param" in data, (
                f"{contract_file.name}: auth_in=query requires auth_query_param"
            )

    def test_endpoints_are_well_formed(self, contract_file: Path) -> None:
        data = yaml.safe_load(contract_file.read_text())
        endpoints = data.get("endpoints", [])
        assert isinstance(endpoints, list), (
            f"{contract_file.name}: endpoints must be a list"
        )

        names = set()
        for idx, ep in enumerate(endpoints):
            assert isinstance(ep, dict), (
                f"{contract_file.name}: endpoint {idx} is not a mapping"
            )
            assert "name" in ep, f"{contract_file.name}: endpoint {idx} missing name"
            assert "method" in ep, f"{contract_file.name}: endpoint {idx} missing method"
            assert ep["method"].upper() in VALID_METHODS, (
                f"{contract_file.name}: endpoint {ep['name']!r} has invalid method {ep['method']!r}"
            )
            assert ep["name"] not in names, (
                f"{contract_file.name}: duplicate endpoint name {ep['name']!r}"
            )
            names.add(ep["name"])

    def test_enabled_endpoints_are_testable(self, contract_file: Path) -> None:
        data = yaml.safe_load(contract_file.read_text())
        for ep in data.get("endpoints", []):
            if not ep.get("enabled", True):
                continue

            # Enabled endpoints must have an expected status and expected fields.
            assert "expected_status" in ep, (
                f"{contract_file.name}: enabled endpoint {ep['name']!r} missing expected_status"
            )
            assert isinstance(ep["expected_status"], int), (
                f"{contract_file.name}: endpoint {ep['name']!r} expected_status must be int"
            )
            assert ep["expected_status"] in {200, 201, 202, 204}, (
                f"{contract_file.name}: endpoint {ep['name']!r} unexpected expected_status "
                f"{ep['expected_status']}"
            )

            if "smoke_test_query" in ep:
                # GraphQL endpoints
                assert "required_fields" in ep, (
                    f"{contract_file.name}: GraphQL endpoint {ep['name']!r} missing required_fields"
                )
            else:
                assert "path" in ep, (
                    f"{contract_file.name}: enabled endpoint {ep['name']!r} needs a path or smoke_test_query"
                )
                path = str(ep["path"])
                # Credential placeholders are resolved at runtime by the drift
                # agent, so only reject unknown placeholders.
                allowed_path_placeholders = {"project_id", "subscription_id", "aws_region", "oci_region", "zone", "bucket", "object"}
                found = set(re.findall(r"\{([a-z_]+)\}", path))
                unknown = found - allowed_path_placeholders
                assert not unknown, (
                    f"{contract_file.name}: enabled endpoint {ep['name']!r} "
                    f"path has unknown placeholder(s): {unknown}"
                )
                assert "drift-test" not in path and "drift_test" not in ep["name"], (
                    f"{contract_file.name}: endpoint {ep['name']!r} looks like a placeholder"
                )

                # Require non-empty expected_response_fields so we don't get trivial healthy reports.
                expected = ep.get("expected_response_fields", [])
                assert expected, (
                    f"{contract_file.name}: enabled endpoint {ep['name']!r} "
                    f"must declare at least one expected_response_fields"
                )

    def test_provider_module_exists(self, contract_file: Path) -> None:
        data = yaml.safe_load(contract_file.read_text())
        provider = data["provider"]
        provider_file = contract_file.parent.parent / f"{provider}_provider.py"
        if not provider_file.exists():
            provider_file = contract_file.parent.parent / f"{provider.replace('_', '')}_provider.py"
        if not provider_file.exists() and provider == "e2enetworks":
            provider_file = contract_file.parent.parent / "e2e_networks_provider.py"
        assert provider_file.exists() or provider in {"aws", "gcp", "azure"}, (
            f"{contract_file.name}: no matching provider module for {provider!r}"
        )
