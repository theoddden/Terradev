"""Schema and sanity tests for ML service drift contract YAMLs."""

import re
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml

CONTRACTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "terradev_cli"
    / "drift_monitor"
    / "ml_service_contracts"
)

VALID_METHODS = {"GET", "POST", "PUT", "DELETE"}


@pytest.mark.unit
class TestMlServiceContracts:
    """Validate that every ML service contract YAML is well-formed and testable."""

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
        assert data.get("auth_required") is False, (
            f"{contract_file.name}: ML service contracts should not require auth"
        )

    def test_base_url_is_valid(self, contract_file: Path) -> None:
        data = yaml.safe_load(contract_file.read_text())
        url = data.get("base_url", "")
        parsed = urlparse(url)
        assert parsed.scheme in {"http", "https"}, (
            f"{contract_file.name}: base_url must be http(s), got {data['base_url']!r}"
        )
        assert parsed.netloc, (
            f"{contract_file.name}: base_url has no host: {data['base_url']!r}"
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

            assert "path" in ep, (
                f"{contract_file.name}: enabled endpoint {ep['name']!r} needs a path"
            )
            path = str(ep["path"])
            found = set(re.findall(r"\{([a-z_]+)\}", path))
            assert not found, (
                f"{contract_file.name}: enabled endpoint {ep['name']!r} "
                f"path has placeholders: {found}"
            )

            is_text = (ep.get("content_type") or data.get("content_type")) == "text/plain"
            if is_text:
                assert "expected_text" in ep, (
                    f"{contract_file.name}: text/plain endpoint {ep['name']!r} "
                    f"must declare expected_text"
                )
            else:
                expected = ep.get("expected_response_fields", [])
                assert expected, (
                    f"{contract_file.name}: endpoint {ep['name']!r} "
                    f"must declare at least one expected_response_fields"
                )
