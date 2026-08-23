#!/usr/bin/env python3
"""Provider API Drift Monitor.

A lightweight agent that reads per-provider API contracts, performs minimal
authenticated smoke-test calls, and reports any deviation from the expected
response shape or auth behavior.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

import requests
import yaml


# -- helpers -----------------------------------------------------------------


def _collect_field_paths(obj: Any, prefix: str = "") -> Set[str]:
    """Return a set of dot-notation paths for every key in a nested response.

    For lists the first non-empty element is used as a representative shape.
    This makes ``expected_response_fields: [data, gpuTypes]`` match a RunPod
    GraphQL response whose top-level keys are ``{"data": {"gpuTypes": [...]}}``.
    """
    paths: Set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            paths.add(path)
            paths.update(_collect_field_paths(value, path))
    elif isinstance(obj, list) and obj:
        for item in obj:
            if isinstance(item, (dict, list)):
                paths.update(_collect_field_paths(item, prefix))
                break
    return paths


def _field_present(paths: Set[str], field: str) -> bool:
    """Check whether ``field`` appears as an exact or terminal path segment."""
    if field in paths:
        return True
    dotted = f".{field}"
    return any(path == field or path.endswith(dotted) for path in paths)


# -- monitor -----------------------------------------------------------------


class DriftMonitor:
    """Check provider API contracts for unexpected drift."""

    def __init__(
        self,
        contracts_dir: str,
        credentials: Dict[str, str],
        timeout: int = 30,
    ):
        self.contracts_dir = Path(contracts_dir)
        self.credentials = credentials
        self.timeout = timeout
        self.results: List[Dict[str, Any]] = []

    def run_all(self) -> List[Dict[str, Any]]:
        """Run drift detection for every contract in ``contracts_dir``."""
        self.results = []
        if not self.contracts_dir.exists():
            raise FileNotFoundError(f"Contracts directory not found: {self.contracts_dir}")

        for contract_path in sorted(self.contracts_dir.glob("*.yaml")):
            self.results.append(self.check_provider(contract_path))
        return self.results

    def check_provider(self, contract_path: Path) -> Dict[str, Any]:
        """Load and check a single provider contract."""
        with open(contract_path, "r", encoding="utf-8") as f:
            contract = yaml.safe_load(f)

        provider = contract.get("provider") or contract_path.stem
        result: Dict[str, Any] = {
            "provider": provider,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "endpoints": [],
            "drift_detected": False,
            "status": "healthy",
        }

        api_key = self.credentials.get(provider)
        if not api_key:
            result["status"] = "skipped_no_credentials"
            return result

        for endpoint in contract.get("endpoints", []):
            if not endpoint.get("enabled", True):
                continue
            endpoint_result = self._check_endpoint(contract, endpoint, api_key)
            result["endpoints"].append(endpoint_result)
            if endpoint_result.get("drift"):
                result["drift_detected"] = True
                result["status"] = "drift"

        return result

    def _check_endpoint(
        self,
        contract: Dict[str, Any],
        endpoint: Dict[str, Any],
        api_key: str,
    ) -> Dict[str, Any]:
        """Call a single endpoint and compare its response to the contract."""
        result: Dict[str, Any] = {
            "endpoint": endpoint.get("name", "unknown"),
            "drift": False,
            "drift_reasons": [],
        }

        try:
            url, headers, payload = self._build_request(contract, endpoint, api_key)
            method = endpoint.get("method", "GET").upper()

            if method == "GET":
                response = requests.get(url, headers=headers, timeout=self.timeout)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            elif method == "PUT":
                response = requests.put(url, headers=headers, json=payload, timeout=self.timeout)
            elif method == "DELETE":
                response = requests.delete(url, headers=headers, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            result["status_code"] = response.status_code
            result["auth_ok"] = response.status_code != 401
            result["missing_fields"] = []
            result["extra_fields"] = []
            result["raw_response_keys"] = []

            if response.ok:
                body = response.json()
                actual_paths = _collect_field_paths(body)
                result["raw_response_keys"] = sorted(actual_paths)

                expected = set(endpoint.get("expected_response_fields", []))
                optional = set(endpoint.get("optional_response_fields", []))
                missing = sorted(f for f in expected if not _field_present(actual_paths, f))
                result["missing_fields"] = missing

                if missing:
                    result["drift"] = True
                    result["drift_reasons"].append(f"missing field(s): {', '.join(missing)}")

                strict = endpoint.get("strict", False) or contract.get("strict", False)
                if strict and isinstance(body, dict):
                    top_level = set(body.keys())
                    allowed = expected | optional
                    extra = sorted(top_level - allowed)
                    result["extra_fields"] = extra
                    if extra:
                        result["drift"] = True
                        result["drift_reasons"].append(f"unexpected field(s): {', '.join(extra)}")
            else:
                if response.status_code in (401, 403):
                    result["drift"] = True
                    result["drift_reasons"].append("auth_header renamed or invalid")
                else:
                    result["drift"] = True
                    result["drift_reasons"].append(f"HTTP {response.status_code}")

            if result["drift"]:
                result["drift_summary"] = "; ".join(result["drift_reasons"])

            return result

        except requests.RequestException as exc:
            result["drift"] = True
            result["error"] = str(exc)
            result["drift_reasons"].append(f"request failed: {exc}")
            result["drift_summary"] = f"request failed: {exc}"
            return result
        except json.JSONDecodeError as exc:
            result["drift"] = True
            result["error"] = f"invalid JSON: {exc}"
            result["drift_reasons"].append("non-JSON response")
            result["drift_summary"] = "non-JSON response"
            return result

    def _build_request(
        self,
        contract: Dict[str, Any],
        endpoint: Dict[str, Any],
        api_key: str,
    ) -> tuple:
        """Construct the URL, headers, and payload for a contract endpoint."""
        base_url = contract["base_url"].rstrip("/")
        path = str(endpoint.get("path", "")).lstrip("/")
        url = f"{base_url}/{path}" if path else base_url

        auth_in = contract.get("auth_in", "header")
        auth_type = contract.get("auth_type", "")
        auth_header = contract.get("auth_header", "Authorization")

        headers: Dict[str, str] = {}
        if auth_in == "query":
            param = contract.get("auth_query_param", "api_key")
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{param}={api_key}"
        else:
            if auth_type:
                if auth_type.lower() == "basic":
                    token = base64.b64encode(f"{api_key}:".encode()).decode()
                    headers[auth_header] = f"Basic {token}"
                else:
                    headers[auth_header] = f"{auth_type} {api_key}"
            else:
                headers[auth_header] = api_key

            if endpoint.get("method", "GET").upper() == "POST":
                headers.setdefault("Content-Type", "application/json")

        payload: Dict[str, Any] = {}
        if endpoint.get("smoke_test_query"):
            payload["query"] = endpoint["smoke_test_query"]
        if endpoint.get("smoke_test_variables"):
            payload["variables"] = endpoint["smoke_test_variables"]
        for field in endpoint.get("required_fields", []):
            if field not in ("query", "variables") and field not in payload:
                payload[field] = endpoint.get(field)

        return url, headers, payload

    def summary(self) -> Dict[str, Any]:
        """Return a roll-up summary of the last ``run_all`` results."""
        total = len(self.results)
        drifted = [r for r in self.results if r.get("drift_detected")]
        skipped = [r for r in self.results if r.get("status") == "skipped_no_credentials"]
        healthy = [
            r
            for r in self.results
            if not r.get("drift_detected") and r.get("status") != "skipped_no_credentials"
        ]

        drift_reasons: List[str] = []
        for r in drifted:
            for ep in r.get("endpoints", []):
                if ep.get("drift"):
                    drift_reasons.append(
                        f"{r['provider']}.{ep['endpoint']}: "
                        f"{ep.get('drift_summary', ep.get('error', 'drift'))}"
                    )

        return {
            "total_providers": total,
            "healthy": len(healthy),
            "drifted": len(drifted),
            "skipped": len(skipped),
            "drift_providers": [r["provider"] for r in drifted],
            "skip_providers": [r["provider"] for r in skipped],
            "drift_reasons": drift_reasons,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "providers": self.results,
        }
