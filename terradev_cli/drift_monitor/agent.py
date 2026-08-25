#!/usr/bin/env python3
"""Provider API Drift Monitor.

A lightweight agent that reads per-provider API contracts, performs minimal
authenticated smoke-test calls, and reports any deviation from the expected
response shape or auth behavior.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set

import requests
import yaml

# Optional signing libraries used for provider-specific auth.
try:
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials as AwsCredentials
except ImportError:  # pragma: no cover
    SigV4Auth = None  # type: ignore
    AWSRequest = None  # type: ignore
    AwsCredentials = None  # type: ignore

try:
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request as GoogleRequest
except ImportError:  # pragma: no cover
    service_account = None  # type: ignore
    GoogleRequest = None  # type: ignore

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


# -- helpers -----------------------------------------------------------------


def _xml_to_dict(element) -> Any:
    """Recursively convert an ElementTree element to a dict or list."""
    children = list(element)
    if not children:
        return element.text or ""

    # If all children share the same tag, treat them as a list.
    tag_counts: Dict[str, int] = {}
    for child in children:
        tag_counts[child.tag] = tag_counts.get(child.tag, 0) + 1

    result: Any = {}
    for child in children:
        child_value = _xml_to_dict(child)
        if tag_counts[child.tag] > 1:
            if child.tag not in result:
                result[child.tag] = []
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(child_value)
        else:
            result[child.tag] = child_value
    return result


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
        credentials: Dict[str, Union[str, Dict[str, Any]]],
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

        auth_required = contract.get("auth_required", True)
        api_key = self.credentials.get(provider, "")
        if auth_required and not api_key:
            result["status"] = "skipped_no_credentials"
            return result

        # If credentials are a dict, ensure all required non-auth query params are present.
        # E2E Networks is kept in the run even if project_id is not supplied; the API key is enough to attempt.
        if auth_required and isinstance(api_key, dict):
            auth_qp = contract.get("auth_query_param")
            required_qps: set = set()
            for endpoint in contract.get("endpoints", []):
                if not endpoint.get("enabled", True):
                    continue
                for qp in endpoint.get("query_params", []):
                    if isinstance(qp, dict):
                        if qp.get("default") is not None:
                            continue
                        name = qp["name"]
                    else:
                        name = qp
                    if name == auth_qp:
                        continue
                    required_qps.add(name)
            missing = [p for p in required_qps if p not in api_key]
            if missing and contract.get("provider") != "e2enetworks":
                result["status"] = "skipped_no_credentials"
                return result

        # For E2E Networks, discover the default project if the user didn't supply one.
        if (
            provider == "e2enetworks"
            and isinstance(api_key, dict)
            and "project_id" not in api_key
        ):
            project_id = self._fetch_e2e_project_id(api_key)
            if project_id:
                api_key["project_id"] = project_id

        for endpoint in contract.get("endpoints", []):
            if not endpoint.get("enabled", True):
                continue
            endpoint_result = self._check_endpoint(contract, endpoint, api_key)
            result["endpoints"].append(endpoint_result)
            if endpoint_result.get("drift"):
                result["drift_detected"] = True
                result["status"] = "drift"

        return result

    def _fetch_e2e_project_id(self, creds: Dict[str, Any]) -> Optional[str]:
        """Resolve the last-used E2E Networks project from the CRN endpoint.

        E2E MyAccount requires a project_id query parameter for most compute
        endpoints. The /iam/multi-crn/ endpoint returns the account's
        `last_used_project` and only needs the apikey/Authorization credentials.
        """
        key_value = str(creds.get("api_key", ""))
        if not key_value:
            return None

        location = str(creds.get("location", "Delhi"))
        base_url = "https://api.e2enetworks.com/myaccount/api/v1"
        url = f"{base_url}/iam/multi-crn/?apikey={key_value}&location={location}"
        headers: Dict[str, str] = {}
        if "bearer_token" in creds:
            headers["Authorization"] = f"Bearer {creds['bearer_token']}"

        try:
            response = requests.get(url, headers=headers, timeout=self.timeout)
            if not response.ok:
                return None
            body = response.json()
            project_id = body.get("data", {}).get("last_used_project")
            if project_id:
                return str(project_id)
        except (requests.RequestException, json.JSONDecodeError, ValueError, AttributeError, TypeError):
            pass
        return None

    def _check_endpoint(
        self,
        contract: Dict[str, Any],
        endpoint: Dict[str, Any],
        api_key: Union[str, Dict[str, Any]],
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

            expected_status = endpoint.get("expected_status")
            if expected_status is not None and response.status_code != expected_status:
                result["drift"] = True
                result["drift_reasons"].append(
                    f"expected HTTP {expected_status}, got {response.status_code}"
                )
                # Include a short snapshot of the response body to diagnose auth failures.
                try:
                    err_text = response.text.strip()[:500]
                    if err_text:
                        result["error_body"] = err_text
                except (ValueError, TypeError, AttributeError):
                    pass
                result["drift_summary"] = "; ".join(result["drift_reasons"])
                return result

            if response.ok:
                try:
                    body = response.json()
                except json.JSONDecodeError:
                    try:
                        root = ET.fromstring(response.text)
                        body = {root.tag: _xml_to_dict(root)}
                    except ET.ParseError as exc:
                        result["drift"] = True
                        result["error"] = f"invalid JSON/XML: {exc}"
                        result["drift_reasons"].append("non-JSON/XML response")
                        result["drift_summary"] = "non-JSON/XML response"
                        return result
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

    def _gcp_access_token(self, creds: Dict[str, Any]) -> Optional[str]:
        """Get a short-lived GCP access token from a service account JSON."""
        if service_account is None or GoogleRequest is None:
            return None
        gcp_creds = creds.get("gcp_credentials")
        if not isinstance(gcp_creds, dict):
            return None
        try:
            credentials = service_account.Credentials.from_service_account_info(
                gcp_creds,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            credentials.refresh(GoogleRequest())
            return credentials.token
        except (ValueError, AttributeError, TypeError):
            return None

    def _aws_presigned_url(
        self, method: str, url: str, creds: Dict[str, Any]
    ) -> Optional[str]:
        """Generate a boto3 SigV4 presigned URL for AWS query APIs."""
        if boto3 is None:
            return None
        access_key = str(creds.get("aws_access_key_id", ""))
        secret_key = str(creds.get("aws_secret_access_key", ""))
        region = str(creds.get("aws_region", "us-east-1"))
        if not access_key or not secret_key:
            return None

        try:
            parsed = urllib.parse.urlparse(url)
            query = urllib.parse.parse_qsl(parsed.query)

            action = None
            params: Dict[str, Any] = {}
            for key, value in query:
                if key == "Action":
                    action = value
                elif key == "Version":
                    continue
                else:
                    # boto3 expects typed values.
                    if value.isdigit():
                        params[key] = int(value)
                    elif value.lower() == "true":
                        params[key] = True
                    elif value.lower() == "false":
                        params[key] = False
                    else:
                        params[key] = value

            if not action:
                return None

            operation = "".join(
                ["_" + c.lower() if c.isupper() else c for c in action]
            ).lstrip("_")

            client = boto3.client(
                "ec2",
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
            return client.generate_presigned_url(
                operation, Params=params, ExpiresIn=60
            )
        except (ValueError, AttributeError, TypeError):
            return None

    def _oci_sign(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        payload: bytes,
        creds: Dict[str, Any],
    ) -> Optional[Dict[str, str]]:
        """Sign an Oracle OCI request with RSA-SHA256."""
        tenancy = str(creds.get("oci_tenancy", ""))
        user = str(creds.get("oci_user", ""))
        fingerprint = str(creds.get("oci_fingerprint", ""))
        private_key = str(creds.get("oci_private_key", ""))
        if not all([tenancy, user, fingerprint, private_key]):
            return None

        try:
            key = serialization.load_pem_private_key(
                private_key.encode("utf-8"), password=None
            )
        except (ValueError, TypeError):
            return None

        now = datetime.now(timezone.utc)
        date = now.strftime("%a, %d %b %Y %H:%M:%S GMT")
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc
        request_target = f"{method.lower()} {parsed.path}"
        if parsed.query:
            request_target += f"?{parsed.query}"

        signing_lines = [
            f"date: {date}",
            f"(request-target): {request_target}",
            f"host: {host}",
        ]
        signing_string = "\n".join(signing_lines)
        signature = key.sign(signing_string.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())
        signature_b64 = base64.b64encode(signature).decode("utf-8")

        key_id = f"{tenancy}/{user}/{fingerprint}"
        auth_header = (
            'Signature version="1",'
            f'keyId="{key_id}",'
            'algorithm="rsa-sha256",'
            'headers="date (request-target) host",'
            f'signature="{signature_b64}"'
        )

        signed = dict(headers)
        signed["Date"] = date
        signed["Authorization"] = auth_header
        return signed

    def _build_request(
        self,
        contract: Dict[str, Any],
        endpoint: Dict[str, Any],
        api_key: Union[str, Dict[str, Any]],
    ) -> tuple:
        """Construct the URL, headers, and payload for a contract endpoint."""
        base_url = contract["base_url"].rstrip("/")
        path = str(endpoint.get("path", "")).lstrip("/")
        url = f"{base_url}/{path}" if path else base_url

        auth_in = contract.get("auth_in", "header")
        auth_type = contract.get("auth_type", "")
        auth_header = contract.get("auth_header", "Authorization")

        headers: Dict[str, str] = {}
        payload: Dict[str, Any] = {}

        # Normalize credentials: may be a raw key string or a dict with extras.
        if isinstance(api_key, dict):
            creds: Dict[str, Any] = api_key
            key_value: str = str(creds.get("api_key", ""))
            bearer_token: str = str(creds.get("bearer_token", key_value))
        else:
            creds = {}
            key_value = str(api_key)
            bearer_token = key_value

        # Substitute credential placeholders in the URL/base_url.
        url = url.replace("{project_id}", str(creds.get("project_id", "")))
        url = url.replace("{aws_region}", str(creds.get("aws_region", "us-east-1")))
        url = url.replace("{oci_region}", str(creds.get("oci_region", "us-ashburn-1")))
        url = url.replace("{zone}", str(creds.get("zone", "us-central1-a")))

        def _add_query_param(name: str, value: Any) -> None:
            nonlocal url
            if value is None or value == "":
                return
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}{name}={value}"

        method = endpoint.get("method", "GET").upper()

        # Build the request payload first; it may be needed for signing POSTs.
        if endpoint.get("smoke_test_query"):
            payload["query"] = endpoint["smoke_test_query"]
        if endpoint.get("smoke_test_variables"):
            payload["variables"] = endpoint["smoke_test_variables"]
        for field in endpoint.get("required_fields", []):
            if field not in ("query", "variables") and field not in payload:
                payload[field] = endpoint.get(field)

        # Append query params (including the auth query param) before signing.
        for qp in endpoint.get("query_params", []):
            if isinstance(qp, dict):
                name = qp["name"]
                source = qp.get("from_credential", name)
                value = creds.get(source, qp.get("default"))
            else:
                name = qp
                value = creds.get(name)
            if value is None and name == contract.get("auth_query_param", "api_key"):
                value = key_value
            _add_query_param(name, value)

        if not contract.get("auth_required", True):
            return url, headers, payload

        if auth_in == "query":
            # The auth query param was already added by the loop above.
            return url, headers, payload

        if method == "POST":
            headers.setdefault("Content-Type", "application/json")

        if auth_type:
            auth_lower = auth_type.lower()
            if auth_lower == "basic":
                token = base64.b64encode(f"{bearer_token}:".encode()).decode()
                headers[auth_header] = f"Basic {token}"
            elif auth_lower == "bearer" and creds.get("gcp_credentials"):
                gcp_token = self._gcp_access_token(creds)
                if gcp_token:
                    headers[auth_header] = f"Bearer {gcp_token}"
                else:
                    headers[auth_header] = f"Bearer {bearer_token}"
            elif auth_lower == "bearer":
                headers[auth_header] = f"Bearer {bearer_token}"
            elif auth_lower == "sigv4":
                presigned = self._aws_presigned_url(method, url, creds)
                if presigned:
                    url = presigned
            elif auth_lower == "rsa":
                body = json.dumps(payload).encode("utf-8") if payload else b""
                signed = self._oci_sign(method, url, headers, body, creds)
                if signed:
                    headers.update(signed)
            else:
                headers[auth_header] = f"{auth_type} {bearer_token}"
        else:
            headers[auth_header] = bearer_token

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
