# Credential Strategy for CI/CD Testing

This document explains which tests require real cloud provider credentials and which use mocks.

---

## Overview

Terradev tests are divided into three categories:

1. **Unit Tests** — No external credentials required. All dependencies are mocked.
2. **Mock Integration Tests** — Provider API calls are mocked. No credentials required.
3. **Live Integration Tests** — Real API calls to cloud providers. Credentials required.

---

## Unit Tests

**Location:** `tests/` (excluding `test_integration.py`, `test_mcp_handlers.py`)

**Credentials Required:** None

**How They Work:**
- All external dependencies (provider APIs, ML services) are mocked using `unittest.mock`
- Tests verify logic, error handling, and data transformation without calling real APIs
- These run on every commit in CI (GitHub Actions, GitLab CI)

**Examples:**
- `test_cli_smoke.py` — CLI command parsing and validation
- `test_flashoptim.py` — FlashOptim auto-injection logic
- `test_vllm_features.py` — vLLM configuration generation
- `test_ai_integrations.py` — Phoenix, Guardrails, Qdrant service initialization (mocked)
- `test_new_providers.py` — Provider auth header formats (mocked)

---

## Mock Integration Tests

**Location:** `tests/test_integration.py`

**Credentials Required:** None

**How They Work:**
- Run with `--mock` flag
- Provider API calls return canned responses
- Tests verify the integration layer works correctly without hitting real APIs
- These run on every commit in CI

**Command:**
```bash
python tests/test_integration.py --mock --suite providers
```

**CI Configuration:**
- GitHub Actions: `integration-mock` job
- GitLab CI: `integration-mock` job

---

## Live Integration Tests

**Location:** `tests/test_integration.py`

**Credentials Required:** Yes (optional, manual trigger only)

**How They Work:**
- Run without `--mock` flag
- Makes real API calls to configured providers
- Tests verify actual provider API behavior, quotas, and edge cases
- These do NOT run automatically on commits to avoid costs and rate limits

**Command:**
```bash
python tests/test_integration.py --suite providers
```

**CI Configuration:**
- GitHub Actions: `integration-live` job (manual trigger via `workflow_dispatch`)
- GitLab CI: `integration-live` job (manual trigger)

---

## Required Secrets for Live Integration Tests

### GitHub Actions

Configure these secrets in your GitHub repository settings:

| Secret | Description | Required For |
|---|---|---|
| `AWS_TEST_KEY` | AWS Access Key ID | AWS provider tests |
| `AWS_TEST_SECRET` | AWS Secret Access Key | AWS provider tests |
| `RUNPOD_TEST_KEY` | RunPod API Key | RunPod provider tests |
| `VAST_TEST_KEY` | Vast.ai API Key | Vast.ai provider tests |

**Setup:**
1. Go to repository → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add each secret with the corresponding value

### GitLab CI

Configure these variables in your GitLab project:

| Variable | Description | Required For |
|---|---|---|---|
| `AWS_ACCESS_KEY_ID` | AWS Access Key ID | AWS provider tests |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret Access Key | AWS provider tests |
| `RUNPOD_API_KEY` | RunPod API Key | RunPod provider tests |
| `VAST_API_KEY` | Vast.ai API Key | Vast.ai provider tests |

**Setup:**
1. Go to project → Settings → CI/CD → Variables
2. Add each variable with the corresponding value
3. Mask sensitive variables

---

## Local Testing with Credentials

To run live integration tests locally:

```bash
# Set environment variables
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export RUNPOD_API_KEY="your-key"
export VAST_API_KEY="your-key"

# Run live integration tests
python tests/test_integration.py --suite providers
```

**Warning:** This will make real API calls and may incur costs. Ensure you have appropriate quotas and budget alerts configured.

---

## Credential Rotation

Test credentials should be rotated regularly:

1. Use dedicated test accounts with limited quotas
2. Set up budget alerts on test accounts
3. Rotate API keys quarterly
4. Revoke old keys immediately after rotation

---

## Provider-Specific Notes

### AWS
- Test account should have EC2 quota for at least 2 instances
- Use a dedicated IAM user with `ec2:RunInstances`, `ec2:DescribeInstances`, `ec2:TerminateInstances` permissions only
- Do not use production AWS credentials

### RunPod
- Test account should have at least $10 credit balance
- Use a dedicated API key, not your main account key
- RunPod may reject API calls from unfunded accounts

### Vast.ai
- Test account should have a verified payment method
- Use a dedicated API key
- Vast.ai uses a bidding model — tests may fail if bids are outbid

---

## Mock Strategy

When adding new provider tests, follow this pattern:

```python
import pytest
from unittest.mock import AsyncMock, patch

class TestNewProvider:
    @pytest.mark.asyncio
    async def test_with_mock(self):
        """Test with mocked API call"""
        provider = NewProvider(credentials={"api_key": "test"})

        # Mock the API call
        with patch.object(provider, "_make_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = {"data": [{"price": 1.50}]}
            result = await provider.get_instance_quotes("A100")
            assert result == [{"price": 1.50}]

    @pytest.mark.asyncio
    async def test_live_integration(self):
        """Test with real API call (requires credentials)"""
        # This test should be in a separate file or marked with a pytest marker
        # that is skipped in CI unless credentials are present
        provider = NewProvider(credentials={"api_key": os.environ.get("TEST_API_KEY")})
        if not os.environ.get("TEST_API_KEY"):
            pytest.skip("TEST_API_KEY not set")
        result = await provider.get_instance_quotes("A100")
        assert len(result) > 0
```

---

## CI/CD Job Dependencies

**GitHub Actions:**
- `lint` → `test-linux` → `integration-mock` → `build` → `security` → `publish`
- `test-windows` and `test-macos` run in parallel with `test-linux`
- `integration-live` runs only on manual trigger
- `docker-build` depends on `build`
- `benchmark` runs only on push to main

**GitLab CI:**
- `lint` → `test` → `integration-mock` → `build` → `security` → `deploy`
- `integration-live` runs only on manual trigger
- `deploy` runs only on tags and requires manual approval

---

## Troubleshooting

### Live Integration Tests Failing

**Symptom:** Tests fail with authentication errors or rate limit errors.

**Fixes:**
1. Verify credentials are set correctly in CI secrets
2. Check that test account has sufficient quota
3. Verify API key hasn't expired or been revoked
4. Check for rate limiting — some providers have strict limits

### Mock Tests Failing

**Symptom:** Mock tests fail with unexpected mock behavior.

**Fixes:**
1. Verify mock return values match actual API response structure
2. Check that mock is applied to the correct method
3. Ensure async context managers are properly mocked
4. Use `autospec=True` for more accurate mocking

### Unit Tests Failing in CI but Passing Locally

**Symptom:** Tests pass locally but fail in CI.

**Fixes:**
1. Check Python version mismatch (CI uses specific versions)
2. Verify all dependencies are installed in CI
3. Check for environment-specific behavior (paths, env vars)
4. Ensure tests don't depend on local files or state

---

## Best Practices

1. **Never commit credentials** to the repository
2. **Use dedicated test accounts** with limited permissions
3. **Set up budget alerts** on all test accounts
4. **Rotate credentials regularly** (quarterly minimum)
5. **Mock external dependencies** in unit tests
6. **Use live integration tests sparingly** (manual trigger only)
7. **Document new credential requirements** in this file
8. **Review test account usage monthly** to catch unexpected costs

---

## Canary Provisioning Tests

**Location:** `tests/test_canary_provisioning.py`

**Credentials Required:** Yes (manual trigger only)

**How They Work:**
- These are end-to-end tests that provision the cheapest available instance for each configured provider.
- Tests poll for RUNNING/ACTIVE status, verify SSH or endpoint connectivity, then immediately terminate the instance.
- Results are recorded in `~/.terradev/canary-results.jsonl`.

**Why They Are Skipped by Default:**
- By default, the entire canary suite is skipped with the message:  
  `Canary tests hit live provider APIs. Set TERRADEV_CANARY_TEST=1 to enable.`
- This prevents CI from accidentally provisioning real (and potentially costly) cloud instances.

**Command to Enable Locally:**

```bash
TERRADEV_CANARY_TEST=1 pytest tests/test_canary_provisioning.py
```

Optional overrides:

```bash
# Default GPU type (default: RTX4090)
export TERRADEV_CANARY_GPU=RTX4090

# Maximum hourly price cap (default: $0.50)
export TERRADEV_CANARY_MAX_PRICE=0.50

# Provisioning timeout in seconds (default: 300)
export TERRADEV_CANARY_TIMEOUT=300

# Comma-separated region override
export TERRADEV_CANARY_REGIONS=us-east-1,us-west-1
```

**CI Configuration:**
- Do not run canary tests on every commit.
- Run them only via a manual `workflow_dispatch` trigger with the required provider secrets.
