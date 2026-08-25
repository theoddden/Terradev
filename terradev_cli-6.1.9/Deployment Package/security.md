# Terradev Security

---

## Credential Model

Terradev uses a **BYOAPI** (Bring Your Own API Key) model. You bring credentials from your own cloud provider accounts. Terradev never issues, manages, or rotates credentials on your behalf.

### Where Credentials Are Stored

All credentials are stored locally on the machine running Terradev:

```
~/.terradev/credentials.json
```

**What this means:**
- Credentials are never transmitted to Terradev servers (there are no Terradev servers in the request path)
- Credentials are never included in logs, telemetry, or error reports
- All API calls go directly from your machine to the provider API endpoint
- Terradev has zero visibility into your cloud accounts

### Credential File

```json
{
  "runpod": { "api_key": "..." },
  "vastai": { "api_key": "..." },
  "aws": {
    "access_key_id": "...",
    "secret_access_key": "...",
    "default_region": "us-east-1"
  }
}
```

The file is created at first `terradev configure` run. It uses `0600` permissions (owner read/write only) on Unix systems.

**Recommendation:** Add `~/.terradev/credentials.json` to your backup exclusions. Do not commit it. Do not copy it into containers.

### SSH Keypairs

Per-provision SSH keypairs are auto-generated at provision time, used for the duration of that instance's lifecycle, and discarded on termination. Keys are not reused across provisions.

---

## Network Security

### Request Path

```
Your machine
    │
    └──▶ Provider API (RunPod/AWS/GCP/etc.)
             Direct HTTPS connection
             No Terradev proxy
             No intermediate servers
```

No traffic is routed through Terradev infrastructure. Your GPU traffic, training data, and model weights go directly to and from provider endpoints.

### TLS

All provider API calls use HTTPS/TLS. Provider SDK clients enforce certificate validation. Do not use `--insecure` flags or custom CA bundles unless you control the endpoint (e.g., self-hosted Qdrant, Phoenix, or Guardrails on a private network).

### Inbound Ports

Provisioned instances open ports as needed for your workload:

| Port | Service | Exposure |
|---|---|---|
| 22 | SSH | Needed for `terradev execute`, `terradev ssh` |
| 8000 | vLLM / inference | Application-controlled |
| 6006 | Arize Phoenix | Internal only recommended |
| 6333 | Qdrant REST | Internal only recommended |
| 8080 | NeMo Guardrails | Internal only recommended |
| 3000 | MCP server | Localhost only |

**Recommendation:** Do not expose Phoenix, Qdrant, or Guardrails ports publicly. Deploy them behind a private network or a reverse proxy with authentication.

---

## Bare Metal and Compliance Workloads

For workloads requiring hardware isolation — HIPAA, FedRAMP, financial services, defense:

Terradev supports **Latitude.sh bare metal** provisioning. Bare metal gives you a dedicated physical server (no hypervisor layer), IPMI out-of-band management, and hardware attestation capability.

```bash
# Provision dedicated bare metal
terradev provision --provider latitude --gpu H100 --instance-type bare-metal
```

IPMI endpoint is returned in instance status:

```bash
terradev status --live --provider latitude
# ipmi_access: true
# ipmi_endpoint: 10.x.x.x
# isolation: bare_metal
```

Bare metal satisfies compliance frameworks that:
- Require dedicated hardware (no co-tenancy)
- Require physical access controls and hardware attestation
- Reject hypervisor layers in their audit scope

---

## Secrets Management

### Recommended: Environment Variables

Instead of storing keys in `credentials.json`, you can set provider keys as environment variables. Terradev reads environment variables for all supported providers:

```bash
export RUNPOD_API_KEY="..."
export VAST_API_KEY="..."
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export LAMBDA_API_KEY="..."
```

Combine with a secrets manager:

```bash
# AWS Secrets Manager
export RUNPOD_API_KEY=$(aws secretsmanager get-secret-value \
  --secret-id prod/terradev/runpod \
  --query SecretString \
  --output text)
```

```bash
# HashiCorp Vault
export RUNPOD_API_KEY=$(vault kv get -field=api_key secret/terradev/runpod)
```

### CI/CD

In GitHub Actions / GitLab CI, store provider keys as repository secrets and inject as environment variables. Never write keys to disk in CI.

```yaml
# GitHub Actions example
env:
  RUNPOD_API_KEY: ${{ secrets.RUNPOD_API_KEY }}
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
```

---

## MCP Server Security

The MCP server (`terradev mcp serve`) runs as a local subprocess communicating over stdio. It does not bind to a network port by default.

**Scope of access:** The MCP server has full access to all Terradev capabilities — provisioning, termination, data staging, training launch. Treat it with the same trust level as direct CLI access.

**Agent guardrails:** The Rust DAG orchestrator enforces idempotency and correct sequencing, but does not apply policy controls on what an agent can do. If you're running agents autonomously:

- Use read-only credentials where possible for initial exploration
- Set `--max-price` ceilings on provision operations
- Review `terradev status --live` after agent sessions to audit what was created
- Set cloud-level budget alerts on your provider accounts independently of Terradev

---

## Data Governance

### Credentials Audit

```bash
# Review what providers are configured
terradev configure --list

# Rotate a credential
terradev configure --provider runpod
# Enter new API key at prompt
```

### Lineage Tracking

For regulated workloads, use Terradev's lineage commands to maintain dataset and model provenance records:

```bash
# Register dataset with URI
terradev lineage register dataset my-dataset s3://my-bucket/dataset

# Track model training execution
terradev lineage add-input my-execution dataset my-dataset
terradev lineage add-output my-execution model my-model
terradev lineage complete my-execution --status completed

# Export lineage records
terradev lineage export --format json --model my-model --env prod
```

### Environment Promotion Controls

Use the environments system to enforce approval gates before production deployments:

```bash
# Promote requires a promotion ID
terradev environments promote my-model --from staging --to prod --user engineer

# Approve with audit trail
terradev environments approve <promotion-id> --user senior-engineer

# Review full history
terradev environments history --artifact my-model
```

---

## Hardening Checklist

### Installation

- [ ] Install via `pip install terradev-cli` from PyPI (do not clone and run `python cli.py` directly in production)
- [ ] Pin to a specific version: `pip install terradev-cli==5.1.5`
- [ ] Run as a non-root user
- [ ] Do not run in the same environment as production model serving

### Credentials

- [ ] `~/.terradev/credentials.json` has `0600` permissions: `chmod 600 ~/.terradev/credentials.json`
- [ ] `~/.terradev/` directory has `0700` permissions: `chmod 700 ~/.terradev/`
- [ ] Not committed to version control (add `~/.terradev/` to `.gitignore` globally)
- [ ] Rotating credentials after team member offboarding
- [ ] Using environment variables instead of `credentials.json` in CI/CD

### Networking

- [ ] Phoenix, Qdrant, and Guardrails not exposed on public ports
- [ ] vLLM inference endpoints behind authentication (API key or VPC)
- [ ] SSH access restricted to known IP ranges (configure at provider level)
- [ ] Kubernetes clusters using private node groups where possible

### Kubernetes

- [ ] Karpenter RBAC scoped to minimum required permissions
- [ ] GPU node security groups restricted to cluster VPC only
- [ ] Secrets stored in Kubernetes Secrets or an external secrets operator, not in ConfigMaps
- [ ] NVIDIA device plugin namespace restricted (`gpu-operator` namespace, not `default`)

### MCP / Agent Usage

- [ ] MCP server not exposed over the network (stdio only by default — do not add `--port` in untrusted environments)
- [ ] Agent sessions reviewed post-run with `terradev status`
- [ ] Cloud-level budget alerts set independently at provider level
- [ ] Destructive operations (`terminate`, `delete-nodepool`) require explicit confirmation

### Compliance-Sensitive Workloads

- [ ] Using Latitude.sh bare metal for HIPAA/FedRAMP-adjacent workloads
- [ ] IPMI endpoint access logged and restricted to SOC team
- [ ] Lineage records exported and archived for audit trail
- [ ] Dataset staging paths audited for PII/PHI before upload
- [ ] NeMo Guardrails deployed with PII filtering enabled for user-facing endpoints

---

## Known Security Boundaries

These are things Terradev **does not protect against** by design:

1. **Compromised provider credentials** — if your RunPod API key is leaked, Terradev cannot revoke it. Revoke at the provider level directly.
2. **Malicious training scripts** — Terradev executes whatever script you point it at with the permissions of the SSH user on the instance. Review training scripts before launch.
3. **Data-in-transit on training clusters** — NCCL collective operations between nodes are not encrypted by default. For sensitive training data, enable NCCL encryption or use a VPC-isolated cluster.
4. **Container image trust** — `terradev run --image ...` pulls and executes arbitrary images. Use trusted registries and pin image digests.

---

## Reporting Security Issues

GitHub Issues: [github.com/theoddden/Terradev/issues](https://github.com/theoddden/Terradev/issues)

For sensitive disclosures, use GitHub's private vulnerability reporting on the same repository.
