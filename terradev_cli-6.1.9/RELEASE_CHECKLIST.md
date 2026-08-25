# Terradev Release Checklist

This checklist must be followed for every PyPI release to prevent `ModuleNotFoundError` and other packaging issues.

## Pre-Release Checks

### 1. Code Review
- [ ] All new code has been reviewed
- [ ] No relative imports in `terradev_cli/cli.py` (must use absolute imports like `from terradev_cli.core.telemetry import ...`)
- [ ] All imports in `cli.py` use the `terradev_cli.` prefix for internal modules
- [ ] No hardcoded version strings in code (use `pyproject.toml` as single source of truth)

### 2. Package Audit
- [ ] Run full repo audit to find all `__init__.py` files:
  ```bash
  find terradev_cli -name "__init__.py" -type f
  ```
- [ ] Verify all packages are under `terradev_cli/` directory
- [ ] Classify packages as MUST SHIP or OPTIONAL:
  - **MUST SHIP**: Any package imported at runtime by `terradev_cli` or its dependencies
  - **OPTIONAL**: Packages only used in development, testing, or documentation

### 3. pyproject.toml Verification
- [ ] Check `[tool.setuptools.packages.find]` section:
  ```toml
  [tool.setuptools.packages.find]
  include = ["terradev_cli", "terradev_cli.*"]
  ```
- [ ] Verify version is incremented (e.g., 5.5.1 → 5.5.2)
- [ ] Update CHANGELOG.md with release notes
- [ ] Update README.md if any user-facing changes

### 4. Import Style Check
- [ ] Search for relative imports in `cli.py`:
  ```bash
  grep -n "^from [a-z_]\+\." terradev_cli/cli.py
  ```
- [ ] Fix any found relative imports to absolute imports:
  - `from core.telemetry` → `from terradev_cli.core.telemetry`
  - `from providers.provider_factory` → `from terradev_cli.providers.provider_factory`
  - `from ml_services.vllm_service` → `from terradev_cli.ml_services.vllm_service`
  - etc.

## Build & Verification

### 5. Clean Build
- [ ] Remove old build artifacts:
  ```bash
  rm -rf dist/ build/
  ```
- [ ] Build wheel and source distribution:
  ```bash
  python3 -m build
  ```
- [ ] Verify build succeeded with no warnings

### 6. Wheel Inspection
- [ ] Inspect wheel contents:
  ```bash
  python3 -c "import zipfile; z=zipfile.ZipFile('dist/terradev_cli-X.X.X-py3-none-any.whl'); print('\n'.join(sorted(z.namelist())))"
  ```
- [ ] Verify all MUST SHIP packages are present:
  - `terradev_cli/core/`
  - `terradev_cli/providers/`
  - `terradev_cli/ml_services/`
  - `terradev_cli/integrations/`
  - `terradev_cli/k8s/`
  - `terradev_cli/optimization/`
  - `terradev_cli/utils/`
  - `terradev_cli/terraform/`
  - `terradev_cli/kubernetes/`
  - `terradev_cli/mcp/`
- [ ] Verify `terradev_cli/cli.py` is present

### 7. Local Installation Test
- [ ] Install wheel locally:
  ```bash
  pip install --force-reinstall dist/terradev_cli-X.X.X-py3-none-any.whl
  ```
- [ ] Verify installation:
  ```bash
  pip show terradev-cli
  ```

## Smoke Tests

### 8. Import Tests
- [ ] Test critical imports:
  ```bash
  python3 -c "from terradev_cli.providers.provider_factory import ProviderFactory; print('✓ ProviderFactory OK')"
  python3 -c "from terradev_cli.core.preflight_validator import PreflightValidator; print('✓ PreflightValidator OK')"
  python3 -c "from terradev_cli.core.telemetry import get_mandatory_telemetry; print('✓ telemetry OK')"
  ```
- [ ] Test all major module imports (core, providers, ml_services, integrations, k8s, optimization, utils)

### 9. CLI Tests
- [ ] Test CLI help:
  ```bash
  terradev --help
  ```
- [ ] Test configure command:
  ```bash
  terradev configure --provider runpod --help
  ```
- [ ] Test quote command:
  ```bash
  terradev quote -g A100
  ```
- [ ] Test provision dry-run:
  ```bash
  terradev provision -g A100 --dry-run --providers runpod
  ```
- [ ] Test preflight help:
  ```bash
  terradev preflight --help
  ```
- [ ] Test vLLM help:
  ```bash
  terradev vllm --help
  ```
- [ ] Test lora help:
  ```bash
  terradev lora --help
  ```

### 9.5. Provision Command Tests (v5.5.3+)
- [ ] terradev provision -g A100 --providers runpod --dry-run
    Shows instance table and prompts for selection
    No "Unsupported cloud type" error
- [ ] terradev provision -g A100 --providers runpod --select 1 --dry-run
    Selects first option without prompt
- [ ] terradev provision -g A100 --providers runpod --auto --dry-run
    Selects cheapest without prompt
- [ ] --spot-strategy cheapest accepted without error
- [ ] --spot-strategy safe accepted without error
- [ ] --select SXM4-40GB selects correct instance variant

## Release

### 10. Version Bump
- [ ] Update version in `pyproject.toml`
- [ ] Commit version bump with message: `Bump version to X.X.X`
- [ ] Tag release: `git tag -a vX.X.X -m "Release X.X.X"`

### 11. PyPI Upload
- [ ] Upload to PyPI:
  ```bash
  python3 -m twine upload dist/terradev_cli-X.X.X-py3-none-any.whl dist/terradev_cli-X.X.X.tar.gz
  ```
- [ ] Verify upload at: https://pypi.org/project/terradev-cli/X.X.X/

## Post-Release Verification

### 12. Fresh Install Test
- [ ] Uninstall local version:
  ```bash
  pip uninstall -y terradev-cli
  ```
- [ ] Install from PyPI:
  ```bash
  pip install terradev-cli==X.X.X
  ```
- [ ] Verify version:
  ```bash
  pip show terradev-cli
  ```
- [ ] Repeat all smoke tests (steps 8-9)

### 13. Documentation
- [ ] Update GitHub Releases page with release notes
- [ ] Update any relevant documentation
- [ ] Announce release to users (if applicable)

## Common Pitfalls

### Relative Imports
**Problem**: `from core.telemetry import ...` fails after pip install because the package structure changes.
**Solution**: Always use absolute imports: `from terradev_cli.core.telemetry import ...`

### Missing Packages
**Problem**: Subpackages not included in wheel due to incorrect `pyproject.toml` configuration.
**Solution**: Ensure `[tool.setuptools.packages.find]` includes `["terradev_cli", "terradev_cli.*"]`

### Version Mismatch
**Problem**: CLI shows wrong version after install.
**Solution**: Version must be in `pyproject.toml` only. Do not hardcode in `cli.py`.

### README.md Location
**Problem**: PyPI shows old README because setup.py looks in wrong location.
**Solution**: Ensure `readme = "README.md"` in `pyproject.toml` points to the correct file.

## Quick Reference Commands

```bash
# Find all __init__.py files
find terradev_cli -name "__init__.py" -type f

# Search for relative imports (bad)
grep -n "^from [a-z_]\+\." terradev_cli/cli.py

# Build wheel
rm -rf dist/ build/ && python3 -m build

# Inspect wheel
python3 -c "import zipfile; z=zipfile.ZipFile('dist/terradev_cli-X.X.X-py3-none-any.whl'); print('\n'.join(sorted(z.namelist())))"

# Install locally
pip install --force-reinstall dist/terradev_cli-X.X.X-py3-none-any.whl

# Upload to PyPI
python3 -m twine upload dist/terradev_cli-X.X.X-py3-none-any.whl dist/terradev_cli-X.X.X.tar.gz

# Test from PyPI
pip install terradev-cli==X.X.X
```

## Release History Template

### X.X.X (YYYY-MM-DD)
- **Fixed**: ModuleNotFoundError on PyPI installs due to relative imports
- **Changed**: All imports in cli.py converted to absolute imports with terradev_cli prefix
- **Verified**: All packages included in wheel, smoke tests pass
