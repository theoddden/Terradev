"""Tests for terradev_cli.core.gitops_manager.

GitOps setup generates Kubernetes manifests and repository structure.
These tests mock `subprocess` and `Path.home` to avoid side effects.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from terradev_cli.core.gitops_manager import (
    GitOpsConfig,
    GitOpsManager,
    GitOpsTool,
    GitProvider,
    GitRepoStructure,
    get_gitops_manager,
)


@pytest.fixture
def mock_home(tmp_path, monkeypatch):
    """Redirect Path.home to a temp directory."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def argocd_config():
    return GitOpsConfig(
        provider=GitProvider.GITHUB,
        repository="terradev-infra",
        tool=GitOpsTool.ARGOCD,
        cluster_name="prod-cluster",
    )


def test_gitops_config_defaults():
    """GitOpsConfig has sensible defaults."""
    config = GitOpsConfig(
        provider=GitProvider.GITHUB,
        repository="repo",
        tool=GitOpsTool.FLUX,
        cluster_name="c",
    )
    assert config.environments == ["dev", "staging", "prod"]
    assert config.namespace == "gitops-system"
    assert config.auto_sync is True


def test_repo_structure_creation(mock_home):
    """GitRepoStructure creates the expected directory tree."""
    base = mock_home / "repo"
    structure = GitRepoStructure()
    structure.create_structure(base)

    for d in ["clusters", "apps", "infra", "policies", "monitoring"]:
        assert (base / d).is_dir()
    for env in ["dev", "staging", "prod"]:
        assert (base / "clusters" / env).is_dir()


def test_gitops_manager_rejects_invalid_repo_name(mock_home):
    """The manager rejects repository names that could escape the work dir."""
    with pytest.raises(ValueError, match="invalid"):
        GitOpsManager(
            GitOpsConfig(
                provider=GitProvider.GITHUB,
                repository="../../.ssh",
                tool=GitOpsTool.ARGOCD,
                cluster_name="c",
            )
        )


@pytest.mark.asyncio
async def test_init_repository_argocd(mock_home, argocd_config, monkeypatch):
    """init_repository creates the work dir, structure, and YAML files."""
    calls = []

    def fake_subprocess(cmd, **kwargs):
        calls.append(cmd)
        # Pretend git init created a .git directory
        if cmd[:2] == ["git", "init"]:
            work_dir = kwargs.get("cwd") or mock_home / ".terradev" / "gitops" / argocd_config.repository
            (Path(work_dir) / ".git").mkdir(parents=True, exist_ok=True)
        return None

    monkeypatch.setattr(
        "terradev_cli.core.gitops_manager.subprocess.run", fake_subprocess
    )

    manager = GitOpsManager(argocd_config)
    assert await manager.init_repository() is True

    work_dir = mock_home / ".terradev" / "gitops" / argocd_config.repository
    assert work_dir.is_dir()
    for d in ["clusters", "apps", "infra", "policies", "monitoring"]:
        assert (work_dir / d).is_dir()

    # ArgoCD-specific files
    assert (work_dir / "infra" / "argocd-namespace.yaml").exists()
    assert (work_dir / "infra" / "argocd-appset.yaml").exists()
    assert (work_dir / "clusters" / "dev" / "cluster-config.yaml").exists()


def test_get_gitops_manager_singleton(argocd_config, mock_home):
    """get_gitops_manager returns a singleton for the same config."""
    m1 = get_gitops_manager(argocd_config)
    m2 = get_gitops_manager(argocd_config)
    assert m1 is m2


@pytest.mark.asyncio
async def test_validate_configuration(mock_home, argocd_config, monkeypatch):
    """validate_configuration checks generated YAML files."""
    monkeypatch.setattr(
        "terradev_cli.core.gitops_manager.subprocess.run", lambda *a, **k: None
    )

    manager = GitOpsManager(argocd_config)
    # Create at least one valid YAML file
    (manager.work_dir / "infra").mkdir(parents=True, exist_ok=True)
    with open(manager.work_dir / "infra" / "test.yaml", "w") as f:
        yaml.dump({"apiVersion": "v1", "kind": "Namespace"}, f)

    result = await manager.validate_configuration(dry_run=True)
    assert result["valid"] is True
    assert result["recommendations"]
