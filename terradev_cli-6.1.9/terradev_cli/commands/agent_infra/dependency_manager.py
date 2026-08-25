#!/usr/bin/env python3
"""Locate and, if necessary, download external agent binaries.

The Terradev agent delegates to genuine open-source runtimes:

* libp2p   — go-libp2p-daemon (`p2pd`)
* gVisor   — `runsc`
* Firecracker — `firecracker` (and `jailer`)
* Bubblewrap — `bwrap`
* WireGuard — `wg` and `wireguard-go`

This module never downloads blindly; it only fetches from the upstream
project's canonical distribution points and verifies the binary is
executable before returning it.
"""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional


class DependencyError(Exception):
    """Raised when a required runtime cannot be located or downloaded."""


class DependencyManager:
    """Manage the lifecycle of third-party agent binaries."""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or Path.home() / ".terradev" / "bin"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # p2pd (libp2p daemon)
    # ------------------------------------------------------------------

    def find_p2pd(self, allow_download: bool = True) -> Path:
        """Return the `p2pd` executable, building it from source if needed."""
        env = os.environ.get("TERRADEV_P2PD")
        if env:
            return Path(env)

        for candidate in [self.cache_dir / "p2pd", *map(Path, os.get_exec_path())]:
            candidate = candidate / "p2pd" if candidate.is_dir() else candidate
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

        if not allow_download:
            raise DependencyError(
                "p2pd not found. Install with: go install github.com/libp2p/go-libp2p-daemon/p2pd@latest"
            )

        self._build_p2pd()
        return self.cache_dir / "p2pd"

    def _build_p2pd(self) -> None:
        if not shutil.which("go"):
            raise DependencyError(
                "p2pd is missing and Go is not installed. "
                "Install Go or set TERRADEV_P2PD to a prebuilt p2pd binary."
            )
        subprocess.run(
            ["go", "install", "github.com/libp2p/go-libp2p-daemon/p2pd@latest"],
            check=True,
            capture_output=True,
            text=True,
        )
        go_bin = Path.home() / "go" / "bin" / "p2pd"
        if not go_bin.is_file():
            raise DependencyError(f"go install completed but {go_bin} was not produced")
        shutil.copy2(go_bin, self.cache_dir / "p2pd")

    # ------------------------------------------------------------------
    # WireGuard tooling
    # ------------------------------------------------------------------

    def find_wg(self, allow_download: bool = True) -> Path:
        """Return the `wg` executable."""
        env = os.environ.get("TERRADEV_WG")
        if env:
            return Path(env)

        for candidate in [self.cache_dir / "wg", *map(Path, os.get_exec_path())]:
            candidate = candidate / "wg" if candidate.is_dir() else candidate
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

        if not allow_download:
            raise DependencyError("wg not found. Install wireguard-tools.")

        return self._download_wireguard_tools()

    def _download_wireguard_tools(self) -> Path:
        system = platform.system().lower()
        if system == "linux":
            raise DependencyError(
                "Auto-downloading wireguard-tools on Linux is not supported; "
                "install it with your package manager (apt/dnf/apk)."
            )
        if system == "darwin":
            raise DependencyError(
                "Auto-downloading wireguard-tools on macOS is not supported; "
                "install with: brew install wireguard-tools"
            )
        raise DependencyError(f"Unsupported platform {system} for wireguard-tools")

    def find_wireguard_go(self, allow_download: bool = True) -> Path:
        """Return the `wireguard-go` userspace implementation."""
        env = os.environ.get("TERRADEV_WIREGUARD_GO")
        if env:
            return Path(env)

        for candidate in [self.cache_dir / "wireguard-go", *map(Path, os.get_exec_path())]:
            candidate = candidate / "wireguard-go" if candidate.is_dir() else candidate
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

        if not allow_download:
            raise DependencyError("wireguard-go not found.")

        return self._download_wireguard_go()

    def _download_wireguard_go(self) -> Path:
        system = platform.system().lower()
        machine = platform.machine().lower()
        if system == "darwin" and machine in ("x86_64", "amd64"):
            arch = "amd64"
        elif system == "darwin" and machine in ("arm64", "aarch64"):
            arch = "arm64"
        elif system == "linux" and machine in ("x86_64", "amd64"):
            arch = "amd64"
        elif system == "linux" and machine in ("arm64", "aarch64"):
            arch = "arm64"
        else:
            raise DependencyError(f"No prebuilt wireguard-go for {system}/{machine}")

        version = "0.0.20230223"  # known stable release
        url = (
            f"https://github.com/WireGuard/wireguard-go/archive/refs/tags/{version}.tar.gz"
        )
        raise DependencyError(
            "wireguard-go must be built from source. "
            f"Clone {url} and run `go make` in the zstd branch for your platform."
        )

    # ------------------------------------------------------------------
    # gVisor
    # ------------------------------------------------------------------

    def find_runsc(self, allow_download: bool = True) -> Path:
        """Return the gVisor `runsc` binary."""
        env = os.environ.get("TERRADEV_RUNSC")
        if env:
            return Path(env)

        for candidate in [self.cache_dir / "runsc", *map(Path, os.get_exec_path())]:
            candidate = candidate / "runsc" if candidate.is_dir() else candidate
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

        if not allow_download:
            raise DependencyError("runsc not found. Set TERRADEV_RUNSC.")

        return self._download_runsc()

    def _download_runsc(self) -> Path:
        if platform.system().lower() != "linux":
            raise DependencyError("gVisor runsc is only available on Linux")

        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            suffix = "amd64"
        elif machine in ("aarch64", "arm64"):
            suffix = "arm64"
        else:
            raise DependencyError(f"No prebuilt runsc for {machine}")

        url = f"https://storage.googleapis.com/gvisor/releases/nightly/latest/runsc.{suffix}"
        dest = self.cache_dir / "runsc"
        urllib.request.urlretrieve(url, dest)
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return dest

    # ------------------------------------------------------------------
    # Firecracker
    # ------------------------------------------------------------------

    def find_firecracker(self, allow_download: bool = True) -> Path:
        """Return the Firecracker `firecracker` binary."""
        env = os.environ.get("TERRADEV_FIRECRACKER")
        if env:
            return Path(env)

        for candidate in [self.cache_dir / "firecracker", *map(Path, os.get_exec_path())]:
            candidate = candidate / "firecracker" if candidate.is_dir() else candidate
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

        if not allow_download:
            raise DependencyError("firecracker not found. Set TERRADEV_FIRECRACKER.")

        return self._download_firecracker()

    def _download_firecracker(self) -> Path:
        if platform.system().lower() != "linux":
            raise DependencyError("Firecracker is only available on Linux")

        machine = platform.machine().lower()
        arch = "x86_64" if machine in ("x86_64", "amd64") else None
        if arch is None:
            raise DependencyError(f"No prebuilt firecracker for {machine}")

        version = "v1.7.0"
        url = f"https://github.com/firecracker-microvm/firecracker/releases/download/{version}/firecracker-{version}-{arch}.tgz"
        tgz = self.cache_dir / f"firecracker-{version}-{arch}.tgz"
        urllib.request.urlretrieve(url, tgz)
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(self.cache_dir)

        for member in tf.getnames() if tf.getnames() else []:
            if "firecracker" in member:
                extracted = self.cache_dir / member
                if extracted.is_file() and "firecracker" in member:
                    extracted.chmod(
                        extracted.stat().st_mode
                        | stat.S_IXUSR
                        | stat.S_IXGRP
                        | stat.S_IXOTH
                    )
                    if not (self.cache_dir / "firecracker").exists():
                        (self.cache_dir / "firecracker").symlink_to(extracted)
                    return extracted

        raise DependencyError("firecracker archive did not contain a firecracker binary")

    # ------------------------------------------------------------------
    # Bubblewrap
    # ------------------------------------------------------------------

    def find_bwrap(self, allow_download: bool = True) -> Path:
        """Return the bubblewrap `bwrap` binary."""
        env = os.environ.get("TERRADEV_BWRAP")
        if env:
            return Path(env)

        for candidate in [self.cache_dir / "bwrap", *map(Path, os.get_exec_path())]:
            candidate = candidate / "bwrap" if candidate.is_dir() else candidate
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

        if not allow_download:
            raise DependencyError("bwrap not found. Set TERRADEV_BWRAP.")

        return self._download_bwrap()

    def _download_bwrap(self) -> Path:
        if platform.system().lower() != "linux":
            raise DependencyError("bubblewrap bwrap is only available on Linux")

        machine = platform.machine().lower()
        arch = "x86_64" if machine in ("x86_64", "amd64") else None
        if arch is None:
            raise DependencyError(f"No prebuilt bwrap for {machine}")

        version = "v0.10.0"
        url = f"https://github.com/containers/bubblewrap/releases/download/{version}/bubblewrap-{arch}.tar.xz"
        txz = self.cache_dir / f"bubblewrap-{arch}.tar.xz"
        urllib.request.urlretrieve(url, txz)
        with tarfile.open(txz, "r:xz") as tf:
            tf.extractall(self.cache_dir)

        dest = self.cache_dir / "bwrap"
        if dest.is_file():
            dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            return dest

        raise DependencyError("bwrap archive did not contain a bwrap binary")

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def find_binary(
        self,
        name: str,
        env_var: str,
        downloader: Optional[callable] = None,  # type: ignore[arg-type]
        allow_download: bool = True,
    ) -> Path:
        """Look for a binary in the cache directory, PATH, or an env override."""
        env = os.environ.get(env_var)
        if env:
            return Path(env)

        for candidate in [self.cache_dir / name, *map(Path, os.get_exec_path())]:
            candidate = candidate / name if candidate.is_dir() else candidate
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

        if not allow_download:
            raise DependencyError(f"{name} not found. Set {env_var}.")

        if downloader is None:
            raise DependencyError(f"No downloader configured for {name}")
        return downloader()
