#!/usr/bin/env python3
"""Landlock LSM helpers for filesystem sandboxing.

This module exposes only what is needed to restrict an untrusted process to a
read-only view of the filesystem plus a single writable scratch directory.  It
is intentionally self-contained and uses raw ``ctypes`` syscalls so it has no
build-time dependencies.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
from typing import List


class LandlockError(Exception):
    """Raised when Landlock setup cannot be completed."""


# Landlock FS access flags (stable since Linux 5.15).
_LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
_LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
_LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
_LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
_LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
_LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
_LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
_LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
_LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
_LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
_LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
_LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
_LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
_LANDLOCK_ACCESS_FS_REFER = 1 << 13
_LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14

_LANDLOCK_ACCESS_FS_ALL = (
    _LANDLOCK_ACCESS_FS_EXECUTE
    | _LANDLOCK_ACCESS_FS_WRITE_FILE
    | _LANDLOCK_ACCESS_FS_READ_FILE
    | _LANDLOCK_ACCESS_FS_READ_DIR
    | _LANDLOCK_ACCESS_FS_REMOVE_DIR
    | _LANDLOCK_ACCESS_FS_REMOVE_FILE
    | _LANDLOCK_ACCESS_FS_MAKE_CHAR
    | _LANDLOCK_ACCESS_FS_MAKE_DIR
    | _LANDLOCK_ACCESS_FS_MAKE_REG
    | _LANDLOCK_ACCESS_FS_MAKE_SOCK
    | _LANDLOCK_ACCESS_FS_MAKE_FIFO
    | _LANDLOCK_ACCESS_FS_MAKE_BLOCK
    | _LANDLOCK_ACCESS_FS_MAKE_SYM
    | _LANDLOCK_ACCESS_FS_REFER
    | _LANDLOCK_ACCESS_FS_TRUNCATE
)


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("_padding", ctypes.c_uint8 * 4),
    ]


def _landlock_syscalls() -> tuple:
    """Return (create, add_rule, restrict_self) syscall numbers for this machine."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return (444, 445, 446)
    if machine in ("aarch64", "arm64"):
        return (277, 278, 279)
    if machine.startswith("arm"):
        return (375, 376, 377)
    if machine.startswith("riscv"):
        return (277, 278, 279)
    raise LandlockError(f"Landlock syscall numbers not known for {machine}")


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _is_available() -> bool:
    if platform.system().lower() != "linux":
        return False
    try:
        _landlock_syscalls()
        libc = _libc()
        attr = _RulesetAttr(handled_access_fs=_LANDLOCK_ACCESS_FS_READ_FILE)
        ret = libc.syscall(_landlock_syscalls()[0], ctypes.byref(attr), ctypes.sizeof(attr), 0)
        if ret >= 0:
            os.close(ret)
            return True
        err = ctypes.get_errno()
        if err == errno.ENOSYS or err == errno.EOPNOTSUPP:
            return False
        return False
    except (LandlockError, OSError, AttributeError):
        return False


def _set_no_new_privs() -> None:
    """Set PR_SET_NO_NEW_PRIVS, required before landlock_restrict_self."""
    libc = _libc()
    PR_SET_NO_NEW_PRIVS = 38
    ret = libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    if ret != 0:
        raise LandlockError(f"prctl(PR_SET_NO_NEW_PRIVS) failed: {os.strerror(ctypes.get_errno())}")


def _create_ruleset() -> int:
    """Create a Landlock ruleset and return its file descriptor."""
    libc = _libc()
    create_nr, _, _ = _landlock_syscalls()
    attr = _RulesetAttr(handled_access_fs=_LANDLOCK_ACCESS_FS_ALL)
    ret = libc.syscall(create_nr, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ret < 0:
        raise LandlockError(
            f"landlock_create_ruleset failed: {os.strerror(ctypes.get_errno())}"
        )
    return int(ret)


def _add_path(ruleset_fd: int, path: str, allowed: int) -> None:
    """Add a path-beneath rule to the ruleset."""
    libc = _libc()
    _, add_rule_nr, _ = _landlock_syscalls()
    fd = os.open(path, os.O_PATH | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        rule = _PathBeneathAttr(allowed_access=allowed, parent_fd=fd)
        ret = libc.syscall(add_rule_nr, ruleset_fd, 0, ctypes.byref(rule), 0)
        if ret != 0:
            raise LandlockError(
                f"landlock_add_rule({path}) failed: {os.strerror(ctypes.get_errno())}"
            )
    finally:
        os.close(fd)


def _restrict_self(ruleset_fd: int) -> None:
    """Install the ruleset on the current thread."""
    libc = _libc()
    _, _, restrict_self_nr = _landlock_syscalls()
    ret = libc.syscall(restrict_self_nr, ruleset_fd, 0)
    if ret != 0:
        raise LandlockError(
            f"landlock_restrict_self failed: {os.strerror(ctypes.get_errno())}"
        )


def apply_landlock(
    read_dirs: List[str],
    write_dirs: List[str],
) -> None:
    """Restrict the current process with Landlock.

    * ``read_dirs`` are made read/execute.
    * ``write_dirs`` are made fully writable.

    The ruleset is installed in the current thread and is inherited by any
    child process created after this call (including ``exec``-ed programs).
    """
    read_access = (
        _LANDLOCK_ACCESS_FS_READ_FILE
        | _LANDLOCK_ACCESS_FS_READ_DIR
        | _LANDLOCK_ACCESS_FS_EXECUTE
    )
    write_access = _LANDLOCK_ACCESS_FS_ALL

    _set_no_new_privs()
    ruleset_fd = _create_ruleset()
    try:
        for path in read_dirs:
            if os.path.isdir(path):
                _add_path(ruleset_fd, path, read_access)
        for path in write_dirs:
            if os.path.isdir(path):
                _add_path(ruleset_fd, path, write_access)
        _restrict_self(ruleset_fd)
    finally:
        os.close(ruleset_fd)


def main() -> None:
    """CLI entry point used by LandlockRuntime to sandbox a command."""
    import json
    import sys

    if len(sys.argv) < 2:
        print("usage: landlock.py <config.json>", file=sys.stderr)
        sys.exit(1)

    cfg = json.loads(sys.argv[1])
    command = cfg["command"]
    env = cfg["env"]
    read_dirs = cfg.get("read_dirs", ["/"])
    write_dir = cfg.get("write_dir")

    if write_dir:
        os.makedirs(write_dir, exist_ok=True)
        os.chdir(write_dir)

    apply_landlock(read_dirs=read_dirs, write_dirs=[write_dir] if write_dir else [])
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
