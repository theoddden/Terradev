#!/usr/bin/env python3
"""Shared base classes and utilities for Terradev CLI commands."""

import asyncio

import click

from terradev_cli.commands._api import TerradevAPI


def get_api() -> TerradevAPI:
    """Resolve TerradevAPI from the current Click context, or instantiate one.

    Enables full dependency injection in tests via obj={"api": mock_api}.
    """
    ctx = click.get_current_context(silent=True)
    if ctx is not None and ctx.obj and ctx.obj.get("api"):
        return ctx.obj["api"]
    return TerradevAPI()


def run_with_timeout(coro, timeout: int = 300, operation: str = "Operation"):
    """Run an async coroutine with a timeout to prevent hangs.

    Exits with code 1 on timeout; propagates all other exceptions.
    """
    try:
        return asyncio.run(asyncio.wait_for(coro, timeout=timeout))
    except asyncio.TimeoutError:
        click.echo(f"ERROR: {operation} timed out after {timeout}s", err=True)
        raise SystemExit(1)


class TerradevCommand(click.Command):
    """Click Command subclass that catches unhandled errors and exits non-zero."""

    def invoke(self, ctx):
        try:
            rv = super().invoke(ctx)
        except (click.ClickException, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            click.echo(f"ERROR: {exc}", err=True)
            raise click.exceptions.Exit(1) from exc

        output = ctx.obj.get("terradev_output") if ctx.obj else None
        if output is not None and (rv is None or rv == 0):
            messages = getattr(output, "_messages", [])
            if any(m.level == "error" for m in messages):
                raise click.exceptions.Exit(1)
        return rv


class TerradevGroup(click.Group):
    """Click Group that propagates TerradevCommand/TerradevGroup to all descendants."""

    def command(self, *args, **kwargs):
        kwargs.setdefault("cls", TerradevCommand)
        return super().command(*args, **kwargs)

    def group(self, *args, **kwargs):
        kwargs.setdefault("cls", TerradevGroup)
        return super().group(*args, **kwargs)
