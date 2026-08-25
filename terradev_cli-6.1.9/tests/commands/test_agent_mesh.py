#!/usr/bin/env python3
"""Tests for ``terradev agent mesh``."""

from __future__ import annotations

import asyncio
import json

import pytest
from click.testing import CliRunner

from terradev_cli.commands.agent_infra.core import MeshError, MeshTransport
from terradev_cli.commands.agent_infra.mesh import (
    AgentCard,
    HttpTransport,
    InMemoryRegistry,
    MeshConfig,
    MeshNode,
    Task,
    TransportFactory,
    mesh,
)


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_mesh_registry():
    """Keep in-memory mesh state isolated between tests."""
    InMemoryRegistry.reset()


def test_agent_card_validation():
    card = AgentCard(name="test", endpoint="http://127.0.0.1:8000", skills=["search"])
    assert "search" in card.skills

    with pytest.raises(ValueError):
        AgentCard(name="bad", endpoint="not-a-url")


def test_transport_factory():
    transport = TransportFactory.create("http")
    assert isinstance(transport, HttpTransport)


def test_libp2p_is_available():
    from terradev_cli.commands.agent_infra.mesh import Libp2pTransport

    _run(Libp2pTransport().is_available())


def test_wireguard_is_not_available_without_wg():
    from terradev_cli.commands.agent_infra.mesh import WireGuardTransport

    available = _run(WireGuardTransport().is_available())
    assert available is False


def test_libp2p_transport_lifecycle():
    from terradev_cli.commands.agent_infra.mesh import Libp2pTransport, MeshConfig

    if not _run(Libp2pTransport().is_available()):
        pytest.skip("libp2p (p2pclient/p2pd) not available")

    async def _main():
        transport = Libp2pTransport()
        await transport.start(MeshConfig(listen="127.0.0.1:0", transport="libp2p"))
        try:
            card = AgentCard(name="node-lp2p", endpoint="/ip4/127.0.0.1/tcp/0", skills=["search"])
            await transport.publish_card(card)
            assert card.endpoint.startswith("/ip4/127.0.0.1/tcp/")
            assert "/p2p/" in card.endpoint
        finally:
            await transport.stop()

    _run(_main())


def test_mesh_node_lifecycle():
    async def _main():
        cfg = MeshConfig(listen="127.0.0.1:0", transport="http")
        node = MeshNode(cfg)
        await node.start()
        try:
            card = AgentCard(
                name="node-a",
                endpoint="http://127.0.0.1:0",
                skills=["search", "math"],
            )
            await node.publish_card(card)
            discovered = await node.discover(["math"])
            assert len(discovered) >= 1
        finally:
            await node.stop()

    _run(_main())


def test_mesh_task_delegation():
    async def _main():
        cfg = MeshConfig(listen="127.0.0.1:0", transport="http")
        node = MeshNode(cfg)
        await node.start()
        try:
            card = AgentCard(
                name="node-b",
                endpoint="http://127.0.0.1:0",
                skills=["echo"],
            )
            await node.publish_card(card)

            task = Task(input="hello mesh", skills=["echo"])
            completed = await node.delegate(card, task)
            assert completed.status == "in_progress"
            assert completed.input == "hello mesh"
        finally:
            await node.stop()

    _run(_main())


def test_route_by_strategy():
    async def _main():
        cfg = MeshConfig(listen="127.0.0.1:0", transport="http", routing="cost")
        node = MeshNode(cfg)
        await node.start()
        try:
            await node.publish_card(
                AgentCard(name="cheap", endpoint="http://127.0.0.1:1", skills=["a"])
            )
            await node.publish_card(
                AgentCard(name="cloud", endpoint="http://127.0.0.1:2", skills=["a"])
            )
            card = await node.route_by_strategy(skills=["a"], strategy="cost")
            assert card is not None
        finally:
            await node.stop()

    _run(_main())


def test_mesh_cli_help(runner: CliRunner):
    result = runner.invoke(mesh, ["--help"])
    assert result.exit_code == 0
    assert "node" in result.output
    assert "card" in result.output
    assert "task" in result.output


def test_mesh_peers_help(runner: CliRunner):
    result = runner.invoke(mesh, ["peers", "--help"])
    assert result.exit_code == 0
    assert "--transport" in result.output


def test_http_card_publish_is_idempotent():
    """Publishing the same card twice updates the existing entry, not duplicates."""

    async def _main():
        cfg = MeshConfig(listen="127.0.0.1:0", transport="http")
        transport = HttpTransport()
        await transport.start(cfg)
        try:
            card = AgentCard(name="node", endpoint="http://127.0.0.1:0", skills=["search"])
            await transport.publish_card(card)
            updated = AgentCard(name="node", endpoint="http://127.0.0.1:0", skills=["search", "math"])
            await transport.publish_card(updated)
            assert len(transport._cards) == 1
            assert InMemoryRegistry.cards[transport._self_endpoint()].skills == ["search", "math"]
        finally:
            await transport.stop()

    _run(_main())


def test_mesh_node_start_fails_gracefully_unavailable_transport():
    """MeshNode.start raises MeshError when the transport is unavailable."""

    class UnavailableTransport(MeshTransport):
        name = "unavailable"

        async def start(self, config: MeshConfig) -> None:
            pass

        async def stop(self) -> None:
            pass

        async def publish_card(self, card: AgentCard) -> None:
            pass

        async def discover_cards(self, skills=None):
            return []

        async def delegate_task(self, card: AgentCard, task: Task) -> Task:
            return task

        async def is_available(self) -> bool:
            return False

    async def _main():
        node = MeshNode(
            MeshConfig(transport="unavailable"),
            transport=UnavailableTransport(),
        )
        with pytest.raises(MeshError):
            await node.start()

    _run(_main())
