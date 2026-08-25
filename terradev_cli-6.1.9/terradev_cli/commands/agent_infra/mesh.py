#!/usr/bin/env python3
"""``terradev agent mesh`` — decentralized agent-to-agent communication.

The mesh is transport-agnostic.  A working HTTP transport is provided out of
the box; libp2p and WireGuard transports are pluggable via the same interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from typing import Any, Dict, List, Optional, Type

import aiohttp
import click
from aiohttp import web

try:
    from multiaddr import Multiaddr, protocols
except ImportError:  # pragma: no cover
    Multiaddr = None  # type: ignore[assignment, misc]
    protocols = None  # type: ignore[assignment, misc]

from .core import (
    AgentCard,
    MeshConfig,
    MeshError,
    MeshRoutingStrategy,
    MeshTopology,
    MeshTransport,
    Task,
)
from .dependency_manager import DependencyError, DependencyManager
from .otel import Tracer, get_tracer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory / HTTP transport
# ---------------------------------------------------------------------------


class InMemoryRegistry:
    """Shared in-memory registry for tests and single-process demos."""

    cards: Dict[str, AgentCard] = {}
    tasks: Dict[str, Task] = {}

    @classmethod
    def reset(cls) -> None:
        cls.cards = {}
        cls.tasks = {}


class HttpTransport(MeshTransport):
    """A2A-over-HTTP transport with an embedded aiohttp server."""

    name = "http"

    def __init__(self):
        self.config: Optional[MeshConfig] = None
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self._cards: Dict[str, AgentCard] = {}

    async def start(self, config: MeshConfig) -> None:
        self.config = config
        self.app = web.Application()
        self.app.router.add_get("/.well-known/agent.json", self._handle_card)
        self.app.router.add_post("/a2a/task", self._handle_task)
        self.app.router.add_get("/a2a/tasks/{task_id}", self._handle_task_status)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        host, port = self._parse_listen(config.listen)
        self.site = web.TCPSite(self.runner, host, port)
        await self.site.start()

        # Resolve the real port when port 0 was used.
        if self.site._server is not None:
            for sock in self.site._server.sockets:
                _, bound_port = sock.getsockname()[:2]
                self.config.listen = f"{host}:{bound_port}"
                break

        logger.info(f"HTTP mesh node listening on {self.config.listen}")

    async def stop(self) -> None:
        if self.site:
            try:
                await self.site.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("HTTP site stop failed: %s", exc)
        if self.runner:
            try:
                await self.runner.cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.warning("HTTP runner cleanup failed: %s", exc)

    async def publish_card(self, card: AgentCard) -> None:
        # Publish this node's actual bound endpoint, not a wildcard like :0.
        card.endpoint = self._self_endpoint()
        self._cards[card.endpoint] = card
        InMemoryRegistry.cards[card.endpoint] = card

    async def discover_cards(self, skills: Optional[List[str]] = None) -> List[AgentCard]:
        cards = list(InMemoryRegistry.cards.values())
        if not skills:
            return cards
        return [c for c in cards if any(s in c.skills for s in skills)]

    async def delegate_task(self, card: AgentCard, task: Task) -> Task:
        payload = task.model_dump_json()
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(f"{card.endpoint}/a2a/task", data=payload) as resp:
                    data = await resp.json()
                    return Task.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            raise MeshError(f"Failed to delegate task to {card.endpoint}: {exc}") from exc

    async def _handle_card(self, request: web.Request) -> web.Response:
        endpoint = self._self_endpoint()
        card = self._cards.get(endpoint)
        if not card:
            card = AgentCard(
                name="default",
                endpoint=endpoint,
                skills=[],
            )
        return web.json_response(card.model_dump())

    async def _handle_task(self, request: web.Request) -> web.Response:
        data = await request.json()
        task = Task.model_validate(data)
        task.status = "in_progress"
        InMemoryRegistry.tasks[task.id] = task
        return web.json_response(task.model_dump())

    async def _handle_task_status(self, request: web.Request) -> web.Response:
        task_id = request.match_info["task_id"]
        task = InMemoryRegistry.tasks.get(task_id)
        if not task:
            raise web.HTTPNotFound()
        return web.json_response(task.model_dump())

    def _self_endpoint(self) -> str:
        if not self.site:
            return "http://127.0.0.1:0"
        return f"http://{self.config.listen}"

    def _parse_listen(self, listen: str) -> tuple:
        if ":" in listen:
            host, port = listen.rsplit(":", 1)
            return host, int(port)
        return "127.0.0.1", int(listen)


class Libp2pTransport(MeshTransport):
    """Real libp2p transport using the go-libp2p-daemon and p2pclient."""

    name = "libp2p"
    PROTOCOL = "/terradev/agent/1.0.0"
    CARD_TOPIC = "terradev:agent-cards"

    def __init__(self) -> None:
        self.config: Optional[MeshConfig] = None
        self._daemon: Any = None
        self._client: Any = None
        self._client_cm: Any = None
        self._peer_id: Any = None
        self._addrs: List[Multiaddr] = []
        self._endpoint: str = ""
        self._cards: List[AgentCard] = []
        self._tracer = get_tracer("terradev.agent.mesh.libp2p")

    async def is_available(self) -> bool:
        try:
            import p2pclient  # noqa: F401
        except (ImportError, OSError):
            return False
        try:
            DependencyManager().find_p2pd(allow_download=False)
            return True
        except DependencyError:
            return False

    async def start(self, config: MeshConfig) -> None:
        from p2pclient.daemon import GoDaemon, get_unused_tcp_port
        from p2pclient.p2pclient import Client

        self.config = config
        p2pd_path = str(DependencyManager().find_p2pd())

        control_port = get_unused_tcp_port()
        listen_port = get_unused_tcp_port()
        control_maddr = Multiaddr(f"/ip4/127.0.0.1/tcp/{control_port}")
        listen_maddr = Multiaddr(f"/ip4/127.0.0.1/tcp/{listen_port}")

        self._daemon = GoDaemon(
            daemon_executable=p2pd_path,
            control_maddr=control_maddr,
            enable_control=True,
            enable_connmgr=False,
            enable_dht=True,
            enable_pubsub=True,
        )

        with self._tracer.trace("libp2p.daemon.start"):
            try:
                await asyncio.wait_for(self._daemon.wait_until_ready(), timeout=30)
            except Exception as exc:  # noqa: BLE001
                raise MeshError(f"libp2p daemon failed to start: {exc}") from exc

        self._client = Client(control_maddr=control_maddr, listen_maddr=listen_maddr)
        self._client_cm = self._client.listen()
        self._client = await self._client_cm.__aenter__()
        await self._client.stream_handler(self.PROTOCOL, self._handle_stream)

        peer_id, addrs = await asyncio.wait_for(self._client.identify(), timeout=5)
        self._peer_id = peer_id
        self._addrs = list(addrs)
        if self._addrs:
            self._endpoint = f"{self._addrs[0]}/p2p/{self._peer_id.to_base58()}"
        else:
            self._endpoint = f"/p2p/{self._peer_id.to_base58()}"

        if config.bootstrap:
            await self._bootstrap()

    async def stop(self) -> None:
        if self._client_cm is not None:
            try:
                await self._client_cm.__aexit__(None, None, None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("libp2p client cleanup error: %s", exc)
        if self._daemon is not None:
            try:
                self._daemon.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("libp2p daemon cleanup error: %s", exc)
        self._client = None
        self._client_cm = None
        self._daemon = None

    async def publish_card(self, card: AgentCard) -> None:
        card.endpoint = self._endpoint
        self._cards.append(card)
        data = json.dumps(card.model_dump()).encode()
        try:
            await self._client.pubsub_publish(self.CARD_TOPIC, data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("libp2p card publish failed: %s", exc)

    async def discover_cards(self, skills: Optional[List[str]] = None, timeout: float = 2.0) -> List[AgentCard]:
        from p2pclient.pb import p2pd_pb2 as p2pd_pb
        from p2pclient.utils import read_pbmsg_safe

        sub_stream = await self._client.pubsub_subscribe(self.CARD_TOPIC)
        seen: set = set()
        result: List[AgentCard] = []
        deadline = asyncio.get_event_loop().time() + timeout
        try:
            while asyncio.get_event_loop().time() < deadline:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                msg = p2pd_pb.PSMessage()
                try:
                    await asyncio.wait_for(read_pbmsg_safe(sub_stream, msg), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                try:
                    card = AgentCard.model_validate_json(msg.data)
                except Exception:  # noqa: BLE001
                    continue
                if card.endpoint not in seen:
                    seen.add(card.endpoint)
                    result.append(card)
        finally:
            try:
                await sub_stream.aclose()
            except Exception:  # noqa: BLE001
                pass

        if not skills:
            return result
        return [c for c in result if any(s in c.skills for s in skills)]

    async def delegate_task(self, card: AgentCard, task: Task) -> Task:
        maddr = Multiaddr(card.endpoint)
        peer_id_bytes = maddr.value_for_protocol(protocols.P_P2P)
        if peer_id_bytes is None:
            raise MeshError(f"libp2p endpoint missing peer id: {card.endpoint}")

        from p2pclient.libp2p_stubs.peer.id import ID

        peer_id = ID(peer_id_bytes.encode() if isinstance(peer_id_bytes, str) else peer_id_bytes)
        # Build a multiaddr without the /p2p suffix for connect.
        addr_str = "/".join(str(p) for p in maddr.protocols() if p.code != protocols.P_P2P)
        maddrs = [Multiaddr(addr_str)]

        with self._tracer.trace("libp2p.task.delegate", {"peer": str(peer_id)}):
            try:
                await self._client.connect(peer_id, maddrs)
                sinfo, stream = await self._client.stream_open(peer_id, [self.PROTOCOL])
                payload = json.dumps(task.model_dump()).encode()
                await self._write_framed(stream, payload)
                raw = await self._read_framed(stream)
                return Task.model_validate_json(raw)
            except Exception as exc:  # noqa: BLE001
                raise MeshError(f"libp2p task delegation failed: {exc}") from exc

    async def _bootstrap(self) -> None:
        from p2pclient.libp2p_stubs.peer.id import ID

        for addr in self.config.bootstrap or []:
            try:
                maddr = Multiaddr(addr)
                peer_id_b58 = maddr.value_for_protocol(protocols.P_P2P)
                if peer_id_b58 is None:
                    logger.warning("libp2p bootstrap addr missing /p2p: %s", addr)
                    continue
                peer_id = ID(peer_id_b58.encode() if isinstance(peer_id_b58, str) else peer_id_b58)
                stripped = "/".join(str(p) for p in maddr.protocols() if p.code != protocols.P_P2P)
                await self._client.connect(peer_id, [Multiaddr(stripped)])
                logger.info("libp2p bootstrapped to %s", addr)
            except Exception as exc:  # noqa: BLE001
                logger.warning("libp2p bootstrap to %s failed: %s", addr, exc)

    async def _handle_stream(self, sinfo: Any, stream: Any) -> None:
        try:
            raw = await self._read_framed(stream)
            task = Task.model_validate_json(raw)
            # Minimal local execution echo: mark as completed.
            task.status = "completed"
            task.result = {"ack": True, "handler": "libp2p"}
            resp = json.dumps(task.model_dump()).encode()
            await self._write_framed(stream, resp)
        except Exception as exc:  # noqa: BLE001
            logger.error("libp2p stream handler error: %s", exc)
        finally:
            try:
                await stream.aclose()
            except Exception:  # noqa: BLE001
                pass

    async def _read_framed(self, stream: Any) -> bytes:
        header = await stream.receive(4)
        if len(header) < 4:
            raise MeshError("libp2p short frame header")
        length = struct.unpack(">I", header)[0]
        chunks: List[bytes] = []
        received = 0
        while received < length:
            chunk = await stream.receive(min(65535, length - received))
            if not chunk:
                raise MeshError("libp2p stream closed before full frame")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    async def _write_framed(self, stream: Any, data: bytes) -> None:
        await stream.send(struct.pack(">I", len(data)) + data)


class WireGuardTransport(MeshTransport):
    """WireGuard-encrypted A2A mesh transport using the wireguard package and `wg` tools."""

    name = "wireguard"

    def __init__(self) -> None:
        self.config: Optional[MeshConfig] = None
        self._interface: str = "tdev0"
        self._port: int = 0
        self._local_ip: str = ""
        self._http: HttpTransport = HttpTransport()
        self._service: Any = None
        self._tracer = get_tracer("terradev.agent.mesh.wireguard")

    async def is_available(self) -> bool:
        try:
            import wireguard  # noqa: F401
        except (ImportError, OSError):
            return False
        try:
            DependencyManager().find_wg(allow_download=False)
            return True
        except DependencyError:
            return False

    async def start(self, config: MeshConfig) -> None:
        import wireguard

        self.config = config
        host, port = self._parse_listen(config.listen)
        self._port = port

        # Pick a stable address in the tdev mesh subnet.
        if config.identity and config.identity.isdigit() and 2 <= int(config.identity) <= 254:
            self._local_ip = f"10.254.0.{config.identity}"
        else:
            self._local_ip = "10.254.0.2"

        peer = wireguard.Peer(
            description=config.identity or "tdev-peer",
            address=f"{self._local_ip}/24",
            allowed_ips="10.254.0.0/24",
            interface=self._interface,
        )

        with self._tracer.trace("wireguard.interface.up"):
            # wireguard writes to /etc/wireguard/<interface>.conf and uses wg-quick
            peer.config.write()
            self._service = wireguard.service.Interface(self._interface)
            self._service.start()

        # Run A2A HTTP over the WireGuard interface.
        self._http.config = MeshConfig(
            listen=f"{self._local_ip}:{self._port}",
            transport="http",
        )
        await self._http.start(self._http.config)

    async def stop(self) -> None:
        if self._http:
            await self._http.stop()
        if self._service:
            try:
                self._service.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("wireguard down failed: %s", exc)

    async def publish_card(self, card: AgentCard) -> None:
        card.endpoint = f"http://{self._local_ip}:{self._port}"
        await self._http.publish_card(card)

    async def discover_cards(self, skills: Optional[List[str]] = None) -> List[AgentCard]:
        return await self._http.discover_cards(skills)

    async def delegate_task(self, card: AgentCard, task: Task) -> Task:
        return await self._http.delegate_task(card, task)

    def _parse_listen(self, listen: str) -> tuple:
        if ":" in listen:
            host, port = listen.rsplit(":", 1)
            return host, int(port)
        return "127.0.0.1", int(listen)


# ---------------------------------------------------------------------------
# Transport factory
# ---------------------------------------------------------------------------


class TransportFactory:
    """Instantiate the selected mesh transport."""

    _transports: Dict[str, Type[MeshTransport]] = {}

    @classmethod
    def register(cls, transport_cls: Type[MeshTransport]) -> Type[MeshTransport]:
        cls._transports[transport_cls.name] = transport_cls
        return transport_cls

    @classmethod
    def get(cls, name: str) -> Optional[Type[MeshTransport]]:
        return cls._transports.get(name)

    @classmethod
    def create(cls, name: str) -> MeshTransport:
        transport_cls = cls._transports.get(name)
        if not transport_cls:
            raise MeshError(f"Unknown mesh transport: {name}")
        return transport_cls()


TransportFactory.register(HttpTransport)
TransportFactory.register(Libp2pTransport)
TransportFactory.register(WireGuardTransport)


# ---------------------------------------------------------------------------
# Mesh node
# ---------------------------------------------------------------------------


class MeshNode:
    """A composable A2A mesh node."""

    def __init__(
        self,
        config: MeshConfig,
        transport: Optional[MeshTransport] = None,
        tracer: Optional[Tracer] = None,
    ):
        self.config = config
        self.transport = transport or TransportFactory.create(config.transport)
        self.tracer = tracer or get_tracer("terradev.agent.mesh")
        self._started = False

    async def start(self) -> None:
        with self.tracer.trace("agent.mesh.start", {"transport": self.config.transport}):
            if self._started:
                raise MeshError("Mesh node is already started")
            if not await self.transport.is_available():
                raise MeshError(
                    f"Mesh transport '{self.config.transport}' is not available "
                    "(missing dependency or unsupported platform)"
                )
            await self.transport.start(self.config)
            self._started = True

    async def stop(self) -> None:
        with self.tracer.trace("agent.mesh.stop", {"transport": self.config.transport}):
            if not self._started:
                return
            try:
                await self.transport.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Mesh transport stop failed: %s", exc)
            self._started = False

    async def publish_card(self, card: AgentCard) -> None:
        with self.tracer.trace("agent.mesh.publish_card", {"name": card.name, "skills": card.skills}):
            await self.transport.publish_card(card)

    async def discover(self, skills: Optional[List[str]] = None) -> List[AgentCard]:
        with self.tracer.trace("agent.mesh.discover", {"skills": skills}):
            return await self.transport.discover_cards(skills)

    async def delegate(self, card: AgentCard, task: Task) -> Task:
        with self.tracer.trace(
            "agent.mesh.delegate",
            {"peer": card.name, "task_id": task.id, "skills": task.skills},
        ):
            return await self.transport.delegate_task(card, task)

    async def route_by_strategy(
        self,
        skills: Optional[List[str]] = None,
        strategy: Optional[MeshRoutingStrategy] = None,
    ) -> Optional[AgentCard]:
        """Select a peer card according to the configured routing objective."""
        strategy = strategy or self.config.routing
        cards = await self.discover(skills)
        if not cards:
            return None

        # For the HTTP demo, latency proxy is just hop count; cost is mocked
        # by provider name.  Real deployments extend this with live telemetry.
        if strategy == MeshRoutingStrategy.COST:
            # Prefer peers on cheaper hypothetical providers (mock).
            return min(cards, key=lambda c: 0 if "local" in c.endpoint else 1)
        if strategy == MeshRoutingStrategy.THROUGHPUT:
            return max(cards, key=lambda c: len(c.skills))
        return cards[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group("mesh")
def mesh():
    """Decentralized agent-to-agent communication and state sync."""
    pass


@mesh.group("node")
def node():
    """Manage a mesh node."""
    pass


@node.command("join")
@click.option(
    "--protocol",
    default="a2a",
    type=click.Choice(["a2a", "anp"]),
    help="Agent protocol to speak",
)
@click.option(
    "--transport",
    default="http",
    type=click.Choice(["http", "libp2p", "wireguard"]),
    help="Underlying transport",
)
@click.option(
    "--topology",
    default="decentralized",
    type=click.Choice(["decentralized", "hierarchical"]),
    help="Mesh topology",
)
@click.option("--listen", default="127.0.0.1:4222", help="Listen address")
@click.option("--bootstrap", multiple=True, help="Bootstrap peer multiaddr or URL")
@click.option("--config", "-c", type=click.Path(exists=True, dir_okay=False), help="Config file")
@click.option("--timeout", default=30, type=int, help="Task delegation timeout (seconds)")
def mesh_node_join(protocol, transport, topology, listen, bootstrap, config, timeout):
    """Join the agent mesh as a node."""
    import asyncio

    async def _main():
        cfg = MeshConfig(
            protocol=protocol,
            transport=transport,
            listen=listen,
            bootstrap=list(bootstrap),
            topology=MeshTopology(topology),
        )

        if config:
            with open(config, "r", encoding="utf-8") as f:
                data = json.load(f) if config.endswith(".json") else {}
            if data:
                cfg = cfg.model_copy(update=data)

        node = MeshNode(cfg)
        try:
            await node.start()
            click.echo(f"OK: Mesh node listening on {listen} ({transport})")
            # Keep alive until interrupted.
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            await node.stop()

    try:
        asyncio.run(_main())
    except (click.ClickException, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1)


@mesh.group("card")
def card():
    """Manage A2A Agent Cards."""
    pass


@card.command("publish")
@click.option("--name", "-n", required=True, help="Agent name")
@click.option("--endpoint", "-e", required=True, help="Agent endpoint URL")
@click.option("--skills", "-s", multiple=True, help="Comma-separated skills")
@click.option("--version", default="1.0.0", help="Agent version")
@click.option("--listen", default="127.0.0.1:4222", help="Local node listen address")
@click.option("--transport", default="http", type=click.Choice(["http", "libp2p", "wireguard"]))
def mesh_card_publish(name, endpoint, skills, version, listen, transport):
    """Publish an Agent Card to the mesh."""
    import asyncio

    async def _main():
        skill_list = [s for item in skills for s in item.split(",") if s]
        cfg = MeshConfig(listen=listen, transport=transport)
        node = MeshNode(cfg)
        await node.start()
        try:
            card = AgentCard(
                name=name,
                endpoint=endpoint,
                skills=skill_list,
                version=version,
            )
            await node.publish_card(card)
            click.echo(f"OK: Published Agent Card for {name}: {endpoint}")
        finally:
            await node.stop()

    try:
        asyncio.run(_main())
    except (click.ClickException, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1)


@mesh.group("task")
def task():
    """Manage mesh tasks."""
    pass


@task.command("create")
@click.option("--input", "-i", required=True, help="Task input/prompt")
@click.option("--skills", "-s", multiple=True, help="Comma-separated skills")
@click.option("--peer", "-p", help="Specific peer endpoint to delegate to")
@click.option("--slo", default="latency", type=click.Choice(["latency", "cost", "throughput"]), help="Routing strategy")
@click.option("--listen", default="127.0.0.1:4222", help="Local node listen address")
@click.option("--transport", default="http", type=click.Choice(["http", "libp2p", "wireguard"]))
@click.option("--format", type=click.Choice(["text", "json"]), default="text")
def mesh_task_create(input, skills, peer, slo, listen, transport, format):
    """Create and delegate a task to the mesh."""
    import asyncio

    async def _main():
        skill_list = [s for item in skills for s in item.split(",") if s]
        cfg = MeshConfig(
            listen=listen,
            transport=transport,
            routing=MeshRoutingStrategy(slo),
        )
        node = MeshNode(cfg)
        await node.start()
        try:
            if peer:
                card = AgentCard(name="selected", endpoint=peer, skills=skill_list)
            else:
                card = await node.route_by_strategy(skills=skill_list)

            if not card:
                click.echo("ERROR: No peer found for requested skills", err=True)
                raise SystemExit(1)

            t = Task(input=input, skills=skill_list)
            completed = await node.delegate(card, t)

            if format == "json":
                click.echo(json.dumps(completed.model_dump(), indent=2, default=str))
            else:
                click.echo(f"OK: Task {completed.id} -> {card.name}")
                click.echo(f"Status: {completed.status}")
                for a in completed.artifacts:
                    for p in a.parts:
                        click.echo(f"-- {a.name} ({p.content_type}) --")
                        click.echo(p.content)

            return completed
        finally:
            await node.stop()

    try:
        asyncio.run(_main())
    except (click.ClickException, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1)


@mesh.command("peers")
@click.option("--skills", "-s", multiple=True, help="Filter by skill")
@click.option("--listen", default="127.0.0.1:4222")
@click.option("--transport", default="http", type=click.Choice(["http", "libp2p", "wireguard"]))
@click.option("--format", type=click.Choice(["text", "json"]), default="text")
def mesh_peers(skills, listen, transport, format):
    """List known peers in the mesh."""
    import asyncio

    async def _main():
        skill_list = [s for item in skills for s in item.split(",") if s]
        cfg = MeshConfig(listen=listen, transport=transport)
        node = MeshNode(cfg)
        await node.start()
        try:
            cards = await node.discover(skill_list)
            if format == "json":
                click.echo(json.dumps([c.model_dump() for c in cards], indent=2))
            else:
                click.echo(f"PEERS ({len(cards)})")
                for c in cards:
                    click.echo(f"  {c.name:<20} {c.endpoint:<40} {','.join(c.skills)}")
        finally:
            await node.stop()

    try:
        asyncio.run(_main())
    except (click.ClickException, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1)


@mesh.command("route")
@click.option("--task-id", help="Task to inspect routing for")
@click.option("--skills", "-s", multiple=True)
@click.option("--listen", default="127.0.0.1:4222")
@click.option("--transport", default="http", type=click.Choice(["http", "libp2p", "wireguard"]))
@click.option("--format", type=click.Choice(["text", "json"]), default="text")
def mesh_route(task_id, skills, listen, transport, format):
    """Show the selected route for a set of skills."""
    import asyncio

    async def _main():
        skill_list = [s for item in skills for s in item.split(",") if s]
        cfg = MeshConfig(listen=listen, transport=transport)
        node = MeshNode(cfg)
        await node.start()
        try:
            card = await node.route_by_strategy(skills=skill_list)
            if not card:
                click.echo("No route available.")
                return
            if format == "json":
                click.echo(json.dumps(card.model_dump(), indent=2))
            else:
                click.echo(f"ROUTE: {card.name} @ {card.endpoint} skills={','.join(card.skills)}")
        finally:
            await node.stop()

    try:
        asyncio.run(_main())
    except (click.ClickException, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1)
