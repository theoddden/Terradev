#!/usr/bin/env python3
"""Letta (formerly MemGPT) integration for the Terradev CLI agent command."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import click


@click.group()
def letta():
    """Letta stateful agents with long-horizon memory management.

    Build LLM agents that manage their own context, memory, and state across
    sessions using Letta's virtual memory system.
    """
    pass


def _get_client(environment: str = "cloud"):
    """Return a configured Letta client or raise a helpful error."""
    try:
        from letta_client import Letta
    except ImportError as e:  # pragma: no cover
        click.echo(
            "ERROR: letta-client is not installed. Install with:\n"
            "  pip install letta-client\n"
            "or see https://docs.letta.com/api/python",
            err=True,
        )
        raise SystemExit(1) from e

    api_key = os.environ.get("LETTA_API_KEY")
    return Letta(
        api_key=api_key,
        environment=environment,
    )


@letta.command("create")
@click.option("--name", "-n", required=True, help="Agent name")
@click.option("--model", "-m", default="openai/gpt-4.1", help="Model to use")
@click.option(
    "--human",
    default="",
    help="Human memory block value",
)
@click.option(
    "--persona",
    default="",
    help="Persona memory block value",
)
@click.option(
    "--memory-blocks",
    default="[]",
    help='JSON list of memory blocks [{"label": ..., "value": ...}]',
)
@click.option(
    "--vector-db",
    default=None,
    help='Vector DB connection string or JSON config for agent memory',
)
@click.option(
    "--skill",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help='Path to a skill.md to embed as an agent memory block',
)
@click.option(
    "--environment",
    default="cloud",
    type=click.Choice(["cloud", "local"]),
    help="Letta environment",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="text",
)
def letta_create(name, model, human, persona, memory_blocks, vector_db, skill, environment, fmt):
    """Create a new stateful Letta agent.

    Examples:
      terradev agent letta create --name my-agent --model openai/gpt-4.1
      terradev agent letta create --name devops \
        --human "Name: Timber" --persona "I am a helpful SRE"
      terradev agent letta create --name rag \
        --vector-db qdrant://localhost:6333 \
        --skill ./research.skill.md
    """
    client = _get_client(environment)

    blocks = json.loads(memory_blocks)
    if human:
        blocks.append({"label": "human", "value": human})
    if persona:
        blocks.append({"label": "persona", "value": persona})
    if vector_db:
        blocks.append({"label": "vector_db", "value": vector_db})
    if skill:
        skill_path = Path(skill)
        if skill_path.exists():
            blocks.append({"label": "skill", "value": skill_path.read_text()})
        else:
            click.echo(f"ERROR: skill file not found: {skill}", err=True)
            sys.exit(1)

    try:
        agent = client.agents.create(
            name=name,
            model=model,
            memory_blocks=blocks,
        )
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: failed to create Letta agent: {e}", err=True)
        sys.exit(1)

    if fmt == "json":
        click.echo(json.dumps({"id": agent.id, "name": name, "model": model}, indent=2))
    else:
        click.echo(f"Created Letta agent: {agent.id}")
        click.echo(f"  Name:  {name}")
        click.echo(f"  Model: {model}")


@letta.command("list")
@click.option(
    "--environment",
    default="cloud",
    type=click.Choice(["cloud", "local"]),
    help="Letta environment",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="text",
)
def letta_list(environment, fmt):
    """List Letta agents."""
    client = _get_client(environment)

    try:
        agents = client.agents.list()
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: failed to list Letta agents: {e}", err=True)
        sys.exit(1)

    if fmt == "json":
        click.echo(json.dumps([{"id": a.id, "name": getattr(a, "name", "")} for a in agents], indent=2))
    else:
        if not agents:
            click.echo("No Letta agents found.")
            return
        click.echo("Letta agents:")
        for agent in agents:
            name = getattr(agent, "name", "")
            click.echo(f"  {agent.id}  {name}")


@letta.command("chat")
@click.option("--agent-id", "-a", required=True, help="Agent ID")
@click.option("--message", "-m", required=True, help="Message to send")
@click.option(
    "--environment",
    default="cloud",
    type=click.Choice(["cloud", "local"]),
    help="Letta environment",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="text",
)
def letta_chat(agent_id, message, environment, fmt):
    """Send a message to a Letta agent."""
    client = _get_client(environment)

    try:
        response = client.agents.messages.create(
            agent_id=agent_id,
            input=message,
        )
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: failed to chat with Letta agent: {e}", err=True)
        sys.exit(1)

    if fmt == "json":
        messages = []
        for msg in response.messages:
            messages.append(
                {
                    "id": getattr(msg, "id", None),
                    "role": getattr(msg, "role", "assistant"),
                    "content": getattr(msg, "content", None),
                }
            )
        click.echo(json.dumps({"agent_id": agent_id, "messages": messages}, indent=2))
    else:
        click.echo(f"Agent {agent_id}:")
        for msg in response.messages:
            content = getattr(msg, "content", None)
            if content:
                click.echo(content)


@letta.command("status")
@click.option("--agent-id", "-a", required=True, help="Agent ID")
@click.option(
    "--environment",
    default="cloud",
    type=click.Choice(["cloud", "local"]),
    help="Letta environment",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "text"]),
    default="text",
)
def letta_status(agent_id, environment, fmt):
    """Show the state of a Letta agent."""
    client = _get_client(environment)

    try:
        agent = client.agents.retrieve(agent_id)
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: failed to retrieve Letta agent: {e}", err=True)
        sys.exit(1)

    if fmt == "json":
        click.echo(json.dumps({"id": agent.id, "name": getattr(agent, "name", "")}, indent=2, default=str))
    else:
        click.echo(f"Agent: {agent.id}")
        click.echo(f"  Name: {getattr(agent, 'name', '')}")


@letta.command("delete")
@click.option("--agent-id", "-a", required=True, help="Agent ID")
@click.option(
    "--environment",
    default="cloud",
    type=click.Choice(["cloud", "local"]),
    help="Letta environment",
)
def letta_delete(agent_id, environment):
    """Delete a Letta agent."""
    client = _get_client(environment)

    try:
        client.agents.delete(agent_id)
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: failed to delete Letta agent: {e}", err=True)
        sys.exit(1)

    click.echo(f"Deleted Letta agent: {agent_id}")


@letta.command("remember")
@click.option("--agent-id", "-a", required=True, help="Agent ID")
@click.option("--text", "-t", required=True, help="Fact to remember")
@click.option("--label", "-l", default="human", help="Memory block label")
@click.option(
    "--environment",
    default="cloud",
    type=click.Choice(["cloud", "local"]),
    help="Letta environment",
)
def letta_remember(agent_id, text, label, environment):
    """Teach a Letta agent a durable fact.

    The text is stored as a memory block and can be recalled in later
    conversations. If a block with the same label exists, a new block is
    appended.
    """
    client = _get_client(environment)

    try:
        # Try V1 core memory append; fall back to sending a system message.
        try:
            client.agents.core_memory.append(
                agent_id=agent_id,
                label=label,
                value=text,
            )
        except AttributeError:
            client.agents.messages.create(
                agent_id=agent_id,
                input=f"Remember this under '{label}': {text}",
            )
    except Exception as e:  # noqa: BLE001
        click.echo(f"ERROR: failed to store memory: {e}", err=True)
        sys.exit(1)

    click.echo(f"Stored memory under label '{label}' for agent {agent_id}")
