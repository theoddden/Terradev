"""MCP tool handlers for the governance domain."""

import logging
import asyncio, json, os

try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from mcp.types import CallToolResult, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    CallToolResult = None
    TextContent = None

from .. import executor

logger = logging.getLogger(__name__)

_load_datadog_creds = executor._load_datadog_creds

HANDLERS = {}


async def _handle_governance_request_consent(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.data_governance import DataGovernance

        gov = DataGovernance()
        result = await gov.request_consent(
            user_id=arguments["user_id"],
            consent_type=arguments["consent_type"],
            dataset_name=arguments["dataset_name"],
            purpose=arguments["purpose"],
            source_location=arguments.get("source_location"),
            target_location=arguments.get("target_location"),
        )
        output_text = "📋 **Consent Request Created**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:2000]}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except ImportError:
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Terradev CLI not found.")
            ],
            isError=True,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['governance_request_consent'] = _handle_governance_request_consent


async def _handle_governance_record_consent(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.data_governance import DataGovernance

        gov = DataGovernance()
        result = await gov.record_consent(
            request_id=arguments["request_id"],
            user_id=arguments["user_id"],
            granted=arguments["granted"],
            conditions=arguments.get("conditions"),
        )
        icon = "✅" if arguments["granted"] else "❌"
        output_text = f"{icon} **Consent {'Granted' if arguments['granted'] else 'Denied'}**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:2000]}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except ImportError:
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Terradev CLI not found.")
            ],
            isError=True,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['governance_record_consent'] = _handle_governance_record_consent


async def _handle_governance_evaluate_opa(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.data_governance import DataGovernance

        gov = DataGovernance()
        result = await gov.evaluate_opa(
            user_id=arguments["user_id"],
            dataset_name=arguments["dataset_name"],
            action=arguments["action"],
            target_location=arguments.get("target_location"),
        )
        allowed = result.get(
            "allowed", result.get("result", {}).get("allow", False)
        )
        icon = "✅" if allowed else "🚫"
        output_text = f"{icon} **OPA Policy Evaluation**\n\n"
        output_text += f"**Action:** {arguments['action']} on {arguments['dataset_name']}\n"
        output_text += (
            f"**Decision:** {'ALLOWED' if allowed else 'DENIED'}\n\n"
        )
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:2000]}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except ImportError:
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Terradev CLI not found.")
            ],
            isError=True,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['governance_evaluate_opa'] = _handle_governance_evaluate_opa


async def _handle_governance_move_data(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.data_governance import DataGovernance

        gov = DataGovernance()
        result = await gov.move_data(
            user_id=arguments["user_id"],
            consent_request_id=arguments["consent_request_id"],
            dataset_name=arguments["dataset_name"],
            source_location=arguments["source_location"],
            target_location=arguments["target_location"],
        )
        output_text = f"📦 **Data Move — {arguments['dataset_name']}**\n\n"
        output_text += f"**From:** {arguments['source_location']}\n"
        output_text += f"**To:** {arguments['target_location']}\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:2000]}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except ImportError:
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Terradev CLI not found.")
            ],
            isError=True,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['governance_move_data'] = _handle_governance_move_data


async def _handle_governance_movement_history(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.data_governance import DataGovernance

        gov = DataGovernance()
        result = await gov.movement_history(
            user_id=arguments.get("user_id"),
            dataset_name=arguments.get("dataset_name"),
            limit=arguments.get("limit", 50),
        )
        output_text = "📜 **Data Movement History**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:4000]}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except ImportError:
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Terradev CLI not found.")
            ],
            isError=True,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['governance_movement_history'] = _handle_governance_movement_history


async def _handle_governance_compliance_report(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.data_governance import DataGovernance

        gov = DataGovernance()
        # Parallel: gather consent stats + movement history + policy evaluations
        report = await gov.compliance_report(
            start_date=arguments["start_date"],
            end_date=arguments["end_date"],
        )
        output_text = "📋 **Compliance Report**\n\n"
        output_text += f"**Period:** {arguments['start_date']} → {arguments['end_date']}\n\n"
        output_text += f"```json\n{json.dumps(report, indent=2, default=str)[:5000]}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except ImportError:
        return CallToolResult(
            content=[
                TextContent(type="text", text="❌ Terradev CLI not found.")
            ],
            isError=True,
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['governance_compliance_report'] = _handle_governance_compliance_report