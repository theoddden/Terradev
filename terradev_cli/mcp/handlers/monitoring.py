"""MCP tool handlers for the monitoring domain."""

import logging
import json, os

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


async def _handle_datadog_status(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.integrations.datadog_integration import (
            get_status_summary,
            METRIC_CATALOG,
            MONITOR_TEMPLATES,
        )

        # Load creds from ~/.terradev/credentials.json
        creds = _load_datadog_creds()
        status = get_status_summary(creds)
        status["available_metrics"] = len(METRIC_CATALOG)
        status["available_monitors"] = list(MONITOR_TEMPLATES.keys())
        output_text = "🐕 **Datadog Integration Status**\n\n"
        output_text += f"```json\n{json.dumps(status, indent=2)}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['datadog_status'] = _handle_datadog_status


async def _handle_datadog_push_metrics(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.integrations.datadog_integration import (
            push_cost_snapshot,
        )

        creds = _load_datadog_creds()
        result = await push_cost_snapshot(creds)
        output_text = "📤 **Datadog Metrics Push**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:3000]}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['datadog_push_metrics'] = _handle_datadog_push_metrics


async def _handle_datadog_send_event(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.integrations.datadog_integration import (
            send_event_async,
        )

        creds = _load_datadog_creds()
        result = await send_event_async(
            creds,
            title=arguments["title"],
            text=arguments["text"],
            alert_type=arguments.get("alert_type", "info"),
        )
        output_text = "📨 **Datadog Event Sent**\n\n"
        output_text += (
            f"```json\n{json.dumps(result, indent=2, default=str)}\n```"
        )
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['datadog_send_event'] = _handle_datadog_send_event


async def _handle_datadog_create_monitors(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.integrations.datadog_integration import (
            create_monitor,
            create_all_monitors,
        )

        creds = _load_datadog_creds()
        template = arguments.get("template")
        if template:
            result = await create_monitor(creds, template)
        else:
            result = await create_all_monitors(creds)
        output_text = "🔔 **Datadog Monitors**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:4000]}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['datadog_create_monitors'] = _handle_datadog_create_monitors


async def _handle_datadog_list_monitors(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.integrations.datadog_integration import (
            list_monitors,
        )

        creds = _load_datadog_creds()
        result = await list_monitors(creds)
        output_text = "📋 **Terradev Monitors in Datadog**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:4000]}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['datadog_list_monitors'] = _handle_datadog_list_monitors


async def _handle_datadog_create_dashboard(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.integrations.datadog_integration import (
            create_dashboard,
        )

        creds = _load_datadog_creds()
        result = await create_dashboard(
            creds, custom_title=arguments.get("title")
        )
        output_text = "📊 **Datadog Dashboard Created**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:3000]}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['datadog_create_dashboard'] = _handle_datadog_create_dashboard


async def _handle_datadog_list_dashboards(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.integrations.datadog_integration import (
            list_dashboards,
        )

        creds = _load_datadog_creds()
        result = await list_dashboards(creds)
        output_text = "📋 **Terradev Dashboards in Datadog**\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:3000]}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['datadog_list_dashboards'] = _handle_datadog_list_dashboards


async def _handle_datadog_query(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.integrations.datadog_integration import (
            query_metrics,
        )

        creds = _load_datadog_creds()
        result = await query_metrics(
            creds,
            query=arguments["query"],
            from_seconds=arguments.get("from_seconds", 3600),
        )
        output_text = f"🔍 **Datadog Query:** `{arguments['query']}`\n\n"
        output_text += f"```json\n{json.dumps(result, indent=2, default=str)[:5000]}\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['datadog_query'] = _handle_datadog_query


async def _handle_datadog_terraform_export(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.integrations.datadog_integration import (
            generate_full_terraform_module,
        )

        creds = _load_datadog_creds()
        out_dir = arguments.get("output_dir", "./datadog-terraform")
        files = generate_full_terraform_module(creds)
        os.makedirs(out_dir, exist_ok=True)
        written = []
        for fname, content in files.items():
            fpath = os.path.join(out_dir, fname)
            with open(fpath, "w") as f:
                f.write(content)
            written.append(fpath)
        output_text = f"🏗️ **Module Exported → `{out_dir}/`**\n\n"
        output_text += "**Files:**\n" + "\n".join(
            f"- `{w}`" for w in written
        )
        output_text += f"\n\n**Next steps:**\n```bash\ncd {out_dir}\ninit\nplan\napply\n```"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['datadog_terraform_export'] = _handle_datadog_terraform_export


async def _handle_datadog_metric_catalog(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.integrations.datadog_integration import (
            METRIC_CATALOG,
        )

        output_text = "📖 **Terradev Metric Catalog for Datadog**\n\n"
        output_text += f"**{len(METRIC_CATALOG)} metrics available:**\n\n"
        for name, meta in METRIC_CATALOG.items():
            tags = ", ".join(meta.get("tags", []))
            output_text += f"- `{name}` ({meta['type']}, {meta['unit']}) — {meta['desc']}"
            if tags:
                output_text += f" [tags: {tags}]"
            output_text += "\n"
        return CallToolResult(
            content=[TextContent(type="text", text=output_text)]
        )
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )
    return cmd_args

HANDLERS['datadog_metric_catalog'] = _handle_datadog_metric_catalog

async def _handle_kv_cache_efficiency(arguments, cmd_args, tool_name, execute_terradev_command):
    try:
        from terradev_cli.core.inference_router import InferenceRouter

        router = InferenceRouter()
        endpoint_id = arguments.get("endpoint_id")
        if endpoint_id:
            result = router.get_kv_cache_stats(endpoint_id)
        else:
            result = router.get_kv_cache_summary()
        output_text = "📦 **KV Cache Efficiency**\n\n"
        output_text += (
            "Avoided tokens = total prompt tokens - cached tokens. "
            "This is the non-gameable cache-efficiency metric.\n\n"
        )
        output_text += f"```json\n{json.dumps(result, indent=2)}\n```"
        return CallToolResult(content=[TextContent(type="text", text=output_text)])
    except Exception as e:  # noqa: BLE001
        return CallToolResult(
            content=[TextContent(type="text", text=f"❌ {e}")], isError=True
        )

HANDLERS['kv_cache_efficiency'] = _handle_kv_cache_efficiency
