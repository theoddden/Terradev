#!/usr/bin/env python3
"""
Formatters - Output formatting utilities for CLI
"""

import json
from typing import List, Any, Dict
from datetime import datetime
import sys


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format data as a table"""
    if not rows:
        return "No data available"

    # Calculate column widths
    col_widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Build table
    lines = []

    # Header row
    header_row = " | ".join(
        header.ljust(col_widths[i]) for i, header in enumerate(headers)
    )
    lines.append(header_row)

    # Separator
    separator = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    lines.append(separator)

    # Data rows
    for row in rows:
        data_row = " | ".join(
            str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
        )
        lines.append(data_row)

    return "\n".join(lines)


def format_json(data: Any) -> str:
    """Format data as JSON"""
    return json.dumps(data, indent=2, default=str)


def format_success(message: str) -> str:
    """Format success message"""
    return f"✅ {message}"


def format_error(message: str) -> str:
    """Format error message"""
    return f"❌ {message}"


def format_warning(message: str) -> str:
    """Format warning message"""
    return f"⚠️ {message}"


def format_info(message: str) -> str:
    """Format info message"""
    return f"ℹ️ {message}"


def format_price(price: float) -> str:
    """Format price with currency symbol"""
    return f"${price:.4f}"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_datetime(dt: datetime) -> str:
    """Format datetime"""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_bytes(bytes_count: int) -> str:
    """Format bytes in human readable format"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} PB"


def format_percentage(value: float) -> str:
    """Format percentage"""
    return f"{value:.1f}%"


def format_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Format progress bar"""
    if total == 0:
        return "[" + "=" * width + "]"

    filled = int((current / total) * width)
    bar = "=" * filled + "-" * (width - filled)
    percentage = (current / total) * 100

    return f"[{bar}] {percentage:.1f}% ({current}/{total})"


def format_list(items: List[str], bullet: str = "•") -> str:
    """Format list with bullets"""
    return "\n".join(f"{bullet} {item}" for item in items)


def format_key_value(pairs: Dict[str, Any], indent: int = 0) -> str:
    """Format key-value pairs"""
    lines = []
    indent_str = "  " * indent

    for key, value in pairs.items():
        if isinstance(value, dict):
            lines.append(f"{indent_str}{key}:")
            lines.append(format_key_value(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{indent_str}{key}:")
            for item in value:
                lines.append(f"{indent_str}  • {item}")
        else:
            lines.append(f"{indent_str}{key}: {value}")

    return "\n".join(lines)


def format_status(status: str) -> str:
    """Format status with color coding"""
    status_colors = {
        "running": "🟢",
        "stopped": "🔴",
        "pending": "🟡",
        "terminating": "🟠",
        "failed": "❌",
        "completed": "✅",
    }

    icon = status_colors.get(status.lower(), "⚪")
    return f"{icon} {status}"


def format_provider(provider: str) -> str:
    """Format provider name"""
    provider_icons = {
        "aws": "🟧",
        "gcp": "🟦",
        "azure": "🟦",
        "runpod": "🚀",
        "vastai": "🌐",
        "lambda_labs": "⚡",
        "coreweave": "🔷",
        "tensordock": "🔶",
    }

    icon = provider_icons.get(provider.lower(), "☁️")
    return f"{icon} {provider.upper()}"


def format_gpu_type(gpu_type: str) -> str:
    """Format GPU type"""
    gpu_icons = {
        "A100": "🔥",
        "V100": "⚡",
        "RTX4090": "🎮",
        "RTX3090": "🎮",
        "H100": "🚀",
        "T4": "🔷",
    }

    icon = gpu_icons.get(gpu_type.upper(), "🔧")
    return f"{icon} {gpu_type}"


def format_cost_savings(savings_percent: float, savings_amount: float) -> str:
    """Format cost savings"""
    if savings_percent > 0:
        return f"💰 Save {savings_percent:.1f}% (${savings_amount:.2f})"
    else:
        return f"💸 No savings available"


def format_optimization_score(score: float) -> str:
    """Format optimization score"""
    if score >= 0.8:
        return f"🟢 {score:.2f}"
    elif score >= 0.6:
        return f"🟡 {score:.2f}"
    else:
        return f"🔴 {score:.2f}"


def print_table(headers: List[str], rows: List[List[str]]) -> None:
    """Print table to stdout"""
    print(format_table(headers, rows))


def print_json(data: Any) -> None:
    """Print JSON to stdout"""
    print(format_json(data))


def print_success(message: str) -> None:
    """Print success message to stdout"""
    print(format_success(message))


def print_error(message: str) -> None:
    """Print error message to stderr"""
    print(format_error(message), file=sys.stderr)


def print_warning(message: str) -> None:
    """Print warning message to stderr"""
    print(format_warning(message), file=sys.stderr)


def print_info(message: str) -> None:
    """Print info message to stdout"""
    print(format_info(message))
