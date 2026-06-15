#!/usr/bin/env python3
"""Tests for utils/formatters.py"""

import pytest
from datetime import datetime
from terradev_cli.utils.formatters import (
    format_table,
    format_json,
    format_success,
    format_error,
    format_warning,
    format_info,
    format_price,
    format_duration,
    format_datetime,
    format_bytes,
    format_percentage,
    format_progress_bar,
    format_list,
    format_key_value,
    format_status,
    format_provider,
    format_gpu_type,
    format_cost_savings,
    format_optimization_score,
    print_table,
    print_json,
    print_success,
    print_error,
    print_warning,
    print_info,
)


class TestFormatTable:
    """Test table formatting"""

    def test_empty_table(self):
        """Empty table returns message"""
        result = format_table(["Name", "Age"], [])
        assert result == "No data available"

    def test_single_row(self):
        """Single row table"""
        result = format_table(["Name", "Age"], [["Alice", "30"]])
        assert "Name" in result
        assert "Alice" in result
        assert "30" in result

    def test_multiple_rows(self):
        """Multiple rows table"""
        result = format_table(["Name", "Age"], [["Alice", "30"], ["Bob", "25"]])
        assert "Alice" in result
        assert "Bob" in result
        assert "30" in result
        assert "25" in result

    def test_column_width_calculation(self):
        """Column widths adjust to content"""
        result = format_table(["Name"], [["VeryLongName"]])
        assert "VeryLongName" in result


class TestFormatJson:
    """Test JSON formatting"""

    def test_dict_formatting(self):
        """Dictionary formatting"""
        result = format_json({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_list_formatting(self):
        """List formatting"""
        result = format_json([1, 2, 3])
        assert "1" in result
        assert "2" in result
        assert "3" in result

    def test_datetime_serialization(self):
        """Datetime objects serialize to string"""
        dt = datetime(2024, 1, 1, 12, 0, 0)
        result = format_json({"time": dt})
        assert "2024" in result


class TestMessageFormatters:
    """Test message formatting functions"""

    def test_format_success(self):
        """Success message formatting"""
        result = format_success("Operation completed")
        assert "✅" in result
        assert "Operation completed" in result

    def test_format_error(self):
        """Error message formatting"""
        result = format_error("Operation failed")
        assert "❌" in result
        assert "Operation failed" in result

    def test_format_warning(self):
        """Warning message formatting"""
        result = format_warning("Caution required")
        assert "⚠️" in result
        assert "Caution required" in result

    def test_format_info(self):
        """Info message formatting"""
        result = format_info("Information")
        assert "ℹ️" in result
        assert "Information" in result


class TestFormatPrice:
    """Test price formatting"""

    def test_price_formatting(self):
        """Price with dollar sign"""
        result = format_price(12.3456)
        assert result == "$12.3456"

    def test_zero_price(self):
        """Zero price"""
        result = format_price(0.0)
        assert result == "$0.0000"


class TestFormatDuration:
    """Test duration formatting"""

    def test_seconds_formatting(self):
        """Duration in seconds"""
        result = format_duration(30.5)
        assert result == "30.5s"

    def test_minutes_formatting(self):
        """Duration in minutes"""
        result = format_duration(120.0)
        assert result == "2.0m"

    def test_hours_formatting(self):
        """Duration in hours"""
        result = format_duration(7200.0)
        assert result == "2.0h"


class TestFormatDatetime:
    """Test datetime formatting"""

    def test_datetime_formatting(self):
        """Datetime string format"""
        dt = datetime(2024, 1, 15, 14, 30, 45)
        result = format_datetime(dt)
        assert result == "2024-01-15 14:30:45"


class TestFormatBytes:
    """Test bytes formatting"""

    def test_bytes_formatting(self):
        """Bytes in B"""
        result = format_bytes(512)
        assert "B" in result

    def test_kilobytes_formatting(self):
        """Bytes in KB"""
        result = format_bytes(2048)
        assert "KB" in result

    def test_megabytes_formatting(self):
        """Bytes in MB"""
        result = format_bytes(2097152)
        assert "MB" in result

    def test_gigabytes_formatting(self):
        """Bytes in GB"""
        result = format_bytes(2147483648)
        assert "GB" in result

    def test_petabytes_formatting(self):
        """Bytes in PB"""
        result = format_bytes(1125899906842624)
        assert "PB" in result


class TestFormatPercentage:
    """Test percentage formatting"""

    def test_percentage_formatting(self):
        """Percentage with decimal"""
        result = format_percentage(75.5)
        assert result == "75.5%"

    def test_zero_percentage(self):
        """Zero percentage"""
        result = format_percentage(0.0)
        assert result == "0.0%"


class TestFormatProgressBar:
    """Test progress bar formatting"""

    def test_zero_progress(self):
        """Zero progress bar"""
        result = format_progress_bar(0, 100)
        assert "[" in result
        assert "]" in result
        assert "0.0%" in result

    def test_half_progress(self):
        """Half progress bar"""
        result = format_progress_bar(50, 100)
        assert "50.0%" in result
        assert "50/100" in result

    def test_full_progress(self):
        """Full progress bar"""
        result = format_progress_bar(100, 100)
        assert "100.0%" in result

    def test_zero_total(self):
        """Zero total progress bar"""
        result = format_progress_bar(0, 0)
        assert "[" in result
        assert "]" in result


class TestFormatList:
    """Test list formatting"""

    def test_list_formatting(self):
        """List with default bullet"""
        result = format_list(["item1", "item2"])
        assert "• item1" in result
        assert "• item2" in result

    def test_custom_bullet(self):
        """List with custom bullet"""
        result = format_list(["item1", "item2"], bullet="-")
        assert "- item1" in result
        assert "- item2" in result


class TestFormatKeyValue:
    """Test key-value formatting"""

    def test_simple_pairs(self):
        """Simple key-value pairs"""
        result = format_key_value({"name": "Alice", "age": "30"})
        assert "name: Alice" in result
        assert "age: 30" in result

    def test_nested_dict(self):
        """Nested dictionary"""
        result = format_key_value({"person": {"name": "Alice"}})
        assert "person:" in result
        assert "name: Alice" in result

    def test_list_values(self):
        """List values"""
        result = format_key_value({"items": ["a", "b"]})
        assert "items:" in result
        assert "• a" in result
        assert "• b" in result

    def test_indentation(self):
        """Indented key-value pairs"""
        result = format_key_value({"key": "value"}, indent=1)
        assert "  key: value" in result


class TestFormatStatus:
    """Test status formatting"""

    def test_running_status(self):
        """Running status"""
        result = format_status("running")
        assert "🟢" in result
        assert "running" in result

    def test_stopped_status(self):
        """Stopped status"""
        result = format_status("stopped")
        assert "🔴" in result
        assert "stopped" in result

    def test_pending_status(self):
        """Pending status"""
        result = format_status("pending")
        assert "🟡" in result
        assert "pending" in result

    def test_unknown_status(self):
        """Unknown status"""
        result = format_status("unknown")
        assert "⚪" in result
        assert "unknown" in result


class TestFormatProvider:
    """Test provider formatting"""

    def test_aws_provider(self):
        """AWS provider"""
        result = format_provider("aws")
        assert "🟧" in result
        assert "AWS" in result

    def test_gcp_provider(self):
        """GCP provider"""
        result = format_provider("gcp")
        assert "🟦" in result
        assert "GCP" in result

    def test_runpod_provider(self):
        """RunPod provider"""
        result = format_provider("runpod")
        assert "🚀" in result
        assert "RUNPOD" in result

    def test_unknown_provider(self):
        """Unknown provider"""
        result = format_provider("unknown")
        assert "☁️" in result
        assert "UNKNOWN" in result


class TestFormatGpuType:
    """Test GPU type formatting"""

    def test_a100_gpu(self):
        """A100 GPU"""
        result = format_gpu_type("A100")
        assert "🔥" in result
        assert "A100" in result

    def test_h100_gpu(self):
        """H100 GPU"""
        result = format_gpu_type("H100")
        assert "🚀" in result
        assert "H100" in result

    def test_unknown_gpu(self):
        """Unknown GPU"""
        result = format_gpu_type("UNKNOWN")
        assert "🔧" in result
        assert "UNKNOWN" in result


class TestFormatCostSavings:
    """Test cost savings formatting"""

    def test_positive_savings(self):
        """Positive cost savings"""
        result = format_cost_savings(25.5, 100.0)
        assert "💰" in result
        assert "25.5%" in result
        assert "$100.00" in result

    def test_no_savings(self):
        """No cost savings"""
        result = format_cost_savings(0.0, 0.0)
        assert "💸" in result
        assert "No savings available" in result


class TestFormatOptimizationScore:
    """Test optimization score formatting"""

    def test_high_score(self):
        """High optimization score"""
        result = format_optimization_score(0.85)
        assert "🟢" in result
        assert "0.85" in result

    def test_medium_score(self):
        """Medium optimization score"""
        result = format_optimization_score(0.65)
        assert "🟡" in result
        assert "0.65" in result

    def test_low_score(self):
        """Low optimization score"""
        result = format_optimization_score(0.45)
        assert "🔴" in result
        assert "0.45" in result


class TestPrintFunctions:
    """Test print functions"""

    def test_print_table(self, capsys):
        """Print table to stdout"""
        print_table(["Name"], [["Alice"]])
        captured = capsys.readouterr()
        assert "Name" in captured.out
        assert "Alice" in captured.out

    def test_print_json(self, capsys):
        """Print JSON to stdout"""
        print_json({"key": "value"})
        captured = capsys.readouterr()
        assert '"key"' in captured.out

    def test_print_success(self, capsys):
        """Print success to stdout"""
        print_success("Done")
        captured = capsys.readouterr()
        assert "✅" in captured.out
        assert "Done" in captured.out

    def test_print_error(self, capsys):
        """Print error to stderr"""
        print_error("Failed")
        captured = capsys.readouterr()
        assert "❌" in captured.err
        assert "Failed" in captured.err

    def test_print_warning(self, capsys):
        """Print warning to stderr"""
        print_warning("Warning")
        captured = capsys.readouterr()
        assert "⚠️" in captured.err
        assert "Warning" in captured.err

    def test_print_info(self, capsys):
        """Print info to stdout"""
        print_info("Info")
        captured = capsys.readouterr()
        assert "ℹ️" in captured.out
        assert "Info" in captured.out
