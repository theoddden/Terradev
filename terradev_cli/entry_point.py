#!/usr/bin/env python3
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2026 Terradev
#
"""
Terradev CLI Entry Point (Open Source - Apache 2.0)
Simple entry point for open source version - no telemetry enforcement
"""

import sys


def main():
    """Main entry point"""
    try:
        from .cli import cli

        cli()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!", file=sys.stderr)
        sys.exit(0)
    except Exception as e:  # noqa: BLE001
        print(f"❌ CLI Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
