#!/usr/bin/env python3
"""
Terradev CLI Module Entry Point - AMAZING UX VERSION
"""

import sys


def main():
    """Main entry point with amazing UX"""
    try:
        # Try the simple CLI first (most reliable)
        try:
            from .cli_optimization_simple import optimize
            optimize()
        except ImportError:
            # Try the fixed CLI
            try:
                from .cli_optimization_fixed import optimize
                optimize()
            except ImportError:
                # Fallback to original CLI
                from .cli import cli
                cli()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
