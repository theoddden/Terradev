#!/usr/bin/env bash
# build_rust.sh — Build all Terradev Rust extension modules and install them
# into the current Python environment.
#
# Usage:
#   ./scripts/build_rust.sh            # release build (default)
#   ./scripts/build_rust.sh --dev      # debug build (faster compile, slower runtime)
#
# Prerequisites:
#   rustup (https://rustup.rs)
#   maturin  — installed automatically if missing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUST_DIR="$REPO_ROOT/rust"

MODE="release"
if [[ "${1:-}" == "--dev" ]]; then
    MODE="debug"
fi

echo "=== Terradev Rust build ($MODE mode) ==="

# Ensure maturin is available
if ! command -v maturin &>/dev/null; then
    echo "maturin not found — installing..."
    pip install "maturin>=1.4,<2.0"
fi

# Key crates that Python imports at runtime
CRATES=(
    "terradev-dag-executor"
    "terradev-credential-vault"
    "terradev-gpu-topology"
    "terradev-semantic-router"
)

for crate in "${CRATES[@]}"; do
    crate_dir="$RUST_DIR/$crate"
    if [[ ! -d "$crate_dir" ]]; then
        echo "  [SKIP] $crate — directory not found"
        continue
    fi

    echo ""
    echo "--- Building $crate ---"
    if [[ "$MODE" == "debug" ]]; then
        maturin develop --manifest-path "$crate_dir/Cargo.toml"
    else
        maturin develop --release --manifest-path "$crate_dir/Cargo.toml"
    fi
    echo "    ✓ $crate installed"
done

echo ""
echo "=== All Rust extensions built and installed ==="
echo "Verify with: python -c \"import terradev_dag_executor, terradev_credential_vault, terradev_gpu_topology, terradev_semantic_router; print('OK')\""
