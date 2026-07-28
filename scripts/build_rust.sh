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

# Detect whether we are running inside a Python virtualenv/conda env.
# Maturin's `develop` command refuses to install into the system interpreter,
# so in CI or other system-Python contexts we build wheels and `pip install` them.
if [[ -n "${VIRTUAL_ENV:-}" ]] || [[ -n "${CONDA_PREFIX:-}" ]]; then
    USE_VENV=1
else
    _in_venv=$(python -c "import sys; print(sys.prefix != sys.base_prefix)" 2>/dev/null || echo "False")
    if [[ "$_in_venv" == "True" ]]; then
        USE_VENV=1
    else
        USE_VENV=0
    fi
fi

if [[ "$USE_VENV" -eq 1 ]]; then
    echo "Using maturin develop for active virtualenv"
else
    WHEEL_DIR=$(mktemp -d)
    # shellcheck disable=SC2064
    trap "rm -rf '$WHEEL_DIR'" EXIT
    echo "No virtualenv detected — maturin will build wheels in $WHEEL_DIR and pip install them"
fi

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
    if [[ "$USE_VENV" -eq 1 ]]; then
        if [[ "$MODE" == "debug" ]]; then
            maturin develop --manifest-path "$crate_dir/Cargo.toml"
        else
            maturin develop --release --manifest-path "$crate_dir/Cargo.toml"
        fi
    else
        if [[ "$MODE" == "debug" ]]; then
            maturin build --manifest-path "$crate_dir/Cargo.toml" --out "$WHEEL_DIR"
        else
            maturin build --release --manifest-path "$crate_dir/Cargo.toml" --out "$WHEEL_DIR"
        fi
    fi
    echo "    ✓ $crate built"
done

if [[ "$USE_VENV" -eq 0 ]]; then
    echo ""
    echo "--- Installing built wheels into system Python ---"
    shopt -s nullglob
    wheels=("$WHEEL_DIR"/*.whl)
    shopt -u nullglob
    if [[ ${#wheels[@]} -eq 0 ]]; then
        echo "ERROR: no wheels were produced" >&2
        exit 1
    fi
    pip install --no-deps --force-reinstall "${wheels[@]}"
fi

echo ""
echo "=== All Rust extensions built and installed ==="
echo "Verify with: python -c \"import terradev_dag_executor, terradev_credential_vault, terradev_gpu_topology, terradev_semantic_router; print('OK')\""
