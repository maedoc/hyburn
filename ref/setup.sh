#!/usr/bin/env bash
# Set up Python virtual environment with tvb-root hybrid-model branch.
#
# Usage:
#   ./ref/setup.sh          # create venv + install tvb-root
#   ./ref/setup.sh --check  # verify installation
#
# The venv is created at ref/.venv using uv.
# tvb-root is installed from the local checkout at ~/src/tvb-root
# (hybrid-model branch) in editable mode.
#
# Requirements: uv, git, ~/src/tvb-root on hybrid-model branch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REF_DIR="$SCRIPT_DIR"
VENV_DIR="$REF_DIR/.venv"
TVB_ROOT="${TVB_ROOT:-$HOME/src/tvb-root}"

check_only=false
if [[ "${1:-}" == "--check" ]]; then
    check_only=true
fi

# --- Verify tvb-root exists ---
if [[ ! -d "$TVB_ROOT/tvb_library" ]]; then
    echo "ERROR: tvb-root not found at $TVB_ROOT"
    echo "Clone with: git clone -b hybrid-model https://github.com/the-virtual-brain/tvb-root.git $TVB_ROOT"
    exit 1
fi

branch=$(cd "$TVB_ROOT" && git branch --show-current)
if [[ "$branch" != "hybrid-model" ]]; then
    echo "WARNING: tvb-root is on branch '$branch', expected 'hybrid-model'"
fi

echo "tvb-root: $TVB_ROOT (branch: $branch)"

# --- Create venv with uv ---
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating venv at $VENV_DIR ..."
    uv venv "$VENV_DIR" --python 3.11
else
    echo "venv already exists at $VENV_DIR"
fi

# --- Install tvb-root + deps ---
if [[ "$check_only" == false ]]; then
    echo "Installing tvb-root (hybrid-model) in editable mode ..."
    uv pip install -e "$TVB_ROOT/tvb_library" --python "$VENV_DIR/bin/python"

    echo "Installing additional deps ..."
    uv pip install scipy --python "$VENV_DIR/bin/python"
fi

# --- Verify ---
echo ""
echo "Verification:"
PYTHON="$VENV_DIR/bin/python"

$PYTHON -c "
from tvb.simulator.hybrid.simulator import Simulator
from tvb.simulator.hybrid.network import NetworkSet
from tvb.simulator.hybrid.subnetwork import Subnetwork
from tvb.simulator.models import Generic2dOscillator, MontbrioPazoRoxin, JansenRit, WilsonCowan, Kuramoto, ReducedWongWang
from tvb.simulator.integrators import HeunDeterministic, EulerDeterministic
from tvb.simulator.coupling import Linear, Sigmoidal, Difference, Kuramoto as KuramotoCoupling
import numpy as np
print('  Simulator     : OK')
print('  NetworkSet    : OK')
print('  Subnetwork    : OK')
print('  Models        : G2DO, MPR, JR, WC, Kuramoto')
print('  Integrators    : Heun, Euler')
print('  Coupling       : Linear, Sigmoidal, Difference, Kuramoto')
print('  NumPy          :', np.__version__)
"

echo ""
echo "Setup complete. Run generation scripts with:"
echo "  $VENV_DIR/bin python ref/generate_single_sim.py"
