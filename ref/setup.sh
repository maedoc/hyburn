#!/usr/bin/env bash
# Set up Python virtual environment with tvb-root hybrid-numba branch.
#
# Usage:
#   ./ref/setup.sh          # create venv + install tvb-root
#   ./ref/setup.sh --check  # verify installation
#
# The venv is created at ref/venv using uv.
# tvb-root is cloned into ref/tvb-root (hybrid-numba branch) if not present,
# then installed in editable mode along with tvb-data.
#
# Requirements: uv, git

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REF_DIR="$SCRIPT_DIR"
VENV_DIR="$REF_DIR/venv"
TVB_ROOT="${TVB_ROOT:-$REF_DIR/tvb-root}"

check_only=false
if [[ "${1:-}" == "--check" ]]; then
    check_only=true
fi

# --- Clone tvb-root if not present ---
if [[ ! -d "$TVB_ROOT/tvb_library" ]]; then
    if [[ "$check_only" == true ]]; then
        echo "ERROR: tvb-root not found at $TVB_ROOT"
        echo "Run ./ref/setup.sh (without --check) to clone and install."
        exit 1
    fi
    echo "Cloning tvb-root (hybrid-numba branch) into $TVB_ROOT ..."
    git clone -b hybrid-numba https://github.com/the-virtual-brain/tvb-root.git "$TVB_ROOT"
fi

# --- Verify tvb-root branch ---
branch=$(cd "$TVB_ROOT" && git branch --show-current)
if [[ "$branch" != "hybrid-numba" ]]; then
    echo "WARNING: tvb-root is on branch '$branch', expected 'hybrid-numba'"
    echo "Switch with: cd $TVB_ROOT && git checkout hybrid-numba"
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
    echo "Installing tvb-root (hybrid-numba) in editable mode ..."
    uv pip install -e "$TVB_ROOT/tvb_library" --python "$VENV_DIR/bin/python"

    echo "Installing additional deps ..."
    uv pip install scipy tvb-data --python "$VENV_DIR/bin/python"
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
import tvb_data
import numpy as np
print('  Simulator     : OK')
print('  NetworkSet    : OK')
print('  Subnetwork    : OK')
print('  Models        : G2DO, MPR, JR, WC, Kuramoto')
print('  Integrators    : Heun, Euler')
print('  Coupling       : Linear, Sigmoidal, Difference, Kuramoto')
print('  tvb-data       :', tvb_data.__path__[0])
print('  NumPy          :', np.__version__)
"

echo ""
echo "Setup complete. Run generation scripts with:"
echo "  $VENV_DIR/bin/python ref/generate_single_sim.py"
echo "  $VENV_DIR/bin/python ref/generate_single_sim.py --full"
echo "  $VENV_DIR/bin/python ref/generate_sweep.py"
echo "  $VENV_DIR/bin/python ref/generate_bold.py"
echo "  $VENV_DIR/bin/python ref/generate_features.py"