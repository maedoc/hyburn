"""
Shared configuration for reference dataset generation.

Defines parameter sets that match hyburn's example TOML files exactly,
so that the TVB hybrid simulator produces comparable outputs.

Each config dict contains:
  - model: TVB model class name
  - model_kwargs: kwargs for the TVB model constructor (matching hyburn params)
  - nnodes: number of brain regions
  - dt: integration time step (ms)
  - sim_length: total simulation time (ms)
  - integrator: "heun" or "euler"
  - initial_state: numpy array or path to .npy
  - coupling: None or dict with coupling config (for single-subnet self-coupling)
  - monitors: list of monitor configs (for BOLD etc.)

IMPORTANT: Parameter ordering must match hyburn's PARAM_NAMES exactly.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Model configurations matching hyburn defaults
# ---------------------------------------------------------------------------

def g2do_default():
    """G2DO with hyburn default params: [tau=1, I=0, a=-2, b=-10, c=0, d=0.02, e=3, f=1, g=0, alpha=1, beta=1, gamma=1]"""
    from tvb.simulator.models import Generic2dOscillator
    return Generic2dOscillator(
        tau=np.array([1.0]),
        I=np.array([0.0]),
        a=np.array([-2.0]),
        b=np.array([-10.0]),
        c=np.array([0.0]),
        d=np.array([0.02]),
        e=np.array([3.0]),
        f=np.array([1.0]),
        g=np.array([0.0]),
        alpha=np.array([1.0]),
        beta=np.array([1.0]),
        gamma=np.array([1.0]),
    )


def mpr_default():
    """MPR with hyburn default params: [tau=1, Delta=1, eta=-5, J=15, I=0, cr=1, cv=0]"""
    from tvb.simulator.models import MontbrioPazoRoxin
    return MontbrioPazoRoxin(
        tau=np.array([1.0]),
        Delta=np.array([1.0]),
        eta=np.array([-5.0]),
        J=np.array([15.0]),
        I=np.array([0.0]),
        cr=np.array([1.0]),
        cv=np.array([0.0]),
    )


def kuramoto_default():
    """Kuramoto with hyburn default: [omega=1]"""
    from tvb.simulator.models import Kuramoto
    return Kuramoto(omega=np.array([1.0]))


def jansen_rit_default():
    """JansenRit with default params (13 params matching hyburn)."""
    from tvb.simulator.models import JansenRit
    return JansenRit()


def wilson_cowan_default():
    """WilsonCowan with default params (22 params matching hyburn)."""
    from tvb.simulator.models import WilsonCowan
    return WilsonCowan()


def rww_default():
    """ReducedWongWang with default params (8 params matching hyburn)."""
    from tvb.simulator.models import ReducedWongWang
    return ReducedWongWang()


# ---------------------------------------------------------------------------
# Simulation configurations
# ---------------------------------------------------------------------------

# Small test configs (fast, for CI-style validation)
SMALL_CONFIGS = {
    "g2do_small": {
        "model_factory": g2do_default,
        "nnodes": 2,
        "dt": 0.1,
        "sim_length": 10.0,
        "integrator": "heun",
    },
    "mpr_small": {
        "model_factory": mpr_default,
        "nnodes": 2,
        "dt": 0.1,
        "sim_length": 10.0,
        "integrator": "euler",
    },
    "kuramoto_small": {
        "model_factory": kuramoto_default,
        "nnodes": 2,
        "dt": 0.1,
        "sim_length": 10.0,
        "integrator": "euler",
    },
}

# Full configs matching hyburn examples (74 nodes, longer sim)
FULL_CONFIGS = {
    "g2do_74": {
        "model_factory": g2do_default,
        "nnodes": 74,
        "dt": 0.1,
        "sim_length": 1000.0,
        "integrator": "heun",
        "initial_state_npy": "examples/init_g2do_74.npy",
    },
}

# Sweep configs matching hyburn sweep.toml
SWEEP_CONFIGS = {
    "g2do_I_ext_sweep": {
        "model_factory": g2do_default,
        "nnodes": 74,
        "dt": 0.1,
        "sim_length": 1000.0,
        "integrator": "heun",
        "initial_state_npy": "examples/init_g2do_74.npy",
        "sweep_param": "I",  # TVB model attribute name
        "sweep_values": [
            -0.50, -0.48, -0.47, -0.45, -0.43, -0.42, -0.40, -0.38,
            -0.37, -0.35, -0.33, -0.32, -0.30, -0.28, -0.27, -0.25,
            -0.23, -0.22, -0.20, -0.18, -0.17, -0.15, -0.13, -0.12,
            -0.10, -0.08, -0.07, -0.05, -0.03, -0.02, 0.00, 0.02,
            0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.13, 0.15,
            0.17, 0.18, 0.20, 0.22, 0.23, 0.25, 0.27, 0.28,
            0.30, 0.32, 0.33, 0.35, 0.37, 0.38, 0.40, 0.42,
            0.43, 0.45, 0.47, 0.48, 0.50, 0.52, 0.53, 0.55,
            0.57, 0.58, 0.60, 0.62, 0.63, 0.65, 0.67, 0.68,
            0.70, 0.72, 0.73, 0.75, 0.77, 0.78, 0.80, 0.82,
            0.83, 0.85, 0.87, 0.88, 0.90, 0.92, 0.93, 0.95,
            0.97, 0.98, 1.00, 1.02, 1.03, 1.05, 1.07, 1.08,
            1.10, 1.12, 1.13, 1.15, 1.17, 1.18, 1.20, 1.22,
            1.23, 1.25, 1.27, 1.28, 1.30, 1.32, 1.33, 1.35,
            1.37, 1.38, 1.40, 1.42, 1.43, 1.45, 1.47, 1.48,
            1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50,
        ],
    },
}

# BOLD monitor config matching hyburn's Balloon-Windkessel model
BOLD_CONFIG = {
    "g2do_bold": {
        "model_factory": g2do_default,
        "nnodes": 74,
        "dt": 0.1,
        "sim_length": 20000.0,  # 20 s in ms for meaningful BOLD
        "integrator": "heun",
        "initial_state_npy": "examples/init_g2do_74.npy",
        "bold_tr": 2.0,        # seconds
        "bold_period": 10,     # neural steps between BW updates
    },
}
