#!/usr/bin/env python3
"""Generate reference single-simulation outputs from TVB hybrid simulator.

For each config in ref/configs.py, runs the TVB hybrid simulator and saves:
  - Final state as ref/single_sim/{name}_final_state.npy  [nvar, nnodes, nmodes]
  - Full trajectory as ref/single_sim/{name}_trajectory.npy  [n_steps, nvar, nnodes, nmodes]

Usage:
    ref/venv/bin/python ref/generate_single_sim.py
    ref/venv/bin/python ref/generate_single_sim.py --config g2do_small  # just one
"""

import os
import sys
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# Add repo root so we can import hyburn's io module for .npy compat
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from configs import SMALL_CONFIGS, FULL_CONFIGS


def make_integrator(kind, dt):
    """Create a TVB integrator matching hyburn's."""
    if kind == "heun":
        from tvb.simulator.integrators import HeunDeterministic
        return HeunDeterministic(dt=dt)
    else:
        from tvb.simulator.integrators import EulerDeterministic
        return EulerDeterministic(dt=dt)


def run_single_sim(name, cfg, output_dir):
    """Run a single subnetwork simulation and save results."""
    from tvb.simulator.hybrid.subnetwork import Subnetwork

    model = cfg["model_factory"]()
    scheme = make_integrator(cfg["integrator"], cfg["dt"])
    nnodes = cfg["nnodes"]

    # Build subnetwork
    sn = Subnetwork(name=name, model=model, scheme=scheme, nnodes=nnodes)

    # Set initial state
    if "initial_state_npy" in cfg:
        # Load from .npy — these are in hyburn's examples/ dir
        # hyburn saves as [nvar, nnodes] (matching TVB convention)
        npy_path = os.path.join(REPO_ROOT, cfg["initial_state_npy"])
        ic = np.load(npy_path).astype(np.float32)
        if ic.ndim == 2:
            # [nvar, nnodes] → [nvar, nnodes, nmodes=1]
            ic = ic[:, :, np.newaxis]
        sn.state = ic.copy()
    else:
        # Use zero initial state matching hyburn's Inline([0.0, ...])
        nmodes = 1
        ic = np.zeros((model.nvar, nnodes, nmodes), dtype=np.float32)
        # Match hyburn's typical init: V=0, W=0.5 for G2DO
        if model.nvar >= 2:
            ic[1, :, :] = 0.5
        sn.state = ic.copy()

    # Configure
    sn.configure(simulation_length=cfg["sim_length"])

    # Run
    n_steps = int(cfg["sim_length"] / cfg["dt"])
    ncvar = len(model.cvar)
    trajectory = np.zeros((n_steps + 1, model.nvar, nnodes, 1), dtype=np.float32)
    trajectory[0] = sn.state.copy()

    c = np.zeros((ncvar, nnodes, 1), dtype=np.float32)  # no external coupling

    for step in range(1, n_steps + 1):
        sn.state = sn.step(step, sn.state, c)
        trajectory[step] = sn.state.copy()

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    final_path = os.path.join(output_dir, f"{name}_final_state.npy")
    traj_path = os.path.join(output_dir, f"{name}_trajectory.npy")

    np.save(final_path, sn.state)
    np.save(traj_path, trajectory)

    print(f"  {name}: {n_steps} steps, final state shape {sn.state.shape}")
    print(f"    trajectory shape: {trajectory.shape}")
    print(f"    saved: {final_path}")
    print(f"    saved: {traj_path}")

    # Verify finite
    assert np.all(np.isfinite(sn.state)), f"{name}: final state contains NaN/Inf"
    assert np.all(np.isfinite(trajectory)), f"{name}: trajectory contains NaN/Inf"

    return sn.state, trajectory


def main():
    parser = argparse.ArgumentParser(description="Generate single-sim reference outputs")
    parser.add_argument("--config", default=None, help="Run only this config name")
    parser.add_argument("--full", action="store_true", help="Include full (74-node) configs")
    args = parser.parse_args()

    output_dir = os.path.join(REPO_ROOT, "ref", "single_sim")

    # Merge configs
    configs = dict(SMALL_CONFIGS)
    if args.full:
        configs.update(FULL_CONFIGS)

    if args.config:
        if args.config not in configs and args.config not in FULL_CONFIGS:
            print(f"ERROR: unknown config '{args.config}'")
            print(f"Available: {list(configs.keys()) + list(FULL_CONFIGS.keys())}")
            sys.exit(1)
        # May need to pull from FULL_CONFIGS
        if args.config in FULL_CONFIGS:
            configs = {args.config: FULL_CONFIGS[args.config]}
        else:
            configs = {args.config: configs[args.config]}

    print(f"Generating {len(configs)} reference single-sim outputs...")
    for name, cfg in configs.items():
        run_single_sim(name, cfg, output_dir)

    print("\nDone. Reference outputs saved to ref/single_sim/")


if __name__ == "__main__":
    main()
