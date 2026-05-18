#!/usr/bin/env python3
"""Generate reference sweep outputs from TVB hybrid simulator.

For each sweep config in ref/configs.py, runs parameter sweeps matching
hyburn's sweep.toml and saves:
  - Final states as ref/sweep/{name}_final_states.npy  [n_sweep, nvar, nnodes, nmodes]
  - Mean trajectories as ref/sweep/{name}_mean_traj.npy  [n_sweep, n_steps+1, nvar]

Usage:
    ref/venv/bin/python ref/generate_sweep.py
    ref/venv/bin/python ref/generate_sweep.py --config g2do_I_ext_sweep
"""

import os
import sys
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from configs import SWEEP_CONFIGS


def make_integrator(kind, dt):
    """Create a TVB integrator matching hyburn's."""
    if kind == "heun":
        from tvb.simulator.integrators import HeunDeterministic
        return HeunDeterministic(dt=dt)
    else:
        from tvb.simulator.integrators import EulerDeterministic
        return EulerDeterministic(dt=dt)


def run_sweep(name, cfg, output_dir):
    """Run a parameter sweep and save results."""
    from tvb.simulator.hybrid.subnetwork import Subnetwork

    nnodes = cfg["nnodes"]
    n_steps = int(cfg["sim_length"] / cfg["dt"])
    sweep_values = cfg["sweep_values"]
    n_sweep = len(sweep_values)
    sweep_param = cfg["sweep_param"]

    # Build base model to get nvar
    base_model = cfg["model_factory"]()
    nvar = base_model.nvar
    ncvar = len(base_model.cvar)

    # Prepare output arrays
    final_states = np.zeros((n_sweep, nvar, nnodes, 1), dtype=np.float32)
    # Store mean over nodes for each variable at each step
    mean_traj = np.zeros((n_sweep, n_steps + 1, nvar), dtype=np.float32)

    print(f"  {name}: {n_sweep} sweep points, {n_steps} steps each")

    for si, sweep_val in enumerate(sweep_values):
        # Create a fresh model with the swept parameter
        model = cfg["model_factory"]()
        setattr(model, sweep_param, np.array([sweep_val]))
        scheme = make_integrator(cfg["integrator"], cfg["dt"])

        # Build subnetwork
        sn = Subnetwork(name=name, model=model, scheme=scheme, nnodes=nnodes)

        # Set initial state
        if "initial_state_npy" in cfg:
            npy_path = os.path.join(REPO_ROOT, cfg["initial_state_npy"])
            ic = np.load(npy_path).astype(np.float32)
            if ic.ndim == 2:
                ic = ic[:, :, np.newaxis]
            sn.state = ic.copy()
        else:
            ic = np.zeros((nvar, nnodes, 1), dtype=np.float32)
            if nvar >= 2:
                ic[1, :, :] = 0.5
            sn.state = ic.copy()

        sn.configure(simulation_length=cfg["sim_length"])

        # Record mean at step 0
        mean_traj[si, 0, :] = sn.state[:, :, 0].mean(axis=1)

        # Run simulation
        c = np.zeros((ncvar, nnodes, 1), dtype=np.float32)
        for step in range(1, n_steps + 1):
            sn.state = sn.step(step, sn.state, c)
            mean_traj[si, step, :] = sn.state[:, :, 0].mean(axis=1)

        final_states[si] = sn.state.copy()

        if (si + 1) % 16 == 0 or si == n_sweep - 1:
            print(f"    sweep point {si + 1}/{n_sweep} "
                  f"({sweep_param}={sweep_val:.2f})")

    # Save
    os.makedirs(output_dir, exist_ok=True)

    final_path = os.path.join(output_dir, f"{name}_final_states.npy")
    traj_path = os.path.join(output_dir, f"{name}_mean_traj.npy")

    np.save(final_path, final_states)
    np.save(traj_path, mean_traj)

    print(f"    final_states shape: {final_states.shape}")
    print(f"    mean_traj shape: {mean_traj.shape}")
    print(f"    saved: {final_path}")
    print(f"    saved: {traj_path}")

    return final_states, mean_traj


def main():
    parser = argparse.ArgumentParser(description="Generate sweep reference outputs")
    parser.add_argument("--config", default=None, help="Run only this config name")
    args = parser.parse_args()

    output_dir = os.path.join(REPO_ROOT, "ref", "sweep")

    configs = dict(SWEEP_CONFIGS)
    if args.config:
        if args.config not in configs:
            print(f"ERROR: unknown config '{args.config}'")
            print(f"Available: {list(configs.keys())}")
            sys.exit(1)
        configs = {args.config: configs[args.config]}

    print(f"Generating {len(configs)} reference sweep outputs...")
    for name, cfg in configs.items():
        run_sweep(name, cfg, output_dir)

    print("\nDone. Reference outputs saved to ref/sweep/")


if __name__ == "__main__":
    main()
