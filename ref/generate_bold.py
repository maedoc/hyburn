#!/usr/bin/env python3
"""Generate reference BOLD monitor outputs from TVB hybrid simulator.

Runs G2DO with the TVB Bold monitor and saves:
  - BOLD time-series as ref/bold/{name}_bold.npy  [n_bold_samples, nnodes]
  - Neural trajectory (var0 mean) as ref/bold/{name}_neural.npy  [n_steps, nnodes]

Usage:
    ref/venv/bin/python ref/generate_bold.py
    ref/venv/bin/python ref/generate_bold.py --config g2do_bold
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

from configs import BOLD_CONFIG


def make_integrator(kind, dt):
    if kind == "heun":
        from tvb.simulator.integrators import HeunDeterministic
        return HeunDeterministic(dt=dt)
    else:
        from tvb.simulator.integrators import EulerDeterministic
        return EulerDeterministic(dt=dt)


def run_bold_sim(name, cfg, output_dir):
    """Run a simulation with BOLD monitor and save results."""
    from tvb.simulator.hybrid.simulator import Simulator
    from tvb.simulator.hybrid.network import NetworkSet
    from tvb.simulator.hybrid.subnetwork import Subnetwork
    from tvb.simulator.monitors import Bold, Raw

    model = cfg["model_factory"]()
    scheme = make_integrator(cfg["integrator"], cfg["dt"])
    nnodes = cfg["nnodes"]

    # Build subnetwork
    sn = Subnetwork(name="g2do", model=model, scheme=scheme, nnodes=nnodes)

    # Set initial state
    if "initial_state_npy" in cfg:
        npy_path = os.path.join(REPO_ROOT, cfg["initial_state_npy"])
        ic = np.load(npy_path).astype(np.float32)
        if ic.ndim == 2:
            ic = ic[:, :, np.newaxis]
        sn.state = ic.copy()
    else:
        ic = np.zeros((model.nvar, nnodes, 1), dtype=np.float32)
        if model.nvar >= 2:
            ic[1, :, :] = 0.5
        sn.state = ic.copy()

    # Build network set (single subnet, no projections)
    nets = NetworkSet(subnets=[sn], projections=[])

    # Build BOLD monitor
    bold_monitor = Bold(period=cfg["bold_tr"] * 1000.0)  # TVB expects ms
    raw_monitor = Raw()

    # Build simulator
    sim = Simulator(
        nets=nets,
        monitors=[bold_monitor, raw_monitor],
        simulation_length=cfg["sim_length"],
    )
    sim.configure()

    # Run with fixed initial conditions
    [(bold_times, bold_data), (raw_times, raw_data)] = sim.run(
        initial_conditions=[ic]
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)

    bold_path = os.path.join(output_dir, f"{name}_bold.npy")
    neural_path = os.path.join(output_dir, f"{name}_neural.npy")

    # bold_data: [n_bold_samples, vois, nnodes, modes] → squeeze
    bold_flat = bold_data.squeeze()  # [n_bold_samples, nnodes]
    np.save(bold_path, bold_flat)

    # raw_data: [n_raw_samples, vois, nnodes, modes] → extract var0
    raw_var0 = raw_data[:, 0, :, 0]  # [n_raw_samples, nnodes]
    # Downsample to bold_period intervals to keep file size manageable
    bold_period = cfg["bold_period"]
    raw_var0_ds = raw_var0[::bold_period]  # [n_bold_period_samples, nnodes]
    np.save(neural_path, raw_var0_ds)

    print(f"  {name}: {cfg['sim_length'] / cfg['dt']:.0f} steps")
    print(f"    BOLD shape: {bold_flat.shape}")
    print(f"    Neural (var0) shape: {raw_var0_ds.shape} (downsampled from {raw_var0.shape})")
    print(f"    saved: {bold_path}")
    print(f"    saved: {neural_path}")

    # Verify
    assert np.all(np.isfinite(bold_flat)), f"{name}: BOLD contains NaN/Inf"
    assert np.all(np.isfinite(raw_var0)), f"{name}: neural contains NaN/Inf"

    return bold_flat, raw_var0


def main():
    parser = argparse.ArgumentParser(description="Generate BOLD reference outputs")
    parser.add_argument("--config", default=None, help="Run only this config name")
    args = parser.parse_args()

    output_dir = os.path.join(REPO_ROOT, "ref", "bold")

    configs = dict(BOLD_CONFIG)
    if args.config:
        if args.config not in configs:
            print(f"ERROR: unknown config '{args.config}'")
            print(f"Available: {list(configs.keys())}")
            sys.exit(1)
        configs = {args.config: configs[args.config]}

    print(f"Generating {len(configs)} reference BOLD outputs...")
    for name, cfg in configs.items():
        run_bold_sim(name, cfg, output_dir)

    print("\nDone. Reference outputs saved to ref/bold/")


if __name__ == "__main__":
    main()
