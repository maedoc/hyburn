#!/usr/bin/env python3
"""Generate reference JR coupling-strength PSD sweep data.

Uses vectorized coupling computation matching hyburn's semantics:
  - JansenRit model with cvar=[0] (NCVAR=1)
  - Linear coupling: G * weights @ y0_src (delayed)
  - Delayed coupling via ring buffer
  - Heun integration at dt=0.1ms

Saves:
  - ref/jr_psd_sweep/conn_weights.npy     [nnodes, nnodes]
  - ref/jr_psd_sweep/conn_delays.npy      [nnodes, nnodes]  (in steps)
  - ref/jr_psd_sweep/coupling_values.npy  [n_sweep]
  - ref/jr_psd_sweep/psd_mean.npy         [n_sweep, n_freqs]
  - ref/jr_psd_sweep/freqs.npy            [n_freqs]

Usage:
    ref/venv/bin/python ref/generate_jr_psd_sweep.py
"""

import os
import sys
import warnings
import numpy as np
from scipy.signal import welch

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# JansenRit dfun (vectorized over nodes, matches hyburn)
# ---------------------------------------------------------------------------

def jr_dfun(state, coupling, params):
    """JansenRit derivative function.

    state:    [nvar, nnodes]
    coupling: [nnodes]
    params:   [13]
    """
    A, B, a, b, v0, nu_max, r, J = params[:8]
    a_1, a_2, a_3, a_4, mu = params[8:13]

    y0, y1, y2, y3, y4, y5 = state
    c_0 = coupling

    def sigm(x):
        return 2.0 * nu_max / (1.0 + np.exp(r * (v0 - x)))

    sigm_y1_y2 = sigm(y1 - y2)
    sigm_y0_1 = sigm(a_1 * J * y0)
    sigm_y0_3 = sigm(a_3 * J * y0)

    dy0 = y3
    dy1 = y4
    dy2 = y5
    dy3 = A * a * sigm_y1_y2 - 2.0 * a * y3 - a * a * y0
    dy4 = (A * a * (sigm_y0_1 * a_2 * J + mu) + c_0) \
          - 2.0 * a * y4 - a * a * y1
    dy5 = B * b * sigm_y0_3 * a_4 * J - 2.0 * b * y5 - b * b * y2

    return np.array([dy0, dy1, dy2, dy3, dy4, dy5])


# ---------------------------------------------------------------------------
# Vectorized coupling computation
# ---------------------------------------------------------------------------

def compute_coupling(history_y0, weights, delay_steps, G, nnodes):
    """Compute delayed linear coupling: c_i = G * sum_j w[j,i] * y0_j[t - d[j,i]].

    history_y0: [max_delay, nnodes] — ring buffer of y0 at past steps
                history_y0[0] = current, history_y0[1] = 1 step ago, etc.
    """
    # For efficiency: precompute which delay values exist
    # coupling[i] = G * sum_j weights[j,i] * history_y0[delay[j,i], j]
    coupling = np.zeros(nnodes, dtype=np.float32)
    if G == 0.0:
        return coupling

    # Vectorized: for each delay d, extract y0 at delay d and compute weighted sum
    for d in range(history_y0.shape[0]):
        mask = delay_steps == d
        if mask.any():
            # w_masked[j,i] = weights[j,i] if delay[j,i] == d else 0
            w_masked = np.where(mask, weights, 0.0)
            # y0_delayed[j] = history_y0[d, j]
            y0_delayed = history_y0[d]  # [nnodes]
            # coupling[i] += G * sum_j w_masked[j,i] * y0_delayed[j]
            coupling += G * (w_masked.T @ y0_delayed)

    return coupling


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_jr_coupled(params, weights, delay_steps, G,
                   nnodes, dt, n_steps, ic, max_delay):
    """Run JR with Heun integration + delayed linear coupling."""
    nvar = 6

    # Ring buffer for y0 history: [max_delay, nnodes]
    history_y0 = np.zeros((max_delay, nnodes), dtype=np.float32)
    for d in range(max_delay):
        history_y0[d] = ic[0].copy()

    state = ic.copy()  # [nvar, nnodes]
    # Only record y0 and y2 for PSD (pyramidal output = y0 - y2)
    voi0_traj = np.zeros((n_steps, nnodes), dtype=np.float32)

    for step in range(n_steps):
        # 1. Compute coupling
        coupling = compute_coupling(history_y0, weights, delay_steps, G, nnodes)

        # 2. Heun step
        k1 = jr_dfun(state, coupling, params)
        x_mid = state + dt * k1
        k2 = jr_dfun(x_mid, coupling, params)
        state = (state + 0.5 * dt * (k1 + k2)).astype(np.float32)

        # 3. Record voi0 = y0 - y2
        voi0_traj[step] = state[0] - state[2]

        # 4. Update ring buffer
        history_y0 = np.roll(history_y0, 1, axis=0)
        history_y0[0] = state[0]

        # Check for instability
        if not np.all(np.isfinite(state)):
            return None, step

    return voi0_traj, n_steps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from tvb.datatypes.connectivity import Connectivity

    output_dir = os.path.join(REPO_ROOT, "ref", "jr_psd_sweep")
    os.makedirs(output_dir, exist_ok=True)

    # Load connectome
    conn = Connectivity.from_file()
    conn.configure()
    nnodes = conn.number_of_regions
    weights = conn.weights.astype(np.float32)
    # Normalize: divide by max so weights ∈ [0, 1]
    wmax = weights.max()
    if wmax > 0:
        weights = weights / wmax

    # Delays in steps
    dt = 0.1  # ms
    delays_ms = conn.delays.astype(np.float32)
    delay_steps = np.clip(np.round(delays_ms / dt).astype(int), 0, None)
    max_delay = int(delay_steps.max()) + 1

    print(f"Connectome: {nnodes} nodes, max_delay={max_delay} steps")
    print(f"  Weights: [{weights.min():.4f}, {weights.max():.4f}]")

    # Save connectome
    np.save(os.path.join(output_dir, "conn_weights.npy"), weights)
    np.save(os.path.join(output_dir, "conn_delays.npy"), delay_steps)

    # JR params
    params = np.array([
        3.25, 22.0, 0.1, 0.05, 5.52, 0.0025, 0.56, 135.0,
        1.0, 0.8, 0.25, 0.25, 0.22
    ], dtype=np.float32)

    # Equilibrium IC
    eq = np.array([5.4573e-02, 1.3888e+01, 6.9978e+00,
                   2.1884e-03, 2.2329e-01, -2.3372e-02], dtype=np.float32)
    ic = np.zeros((6, nnodes), dtype=np.float32)
    for i in range(6):
        ic[i, :] = eq[i]

    # Simulation
    sim_length = 2000.0  # 2s — enough for PSD with welch nperseg=1024
    n_steps = int(sim_length / dt)  # 20000
    fs = 1000.0 / dt  # 10000 Hz
    transient_steps = 2000  # discard first 200ms

    # PSD params
    nperseg = 1024
    noverlap = nperseg // 2

    # Coupling sweep: 10 points covering subcritical → near bifurcation → oscillatory
    coupling_values = np.array([
        0.0, 0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05,
    ], dtype=np.float32)
    n_sweep = len(coupling_values)

    print(f"\nSweep: {n_sweep} coupling values")
    print(f"Simulation: {sim_length}ms ({n_steps} steps)")

    psd_results = []
    freqs_ref = None

    for si, G in enumerate(coupling_values):
        print(f"  G={G:7.3f} ({si+1}/{n_sweep}) ...", end="", flush=True)

        voi0_traj, actual_steps = run_jr_coupled(
            params, weights, delay_steps, float(G),
            nnodes, dt, n_steps, ic, max_delay
        )

        if voi0_traj is None:
            print(f" NaN at step {actual_steps}")
            psd_results.append(np.full(len(freqs_ref) if freqs_ref is not None else 1025, np.nan))
            continue

        # Discard transient
        voi0_ss = voi0_traj[transient_steps:]

        # PSD per node, average
        psd_nodes = []
        for node in range(nnodes):
            freqs, pxx = welch(voi0_ss[:, node], fs=fs, nperseg=nperseg, noverlap=noverlap)
            psd_nodes.append(pxx)
        psd_mean = np.mean(psd_nodes, axis=0).astype(np.float32)

        if freqs_ref is None:
            freqs_ref = freqs.astype(np.float32)

        psd_results.append(psd_mean)

        alpha_idx = (freqs >= 8) & (freqs <= 13)
        alpha_frac = psd_mean[alpha_idx].sum() / psd_mean.sum() if psd_mean.sum() > 0 else 0
        peak_freq = freqs[np.argmax(psd_mean)]
        print(f" alpha_frac={alpha_frac:.4f}, peak={peak_freq:.1f}Hz")

    # Save
    np.save(os.path.join(output_dir, "coupling_values.npy"), coupling_values)
    np.save(os.path.join(output_dir, "psd_mean.npy"), np.array(psd_results, dtype=np.float32))
    np.save(os.path.join(output_dir, "freqs.npy"), freqs_ref)

    print(f"\nSaved {len(psd_results)} PSDs to {output_dir}")


if __name__ == "__main__":
    main()
