#!/usr/bin/env python3
"""Generate coupling reference traces for hyburn unit tests.

Produces .npy files for each coupling function using the EXACT classic TVB
coupling pipeline:

    result[i, k] = cfun.post(sum_j(W[i,j] * cfun.pre(x_i_broadcast, x_j_expanded)[k, j, :]))

Where:
- x_j_expanded broadcasts source states across target nodes
- x_i_broadcast broadcasts target states across source nodes
- The weighted sum is over source nodes j

Also generates hyburn-style outputs (post_with_target) for Kuramoto and
Difference, plus E2E simulation traces.

Output directory: ref/coupling_semantics/
"""

import os
import sys
import json
import importlib

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(REPO_ROOT, "ref", "coupling_semantics")

TVB_ROOT = os.path.join(os.path.abspath(os.path.dirname(__file__)), "tvb-root")
TVB_LIB = os.path.join(TVB_ROOT, "tvb_library")


def import_tvb():
    try:
        sys.path.insert(0, TVB_LIB)
        import tvb.simulator.coupling as classic_coupling
        import tvb.simulator.hybrid.coupling as hybrid_coupling
        return classic_coupling, hybrid_coupling
    except ImportError as e:
        print(f"ERROR: Cannot import TVB coupling modules: {e}")
        print(f"  TVB_ROOT={TVB_ROOT}")
        print(f"  TVB_LIB={TVB_LIB}")
        print("  Run: ./ref/setup.sh to install TVB dependencies")
        sys.exit(1)


# =========================================================================
# Classic TVB coupling pipeline implementations (pure numpy)
# =========================================================================

def classic_linear(W, x_j, x_i=None, a=0.00390625, b=0.0):
    """Linear(a, b): pre=x_j, post=a*gx+b.
    result = a * (W @ x_j) + b
    """
    gx = W @ x_j
    return a * gx + b


def classic_sigmoidal(W, x_j, x_i=None, cmin=-1.0, cmax=1.0, midpoint=0.0, a=1.0, sigma=1.0):
    """Sigmoidal: pre=x_j, post=cmin+(cmax-cmin)/(1+exp(-a*(gx-midpoint)/sigma))."""
    gx = W @ x_j
    return cmin + (cmax - cmin) / (1.0 + np.exp(-a * ((gx - midpoint) / sigma)))


def classic_tanh(W, x_j, x_i=None, a=1.0, b=1.0, midpoint=0.0, sigma=1.0):
    """HyperbolicTangent: pre=a*(1+tanh((b*x_j-midpoint)/sigma)), post=identity.
    result = W @ [a * (1 + tanh((b*x_j - midpoint) / sigma))]
    """
    pre = a * (1.0 + np.tanh((b * x_j - midpoint) / sigma))
    gx = W @ pre
    return gx


def classic_kuramoto(W, x_j, x_i, a=1.0):
    """Kuramoto: pre=sin(x_j - x_i), post=a/gx.shape[0]*gx.
    result[i,k] = (a/N) * sum_j(W[i,j] * sin(x_j[k,j] - x_i[k,i]))
    Uses x_i for per-edge phase difference.
    """
    n_tgt = W.shape[0]
    n_src = W.shape[1]
    ncvar = x_j.shape[1] if x_j.ndim > 1 else 1

    if x_j.ndim == 1:
        x_j_2d = x_j.reshape(n_src, 1)
    else:
        x_j_2d = x_j
    if x_i.ndim == 1:
        x_i_2d = x_i.reshape(n_tgt, 1)
    else:
        x_i_2d = x_i

    ncvar = x_j_2d.shape[1]
    result = np.zeros((n_tgt, ncvar), dtype=np.float64)

    for k in range(ncvar):
        for i in range(n_tgt):
            s = 0.0
            for j in range(n_src):
                s += W[i, j] * np.sin(x_j_2d[j, k] - x_i_2d[i, k])
            result[i, k] = a / n_src * s

    return result


def classic_difference(W, x_j, x_i, a=0.1):
    """Difference: pre=x_j-x_i, post=a*gx.
    result[i,k] = a * sum_j(W[i,j] * (x_j[k,j] - x_i[k,i]))
    """
    n_tgt = W.shape[0]
    n_src = W.shape[1]

    if x_j.ndim == 1:
        x_j_2d = x_j.reshape(n_src, 1)
    else:
        x_j_2d = x_j
    if x_i.ndim == 1:
        x_i_2d = x_i.reshape(n_tgt, 1)
    else:
        x_i_2d = x_i

    ncvar = x_j_2d.shape[1]
    result = np.zeros((n_tgt, ncvar), dtype=np.float64)

    for k in range(ncvar):
        diff = x_j_2d[:, k][np.newaxis, :] - x_i_2d[:, k][:, np.newaxis]
        result[:, k] = a * np.sum(W * diff, axis=1)

    return result


def classic_sigmoidal_jr(W, x_j_2cvar, x_i=None, a=1.0, cmin=0.0, cmax=0.005, r=0.56, midpoint=6.0):
    """SigmoidalJansenRit (classic mode): pre uses x_j[:,0]-x_j[:,1], post=a*gx.
    pre[j] = cmin + (cmax-cmin)/(1+exp(r*(midpoint-(x_j[j,0]-x_j[j,1]))))
    result = a * (W @ pre)
    """
    diff = x_j_2cvar[:, 0] - x_j_2cvar[:, 1]
    pre = cmin + (cmax - cmin) / (1.0 + np.exp(r * (midpoint - diff)))
    gx = W @ pre
    return a * gx


def sjr_hyburn_legacy(W, x_j_2cvar, a=1.0, e0=0.005, r=0.56, v0=6.0):
    """SigmoidalJansenRit legacy hyburn formula: a*(2*e0)/(1+exp(r*(v0-x))).
    x = x_j[:,0] - x_j[:,1].
    """
    diff = x_j_2cvar[:, 0] - x_j_2cvar[:, 1]
    pre = a * (2 * e0) / (1.0 + np.exp(r * (v0 - diff)))
    gx = W @ pre
    return gx


def classic_presigmoidal_static(W, x_j, x_i=None, H=1.0, Q=0.0, G=1.0, P=1.0, theta=0.5):
    """PreSigmoidal static: H*(Q+tanh(G*(P*x_j - theta))).
    Standard pre-summation, single cvar input.
    """
    pre = H * (Q + np.tanh(G * (P * x_j - theta)))
    if pre.ndim == 1:
        gx = W @ pre
    else:
        gx = W @ pre
    return gx


def classic_presigmoidal_dynamic(W, x_j_2cvar, x_i=None, H=0.5, Q=1.0, G=60.0, P=1.0, globalT=False):
    """PreSigmoidal dynamic: H*(Q+tanh(G*(P*x_j[:,0]-x_j[:,1]))).
    Uses 2 cvar input: x_j[:,0] = afferent signal, x_j[:,1] = dynamic threshold.
    Result shape: [2, n_tgt] — first row is weighted sum of A_j, second is diagonal self-connection.
    Matches classic TVB PreSigmoidal.__call__ for dynamic=True, globalT=False.
    """
    n_tgt = W.shape[0]
    n_src = W.shape[1]
    x0 = x_j_2cvar[:, 0]
    x1 = x_j_2cvar[:, 1]
    arg = P * x0 - x1
    A_j = H * (Q + np.tanh(G * arg))

    c_0 = W @ A_j

    c_1 = A_j.copy()
    if globalT:
        c_1 = np.full_like(c_1, c_1.mean())

    return np.stack([c_0, c_1], axis=0)


def hyburn_kuramoto(W, x_j, x_i, a=1.0):
    """Hyburn Kuramoto: 2-channel pre(sin,cos) + post_with_target.
    pre = [sin(x_j), cos(x_j)]  (concatenated)
    gx = W @ pre  →  [W@sin, W@cos]
    result = (a/n_src) * (cos(x_i) * gx_sin - sin(x_i) * gx_cos)
    """
    n_src = x_j.shape[0] if x_j.ndim > 1 else x_j.shape[0]
    sin_x = np.sin(x_j)
    cos_x = np.cos(x_j)
    pre = np.concatenate([sin_x, cos_x], axis=1)
    gx = W @ pre

    if x_j.ndim == 1 or x_j.shape[1] == 1:
        ncvar = 1
        gx_sin = gx[:, :ncvar]
        gx_cos = gx[:, ncvar:]
        x_i_2d = x_i.reshape(-1, 1) if x_i.ndim == 1 else x_i
        result = np.cos(x_i_2d) * gx_sin - np.sin(x_i_2d) * gx_cos
    else:
        ncvar = x_j.shape[1]
        gx_sin = gx[:, :ncvar]
        gx_cos = gx[:, ncvar:]
        result = np.cos(x_i) * gx_sin - np.sin(x_i) * gx_cos

    return (a / n_src) * result


def hyburn_difference(W, x_j, x_i, a=0.1):
    """Hyburn Difference: W @ x_j then post_with_target rowsums.
    result = a * (W @ x_j - x_i * rowsums)
    where rowsums[i] = sum_j(W[i,j]).
    """
    rowsums = W.sum(axis=1, keepdims=True)
    gx = W @ x_j
    return a * (gx - x_i * rowsums)


def classic_scaled_linear(W, x_j, x_i=None, a=1.0, b=0.001):
    """ScaledLinear: pre=x_j, post=a*(x-b).
    result = a * (W @ x_j - b)
    Note: classic TVB 'Scaling' is a*gx (no offset b). hyburn's ScaledLinear
    uses a*(gx - b). We generate the hyburn version.
    """
    gx = W @ x_j
    return a * (gx - b)


# =========================================================================
# Simple model dfuns for E2E simulation traces
# =========================================================================

def g2do_dfun(state, coupling, params=None):
    """Generic2dOscillator dfun matching hyburn's full parameterisation.

    Params: [tau, I, a, b, c, d, e, f, g, alpha, beta, gamma]
    Defaults: [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0]

    Equations:
      dV = d*tau*(alpha*W + gamma*(I + c0) - f*V^3 + e*V^2 + g*V)
      dW = d/tau*(a + b*V + c*V^2 - beta*W)
    """
    if params is None:
        params = [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    tau, I_ext, a, b, c, d, e, f, g, alpha, beta, gamma = params

    V = state[0]
    W = state[1]
    c0 = coupling[0] if coupling is not None else np.zeros_like(V)
    i_ext_term = gamma * (I_ext + c0)
    dtau = d * tau
    d_over_tau = d / tau

    dV = dtau * (alpha * W + i_ext_term - f * V**3 + e * V**2 + g * V)
    dW = d_over_tau * (a + b * V + c * V**2 - beta * W)
    return np.array([dV, dW])


def jr_dfun(state, coupling, params=None):
    """JansenRit dfun matching hyburn.

    Params: [A, B, a, b, j5a, v, numax, r, v0, j2pi, pmin, pmax, e0]
    Defaults: [3.25, 22.0, 100.0, 50.0, 135.0, 0.001, 0.56, 512.0, 6.0, 1000.0, 0.08, 0.28, 0.005]
    """
    if params is None:
        params = [3.25, 22.0, 100.0, 50.0, 135.0, 0.001, 0.56, 512.0, 6.0, 1000.0, 0.08, 0.28, 0.005]
    A, B, a, b = params[0], params[1], params[2], params[3]

    y0, y1, y2, y3, y4, y5 = state

    cy1 = coupling[0] if coupling is not None else np.zeros_like(y1)
    cy2 = coupling[1] if coupling is not None else np.zeros_like(y2) if coupling is not None else np.zeros_like(y2)

    sig_y1_y3 = 2.0 * 0.0025 / (1.0 + np.exp(0.56 * (6.0 - (y1 - y3))))
    sig_y2 = 2.0 * 0.0025 / (1.0 + np.exp(0.56 * (6.0 - y2)))

    dy0 = y3
    dy1 = A * a * sig_y1_y3
    dy2 = y4 - y5 + cy1
    dy3 = A * a * sig_y2 - b * y3
    dy4 = B * a * sig_y1_y3 - b * y4
    dy5 = b * y5
    return np.array([dy0, dy1, dy2, dy3, dy4, dy5])


# =========================================================================
# TRACES dict: defines all reference data to generate
# =========================================================================

TRACES = {
    # NOTE: Input data is generated in main() from a single RandomState(42)
    # to match verify_coupling_semantics.py exactly.
    "weights": {
        "type": "input",
        "desc": "4x4 deterministic weight matrix (zero diagonal)",
    },
    "x_j": {
        "type": "input",
        "desc": "(4,1) random source states, values in [-2,2)",
    },
    "x_i": {
        "type": "input",
        "desc": "(4,1) random target states, values in [-2,2)",
    },
    "x_j_2cvar": {
        "type": "input",
        "desc": "(4,2) random for 2-cvar input, values in [-2,2)",
    },

    # --- Coupling function outputs (classic TVB pipeline) ---

    "Linear_b0_classic": {
        "type": "coupling",
        "desc": "Linear(a=0.004, b=0): 0.004 * (W @ x_j)",
        "fn": lambda W, xj, xi, xj2: classic_linear(W, xj, xi, a=0.004, b=0.0),
        "input": "x_j",
    },
    "Linear_b0.1_classic": {
        "type": "coupling",
        "desc": "Linear(a=0.004, b=0.1): 0.004 * (W @ x_j) + 0.1",
        "fn": lambda W, xj, xi, xj2: classic_linear(W, xj, xi, a=0.004, b=0.1),
        "input": "x_j",
    },
    "Sigmoidal_classic": {
        "type": "coupling",
        "desc": "Sigmoidal(cmin=-1, cmax=1, a=1, midpoint=0, sigma=1)",
        "fn": lambda W, xj, xi, xj2: classic_sigmoidal(W, xj, xi, cmin=-1.0, cmax=1.0, midpoint=0.0, a=1.0, sigma=1.0),
        "input": "x_j",
    },
    "Tanh_classic": {
        "type": "coupling",
        "desc": "HyperbolicTangent(a=1, b=2, midpoint=0, sigma=1) — NOTE b=2 not b=1",
        "fn": lambda W, xj, xi, xj2: classic_tanh(W, xj, xi, a=1.0, b=2.0, midpoint=0.0, sigma=1.0),
        "input": "x_j",
    },
    "Kuramoto_classic": {
        "type": "coupling",
        "desc": "Kuramoto(a=1): (1/N) * sum_j(w_ij * sin(x_j - x_i)) per edge",
        "fn": lambda W, xj, xi, xj2: classic_kuramoto(W, xj, xi, a=1.0),
        "input": "x_j",
        "needs_xi": True,
    },
    "Kuramoto_hyburn": {
        "type": "coupling",
        "desc": "Kuramoto(a=1): hyburn 2-channel (sin,cos) post_with_target",
        "fn": lambda W, xj, xi, xj2: hyburn_kuramoto(W, xj, xi, a=1.0),
        "input": "x_j",
        "needs_xi": True,
    },
    "Difference_classic": {
        "type": "coupling",
        "desc": "Difference(a=0.1): a * sum_j(w_ij * (x_j - x_i)) per edge",
        "fn": lambda W, xj, xi, xj2: classic_difference(W, xj, xi, a=0.1),
        "input": "x_j",
        "needs_xi": True,
    },
    "Difference_hyburn": {
        "type": "coupling",
        "desc": "Difference(a=0.1): hyburn rowsums approach: a*(W@x_j - x_i*rowsums)",
        "fn": lambda W, xj, xi, xj2: hyburn_difference(W, xj, xi, a=0.1),
        "input": "x_j",
        "needs_xi": True,
    },
    "SigmoidalJansenRit_classic": {
        "type": "coupling",
        "desc": "SigmoidalJansenRit classic mode (cmin=0, cmax=0.005, a=1, r=0.56, midpoint=6)",
        "fn": lambda W, xj, xi, xj2: classic_sigmoidal_jr(W, xj2, a=1.0, cmin=0.0, cmax=0.005, r=0.56, midpoint=6.0),
        "input": "x_j_2cvar",
    },
    "SigmoidalJansenRit_hyburn": {
        "type": "coupling",
        "desc": "SigmoidalJansenRit legacy hyburn: a*(2*e0)/(1+exp(r*(v0-(x0-x1))))",
        "fn": lambda W, xj, xi, xj2: sjr_hyburn_legacy(W, xj2, a=1.0, e0=0.005, r=0.56, v0=6.0),
        "input": "x_j_2cvar",
    },
    "PreSigmoidal_static_classic": {
        "type": "coupling",
        "desc": "PreSigmoidal static (H=1, Q=0, G=1, P=1, theta=0.5): H*(Q+tanh(G*(P*x-theta)))",
        "fn": lambda W, xj, xi, xj2: classic_presigmoidal_static(W, xj, H=1.0, Q=0.0, G=1.0, P=1.0, theta=0.5),
        "input": "x_j",
    },
    "PreSigmoidal_dynamic_classic": {
        "type": "coupling",
        "desc": "PreSigmoidal dynamic (H=0.5, Q=1, G=60, P=1, dynamic=True): 2-cvar H*(Q+tanh(G*(P*x0-x1)))",
        "fn": lambda W, xj, xi, xj2: classic_presigmoidal_dynamic(W, xj2, H=0.5, Q=1.0, G=60.0, P=1.0),
        "input": "x_j_2cvar",
    },
    "ScaledLinear_classic": {
        "type": "coupling",
        "desc": "ScaledLinear(a=0.004, b=0.001): a*(W@x_j - b)",
        "fn": lambda W, xj, xi, xj2: classic_scaled_linear(W, xj, a=0.004, b=0.001),
        "input": "x_j",
    },

    # --- E2E simulation traces ---

    "g2do_linear_weak_final": {
        "type": "e2e",
        "desc": "G2DO with Linear(a=0.001,b=0) weak coupling, 4 nodes, 200 steps, dt=0.1",
        "model": "g2do",
        "nnodes": 4,
        "dt": 0.1,
        "steps": 200,
        "sim_length": 20.0,
        "weights": np.array([[0,0.01,0.01,0.01],[0.01,0,0.01,0.01],[0.01,0.01,0,0.01],[0.01,0.01,0.01,0]], dtype=np.float64),
        "coupling_fn": "linear",
        "coupling_params": {"a": 0.001, "b": 0.0},
        "cvar": [0],
        "ic": np.array([[0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]], dtype=np.float64),
    },
}


def heun_step(dfuns, state, params, coupling, dt):
    """One Heun integration step."""
    k1 = dfuns(state, params, coupling)
    k2 = dfuns(state + dt * k1, params, coupling)
    return state + dt * 0.5 * (k1 + k2)


def compute_coupling_for_sim(state, weights, cvar_indices, cfun_name, cfun_params):
    """Compute coupling for E2E simulation traces."""
    nnodes = weights.shape[0]
    ncvar = len(cvar_indices)

    if cfun_name == "linear":
        x_j = np.zeros((nnodes, ncvar), dtype=np.float64)
        for j, cv in enumerate(cvar_indices):
            x_j[:, j] = state[cv]
        a = cfun_params["a"]
        b = cfun_params.get("b", 0.0)
        result = classic_linear(weights, x_j, a=a, b=b)
        coupling = np.zeros((ncvar, nnodes), dtype=np.float64)
        for j in range(ncvar):
            coupling[j] = result[:, j] if result.ndim > 1 else result

    elif cfun_name == "difference":
        x_j = np.zeros((nnodes, ncvar), dtype=np.float64)
        x_i = np.zeros((nnodes, ncvar), dtype=np.float64)
        for j, cv in enumerate(cvar_indices):
            x_j[:, j] = state[cv]
            x_i[:, j] = state[cv]
        a = cfun_params["a"]
        result = hyburn_difference(weights, x_j, x_i, a=a)
        coupling = np.zeros((ncvar, nnodes), dtype=np.float64)
        for j in range(ncvar):
            coupling[j] = result[:, j]

    elif cfun_name == "sjr_classic":
        x_j_2cvar = np.zeros((nnodes, 2), dtype=np.float64)
        for j, cv in enumerate(cvar_indices):
            x_j_2cvar[:, j] = state[cv]
        result = classic_sigmoidal_jr(weights, x_j_2cvar, **cfun_params)
        coupling = np.zeros((1, nnodes), dtype=np.float64)
        coupling[0] = result if result.ndim == 1 else result[:, 0] if result.shape[0] == nnodes else result

    elif cfun_name == "sjr_hyburn":
        x_j_2cvar = np.zeros((nnodes, 2), dtype=np.float64)
        for j, cv in enumerate(cvar_indices):
            x_j_2cvar[:, j] = state[cv]
        result = sjr_hyburn_legacy(weights, x_j_2cvar, **cfun_params)
        coupling = np.zeros((1, nnodes), dtype=np.float64)
        coupling[0] = result

    else:
        raise ValueError(f"Unknown coupling function: {cfun_name}")

    return coupling


def run_e2e_trace(name, cfg, rng):
    """Run an E2E simulation trace and save final state."""
    model = cfg["model"]
    nnodes = cfg["nnodes"]
    dt = cfg["dt"]
    steps = cfg["steps"]
    weights = cfg["weights"].astype(np.float64)
    cvar_indices = cfg["cvar"]
    cfun_name = cfg["coupling_fn"]
    cfun_params = cfg["coupling_params"]

    if model == "g2do":
        if "ic" in cfg:
            state = cfg["ic"].astype(np.float64).copy()
        else:
            state = np.zeros((2, nnodes), dtype=np.float64)
            state[1, :] = 0.5
        dfun = lambda s, p, c: g2do_dfun(s, c)
    elif model == "jr":
        state = np.zeros((6, nnodes), dtype=np.float64)
        dfun = lambda s, p, c: jr_dfun(s, c)
    else:
        raise ValueError(f"Unknown model: {model}")

    trajectory = np.zeros((steps + 1, state.shape[0], nnodes), dtype=np.float32)
    trajectory[0] = state.copy().astype(np.float32)

    for step in range(1, steps + 1):
        coupling = compute_coupling_for_sim(state, weights, cvar_indices, cfun_name, cfun_params)
        k1 = dfun(state, None, coupling)
        k2 = dfun(state + dt * k1, None, coupling)
        state = state + dt * 0.5 * (k1 + k2)
        if not np.all(np.isfinite(state)):
            print(f"  WARNING: {name} diverged at step {step}")
            trajectory[step:] = np.nan
            break
        trajectory[step] = state.copy().astype(np.float32)

    final = trajectory[-1]
    final_3d = final[:, :, np.newaxis]
    return final_3d


def main():
    print(f"Generating coupling parity reference data...")
    print(f"Output directory: {OUTPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate input data from a single RandomState(42) — same as verify_coupling_semantics.py
    rs = np.random.RandomState(42)
    W = np.array([
        [0.0, 0.2, 0.3, 0.1],
        [0.2, 0.0, 0.1, 0.3],
        [0.3, 0.1, 0.0, 0.2],
        [0.1, 0.3, 0.2, 0.0],
    ], dtype=np.float64)
    x_j = (rs.randn(4, 1) * 2).astype(np.float64)
    x_i = (rs.randn(4, 1) * 2).astype(np.float64)
    x_j_2cvar = (rs.randn(4, 2) * 2).astype(np.float64)

    # Save input data
    np.save(os.path.join(OUTPUT_DIR, "weights.npy"), W.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "x_j.npy"), x_j.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "x_i.npy"), x_i.astype(np.float32))
    np.save(os.path.join(OUTPUT_DIR, "x_j_2cvar.npy"), x_j_2cvar.astype(np.float32))
    print(f"  Saved weights.npy: shape={W.shape}")
    print(f"  Saved x_j.npy: shape={x_j.shape}")
    print(f"  Saved x_i.npy: shape={x_i.shape}")
    print(f"  Saved x_j_2cvar.npy: shape={x_j_2cvar.shape}")

    rng = np.random.default_rng(42)  # For E2E traces that need their own rng

    parity_results = {}

    for name, cfg in TRACES.items():
        if cfg["type"] == "input":
            continue

        if cfg["type"] == "coupling":
            x_input = x_j if cfg.get("input") == "x_j" else x_j_2cvar
            xi_input = x_i if cfg.get("needs_xi") else None

            if xi_input is not None:
                result = cfg["fn"](W, x_input, xi_input, x_j_2cvar)
            else:
                result = cfg["fn"](W, x_input, None, x_j_2cvar)

        elif cfg["type"] == "e2e":
            result = run_e2e_trace(name, cfg, rng)

        result_f32 = result.astype(np.float32)
        path = os.path.join(OUTPUT_DIR, f"{name}.npy")
        np.save(path, result_f32)
        print(f"  Saved {name}.npy: shape={result_f32.shape}, dtype={result_f32.dtype}")

        config = {
            "name": name,
            "type": cfg["type"],
            "desc": cfg.get("desc", ""),
        }
        for k, v in cfg.items():
            if k in ("fn", "gen", "type"):
                continue
            if isinstance(v, np.ndarray):
                config[k] = v.tolist()
            else:
                config[k] = v

        config_path = os.path.join(OUTPUT_DIR, f"{name}_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        if cfg["type"] == "coupling":
            parity_results[name] = {
                "shape": list(result_f32.shape),
                "desc": cfg.get("desc", ""),
            }

    print(f"\nVerifying Kuramoto/Difference parity (hyburn == classic):")

    kur_classic = np.load(os.path.join(OUTPUT_DIR, "Kuramoto_classic.npy"))
    kur_hyburn = np.load(os.path.join(OUTPUT_DIR, "Kuramoto_hyburn.npy"))
    diff = np.max(np.abs(kur_classic - kur_hyburn))
    print(f"  Kuramoto: max|classic - hyburn| = {diff:.2e}  {'AGREE' if diff < 1e-5 else 'DISAGREE'}")

    diff_classic = np.load(os.path.join(OUTPUT_DIR, "Difference_classic.npy"))
    diff_hyburn = np.load(os.path.join(OUTPUT_DIR, "Difference_hyburn.npy"))
    diff = np.max(np.abs(diff_classic - diff_hyburn))
    print(f"  Difference: max|classic - hyburn| = {diff:.2e}  {'AGREE' if diff < 1e-5 else 'DISAGREE'}")

    output_files = sorted(os.listdir(OUTPUT_DIR))
    print(f"\nGenerated {len(output_files)} files in {OUTPUT_DIR}:")
    for f in output_files:
        fpath = os.path.join(OUTPUT_DIR, f)
        if f.endswith(".npy"):
            arr = np.load(fpath)
            print(f"  {f}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"  {f}")


if __name__ == "__main__":
    main()