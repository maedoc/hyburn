"""
Coupling semantics verification across TVB classic, TVB hybrid (fixed), and hyburn.

All three now implement the same pipeline for functions where pre() doesn't need x_i:

    result = scale * post( Σ w · pre(x_j) )

For Kuramoto and Difference, hyburn now uses correct classic TVB semantics:
- Kuramoto: a/N * Σ w_ij * sin(x_j - x_i) via 2-channel pre + post_with_target
- Difference: a * Σ w_ij * (x_j - x_i) via rowsum preprocessing or diagonal modification

Pipelines compared:
1. TVB Classic: post(Σ w · pre(x_i, x_j))  — pre per-edge, has x_i access
2. TVB Hybrid (fixed): scale * post(Σ w · pre(x_j))  — pre per-edge, no x_i
3. Hyburn: post_with_target(W @ pre(x_j), x_i)  — has x_i access via post_with_target
"""

import numpy as np
import sys
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ========================================================================
# Coupling function implementations
# ========================================================================

def linear_post(gx, a, b):
    return a * gx + b

def sigmoidal_post(gx, cmin, cmax, midpoint, a, sigma):
    return cmin + (cmax - cmin) / (1.0 + np.exp(-a * ((gx - midpoint) / sigma)))

def tanh_pre(x_j, a, b, midpoint, sigma):
    return a * (1.0 + np.tanh((b * x_j - midpoint) / sigma))

def sigmoidal_jr_pre(x_j, a, e0, r, v0):
    diff = x_j[:, 0] - x_j[:, 1]
    return a * (2 * e0) / (1.0 + np.exp(r * (v0 - diff)))


# ========================================================================
# Pipeline: TVB Classic — post(Σ w · pre(x_i, x_j))
# ========================================================================

def pipeline_classic(W, x_i, x_j, cfun_name, cfun_params):
    """Classic TVB: pre has x_i access."""
    if cfun_name == "Linear":
        gx = W @ x_j
        return linear_post(gx, cfun_params['a'], cfun_params['b'])

    elif cfun_name == "Sigmoidal":
        gx = W @ x_j
        return sigmoidal_post(gx, cfun_params['cmin'], cfun_params['cmax'], cfun_params['midpoint'], cfun_params['a'], cfun_params['sigma'])

    elif cfun_name == "HyperbolicTangent":
        pre_vals = tanh_pre(x_j, **{k: cfun_params[k] for k in ['a','b','midpoint','sigma']})
        gx = W @ pre_vals
        return gx  # post is identity

    elif cfun_name == "Kuramoto":
        a = cfun_params['a']
        x_i_flat = x_i[:, 0] if x_i.ndim > 1 else x_i
        x_j_flat = x_j[:, 0] if x_j.ndim > 1 else x_j
        diff = x_j_flat[np.newaxis, :] - x_i_flat[:, np.newaxis]
        pre_vals = np.sin(diff)
        gx = np.sum(W * pre_vals, axis=1, keepdims=True)
        return a / x_j.shape[0] * gx  # classic normalizes by N

    elif cfun_name == "Difference":
        a = cfun_params['a']
        x_i_flat = x_i[:, 0] if x_i.ndim > 1 else x_i
        x_j_flat = x_j[:, 0] if x_j.ndim > 1 else x_j
        diff = x_j_flat[np.newaxis, :] - x_i_flat[:, np.newaxis]
        gx = np.sum(W * diff, axis=1, keepdims=True)
        return a * gx

    elif cfun_name == "SigmoidalJansenRit":
        pre_vals = sigmoidal_jr_pre(x_j, **{k: cfun_params[k] for k in ['a','e0','r','v0']})
        gx = W @ pre_vals
        return gx  # post is identity

    raise ValueError(f"Unknown cfun: {cfun_name}")


# ========================================================================
# Pipeline: TVB Hybrid (FIXED) & Hyburn — scale * post(W @ pre(x_j))
# ========================================================================

def pipeline_hybrid_fixed(W, x_j, cfun_name, cfun_params, scale=1.0):
    """Fixed hybrid and hyburn: scale * post(W @ pre(x_j))."""
    if cfun_name == "Linear":
        pre_vals = x_j  # identity
        gx = W @ pre_vals
        return scale * linear_post(gx, cfun_params['a'], cfun_params['b'])

    elif cfun_name == "Sigmoidal":
        pre_vals = x_j  # identity
        gx = W @ pre_vals
        return scale * sigmoidal_post(gx, cfun_params['cmin'], cfun_params['cmax'], cfun_params['midpoint'], cfun_params['a'], cfun_params['sigma'])

    elif cfun_name == "HyperbolicTangent":
        pre_vals = tanh_pre(x_j, **{k: cfun_params[k] for k in ['a','b','midpoint','sigma']})
        gx = W @ pre_vals
        return scale * gx  # post is identity

    elif cfun_name == "Kuramoto":
        a = cfun_params['a']
        pre_vals = np.sin(x_j)  # hybrid: sin(x_j), not sin(x_j - x_i)
        gx = W @ pre_vals
        return scale * (a * gx)  # post is a*gx

    elif cfun_name == "Difference":
        a = cfun_params['a']
        pre_vals = x_j  # identity pre (no x_i access)
        gx = W @ pre_vals
        return scale * (a * gx)

    elif cfun_name == "SigmoidalJansenRit":
        pre_vals = sigmoidal_jr_pre(x_j, **{k: cfun_params[k] for k in ['a','e0','r','v0']})
        gx = W @ pre_vals
        return scale * gx  # post is identity

    raise ValueError(f"Unknown cfun: {cfun_name}")


# Alias: hyburn pipeline for non-x_i functions IS the fixed hybrid pipeline
pipeline_hybrid_fixed_no_xi = pipeline_hybrid_fixed


def pipeline_hyburn(W, x_i, x_j, cfun_name, cfun_params):
    """Hyburn pipeline: post_with_target(W @ pre(x_j), x_i).
    
    For Kuramoto and Difference, uses x_i in post_with_target to match classic.
    For all others, delegates to the fixed hybrid pipeline (no x_i needed).
    """
    if cfun_name == "Kuramoto":
        a = cfun_params['a']
        n_src = x_j.shape[0]
        sin_x = np.sin(x_j)  # pre channel 1
        cos_x = np.cos(x_j)  # pre channel 2
        pre_vals = np.concatenate([sin_x, cos_x], axis=1)  # [nsrc, 2*ncvar]
        weighted_sum = W @ pre_vals  # [ntgt, 2*ncvar]
        ncvar = x_i.shape[1]
        gx_sin = weighted_sum[:, :ncvar]
        gx_cos = weighted_sum[:, ncvar:]
        result = np.cos(x_i) * gx_sin - np.sin(x_i) * gx_cos
        return (a / n_src) * result
    
    elif cfun_name == "Difference":
        a = cfun_params['a']
        gx = W @ x_j  # [ntgt, ncvar]
        rowsums = W.sum(axis=1, keepdims=True)  # [ntgt, 1]
        return a * (gx - x_i * rowsums)
    
    else:
        return pipeline_hybrid_fixed_no_xi(W, x_j, cfun_name, cfun_params)


# ========================================================================
# Comparison
# ========================================================================

def compare_all(cfun_name, cfun_params, W, x_i, x_j, tol=1e-6):
    r_classic = pipeline_classic(W, x_i, x_j, cfun_name, cfun_params)
    r_hybrid_fixed = pipeline_hybrid_fixed(W, x_j, cfun_name, cfun_params)
    r_hyburn = pipeline_hyburn(W, x_i, x_j, cfun_name, cfun_params)

    diff_fixed_vs_classic = np.max(np.abs(r_hybrid_fixed - r_classic))
    diff_hyburn_vs_classic = np.max(np.abs(r_hyburn - r_classic))
    diff_hyburn_vs_fixed = np.max(np.abs(r_hyburn - r_hybrid_fixed))

    return {
        'classic': r_classic,
        'hybrid_fixed': r_hybrid_fixed,
        'hyburn': r_hyburn,
        'diff_fixed_vs_classic': diff_fixed_vs_classic,
        'diff_hyburn_vs_classic': diff_hyburn_vs_classic,
        'diff_hyburn_vs_fixed': diff_hyburn_vs_fixed,
        'agree_fixed_classic': diff_fixed_vs_classic < tol,
        'agree_hyburn_classic': diff_hyburn_vs_classic < tol,
        'agree_hyburn_fixed': diff_hyburn_vs_fixed < tol,
    }


def main():
    np.random.seed(42)

    n_nodes = 4
    W = np.array([
        [0.0, 0.2, 0.3, 0.1],
        [0.2, 0.0, 0.1, 0.3],
        [0.3, 0.1, 0.0, 0.2],
        [0.1, 0.3, 0.2, 0.0],
    ], dtype=np.float32)

    x_j = np.random.randn(n_nodes, 1).astype(np.float32) * 2
    x_i = np.random.randn(n_nodes, 1).astype(np.float32) * 2
    x_j_2cvar = np.random.randn(n_nodes, 2).astype(np.float32) * 2

    tests = [
        ("Linear_b0", "Linear", {'a': 0.004, 'b': 0.0}, x_j),
        ("Linear_b0.1", "Linear", {'a': 0.004, 'b': 0.1}, x_j),
        ("Sigmoidal", "Sigmoidal", {'cmin': -1.0, 'cmax': 1.0, 'midpoint': 0.0, 'a': 1.0, 'sigma': 1.0}, x_j),
        ("Tanh", "HyperbolicTangent", {'a': 1.0, 'b': 1.0, 'midpoint': 0.0, 'sigma': 1.0}, x_j),
        ("Kuramoto", "Kuramoto", {'a': 1.0}, x_j),
        ("Difference", "Difference", {'a': 0.1}, x_j),
        ("SigmoidalJansenRit", "SigmoidalJansenRit", {'a': 1.0, 'e0': 0.005, 'r': 0.56, 'v0': 6.0}, x_j_2cvar),
    ]

    results = {}
    print("=" * 80)
    print("COUPLING SEMANTICS VERIFICATION (after fix)")
    print("=" * 80)

    for name, cfun_name, params, xj in tests:
        xi = x_i if xj.shape == x_i.shape else x_i[:, :xj.shape[1]]
        r = compare_all(cfun_name, params, W, xi, xj)

        print(f"\n--- {name} ---")
        print(f"  fixed_hybrid vs classic:  {'AGREE' if r['agree_fixed_classic'] else 'DISAGREE'} (max_diff={r['diff_fixed_vs_classic']:.2e})")
        print(f"  hyburn       vs classic:  {'AGREE' if r['agree_hyburn_classic'] else 'DISAGREE'} (max_diff={r['diff_hyburn_vs_classic']:.2e})")
        print(f"  hyburn       vs fixed:    {'AGREE' if r['agree_hyburn_fixed'] else 'DISAGREE'} (max_diff={r['diff_hyburn_vs_fixed']:.2e})")
        results[name] = r

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Function':<25} {'Fixed=Classic':>14} {'Hyburn=Classic':>16} {'Hyburn=Fixed':>14}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<25} {'✅' if r['agree_fixed_classic'] else '❌':>14} {'✅' if r['agree_hyburn_classic'] else '❌':>16} {'✅' if r['agree_hyburn_fixed'] else '❌':>14}")

    # Save traces
    out_dir = os.path.join(os.path.dirname(__file__), "coupling_semantics")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "weights.npy"), W.astype(np.float32))
    np.save(os.path.join(out_dir, "x_j.npy"), x_j.astype(np.float32))
    np.save(os.path.join(out_dir, "x_i.npy"), x_i.astype(np.float32))
    np.save(os.path.join(out_dir, "x_j_2cvar.npy"), x_j_2cvar.astype(np.float32))

    for name, r in results.items():
        np.save(os.path.join(out_dir, f"{name}_classic.npy"), r['classic'].astype(np.float32))
        np.save(os.path.join(out_dir, f"{name}_hybrid_fixed.npy"), r['hybrid_fixed'].astype(np.float32))
        np.save(os.path.join(out_dir, f"{name}_hyburn.npy"), r['hyburn'].astype(np.float32))

    comparison = {}
    for name, r in results.items():
        comparison[name] = {
            'diff_fixed_vs_classic': float(r['diff_fixed_vs_classic']),
            'diff_hyburn_vs_classic': float(r['diff_hyburn_vs_classic']),
            'diff_hyburn_vs_fixed': float(r['diff_hyburn_vs_fixed']),
            'agree_fixed_classic': bool(r['agree_fixed_classic']),
            'agree_hyburn_classic': bool(r['agree_hyburn_classic']),
            'agree_hyburn_fixed': bool(r['agree_hyburn_fixed']),
        }
    with open(os.path.join(out_dir, "comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nSaved {len(results)} comparison sets to {out_dir}/")


if __name__ == "__main__":
    main()
