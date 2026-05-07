# Reference Test Fixtures for `hyburn`

This directory contains `.npy` files that serve as **ground-truth reference
data** for validating the Rust/Burn hybrid simulator against the original
Python TVB classic simulator.

## Regenerating all fixtures

From the repository root:

```bash
python generate_reference.py
```

Requirements:
- `tvb-hybrid-numba` (sibling checkout) must be importable.
- Standard scientific Python stack: `numpy`, `scipy`.

## Parameters used

| Parameter | Value | Notes |
|---|---|---|
| Model | `Generic2dOscillator` | Classic TVB, **full** parameter set |
| Integrator | `HeunDeterministic` | Predictor-corrector (modified trapezoidal) |
| `dt` | **0.1 ms** | 1000 steps for 100 ms |
| Total time | **100.0 ms** | |
| Nodes | **2** | Fully connected undirected pair |
| Weights | `[[0, 1], [1, 0]]` | Unit weight |
| Tract lengths | All **0.0** | Eliminates delays for deterministic comparison |
| Coupling | `Linear(a=0.004)` | Affine coupling with zero offset |
| Monitor | `Raw` | Every integration step is recorded |
| RNG seed | **42** | For reproducible initial conditions |

### Generic2dOscillator full Python default parameters

All values are scalar `numpy.array([x])`:

| Parameter | Default | Description |
|---|---|---|
| `tau`   | 1.0   | Time-scale hierarchy |
| `I`     | 0.0   | Baseline shift of cubic nullcline |
| `a`     | -2.0  | Vertical shift of configurable nullcline |
| `b`     | -10.0 | Linear slope of configurable nullcline |
| `c`     | 0.0   | Parabolic term |
| `d`     | 0.02  | Temporal scale factor |
| `e`     | 3.0   | Quadratic coeff. of cubic nullcline |
| `f`     | 1.0   | Cubic coeff. of cubic nullcline |
| `g`     | 0.0   | Linear coeff. of cubic nullcline |
| `alpha` | 1.0   | Feedback fast->slow scale |
| `beta`  | 1.0   | Feedback slow->slow scale |
| `gamma` | 1.0   | Scales both I_ext and coupling term |

## Output files

| File | Shape | Description |
|---|---|---|
| `g2do_2node_heun_dt01_100ms_traj.npy`   | `(1000, 2, 2, 1)` | Full state trajectory `[step, var, node, mode]` |
| `g2do_2node_heun_dt01_100ms_times.npy`  | `(1000,)`         | Time stamp for each step `[ms]` |
| `g2do_2node_heun_dt01_100ms_ic.npy`     | `(2, 2, 1)`       | Initial state after `configure()` `[var, node, mode]` |
| `g2do_2node_weights.npy`                | `(2, 2)`          | Connectivity weight matrix |

All arrays are `float64`.

## Differences from `BURN.md` / Rust parameterisation

The Rust `hyburn` code uses a **simplified** G2DO formulation:

```
dV/dt = tau * (V - V^3/3 - W + I_ext + coupling[:,0])
dW/dt = (V - a*W + b) / tau
Parameters: [tau, I_ext, a, b]
```

This is a **subset** of the full Python model.  To compare the Rust output
against these reference arrays you must either:

1. **Map** the Rust parameters to the Python equivalents, or
2. **Align** the Python model to the reduced form by setting:
   - `d=1.0`, `e=0.0`, `f=1/3`, `g=1.0`, `alpha=1.0`, `beta=1.0`, `gamma=1.0`
   - and then adjusting `a`, `b`, `I` accordingly.

### Key gotchas

- **Coupling sign**: in the full Python model the coupling term is multiplied
  by `gamma`.  With default `gamma=1.0` the sign is positive; if you set
  `gamma=-1.0` the coupling is inverted.
- **Initial conditions**: TVB's classic simulator draws random ICs from the
  model's `state_variable_range` during `configure()` (seed=42 here).  The Rust
  code should use the saved `ic.npy` file directly to avoid any mismatch.
- **Variable ordering**: state variables are ordered `(V, W)` — index 0 is V,
  index 1 is W.  `cvar=[0]` means only `V` participates in coupling.
- **dt / step count**: `HeunDeterministic` is a predictor-corrector.  With
  `dt=0.1` and `simulation_length=100.0` the classic simulator produces exactly
  1000 steps (Raw monitor records every step).
- **Node layout**: for 2 nodes the connectivity weights are dense `[2, 2]`.
  In hybrid TVB the coupling is computed as `weights @ delayed_state`, yielding
  shape `[ntgt, ncvar]`.
