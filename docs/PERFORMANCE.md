# Performance Guide

This document covers backend selection, autotuning, and parameter sweep usage for `hyburn`.

## Backend Selection

`hyburn` supports multiple Burn backends. The desired backend is selected via the `--backend` CLI flag.

| Backend   | Flag       | Hardware | Feature gate | Notes |
|-----------|------------|----------|--------------|-------|
| NdArray   | `ndarray`  | CPU      | Always on    | Default, most portable. |
| WGPU      | `wgpu`     | GPU      | `wgpu`       | Cross-platform GPU via WebGPU. |
| CUDA      | `cuda`     | NVIDIA   | `cuda`       | Requires CUDA 12.x toolkit. |

### Examples

```bash
# CPU (default)
hyburn run --config sim.toml --backend ndarray

# GPU via WGPU (compile with --features wgpu)
cargo run --features wgpu -- run --config sim.toml --backend wgpu

# NVIDIA GPU via CUDA (compile with --features cuda)
cargo run --features cuda -- run --config sim.toml --backend cuda
```

If a requested backend is not compiled in (e.g. `cuda` without the `cuda` feature), the simulator automatically falls back to `ndarray` with a warning.

### Selecting the Right Backend

* **NdArray** is the safest default and works everywhere. Use it for development, debugging, and small networks (< 500 nodes).
* **WGPU** shines on medium-to-large networks (500–2000+ nodes) on any modern GPU (NVIDIA, AMD, Intel, Apple Metal). It compiles to native GPU code via WGSL or SPIR-V.
* **CUDA** offers the highest throughput on NVIDIA hardware, especially when kernel fusion and TensorCores are active. Enable it for production runs on large networks.

## Expected Performance Characteristics

Performance is dominated by three factors:

1. **Coupling computation** — dense `matmul` for small networks, sparse CSR for medium, tiled GPU kernels for large.
2. **Integration step** — the `dfun` evaluation per node per step.
3. **History / delay buffer** — memory bandwidth for delayed state lookups.

| Network size | Recommended strategy | Typical relative speed |
|--------------|---------------------|------------------------|
| < 500 nodes  | Dense CPU           | 1× (baseline)          |
| 500–2000     | Sparse CSR CPU      | 2–5× over dense        |
| > 2000       | Tiled GPU (WGPU/CUDA) | 5–20× over dense       |

These are rough estimates; actual speedups depend on GPU model, sparsity, and integration step size.

## Autotuning

The built-in autotuner benchmarks Dense vs SparseCSR coupling on a synthetic network matching your configuration's node count and recommends the fastest strategy.

### Running autotune

```bash
hyburn autotune --config sim.toml
```

Output looks like:

```
AutotuneResult {
  optimal_strategy: SparseCSR,
  optimal_block_size: 128,
  benchmark_time_ns: 1523400,
}
```

You can use the reported `optimal_strategy` to inform engine configuration, or simply trust the heuristics used automatically by the simulator.

## Parameter Sweep Mode

Parameter sweeps let you run embarrassingly parallel simulations across a grid of parameter values, producing one `.npy` output per point.

### Sweep configuration (TOML)

```toml
parameter_name = "subnetworks[0].params[2]"
values = [1.0, 1.5, 2.0, 2.5, 3.0]
```

Or using a range:

```toml
parameter_name = "dt"

[range]
start = 0.01
step  = 0.01
end   = 0.1
```

Supported `parameter_name` formats:
* `dt` — integration step size
* `nsig` — noise amplitude
* `subnetworks[0].params[2]` — any param inside a subnetwork

### Running a sweep

```bash
hyburn run --config sim.toml --sweep sweep.toml --output sweep_results
```

This creates subdirectories:

```
sweep_results/
  sweep_0/
    state_final_sub0.npy
    state_traj.npy
  sweep_1/
    ...
```

Each subdirectory is a fully independent simulation. Checkpoints and resume are disabled during sweep mode.

## Compilation Tips

* **Release builds** are strongly recommended for performance work:
  ```bash
  cargo build --release
  ```
* To compile without any GPU dependencies (e.g. for a headless server):
  ```bash
  cargo build --no-default-features
  ```
* Enable autotuning and fusion where available:
  ```bash
  cargo run --features "wgpu" -- run ...
  ```
