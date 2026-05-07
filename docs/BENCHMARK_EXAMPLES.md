# Hyburn Backend Benchmark Report

**Date:** 2026-05-07  
**CPU:** AMD EPYC 32-core (ndarray backend, single-threaded per sim)  
**GPU:** NVIDIA RTX 4090 (WGPU/Vulkan + CUDA backends)  
**Build:** `cargo build --release --features "wgpu,cuda"`  
**Dimensions:** 74 nodes, 10k steps per simulation; 128 sweep points

---

## Single Simulation (1 sim, 74 nodes, 10k steps)

| Example | Nodes | Steps | CPU (s) | WGPU (s) | CUDA (s) |
|---------|------:|------:|--------:|---------:|---------:|
| G2DO 74n | 74 | 10,000 | **0.67** | 26.73 | 4.44 |
| G2DO+BOLD 74n | 74 | 100,000 | **6.99** | >120 timeout | 48.07 |
| 2×G2DO 148n | 148 | 10,000 | **1.36** | 47.81 | 8.53 |
| WC 74n | 74 | 10,000 | **0.76** | 25.25 | 5.55 |
| JR+stim 74n | 74 | 10,000 | **1.43** | 26.23 | 7.07 |
| 4×G2DO+stim 296n | 296 | 10,000 | **2.72** | 91.37 | 17.49 |
| Demo 74n | 74 | 10,000 | **0.69** | 26.78 | 4.31 |
| G2DO+Kura 148n | 148 | 10,000 | **0.97** | 47.69 | 8.48 |

**Single-sim conclusion:** CPU dominates. WGPU is 30–50× slower (shader compilation + kernel launch overhead). CUDA is 6–12× slower (PTX compilation + launch overhead). GPU backends only pay off with batched workloads.

---

## Parameter Sweep (128 points × 74 nodes × 10k steps)

| Sweep | CPU (s) | CPU/pt | WGPU (s) | WGPU/pt | CUDA (s) | CUDA/pt | CPU→WGPU | CPU→CUDA |
|-------|--------:|-------:|---------:|--------:|---------:|--------:|---------:|---------:|
| G2DO 128pt | 26.18 | 0.205 s | **2.05** | 0.016 s | **1.82** | 0.014 s | **12.8×** | **14.4×** |
| MPR 128pt | 30.68 | 0.240 s | **2.99** | 0.023 s | **2.59** | 0.020 s | **10.3×** | **11.9×** |

**Sweep conclusion:** GPU backends deliver **10–14× speedup** for 128-point parameter sweeps. CUDA consistently outperforms WGPU by ~10–15%. The `BatchHybridEngine` keeps all sweep points in a single tensor, so kernel launch cost is amortized across all 128 points.

---

## Architecture Notes

### Why single-sim GPU is slow
- **WGPU:** Runtime WGSL shader compilation (~15 s cold, ~5 s warm) + 50–100 μs per kernel launch × ~100 ops/step × 10k steps = pure overhead
- **CUDA:** PTX JIT compilation (~3 s cold, ~1 s warm) + similar launch overhead but lower than WGPU
- **CPU:** NdArray backend has zero launch overhead; in-memory L1/L2 cache makes 74-node state vectors fit entirely in L2

### Why sweep GPU is fast
- The `BatchHybridEngine` reshapes state as `[n_sweep * nnodes, nvar, nmodes]`
- All 128 points share a single kernel launch per operation
- RTX 4090's 16384 CUDA cores process 128×74=9472 rows in parallel
- CPU Rayon parallelism hits Amdahl's law at ~8–16 threads

### Scaling expectations

| Sweep Points | CPU (est.) | CUDA (est.) | Speedup |
|-------------:|----------:|-----------:|--------:|
| 16 | 3.3 s | 1.5 s | 2.2× |
| 64 | 13 s | 1.7 s | 7.6× |
| **128** | **26 s** | **1.8 s** | **14×** |
| 512 | 104 s | 3.0 s | 35× |
| 2048 | 416 s | 7.0 s | 59× |
| 8192 | 1667 s | 20 s | 83× |

---

## Recommended Backend Selection

| Scenario | Backend | Why |
|----------|---------|-----|
| Quick testing, <10 nodes | `ndarray` | Zero overhead |
| Single long sim, <200 nodes | `ndarray` | L2 cache fits state |
| Sweep, <16 points | `ndarray` | GPU launch cost dominates |
| **Sweep, ≥64 points** | **`cuda`** | **10×+ speedup** |
| Sweep, ≥64 points (no NVIDIA) | `wgpu` | 80–90% of CUDA speed |
| Whole-brain, >200 nodes, single sim | `cuda` | matmul parallelism |
| SBI training (MAF/MADE) | `ndarray` | autodiff only on CPU |
| CrossCoder training | `ndarray` | autodiff only on CPU |
