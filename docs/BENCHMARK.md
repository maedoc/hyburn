# hyburn Benchmark Results

## Test Machine
- **CPU**: 16-core (Ryzen/EPYC-class)
- **GPU**: NVIDIA GeForce RTX 4090 (Compute 8.9)
- **Rust**: 1.94, `--release` (LTO, opt-level=3)
- **Burn**: 0.16 (NdArray, WGPU, CUDA backends)

## Realistic 3-Subnetwork Sweep (1024 points × 1000 steps)

**Network**: G2DO (2 vars) → JansenRit (6 vars) → WilsonCowan (2 vars) — coupled ring  
**Coupling**: All-to-all scalar Linear, weight=0.01  
**Sweep**: G2DO I_ext over 1024 points, dt=0.1ms  
**Integration**: Heun for G2DO, Euler for JR/WC (matching Numba reference)

### 76 nodes/subnet (760 total state vars)

| Method | Time | Per-point | vs Numba CUDA |
|--------|------|-----------|---------------|
| **Hyburn CUDA (generic, hybrid)** | **360 ms** | **0.35 ms** | **1.14×** ⚡ |
| **Hyburn CUDA (hardcoded)** | **379 ms** | **0.37 ms** | **1.08×** |
| **Hyburn WGPU batch** | **505 ms** | **0.49 ms** | **0.8×** |
| **Numba CUDA (RTX 4090)** | **413 ms** | **0.40 ms** | **1.0×** |
| Hyburn Rayon (16 cores) | 8,923 ms | 8.71 ms | 0.05× (21× slower) |
| Python TVB (1 core) | 73,059 ms | 71.3 ms | 0.006× |
| Hyburn NdArray batch | 52,644 ms | 51.4 ms | 0.07× |
| Hyburn NdArray serial | 139,927 ms | 136.6 ms | 0.003× |
| Hyburn WGPU serial | — | 4,774 ms* | 0.00008× |

### 164 nodes/subnet (1,640 total state vars)

| Method | Time | Per-point | vs Numba CUDA |
|--------|------|-----------|---------------|
| **Hyburn CUDA (generic, hybrid)** | **341 ms** | **0.33 ms** | **2.6×** ⚡ |
| **Hyburn CUDA (hardcoded)** | **371 ms** | **0.36 ms** | **2.4×** |
| **Hyburn WGPU batch** | **497 ms** | **0.49 ms** | **1.8×** |
| **Numba CUDA (RTX 4090)** | **893 ms** | **0.87 ms** | **1.0×** |
| Hyburn Rayon (16 cores) | 18,360 ms | 17.9 ms | 0.05× (20× slower) |

### Speedup Summary

| Comparison | 76 nodes | 164 nodes |
|-----------|----------|-----------|
| Hyburn CUDA (generic, hybrid) vs Numba CUDA | **1.14×** (faster!) | **2.6×** (faster!) |
| Hyburn CUDA (hardcoded) vs Numba CUDA | **1.08×** (matched!) | **2.4×** (faster!) |
| Hyburn WGPU batch vs Numba CUDA | **0.8×** | **1.8×** |
| Hyburn CUDA vs NdArray serial | **388×** | **411×** |
| Hyburn CUDA vs Rayon (16 cores) | **25×** | **54×** |

### Single-Model Benchmark (CUDA, 256 pts × 76 nodes × 1000 steps)

| Model | ms/pt | NVAR | Notes |
|-------|-------|------|-------|
| G2DO | 0.28 | 2 | Faster than Numba! |
| WilsonCowan | 0.35 | 2 | |
| MPR | 0.37 | 2 | |
| JansenRit | 0.41 | 6 | |
| Kuramoto | ~0.30 | 1 | |

### Generic BatchHybridEngine vs Hardcoded

With the hybrid integrator (Heun for G2DO, Euler for JR/WC), the generic engine now matches
or exceeds the hardcoded 3-subnet kernel. The generic engine handles any `SimConfig` without code changes.

| Config | Hardcoded | Generic (hybrid) | Ratio |
|--------|-----------|-------------------|-------|
| 3-subnet ring, 76 nodes × 1024 pts | 0.37 ms/pt | 0.35 ms/pt | **0.95×** (faster!) |
| 3-subnet ring, 164 nodes × 1024 pts | 0.36 ms/pt | 0.33 ms/pt | **0.92×** (faster!) |
| Single G2DO, 76 nodes × 256 pts | — | 0.28 ms/pt | — |

---

## Burn GPU Backend Analysis

### Per-step overhead dominated old approach

| Backend | 76 nodes (20 pts × 1000 steps) | Per-point | vs NdArray |
|---------|-------------------------------|-----------|------------|
| NdArray serial | 2,721 ms | 136 ms | 1.0× |
| **WGPU serial** | **95,475 ms** | **4,774 ms** | **0.03× (35× slower!)** |
| **CUDA serial** | **14,772 ms** | **739 ms** | **0.18× (5.5× slower!)** |

### Batch-dim approach eliminates the overhead

| Approach | Kernel launches (total) | Performance |
|----------|----------------------|-------------|
| Numba CUDA (`@cuda.jit`) | 1 (entire simulation) | **0.40 ms/pt** |
| **Burn CUDA (generic, hybrid)** | ~25,000 | **0.35 ms/pt** |
| **Burn CUDA (hardcoded)** | ~30,000 | **0.37 ms/pt** |
| **Burn WGPU batch** | ~30,000 | **0.49 ms/pt** |
| Burn WGPU (serial, per-point) | ~30,000×1024 | 4,774 ms/pt |

---

## Simple Single-Subnet Sweep (20 points × 1000 steps)

| Nodes | NdArray Serial | NdArray Rayon | Rayon Speedup |
|------:|--------------:|-------------:|---------------:|
|     2 |        153 ms |       16 ms  |         9.4×  |
|    34 |        282 ms |       30 ms  |         9.2×  |
|   256 |       1208 ms |      128 ms  |         9.4×  |

---

## Architecture

| Component | Description |
|-----------|-------------|
| `engine::batch_gpu::BatchHybridEngine` | **Generic batch-dim engine** — 0.35 ms/pt (any SimConfig) |
| `engine::sweep_gpu::batch_sweep_3subnet()` | Hardcoded 3-subnet batch (reference) |
| `engine::sweep::parallel_sweep()` | `rayon::par_iter` — 9-16× on 16 cores |
| `engine::Projection.cvar_map` | Cross-model coupling routing |
| `engine::sparse` | CSR sparse coupling with per-edge delays |