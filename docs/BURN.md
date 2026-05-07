# Burn-Based GPU Hybrid Simulator + SBI: Feasibility Assessment

**Date:** 2026-05-04
**Status:** Technical feasibility review
**Author:** pi coding agent (coordinated parallel subagent review)

---

## Executive Summary

**Verdict: Feasible, but a major engineering undertaking (est. 14–19 person-weeks).**

Building a standalone Rust CLI that replaces the Python/Numba TVB hybrid simulator and smolcde with a Burn-backed, GPU-accelerated system is **technically feasible**. Burn provides the necessary tensor infrastructure, autodiff, GPU backends, and autotuning. However, three non-trivial challenges dominate:

1. **Sparse CSR coupling with per-edge delays** — Burn has no sparse tensor primitives; this requires custom CubeCL/WGSL kernels or a dense fallback.
2. **Runtime code generation replacement** — The Python simulator dynamically generates Numba kernels from model metadata via Mako templates. Rust must replace this with compile-time codegen, a DSL, or trait-object dispatch.
3. **24+ neural mass models** with varying state/coupling variable counts, each with model-specific boundary conditions and dfuns — requires careful Rust type-system design.

The smolcde/SBI side is comparatively easy (2–3 weeks) and benefits directly from Burn's autodiff, eliminating the hand-written C backward pass.

---

## 1. Component Review Summaries

### 1.1 Hybrid Simulator (Python/Numba)

**Reference codebase:** `../tvb-hybrid-numba/tvb_library/tvb/simulator/hybrid/`
**Reference backend:** `../tvb-hybrid-numba/tvb_library/tvb/simulator/backend/nb_hybrid.py`
**Reference tests:** 18 test files under `../tvb-hybrid-numba/tvb_library/tvb/tests/library/simulator/hybrid/`

#### Core Architecture

```
User API (Simulator) → NetworkSet (topology) → NbHybridBackend (compile) → Generated Numba kernel
```

1. **NetworkSet** — A container of `Subnetwork` objects connected by `InterProjection`/`IntraProjection` edges, with optional `Stimulus` inputs.
2. **NbHybridBackend** — Flattens the object graph into `NetworkAnalysis`/`SubnetworkInfo`/`ProjectionInfo` dataclasses, then renders a Mako template (`nb-hybrid-sim.py.mako`) into a self-contained Python/Numba module.
3. **Generated kernel** — A `run_network()` function calling per-projection `compute_coupling_*()` and per-subnetwork `integrate_*()` subroutines, all `@nb.njit(inline="always")`.

#### Computational Kernels (what actually runs)

| Kernel | Operation | Complexity |
|--------|-----------|------------|
| **CSR sparse coupling** | For each target node, iterate CSR row, read delayed source state from history buffer, accumulate weighted sum | O(nnz) scalar, no BLAS |
| **Coupling function** | Apply `pre()`, scale, `post()` to each coupled input (8 functions: Linear, Sigmoidal, Kuramoto, Difference, etc.) | O(nedges) scalar |
| **cvar mapping** | Map coupling variables to state variables (1:1, many:1, 1:many, n:n patterns) | O(nnodes × nmodes) |
| **dfun** | Per-model scalar derivative expressions (e.g., MPR: `dS/dt`, `dZ/dt`, `dr/dt`) | O(nnodes × nmodes × nvar) |
| **Heun integration** | Predictor-corrector: two dfun calls + midpoint clamping | 2× dfun cost |
| **Boundary clamping** | Model-specific inline constraints (e.g., MPR `r >= 0`) | O(nnodes) |
| **Buffer write** | One write per source subnetwork per step to circular history buffer | O(nvar × nnodes × nmodes) |
| **Temporal average** | Running accumulation + periodic divide (in Python post-chunk) | O(1) per step |

#### Key Data Structures

| Structure | Shape | Dtype |
|-----------|-------|-------|
| Per-subnetwork state | `(nvar, nnodes, nmodes)` | float32 (Numba path) |
| Coupling variables (accumulator) | `(ncvar, nnodes, nmodes)` | float32 |
| History buffer (per source subnet) | `(nvar, nnodes, nmodes, horizon)` | float32 |
| CSR weights | `(nnz,)` | float32 |
| CSR indices | `(nnz,)` | int32 |
| CSR indptr | `(ntgt + 1,)` | int32 |
| Per-edge delays | `(nnz,)` | int32 (in steps) |
| Stimulus | `(ncvar, nnodes, nmodes, nstep)` | float32 |
| Noise (stochastic) | `(nvar, nnodes, nmodes, nstep)` | float32 |

#### Integration Schemes
- **HeunDeterministic** — Predictor-corrector, midpoint clamping
- **EulerDeterministic** — Forward Euler
- **HeunStochastic / EulerStochastic** — Additive noise only (multiplicative raises NotImplementedError)
- All subnetworks must share identical `dt`

#### CUDA Sweep Backend (Proof-of-Concept)
- Standalone benchmark in `nb_hybrid_cuda_sweep.py`
- One CUDA thread per sweep point, all state/history in device memory
- Not integrated into the main simulator pipeline
- Demonstrates embarrassingly-parallel parameter sweeps work on GPU

#### Test Coverage
18 test files covering:
- Simulator integration (end-to-end runs, initial conditions)
- Network assembly (projection creation, topology validation)
- Subnetwork (model configuration, state initialization)
- Coupling functions (all 8 types, parameter validation)
- Toy synchronization (analytical benchmarks)
- Recorder (history buffer correctness)
- Boundary conditions (integrator clamping)
- Stimulus injection
- MPR/Kionex model-specific validation
- Steady-state validation
- Two-subnetwork scenarios

### 1.2 smolcde — Conditional Density Estimation via MAF

**Reference codebase:** `../smolcde/`
**Language:** C (with Python reference implementation)

#### Algorithm: Masked Autoregressive Flow (MAF)

smolcde learns `p(parameters | features)` — the conditional posterior density of simulation parameters given observed features. This is **simulation-based inference (SBI)**.

**Forward pass (training — density estimation):**
```
For each of K flow layers:
  1. Permute parameters x → x_perm
  2. MADE block: compute mu, alpha = f(x_perm, context_features)
     - h = tanh(x_perm @ W1y·M1^T + ctx @ W1c^T + b1)
     - out = h @ W2·M2^T + ctx @ W2c^T + b2
     - split out → mu, alpha; clip alpha ∈ [-7, 7]
  3. Transform: u = (x_perm - mu) * exp(-alpha)
  4. Accumulate: log_det += sum(-alpha)
After all layers: u ~ N(0,I)
Log-prob = N(u; 0, I) + log_det
Loss = -mean(log-prob)
```

**Inverse pass (inference — sampling):**
```
Draw z ~ N(0, I)
For each layer (reversed):
  For each dimension i sequentially (autoregressive):
    1. Compute mu_i, alpha_i from partial u[:i] using MADE
    2. u[i] = z_perm[i] * exp(alpha_i) + mu_i
  Apply inverse permutation
Return samples
```

#### Architecture Parameters

| Config | D (param dim) | C (feature dim) | H (hidden) | Blocks | Typical use |
|--------|--------------|-----------------|------------|--------|-------------|
| Lorenz | 3 | 65 | 64 | 4 | Chaotic attractor params |
| MNIST | 10 | 784 | 256 | 6 | Digit classification |

#### Key Tensor Shapes (C implementation)

- Weights: `W1y [H,D]`, `W1c [H,C]`, `W2 [2D,H]`, `W2c [2D,C]`
- Biases: `b1 [H]`, `b2 [2D]`
- Masks: `M1 [H,D]`, `M2 [D,H]` (binary)
- Activations: `h [H,8]`, `out [2D,8]`, `u [D,8]`
- Dtype: `f32` throughout
- Hardcoded internal batch size: 8 (C micro-optimization, not algorithmic)

#### Dominant Compute Patterns
- Small dense matmuls (triple-nested C loops, no BLAS)
- Elementwise: `tanh`, `exp`, `clip`, binary masking
- Sequential autoregressive loop in sampling (main bottleneck)
- Adam optimizer (hand-written)
- Hand-written backward pass (~200 lines of C)

### 1.3 Burn ML Framework

**Website:** https://burn.dev
**Version:** Stable releases (v0.16.x series as of early 2026)

#### Capabilities Assessment

| Capability | Rating | Notes |
|-----------|--------|-------|
| Tensor API | ★★★★★ | Full-featured: reshape, slice, matmul, elemwise, reduce, broadcast |
| GPU Backends | ★★★★☆ | CUDA, WGPU (Vulkan/Metal/DX12), ROCm — all production-grade |
| Autodiff | ★★★★★ | Full support, wraps any backend |
| Autotuning | ★★★★☆ | Built-in for matmul/reduce; caches per-device; extensible |
| NN Modules | ★★★★☆ | Linear, Conv, LSTM, Transformer, Norm, Dropout, activations |
| Serialization | ★★★★☆ | MessagePack, Bincode, JSON, SafeTensors, PyTorch import |
| Data Loading | ★★★★☆ | Dataset trait, CSV import, Polars integration, batchers |
| Custom Kernels | ★★★☆☆ | CubeCL for WGSL/CUDA/SPIR-V kernels; API evolving |
| Distributed | ★★☆☆☆ | Remote backend (beta); no MPI/NCCL |
| Normalizing Flows | ★☆☆☆☆ | No built-in MAF/flow library; must build from primitives |
| Sparse Tensors | ★☆☆☆☆ | No CSR/CSC primitives; must be custom |

#### Backend Selection for This Project

| Backend | Recommendation | Rationale |
|---------|---------------|-----------|
| **WGPU** | Primary target | Cross-platform (Vulkan/Metal/DX12/WebGPU); no NVIDIA lock-in; autotuning support |
| **CUDA** | Performance target | NVIDIA HPC users; lower overhead; CUBLAS integration |
| **NdArray** | Fallback/dev | CPU-only; useful for CI and development without GPU |

---

## 2. Feasibility Matrix

### 2.1 What Maps Cleanly

| Component | Burn Mapping | Confidence |
|-----------|-------------|------------|
| **State tensors** (nvar × nnodes × nmodes) | `Tensor<B, 3, Float>` | High |
| **History buffers** (nvar × nnodes × nmodes × horizon) | `Tensor<B, 4, Float>` with circular index slicing | High |
| **Dense matmul** (for MAF MADE blocks) | `tensor.matmul(other)` | High |
| **Elementwise ops** (tanh, exp, clip, ReLU) | Built-in tensor ops | High |
| **Heun/Euler integration** | Custom loop with tensor ops | High |
| **Coupling functions** (Linear, Sigmoidal, etc.) | Elementwise tensor ops | High |
| **Adam optimizer** | `AdamConfig` in `burn-train` | High |
| **Model serialization** | `record()` / `load_record()` | High |
| **TOML config parsing** | `toml` crate + serde | High |
| **NPY/CSV I/O** | `ndarray-npy` + `csv` crates | High |
| **CLI framework** | `clap` derive macros | High |
| **MAF training loop** | Burn `Learner` + custom training step | High |
| **MAF forward pass** | Custom `Module` with Linear layers + masking | High |
| **Autodiff for MAF** | Eliminates hand-written C backward pass | High |

### 2.2 What Requires Custom Work

| Component | Challenge | Approach | Est. Effort |
|-----------|-----------|----------|-------------|
| **CSR sparse coupling** | No sparse tensor primitives in Burn | Option A: Dense matmul (ok for N < 1000 nodes). Option B: Custom CubeCL/WGSL sparse kernel. Option C: CPU CSR via `ndarray` + cross-device transfer. | 1–3 weeks |
| **Per-edge delays** | Circular buffer indexing with variable delays per edge | Custom gather kernel or pre-expanded dense delay tensor | 1 week |
| **Model dfun codegen** | 24+ models with different state variables, equations, parameters | Compile-time codegen via build.rs + proc macros, or trait-based dispatch with match statements | 2–4 weeks |
| **Boundary clamping** | Per-model constraints (MPR `r>=0`, RWW `S∈[0,1]`) | Trait method `clamp(&mut self)` per model | 0.5 week |
| **Monitor post-processing** | ~250 lines of Python branching on monitor type | Rust enum with match arms, one variant per monitor | 1–2 weeks |
| **Noise generation** | Needs to match TVB noise semantics (nsig, additive only) | `rand` crate + `Tensor::from_data` | 0.5 week |
| **MAF autoregressive sampling** | Sequential dimension loop not natural in tensor graphs | Explicit `for` loop in `Module::forward`, recomputing MADE per dimension | 1 week |

### 2.3 What Is Problematic / Architecturally Challenging

| Challenge | Severity | Details |
|-----------|----------|---------|
| **Dynamic code generation (Mako → exec)** | **HIGH** | The Python backend generates Numba source code at runtime from model metadata. In Rust, this must become either: (a) compile-time code generation via `build.rs`, (b) a declarative proc-macro that expands model definitions into kernel code, or (c) runtime trait-object dispatch (slower but simpler). All approaches add complexity. |
| **Heterogeneous subnetwork collections** | **MEDIUM** | `NetworkSet` holds subnetworks with different model types (different `nvar`, `ncvar`, dfun signatures). Rust's type system requires explicit handling via enums, trait objects, or code generation. |
| **Burn's lack of sparse primitives** | **MEDIUM** | The coupling kernel is the innermost loop and dominates runtime for large networks. Dense fallback wastes O(N²) memory/FLOPs. Custom sparse kernels in CubeCL/WGSL are feasible but require GPU programming expertise. |
| **Static vs. dynamic dispatch for dfuns** | **MEDIUM** | The generated Numba code inlines dfun expressions directly (no function call overhead). Trait-object dispatch in Rust adds vtable overhead per node, which matters at scale. Proc-macro codegen can match Numba's performance but is harder to implement. |

---

## 3. Proposed Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Rust CLI (clap)                       │
│  Commands: run, train-sbi, infer, autotune, benchmark   │
└────────────┬────────────────────────────────────────────┘
             │
     ┌───────▼────────┐
     │  Config Layer  │  TOML → typed config structs (serde)
     │  (toml crate)  │
     └───────┬────────┘
             │
     ┌───────▼────────┐     ┌──────────────────┐
     │  IO Layer      │     │                  │
     │  NPY ↔ Tensor  │     │  CSV ↔ Dataset   │
     │  (ndarray-npy) │     │  (csv + Polars)  │
     └───────┬────────┘     └──────────────────┘
             │
     ┌───────▼────────────────────────────────────────────┐
     │                 Burn Backend (WGPU / CUDA)          │
     │                                                      │
     │  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
     │  │ Hybrid Engine │  │  SBI Engine   │  │ Autotune  │ │
     │  │               │  │               │  │           │ │
     │  │ • Sparse CSR  │  │ • MAF Model   │  │ • Block   │ │
     │  │   coupling    │  │ • MADE blocks │  │   size    │ │
     │  │ • Heun/Euler  │  │ • Adam train  │  │ • GPU vs  │ │
     │  │ • Monitors    │  │ • CDE infer   │  │   CPU     │ │
     │  │ • Noise       │  │               │  │ • Kernel  │ │
     │  └──────────────┘  └──────────────┘  │   strategy │ │
     │                                       └───────────┘ │
     └──────────────────────────────────────────────────────┘
```

### 3.1 Hybrid Engine Design

```rust
// Per-model trait (compile-time dispatch via macros)
trait NeuralMassModel {
    const NVAR: usize;
    const NCVAR: usize;
    const PARAM_NAMES: &'static [&'static str];
    
    fn dfun<B: Backend>(
        state: Tensor<B, 2>,       // [nnodes, nvar]
        coupling: Tensor<B, 2>,    // [nnodes, ncvar]  
        params: &[f32],
    ) -> Tensor<B, 2>;            // [nnodes, nvar] derivatives
    
    fn clamp<B: Backend>(state: &mut Tensor<B, 2>);
}

// Codegen approach: use a proc macro to generate dfun/clamp from model spec
#[neural_mass_model(
    nvar = 6, ncvar = 1,
    equations = [
        "dV/dt = …",
        "dW/dt = …",
        …
    ],
    clamp = ["r >= 0.0", "S in [0.0, 1.0]"]
)]
struct MPRModel;
```

### 3.2 Sparse Coupling Strategy

**Tiered approach based on network size:**

| Node count | Strategy | Rationale |
|------------|----------|-----------|
| N < 500 | Dense coupling matrix | Memory overhead acceptable; pure Burn matmuls |
| 500 ≤ N < 2000 | Tiled CSR with shared memory | Custom CubeCL kernel, 2–3× speedup over dense |
| N ≥ 2000 | Full CSR + autotuned tiling | Custom kernel + autotuning for block sizes |

### 3.3 SBI Engine Design

```rust
#[derive(Module, Debug)]
struct MADE<B: Backend> {
    linear_y: Linear<B>,     // [D → H] with mask M1
    linear_c: Linear<B>,     // [C → H]
    linear_out: Linear<B>,   // [H → 2D] with mask M2  
    linear_out_c: Linear<B>, // [C → 2D]
    mask1: Tensor<B, 2>,
    mask2: Tensor<B, 2>,
}

#[derive(Module, Debug)]
struct MAF<B: Backend> {
    layers: Vec<MADE<B>>,
    permutations: Vec<Tensor<B, 1, Int>>,
}

impl<B: Backend> MAF<B> {
    fn log_prob(&self, x: Tensor<B, 2>, ctx: Tensor<B, 2>) -> Tensor<B, 1>;
    fn sample(&self, ctx: Tensor<B, 2>, n_samples: usize) -> Tensor<B, 3>;
}
```

---

## 4. Implementation Phases

### Phase 0: Prototype Validation (1 week)
- Build minimal Rust project with Burn (WGPU backend)
- Port ONE model (e.g., Generic2dOscillator) with dense coupling
- Run Heun integration on GPU, validate against Python
- **Gate:** GPU integration works, results match Python to 1e-5

### Phase 1: Hybrid Engine Core (3–5 weeks)
- Implement model trait + proc-macro codegen for 5–8 priority models
- Dense coupling path for small networks
- History buffer with circular indexing
- Heun/Euler integrators (deterministic + stochastic)
- Coupling functions (Linear, Sigmoidal, Kuramoto, Difference)
- TemporalAverage monitor
- TOML config schema + parser
- NPY input/output
- **Gate:** End-to-end simulation matches Python for 2-subnetwork tests

### Phase 2: Sparse Coupling + Scaling (2–3 weeks)
- Custom CSR sparse coupling kernel (CubeCL/WGSL) for medium/large networks
- Autotuning integration (block sizes, tiling strategies)
- Benchmark against Python/Numba for sweep workloads
- **Gate:** GPU simulation is faster than Numba CPU for N ≥ 100 nodes

### Phase 3: SBI Engine (2–3 weeks)
- MADE module with masked Linear layers
- MAF module with forward (log-prob) and inverse (sample)
- Adam training loop via Burn Learner
- TOML config for MAF architecture
- NPY/CSV data loading for training sets
- CLI subcommands: `train-sbi`, `infer`
- **Gate:** CDE matches smolcde log-prob within 1e-4 on Lorenz benchmark

### Phase 4: Monitors + Production Features (2–3 weeks)
- Full monitor suite (Raw, SubSample, GlobalAverage, AfferentCoupling, SpatialAverage, Projection, Bold)
- Checkpoint/resume for long simulations
- Progress reporting
- Error handling and validation (match Python simulator's validation suite)
- CI pipeline with test parity against Python

### Phase 5: Autotuning + Optimization (1–2 weeks)
- End-to-end autotuning: block size, GPU vs CPU routing per subnetwork
- CUDA backend support (in addition to WGPU)
- Parameter sweep mode (embarrassingly parallel, like the CUDA sweep POC)
- Performance documentation

**Total estimate: 12–17 weeks (3–4 months) for a single experienced Rust engineer.**

Add 4–6 weeks for comprehensive test coverage matching the Python test suite.

---

## 5. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Burn/CubeCL API breaking changes | Medium | High | Pin version; wrap in internal abstraction layer |
| Sparse CSR kernel performance insufficient | Medium | High | Dense fallback always available; can also use CPU CSR + async transfer |
| Proc-macro codegen for 24+ models too complex | Medium | Medium | Start with trait-object dispatch (slower but correct); optimize hot models only |
| GPU memory limits for large networks | Low | High | Streaming/chunking already in design; can spill to CPU |
| WGPU backend bugs on specific hardware | Low | Medium | CUDA backend as fallback for NVIDIA users |
| MAF training convergence issues on new problems | Low | Medium | smolcde's algorithm is well-characterized; can fall back to its hyperparameters |

---

## 6. Alternatives Considered

| Alternative | Pros | Cons | Verdict |
|-------------|------|------|---------|
| **Keep Python + accelerate with JAX** | Faster to implement; GPU via XLA | Still Python runtime overhead; no standalone CLI | Rejected: defeats standalone CLI goal |
| **Rust + Candle** | Lighter weight; HuggingFace ecosystem | Less mature autotuning; no CubeCL for custom kernels | Possible but Burn is better fit |
| **Rust + raw CUDA/WGSL (no framework)** | Maximum control; no dependency risk | 3× development time; reinventing autodiff, serialization, training loops | Rejected: excessive effort |
| **Rust + Burn (this proposal)** | Good balance of control and leverage | Burn still maturing; sparse primitives missing | **Selected** |
| **Hybrid: Rust CLI orchestrating Python subprocess** | Lowers porting risk | Complexity of IPC; loses GPU unification | Fallback if Phase 0 fails |

---

## 7. Conclusion

Building a Burn-based GPU hybrid simulator with SBI is **technically achievable** and would produce a significantly more deployable artifact than the current Python/Numba system. The Burn framework provides 80% of what's needed (tensor ops, GPU backends, autodiff, autotuning, serialization). The remaining 20% — sparse CSR coupling, model codegen, and monitor post-processing — requires custom engineering but no fundamental blockers.

**Recommendation:** Proceed with Phase 0 (1-week prototype) to validate the GPU integration path. If successful, commit to Phase 1 with a clear eye on the sparse coupling challenge in Phase 2.

---

*Sub-reviews synthesized from parallel agent analysis of the above reference codebases and burn.dev documentation.*
