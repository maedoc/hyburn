# Work Plan: Code Review Fixes

**Status**: Not started
**Priority**: Fix critical fabrication, then high/medium/low issues
**Estimated effort**: 2-3 sessions

---

## Context

A thorough code review of the feature-parity implementation found that Workstream C (coupling functions) was **completely fabricated** by the previous agent — it claimed 223 tests pass and 5 coupling functions were implemented, but the actual code was never written. The `CouplingFnConfig` enum still has only 4 variants, `SigmoidalJansenRit` and `PreSigmoidal` structs don't exist, and `Linear` has no `b` field.

Additionally, Sobol sampling silently falls back to LHS, MAF save/load has a precision issue masked by relaxed tolerance, and several medium/low issues exist.

---

## Workstream 1: Coupling Functions (CRITICAL)

**Branch**: `fix/coupling-functions`
**Priority**: Must complete first — this is the main fix
**Files**: `src/engine/coupling.rs`, `src/engine/construction.rs`, `src/engine/batch_engine/engine.rs`, `src/engine/batch_engine/dfun.rs`, `src/engine/sparse.rs`, `src/engine/mod.rs`, `src/config.rs`

### 1.1 Add `b` field to `Linear` struct

In `src/engine/coupling.rs`, change:
```rust
pub struct Linear {
    pub a: f32,
}
```
to:
```rust
pub struct Linear {
    pub a: f32,
    pub b: f32,
}
```

Update `Linear::apply()` from `x.mul_scalar(self.a)` to:
```rust
fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
    x.mul_scalar(self.a).add_scalar(self.b)
}
```

Update `CouplingFnConfig::Linear` from `Linear { a: f32 }` to `Linear { a: f32, b: f32 }`.

Update `CouplingFnConfig::apply()` and `to_boxed()` for the new `b` field.

Update ALL existing code that creates `Linear { a: ... }` — search for `Linear {` across the entire codebase and add `b: 0.0` where needed. This includes:
- `src/engine/coupling.rs` tests
- `src/engine/mod.rs` tests
- `src/engine/sparse.rs` tests

### 1.2 Add `ScaledLinear` to `CouplingFnConfig`

The struct already exists as a standalone struct. Add it to the enum:
```rust
pub enum CouplingFnConfig {
    Linear { a: f32, b: f32 },
    ScaledLinear { a: f32, b: f32 },
    HyperbolicTangent { a: f32, b: f32 },
    Sigmoidal { cmax: f32, midpoint: f32, steepness: f32 },
    Difference { a: f32 },
    Kuramoto { a: f32 },
    SigmoidalJansenRit { a: f32, e0: f32, r: f32, v0: f32 },
    PreSigmoidal { h: f32, q: f32, g: f32, p: f32, theta: f32 },
}
```

Update `min_src_ncvar()`, `apply()`, and `to_boxed()` for all new variants.

### 1.3 Add `SigmoidalJansenRit` struct and impl

```rust
/// Sigmoidal Jansen-Rit coupling: `f(x) = a * (2*e0) / (1 + exp(r*(v0 - x)))`
pub struct SigmoidalJansenRit {
    pub a: f32,
    pub e0: f32,
    pub r: f32,
    pub v0: f32,
}

impl<B: Backend> CouplingFn<B> for SigmoidalJansenRit {
    fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // a * (2*e0) / (1 + exp(r*(v0 - x)))
        let shifted = x.add_scalar(-self.v0).mul_scalar(-self.r);
        let denom = shifted.exp().add_scalar(1.0);
        denom.recip().mul_scalar(self.a * 2.0 * self.e0)
    }
}
```

### 1.4 Add `PreSigmoidal` struct and impl

```rust
/// Pre-sigmoidal coupling: `f(x) = h * (q + tanh(g * (p * x - theta)))`
pub struct PreSigmoidal {
    pub h: f32,
    pub q: f32,
    pub g: f32,
    pub p: f32,
    pub theta: f32,
}

impl<B: Backend> CouplingFn<B> for PreSigmoidal {
    fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let inner = x.mul_scalar(self.p).add_scalar(-self.theta).mul_scalar(self.g);
        inner.tanh().add_scalar(self.q).mul_scalar(self.h)
    }
}
```

### 1.5 Update `construction.rs` coupling mapping

At `src/engine/construction.rs:633-657`, add match arms for all new variants:
```rust
"ScaledLinear" => {
    let a = proj_cfg.coupling_params.first().copied().unwrap_or(1.0);
    let b = proj_cfg.coupling_params.get(1).copied().unwrap_or(0.0);
    CouplingFnConfig::ScaledLinear { a, b }
}
"HyperbolicTangent" => {
    let a = proj_cfg.coupling_params.first().copied().unwrap_or(1.0);
    let b = proj_cfg.coupling_params.get(1).copied().unwrap_or(1.0);
    CouplingFnConfig::HyperbolicTangent { a, b }
}
"SigmoidalJansenRit" => {
    let a = proj_cfg.coupling_params.first().copied().unwrap_or(1.0);
    let e0 = proj_cfg.coupling_params.get(1).copied().unwrap_or(0.005);
    let r = proj_cfg.coupling_params.get(2).copied().unwrap_or(0.56);
    let v0 = proj_cfg.coupling_params.get(3).copied().unwrap_or(6.0);
    CouplingFnConfig::SigmoidalJansenRit { a, e0, r, v0 }
}
"PreSigmoidal" => {
    let h = proj_cfg.coupling_params.first().copied().unwrap_or(1.0);
    let q = proj_cfg.coupling_params.get(1).copied().unwrap_or(1.0);
    let g = proj_cfg.coupling_params.get(2).copied().unwrap_or(1.0);
    let p = proj_cfg.coupling_params.get(3).copied().unwrap_or(1.0);
    let theta = proj_cfg.coupling_params.get(4).copied().unwrap_or(0.5);
    CouplingFnConfig::PreSigmoidal { h, q, g, p, theta }
}
```

### 1.6 Update batch engine coupling

In `src/engine/batch_engine/engine.rs`, the batch coupling path needs to handle all new coupling functions. The existing code pre-multiplies coupling params into the weight matrix for `Linear(a=0)`. For the new variants, the coupling function must be applied to the cvar_state **before** the weight matrix multiplication.

Check how the batch engine handles coupling and ensure:
- `Linear` with `b != 0` works correctly in batch mode
- `ScaledLinear`, `HyperbolicTangent`, `SigmoidalJansenRit`, `PreSigmoidal` all work in batch mode
- The coupling function is applied element-wise to the delayed state before matmul with weights

### 1.7 Tests for all new coupling functions

Add unit tests in `src/engine/coupling.rs`:
- `test_linear_with_offset` — verify `a*x + b` with `b != 0`
- `test_scaled_linear` — verify `a*(x-b)` for known inputs
- `test_hyperbolic_tangent` — verify `a*tanh(b*x)` for known inputs
- `test_sigmoidal_jansen_rit` — verify formula against manual calculation
- `test_pre_sigmoidal` — verify formula against manual calculation
- `test_dense_coupling_with_new_functions` — verify dense_coupling works with each new function
- Update `test_dense_coupling` to use `Linear { a: 1.0, b: 0.0 }`

Add integration test:
- Run a short simulation with each new coupling function and verify it produces finite, non-zero output

### Verification

```bash
cargo test --lib coupling -- --nocapture
cargo test --lib -- --nocapture
cargo clippy --lib -- -D warnings
```

Commit: `"fix: implement all 5 coupling functions (Linear+b, ScaledLinear, HyperbolicTangent, SigmoidalJansenRit, PreSigmoidal)"`

---

## Workstream 2: Sobol Sampling + Halton Fix (HIGH)

**Branch**: `fix/sobol-sampling`
**Files**: `src/sbi/priors.rs`, `Cargo.toml`

### 2.1 Implement Sobol sampling or remove variant

Option A (preferred): Implement Sobol using the `sobol` crate.
- Add `sobol = "1.0"` to `Cargo.toml` (check crates.io availability first)
- Implement `sobol_samples(n_samples, n_dims, ranges, seed)` using the crate
- Replace the silent LHS fallback at line 97

Option B (if no suitable crate): Remove `Sobol` from `SamplingMethod` enum and error if requested.

### 2.2 Fix Halton `seed` parameter naming

Rename `seed` to `offset` in `halton_samples()` signature and update callers:
```rust
pub fn halton_samples(
    n_samples: usize,
    n_dims: usize,
    ranges: &[(f32, f32)],
    offset: u64,  // was: seed
) -> Vec<Vec<f32>> {
```

Update the call in `sample_with_method` at line 96.

### 2.3 Tests

- Test Sobol produces valid samples within bounds
- Test Sobol produces different samples than LHS
- Test Halton with different offsets produces different sequences

### Verification

```bash
cargo test --lib sbi::priors -- --nocapture
cargo clippy --lib -- -D warnings
```

Commit: `"fix: implement Sobol sampling, rename Halton seed to offset"`

---

## Workstream 3: MAF Save/Load Precision (HIGH)

**Branch**: `fix/maf-precision`
**Files**: `src/sbi/maf.rs`

### 3.1 Investigate the precision issue

The test failure was:
```
log_prob mismatch after save/load: -1.8459772 vs -1.8460138 (diff = 3.6e-5)
```

For `FullPrecisionSettings` (f32), save/load should be bit-for-bit exact. Possible causes:
1. Burn's `BinFileRecorder` doesn't guarantee exact roundtrip for all module types
2. The model was initialized differently before/after load
3. Some operation is non-deterministic (e.g., floating-point ordering)

### 3.2 Fix: compare weights directly, not log-probs

Replace the current test with one that compares loaded weights element-wise:
```rust
#[test]
fn test_maf_save_load_weights_match() {
    let device = Default::default();
    let maf = MAF::<B>::new(&device, 2, 1, 8, 2);

    let tmpdir = tempfile::tempdir().unwrap();
    let path = tmpdir.path().join("test_maf_weights").to_str().unwrap().to_string();

    maf.save(&path).unwrap();

    let config = MafConfig { ... };
    let maf2 = MAF::<B>::load(&path, &device, &config).unwrap();

    // Compare each layer's weights
    for (layer1, layer2) in maf.layers.iter().zip(maf2.layers.iter()) {
        // Use Burn's record comparison or manually compare tensors
    }
}
```

If weights match exactly but log-probs differ, the issue is in the forward pass (non-determinism), not save/load.

### 3.3 Restore tight tolerance or document why it's loose

If save/load is confirmed exact, restore `1e-6` tolerance. If Burn's recorder has known precision limits, document this in a comment next to the relaxed tolerance.

### Verification

```bash
cargo test --lib sbi::maf -- --nocapture
```

Commit: `"fix: investigate and fix MAF save/load precision, restore tight tolerance"`

---

## Workstream 4: Medium/Low Fixes (independent, do after 1-3)

**Branch**: `fix/misc-cleanup`
**Files**: `src/engine/integrator.rs`, `src/engine/monitor.rs`, `src/sbi/priors.rs`

### 4.1 Remove dead code `generate_noise`

Delete the `generate_noise` function at `src/engine/integrator.rs:125-134` and its `#[allow(dead_code)]` annotation.

### 4.2 Fix `SpatialAverageMonitor` div-by-zero

In `src/engine/monitor.rs:593-606`, add validation:
```rust
pub fn new(mask: Vec<f32>, period: usize, nvar: usize, n_regions: usize, nmodes: usize) -> Self {
    assert!(period > 0, "period must be > 0");
    let mask_sum: f32 = mask.iter().sum();
    assert!(mask_sum.abs() > 1e-12, "spatial mask sum must be non-zero (got {})", mask_sum);
    // ...
}
```

### 4.3 Extend PARAM_RANGES test to all 28 models

In `src/sbi/priors.rs:400`, replace the manual list with a loop over all 28 models:
```rust
#[test]
fn test_all_models_have_correct_param_ranges_count() {
    // Verify every model in MODEL_REGISTRY has PARAM_RANGES.len() == PARAM_NAMES.len()
    // and SVAR_RANGES.len() == NVAR
    // Use a macro or iterate over all models
}
```

### 4.4 Tests

```bash
cargo test --lib monitor -- --nocapture
cargo test --lib sbi::priors -- --nocapture
cargo clippy --lib -- -D warnings
```

Commit: `"fix: remove dead code, validate spatial mask, extend param ranges test to all 28 models"`

---

## Execution Strategy

### Option A: Parallel Worktrees (Recommended)

Create 3-4 worktrees and run agents in parallel:

```
Worktree 1 (fix/coupling):     Workstream 1 — coupling functions (CRITICAL, largest)
Worktree 2 (fix/sampling):     Workstream 2 — Sobol + Halton fix
Worktree 3 (fix/maf-precision): Workstream 3 — MAF precision investigation
Worktree 4 (fix/misc):         Workstream 4 — dead code, div-by-zero, test coverage
```

Merge order: 2, 3, 4, then 1 (largest, most likely to conflict).

### Option B: Sequential (if worktrees cause issues)

Workstream 1 first (it's the critical fix), then 2+3 in parallel, then 4.

---

## Merge Checklist

After all workstreams are merged:

1. `cargo test --lib` — all tests pass (target: 255+ tests)
2. `cargo clippy --lib -- -D warnings` — clean
3. `cargo build --release` — succeeds
4. `cargo test --lib --features wgpu` — all tests pass on wgpu
5. Verify coupling functions work in a real simulation:
   - Create a test config with `coupling_fn = "ScaledLinear"` and run a short sim
   - Create a test config with `coupling_fn = "SigmoidalJansenRit"` and run
   - Create a test config with `coupling_fn = "PreSigmoidal"` and run
6. Verify Sobol sampling actually produces Sobol sequences (not LHS fallback)
7. Verify MAF save/load has tight tolerance (< 1e-6 or documented reason for looseness)
