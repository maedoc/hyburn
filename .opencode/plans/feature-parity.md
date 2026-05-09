# Implementation Plan: Full Hybrid Simulation Feature Parity

**Goal**: Achieve feature parity with TVB hybrid-numba API for simulation and sweeps, enabling a complete SBI workflow with multi-parameter batch sweeps and model-defined parameter ranges.

**Status**: Planning complete, execution not started.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| nsig config format | Polymorphic: `nsig = 0.01` (scalar broadcast) or `nsig = [0.01, 0.02]` (per-var) | Full backward compat, clean TOML |
| Cartesian product cap | Warn + soft cap at 10K sweep points, `--force` to exceed | Prevent accidental GPU OOM |
| Parameter ranges in priors | User/config explicitly specifies prior params + ranges; model metadata provides suggestions | No heuristic guessing; missing ranges must be user-specified |
| Cortical surface support | Not needed | Removed in TVB hybrid API |
| Colored noise | Deferred | Per-svar noise is higher priority |
| Sensor monitors | Simple: tavg x gain matrix | No forward model physics, just projection |

---

## Phase 0: Reference Trace Generation

Generate reference traces from TVB hybrid-numba for all new features. These become test fixtures.

### 0a. Model Parameter Range Extraction

**Script**: `/tmp/extract_param_ranges.py`

- Iterate all 27 TVB models
- For each model: extract `parameter_names`, per-param `domain` (lo/hi/step), `default`
- For each model: extract `state_variable_range`, `state_variable_boundaries` (if any), `cvar`, `stvar`
- Output: `/tmp/tvb_param_ranges.json`
  ```json
  {
    "Generic2dOscillator": {
      "nvar": 2, "ncvar": 1,
      "params": {
        "tau": {"default": 1.0, "lo": 1.0, "hi": 5.0, "step": 0.01},
        ...
      },
      "state_variables": {
        "V": {"lo": -2.0, "hi": 4.0},
        "W": {"lo": -6.0, "hi": 6.0}
      },
      "stvar": [0],
      "boundaries": null
    },
    ...
  }
  ```
- Note: params without `domain` get `"lo": null, "hi": null`

### 0b. Integration Reference Traces

**Script**: `/tmp/gen_ref_traces_phase2.py`

| Trace ID | Model | Integrator | Steps | dt | Notes |
|---|---|---|---|---|---|
| `rk4_g2do` | G2DO | RK4 | 100 | 0.1 | Compare vs Heun |
| `rk4_jr` | JansenRit | RK4 | 100 | 0.1 | 6-var model |
| `rk4_epileptor` | Epileptor | RK4 | 100 | 0.1 | Multi-timescale |

Output: `tests/fixtures/{id}.npy` -- final state vectors

### 0c. Per-Variable Noise Reference Traces

| Trace ID | Model | Integrator | nsig | Steps | Seed |
|---|---|---|---|---|---|
| `per_svar_noise_g2do` | G2DO | HeunStochastic | [0.02, 0.005] | 1000 | 42 |

Output: `tests/fixtures/{id}.npy` -- final state

Note: stochastic traces need matching RNG. Test by comparing with small nsig where deterministic drift dominates, within a tolerance.

### 0d. Coupling Function Reference Traces

Use TVB hybrid simulator with specific coupling functions:

| Trace ID | Models | Coupling | Config | Steps |
|---|---|---|---|---|
| `linear_offset_g2do` | G2DO -> G2DO | Linear(a=0.01, b=0.5) | 2 nodes, CSR | 200 |
| `sigr_jr` | JR -> JR | SigmoidalJansenRit | 2 nodes | 200 |
| `presig_epileptor` | Epileptor -> Epileptor | PreSigmoidal(H=1,Q=0,G=60,P=1,theta=0.5) | 2 nodes | 200 |
| `tanh_g2do` | G2DO -> G2DO | HyperbolicTangent(a=1,b=1) | 2 nodes | 200 |

Output: `tests/fixtures/{id}.npy` -- temporal average

### 0e. Sensor Monitor Reference Traces

| Trace ID | Model | Monitor | Config |
|---|---|---|---|
| `eeg_monitor_jr` | JR | EEG (gain [3x2]) | 2 regions, period=1.0ms |
| `spatial_avg_jr` | JR | SpatialAverage (mask [1x2]) | 2 regions |

Output: `tests/fixtures/{id}.npy` -- monitor output (flattened)

### 0f. BOLD Batch Sweep Reference

| Trace ID | Model | Sweep | BOLD | Steps |
|---|---|---|---|---|
| `bold_sweep_g2do` | G2DO | param[0] x 5 values | period=10, TR=2s | 2000 |

Output: `tests/fixtures/{id}.npy` -- BOLD signal per sweep point

### 0g. Multi-Param Sweep Reference

| Trace ID | Model | Params | Values | Steps |
|---|---|---|---|---|
| `multiparam_g2do` | G2DO | param[1] x param[2] | 5x5 = 25 points | 100 |

Output: `tests/fixtures/{id}_tavg.npy`, `tests/fixtures/{id}_final.npy`

---

## Phase 1: Core Engine Features (ndarray)

Each workstream is an independent worktree. Compile only touched modules.

### Workstream A: RK4 Integrator

**Files touched**: `src/engine/integrator.rs`, `src/engine/batch_engine/engine.rs`, `src/engine/batch_engine/dfun.rs`, `src/config.rs`, `src/wasm.rs`

**Changes**:
1. Add `Rk4` variant to `IntegratorKind` enum
2. Implement `rk4_step<B: Backend>(state, coupling, dt, dfun, clamp)` -- 4 dfun evaluations:
   - k1 = dfun(state, coupling)
   - k2 = dfun(state + dt/2 * k1, coupling)
   - k3 = dfun(state + dt/2 * k2, coupling)
   - k4 = dfun(state + dt * k3, coupling)
   - result = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
3. Add dispatch in runtime.rs for `IntegratorKind::Rk4`
4. Add `Rk4Stochastic` variant (RK4 deterministic + additive noise at end)
5. BatchHybridEngine: add RK4 support (4x dfun_batch calls per step)
6. Config: `integrator = "rk4"` / `"rk4_stochastic"`
7. Checkpoint: add Rk4=5, Rk4Stochastic=6
8. WASM: expose RK4 option
9. Tests: unit test (G2DO + JR RK4 vs reference), reference trace comparison

**Test command**: `cargo test --lib integrator -- --nocapture`

### Workstream B: Per-Variable Noise

**Files touched**: `src/engine/integrator.rs`, `src/engine/batch_engine/engine.rs`, `src/config.rs`, `src/wasm.rs`, `src/cli.rs`

**Changes**:
1. Change `nsig: f32` in SimConfig to `nsig: NsigConfig`:
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]
   #[serde(untagged)]
   pub enum NsigConfig {
       Scalar(f32),
       PerVariable(Vec<f32>),
   }
   ```
2. Implement `NsigConfig::to_vec(nvar: usize) -> Vec<f32>` -- scalar broadcasts, per-var validated
3. Update stochastic step functions to accept `nsig: &[f32]` instead of `nsig: f32`
4. In stochastic steps: noise per element scaled by per-variable nsig
5. BatchHybridEngine: store `nsig_vec: Vec<f32>` per subnetwork, broadcast in batch noise generation
6. Checkpoint format: version bump (v1->v2), add `nsig_len` + `nsig_vec` after `nsig` field
7. WASM: `nsig` can be number or array in JSON config
8. CLI: accept `--nsig 0.01` or `--nsig 0.01,0.02,...`
9. Backward compat: `nsig = 0.01` in TOML deserializes as `NsigConfig::Scalar(0.01)`
10. Tests: per-variable noise unit test, reference comparison

**Test command**: `cargo test --lib integrator -- --nocapture`

### Workstream C: Coupling Functions

**Files touched**: `src/engine/coupling.rs`, `src/engine/batch_engine/engine.rs`, `src/config.rs`

**Changes**:
1. Add `b: f32` to `CouplingFnConfig::Linear { a, b }` (default b=0.0)
2. Add `ScaledLinear { a, b }` to `CouplingFnConfig` (struct already exists)
3. Add `HyperbolicTangent { a, b }` to `CouplingFnConfig` (struct already exists)
4. Add `SigmoidalJansenRit { a, e0, r, v0 }` to `CouplingFnConfig`:
   - `a * (2*e0) / (1 + exp(r * (v0 - x)))`
5. Add `PreSigmoidal { h, q, g, p, theta, dynamic, global_t }` to `CouplingFnConfig`:
   - `h * (q + tanh(g * (p * x - theta)))`
   - `dynamic` flag: if true, theta is computed from mean activity; if false, use static theta
   - `global_t` flag: if true, use global mean for theta; if false, per-node
6. Update `CouplingFnConfig::min_src_ncvar()` for new variants
7. BatchHybridEngine: implement batch coupling for all new functions (tensor ops)
8. Config: TOML deserialization for new variants
9. Tests: unit test per coupling function, reference trace comparison

**Test command**: `cargo test --lib coupling -- --nocapture`

### Workstream D: Sensor Monitors + Batch BOLD

**Files touched**: `src/engine/monitor.rs`, `src/engine/bold_monitor.rs`, `src/engine/batch_engine/engine.rs`, `src/config.rs`

**Changes**:

**Sensor monitors** (EEG/MEG/iEEG -- same implementation, different naming):
1. Add `SensorProjectionMonitor` struct:
   ```rust
   pub struct SensorProjectionMonitor<B: Backend> {
       gain: Tensor<B, 2>,  // [n_sensors, n_regions]
       period: usize,
       accumulator: Vec<f32>,
       accumulator_count: usize,
       data: Vec<f32>,
       // ... dims
   }
   ```
2. Implement `Monitor<B>` for `SensorProjectionMonitor`:
   - `record()`: take state `[nvar, nnodes, nmodes]`, select VOI (e.g. cvar[0]), flatten spatial, matmul gain, accumulate
   - `flush()`: finalize temporal average
3. Add `MonitorConfig` variants: `Eeg`, `Meg`, `Ieeg` -- each with `gain_path` (npy) or `gain` (inline)
4. Add `SpatialAverageMonitor` struct:
   - Takes `spatial_mask: Vec<f32>` [n_regions]
   - Each step: `dot(mask, state_per_var)` -> scalar per var per step
   - Temporal average over period
5. Config: `MonitorConfig` gains `gain`, `spatial_mask` fields

**Batch BOLD**:
6. Add `bold_monitors: Vec<BoldMonitor>` to `BatchHybridEngine`
7. In sweep loop: after temporal average accumulation, feed neural input to BoldMonitor
8. `BatchSweepResult` gains `bold: Option<Vec<Vec<f32>>>` (per sweep point)
9. `run_sweep_with_bold()` method
10. Config: `monitors` can include `"Bold"` in batch mode

**Test command**: `cargo test --lib monitor -- --nocapture`

### Workstream E: Speed -> Delay Conversion (small, sequential after D)

**Files touched**: `src/config.rs`

**Changes**:
1. Add `speed: f32` field to `SimConfig` (default 3.0, unit: mm/ms)
2. Add `tract_lengths: Option<Vec<Vec<f32>>>` to `ProjectionConfig`
3. In `SimConfig::validate()`: if `tract_lengths` present and `delays` not, compute `idelays = ceil(tract_lengths / speed / dt)`
4. If both `tract_lengths` and `delays` present, `delays` takes precedence (explicit wins)
5. Error if `speed <= 0` and `tract_lengths` present

**Test command**: `cargo test --lib config -- --nocapture`

---

## Phase 2: Sweep & SBI Infrastructure

### Workstream F: Multi-Parameter Batch Sweep

**Files touched**: `src/engine/batch_engine/engine.rs`, `src/engine/batch_engine/dfun.rs`, `src/sbi/`, `src/cli.rs`

**Changes**:
1. Replace `SweepParam` with `SweepSpec`:
   ```rust
   pub struct SweepSpec {
       pub params: Vec<SweepPoint>,  // each has sub_idx, param_idx, values
   }

   pub struct SweepPoint {
       pub sub_idx: usize,
       pub param_idx: usize,
       pub values: Vec<f32>,
   }
   ```
2. Compute Cartesian product of all `SweepPoint.values` -> `n_sweep` points, each with a param assignment vector
3. Warn if product > 10,000; error if > 10,000 without `--force`
4. Build param tensor `[n_sweep, n_params]` where each row has full params for that sweep point
5. In `dfun_batch()`: for swept subnetworks, use per-row param instead of uniform param
6. `BatchHybridEngine::run_sweep()` takes `&SweepSpec` instead of `&SweepParam`
7. Backward compat: `run_sweep_single(param, values, n_steps)` convenience method
8. Pipeline integration: `BoxUniform` with multiple named params -> `SweepSpec`
9. CLI: `--sweep-param subnetworks[0].params[1] --sweep-values 0.1,0.2,0.3` (multiple allowed)
10. Tests: multi-param sweep vs serial, reference comparison

**Test command**: `cargo test --lib batch_engine -- --nocapture`

### Workstream G: Model Parameter Ranges + Prior Sampling

**Files touched**: `src/model/*.rs` (28 files), `src/sbi/priors.rs`, `src/config.rs`

**Changes**:

**Model metadata**:
1. Add `PARAM_RANGES: &[(f32, f32)]` constant to each model struct (from TVB extraction)
   - Params without TVB domain: `(f32::NAN, f32::NAN)` -- signals "no range"
2. Add `SVAR_RANGES: &[(f32, f32)]` constant to each model struct
3. Add `STVAR: &[usize]` constant (stochastic variable indices)
4. Add `param_ranges()` and `svar_ranges()` methods to `NeuralMassModel<B>` trait
5. Example for G2DO:
   ```rust
   pub const PARAM_RANGES: &[(f32, f32)] = &[
       (1.0, 5.0),     // tau
       (-5.0, 5.0),    // I
       (-5.0, 5.0),    // a
       (-20.0, 15.0),  // b
       (-10.0, 10.0),  // c
       (0.0001, 1.0),  // d
       (-5.0, 5.0),    // e
       (-5.0, 5.0),    // f
       (-5.0, 5.0),    // g
       (-5.0, 5.0),    // alpha
       (-5.0, 5.0),    // beta
       (-1.0, 1.0),    // gamma
   ];
   ```

**Prior config**:
6. Update `PriorConfig` to reference model param names:
   ```rust
   pub struct PriorParam {
       pub name: String,          // "subnetworks[0].params[1]" or "cortex.tau"
       pub lo: Option<f32>,       // None = look up from model PARAM_RANGES
       pub hi: Option<f32>,       // None = look up from model PARAM_RANGES
   }
   ```
7. If user doesn't specify `lo`/`hi`, look up from model `PARAM_RANGES` (error if NaN)
8. Validate: all prior param names must resolve to valid (sub_idx, param_idx)

**Sampling**:
9. Implement Latin Hypercube Sampling:
   ```rust
   pub fn latin_hypercube(n_samples: usize, n_dims: usize, ranges: &[(f32, f32)], seed: u64) -> Vec<Vec<f32>>
   ```
10. Implement Sobol sequence:
    ```rust
    pub fn sobol_samples(n_samples: usize, n_dims: usize, ranges: &[(f32, f32)]) -> Vec<Vec<f32>>
    ```
    - Use `sobol_rs` or `low-discrepancy` crate, or implement Owen-scrambled Sobol
11. Add `sampling` field to `PriorConfig`: `"uniform"` | `"lhs"` | `"sobol"` (default: `"lhs"`)
12. Pipeline: generate prior samples -> build SweepSpec -> run batch sweep
13. Tests: LHS/Sobol correctness (coverage, range adherence), prior config validation

**Test command**: `cargo test --lib sbi -- --nocapture`

### Workstream H: MAF Model I/O

**Files touched**: `src/sbi/maf.rs`, `src/sbi/train.rs`, `src/cli.rs`

**Changes**:
1. Implement `MAF::save(&self, path: &str)`:
   ```rust
   pub fn save(&self, path: &str) {
       let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
       self.clone().save_file(path, &recorder).expect("MAF save failed");
   }
   ```
2. Implement `MAF::load(path: &str, device: &B::Device, config: &MafConfig) -> Self`:
   ```rust
   pub fn load(path: &str, device: &B::Device, config: &MafConfig) -> Self {
       let maf = MAF::new(device, config.param_dim, config.feature_dim, config.hidden_units, config.n_flows);
       let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
       maf.load_file(path, &recorder, device).expect("MAF load failed")
   }
   ```
3. Save MafConfig alongside model (as JSON sidecar: `path.json`)
4. CLI `train-sbi`: add `--output model.bin` flag to save trained MAF
5. CLI `infer`: implement full workflow:
   - Load MafConfig from `model.bin.json`
   - Load MAF from `model.bin`
   - Load features from NPY or CSV
   - Run `maf.inverse_sample(features, n_samples)`
   - Write posterior samples to NPY
6. CLI `pipeline`: auto-save MAF after training
7. WASM: add `load_maf` / `save_maf` bindings (using serde + base64 for browser storage)
8. Tests: roundtrip save/load (train -> save -> load -> sample -> compare), CLI infer integration

**Test command**: `cargo test --lib sbi::maf -- --nocapture`

---

## Phase 3: GPU Backend Validation

Most GPU support comes for free with Burn tensor ops. Phase 3 is validation, not new implementation.

### Workstream I: GPU Integrator + Coupling Validation

1. Verify batch RK4 on wgpu (Metal on macOS)
2. Verify new coupling functions on wgpu
3. Verify per-variable noise on wgpu
4. If failures: debug tensor op compatibility

**Test command**: `cargo test --lib --features wgpu batch_engine -- --nocapture`
**Timeout**: 600s (wgpu compile is slow)

### Workstream J: GPU Batch BOLD + Monitors Validation

1. Verify BatchHybridEngine BOLD output on wgpu
2. Verify SensorProjectionMonitor on wgpu
3. Verify SpatialAverageMonitor on wgpu
4. Integration test: multi-subnet + BOLD + sensor monitors on wgpu

**Test command**: `cargo test --lib --features wgpu monitor -- --nocapture`

### Workstream K: GPU Multi-Param Sweep Validation

1. Verify multi-param batch sweep on wgpu
2. Verify multi-param sweep + BOLD on wgpu
3. Verify LHS/Sobol sampling -> batch sweep on wgpu
4. Performance: compare ndarray vs wgpu sweep times

**Test command**: `cargo test --lib --features wgpu -- --nocapture`

---

## Phase 4: Integration & Full Testing

1. Merge all worktrees into main branch
2. Full recompile: `cargo build --release`
3. Full lib tests: `cargo test --lib`
4. Full reference tests: `cargo test --test model_reference_test`
5. Clippy: `cargo clippy --lib -- -D warnings`
6. WASM build: `wasm-pack build --target web --no-default-features --features wasm`
7. WASM E2E: `npx playwright test`
8. Update `AGENTS.md` with new features
9. Update presets if needed
10. Update `MODEL_REGISTRY` entries if param ranges affect validation

---

## Parallelization Strategy

### Phase 0: Sequential (1 agent)
- One Python subagent generates all reference traces + extracts param ranges
- Output: `/tmp/tvb_param_ranges.json` + `tests/fixtures/*.npy`

### Phase 1: 3 parallel worktrees (limited by 4-core laptop)

```
Worktree 1: RK4 + Per-svar noise (both touch integrator.rs)
Worktree 2: Coupling functions
Worktree 3: Monitors + Batch BOLD
  | (sequential after merge)
Worktree E: Speed -> delay (small, touches config.rs after other merges)
```

Merge order: 2, 3, then 1 (1 is largest change), then E.
Resolve config.rs conflicts during merge (each adds independent fields).

### Phase 2: 3 parallel worktrees

```
Worktree 4: Multi-param batch sweep
Worktree 5: Model ranges + priors + LHS/Sobol
Worktree 6: MAF I/O
```

All touch different files. Merge in order: 6, 5, 4.

### Phase 3: Sequential (validation only)

Run on wgpu backend. If no CUDA GPU available, skip CUDA tests.

### Phase 4: Sequential (final integration)

Single session: merge + full recompile + full test suite.

---

## Dependency Graph

```
Phase 0 (ref traces) --> all of Phase 1
Phase 1A (RK4) ---------> Phase 3I (GPU RK4 validation)
Phase 1B (per-svar) -----> Phase 3I (GPU noise validation)
Phase 1C (coupling) -----> Phase 3I (GPU coupling validation)
Phase 1D (monitors) -----> Phase 3J (GPU monitor validation)
Phase 1E (speed/delay) -> (no GPU dependency)
Phase 2F (multi-sweep) -> Phase 3K (GPU sweep validation)
Phase 2G (ranges) -------> Phase 2F (sweep needs param ranges for LHS)
Phase 2H (MAF I/O) -----> (independent)

So: 2G must complete before 2F can use model ranges in LHS sampling.
    But 2F can proceed independently with manual prior config.
```

---

## Risk & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Per-svar noise changes SimConfig + checkpoint format | Breaks existing configs | Polymorphic nsig (scalar = backward compat); checkpoint version bump with migration |
| RK4 in batch: 4x dfun evals per step | Slow on GPU | Only use when explicitly requested; Heun remains default |
| Multi-param Cartesian product explosion | GPU OOM, long runs | Warn at >10K points; error without --force |
| MAF BinFileRecorder may not serialize masks/tensors | Save/load fails | Test roundtrip early; fall back to safetensors if needed |
| Stochastic reference traces non-deterministic | Can't compare exact values | Use small nsig where deterministic drift dominates; or compare stats |
| Burn wgpu tensor ops may differ from ndarray | GPU validation failures | Debug per-op; most ops are simple arithmetic |
| 28 model PARAM_RANGES extraction is tedious | Errors in range values | Script-driven from TVB; validate against TVB test suite |
| config.rs merge conflicts across worktrees | Manual resolution | Each worktree adds independent fields; straightforward merge |
| Laptop 4-core limit during parallel compiles | Slow builds | `cargo test --lib <module>` only; skip full recompile until Phase 4 |

---

## Estimated Effort

| Phase | Sessions | Notes |
|---|---|---|
| Phase 0: Ref traces | 1 | Python scripting, one subagent |
| Phase 1: Core features | 3-4 | 3 parallel worktrees + 1 sequential merge |
| Phase 2: Sweep/SBI | 2-3 | 3 parallel worktrees |
| Phase 3: GPU validation | 1 | Validation only, fixes if needed |
| Phase 4: Integration | 1 | Full recompile + test suite |
| **Total** | **8-10** | Focused sessions with subagents |

---

## Feature Checklist (Final State)

| Feature | Phase | Status |
|---|---|---|
| 28 neural mass models | -- | Done |
| 7 reference tests | -- | Done |
| RK4 integrator | 1A | Pending |
| RK4 stochastic | 1A | Pending |
| Per-variable noise | 1B | Pending |
| Linear coupling b offset | 1C | Pending |
| ScaledLinear coupling | 1C | Pending |
| HyperbolicTangent coupling | 1C | Pending |
| SigmoidalJansenRit coupling | 1C | Pending |
| PreSigmoidal coupling | 1C | Pending |
| EEG/MEG/iEEG monitors | 1D | Pending |
| SpatialAverage monitor | 1D | Pending |
| BOLD in batch sweeps | 1D | Pending |
| Speed -> delay conversion | 1E | Pending |
| Multi-parameter batch sweep | 2F | Pending |
| Model PARAM_RANGES metadata | 2G | Pending |
| Model SVAR_RANGES metadata | 2G | Pending |
| Latin Hypercube Sampling | 2G | Pending |
| Sobol sequence sampling | 2G | Pending |
| PriorConfig with named params + model ranges | 2G | Pending |
| MAF save/load (BinFileRecorder) | 2H | Pending |
| CLI infer subcommand | 2H | Pending |
| GPU validation (all new features) | 3I-K | Pending |
