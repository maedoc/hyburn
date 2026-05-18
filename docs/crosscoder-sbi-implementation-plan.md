# CrossCoder SBI Implementation Plan

Detailed implementation plan for achieving feature parity with vcc and cap-vep.
Organized into 3 phases with concrete file changes, test strategies, and
dependency ordering.

**Prerequisite reading**: `docs/crosscoder-sbi-feature-analysis.md` (the
companion analysis document; note corrections in §1.4 of this plan).

---

## 1. Corrections to the Feature Analysis

The glm-5.1 analysis contains several factual errors verified against the
actual codebase. These corrections inform the plan below.

### 1.1 β-annealing already exists

The analysis claims "hyburn has fixed β (no annealing)". **Wrong.**
`crosscoder_train.rs:46-57` implements linear β warmup from 0.01 to the
target β over the first 20% of epochs. No work needed here.

### 1.2 Hopf model already exists

The analysis claims "No Hopf oscillator model". **Wrong.** `src/model/sup_hopf.rs`
implements the Stuart-Landau / supercritical Hopf with parameters `a`
(bifurcation) and `omega` (frequency). The dfun is at `dfun.rs:534-556`:
```
dx = (a - x² - y²) * x - ω * y + coupling_x
dy = (a - x² - y²) * y + ω * x + coupling_y
```
Coupling strength K and noise D are handled at the engine level, not baked
into the model. This is more flexible than vcc's approach.

### 1.3 Empirical MVN prior infrastructure exists

The analysis claims "No empirical MVN prior for NPE". **Partially wrong.**
`crosscoder_cohort.rs` has `MvnPrior` with full covariance + Cholesky
sampling. The infrastructure exists but isn't wired into the NPE training
loop (which uses `PriorDistribution::BoxUniform` or `MultivariateNormal`
with user-provided diagonal stds). The gap is wiring, not infrastructure.

### 1.4 Feature capabilities undersold

The analysis describes hyburn features as "Generic feature extraction from
trajectories". **Undersold.** hyburn has 7 feature sets via `FeatureSet`:

| Set | Per-var count | Domain |
|---|---|---|
| Classic | 3 (mean, var, lag1_ac) | Temporal |
| Catch22 | 22 (hctsa dynamical) | Temporal |
| Catch24 | 24 (catch22 + mean + std) | Temporal |
| Fc | 7 (FC matrix stats + homotopic) | Connectivity |
| Spectral | 9 (band-power + spectral moments) | Spectral |
| TemporalStat | 7 (energy, entropy, burstiness, etc.) | Statistical |
| Combined | sum of parts | MultiDomain |

Plus `normalize_features()` and `apply_normalization()` for cohort-level
z-scoring. This is richer than either vcc (1-dim variance) or cap-vep
(2×nnodes ptp+mean_early).

### 1.5 Epileptor2D ≠ cap-vep BVEP

The analysis implies hyburn's `Epileptor2D` could substitute for cap-vep's
BVEP. **They're different models.** cap-vep uses a simplified 3-param BVEP:
```
dx = 1 - x³ - 2x²z
dz = (-z + I₁ + x₀ + 3x + c) / τ₀
```
hyburn's `Epileptor2D` has 12 params with piecewise functions, a `modification`
flag, and separate `Kvf`/`Ks` coupling scalars. For cap-vep parity, a new
simplified BVEP model would be needed, or the existing `Epileptor2D` could
be configured to approximate it.

### 1.6 Per-simulation z-scoring partially exists

`features/mod.rs:317-364` has `normalize_features` (cohort-level) and
`apply_normalization`. cap-vep's per-simulation z-scoring (normalize within
each sim before population normalization) is indeed missing as a pipeline
option, but the mathematical machinery exists.

---

## 2. Architecture Overview

### 2.1 The Core Problem

hyburn's `crosscoder_pipeline.rs` returns flat decoded SC matrices as the
parameter vector:

```rust
// crosscoder_pipeline.rs:174 — params are flat SC (nnodes² dim)
all_params.extend_from_slice(sc);  // dim = nnodes × nnodes
```

For 74-region parcellation: param_dim = 5,476. For 162-region: param_dim = 26,244.
This is 3 orders of magnitude larger than vcc (nlat+2 ≈ 18) or cap-vep
(K_total+3 ≈ 23). A MAF cannot learn a useful posterior over 26K-dimensional
parameter space from 324-dim features.

vcc and cap-vep solve this by using the **latent code itself** as the parameter:
θ = [z(nlat), dynamics_params], where nlat ∈ {8, 16, 32}. The decoded SC is
a deterministic function of z, so the MAF only needs to infer the low-dimensional
latent code + dynamics params.

### 2.2 The Solution: ParamBasis Abstraction

Introduce a `ParamBasis` enum that controls what goes into the θ vector:

```rust
pub enum ParamBasis {
    /// Flat decoded SC matrix (current behavior). param_dim = nnodes².
    Flat,
    /// CrossCoder latent code + appended dynamics params.
    /// param_dim = nlat + n_dynamics_params.
    LatentCode { nlat: usize, view_idx: usize },
    /// Spatial basis decomposition (PCA+SVD, cap-vep style).
    /// param_dim = K_total + n_dynamics_params.
    SpatialBasis { v_path: String, mu_path: String, k_total: usize },
}
```

The pipeline would return θ vectors whose composition depends on `ParamBasis`:
- `Flat`: θ = [sc_flat(nnodes²)] — current behavior
- `LatentCode`: θ = [z(nlat), K, D, ...] — vcc pattern
- `SpatialBasis`: θ = [alpha(K_total), K, τ₀, μ_base] — cap-vep pattern

### 2.3 File Map

```
src/sbi/
  param_basis.rs          NEW — ParamBasis enum, theta composition/decomposition
  crosscoder_pipeline.rs  MODIFY — use ParamBasis, return joint theta
  crosscoder_cohort.rs    MODIFY — add confusion_rate, per-parc reconstruction
  crosscoder_validate.rs  MODIFY — add confusion rate test
  diagnostics.rs          MODIFY — add coverage (90% CI), per-point diagnostics
  per_subject.rs          NEW — per-subject SBI baseline
  features/mod.rs         MODIFY — add per-sim z-scoring option
  clinical_eval.rs        NEW — F1, IoU, EZ localization (cap-vep parity)
  priors.rs               MODIFY — add MvnPrior as PriorDistribution variant
  mod.rs                  MODIFY — re-export new modules
src/model/
  bvep_simple.rs          NEW — simplified 3-param BVEP (cap-vep parity)
src/cli.rs                MODIFY — add crosscoder-pipeline subcommand
```

---

## 3. Phase 1: Joint Parameter SBI Pipeline

**Goal**: Make `crosscoder_pipeline` return θ = [z(nlat), dynamics_params]
instead of flat SC matrices. This is the critical gap that blocks all
downstream work.

### 3.1 Task 1.1: Create `src/sbi/param_basis.rs`

New file defining the parameter basis abstraction.

```rust
/// How the SBI parameter vector is composed.
#[derive(Debug, Clone)]
pub enum ParamBasis {
    /// Flat decoded SC matrix. param_dim = nnodes².
    /// Current behavior — keep for backward compatibility.
    Flat,

    /// CrossCoder latent code + dynamics parameters.
    /// θ = [z_0, z_1, ..., z_{nlat-1}, dyn_0, dyn_1, ...]
    /// The decoded SC is a deterministic function of z, so the MAF
    /// only needs to infer the low-dimensional latent code.
    LatentCode {
        /// Number of latent dimensions from CrossCoder.
        nlat: usize,
        /// Which CrossCoder view to decode through.
        view_idx: usize,
    },
}

/// Dynamics parameter specification for the joint theta vector.
#[derive(Debug, Clone)]
pub struct DynamicsParamSpec {
    /// Parameter name (e.g., "coupling_strength", "noise_amplitude").
    pub name: String,
    /// Prior minimum.
    pub min: f32,
    /// Prior maximum.
    pub max: f32,
}

impl ParamBasis {
    /// Total parameter dimensionality given the basis type.
    pub fn param_dim(&self, nnodes: usize, n_dynamics: usize) -> usize {
        match self {
            ParamBasis::Flat => nnodes * nnodes,
            ParamBasis::LatentCode { nlat, .. } => nlat + n_dynamics,
        }
    }

    /// Whether this basis requires a CrossCoder model.
    pub fn needs_crosscoder(&self) -> bool {
        match self {
            ParamBasis::Flat => false,
            ParamBasis::LatentCode { .. } => true,
        }
    }
}
```

**Key design decisions**:
- `SpatialBasis` deferred to Phase 3 (cap-vep specific, needs PCA+SVD pipeline)
- `DynamicsParamSpec` is generic — works for [k, D] (vcc) or [K, τ₀, μ_base] (cap-vep)
- `Flat` kept for backward compatibility

**Tests**:
- `test_param_basis_dim_flat`: Flat with nnodes=10 → param_dim=100
- `test_param_basis_dim_latent`: LatentCode(nlat=16) + 2 dynamics → param_dim=18
- `test_param_basis_needs_crosscoder`: Flat→false, LatentCode→true

### 3.2 Task 1.2: Modify `CrossCoderPipelineConfig`

**File**: `src/sbi/crosscoder_pipeline.rs`

Add `ParamBasis` and dynamics params to the pipeline config:

```rust
pub struct CrossCoderPipelineConfig<'a, B: Backend> {
    // ... existing fields ...
    pub model: &'a CrossCoder<B>,
    pub prior: &'a MvnPrior,
    pub view_idx: usize,
    pub template: &'a SimConfig,
    pub nnodes: usize,
    pub feature_set: &'a FeatureSet,
    pub n_samples: usize,
    pub device: &'a B::Device,
    pub seed: Option<u64>,

    // NEW: parameter basis control
    pub param_basis: ParamBasis,

    // NEW: dynamics parameter specs (for LatentCode basis)
    pub dynamics_params: Vec<DynamicsParamSpec>,

    // NEW: feature normalization
    pub normalize_features: bool,
}
```

**Backward compatibility**: Default `param_basis` to `Flat`, `dynamics_params`
to empty, `normalize_features` to false. Existing callers don't break.

### 3.3 Task 1.3: Implement joint theta composition

**File**: `src/sbi/crosscoder_pipeline.rs`

New function `compose_theta` that builds the θ vector based on `ParamBasis`:

```rust
/// Sample dynamics parameters from their prior ranges.
fn sample_dynamics_params(
    specs: &[DynamicsParamSpec],
    n: usize,
    seed: Option<u64>,
) -> Vec<f32> {
    // LHS or uniform sampling from [min, max] per spec
}

/// Compose a joint theta vector from latent code + dynamics params.
fn compose_theta_latent(
    z: &[f32],           // latent code (nlat dims)
    dynamics: &[f32],    // dynamics params (n_dynamics dims)
) -> Vec<f32> {
    let mut theta = Vec::with_capacity(z.len() + dynamics.len());
    theta.extend_from_slice(z);
    theta.extend_from_slice(dynamics);
    theta
}
```

Modify `run_crosscoder_simulation_pipeline_with_config` to branch on
`ParamBasis`:

```rust
match config.param_basis {
    ParamBasis::Flat => {
        // Existing behavior: params = flat SC matrix
        all_params.extend_from_slice(sc);
    }
    ParamBasis::LatentCode { nlat, view_idx } => {
        // NEW: params = [z(nlat), K, D, ...]
        let z_flat = &latent_codes[s * nlat..(s + 1) * nlat];
        let dynamics = sample_dynamics_params(&config.dynamics_params, 1, seed);
        let theta = compose_theta_latent(z_flat, &dynamics);
        all_params.extend_from_slice(&theta);
    }
}
```

**Critical detail**: For `LatentCode`, each simulation gets:
1. Sample z ~ MVN (existing)
2. Sample dynamics params from their priors (new)
3. Decode z → SC matrix (existing)
4. Build SimConfig with SC + dynamics params injected (new)
5. Run simulation (existing)
6. Extract features (existing)
7. Return θ = [z, dynamics] and features (new)

Step 4 requires injecting dynamics params into the SimConfig. For coupling
strength K, this means setting `coupling_params[0] = K` on the projection.
For noise D, setting `nsig = D`. This needs a helper:

```rust
/// Inject dynamics parameters into a SimConfig.
fn inject_dynamics_params(
    cfg: &mut SimConfig,
    specs: &[DynamicsParamSpec],
    values: &[f32],
) {
    for (spec, val) in specs.iter().zip(values.iter()) {
        match spec.name.as_str() {
            "coupling_strength" => {
                if !cfg.network.projections.is_empty() {
                    cfg.network.projections[0].coupling_params[0] = *val;
                }
            }
            "noise_amplitude" => {
                cfg.nsig = crate::config::NsigConfig::Scalar(*val);
            }
            _ => {
                log::warn!("Unknown dynamics param: {}", spec.name);
            }
        }
    }
}
```

### 3.4 Task 1.4: Wire MvnPrior into NPE training

**File**: `src/sbi/priors.rs`

Add `MvnPrior` (from `crosscoder_cohort`) as a `PriorDistribution` variant:

```rust
pub enum PriorDistribution {
    // ... existing variants ...
    /// Multivariate normal with full covariance from CrossCoder cohort.
    /// Loaded from mean/cov NPY files generated by encode_cohort + fit_mvn.
    CrosscoderMvn {
        mean_path: String,
        cov_path: String,
    },
}
```

Implement `sample` for this variant using `MvnPrior::from_mean_cov` +
`MvnPrior::sample`. This connects the CrossCoder's latent distribution
to the NPE training prior.

**Alternative**: Instead of a new enum variant, add a utility function
that converts `MvnPrior` → `PriorDistribution::MultivariateNormal`
(diagonal approximation). This is simpler but loses cross-dimension
correlations. The full-covariance variant is preferred.

### 3.5 Task 1.5: Update `train_maf_with_data_and_log` for conditional prior

**File**: `src/sbi/train.rs`

Currently, `train_maf_with_data_and_log` does density estimation on θ
without an explicit prior. For the `LatentCode` basis, the NPE should
use the MVN as the base distribution. This requires passing the prior
through to the MAF's loss function.

**Option A** (simpler): Train the MAF on the joint θ directly. The MAF
learns the prior implicitly from the training data distribution. This is
what vcc/cap-vep do (the sbi library's `NPE_C` uses an empirical MVN
prior internally, but the MAF just learns p(θ|x)).

**Option B** (principled): Pass the MVN prior to the MAF and use it as
the base distribution for the normalizing flow. This requires changes
to the MADE/MAF architecture.

**Recommendation**: Start with Option A (no MAF architecture changes).
The MAF learns the joint distribution from the training data. If
calibration is poor, add Option B later.

### 3.6 Task 1.6: Update CLI for crosscoder pipeline

**File**: `src/cli.rs`

Add a new subcommand `crosscoder-pipeline` that orchestrates the full
CrossCoder → simulate → train workflow:

```
hyburn crosscoder-pipeline \
    --config sim.toml \
    --crosscoder model.cc.bin \
    --cohort-data cohort_view0.npy cohort_view1.npy \
    --nlat 16 \
    --param-basis latent-code \
    --dynamics-params coupling_strength:0.1:0.3,noise_amplitude:0.2:0.4 \
    --n-samples 4096 \
    --output output/ \
    --backend ndarray
```

This replaces the manual Python orchestration in vcc/cap-vep with a
single CLI command.

### 3.7 Task 1.7: Tests for Phase 1

| Test | File | What it verifies |
|---|---|---|
| `test_param_basis_flat_backward_compat` | `param_basis.rs` | Flat basis returns nnodes² params, same as before |
| `test_param_basis_latent_code_dim` | `param_basis.rs` | LatentCode(nlat=16) + 2 dynamics → 18 dims |
| `test_compose_theta_latent` | `crosscoder_pipeline.rs` | θ = [z₀..z₁₅, k, D] with correct ordering |
| `test_inject_dynamics_params` | `crosscoder_pipeline.rs` | coupling_strength and noise_amplitude injected into SimConfig |
| `test_pipeline_latent_code_e2e` | `crosscoder_pipeline.rs` | Full pipeline with LatentCode basis, verify θ dim and features finite |
| `test_mvn_prior_as_distribution` | `priors.rs` | CrosscoderMvn variant samples correctly |
| `test_crosscoder_pipeline_cli` | `cli.rs` | CLI parses crosscoder-pipeline subcommand |

---

## 4. Phase 2: Validation & Diagnostics

**Goal**: Add confusion rate, coverage, per-subject SBI, and per-parcellation
benchmarking. These are needed for paper-quality results.

### 4.1 Task 2.1: Confusion rate

**File**: `src/sbi/crosscoder_cohort.rs`

Add confusion rate computation:

```rust
/// Compute cross-parcellation confusion rate.
///
/// For each subject and source parcellation, encode to latent z,
/// decode through all other parcellations, re-encode, and check
/// if the closest latent neighbor is the same subject.
///
/// Returns (confusion_matrix[nparc × nparc], mean_confusion_rate).
pub fn confusion_rate<B: Backend>(
    model: &CrossCoder<B>,
    data: &[Vec<f32>],
    shapes: &[(usize, usize)],
    device: &B::Device,
) -> (Vec<f32>, f32) {
    let n_views = model.views.len();
    let n_samples = shapes[0].0;
    let latent_dim = model.latent_dim;

    // Encode all views
    let mut encodings: Vec<Vec<f32>> = Vec::with_capacity(n_views);
    for (v, &dim) in shapes.iter().map(|s| s.1).enumerate() {
        let t = Tensor::<B, 2>::from_data(
            TensorData::new::<f32, Vec<usize>>(data[v].clone(), vec![n_samples, dim]),
            device,
        );
        let mu = model.views[v].encode_deterministic(t);
        encodings.push(mu.into_data().as_slice::<f32>().unwrap().to_vec());
    }

    // For each (src, tgt) pair, compute identification rate
    let mut conf_matrix = vec![0.0f32; n_views * n_views];
    for i in 0..n_views {
        for j in 0..n_views {
            if i == j {
                conf_matrix[i * n_views + j] = 1.0; // diagonal = self
                continue;
            }
            let mut correct = 0;
            for s in 0..n_samples {
                let z_src: Vec<f32> = (0..latent_dim)
                    .map(|d| encodings[i][s * latent_dim + d])
                    .collect();
                // Find closest in target encoding
                let mut min_dist = f32::INFINITY;
                let mut min_idx = 0;
                for t in 0..n_samples {
                    let dist: f32 = (0..latent_dim)
                        .map(|d| (encodings[j][t * latent_dim + d] - z_src[d]).powi(2))
                        .sum();
                    if dist < min_dist {
                        min_dist = dist;
                        min_idx = t;
                    }
                }
                if min_idx == s {
                    correct += 1;
                }
            }
            conf_matrix[i * n_views + j] = correct as f32 / n_samples as f32;
        }
    }

    let mean_conf = conf_matrix.iter().sum::<f32>() / (n_views * n_views) as f32;
    (conf_matrix, mean_conf)
}
```

**Tests**:
- `test_confusion_rate_identity`: 2 views, correlated data → confusion > 0.8
- `test_confusion_rate_random`: 2 views, independent data → confusion ≈ 1/n_samples

### 4.2 Task 2.2: Per-parcellation reconstruction correlation

**File**: `src/sbi/crosscoder_cohort.rs`

```rust
/// Compute per-view reconstruction correlation.
///
/// For each view, encode all samples, decode back, and compute
/// Pearson correlation between original and reconstructed.
///
/// Returns per-view correlation coefficients.
pub fn reconstruction_correlation<B: Backend>(
    model: &CrossCoder<B>,
    data: &[Vec<f32>],
    shapes: &[(usize, usize)],
    device: &B::Device,
) -> Vec<f32> {
    // For each view: encode → decode → pearson(original, reconstructed)
}
```

### 4.3 Task 2.3: Coverage (90% CI) diagnostic

**File**: `src/sbi/diagnostics.rs`

Add to `SbiDiagnostics`:

```rust
pub struct SbiDiagnostics {
    // ... existing fields ...
    /// Per-parameter 90% CI coverage: fraction of test points where
    /// true param falls within [q5, q95] of posterior.
    pub coverage_90: Vec<f32>,
    /// Mean coverage across all parameters.
    pub mean_coverage_90: f32,
}
```

Update `from_samples` to compute coverage:

```rust
// In the per-parameter loop:
let mut q5_sum = 0.0f32;
let mut q95_sum = 0.0f32;
for i in 0..n_test {
    // ... existing posterior mean/var computation ...
    // Compute quantiles from posterior samples
    let mut sorted: Vec<f32> = (0..n_samples)
        .map(|s| posterior_samples[(i * n_samples + s) * param_dim + d])
        .collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q5 = sorted[(n_samples as f32 * 0.05) as usize];
    let q95 = sorted[(n_samples as f32 * 0.95) as usize];
    let true_val = true_params[i * param_dim + d];
    if true_val >= q5 && true_val <= q95 {
        coverage_sum += 1.0;
    }
}
coverage_90[d] = coverage_sum / n_test as f32;
```

**Tests**:
- `test_coverage_perfect_posterior`: Tight posterior → coverage ≈ 1.0
- `test_coverage_uninformative`: Prior-width posterior → coverage ≈ 0.9

### 4.4 Task 2.4: Per-subject SBI baseline

**File**: `src/sbi/per_subject.rs` (new)

```rust
/// Per-subject SBI: fixed SC, infer only dynamics parameters.
///
/// For each subject in a cohort:
/// 1. Fix SC = their actual connectome
/// 2. Sample dynamics params from prior
/// 3. Simulate with fixed SC + varied dynamics
/// 4. Train separate NPE per subject
/// 5. Compare diagnostics to cohort-level SBI
pub struct PerSubjectConfig {
    pub sc_path: String,          // NPY file: [nnodes, nnodes]
    pub dynamics_params: Vec<DynamicsParamSpec>,
    pub template: SimConfig,
    pub feature_set: FeatureSet,
    pub n_samples: usize,
    pub maf_config: MafConfig,
}

pub fn run_per_subject_sbi<B: Backend>(
    config: &PerSubjectConfig,
    device: &B::Device,
) -> (SbiDiagnostics, MAF<Autodiff<NdArray<f32>>>) {
    // 1. Load SC
    // 2. Sample dynamics params
    // 3. For each sample: inject SC + dynamics → simulate → features
    // 4. Train MAF
    // 5. Compute diagnostics
}
```

**Tests**:
- `test_per_subject_sbi_produces_diagnostics`: Toy SC + G2DO → finite diagnostics

### 4.5 Task 2.5: Cohort vs per-subject comparison utility

**File**: `src/sbi/per_subject.rs`

```rust
/// Compare cohort-level vs per-subject SBI diagnostics.
///
/// Returns a comparison struct with per-parameter and aggregate metrics
/// for both approaches, plus the amortization benefit (cohort - per-subject).
pub fn compare_cohort_vs_subject(
    cohort_diags: &SbiDiagnostics,
    subject_diags: &[SbiDiagnostics],
) -> CohortComparison {
}
```

### 4.6 Task 2.6: Per-parcellation SBI benchmarking

**File**: `src/sbi/crosscoder_pipeline.rs`

```rust
/// Run the full pipeline for each parcellation and collect diagnostics.
///
/// For multi-view CrossCoders, iterate over all views and compute
/// per-parcellation shrinkage/z-score to demonstrate parcellation invariance.
pub fn benchmark_parcellations<B: Backend>(
    model: &CrossCoder<B>,
    prior: &MvnPrior,
    template: &SimConfig,
    feature_set: &FeatureSet,
    n_samples: usize,
    device: &B::Device,
) -> Vec<(String, SbiDiagnostics)> {
}
```

### 4.7 Task 2.7: Tests for Phase 2

| Test | File | What it verifies |
|---|---|---|
| `test_confusion_rate_basic` | `crosscoder_cohort.rs` | 2-view CC with correlated data → rate > 0.5 |
| `test_reconstruction_correlation` | `crosscoder_cohort.rs` | Trained CC → r > 0.9 |
| `test_coverage_computation` | `diagnostics.rs` | 90% CI coverage within [0.85, 0.95] for calibrated posterior |
| `test_per_subject_sbi_e2e` | `per_subject.rs` | Full per-subject pipeline produces finite diagnostics |
| `test_cohort_vs_subject_comparison` | `per_subject.rs` | Comparison struct has correct fields |

---

## 5. Phase 3: Domain-Specific Extensions

**Goal**: Enable cap-vep style workflows — spatial basis parameterization,
clinical evaluation, simplified BVEP model.

### 5.1 Task 3.1: Spatial basis parameterization

**File**: `src/sbi/param_basis.rs`

Extend `ParamBasis` with the cap-vep pattern:

```rust
pub enum ParamBasis {
    Flat,
    LatentCode { nlat: usize, view_idx: usize },

    /// Spatial basis decomposition (cap-vep style).
    /// θ = [alpha(K_total), K_global, tau0, x0_mu_base]
    /// x0 = x0_mu_base + V_combined @ alpha
    /// where V_combined = [PCA(x0_population) | SVD(mean_SC)]
    SpatialBasis {
        /// Path to V_combined basis matrix NPY [nnodes, K_total].
        v_path: String,
        /// Path to x0 population mean NPY [nnodes].
        mu_path: String,
        /// Number of basis components.
        k_total: usize,
    },
}
```

**File**: `src/sbi/spatial_basis.rs` (new)

```rust
/// Compute combined PCA(x0) + SVD(SC) basis for x0 parameterization.
///
/// Equivalent to cap-vep's `compute_combined_basis`.
pub fn compute_combined_basis(
    x0_population: &[f32],   // [n_patients, nnodes]
    sc_pool: &[f32],          // [n_sc, nnodes, nnodes]
    n_patients: usize,
    n_sc: usize,
    nnodes: usize,
    n_x0_comp: Option<usize>,
    n_sc_comp: usize,
    variance_frac: f32,
) -> CombinedBasis {
    // 1. PCA on x0_population → V_pca [nnodes, K_x0]
    // 2. SVD of mean_SC → V_sc [nnodes, K_sc]
    // 3. Concatenate → V_combined [nnodes, K_total]
}

pub struct CombinedBasis {
    pub v_combined: Vec<f32>,  // [nnodes, K_total]
    pub x0_mean: Vec<f32>,     // [nnodes]
    pub k_x0: usize,
    pub k_sc: usize,
    pub k_total: usize,
    pub var_ratio: Vec<f32>,   // [K_x0] explained variance
}
```

**Tests**:
- `test_combined_basis_shapes`: 162 regions, 8 PCA + 5 SC → K_total=13
- `test_x0_reconstruction`: V @ alpha + mu → valid x0 vector

### 5.2 Task 3.2: Simplified BVEP model

**File**: `src/model/bvep_simple.rs` (new)

```rust
/// Simplified 2D BVEP (Virtual Epileptic Patient) model.
///
/// Matches cap-vep's 3-parameter BVEP:
///   dx = 1 - x³ - 2x²z + coupling
///   dz = (-z + I₁ + x₀ + 3x + coupling) / τ₀
///
/// Parameters: x0 (excitability), K (coupling strength), tau0 (time scale)
/// Fixed: I₁ = 3.1
pub struct BvepSimple;

impl<B: Backend> NeuralMassModel<B> for BvepSimple {
    const NVAR: usize = 2;
    const NCVAR: usize = 1;
    const CVAR: &'static [usize] = &[0];
    const PARAM_NAMES: &'static [&'static str] = &["x0", "K", "tau0"];
    // ...
}
```

**Tests**:
- `test_bvep_simple_equilibrium`: x0=-2.5 → steady state near (-2, 3.5)
- `test_bvep_simple_seizure`: x0=-1.5 → seizure dynamics (x > -0.5)

### 5.3 Task 3.3: Per-simulation feature z-scoring

**File**: `src/sbi/features/mod.rs`

Add per-simulation normalization option to the feature extraction:

```rust
/// Extract features with optional per-simulation z-scoring.
///
/// When `per_sim_normalize` is true, features are z-scored within each
/// simulation before being returned. This preserves spatial rank ordering
/// across diverse SC matrices (cap-vep pattern).
pub fn extract_features_with_options(
    trajectory: &[f32],
    shape: &[usize],
    feature_set: &FeatureSet,
    per_sim_normalize: bool,
) -> Vec<f32> {
    let features = extract_features_with(trajectory, shape, feature_set);
    if per_sim_normalize {
        zscore_in_place(&features)
    } else {
        features
    }
}
```

**Tests**:
- `test_per_sim_zscore`: Features have mean≈0, std≈1 after normalization
- `test_per_sim_zscore_preserves_rank`: Ordering of feature values preserved

### 5.4 Task 3.4: Clinical evaluation metrics

**File**: `src/sbi/clinical_eval.rs` (new)

```rust
/// EZ localization accuracy metrics (cap-vep parity).
pub struct EzMetrics {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub iou: f32,
    pub overlap: usize,
}

/// Identify epileptogenic zones from posterior x0 estimates.
///
/// Regions with x0 > threshold are seizure-prone.
/// Alternatively, select top-k regions by x0 value.
pub fn localize_ez(
    x0_posterior: &[f32],  // [n_samples, nnodes] or [nnodes]
    n_samples: usize,
    nnodes: usize,
    threshold: Option<f32>,
    top_k: Option<usize>,
) -> (Vec<usize>, Vec<f32>) {
    // If 2D: compute median across samples
    // If threshold: select x0 > threshold
    // If top_k: select top-k by x0 value
}

/// Compute EZ localization accuracy against clinical ground truth.
pub fn compute_ez_accuracy(
    predicted_ez: &[usize],
    true_ez: &[usize],
    nnodes: usize,
) -> EzMetrics {
}

/// Load clinical ground truth from JSON (ei-vep.json format).
pub fn load_clinical_ground_truth(path: &str) -> anyhow::Result<HashMap<String, GroundTruth>> {
}

pub struct GroundTruth {
    pub ez_indices: Vec<usize>,
    pub pz_indices: Vec<usize>,
    pub seizure_free: Option<bool>,
}
```

**Tests**:
- `test_localize_ez_threshold`: x0 > -2.05 → correct EZ indices
- `test_localize_ez_top_k`: top-3 selection → 3 indices
- `test_ez_accuracy_perfect`: pred=true → F1=1.0, IoU=1.0
- `test_ez_accuracy_partial`: 2/3 overlap → F1=0.8

### 5.5 Task 3.5: Feature-parameter correlation diagnostic

**File**: `src/sbi/diagnostics.rs`

Add x0↔feature correlation check (cap-vep's sanity diagnostic):

```rust
/// Compute per-region correlation between x0 and feature values.
///
/// High correlation (ρ > 0.8) indicates features are informative about
/// the spatial pattern of excitability. Low correlation (< 0.3) suggests
/// the feature pipeline is not capturing the relevant signal.
pub fn x0_feature_correlation(
    x0_samples: &[f32],     // [n_check, nnodes]
    features: &[f32],        // [n_check, feature_dim]
    n_check: usize,
    nnodes: usize,
    feature_dim: usize,
) -> Vec<f32> {
    // For each region r: pearson(x0[:, r], features[:, r])
}
```

### 5.6 Task 3.6: Gain-inverse feature mapping

**File**: `src/sbi/features/gain_inverse.rs` (new)

```rust
/// Map sensor-space features to source space via gain transpose.
///
/// Equivalent to cap-vep's `source_slp` and `source_power`.
/// gain_norm = gain / gain.sum(axis=1, keepdims=True)
/// src = gain_norm.T @ sensor_data
pub fn gain_inverse(
    sensor_data: &[f32],  // [nt, ns] or [ns]
    gain: &[f32],          // [ns, nn]
    nt: usize,
    ns: usize,
    nn: usize,
) -> Vec<f32> {
}
```

### 5.7 Task 3.7: Tests for Phase 3

| Test | File | What it verifies |
|---|---|---|
| `test_spatial_basis_computation` | `spatial_basis.rs` | PCA+SVD → correct K_total |
| `test_bvep_simple_dynamics` | `bvep_simple.rs` | Seizure and equilibrium regimes |
| `test_per_sim_zscore` | `features/mod.rs` | Mean≈0, std≈1, rank preserved |
| `test_ez_localization` | `clinical_eval.rs` | Threshold and top-k modes |
| `test_gain_inverse` | `features/gain_inverse.rs` | sensor→source mapping |
| `test_x0_feature_correlation` | `diagnostics.rs` | Correlation in [0, 1] |

---

## 6. Implementation Order & Dependencies

```
Phase 1 (critical path):
  1.1 param_basis.rs          ← no deps
  1.2 CrossCoderPipelineConfig ← deps on 1.1
  1.3 compose_theta + inject  ← deps on 1.2
  1.4 MvnPrior → PriorDist    ← no deps (parallel with 1.1-1.3)
  1.5 train_maf update        ← deps on 1.4
  1.6 CLI subcommand          ← deps on 1.3
  1.7 tests                   ← deps on all above

Phase 2 (can start 1.1 in parallel):
  2.1 confusion_rate          ← no deps on Phase 1
  2.2 reconstruction_corr     ← no deps on Phase 1
  2.3 coverage diagnostic     ← no deps on Phase 1
  2.4 per_subject.rs          ← deps on 1.3 (uses ParamBasis)
  2.5 cohort vs subject       ← deps on 2.4
  2.6 per-parc benchmarking   ← deps on 1.3
  2.7 tests                   ← deps on all above

Phase 3 (after Phase 1):
  3.1 SpatialBasis            ← deps on 1.1 (extends ParamBasis)
  3.2 bvep_simple.rs          ← no deps
  3.3 per-sim z-scoring       ← no deps
  3.4 clinical_eval.rs        ← no deps
  3.5 x0_feature_correlation  ← deps on 3.4
  3.6 gain_inverse            ← no deps
  3.7 tests                   ← deps on all above
```

**Parallelizable work**:
- 1.1 + 1.4 + 2.1 + 2.2 + 2.3 can all start immediately
- 3.2 + 3.3 + 3.4 + 3.6 can all start after 1.1

---

## 7. Risk Assessment

### 7.1 MAF on high-dimensional θ

Even with `LatentCode` basis, the MAF must learn p(z|x) where z ∈ ℝ^16
and x ∈ ℝ^324 (catch22 features). This is feasible but may require:
- More training samples (4096+)
- More flow layers (8-10)
- Larger hidden units (128-256)

**Mitigation**: The existing `MafConfig` already supports these knobs.
Start with vcc's hyperparameters (nlat=16, 4096 sims, 4-layer MAF)
and tune from there.

### 7.2 SimConfig injection fidelity

Injecting dynamics params into `SimConfig` (coupling_strength, noise_amplitude)
requires modifying the config before engine construction. This is straightforward
for scalar params but may need care for per-node params (e.g., x0 vector in
cap-vep).

**Mitigation**: Phase 1 handles scalar params only. Per-node params (x0 vector)
are Phase 3 territory with `SpatialBasis`.

### 7.3 CrossCoder .pkl ↔ .cc.bin interop

vcc/cap-vep use vbjax `.pkl` files. hyburn uses Burn `.cc.bin` files.
To run the full pipeline on real cohort data, we need either:
1. A Python script to convert .pkl → NPY arrays → load in Rust
2. A Rust pickle reader (complex, fragile)
3. Train the CrossCoder in hyburn directly from NPY cohort data

**Recommendation**: Option 3 is cleanest. The cohort data (connectome
matrices) can be saved as NPY from Python, then loaded via
`load_cohort_from_npy` and trained with `train_crosscoder`. This
avoids pickle format dependency entirely.

### 7.4 Test data generation

Most tests need synthetic data. Strategy:
- Use existing `tests/validate_output/` NPY files for CrossCoder tests
- Generate synthetic SC matrices (random symmetric positive matrices)
- Use G2DO or SupHopf for fast simulation tests
- Use `test_crosscoder_matches_vbjax_reference` as the integration gold standard

---

## 8. Success Criteria

### Phase 1 complete when:
- [ ] `ParamBasis::LatentCode` produces θ = [z(nlat), K, D] with correct dim
- [ ] Pipeline runs end-to-end: CrossCoder → MVN → simulate → features → MAF → posterior
- [ ] MAF posterior on toy data (G2DO, 2 nodes) shows shrinkage > 0.3
- [ ] CLI `crosscoder-pipeline` subcommand works
- [ ] All existing tests still pass (backward compatibility)

### Phase 2 complete when:
- [ ] Confusion rate computed on 2-view toy CrossCoder
- [ ] Coverage (90% CI) added to `SbiDiagnostics`
- [ ] Per-subject SBI produces diagnostics for comparison
- [ ] Per-parcellation benchmarking iterates over views

### Phase 3 complete when:
- [ ] `SpatialBasis` parameterization works for cap-vep style x0
- [ ] Simplified BVEP model simulates correctly
- [ ] Per-sim z-scoring option in feature extraction
- [ ] F1/IoU metrics computed against synthetic ground truth
- [ ] Gain-inverse mapping produces source-space features

---

## 9. Appendix: Key File References

| File | Lines | Key content |
|---|---|---|
| `src/sbi/crosscoder.rs` | 378 | CrossCoderView, CrossCoder, loss, save/load |
| `src/sbi/crosscoder_train.rs` | 169 | train_crosscoder with β-annealing (line 46-57) |
| `src/sbi/crosscoder_cohort.rs` | 292 | MvnPrior, fit_mvn, encode_cohort, cholesky |
| `src/sbi/crosscoder_pipeline.rs` | 323 | Pipeline config, SC decode, simulate, features |
| `src/sbi/crosscoder_validate.rs` | 223 | vbjax reference validation test |
| `src/sbi/diagnostics.rs` | 274 | SbiDiagnostics: z-score, shrinkage |
| `src/sbi/features/mod.rs` | 462 | FeatureSet enum, 7 feature sets, normalize |
| `src/sbi/priors.rs` | 733 | PriorDistribution, sampling methods |
| `src/sbi/train.rs` | 232 | train_maf_with_data_and_log |
| `src/sbi/maf.rs` | 410 | MAF normalizing flow |
| `src/model/sup_hopf.rs` | 56 | SupHopf (Stuart-Landau Hopf) |
| `src/model/epileptor2d.rs` | 69 | Epileptor2D (12-param, not cap-vep BVEP) |
| `src/model/mod.rs` | — | NeuralMassModel trait, 28 models |
| `src/cli.rs` | 1252 | CLI with pipeline subcommand |
| `src/engine/batch_engine/dfun.rs` | 1907 | All model dfuns including sup_hopf_dfun_batch |