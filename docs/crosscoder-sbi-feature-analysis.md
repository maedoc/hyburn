# CrossCoder & SBI Feature Analysis: vcc / cap-vep / hyburn

Comprehensive comparison of three codebases implementing CrossCoder-based
simulation-based inference (SBI) for brain network modeling. This document
surveys what exists, what's missing, and what design patterns emerge—without
prescribing a specific implementation plan.

---

## 1. Codebase Summaries

### 1.1 vcc (Cohort Connectome SBI)

**Location**: `rtx4:~/src/apvbt/vcc/` (Python, JAX, vbjax)
**Domain**: Multi-parcellation connectome embedding + amortized SBI for
Hopf/MPR dynamics on healthy cohorts (HCP + 1000 Brains, 461 subjects).

**Key files**:

| File | Purpose |
|---|---|
| `vcc_utils.py` | `bench_model_vcc`, `sample_model_vcc`, `sample_subj_model` |
| `synth_latent.py` | End-to-end: crosscoder → simulate → SBI → diagnostics |
| `step0-train-crosscoder.ipynb` | Train variational + deterministic crosscoders |
| `step1-latent-eval.ipynb` | MVN computation, latent weight plots |
| `step2-id-conn-hopf.ipynb` | Hopf cohort-level SBI |
| `step2-id-conn-parc.ipynb` | Per-parcellation SBI consistency |
| `step3-id-subj-hopf.ipynb` | Cohort vs per-subject comparison (Hopf) |
| `step3-id-subj-mpr.ipynb` | Cohort vs per-subject comparison (MPR) |
| `step9-test-fcd.ipynb` | FCD-based evaluation |
| `supplement/gen_figures.py` | 10 publication figures |
| `CROSSCODER_DRAFT_PLAN.md` | Paper section mapping |

**CrossCoder**: 20 parcellation views (31–294 regions), `vbjax.CrossCoder`
with variational mode (μ + logσ² encoder, β-annealed KL), confusion rate,
per-parcellation reconstruction correlation.

**SBI parameterization**: θ = [z_latent(nlat), k, D] where z ~ MVN(cohort),
k ∈ [0.1, 0.3] coupling, D ∈ [0.2, 0.4] noise.

**Features**: Variance of fast variable (Hopf), mean of rate variable (MPR).

**Diagnostics**: Shrinkage, z-score, 90% CI coverage.

**Key results (MPR, 231 test subjects)**:

| Metric | Cohort | Per-subject |
|---|---|---|
| Mean shrinkage | 0.633 | 0.636 |
| Mean z-score | 0.577 | 0.480 |
| Coverage (90% CI) | 99.6% | 95.2% |
| k shrinkage | — | 0.543 |
| D shrinkage | — | 0.728 |
| k z-score | — | 0.497 |
| D z-score | — | 0.464 |

---

### 1.2 cap-vep (Clinical Epilepsy SBI)

**Location**: `rtx4:~/src/cap-vep/` (Python, JAX/numpy, sbi)
**Domain**: Epileptogenic zone (EZ) localization from SEEG data.
~30 epilepsy patients, 162-region VEP parcellation, BVEP (2D Epileptor) model.

**Key files**:

| File | Lines | Purpose |
|---|---|---|
| `crosscoder_fit.py` | 56 | Fit single-view CrossCoder, decode SCs |
| `priors.py` | 243 | Combined PCA(x0)+SVD(SC) basis, theta sampling |
| `simulator.py` | 84 | BVEP numpy Heun simulator |
| `simulator_jax.py` | 147 | BVEP JAX vmap GPU simulator (6ms/sim) |
| `features.py` | 125 | Source-space ptp_z + mean_early_z, gain-inverse |
| `sbi_pipeline.py` | 198 | VEPModel, sample_cohort, train_sbi, personalize |
| `per_subject.py` | 116 | Per-subject SBI baseline |
| `inverse.py` | 111 | Gain transpose mapping (sensor → source) |
| `evaluation.py` | 183 | EZ localization: F1, precision, recall, IoU |

**CrossCoder**: Single-view (vep162), variational, 8 latent dims. Used only
for SC synthesis—no multi-view alignment.

**SBI parameterization**: θ = [alpha_x0(K_x0), alpha_sc(K_sc), K, τ₀, x0_mu_base]
where:
- x0 = x0_mu_base + V_combined @ alpha
- V_combined = [PCA(population_x0) | SVD(mean_SC)] ∈ ℝ^(162 × K_total)
- k ∈ [0.5, 1.5] global coupling, τ₀ ∈ [10, 20] time scale
- Mixture prior: 70% N(-2.3, 0.2), 30% N(-2.7, 0.3) for x0_mu_base

**Features**: 2 × 162 = 324 dim: [ptp_z(162), mean_early_z(162)]
- Per-simulation z-scoring preserves spatial rank ordering
- Population z-scoring on top for SBI training stability

**Diagnostics**: EZ F1, precision, recall, IoU against clinical ground truth
from ei-vep.json (surgical outcomes).

**Key results**:

| Config | F1 mean | F1 median | F1 > 0.5 |
|---|---|---|---|
| Random baseline | 0.02 | 0.0 | 0/21 |
| x0 threshold only | 0.48 | — | — |
| Cohort SBI (n_sc=12, 10k sims) | 0.55 | 0.67 | 14/21 |
| Cohort PCA-only (fair comp) | 0.37 | 0.33 | 8/21 |
| Per-subject (PCA-only) | 0.34 | 0.33 | 8/21 |
| Stan HMC (original VEP) | 0.46 | 0.44 | 8/21 |

**Critical finding**: Cohort amortization advantage is +0.03 F1 over
per-subject when using the same PCA-only basis. The larger advantage
(+0.16 F1 with PCA+SC basis) comes from SC eigenvector columns in the
spatial basis, not amortization per se.

---

### 1.3 hyburn

**Location**: `~/src/hyburn/` (Rust, Burn)
**Domain**: Multi-model brain network simulator with SBI pipeline.

**CrossCoder/SBI files**:

| File | Lines | Purpose |
|---|---|---|
| `src/sbi/crosscoder.rs` | 378 | CrossCoder model: LinearVae per view, encode/decode, loss |
| `src/sbi/crosscoder_train.rs` | 169 | Training loop with Adam optimizer |
| `src/sbi/crosscoder_cohort.rs` | 292 | Cohort data loading, MVN fitting, Cholesky sampling |
| `src/sbi/crosscoder_validate.rs` | 223 | Validation against vbjax reference (correlation + MSE) |
| `src/sbi/crosscoder_pipeline.rs` | 323 | SC decode → SimConfig inject → simulate → features → MAF |
| `src/sbi/priors.rs` | 733 | BoxUniform, MVN, LHS/Sobol/Halton sampling |
| `src/sbi/maf.rs` | 410 | MAF density estimator (Burn) |
| `src/sbi/train.rs` | 232 | MAF training loop |
| `src/sbi/diagnostics.rs` | 274 | z-score, shrinkage (no coverage) |
| `src/sbi/features/` | — | Feature extraction from trajectories |
| `src/model/` | 28 models | G2DO, MPR, JansenRit, WilsonCowan, Kuramoto, Epileptor, etc. |
| `src/engine/` | — | HybridEngine, batch engine, coupling, runtime |

**CrossCoder**: Multi-view linear VAE (same architecture as vcc). Fixed β
(no annealing), no confusion rate, no per-parcellation metrics. Validation
is reconstruction MSE + best-match correlation against vbjax reference.

**SBI parameterization**: Flat SC matrix (nnodes² params) decoded from
CrossCoder, injected as projection weights. Does NOT jointly train on
[latent_code, dynamics_params]. No spatial basis decomposition for model
parameters (e.g., x0).

**Features**: Generic feature extraction from trajectories (see
`src/sbi/features/`).

**Diagnostics**: z-score + shrinkage only. No coverage (90% CI),
no clinical metrics (F1, IoU).

**Models**: 28 neural mass models including Epileptor (6-var) but NOT
the 2D BVEP variant used in cap-vep. No Hopf oscillator model.

---

## 2. Feature-by-Feature Comparison

### 2.1 CrossCoder Architecture

| Feature | vcc | cap-vep | hyburn | Notes |
|---|---|---|---|---|
| Linear encoder/decoder | ✓ (per-parc) | ✓ (single view) | ✓ (per view) | Same architecture |
| Variational (μ + logσ²) | ✓ | ✓ | ✓ | vcc has dedicated variational flag |
| Deterministic (μ only) | ✓ | ✗ | ✓ `encode_deterministic` | |
| Multi-view (≥2 parcellations) | ✓ 20 views | ✗ 1 view | ✓ generic | vcc is the only real multi-view user |
| Per-view normalization | ✓ center/zscore/logit | ✓ center only | ✗ | hyburn expects pre-processed data |
| Denormalization (inverse) | ✓ `_denorm` | ✗ (nonneg=False) | ✗ | |
| β-annealing | ✓ 0→β_end over anneal_steps | ✗ fixed KL weight | ✗ fixed β | |
| Chunked training | ✓ `lax.scan` | ✗ | ✗ | |
| Train/test split tracking | ✓ `tts` field | ✓ `tts` field | ✗ | External in hyburn |
| Confusion rate | ✓ `calc_confusion_rate` | ✗ | ✗ | Key validation metric |
| Per-parcellation reconstruction | ✓ per-parc r>0.95 | ✗ | ✗ | |
| Checkpoint save/load | ✓ `to_pkl`/`from_pkl` | ✓ | ✓ `.cc.bin` | |
| SC decode with clip_positive | ✗ | ✓ `clip_positive=True` | ✗ | cap-vep needs non-negative SC |
| SC from latent batch decode | ✓ | ✓ `decode_sc_batch` | ✓ `generate_synthetic_sc_matrices` | |

### 2.2 SBI Pipeline

| Feature | vcc | cap-vep | hyburn | Notes |
|---|---|---|---|---|
| Joint latent+dynamics SBI | ✓ θ=[z, k, D] | ✓ θ=[α, K, τ, μ] | ✗ flat SC only | **Critical gap** |
| Parameter space | nlat + 2 | K_total + 3 | nnodes² | hyburn param dim is O(nnodes²) |
| Spatial basis decomposition | ✗ | ✓ PCA(x0)+SVD(SC) | ✗ | cap-vep novel contribution |
| Per-subject SBI | ✓ `sample_subj_model` | ✓ `PerSubjectModel` | ✗ | |
| Per-parcellation SBI | ✓ loop over 20 parcs | ✗ (single parc) | ✗ | |
| Cohort vs per-subject comparison | ✓ side-by-side diags | ✓ F1 head-to-head | ✗ | |
| Dynamics model | Hopf/MPR | BVEP (2D Epileptor) | 28 models (not BVEP 2D) | |
| Feature extraction | Variance (1 dim) | ptp_z + mean_early_z (324 dim) | Generic | |
| Gain-inverse feature mapping | ✗ | ✓ `gain_norm.T @ sensor_data` | ✗ | Domain-specific |
| Per-simulation z-scoring | ✗ | ✓ (ptp, mean_early) | ✗ | Normalizes spatial rank |
| Population z-scoring | ✗ | ✓ feat_mean/feat_std | ✗ | |
| NaN filtering | ✗ | ✓ (per-sim check) | ✓ (skip on engine error) | |
| GPU batched simulation | ✓ JAX pmap | ✓ JAX vmap (6ms/sim) | ✓ Burn wgpu | |
| NPE-C/MAF trainer | ✓ sbi library (PyTorch) | ✓ sbi library (PyTorch) | ✓ custom Burn MAF | |
| Empirical MVN prior for NPE | ✓ | ✓ | ✗ (uses BoxUniform) | |
| Coverage (90% CI) | ✓ `ci90` | ✗ | ✗ | |
| Posterior sampling | ✓ `posterior.sample_batched` | ✓ `personalize` | ✓ MAF forward | |
| k_per_parc optimization | ✓ pre-tuned per parcellation | ✗ | ✗ | |

### 2.3 Clinical Evaluation

| Feature | vcc | cap-vep | hyburn | Notes |
|---|---|---|---|---|
| Shrinkage diagnostic | ✓ | ✗ | ✓ | |
| z-score diagnostic | ✓ | ✗ | ✓ | |
| Coverage (90% CI) | ✓ | ✗ | ✗ | |
| EZ localization (threshold) | ✗ | ✓ | ✗ | |
| EZ localization (top-k) | ✗ | ✓ | ✗ | |
| F1 / precision / recall | ✗ | ✓ | ✗ | |
| IoU (intersection over union) | ✗ | ✓ | ✗ | |
| Seizure freedom stratification | ✗ | ✓ | ✗ | |
| Clinical ground truth loading | ✗ | ✓ `ei-vep.json` | ✗ | |
| Per-patient comparison table | ✗ | ✓ head-to-head | ✗ | |

### 2.4 Dynamics Models

| Model | vcc | cap-vep | hyburn | Notes |
|---|---|---|---|---|
| Hopf oscillator | ✓ | ✗ | ✗ | Key model for healthy dynamics |
| MPR (multi-pop rate) | ✓ | ✗ | ✓ | |
| Generic 2D Oscillator | ✗ | ✗ | ✓ | Similar to Hopf but different params |
| BVEP (2D Epileptor) | ✗ | ✓ | ✗ | 2-var epileptor, Heun + state clip |
| JansenRit | ✗ | ✗ | ✓ | |
| WilsonCowan | ✗ | ✗ | ✓ | |
| Kuramoto | ✗ | ✗ | ✓ | |
| Epileptor (6-var) | ✗ | ✗ | ✓ | Full 6-var, not 2D BVEP |
| Epileptor2D | ✗ | ✗ | ✓ | Exists but not same as BVEP |
| Reduced FHN | ✗ | ✗ | ✓ | |
| RWW (Wong-Wang) | ✗ | ✗ | ✓ | |
| + 18 more | ✗ | ✗ | ✓ | |

### 2.5 Simulation Infrastructure

| Feature | vcc | cap-vep | hyburn | Notes |
|---|---|---|---|---|
| Heun integrator | ✓ JAX | ✓ numpy + JAX | ✓ | |
| Euler integrator | ✗ | ✗ | ✓ | |
| Heun with state clipping | ✗ | ✓ clip_val=10 | ✗ | cap-vep specific for BVEP |
| Stochastic (SDE) | ✓ `vbjax.make_sde` | ✗ (deterministic) | ✓ noise_mode | |
| CPU simulation | ✓ | ✓ 170ms/sim | ✓ | |
| GPU simulation | ✓ JAX pmap | ✓ JAX vmap 6ms/sim | ✓ Burn wgpu | |
| Time-window averaging | ✓ nwin with scan | ✓ nt=151 steps | ✓ | |
| Connectome normalization | ✓ W / W.max() | ✗ | ✗ (uses raw weights) | |
| Coupling computation | ✓ `K * Σ SC * D` | ✓ `K * Σ SC * (x_j - x_i)` | ✓ generic coupling | |

---

## 3. Data Flow Comparison

### 3.1 vcc Data Flow

```
both.pkl (461 subjects × 20 parcellations)
    │
    ▼
CrossCoder(variational=True, 20 views)
    ├── encode(nlat, parc) → z ∈ ℝ^nlat per subject per parc
    ├── decode_conn(nlat, parc, z) → SC matrix
    ├── calc_mvn(nlat) → MvNorm(mean, cov) over latent space
    └── calc_confusion_rate(nlat) → self-ID rate
         │
         ▼
sample_model_vcc(cc, model, mvn, parc, nlat, num_batch, batch_size)
    ├── z = mvn.sample(batch_size)
    ├── SC = cc.decode_conn(nlat, parc, z)
    ├── K = 0.1 + rand * 0.2
    ├── D = 0.2 + rand * 0.2
    ├── features = model(SC, K, D)           ← dynamics simulation
    └── θ = [z, K, D]                         ← joint parameter vector
         │
         ▼
run_sbi(θ, features) → NPE_C posterior          ← PyTorch sbi library
         │
         ▼
posterior_diags(θ_prior, θ_posterior, θ_true)
    ├── shrinkage: 1 - σ²_post / σ²_prior
    ├── z-score: |μ_post - θ_true| / σ_prior
    └── ci90: fraction of true params in 90% CI
```

### 3.2 cap-vep Data Flow

```
vep_data.pkl (30 patients, SC + SEEG + gain)
ei-vep.json (clinical ground truth: EZ/PZ indices, seizure freedom)
    │
    ├──► CrossCoder (single view, 162 regions, nlat=8)
    │       ├── decode_conn → SC pool (100 synthetic matrices)
    │       └── to_pkl(crosscoder.pkl)
    │
    ├──► build_population_x0(ei-vep.json)
    │       └── X0 ∈ ℝ^(n_patients × 162)  (EZ=+1.5, PZ=+0.5, rest=0)
    │
    ├──► compute_combined_basis(X0, sc_pool)
    │       ├── PCA(X0) → V_pca ∈ ℝ^(162 × K_x0)   ← focal modes
    │       ├── SVD(mean_SC) → V_sc ∈ ℝ^(162 × K_sc) ← global modes
    │       └── V_combined = [V_pca | V_sc] ∈ ℝ^(162 × K_total)
    │
    ├──► sample_theta_prior(key, batch_size, nn, V_x0, x0_mean, ...)
    │       ├── alpha ~ N(0, 1/sqrt(sv_pca))  ← PCA coefficients
    │       ├── K ~ |N(0,1)| * 0.5 + 0.5     ← global coupling
    │       ├── τ₀ ~ |N(0,1)| * 5 + 10       ← time scale
    │       ├── x0_mu_base ~ mixture(-2.3, -2.7) ← baseline excitability
    │       └── θ = [alpha, K, τ₀, x0_mu_base]
    │
    ├──► VEPModel.simulate_one(θ, SC=random from pool)
    │       ├── x0 = x0_mu_base + V_combined @ alpha
    │       ├── BVEP_dfun(x, z, c, x0, τ₀)  ← 2D Epileptor
    │       ├── Heun integration (nt=151, dt=0.1)
    │       └── features = [ptp_z(162), mean_early_z(162)]  ← z-scored per sim
    │
    ├──► train_sbi(θ, features) → NPE_C posterior
    │
    ├──► Personalize: posterior.sample(100, x=patient_features)
    │       └── x0_posterior ∈ ℝ^(100 × 162)
    │
    └──► localize_ez(x0_posterior, threshold=-2.05 or top_k=3)
            ├── compute_ez_accuracy(pred_ez, true_ez)
            └── F1, precision, recall, IoU
```

### 3.3 hyburn Data Flow

```
SimConfig (TOML) — single parcellation
    │
    ├──► HybridEngine::from_config(cfg, device)
    │       ├── construct subnetworks + projections
    │       ├── precompute weights, coupling, target_cvar_cpl
    │       └── allocate GPU tensors
    │
    ├──► engine.run(n_steps) → trajectory [n_steps, nvar, nnodes, nmodes]
    │
    ├──► extract_features_with(traj, shape, feature_set)
    │
    ├──► CrossCoder (optional, loaded from .cc.bin)
    │       ├── encode cohort → consensus latents
    │       ├── fit_mvn → MvnPrior (mean, cov, chol)
    │       ├── mvn.sample → latent codes
    │       └── decode → SC matrices
    │
    ├──► crosscoder_pipeline: SC → inject SimConfig → simulate → features
    │       └── trains MAF on flat SC matrices (nnodes² param dim)
    │
    └──► SbiDiagnostics: z-score, shrinkage (no coverage, no clinical eval)
```

**Key structural difference**: In hyburn, the CrossCoder pipeline treats
the decoded SC as a flat parameter vector (nnodes² dims). In vcc and
cap-vep, the latent code itself is the parameter (nlat or K_total dims,
both << nnodes²), and dynamics parameters are appended.

---

## 4. Parameterization Analysis

### 4.1 Parameter Dimensions

| System | Parameter vector | Dim (162 regions) | Dim (74 regions) | Notes |
|---|---|---|---|---|
| vcc | [z(nlat), k, D] | 18 | 18 | nlat=16 + 2 dynamics |
| cap-vep | [α(K_total), K, τ₀, μ] | 23 | — | K_total=20 + 3 |
| hyburn (current) | flat SC matrix | 26244 | 5476 | nnodes² |
| hyburn (latent, proposed) | [z(nlat), dynamics] | ~20 | ~18 | Matches vcc pattern |

The difference is 3 orders of magnitude. A 26K-dim parameter space is
essentially uninformable from 324-dim features. The vcc/cap-vep approach
of using latent codes as parameters is not optional—it's fundamental to
making SBI tractable.

### 4.2 Spatial Basis (cap-vep specific)

The `V_combined = [PCA(x0) | SVD(SC)]` basis is a key innovation:

- **PCA on population x0**: Captures focal spatial patterns from clinical
  ground truth (EZ/PZ masks). 8 components capture 78.3% of population
  x0 variance. These are *focal*—they represent specific brain regions
  being epileptogenic.

- **SVD of mean SC**: Captures global structural connectivity modes.
  12 SC eigenvectors was the sweet spot (n_sc_comp sweep). These are
  *smooth*—they represent brain-wide connectivity gradients.

- **Interaction**: The SC eigenvectors in the x0 basis help SBI learn
  that x0 patterns should correlate with structural connectivity topology.
  Adding them improves F1 from 0.41 (PCA-only) to 0.51 (PCA+SC(12)).

- **Relevance to hyburn**: For the Epileptor model, a similar x0
  parameterization could replace per-region independent x0 sampling.
  V_combined would need to be computed from the cohort's clinical data.

---

## 5. Feature Extraction Comparison

### 5.1 Current Features

| System | Feature | Dim | Computation |
|---|---|---|---|
| vcc (Hopf) | Var(x_fast) | 1 | Variance of first variable across time × nodes |
| vcc (MPR) | Mean(x_rate) | nnodes | Mean of second variable across time |
| cap-vep | [ptp_z, mean_early_z] | 2×nnodes | Peak-to-peak + mean early x, z-scored per sim |
| hyburn | Generic feature set | varies | Var, mean, spectral, etc. from feature enum |

### 5.2 Key Feature Engineering Insights (cap-vep)

1. **Per-simulation z-scoring**: Different simulations produce features
   at wildly different absolute scales. Z-scoring within each sim
   preserves spatial rank ordering (which regions have highest activity).
   This is critical for amortized inference across diverse SC matrices.

2. **Gain-inverse mapping**: `src = gain_norm.T @ sensor_data` maps SEEG
   sensor data to brain source space. This makes sim features and data
   features live in the same space. Without it, x0↔ptp correlation
   drops from 0.93 (source) to 0.01 (after gain round-trip).

3. **Early time window**: `mean_early` captures the initial seizure
   propagation before all regions have synchronized. Combined with ptp,
   this provides two views of the spatial seizure pattern.

4. **Population z-scoring on top**: After per-sim normalization,
   apply population-level z-scoring for SBI training stability.

### 5.3 Feature-Parameter Correlation

vcc uses very low-dimensional features (1 dim for Hopf, nnodes for MPR).
This works because the parameter space is also low-dimensional (nlat+2).
cap-vep uses 2×nnodes features for K_total+3 parameters—roughly balanced.

hyburn's current pipeline would use flat SC (nnodes² params) with generic
features. The parameter-to-feature ratio is >10:1, making inference
identifiability extremely challenging.

---

## 6. Evaluation Metrics Comparison

### 6.1 Statistical Diagnostics (simulation-based)

| Metric | Definition | vcc | cap-vep | hyburn |
|---|---|---|---|---|
| Shrinkage | 1 - σ²_post / σ²_prior | ✓ | ✗ | ✓ |
| z-score | |μ_post - θ_true| / σ_prior | ✓ | ✗ | ✓ |
| 90% CI coverage | P(θ_true ∈ [q5, q95]) | ✓ | ✗ | ✗ |
| Posterior calibration | SBC ranks | ✗ | ✗ | ✗ |

### 6.2 Clinical Metrics (domain-specific)

| Metric | Definition | vcc | cap-vep | hyburn |
|---|---|---|---|---|
| EZ precision | |pred ∩ true| / |pred| | ✗ | ✓ | ✗ |
| EZ recall | |pred ∩ true| / |true| | ✗ | ✓ | ✗ |
| F1 | 2·P·R / (P+R) | ✗ | ✓ | ✗ |
| IoU | |pred ∩ true| / |pred ∪ true| | ✗ | ✓ | ✗ |
| Seizure freedom stratification | SF vs NSF analysis | ✗ | ✓ | ✗ |

### 6.3 CrossCoder-Specific Metrics

| Metric | vcc | cap-vep | hyburn |
|---|---|---|---|
| Confusion rate (self-ID) | ✓ | ✗ | ✗ |
| Per-parcellation reconstruction r | ✓ | ✗ | ✗ |
| Latent space SVD structure | ✓ (3-4 dominant) | ✗ | ✗ |
| One-hot decoded patterns | ✓ | ✗ | ✗ |
| Best-match correlation vs vbjax | ✗ | ✗ | ✓ |

---

## 7. API Pattern Analysis

### 7.1 Common Pipeline Pattern

All three systems share this structure, despite different implementations:

```
1. Load cohort connectomes → fit CrossCoder → MVN over latent space
2. Prior:   sample z ~ MVN → decode to SC matrix
3. Dynamics: simulate(SC, theta_dynamics) → trajectory
4. Features: extract(trajectory) → feature vector
5. Training: NPE(theta_joint, features) → posterior estimator
6. Inference: posterior.sample(x=new_data) → parameter estimates
7. Evaluation: compare estimates to ground truth
```

### 7.2 Where They Diverge

**Step 2 (Prior)**: 
- vcc: z directly from MVN, append [k, D]
- cap-vep: alpha_x0 from scaled Gaussian, append [K, τ₀, μ_base], x0 = V @ alpha + μ_base
- hyburn: flat SC decoded from z, no dynamics params appended

**Step 3 (Dynamics)**:
- vcc: JAX `make_sde` with `lax.scan` time-stepping
- cap-vep: numpy Heun with state clipping, or JAX vmap for GPU
- hyburn: Burn backend (ndarray/wgpu/cuda) with generic HybridEngine

**Step 5 (Training)**:
- vcc/cap-vep: `sbi.NPE_C` (PyTorch, mature, GPU)
- hyburn: custom MAF (Burn, less validated, but self-contained)

**Step 7 (Evaluation)**:
- vcc: shrinkage/z-score/coverage on continuous parameters
- cap-vep: F1/IoU on discrete EZ localization against clinical ground truth
- hyburn: shrinkage/z-score on flat SC reconstruction (weaker signal)

### 7.3 CrossCoder API Differences

| Operation | vcc | cap-vep | hyburn |
|---|---|---|---|
| Create | `CrossCoder(variational=True)` | `CrossCoder(variational=True, chunked_training=True)` | `CrossCoder::new(device, input_dims, latent_dim, beta)` |
| Add view | `cc.add_view(data, name, normalize='zscore')` | `cc.add_view(triu, 'vep162', normalize='center')` | Construct with `input_dims` array |
| Train | `cc.train(nlat, lr, niter, beta_end, anneal_steps)` | `cc.train(nlat, lr, niter, key)` | `train_crosscoder(data, shapes, &cfg, ...)` |
| Encode | `cc.encode(nlat, parc, tts=cc.tts)` | `cc.encode(0, 'vep162', z, sample=True)` | `model.views[i].encode(x)` per-view |
| Decode | `cc.decode_conn(nlat, parc, z)` | `cc.decode_conn(0, 'vep162', z, clip_positive=True)` | `model.views[i].decode(z)` per-view |
| MVN | `cc.calc_mvn(nlat)` | manual | `fit_mvn_over_latents(flat, n, dim)` |
| Save | `cc.to_pkl(path)` | `cc.to_pkl(path)` | `model.save(path)` |
| Load | `CrossCoder.from_pkl(path)` | `CrossCoder.from_pkl(path)` | `load_crosscoder(path, device, dims, ...)` |

**Key friction**: hyburn's CrossCoder doesn't track parcellation names,
normalization, or train/test splits. Each encode/decode must specify
the view index manually. vcc/cap-vep use string-based parcellation names.

---

## 8. Gap Summary (hyburn vs both reference systems)

### 8.1 Critical Gaps (blocks core use case)

| # | Gap | vcc | cap-vep | Impact |
|---|---|---|---|---|
| G1 | No joint [latent+dynamics] SBI training | ✓ | ✓ | hyburn's CC pipeline trains on flat SC (nnodes² dims)—uninformable from low-dim features |
| G2 | No latent code as parameter | ✓ | ✓ | Must append dynamics params to latent vector; current pipeline only uses SC as param |
| G3 | No per-subject SBI baseline | ✓ | ✓ | Can't measure amortization benefit without per-subject comparison |

### 8.2 High Gaps (needed for publication-quality results)

| # | Gap | vcc | cap-vep | Impact |
|---|---|---|---|---|
| G4 | No confusion rate metric | ✓ | — | Key cross-coder validation for paper |
| G5 | No coverage (90% CI) diagnostic | ✓ | — | Standard SBC metric |
| G6 | No clinical evaluation (F1, IoU, EZ) | — | ✓ | Needed for epilepsy use case |
| G7 | No spatial basis parameterization | — | ✓ | Reduces param dim from nnodes² to K_total; enables focal x0 patterns |
| G8 | No per-simulation feature z-scoring | — | ✓ | Normalizes feature scale across diverse SCs |
| G9 | No gain-inverse feature mapping | — | ✓ | Maps sensor → source space for SEEG |

### 8.3 Medium Gaps (quality and completeness)

| # | Gap | vcc | cap-vep | Impact |
|---|---|---|---|---|
| G10 | No β-annealing schedule | ✓ | — | Training quality; fixed β can cause KL collapse |
| G11 | No per-parcellation SBI benchmarking | ✓ | — | Demonstrates parcellation-invariance |
| G12 | No parcellation naming/indexing | ✓ | ✓ | API ergonomics; string-based > index-based |
| G13 | No per-view normalization pipeline | ✓ | ✓ | Pre-processing convenience |
| G14 | No Hopf oscillator model | ✓ | — | Used in healthy-dynamics SBI papers |
| G15 | No BVEP (2D Epileptor) simulator variant | — | ✓ | Different from 6-var Epileptor |
| G16 | No k_per_parc tuning | ✓ | — | Optimal coupling per parcellation size |

### 8.4 Low Gaps (nice to have)

| # | Gap | vcc | cap-vep | Impact |
|---|---|---|---|---|
| G17 | No SC positive clipping | — | ✓ | BVEP needs non-negative SC |
| G18 | No chunked training | ✓ | — | GPU/TPU memory management |
| G19 | No empirical MVN prior for NPE | ✓ | ✓ | Better NPE training than BoxUniform |
| G20 | No SBC rank-based calibration | — | — | Gold standard but expensive |

---

## 9. Design Patterns Worth Preserving

### 9.1 From vcc

1. **Multi-view CrossCoder with confusion rate**: The 20-parcellation
   CrossCoder with self-identification rate is a novel validation tool.
   hyburn's generic multi-view architecture supports this, but needs the
   metrics.

2. **`bench_model_vcc(return_everything=True)`**: Returning the full
   locals dict for interactive inspection is very convenient for notebook
   workflows. hyburn's pipeline returns only (params, features).

3. **Inflated MVN covariance**: `mvn.cov = mvn.cov * inflate` widens the
   latent prior slightly, improving NPE training stability. Simple but
   effective.

4. **Separate `sample_model_vcc` and `sample_subj_model`**: Clean
   separation between cohort-level (SC sampled from latent) and
   per-subject (SC fixed) simulation.

### 9.2 From cap-vep

1. **Combined spatial basis**: `V_combined = [V_pca | V_sc]` is the most
   impactful finding. PCA provides focal patterns, SVD(SC) provides
   global modes. The combination outperforms either alone.

2. **Per-simulation z-scoring before population z-scoring**: Two-stage
   normalization: first within each simulation (removes absolute scale
   differences across SC matrices), then across the cohort (standardizes
   for NPE training). This is critical for amortized inference.

3. **Gain-inverse as preprocessing, not model component**: Instead of
   including the gain matrix in the generative model (expensive,
   patient-specific), apply gain-inverse to the data and work entirely in
   source space. This decouples the simulator from patient-specific
   sensor geometry.

4. **Mixture prior for x0_mu_base**: Single Gaussian produces unrealistically
   uniform EZ sizes. Mixture prior (70% moderate, 30% low baseline) gives
   plausible EZ size distribution (1-12 regions).

5. **Top-k EZ selection vs thresholding**: Thresholding at -2.05 gives
   variable EZ sizes (sometimes 0, sometimes 20+). Top-3 selection gives
   stable, clinically plausible predictions.

6. **State clipping in BVEP**: `clip(x, -10, 10)` prevents numerical
   divergence during Heun integration without affecting steady-state
   dynamics. Simple but essential for batch robustness.

### 9.3 From hyburn

1. **Type-safe CrossCoder in Burn**: The Rust/Burn CrossCoder is
   type-safe, serializable, and GPU-agnostic (ndarray/wgpu/cuda). This
   is a significant engineering advantage over Python's dynamic typing.

2. **Generic feature extraction via `FeatureSet` enum**: Extensible
   feature extraction without hardcoding. vcc/cap-vep use ad-hoc
   feature functions.

3. **`target_cvar_cpl` mapping**: The coupling variable index mapping
   (recently added) correctly handles multi-cvar models. vcc/cap-vep
   assume single coupling variables.

4. **28 neural mass models**: Far broader model library than either
   reference system. The infrastructure for adding new models is
   well-established.

5. **WASM target**: hyburn can run in the browser. No Python SBI system
   can do this.

---

## 10. Quantitative Benchmarks

### 10.1 Simulation Speed

| System | Backend | Time/sim | Batch size | Notes |
|---|---|---|---|---|
| vcc (Hopf) | JAX pmap (8 GPU) | ~1ms | 128 | 1000 time steps |
| vcc (MPR) | JAX pmap (8 GPU) | ~2ms | 128 | 1000 time steps |
| cap-vep (BVEP) | JAX vmap (1 GPU) | 6ms | 256 | 151 time steps |
| cap-vep (BVEP) | numpy CPU | 170ms | 1 | 151 time steps |
| hyburn | ndarray CPU | varies | 1 | Config-dependent |
| hyburn | wgpu | varies | 1 | 10 min compile |

### 10.2 SBI Training Speed

| System | Library | N sims | θ dim | Feature dim | Train time |
|---|---|---|---|---|---|
| vcc (Hopf) | sbi NPE_C | 4096 | 18 | 1 | ~2 min |
| vcc (MPR) | sbi NPE_C | 4096 | 18 | nnodes | ~5 min |
| cap-vep (BVEP) | sbi NPE_C GPU | 5000 | 23 | 324 | ~80s |
| cap-vep (BVEP) | sbi NPE_C CPU | 5000 | 23 | 324 | ~8 min |
| hyburn | Burn MAF | varies | varies | varies | TBD |

### 10.3 Total Pipeline Time

| System | Phases | Total time |
|---|---|---|
| vcc | CC train + SBI train + eval | ~15 min |
| cap-vep (GPU) | CC(60s) + Sim(30s) + Train(80s) | ~3 min |
| cap-vep (CPU) | CC(60s) + Sim(850s) + Train(480s) | ~25 min |
| hyburn | CC + build + sim + features + MAF | TBD |

---

## 11. CrossCoder Mathematical Details

### 11.1 vcc/cap-vep (vbjax) CrossCoder

**Encoder** (variational):
```
μ = x @ W_enc_μ + b_enc_μ   ∈ ℝ^nlat
log σ² = x @ W_enc_logvar + b_enc_logvar   ∈ ℝ^nlat
z = μ + σ ⊙ ε,  ε ~ N(0, I)
```

**Decoder**:
```
x̂ = z @ W_dec + b_dec   ∈ ℝ^d
```

**Loss**:
```
L = Σ_i Σ_j MSE(decode_j(z_i), x_j) + β · KL(q_i || N(0, I))
```
where β is annealed from 0 to β_end over anneal_steps iterations.

**Normalization**: Per-view center/zscore/logit with `_denorm` inverse.

### 11.2 hyburn CrossCoder

**Encoder** (same variational, unified weights):
```
[μ || log σ²] = x @ W_enc + b_enc   ∈ ℝ^(2·nlat)
μ = [0:nlat], log σ² = [nlat:2·nlat]
```

**Decoder**:
```
x̂ = z @ W_dec + b_dec   ∈ ℝ^d
```

**Loss**: Same structure but fixed β (no annealing).

**Difference**: hyburn uses a single encoder weight matrix for both μ and
log σ², while vbjax uses separate `W_enc_μ` and `W_enc_logvar`. This is
equivalent (the split happens after the linear layer) but affects
initialization and weight saving format.

---

## 12. File Format Compatibility

| Format | vcc | cap-vep | hyburn | Convertible? |
|---|---|---|---|---|
| CrossCoder `.pkl` (vbjax) | ✓ read/write | ✓ read/write | ✗ | Need Python bridge |
| CrossCoder `.cc.bin` (Burn) | ✗ | ✗ | ✓ read/write | Need Rust bridge |
| NPY (numpy) | ✓ | ✓ | ✓ (via `io::read_npy_f32`) | ✓ |
| NPY (scipy) | ✓ | ✓ | ✓ | ✓ |
| TOML (SimConfig) | ✗ | ✗ | ✓ | hyburn-specific |
| JSON (ground truth) | ✗ | ✓ `ei-vep.json` | ✗ | Would need schema |
| NPZ (diagnostics) | ✓ | ✗ | ✗ | Need npz reader |

The `.pkl` ↔ `.cc.bin` conversion is the main interoperability gap.
A Python script could load vbjax `.pkl`, extract W_enc/W_dec/b_enc/b_dec,
save as NPY arrays, then hyburn loads NPY and constructs Burn tensors.

---

## 13. Model-Specific Considerations

### 13.1 BVEP (2D Epileptor) for hyburn

cap-vep's BVEP is a 2-variable model:

```
dx = 1 - x³ - 2x²z
dz = (-z + I₁ + x₀ + 3x + c) / τ₀
```

where c = K · Σ_j SC_ij · (x_j - x_i) is difference coupling.

hyburn has `Epileptor` (6-var) and `Epileptor2D` (2-var). The `Epileptor2D`
model should be checked for formula compatibility with cap-vep's BVEP.
Key differences to verify:
- Coupling type: cap-vep uses difference coupling (c = K·ΣSC·(x_j-x_i))
- State clipping: cap-vep clips both x and z to [-10, 10]
- Parameters: cap-vep uses x₀, K, τ₀, I₁=3.1

### 13.2 Hopf Oscillator for hyburn

vcc's Hopf model has parameters η (bifurcation), ω (frequency), K (coupling), D (noise):

```
dx = (η - x² - y²)x - ωy + K·ΣSC·(x_j - x_i) + D·ξ_x
dy = (η - x² - y²)y + ωx + K·ΣSC·(y_j - y_i) + D·ξ_y
```

This is different from hyburn's `Generic2dOscillator` which uses α, β, γ, τ parameters.
The Hopf model is a specific case; adding it would broaden comparability.

### 13.3 Feature Extraction for Epileptor Models

cap-vep's feature pipeline (ptp_z + mean_early_z) is tailored to the
BVEP model's seizure dynamics. For hyburn's Epileptor models, similar
features could be defined but may need adaptation for the 6-variable
model's richer dynamics. The key insight—per-simulation z-scoring to
preserve spatial rank ordering—is model-agnostic.

---

## 14. Summary Statistics

| Metric | vcc | cap-vep | hyburn |
|---|---|---|---|
| Language | Python/JAX | Python/JAX+numpy | Rust/Burn |
| CrossCoder lines | ~200 (vbjax lib) | 56 | 378 |
| SBI pipeline lines | ~150 (vcc_utils) | 198 | 323 |
| Total SBI code | ~350 | ~1263 | ~3520 |
| Models supported | 2 (Hopf, MPR) | 1 (BVEP) | 28 |
| Multi-view CC | ✓ 20 parcs | ✗ 1 view | ✓ generic |
| Joint latent+dynamics SBI | ✓ | ✓ | ✗ |
| Clinical eval | ✗ | ✓ | ✗ |
| GPU accel | ✓ JAX | ✓ JAX | ✓ Burn |
| Browser (WASM) | ✗ | ✗ | ✓ |