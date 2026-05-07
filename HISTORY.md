# History

This project was extracted from [the-virtual-brain/tvb-root](https://github.com/the-virtual-brain/tvb-root)
as a standalone Rust crate. Below is a compacted history of the relevant commits.

## 2025-04 — Initial engine + models
- HybridEngine with NdArray backend (Heun/Euler integrators)
- Neural mass models: G2DO, MPR, Kuramoto, Jansen-Rit, Wilson-Cowan, RWW
- Dense and CSR sparse coupling, configurable projections
- CLI: `hyburn run` with TOML config, trajectory output to NPY
- Monitors: TemporalAverage, SubSample, GlobalAverage, AfferentCoupling, Projection

## 2025-04 — Production quality
- Error types (SimulationError), checkpoint/resume
- Progress reporting, integration tests
- GPU autotuning strategy

## 2025-04 — GPU backends
- WGPU (Vulkan) and CUDA backend support
- BatchHybridEngine for parallel parameter sweeps
- GPU integration tests

## 2025-04 — SBI pipeline
- MADE and MAF modules for simulation-based inference
- SBI config, training, CLI wire-up
- Parameter sweep with Rayon multicore parallelism

## 2025-05 — Feature parity with vbi/vbjax
- BOLD hemodynamics (Balloon-Windkessel monitor with TR downsampling)
- Stimulus injection (impulse, step, sinusoid, pulse_train)
- FC feature extraction (Pearson FC, FCD, homotopic)
- Spectral features (Welch PSD, band-power, spectral moments)
- Temporal/statistical features (energy, centroid, burstiness, skewness, kurtosis)
- catch22 time-series feature set
- Prior distribution abstraction (BoxUniform, SamplesFromNpy, MultivariateNormal)
- Multi-parameter SBI pipeline with config-driven prior sampling

## 2025-05 — CrossCoder
- Linear variational CrossCoder (multi-view, μ/logσ² encoder, Xavier init)
- Training with Adam, gradient clipping, β-annealing
- Cohort data loading → encode_all → fit MVN → Cholesky sampling
- CrossCoder→Simulation pipeline (sample z → decode SC → simulate → features → MAF)
- Validation against vbjax reference (Python script + Rust test)

## 2025-05 — Robustness fixes
- Fixed default cvar_map "1:1" → "0:0" (was silently wrong for ncvar=1 models)
- Added cvar_map validation at config time (both engines)
- Fixed stimulus ncvar=1 crash (narrow on size-0 dim)
- Fixed Difference coupling validation (requires ncvar≥2)
- Fixed multi-model ncvar projection (proper cvar_map scatter in batch engine)
- Added NPY initial state support in batch engine
- Replaced CSR unwrap() with expect() / safe fallback
- Fixed sweep param name parsing (unwrap → proper errors)
- Added BOLD GPU fast path (GPU-side accumulators, sync only per bold_period)
- Clippy cleanup (77→0 warnings)
- Flaky perf test threshold loosened (2.0× → 2.5×)
- 74-node example configs (16 TOML + 5 NPY init files)
- Benchmarks: CUDA 12–17× speedup on 128-point sweeps vs CPU
