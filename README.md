# hyburn

Burn-based GPU hybrid neural mass simulator with simulation-based inference (SBI).

## What it does

Hyburn simulates whole-brain neural mass models on GPU (CUDA/WGPU) or CPU (NdArray), runs parameter sweeps, extracts time-series features, and trains Masked Autoregressive Flows (MAF) for simulation-based inference — all from a single Rust binary.

## Quick start

```bash
cargo build --release
./target/release/hyburn run -c examples/demo.toml -o output/demo
```

### With GPU backends

```bash
cargo build --release --features cuda    # NVIDIA CUDA
cargo build --release --features wgpu    # Vulkan (cross-platform GPU)
./target/release/hyburn run -c examples/demo.toml -o output/demo --backend cuda
```

### Parameter sweeps

```bash
# 128-point sweep (CPU)
./target/release/hyburn run -c examples/g2do_sweep.toml \
  -o output/sweep --sweep examples/sweep.toml --backend ndarray

# 128-point sweep (CUDA — ~14× faster than CPU)
./target/release/hyburn run -c examples/g2do_sweep.toml \
  -o output/sweep --sweep examples/sweep.toml --backend cuda
```

### SBI pipeline

```bash
./target/release/hyburn pipeline \
  -c examples/demo.toml -p examples/pipeline_prior.toml \
  -o output/pipeline
```

## Neural mass models

| Model | State vars | Coupling vars | Description |
|-------|----------:|--------------:|-------------|
| Generic2dOscillator | 2 | 1 | Canonical TVB oscillator |
| MontbrioPazoRoxin | 2 | 2 | Mean-field spiking network |
| JansenRit | 6 | 1 | Cortical column (E/I) |
| WilsonCowan | 2 | 1 | Excitatory-inhibitory rate model |
| Kuramoto | 1 | 1 | Phase oscillator |
| ReducedWongWang | 1 | 1 | Bistable mean-field |

## Features

- **Multi-backend:** NdArray (CPU), WGPU (Vulkan), CUDA (NVIDIA)
- **Batch sweeps:** `BatchHybridEngine` processes all sweep points in a single tensor — 12–17× speedup on GPU
- **BOLD monitor:** Balloon-Windkessel hemodynamics with TR downsampling
- **Stimulus injection:** Impulse, step, sinusoid, pulse train
- **CrossCoder:** Multi-view variational autoencoder for cohort connectome priors
- **SBI pipeline:** BoxUniform/MultivariateNormal priors → simulation → feature extraction → MAF training → diagnostics
- **Feature extraction:** FC, FCD, spectral (Welch PSD, band-power), temporal statistics, catch22
- **Checkpoint/resume:** Binary checkpoint files for long simulations
- **NPY I/O:** Numpy-compatible .npy files for all outputs

## Examples

See `examples/` for 16+ configuration files covering single-model sims, BOLD, stimulus, sweeps, multi-model networks, and CrossCoder pipelines.

## Benchmarks

| Scenario | Backend | Time | Speedup |
|----------|---------|------|--------:|
| Single sim, 74 nodes, 10k steps | CPU | 0.67s | 1.0× |
| Single sim, 74 nodes, 10k steps | CUDA | 4.4s | 0.15× |
| 128-point sweep, 74 nodes | CPU | 26.7s | 1.0× |
| 128-point sweep, 74 nodes | **CUDA** | **1.6s** | **17×** |

GPU backends dominate for batch sweeps; CPU is faster for single simulations due to kernel launch overhead. See `docs/BENCHMARK_EXAMPLES.md` for full results.

## Building

```bash
cargo build --release                          # CPU only
cargo build --release --features wgpu         # + Vulkan GPU
cargo build --release --features cuda         # + NVIDIA CUDA
cargo build --release --features "wgpu,cuda"  # all backends
```

Requires Rust 1.85+ (edition 2024). CUDA requires CUDA toolkit 12+ and a compatible NVIDIA GPU.

## Testing

```bash
cargo test --lib                              # 184 unit/integration tests
cargo test --lib --features cuda              # + GPU tests
cargo clippy --lib                            # zero warnings
```

## License

GPL-3.0 — see [LICENSE](LICENSE).
