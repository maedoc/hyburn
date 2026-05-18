# AGENTS.md — Project conventions for AI assistants

## Environment

- Rust might not be on the default PATH. Check and then prefix cargo/rustc commands with:
  ```
  PATH="$HOME/.cargo/bin:$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"
  ```
  Or use the full path: `$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin/cargo`
- Rust edition 2024, requires Rust 1.85+
- Platform: macOS aarch64 (Apple Silicon) for local dev

## Build Commands

```bash
cargo build --release                                    # CPU (ndarray) only
cargo build --release --features wgpu                    # + Vulkan/Metal GPU
cargo build --release --features cuda                   # + NVIDIA CUDA (needs toolkit 12+)
cargo build --release --features "wgpu,cuda"            # all backends
cargo build --release --no-default-features --features "cli,report,parallel"  # ndarray only, explicit
```

WGPU builds are slow (~10 min on M-series). Use timeout >= 600000ms for bash commands.

## Test Commands

```bash
cargo test --lib                                         # unit/integration tests (~293)
cargo test --lib --features cuda                        # + GPU tests (needs CUDA GPU)
cargo clippy --lib -- -D warnings                       # lint
```

## Reference Trace Generation

Reference `.npy` traces are generated with the TVB hybrid simulator for numerical validation.

```bash
./ref/setup.sh              # clone tvb-root (hybrid-numba), create venv, install deps
./ref/setup.sh --check      # verify installation
ref/venv/bin/python ref/generate_single_sim.py          # small 2-node configs
ref/venv/bin/python ref/generate_single_sim.py --full   # include 74-node config
ref/venv/bin/python ref/generate_sweep.py                # parameter sweeps
ref/venv/bin/python ref/generate_bold.py                 # BOLD monitor outputs
ref/venv/bin/python ref/generate_features.py             # feature extraction
```

- Venv at `ref/venv/`, tvb-root at `ref/tvb-root/` (both git-ignored). Run `./ref/setup.sh` to create.
- Requires: [uv](https://docs.astral.sh/uv/), git.
- Teardown: `rm -rf ref/venv ref/tvb-root`.

## WASM Build

```bash
pip install numpy tomli                                 # needed for gen_presets.py
python3 scripts/gen_presets.py                           # generate preset JSON into src/presets.rs
wasm-pack build --target web --no-default-features --features wasm
```

WASM output goes to `pkg/`. For the web app, copy `pkg/*` to `web/pkg/`.

## WASM E2E Tests (Playwright)

```bash
cd web
npm install
npx playwright install --with-deps firefox
npx playwright test --reporter=list
```

- Uses **Firefox only** (Chromium headless has externref table grow bug with wasm-bindgen 0.2.120)
- Config: `web/playwright.config.ts` — firefox project, python3 http.server on port 8080
- Test file: `web/tests/app.spec.ts` — 14 tests
- Harness: `web/test-harness.html` — minimal page (not index.html) to avoid script conflicts
- Each `page.evaluate` self-contains WASM init (`await mod.default()`) to work around Chromium bug

## Binary Sizes (reference)

| Build | Stripped size | ZIP (CI artifact) |
|-------|--------------|-------------------|
| ndarray only (macOS) | ~4 MB | ~2 MB |
| ndarray + wgpu (macOS) | ~15 MB | ~6 MB |
| ndarray + wgpu + cuda (Linux) | ~20 MB | ~7 MB |

CI artifact sizes are ZIP-compressed. The backends ARE included — the compressed size is misleading.

## Cargo Features

```toml
default = ["cli", "report", "parallel"]
wgpu = ["dep:burn-wgpu"]          # Vulkan/Metal GPU backend
cuda = ["dep:burn-cuda"]          # NVIDIA CUDA backend
cli = ["dep:clap", "dep:env_logger", "dep:rayon"]
report = ["dep:plotters"]         # HTML SBI report generation
parallel = ["dep:rayon"]
wasm = ["dep:wasm-bindgen", "dep:js-sys", "dep:web-sys", "dep:console_log", "dep:getrandom"]
```

- NdArray backend is always compiled in (not optional)
- `--features wgpu` is **additive** on top of defaults (does NOT replace them)
- The binary `hyburn` requires `cli` feature (in `required-features`)
- WASM builds must use `--no-default-features --features wasm` to exclude native-only deps

## CLI Subcommands

| Command | Description |
|---------|-------------|
| `run` | Run a simulation (`--backend ndarray\|wgpu\|cuda`) |
| `benchmark` | Benchmark a config |
| `pipeline` | End-to-end SBI: sweep → features → train → infer |
| `sbi-report` | Generate self-contained HTML diagnostic report |
| `train-sbi` | Train MAF conditional density estimator |
| `infer` | Run inference with trained SBI model |
| `autotune` | Autotune GPU kernel parameters |

## Project Structure

```
src/
  main.rs          # entry point (requires cli feature)
  cli.rs           # clap derive CLI (1100+ lines, all subcommands)
  config.rs        # SimConfig deserialization
  engine/          # HybridEngine, BatchHybridEngine
  model/           # 6 neural mass models (g2do, mpr, jansen_rit, wilson_cowan, kuramoto, rww)
  sbi/             # MAF, feature extraction, diagnostics, priors
  report.rs        # HTML SBI report generation (plotters SVG)
  presets.rs       # Auto-generated WASM preset configs (gen_presets.py)
  wasm.rs          # WASM bindings (WebEngine)
  io.rs            # NPY I/O
  bin/             # Standalone benchmark binaries
examples/          # 16+ TOML config files
web/               # WASM web app + Playwright E2E tests
scripts/           # gen_presets.py, build.sh
docs/              # Benchmarks, sample reports
```

## CI (GitHub Actions)

Workflow: `.github/workflows/ci.yml` with 5 jobs:
1. **Test** (ubuntu-latest) — `cargo test --lib`, clippy
2. **Test (macOS)** (macos-14) — `cargo build --release --features wgpu`, test
3. **Build macOS** (macos-14) — `cargo build --release --features wgpu`, strip, upload artifact
4. **Build Linux** (ubuntu-22.04) — CUDA 12.6 + Vulkan, `cargo build --release --features "wgpu,cuda"`, strip, upload artifact
5. **WASM E2E** (ubuntu-latest) — wasm-pack build, Playwright (Firefox)

The WASM build fails sometimes, remember to investigate.

Release job triggers on `v*` tags, downloads both artifacts, creates GitHub Release.

## Key Gotchas

- The `WebEngine` constructor takes a JSON string directly: `new WebEngine(jsonConfig)`. `from_json` does NOT exist as a constructor — `from_toml` is a static method for TOML input.
- wasm-bindgen 0.2.120 uses externref tables that fail in headless Chromium with `WebAssembly.Table.grow(): failed to grow table by 4`. Use Firefox for browser tests.
- `initial_state` can be an inline array or a path to a `.npy` file.
- `cargo metadata` may fail if PATH isn't set correctly on this machine.
