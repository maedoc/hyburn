//! Generic batch-dim GPU sweep engine.
//!
//! Unlike `sweep_gpu::batch_sweep_3subnet` (hardcoded for a specific network),
//! this module provides a **generic** `BatchHybridEngine` that reads a `SimConfig`
//! and runs all sweep points as a batch. Any network topology supported by
//! `SimConfig` works — different models, coupling patterns, node counts.
//!
//! Architecture:
//! - Adds a leading batch dimension `[n_sweep]` to all tensors
//! - State: `[n_sweep, nnodes, nvar]` per subnetwork  (note: actually `[n_sweep, nnodes*nmodes, nvar]`)
//! - Coupling: computed per-projection across batch
//! - Supported models: G2DO, JansenRit, WilsonCowan, Mpr, Kuramoto, Rww (batch-native)
//!   Other models delegate to the 2D dfun per-sweep-point (slower fallback)
//! - Supported coupling: all-to-all scalar, dense weight matrix
//!   (delayed coupling and sparse CSR will fall back to serial)

pub mod engine;
pub(crate) mod dfun;
pub(crate) mod projection;

pub use engine::{
    BatchHybridEngine, BatchSweepResult, SweepParam, rayon_batch_sweep,
};
