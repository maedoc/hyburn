pub mod model;
pub mod engine;
pub mod io;
pub mod config;
pub mod sbi;
pub mod error;

#[cfg(feature = "cli")]
pub mod cli;

#[cfg(feature = "report")]
pub mod report;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(all(test, feature = "wasm"))]
mod wasm_tests;

#[cfg(feature = "wasm")]
mod presets;

/// Re-export Burn backend types for convenience
pub use burn::tensor::Tensor;
