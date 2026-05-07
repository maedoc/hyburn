//! Error types for hyburn.

use thiserror::Error;

/// Top-level simulation error.
#[derive(Error, Debug)]
pub enum SimulationError {
    /// Invalid configuration value or missing field.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Invalid state tensor (NaN, Inf, or shape mismatch).
    #[error("invalid state: {0}")]
    InvalidState(String),

    /// Backend-specific tensor or device error.
    #[error("backend error: {0}")]
    BackendError(String),

    /// I/O error during file read/write.
    #[error("I/O error: {0}")]
    IOError(#[from] std::io::Error),
}

/// Convenience result alias.
pub type Result<T> = std::result::Result<T, SimulationError>;
