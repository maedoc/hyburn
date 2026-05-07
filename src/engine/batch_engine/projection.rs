//! Pre-computed projection types for batched coupling.
//!
//! Used by [`super::engine::BatchHybridEngine`] to cache weight tensors
//! across integration steps.

use burn::prelude::Backend;
use burn::tensor::Tensor;

/// Pre-computed projection with weight tensor materialized once.
pub(crate) struct PrecomputedProjection<B: Backend> {
    pub src: usize,
    pub tgt: usize,
    pub delay: u32,
    pub cvar_map: Vec<(usize, usize)>,
    pub weight_kind: ProjectionWeightKind<B>,
}

pub(crate) enum ProjectionWeightKind<B: Backend> {
    Scalar { weight: f32 },
    Dense { weights: Tensor<B, 2> },
    Csr { weights: Tensor<B, 2> },
}
