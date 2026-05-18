//! Pre-computed projection types for batched coupling.
//!
//! Used by [`super::engine::BatchHybridEngine`] to cache weight tensors
//! across integration steps.

use burn::prelude::Backend;
use burn::tensor::Tensor;

use crate::engine::coupling::CouplingFnConfig;

pub(crate) struct PrecomputedProjection<B: Backend> {
    pub src: usize,
    pub tgt: usize,
    pub delay: u32,
    pub cvar_map: Vec<(usize, usize)>,
    pub weight_kind: ProjectionWeightKind<B>,
    pub coupling_fn: CouplingFnConfig,
    pub rowsums_tensor: Option<Tensor<B, 2>>,
}

pub(crate) enum ProjectionWeightKind<B: Backend> {
    Scalar { weight: f32 },
    Dense { weights: Tensor<B, 2>, weights_3d: Option<Tensor<B, 3>> },
    Csr { weights: Tensor<B, 2>, weights_3d: Option<Tensor<B, 3>> },
}
