//! Per-subnetwork metadata and helpers.

use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::EngineModel;
use crate::error::{Result, SimulationError};

/// Metadata for a single subnetwork within a hybrid simulation.
pub struct Subnetwork<B: Backend> {
    pub model_name: String,
    pub params: Vec<f32>,
    pub nnodes: usize,
    pub nmodes: usize,
    pub nvar: usize,
    pub ncvar: usize,
    /// State-variable indices for each coupling variable position.
    /// CVAR[i] is the state-variable index that coupling input i maps to.
    /// Length equals NCVAR.
    pub cvar: Vec<usize>,
    /// Offset in a flat global state array (for future flat-state designs).
    pub state_offset: usize,
    /// Length of the flat state slice (= nvar * nnodes * nmodes).
    pub state_len: usize,
    /// Phantom data for backend.
    pub _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Subnetwork<B> {
    /// Create a new subnetwork descriptor.
    pub fn new(
        model_name: String,
        params: Vec<f32>,
        nnodes: usize,
        nmodes: usize,
        state_offset: usize,
    ) -> Result<Self> {
        let model = EngineModel::<B>::from_config(&model_name, params.clone())
            .map_err(|e| SimulationError::InvalidConfig(e.to_string()))?;
        let nvar = model.nvar();
        let ncvar = model.ncvar();
        let cvar = model.cvar().to_vec();
        Ok(Self {
            model_name,
            params,
            nnodes,
            nmodes,
            nvar,
            ncvar,
            cvar,
            state_offset,
            state_len: nvar * nnodes * nmodes,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Convenience: build the engine model from cached name + params.
    pub fn make_model(&self) -> EngineModel<B> {
        EngineModel::<B>::from_config(&self.model_name, self.params.clone()).unwrap()
    }

    /// Compute `dfun` on a 2-D state slice.
    ///
    /// - `state`: shape `[nnodes * nmodes, nvar]`
    /// - `coupling`: shape `[nnodes * nmodes, ncvar]`
    pub fn dfun(&self, state: Tensor<B, 2>, coupling: Tensor<B, 2>) -> Tensor<B, 2> {
        self.make_model().dfun(state, coupling)
    }

    /// Clamp a 2-D state slice.
    pub fn clamp(&self, state: &mut Tensor<B, 2>) {
        self.make_model().clamp(state);
    }
}
