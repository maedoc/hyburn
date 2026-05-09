pub mod g2do;
pub mod mpr;
pub mod rww;
pub mod kuramoto_model;
pub mod jansen_rit;
pub mod wilson_cowan;
pub mod linear;
pub mod sup_hopf;
pub mod hopfield;
pub mod coombes_byrne2d;
pub mod coombes_byrne;
pub mod gast_schmidt_knosche_sd;
pub mod gast_schmidt_knosche_sf;
pub mod larter_breakspear;
pub mod epileptor2d;
pub mod epileptor;
pub mod rww_exc_inh;
pub mod deco_balanced_exc_inh;
pub mod epileptor_codim3;
pub mod epileptor_codim3_slowmod;
pub mod epileptor_rs;
pub mod zetterberg_jansen;
pub mod reduced_fhn;
pub mod reduced_hr;
pub mod dumont_gutkin;
pub mod zerlaut_first;
pub mod zerlaut_second;
pub mod kionex;

use burn::prelude::Backend;

/// Core trait for neural mass models.
///
/// Each model defines its own state variables, coupling variables,
/// derivative function (dfun), and boundary clamping rules.
pub trait NeuralMassModel<B: Backend> {
    /// Number of state variables per node (e.g., 2 for G2DO).
    const NVAR: usize;

    /// Number of coupling variables (input modes) this model receives.
    const NCVAR: usize;

    /// Human-readable parameter names (for config/help).
    const PARAM_NAMES: &'static [&'static str];

    /// Valid parameter ranges: `(lo, hi)` per parameter.
    /// Use `(f32::NAN, f32::NAN)` for params without a clear domain.
    const PARAM_RANGES: &'static [(f32, f32)];

    /// Valid state variable ranges: `(lo, hi)` per state variable.
    /// Use `(f32::NAN, f32::NAN)` for variables without a clear domain.
    const SVAR_RANGES: &'static [(f32, f32)];

    /// Indices of state variables that receive stochastic noise.
    const STVAR: &'static [usize];

    /// Compute state derivatives given current state and coupling input.
    ///
    /// - `state`: shape `[nnodes, nvar]`
    /// - `coupling`: shape `[nnodes, ncvar]`
    /// - `params`: model-specific parameter slice
    ///
    /// Returns derivatives with same shape as `state`.
    fn dfun(
        state: burn::tensor::Tensor<B, 2>,
        coupling: burn::tensor::Tensor<B, 2>,
        params: &[f32],
    ) -> burn::tensor::Tensor<B, 2>;

    /// Clamp state values to model-specific boundaries (e.g., r >= 0 for MPR).
    fn clamp(state: &mut burn::tensor::Tensor<B, 2>);
}
