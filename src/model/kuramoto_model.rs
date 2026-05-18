use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

/// Kuramoto phase oscillator.
///
/// State variable: theta (index 0)
///
/// Equation:
///   dtheta = omega + c_0
///
/// Parameters: [omega]
/// Default:    [1.0]
///
/// NCVAR=1 (cvar=[0]).
pub struct Kuramoto;

impl<B: Backend> NeuralMassModel<B> for Kuramoto {
    const NVAR: usize = 1;
    const NCVAR: usize = 1;
    const CVAR: &'static [usize] = &[0];
    const PARAM_NAMES: &'static [&'static str] = &["omega"];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (-10.0, 10.0),    // omega
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (0.0, std::f32::consts::TAU),
    ];

    const STVAR: &'static [usize] = &[0];

    fn dfun(
        state: Tensor<B, 2>,
        coupling: Tensor<B, 2>,
        params: &[f32],
    ) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::kuramoto_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(state: &mut Tensor<B, 2>) {
        // Phase normalization: wrap θ into [0, 2π) to prevent precision loss
        let two_pi = 2.0 * std::f32::consts::PI;
        *state = state.clone() - (state.clone() / two_pi).floor() * two_pi;
    }
}

/// Convenience: create Kuramoto parameters with Python TVB defaults.
pub fn kuramoto_default_params() -> Vec<f32> {
    vec![1.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_kuramoto_dfun() {
        let state = Tensor::<B, 2>::from_floats(
            [[0.5_f32]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::from_floats(
            [[0.3_f32]],
            &Default::default(),
        );
        let params = kuramoto_default_params();
        let d = Kuramoto::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();

        assert!((d[0] - 1.3).abs() < 1e-5,
            "dtheta = {} (expected 1.3)", d[0]);
    }
}
