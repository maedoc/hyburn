use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct GastSchmidtKnoscheSD;

impl<B: Backend> NeuralMassModel<B> for GastSchmidtKnoscheSD {
    const NVAR: usize = 4;
    const NCVAR: usize = 4;
    const CVAR: &'static [usize] = &[0, 1, 2, 3];
    const PARAM_NAMES: &'static [&'static str] = &[
        "tau", "tau_A", "alpha", "I", "Delta", "J", "eta", "cr", "cv"
    ];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (0.01, 100.0),    // tau
        (0.1, 100.0),     // tau_A
        (0.01, 20.0),     // alpha
        (-5.0, 5.0),      // I
        (0.001, 10.0),    // Delta
        (0.0, 50.0),      // J
        (-10.0, 5.0),     // eta
        (-5.0, 5.0),      // cr
        (-5.0, 5.0),      // cv
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (0.0, 10.0),      // r (clamped >= 0)
        (-10.0, 10.0),    // V
        (-10.0, 10.0),    // a
        (-10.0, 10.0),    // b
    ];

    const STVAR: &'static [usize] = &[0, 1, 2, 3];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::gast_sd_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(state: &mut Tensor<B, 2>) {
        let r = state.clone().narrow(1, 0, 1).clamp(0.0, f32::INFINITY);
        let v = state.clone().narrow(1, 1, 1);
        let a = state.clone().narrow(1, 2, 1);
        let b = state.clone().narrow(1, 3, 1);
        *state = Tensor::cat(vec![r, v, a, b], 1);
    }
}

pub fn gast_sd_default_params() -> Vec<f32> {
    vec![1.0, 10.0, 0.5, 0.0, 2.0, 21.2132, -6.0, 1.0, 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_gast_sd_at_zero() {
        let state = Tensor::<B, 2>::from_floats([[0.0_f32, 0.0, 0.0, 0.0]], &Default::default());
        let coupling = Tensor::<B, 2>::zeros([1, 4], &Default::default());
        let params = gast_sd_default_params();
        let d = GastSchmidtKnoscheSD::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        let pi = std::f32::consts::PI;
        let tau = params[0];
        let expected_dr = params[4] / (pi * tau * tau);
        assert!((d[0] - expected_dr).abs() < 1e-4, "dr = {}", d[0]);
    }
}
