use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct CoombesByrne;

impl<B: Backend> NeuralMassModel<B> for CoombesByrne {
    const NVAR: usize = 4;
    const NCVAR: usize = 4;
    const PARAM_NAMES: &'static [&'static str] = &["Delta", "alpha", "v_syn", "k", "eta"];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (0.001, 10.0),    // Delta
        (0.01, 2.0),      // alpha
        (-20.0, 20.0),    // v_syn
        (0.0, 10.0),      // k
        (-10.0, 30.0),    // eta
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (0.0, 10.0),      // r (clamped >= 0)
        (-20.0, 20.0),    // V
        (-10.0, 10.0),    // g
        (-10.0, 10.0),    // q
    ];

    const STVAR: &'static [usize] = &[0, 1, 2, 3];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::coombes_byrne_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(state: &mut Tensor<B, 2>) {
        let r = state.clone().narrow(1, 0, 1).clamp(0.0, f32::INFINITY);
        let v = state.clone().narrow(1, 1, 1);
        let g = state.clone().narrow(1, 2, 1);
        let q = state.clone().narrow(1, 3, 1);
        *state = Tensor::cat(vec![r, v, g, q], 1);
    }
}

pub fn coombes_byrne_default_params() -> Vec<f32> {
    vec![0.5, 0.95, -10.0, 1.0, 20.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_coombes_byrne_at_zero() {
        let state = Tensor::<B, 2>::from_floats([[0.0_f32, 0.0, 0.0, 0.0]], &Default::default());
        let coupling = Tensor::<B, 2>::zeros([1, 4], &Default::default());
        let params = coombes_byrne_default_params();
        let d = CoombesByrne::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        let pi = std::f32::consts::PI;
        let expected_dr = params[0] / pi;
        assert!((d[0] - expected_dr).abs() < 1e-4, "dr = {}", d[0]);
        let expected_dv = params[4];
        assert!((d[1] - expected_dv).abs() < 1e-4, "dV = {}", d[1]);
    }
}
