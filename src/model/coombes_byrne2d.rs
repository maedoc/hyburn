use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct CoombesByrne2D;

impl<B: Backend> NeuralMassModel<B> for CoombesByrne2D {
    const NVAR: usize = 2;
    const NCVAR: usize = 2;
    const CVAR: &'static [usize] = &[0, 1];
    const PARAM_NAMES: &'static [&'static str] = &["Delta", "v_syn", "k", "eta"];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (0.001, 10.0),    // Delta
        (-20.0, 20.0),    // v_syn
        (0.0, 10.0),      // k
        (-10.0, 30.0),    // eta
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (0.0, 10.0),      // r (clamped >= 0)
        (-20.0, 20.0),    // V
    ];

    const STVAR: &'static [usize] = &[0, 1];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::coombes_byrne2d_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(state: &mut Tensor<B, 2>) {
        let r = state.clone().narrow(1, 0, 1).clamp(0.0, f32::INFINITY);
        let v = state.clone().narrow(1, 1, 1);
        *state = Tensor::cat(vec![r, v], 1);
    }
}

pub fn coombes_byrne2d_default_params() -> Vec<f32> {
    vec![1.0, -4.0, 1.0, 2.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_coombes_byrne2d_at_zero() {
        let state = Tensor::<B, 2>::from_floats([[0.0_f32, 0.0]], &Default::default());
        let coupling = Tensor::<B, 2>::zeros([1, 2], &Default::default());
        let params = coombes_byrne2d_default_params();
        let d = CoombesByrne2D::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        let pi = std::f32::consts::PI;
        let expected_dr = params[0] / pi;
        assert!((d[0] - expected_dr).abs() < 1e-4, "dr = {} (expected {})", d[0], expected_dr);
        let expected_dv = params[3];
        assert!((d[1] - expected_dv).abs() < 1e-4, "dV = {} (expected {})", d[1], expected_dv);
    }
}
