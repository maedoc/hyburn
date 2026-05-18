use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct SupHopf;

impl<B: Backend> NeuralMassModel<B> for SupHopf {
    const NVAR: usize = 2;
    const NCVAR: usize = 2;
    const CVAR: &'static [usize] = &[0, 1];
    const PARAM_NAMES: &'static [&'static str] = &["a", "omega"];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (-2.0, 2.0),      // a
        (0.01, 10.0),     // omega
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (-10.0, 10.0),    // x
        (-10.0, 10.0),    // y
    ];

    const STVAR: &'static [usize] = &[0, 1];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::sup_hopf_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {}
}

pub fn sup_hopf_default_params() -> Vec<f32> {
    vec![-0.5, 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_sup_hopf_at_zero() {
        let state = Tensor::<B, 2>::from_floats([[0.0_f32, 0.0]], &Default::default());
        let coupling = Tensor::<B, 2>::zeros([1, 2], &Default::default());
        let params = sup_hopf_default_params();
        let d = SupHopf::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        assert!(d[0].abs() < 1e-5, "dx = {} (expected 0)", d[0]);
        assert!(d[1].abs() < 1e-5, "dy = {} (expected 0)", d[1]);
    }
}
