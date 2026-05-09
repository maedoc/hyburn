use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct Linear;

impl<B: Backend> NeuralMassModel<B> for Linear {
    const NVAR: usize = 1;
    const NCVAR: usize = 1;
    const PARAM_NAMES: &'static [&'static str] = &["gamma"];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (-100.0, 0.0),    // gamma
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (f32::NAN, f32::NAN), // x (unbounded)
    ];

    const STVAR: &'static [usize] = &[0];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::linear_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {}
}

pub fn linear_default_params() -> Vec<f32> {
    vec![-10.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_linear_dfun() {
        let state = Tensor::<B, 2>::from_floats([[1.0_f32]], &Default::default());
        let coupling = Tensor::<B, 2>::zeros([1, 1], &Default::default());
        let params = linear_default_params();
        let d = Linear::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        assert!((d[0] + 10.0).abs() < 1e-4, "dx = {} (expected -10)", d[0]);
    }
}
