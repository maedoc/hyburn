use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct Hopfield;

impl<B: Backend> NeuralMassModel<B> for Hopfield {
    const NVAR: usize = 2;
    const NCVAR: usize = 2;
    const PARAM_NAMES: &'static [&'static str] = &["taux", "tauT", "dynamic"];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::hopfield_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {}
}

pub fn hopfield_default_params() -> Vec<f32> {
    vec![1.0, 5.0, 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_hopfield_static() {
        let state = Tensor::<B, 2>::from_floats([[2.0_f32, 0.0]], &Default::default());
        let coupling = Tensor::<B, 2>::from_floats([[1.0_f32, 0.0]], &Default::default());
        let params = hopfield_default_params();
        let d = Hopfield::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        assert!((d[0] + 1.0).abs() < 1e-4, "dx = {} (expected -1)", d[0]);
    }
}
