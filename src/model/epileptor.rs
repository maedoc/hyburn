use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct Epileptor;

impl<B: Backend> NeuralMassModel<B> for Epileptor {
    const NVAR: usize = 6;
    const NCVAR: usize = 2;
    const PARAM_NAMES: &'static [&'static str] = &[
        "Iext", "Iext2", "x0", "a", "b", "c", "d", "r",
        "slope", "tau", "aa", "bb", "Kvf", "Kf", "Ks", "tt", "modification",
    ];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::epileptor_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {}
}

pub fn epileptor_default_params() -> Vec<f32> {
    vec![
        3.1, 0.45, -1.6, 1.0, 3.0, 1.0, 5.0, 0.00035,
        0.0, 10.0, 6.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_epileptor_dfun_finite() {
        let state = Tensor::<B, 2>::from_floats(
            [[-1.0_f32, 1.0, 2.0, -1.0, 0.5, 0.1]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::zeros([1, 2], &Default::default());
        let params = epileptor_default_params();
        let d = Epileptor::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        for i in 0..6 {
            assert!(d[i].is_finite(), "d[{}] is not finite: {}", i, d[i]);
        }
    }
}
