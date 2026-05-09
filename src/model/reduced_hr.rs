use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct ReducedSetHindmarshRose;

impl<B: Backend> NeuralMassModel<B> for ReducedSetHindmarshRose {
    const NVAR: usize = 6;
    const NCVAR: usize = 2;
    const PARAM_NAMES: &'static [&'static str] = &[
        "r", "a", "b", "c", "d", "s", "xo",
        "K11", "K12", "K21", "sigma", "mu",
    ];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::reduced_hr_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {}
}

pub fn reduced_hr_default_params() -> Vec<f32> {
    vec![0.006, 1.0, 3.0, 1.0, 5.0, 4.0, -1.6, 0.5, 0.1, 0.15, 0.3, 3.3]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_reduced_hr_dfun_finite() {
        let state = Tensor::<B, 2>::from_floats(
            [[-1.0_f32, -5.0, 1.0, -0.5, -5.0, 1.0]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::zeros([1, 2], &Default::default());
        let params = reduced_hr_default_params();
        let d = ReducedSetHindmarshRose::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        for i in 0..6 {
            assert!(d[i].is_finite(), "d[{}] not finite: {}", i, d[i]);
        }
    }
}
