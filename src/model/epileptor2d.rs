use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct Epileptor2D;

impl<B: Backend> NeuralMassModel<B> for Epileptor2D {
    const NVAR: usize = 2;
    const NCVAR: usize = 1;
    const PARAM_NAMES: &'static [&'static str] = &[
        "Iext", "x0", "a", "b", "slope", "c", "d", "r", "Kvf", "Ks", "tt", "modification",
    ];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (-2.0, 10.0),     // Iext
        (-5.0, 0.0),      // x0
        (0.01, 10.0),     // a
        (0.01, 10.0),     // b
        (0.0, 10.0),      // slope
        (0.01, 5.0),      // c
        (0.01, 10.0),     // d
        (0.00001, 0.01),  // r
        (-5.0, 5.0),      // Kvf
        (-5.0, 5.0),      // Ks
        (0.1, 10.0),      // tt
        (0.0, 2.0),       // modification
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (-5.0, 5.0),      // x1
        (-5.0, 5.0),      // z
    ];

    const STVAR: &'static [usize] = &[0, 1];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::epileptor2d_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {}
}

pub fn epileptor2d_default_params() -> Vec<f32> {
    vec![3.1, -1.6, 1.0, 3.0, 0.0, 1.0, 5.0, 0.00035, 0.0, 0.0, 1.0, 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_epileptor2d_dfun_finite() {
        let state = Tensor::<B, 2>::from_floats([[-1.0_f32, 1.0]], &Default::default());
        let coupling = Tensor::<B, 2>::zeros([1, 1], &Default::default());
        let params = epileptor2d_default_params();
        let d = Epileptor2D::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        for i in 0..2 {
            assert!(d[i].is_finite(), "d[{}] is not finite: {}", i, d[i]);
        }
    }
}
