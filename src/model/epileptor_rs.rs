use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct EpileptorRestingState;

impl<B: Backend> NeuralMassModel<B> for EpileptorRestingState {
    const NVAR: usize = 8;
    const NCVAR: usize = 3;
    const PARAM_NAMES: &'static [&'static str] = &[
        "Iext", "Iext2", "x0", "a", "b", "c", "d", "r",
        "slope", "tau", "aa", "bb", "Kvf", "Kf", "Ks", "tt", "modification",
        "tau_rs", "I_rs", "a_rs", "b_rs", "d_rs", "e_rs", "f_rs",
        "alpha_rs", "beta_rs", "gamma_rs", "K_rs",
    ];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::epileptor_rs_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {}
}

pub fn epileptor_rs_default_params() -> Vec<f32> {
    vec![
        3.1, 0.45, -1.6, 1.0, 3.0, 1.0, 5.0, 0.00035,
        0.0, 10.0, 6.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        1.0, 0.0, -2.0, -10.0, 0.02, 3.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_epileptor_rs_dfun_finite() {
        let state = Tensor::<B, 2>::from_floats(
            [[-1.6_f32, -12.5, 3.8, -1.0, 0.005, 0.0, 1.0, 0.0]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::zeros([1, 3], &Default::default());
        let params = epileptor_rs_default_params();
        let d = EpileptorRestingState::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        let names = ["x1", "y1", "z", "x2", "y2", "g", "x_rs", "y_rs"];
        let expected = [-1.424_f32, 0.7, -0.00133, 0.355, -0.0005, -0.0016, 0.04, -0.24];
        for i in 0..8 {
            assert!(d[i].is_finite(), "d[{}] not finite: {}", names[i], d[i]);
            let rel_err = if expected[i].abs() > 1e-4 { (d[i] - expected[i]).abs() / expected[i].abs() } else { (d[i] - expected[i]).abs() };
            assert!(rel_err < 0.05, "d[{}] mismatch: actual={}, expected={}", names[i], d[i], expected[i]);
        }
    }
}
