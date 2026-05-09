use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct KIonEx;

impl<B: Backend> NeuralMassModel<B> for KIonEx {
    const NVAR: usize = 5;
    const NCVAR: usize = 1;
    const PARAM_NAMES: &'static [&'static str] = &[
        "E", "K_bath", "J", "eta", "Delta",
        "c_minus", "R_minus", "c_plus", "R_plus", "Vstar",
        "Cm", "tau_n", "gamma", "epsilon",
    ];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::kionex_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(state: &mut Tensor<B, 2>) {
        let x = state.clone().narrow(1, 0, 1).clamp(0.0, f32::INFINITY);
        let rest = state.clone().narrow(1, 1, 4);
        *state = Tensor::cat(vec![x, rest], 1);
    }
}

pub fn kionex_default_params() -> Vec<f32> {
    vec![
        0.0, 5.5, 0.1, 0.0, 1.0,
        -40.0, 0.5, -20.0, -0.5, -31.0,
        1.0, 4.0, 0.04, 0.001,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_kionex_dfun_finite() {
        let state = Tensor::<B, 2>::from_floats(
            [[0.1_f32, -65.0, 0.5, -0.5, -2.0]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::zeros([1, 1], &Default::default());
        let params = kionex_default_params();
        let d = KIonEx::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        let names = ["x", "V", "n", "DKi", "Kg"];
        let expected = [-1.50015915_f32, -359.477, -0.107, -0.00509, 0.0012];
        for i in 0..5 {
            assert!(d[i].is_finite(), "d[{}] not finite: {}", names[i], d[i]);
            let rel_err = if expected[i].abs() > 0.01 { (d[i] - expected[i]).abs() / expected[i].abs() } else { (d[i] - expected[i]).abs() };
            assert!(rel_err < 0.05, "d[{}] mismatch: actual={}, expected={}", names[i], d[i], expected[i]);
        }
    }
}
