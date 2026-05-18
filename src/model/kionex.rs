use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct KIonEx;

impl<B: Backend> NeuralMassModel<B> for KIonEx {
    const NVAR: usize = 5;
    const NCVAR: usize = 1;
    const CVAR: &'static [usize] = &[0];
    const PARAM_NAMES: &'static [&'static str] = &[
        "E", "K_bath", "J", "eta", "Delta",
        "c_minus", "R_minus", "c_plus", "R_plus", "Vstar",
        "Cm", "tau_n", "gamma", "epsilon",
    ];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (-5.0, 5.0),      // E
        (0.0, 20.0),      // K_bath
        (0.0, 10.0),      // J
        (-10.0, 5.0),     // eta
        (0.001, 10.0),    // Delta
        (-100.0, 0.0),    // c_minus
        (0.0, 10.0),      // R_minus
        (-100.0, 0.0),    // c_plus
        (-10.0, 0.0),     // R_plus
        (-100.0, 0.0),    // Vstar
        (0.01, 10.0),     // Cm
        (0.01, 100.0),    // tau_n
        (0.001, 10.0),    // gamma
        (0.0, 1.0),       // epsilon
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (0.0, 10.0),      // x (clamped >= 0)
        (-100.0, 50.0),   // V
        (-1.0, 1.0),      // n
        (-10.0, 10.0),    // DKi
        (-10.0, 10.0),    // Kg
    ];

    const STVAR: &'static [usize] = &[0, 1, 2, 3, 4];

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
