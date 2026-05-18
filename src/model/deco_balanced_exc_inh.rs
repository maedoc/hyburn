use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct DecoBalancedExcInh;

impl<B: Backend> NeuralMassModel<B> for DecoBalancedExcInh {
    const NVAR: usize = 2;
    const NCVAR: usize = 1;
    const CVAR: &'static [usize] = &[0];
    const PARAM_NAMES: &'static [&'static str] = &[
        "a_e", "b_e", "d_e", "gamma_e", "tau_e",
        "w_p", "J_N", "W_e",
        "a_i", "b_i", "d_i", "gamma_i", "tau_i",
        "J_i", "W_i", "I_o", "I_ext", "G", "lamda", "M_i",
    ];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (1.0, 1000.0),    // a_e
        (1.0, 500.0),     // b_e
        (0.01, 1.0),      // d_e
        (0.0001, 0.01),   // gamma_e
        (1.0, 1000.0),    // tau_e
        (0.0, 5.0),       // w_p
        (0.0, 5.0),       // J_N
        (0.0, 5.0),       // W_e
        (1.0, 1000.0),    // a_i
        (1.0, 500.0),     // b_i
        (0.01, 1.0),      // d_i
        (0.0001, 0.01),   // gamma_i
        (0.1, 100.0),     // tau_i
        (0.0, 5.0),       // J_i
        (0.0, 5.0),       // W_i
        (-1.0, 1.0),      // I_o
        (-5.0, 5.0),      // I_ext
        (0.0, 10.0),      // G
        (-5.0, 5.0),      // lamda
        (0.0, 10.0),      // M_i
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (0.0, 1.0),       // S_e (clamped [0,1])
        (0.0, 1.0),       // S_i (clamped [0,1])
    ];

    const STVAR: &'static [usize] = &[0, 1];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::deco_balanced_exc_inh_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(state: &mut Tensor<B, 2>) {
        let se = state.clone().narrow(1, 0, 1).clamp(0.0, 1.0);
        let si = state.clone().narrow(1, 1, 1).clamp(0.0, 1.0);
        *state = Tensor::cat(vec![se, si], 1);
    }
}

pub fn deco_balanced_exc_inh_default_params() -> Vec<f32> {
    vec![
        310.0, 125.0, 0.160, 0.000641, 100.0,
        1.4, 0.15, 1.0,
        615.0, 177.0, 0.087, 0.001, 10.0,
        1.0, 0.7, 0.382, 0.0, 2.0, 0.0, 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_deco_balanced_clamp() {
        let mut state = Tensor::<B, 2>::from_floats([[-0.5_f32, 1.5]], &Default::default());
        DecoBalancedExcInh::clamp(&mut state);
        let vals = state.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-6, "S_e clamped: {}", d[0]);
        assert!((d[1] - 1.0).abs() < 1e-6, "S_i clamped: {}", d[1]);
    }
}
