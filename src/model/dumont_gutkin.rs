use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct DumontGutkin;

impl<B: Backend> NeuralMassModel<B> for DumontGutkin {
    const NVAR: usize = 8;
    const NCVAR: usize = 4;
    const CVAR: &'static [usize] = &[0, 1, 4, 5];
    const PARAM_NAMES: &'static [&'static str] = &[
        "I_e", "Delta_e", "eta_e", "tau_e",
        "I_i", "Delta_i", "eta_i", "tau_i",
        "tau_s", "J_ee", "J_ei", "J_ie", "J_ii", "Gamma",
    ];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (-5.0, 5.0),      // I_e
        (0.001, 10.0),    // Delta_e
        (-10.0, 5.0),     // eta_e
        (0.1, 100.0),     // tau_e
        (-5.0, 5.0),      // I_i
        (0.001, 10.0),    // Delta_i
        (-10.0, 5.0),     // eta_i
        (0.1, 100.0),     // tau_i
        (0.1, 100.0),     // tau_s
        (0.0, 50.0),      // J_ee
        (0.0, 50.0),      // J_ei
        (0.0, 50.0),      // J_ie
        (0.0, 50.0),      // J_ii
        (0.01, 20.0),     // Gamma
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (0.0, 10.0),      // r_e (clamped >= 0)
        (-10.0, 10.0),    // V_e
        (-10.0, 10.0),    // s_ee
        (-10.0, 10.0),    // s_ei
        (0.0, 10.0),      // r_i (clamped >= 0)
        (-10.0, 10.0),    // V_i
        (-10.0, 10.0),    // s_ie
        (-10.0, 10.0),    // s_ii
    ];

    const STVAR: &'static [usize] = &[0, 1, 2, 3, 4, 5, 6, 7];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::dumont_gutkin_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(state: &mut Tensor<B, 2>) {
        let r_e = state.clone().narrow(1, 0, 1).clamp(0.0, f32::INFINITY);
        let v_e = state.clone().narrow(1, 1, 1);
        let s_ee = state.clone().narrow(1, 2, 1);
        let s_ei = state.clone().narrow(1, 3, 1);
        let r_i = state.clone().narrow(1, 4, 1).clamp(0.0, f32::INFINITY);
        let v_i = state.clone().narrow(1, 5, 1);
        let s_ie = state.clone().narrow(1, 6, 1);
        let s_ii = state.clone().narrow(1, 7, 1);
        *state = Tensor::cat(vec![r_e, v_e, s_ee, s_ei, r_i, v_i, s_ie, s_ii], 1);
    }
}

pub fn dumont_gutkin_default_params() -> Vec<f32> {
    vec![
        0.0, 1.0, -5.0, 10.0,
        0.0, 1.0, -5.0, 10.0,
        1.0, 0.0, 10.0, 0.0, 15.0, 5.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_dumont_gutkin_at_zero() {
        let state = Tensor::<B, 2>::from_floats(
            [[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::zeros([1, 4], &Default::default());
        let params = dumont_gutkin_default_params();
        let d = DumontGutkin::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        let pi = std::f32::consts::PI;
        let tau_e = params[3];
        let expected_dr_e = params[1] / (pi * tau_e * tau_e);
        assert!((d[0] - expected_dr_e).abs() < 1e-4, "dr_e = {}", d[0]);
    }
}
