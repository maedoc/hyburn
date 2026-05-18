use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct ZerlautAdaptationSecondOrder;

impl<B: Backend> NeuralMassModel<B> for ZerlautAdaptationSecondOrder {
    const NVAR: usize = 8;
    const NCVAR: usize = 1;
    const CVAR: &'static [usize] = &[0];
    const PARAM_NAMES: &'static [&'static str] = &[
        "g_L", "E_L_e", "E_L_i", "C_m",
        "b_e", "a_e", "b_i", "a_i",
        "tau_w_e", "tau_w_i",
        "E_e", "E_i", "Q_e", "Q_i",
        "tau_e", "tau_i", "N_tot",
        "p_connect_e", "p_connect_i", "g",
        "K_ext_e", "K_ext_i", "T",
        "external_input_ex_ex", "external_input_ex_in",
        "external_input_in_ex", "external_input_in_in",
        "tau_OU", "weight_noise", "S_i",
        "P_e_0", "P_e_1", "P_e_2", "P_e_3", "P_e_4",
        "P_e_5", "P_e_6", "P_e_7", "P_e_8", "P_e_9",
        "P_i_0", "P_i_1", "P_i_2", "P_i_3", "P_i_4",
        "P_i_5", "P_i_6", "P_i_7", "P_i_8", "P_i_9",
    ];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (0.1, 100.0),     // g_L
        (-100.0, -30.0),  // E_L_e
        (-100.0, -30.0),  // E_L_i
        (10.0, 1000.0),   // C_m
        (0.0, 200.0),     // b_e
        (0.0, 20.0),      // a_e
        (0.0, 200.0),     // b_i
        (0.0, 20.0),      // a_i
        (1.0, 5000.0),    // tau_w_e
        (0.1, 100.0),     // tau_w_i
        (-10.0, 10.0),    // E_e
        (-100.0, -50.0),  // E_i
        (0.1, 20.0),      // Q_e
        (0.1, 20.0),      // Q_i
        (0.1, 100.0),     // tau_e
        (0.1, 100.0),     // tau_i
        (100.0, 100000.0), // N_tot
        (0.0, 1.0),       // p_connect_e
        (0.0, 1.0),       // p_connect_i
        (0.0, 5.0),       // g
        (0.0, 1000.0),    // K_ext_e
        (0.0, 1000.0),    // K_ext_i
        (0.1, 100.0),     // T
        (-10.0, 10.0),    // external_input_ex_ex
        (-10.0, 10.0),    // external_input_ex_in
        (-10.0, 10.0),    // external_input_in_ex
        (-10.0, 10.0),    // external_input_in_in
        (0.1, 100.0),     // tau_OU
        (0.0, 100.0),     // weight_noise
        (0.0, 10.0),      // S_i
        (-1.0, 1.0),      // P_e_0
        (-1.0, 1.0),      // P_e_1
        (-1.0, 1.0),      // P_e_2
        (-1.0, 1.0),      // P_e_3
        (-1.0, 1.0),      // P_e_4
        (-1.0, 1.0),      // P_e_5
        (-1.0, 1.0),      // P_e_6
        (-1.0, 1.0),      // P_e_7
        (-1.0, 1.0),      // P_e_8
        (-1.0, 1.0),      // P_e_9
        (-1.0, 1.0),      // P_i_0
        (-1.0, 1.0),      // P_i_1
        (-1.0, 1.0),      // P_i_2
        (-1.0, 1.0),      // P_i_3
        (-1.0, 1.0),      // P_i_4
        (-1.0, 1.0),      // P_i_5
        (-1.0, 1.0),      // P_i_6
        (-1.0, 1.0),      // P_i_7
        (-1.0, 1.0),      // P_i_8
        (-1.0, 1.0),      // P_i_9
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (0.0, 1.0),       // E (clamped [0,1])
        (0.0, 1.0),       // I (clamped [0,1])
        (-10.0, 10.0),    // dE
        (-10.0, 10.0),    // dI
        (-10.0, 10.0),    // w_e
        (-10.0, 10.0),    // w_i
        (-10.0, 10.0),    // ou_e
        (-10.0, 10.0),    // ou_i
    ];

    const STVAR: &'static [usize] = &[0, 1, 2, 3, 4, 5, 6, 7];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::zerlaut_second_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(state: &mut Tensor<B, 2>) {
        let e = state.clone().narrow(1, 0, 1).clamp(0.0, 1.0);
        let i_val = state.clone().narrow(1, 1, 1).clamp(0.0, 1.0);
        let rest = state.clone().narrow(1, 2, 6);
        *state = Tensor::cat(vec![e, i_val, rest], 1);
    }
}

pub fn zerlaut_second_default_params() -> Vec<f32> {
    vec![
        10.0, -65.0, -65.0, 200.0,
        60.0, 4.0, 0.0, 0.0,
        500.0, 1.0,
        0.0, -80.0, 1.5, 5.0,
        5.0, 5.0, 10000.0,
        0.05, 0.05, 0.2,
        400.0, 0.0, 20.0,
        0.0, 0.0, 0.0, 0.0,
        5.0, 10.5, 1.0,
        -0.04983106, 0.005063551, -0.023470122,
        0.0022951514, -0.00041053028,
        0.010547051, -0.03659253,
        0.0074374876, 0.0012650647, -0.040721614,
        -0.05149122, 0.004003689, -0.008352013,
        0.0002414238, -0.0005070645,
        0.0014345394, -0.01468669,
        0.004502706, 0.0028472191, -0.0153578045,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_zerlaut_second_dfun_finite() {
        let state = Tensor::<B, 2>::from_floats(
            [[0.01_f32, 0.01, 0.0001, 0.0, 0.0001, 0.0, 0.0, 0.0]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::zeros([1, 1], &Default::default());
        let params = zerlaut_second_default_params();
        let d = ZerlautAdaptationSecondOrder::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        for i in 0..8 {
            assert!(d[i].is_finite(), "d[{}] not finite: {}", i, d[i]);
        }
        let rel_err_de = (d[0] - 0.00365182).abs() / 0.00365182;
        assert!(rel_err_de < 0.01, "dE rel_err={rel_err_de:.6}");
    }
}
