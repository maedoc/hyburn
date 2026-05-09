use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct EpileptorCodim3SlowMod;

impl<B: Backend> NeuralMassModel<B> for EpileptorCodim3SlowMod {
    const NVAR: usize = 5;
    const NCVAR: usize = 1;
    const PARAM_NAMES: &'static [&'static str] = &[
        "mu1_start", "mu2_start", "nu_start",
        "mu1_stop", "mu2_stop", "nu_stop",
        "b", "R", "c", "dstar", "Ks", "N", "modification",
        "G0", "G1", "G2",
        "L0", "L1", "L2",
        "H0", "H1", "H2",
        "M0", "M1", "M2",
        "cA", "cB",
    ];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (-1.0, 1.0),      // mu1_start
        (-1.0, 1.0),      // mu2_start
        (-1.0, 1.0),      // nu_start
        (-1.0, 1.0),      // mu1_stop
        (-1.0, 1.0),      // mu2_stop
        (-1.0, 1.0),      // nu_stop
        (0.01, 10.0),     // b
        (0.01, 10.0),     // R
        (0.0, 0.1),       // c
        (0.01, 5.0),      // dstar
        (-5.0, 5.0),      // Ks
        (0.1, 10.0),      // N
        (0.0, 2.0),       // modification
        (-2.0, 2.0),      // G0
        (-2.0, 2.0),      // G1
        (-2.0, 2.0),      // G2
        (-2.0, 2.0),      // L0
        (-2.0, 2.0),      // L1
        (-2.0, 2.0),      // L2
        (-2.0, 2.0),      // H0
        (-2.0, 2.0),      // H1
        (-2.0, 2.0),      // H2
        (-2.0, 2.0),      // M0
        (-2.0, 2.0),      // M1
        (-2.0, 2.0),      // M2
        (0.0, 0.1),       // cA
        (0.0, 0.1),       // cB
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (-5.0, 5.0),      // x
        (-5.0, 5.0),      // y
        (-5.0, 5.0),      // z
        (-5.0, 5.0),      // slow1
        (-5.0, 5.0),      // slow2
    ];

    const STVAR: &'static [usize] = &[0, 1, 2, 3, 4];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::epileptor_codim3_slowmod_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {}
}

pub fn epileptor_codim3_slowmod_default_params() -> Vec<f32> {
    vec![
        -0.02285, 0.3448, 0.2014,
        -0.07465, 0.3351, 0.2053,
        1.0, 0.4, 0.002, 0.3, 0.0, 1.0, 1.0,
        0.68281186, -0.13736244, 0.717565,
        0.60745907, 0.11524223, 0.785947,
        0.6668986, -0.2839006, -0.6889461,
        0.79066, 0.00755125, -0.6122089,
        0.0001, 0.00012,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_epileptor_codim3_slowmod_finite() {
        let state = Tensor::<B, 2>::from_floats([[0.1_f32, 0.1, 0.0, 0.0, 0.0]], &Default::default());
        let coupling = Tensor::<B, 2>::zeros([1, 1], &Default::default());
        let params = epileptor_codim3_slowmod_default_params();
        let d = EpileptorCodim3SlowMod::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        for i in 0..5 {
            assert!(d[i].is_finite(), "d[{}] not finite: {}", i, d[i]);
        }
    }
}
