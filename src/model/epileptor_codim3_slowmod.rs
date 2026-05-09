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
        "mu1_Ain", "mu2_Ain", "nu_Ain",
        "mu1_Bin", "mu2_Bin", "nu_Bin",
        "mu1_Aend", "mu2_Aend", "nu_Aend",
        "mu1_Bend", "mu2_Bend", "nu_Bend",
        "cA", "cB",
    ];

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
        1.0, 0.4, 0.001, 0.3, 0.0, 1.0, 1.0,
        0.05494, 0.2731, 0.287,
        -0.0461, 0.243, 0.3144,
        0.06485, 0.07337, -0.3878,
        0.03676, -0.02792, -0.3973,
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
