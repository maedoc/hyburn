use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct EpileptorCodim3;

impl<B: Backend> NeuralMassModel<B> for EpileptorCodim3 {
    const NVAR: usize = 3;
    const NCVAR: usize = 1;
    const PARAM_NAMES: &'static [&'static str] = &[
        "mu1_start", "mu2_start", "nu_start",
        "mu1_stop", "mu2_stop", "nu_stop",
        "b", "R", "c", "dstar", "Ks", "N", "modification",
    ];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::epileptor_codim3_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {}
}

pub fn epileptor_codim3_default_params() -> Vec<f32> {
    vec![
        -0.02285, 0.3448, 0.2014,
        -0.07465, 0.3351, 0.2053,
        1.0, 0.4, 0.001, 0.3, 0.0, 1.0, 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_epileptor_codim3_dfun_finite() {
        let state = Tensor::<B, 2>::from_floats([[0.1_f32, 0.1, 0.0]], &Default::default());
        let coupling = Tensor::<B, 2>::zeros([1, 1], &Default::default());
        let params = epileptor_codim3_default_params();
        let d = EpileptorCodim3::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        for i in 0..3 {
            assert!(d[i].is_finite(), "d[{}] not finite: {}", i, d[i]);
        }
    }
}
