use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct ZetterbergJansen;

impl<B: Backend> NeuralMassModel<B> for ZetterbergJansen {
    const NVAR: usize = 12;
    const NCVAR: usize = 1;
    const PARAM_NAMES: &'static [&'static str] = &[
        "He", "Hi", "ke", "ki",
        "e0", "rho_2", "rho_1",
        "gamma_1", "gamma_2", "gamma_3", "gamma_4", "gamma_5",
        "gamma_1T", "gamma_2T", "gamma_3T",
        "P", "U", "Q",
    ];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::zetterberg_jansen_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {}
}

pub fn zetterberg_jansen_default_params() -> Vec<f32> {
    vec![
        3.25, 22.0, 0.1, 0.05,
        0.0025, 6.0, 0.56,
        135.0, 108.0, 33.75, 33.75, 15.0,
        1.0, 1.0, 1.0,
        0.12, 0.12, 0.12,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_zetterberg_jansen_dfun_finite() {
        let state = Tensor::<B, 2>::from_floats(
            [[0.0_f32; 12]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::zeros([1, 1], &Default::default());
        let params = zetterberg_jansen_default_params();
        let d = ZetterbergJansen::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        for i in 0..12 {
            assert!(d[i].is_finite(), "d[{}] not finite: {}", i, d[i]);
        }
    }
}
