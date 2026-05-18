use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct ReducedSetFitzHughNagumo;

impl<B: Backend> NeuralMassModel<B> for ReducedSetFitzHughNagumo {
    const NVAR: usize = 4;
    const NCVAR: usize = 2;
    const CVAR: &'static [usize] = &[0, 2];
    const PARAM_NAMES: &'static [&'static str] = &[
        "tau", "a", "b", "K11", "K12", "K21", "sigma", "mu",
    ];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (0.01, 100.0),    // tau
        (-5.0, 5.0),      // a
        (-5.0, 5.0),      // b
        (0.0, 10.0),      // K11
        (0.0, 10.0),      // K12
        (0.0, 10.0),      // K21
        (-5.0, 5.0),      // sigma
        (-5.0, 5.0),      // mu
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (-5.0, 5.0),      // V
        (-5.0, 5.0),      // W
        (-5.0, 5.0),      // V2
        (-5.0, 5.0),      // W2
    ];

    const STVAR: &'static [usize] = &[0, 1, 2, 3];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::reduced_fhn_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {}
}

pub fn reduced_fhn_default_params() -> Vec<f32> {
    vec![3.0, 0.45, 0.9, 0.5, 0.15, 0.15, 0.35, 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_reduced_fhn_dfun_finite() {
        let state = Tensor::<B, 2>::from_floats([[0.0_f32, 0.0, 0.0, 0.0]], &Default::default());
        let coupling = Tensor::<B, 2>::zeros([1, 2], &Default::default());
        let params = reduced_fhn_default_params();
        let d = ReducedSetFitzHughNagumo::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        for i in 0..4 {
            assert!(d[i].is_finite(), "d[{}] not finite: {}", i, d[i]);
        }
    }
}
