use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

pub struct LarterBreakspear;

impl<B: Backend> NeuralMassModel<B> for LarterBreakspear {
    const NVAR: usize = 3;
    const NCVAR: usize = 1;
    const PARAM_NAMES: &'static [&'static str] = &[
        "gCa", "gK", "gL", "gNa", "phi",
        "VCa", "VK", "VL", "VNa",
        "TCa", "TNa", "TK",
        "d_Ca", "d_Na", "d_K", "d_V", "d_Z",
        "aei", "aie", "aee", "ane", "ani",
        "b", "C", "Iext", "rNMDA",
        "VT", "ZT", "QV_max", "QZ_max", "t_scale", "tau_K",
    ];

    fn dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>, params: &[f32]) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::larter_breakspear_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {}
}

pub fn larter_breakspear_default_params() -> Vec<f32> {
    vec![
        1.1, 2.0, 0.5, 6.7, 0.7,
        1.0, -0.7, -0.5, 0.53,
        -0.01, 0.3, 0.0,
        0.15, 0.15, 0.3, 0.65, 0.7,
        2.0, 2.0, 0.4, 1.0, 0.4,
        0.1, 0.1, 0.3, 0.25,
        0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    type B = NdArray<f32>;

    #[test]
    fn test_larter_breakspear_dfun() {
        let state = Tensor::<B, 2>::from_floats([[0.0_f32, 0.0, 0.0]], &Default::default());
        let coupling = Tensor::<B, 2>::zeros([1, 1], &Default::default());
        let params = larter_breakspear_default_params();
        let d = LarterBreakspear::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        for i in 0..3 {
            assert!(d[i].is_finite(), "d[{}] is not finite: {}", i, d[i]);
        }
    }
}
