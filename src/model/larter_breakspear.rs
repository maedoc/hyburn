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

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (0.01, 10.0),     // gCa
        (0.01, 10.0),     // gK
        (0.01, 5.0),      // gL
        (0.01, 20.0),     // gNa
        (0.01, 10.0),     // phi
        (-5.0, 5.0),      // VCa
        (-5.0, 1.0),      // VK
        (-1.0, 1.0),      // VL
        (-1.0, 5.0),      // VNa
        (-1.0, 5.0),      // TCa
        (0.0, 5.0),       // TNa
        (-5.0, 5.0),      // TK
        (0.01, 5.0),      // d_Ca
        (0.01, 5.0),      // d_Na
        (0.01, 5.0),      // d_K
        (0.01, 5.0),      // d_V
        (0.01, 5.0),      // d_Z
        (0.0, 10.0),      // aei
        (0.0, 10.0),      // aie
        (0.0, 10.0),      // aee
        (0.0, 10.0),      // ane
        (0.0, 10.0),      // ani
        (0.0, 1.0),       // b
        (0.0, 1.0),       // C
        (-5.0, 5.0),      // Iext
        (0.0, 1.0),       // rNMDA
        (-5.0, 5.0),      // VT
        (-5.0, 5.0),      // ZT
        (0.01, 10.0),     // QV_max
        (0.01, 10.0),     // QZ_max
        (0.01, 10.0),     // t_scale
        (0.01, 10.0),     // tau_K
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (-2.0, 2.0),      // V
        (-10.0, 10.0),    // W
        (-10.0, 10.0),    // Z
    ];

    const STVAR: &'static [usize] = &[0, 1, 2];

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
