use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

/// Jansen-Rit neural mass model.
///
/// State variables: y0, y1, y2, y3, y4, y5 (indices 0..5)
///
/// Equations:
///   sigm(v) = 2*nu_max / (1 + exp(r*(v0 - v)))
///   dy0 = y3
///   dy1 = y4
///   dy2 = y5
///   dy3 = A*a*sigm(y1 - y2) - 2*a*y3 - a^2*y0
///   dy4 = A*a*(mu + a_2*J*sigm(a_1*J*y0) + c_0) - 2*a*y4 - a^2*y1
///   dy5 = B*b*(a_4*J*sigm(a_3*J*y0)) - 2*b*y5 - b^2*y2
///
/// Parameters: [A, B, a, b, v0, nu_max, r, J, a_1, a_2, a_3, a_4, mu]
/// Default:    [3.25, 22.0, 0.1, 0.05, 5.52, 0.0025, 0.56, 135.0, 1.0, 0.8, 0.25, 0.25, 0.22]
///
/// NCVAR=1.
pub struct JansenRit;

impl<B: Backend> NeuralMassModel<B> for JansenRit {
    const NVAR: usize = 6;
    const NCVAR: usize = 1;
    const PARAM_NAMES: &'static [&'static str] = &[
        "A", "B", "a", "b", "v0", "nu_max", "r", "J",
        "a_1", "a_2", "a_3", "a_4", "mu",
    ];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (0.1, 20.0),      // A
        (0.1, 50.0),      // B
        (0.01, 1.0),      // a
        (0.01, 1.0),      // b
        (0.0, 10.0),      // v0
        (0.0001, 0.01),   // nu_max
        (0.01, 2.0),      // r
        (1.0, 500.0),     // J
        (0.01, 5.0),      // a_1
        (0.01, 5.0),      // a_2
        (0.01, 5.0),      // a_3
        (0.01, 5.0),      // a_4
        (-1.0, 5.0),      // mu
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (-500.0, 500.0),  // y0
        (-500.0, 500.0),  // y1
        (-500.0, 500.0),  // y2
        (-50.0, 50.0),    // y3
        (-50.0, 50.0),    // y4
        (-50.0, 50.0),    // y5
    ];

    const STVAR: &'static [usize] = &[0, 1, 2, 3, 4, 5];

    fn dfun(
        state: Tensor<B, 2>,
        coupling: Tensor<B, 2>,
        params: &[f32],
    ) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::jansen_rit_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {
        // No explicit boundary clamping
    }
}

/// Convenience: create Jansen-Rit parameters with Python TVB defaults.
pub fn jansen_rit_default_params() -> Vec<f32> {
    vec![
        3.25, 22.0, 0.1, 0.05, 5.52, 0.0025, 0.56, 135.0,
        1.0, 0.8, 0.25, 0.25, 0.22,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_jr_dfun_at_zero() {
        let state = Tensor::<B, 2>::zeros([1, 6], &Default::default());
        let coupling = Tensor::<B, 2>::zeros([1, 1], &Default::default());
        let params = jansen_rit_default_params();
        let d = JansenRit::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();

        // With all y=0, c=0:
        // sigm(0) = 2*nu_max / (1 + exp(r*v0))
        let nu_max = 0.0025_f32;
        let r = 0.56_f32;
        let v0 = 5.52_f32;
        let sigm_0 = 2.0 * nu_max / (1.0 + (r * v0).exp());

        let a_p = 3.25_f32;
        let a = 0.1_f32;
        let b_p = 22.0_f32;
        let b = 0.05_f32;
        let j = 135.0_f32;
        let _a_1 = 1.0_f32;
        let a_2 = 0.8_f32;
        let _a_3 = 0.25_f32;
        let a_4 = 0.25_f32;
        let mu = 0.22_f32;

        // dy0 = 0
        assert!(d[0].abs() < 1e-6, "dy0 = {} (expected 0)", d[0]);
        // dy1 = 0
        assert!(d[1].abs() < 1e-6, "dy1 = {} (expected 0)", d[1]);
        // dy2 = 0
        assert!(d[2].abs() < 1e-6, "dy2 = {} (expected 0)", d[2]);
        // dy3 = A*a*sigm(0) - 0 - 0
        let expected_dy3 = a_p * a * sigm_0;
        assert!((d[3] - expected_dy3).abs() < 1e-6, "dy3 = {} (expected {})", d[3], expected_dy3);
        // dy4 = A*a*(mu + a_2*J*sigm(0) + 0) - 0 - 0
        let expected_dy4 = a_p * a * (mu + a_2 * j * sigm_0);
        assert!((d[4] - expected_dy4).abs() < 1e-5, "dy4 = {} (expected {})", d[4], expected_dy4);
        // dy5 = B*b*(a_4*J*sigm(0)) - 0 - 0
        let expected_dy5 = b_p * b * a_4 * j * sigm_0;
        assert!((d[5] - expected_dy5).abs() < 1e-5, "dy5 = {} (expected {})", d[5], expected_dy5);
    }

    #[test]
    fn test_jr_dfun_nonzero_state_and_coupling() {
        // Non-zero state and coupling to exercise the sigmoid computation.
        // state = [y0=0.1, y1=0.2, y2=0.05, y3=0, y4=0, y5=0, c_0=0.1]
        //
        // sigm(v) = 2*nu_max / (1 + exp(r*(v0 - v)))
        //
        // sigm(y1-y2) = sigm(0.15) ≈ 0.0002355
        //   arg = 0.56*(5.52-0.15) = 3.0072
        //   denom = exp(3.0072)+1 ≈ 21.23
        //   sigm(0.15) = 0.005/21.23 ≈ 0.0002355
        //
        // sigm(a1*J*y0) = sigm(13.5) ≈ 0.004943
        //   arg = 0.56*(5.52-13.5) = -4.4688
        //   denom = exp(-4.4688)+1 ≈ 1.01147
        //   sigm(13.5) ≈ 0.004943
        //
        // sigm(a3*J*y0) = sigm(3.375) ≈ 0.001156
        //   arg = 0.56*(5.52-3.375) = 1.2012
        //   denom = exp(1.2012)+1 ≈ 4.324
        //   sigm(3.375) ≈ 0.001156
        //
        // Expected derivatives (with all params from jansen_rit_default_params):
        // dy0 = y3 = 0
        // dy1 = y4 = 0
        // dy2 = y5 = 0
        // dy3 = A*a*sigm(y1-y2) - a²*y0 ≈ 0.0000765 - 0.001 ≈ -0.000924
        // dy4 = A*a*(mu + a2*J*sigm(a1*J*y0) + c_0) - a²*y1 ≈ 0.325*0.85384 - 0.002 ≈ 0.2755
        // dy5 = B*b*a4*J*sigm(a3*J*y0) - b²*y2 ≈ 1.1*0.03902 - 0.000125 ≈ 0.0428

        let state = Tensor::<B, 2>::from_floats(
            [[0.1_f32, 0.2, 0.05, 0.0, 0.0, 0.0]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::from_floats(
            [[0.1_f32]],
            &Default::default(),
        );
        let params = jansen_rit_default_params();
        let d = JansenRit::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();

        // dy0, dy1, dy2 are velocity variables (y3, y4, y5) which are 0
        assert!(d[0].abs() < 1e-6, "dy0 = {} (expected 0)", d[0]);
        assert!(d[1].abs() < 1e-6, "dy1 = {} (expected 0)", d[1]);
        assert!(d[2].abs() < 1e-6, "dy2 = {} (expected 0)", d[2]);

        // Verify derivatives numerically (wide tolerance due to f32 arithmetic)
        assert!((d[3] + 9.24e-4).abs() < 5e-4,
            "dy3 = {} (expected ≈-0.000924)", d[3]);
        assert!((d[4] - 0.2755).abs() < 0.01,
            "dy4 = {} (expected ≈0.276)", d[4]);
        assert!((d[5] - 0.0428).abs() < 0.005,
            "dy5 = {} (expected ≈0.043)", d[5]);
    }
}
