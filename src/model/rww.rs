use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

/// Reduced Wong-Wang (RWW) model.
///
/// State variable: S (index 0)
///
/// Equations:
///   x = w*J_N*S + I_o + J_N*c_0
///   H = (a*x - b) / (1 - exp(-d*(a*x - b)))
///   dS = -S/tau_s + (1 - S)*H*gamma
///
/// Parameters: [a, b, d, gamma, tau_s, w, J_N, I_o]
/// Default:    [0.270, 0.108, 154.0, 0.641, 100.0, 0.6, 0.2609, 0.33]
///
/// NCVAR=1 (cvar=[0]).
pub struct ReducedWongWang;

impl<B: Backend> NeuralMassModel<B> for ReducedWongWang {
    const NVAR: usize = 1;
    const NCVAR: usize = 1;
    const CVAR: &'static [usize] = &[0];
    const PARAM_NAMES: &'static [&'static str] = &["a", "b", "d", "gamma", "tau_s", "w", "J_N", "I_o"];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (0.01, 1.0),      // a
        (0.0, 1.0),       // b
        (1.0, 500.0),     // d
        (0.01, 10.0),     // gamma
        (1.0, 1000.0),    // tau_s
        (0.0, 2.0),       // w
        (0.0, 5.0),       // J_N
        (-1.0, 1.0),      // I_o
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (0.0, 1.0),       // S (clamped [0,1])
    ];

    const STVAR: &'static [usize] = &[0];

    fn dfun(
        state: Tensor<B, 2>,
        coupling: Tensor<B, 2>,
        params: &[f32],
    ) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::rww_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(state: &mut Tensor<B, 2>) {
        let s = state.clone().narrow(1, 0, 1).clamp(0.0, 1.0);
        *state = s;
    }
}

/// Convenience: create RWW parameters with Python TVB defaults.
pub fn rww_default_params() -> Vec<f32> {
    vec![0.270, 0.108, 154.0, 0.641, 100.0, 0.6, 0.2609, 0.33]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_rww_dfun_at_zero() {
        // S=0, c=0 with defaults:
        // x = 0 + 0.33 + 0 = 0.33
        // a*x - b = 0.270*0.33 - 0.108 = 0.0891 - 0.108 = -0.0189
        // H = -0.0189 / (1 - exp(-154*(-0.0189)))
        //   = -0.0189 / (1 - exp(2.9106))
        //   = -0.0189 / (1 - 18.37)
        //   = -0.0189 / (-17.37)
        //   ≈ 0.001088
        // dS = -0/100 + (1-0)*0.001088*0.641 ≈ 0.000697
        let state = Tensor::<B, 2>::from_floats(
            [[0.0_f32]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::zeros(
            [1, 1],
            &Default::default(),
        );
        let params = rww_default_params();
        let d = ReducedWongWang::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();

        let x = 0.33_f32;
        let ax_b = 0.270 * x - 0.108;
        let h = ax_b / (1.0 - (-154.0 * ax_b).exp());
        let expected = (1.0 - 0.0) * h * 0.641;
        assert!((d[0] - expected).abs() < 1e-5,
            "dS = {} (expected {})", d[0], expected);
    }

    #[test]
    fn test_rww_clamp() {
        let mut state = Tensor::<B, 2>::from_floats(
            [[-0.2_f32]],
            &Default::default(),
        );
        ReducedWongWang::clamp(&mut state);
        let vals = state.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-6, "S was not clamped to 0: {}", d[0]);

        let mut state = Tensor::<B, 2>::from_floats(
            [[1.5_f32]],
            &Default::default(),
        );
        ReducedWongWang::clamp(&mut state);
        let vals = state.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6, "S was not clamped to 1: {}", d[0]);
    }
}
