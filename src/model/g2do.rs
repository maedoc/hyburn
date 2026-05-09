use burn::prelude::Backend;
use burn::tensor::Tensor;

use super::NeuralMassModel;

/// Generic 2D Oscillator model — full parameterisation matching Python TVB.
///
/// State variables: V (index 0), W (index 1)
///
/// Full equations:
///   dV/dt = d * tau * (alpha * W - f * V^3 + e * V^2 + g * V + gamma * (I + c_0) + lc_0)
///   dW/dt = d * (a + b * V + c * V^2 - beta * W) / tau
///
/// Where `c_0` is the coupling variable (from coupling[:,0]) and `lc_0` is local coupling
/// (not yet implemented — set to 0).
///
/// Parameters (12 total, matching Python TVB defaults):
///   [tau, I, a, b, c, d, e, f, g, alpha, beta, gamma]
///
/// Default values: [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0]
///
/// Simplified form (d=1, e=0, f=1/3, g=1, alpha=1, beta=1, gamma=1):
///   dV/dt = tau * (W - V^3/3 + V + I + c_0)
///   dW/dt = (a + b*V + c*V^2 - W) / tau
/// With further simplification (c=0, g=1): dV/dt = tau*(V - V^3/3 - W + I + c_0)
///                                           dW/dt = (V - a*W + b) / tau  [if b maps differently]
pub struct Generic2dOscillator;

impl<B: Backend> NeuralMassModel<B> for Generic2dOscillator {
    const NVAR: usize = 2;
    const NCVAR: usize = 1;
    const PARAM_NAMES: &'static [&'static str] = &[
        "tau", "I", "a", "b", "c", "d", "e", "f", "g", "alpha", "beta", "gamma"
    ];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (0.01, 100.0),    // tau
        (-5.0, 5.0),      // I
        (-5.0, 5.0),      // a
        (-20.0, 15.0),    // b
        (-10.0, 10.0),    // c
        (0.0001, 1.0),    // d
        (-5.0, 5.0),      // e
        (0.01, 10.0),     // f
        (-5.0, 5.0),      // g
        (-5.0, 5.0),      // alpha
        (0.01, 10.0),     // beta
        (-1.0, 1.0),      // gamma
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (-2.0, 4.0),      // V
        (-6.0, 6.0),      // W
    ];

    const STVAR: &'static [usize] = &[0, 1];

    fn dfun(
        state: Tensor<B, 2>,
        coupling: Tensor<B, 2>,
        params: &[f32],
    ) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::g2do_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(_state: &mut Tensor<B, 2>) {
        // G2DO has no explicit boundary clamping
    }
}

/// Convenience: create G2DO parameters with Python TVB defaults.
pub fn g2do_default_params() -> Vec<f32> {
    vec![1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_g2do_equilibrium_with_defaults() {
        // With full default params: d=0.02, tau=1, a=-2, b=-10, c=0, etc.
        // At V=0, W=0, I=0, c_0=0:
        // dV = d*tau*(alpha*W + gamma*(I+c_0) - f*V^3 + e*V^2 + g*V)
        //    = 0.02*1*(1*0 + 1*(0+0) - 1*0 + 3*0 + 0*0)
        //    = 0
        // dW = d*(a + b*V + c*V^2 - beta*W) / tau
        //    = 0.02*(-2 + -10*0 + 0*0 - 1*0) / 1
        //    = 0.02 * (-2) = -0.04
        let state = Tensor::<B, 2>::from_floats(
            [[0.0_f32, 0.0]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::zeros(
            [1, 1],
            &Default::default(),
        );
        let params = g2do_default_params();
        let d = Generic2dOscillator::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();

        assert!(d[0].abs() < 1e-5, "dV = {} (expected 0)", d[0]);
        assert!((d[1] + 0.04).abs() < 1e-4, "dW = {} (expected -0.04)", d[1]);
    }

    #[test]
    fn test_g2do_simplified_form() {
        // Simplified form: d=1, e=0, f=1/3, g=1, alpha=1, beta=1, gamma=1
        // This reduces to: dV/dt = tau*(W - V^3/3 + V + I + c_0)
        //                     dW/dt = (a + b*V + c*V^2 - W) / tau
        // With a=1, b=1, c=0, tau=1, I=0, c_0=0:
        //   V=0.5, W=0.3:
        //   dV = 1*(0.3 - 0.125/3 + 0.5 + 0) = 0.3 - 0.04167 + 0.5 = 0.7583
        //   dW = (1 + 1*0.5 + 0*0.25 - 0.3) / 1 = 1.2
        let state = Tensor::<B, 2>::from_floats(
            [[0.5_f32, 0.3]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::zeros(
            [1, 1],
            &Default::default(),
        );
        // Simplified params: [tau=1, I=0, a=1, b=1, c=0, d=1, e=0, f=1/3, g=1, alpha=1, beta=1, gamma=1]
        let params = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0/3.0, 1.0, 1.0, 1.0, 1.0];
        let d = Generic2dOscillator::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();

        // dV = 1 * 1 * (1*0.3 + 1*(0+0) - 1/3*0.125 + 0*0.25 + 1*0.5)
        //    = 0.3 - 0.04167 + 0.5 = 0.7583
        // dW = 1 * (1 + 1*0.5 + 0*0.25 - 1*0.3) / 1
        //    = 1 + 0.5 - 0.3 = 1.2
        assert!((d[0] - 0.7583).abs() < 1e-3, "dV = {} (expected ~0.758)", d[0]);
        assert!((d[1] - 1.2).abs() < 1e-3, "dW = {} (expected 1.2)", d[1]);
    }

    #[test]
    fn test_narrow_extract() {
        let state = Tensor::<B, 2>::from_floats(
            [[0.0_f32, 0.5]],
            &Default::default(),
        );
        let data = state.clone().into_data();
        let d = data.as_slice::<f32>().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-6, "d[0] = {}", d[0]);
        assert!((d[1] - 0.5).abs() < 1e-6, "d[1] = {}", d[1]);

        let v = state.clone().narrow(1, 0, 1);
        let w = state.narrow(1, 1, 1);
        let v_data = v.into_data();
        let w_data = w.into_data();
        let v = v_data.as_slice::<f32>().unwrap();
        let w = w_data.as_slice::<f32>().unwrap();
        assert!((v[0] - 0.0).abs() < 1e-6, "V = {}", v[0]);
        assert!((w[0] - 0.5).abs() < 1e-6, "W = {}", w[0]);
    }
}