use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

/// Montbrió-Pazo-Roxin (MPR) infinite QIF 2D model.
///
/// State variables: r (index 0), V (index 1)
///
/// Equations:
///   dr = (1/tau) * (Delta/(pi*tau) + 2*V*r)
///   dV = (1/tau) * (V^2 - pi^2*tau^2*r^2 + eta + J*tau*r + I + cr*c_r + cv*c_v)
///
/// Parameters: [tau, Delta, eta, J, I, cr, cv]
/// Default:    [1.0,  1.0, -5.0, 15.0, 0.0, 1.0, 0.0]
///
/// NCVAR=2 because cvar=[0,1] (both r and V are coupling variables).
pub struct MontbrioPazoRoxin;

impl<B: Backend> NeuralMassModel<B> for MontbrioPazoRoxin {
    const NVAR: usize = 2;
    const NCVAR: usize = 2;
    const CVAR: &'static [usize] = &[0, 1];
    const PARAM_NAMES: &'static [&'static str] = &["tau", "Delta", "eta", "J", "I", "cr", "cv"];

    const PARAM_RANGES: &'static [(f32, f32)] = &[
        (0.01, 100.0),    // tau
        (0.001, 10.0),    // Delta
        (-10.0, 5.0),     // eta
        (0.0, 50.0),      // J
        (-5.0, 5.0),      // I
        (-5.0, 5.0),      // cr
        (-5.0, 5.0),      // cv
    ];

    const SVAR_RANGES: &'static [(f32, f32)] = &[
        (0.0, 5.0),       // r (clamped >= 0)
        (-10.0, 10.0),    // V
    ];

    const STVAR: &'static [usize] = &[0, 1];

    fn dfun(
        state: Tensor<B, 2>,
        coupling: Tensor<B, 2>,
        params: &[f32],
    ) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::mpr_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(state: &mut Tensor<B, 2>) {
        let r = state.clone().narrow(1, 0, 1).clamp(0.0, f32::INFINITY);
        let v = state.clone().narrow(1, 1, 1);
        *state = Tensor::cat(vec![r, v], 1);
    }
}

/// Convenience: create MPR parameters with Python TVB defaults.
pub fn mpr_default_params() -> Vec<f32> {
    vec![1.0, 1.0, -5.0, 15.0, 0.0, 1.0, 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_mpr_dfun_at_zero() {
        // At r=0, V=0, c_r=0, c_v=0, with defaults:
        // dr = (1/1)*(1/(pi*1) + 0) = 1/pi ≈ 0.3183
        // dV = (1/1)*(0 - 0 + (-5) + 0 + 0 + 0 + 0) = -5
        let state = Tensor::<B, 2>::from_floats(
            [[0.0_f32, 0.0]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::zeros(
            [1, 2],
            &Default::default(),
        );
        let params = mpr_default_params();
        let d = MontbrioPazoRoxin::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();

        let expected_dr = 1.0 / std::f32::consts::PI;
        assert!((d[0] - expected_dr).abs() < 1e-4,
            "dr = {} (expected {})", d[0], expected_dr);
        assert!((d[1] + 5.0).abs() < 1e-4,
            "dV = {} (expected -5)", d[1]);
    }

    #[test]
    fn test_mpr_dfun_with_state() {
        // r=0.5, V=0.3, tau=1, Delta=1, eta=-5, J=15, I=0, cr=1, cv=0, c=0
        // dr = 1*(1/pi + 2*0.3*0.5) = 0.3183 + 0.3 = 0.6183
        // dV = 1*(0.09 - pi^2*0.25 + (-5) + 15*1*0.5 + 0 + 0 + 0)
        //    = 0.09 - 2.4674 - 5 + 7.5 = 0.1226
        let state = Tensor::<B, 2>::from_floats(
            [[0.5_f32, 0.3]],
            &Default::default(),
        );
        let coupling = Tensor::<B, 2>::zeros(
            [1, 2],
            &Default::default(),
        );
        let params = mpr_default_params();
        let d = MontbrioPazoRoxin::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();

        let expected_dr = 1.0 / std::f32::consts::PI + 0.3;
        assert!((d[0] - expected_dr).abs() < 1e-3,
            "dr = {} (expected {})", d[0], expected_dr);
        let expected_dv = 0.09 - std::f32::consts::PI * std::f32::consts::PI * 0.25 - 5.0 + 7.5;
        assert!((d[1] - expected_dv).abs() < 1e-3,
            "dV = {} (expected {})", d[1], expected_dv);
    }

    #[test]
    fn test_mpr_clamp() {
        let mut state = Tensor::<B, 2>::from_floats(
            [[-0.5_f32, 1.0]],
            &Default::default(),
        );
        MontbrioPazoRoxin::clamp(&mut state);
        let vals = state.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-6, "r was not clamped to 0: {}", d[0]);
        assert!((d[1] - 1.0).abs() < 1e-6, "V should stay 1.0: {}", d[1]);
    }
}
