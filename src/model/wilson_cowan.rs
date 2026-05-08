use burn::prelude::Backend;
use burn::tensor::Tensor;
use super::NeuralMassModel;

/// Wilson-Cowan model.
///
/// State variables: E (index 0), I (index 1)
///
/// Equations (with shift_sigmoid = True, hardcoded):
///   x_e = alpha_e * (c_ee*E - c_ei*I + P - theta_e + c_0)
///   x_i = alpha_i * (c_ie*E - c_ii*I + Q - theta_i)
///   s_e = c_e * (1/(1+exp(-a_e*(x_e - b_e))) - 1/(1+exp(a_e*b_e)))
///   s_i = c_i * (1/(1+exp(-a_i*(x_i - b_i))) - 1/(1+exp(a_i*b_i)))
///   dE = (-E + (k_e - r_e*E) * s_e) / tau_e
///   dI = (-I + (k_i - r_i*I) * s_i) / tau_i
///
/// Parameters: [c_ee, c_ei, c_ie, c_ii, tau_e, tau_i, a_e, b_e, c_e, theta_e,
///              a_i, b_i, c_i, theta_i, r_e, r_i, k_e, k_i, P, Q, alpha_e, alpha_i]
/// Default:    [12.0, 4.0, 13.0, 11.0, 10.0, 10.0, 1.2, 2.8, 1.0, 0.0,
///              1.0, 4.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
///
/// NCVAR=1 (Coupling_Term_E only).
pub struct WilsonCowan;

impl<B: Backend> NeuralMassModel<B> for WilsonCowan {
    const NVAR: usize = 2;
    const NCVAR: usize = 1;
    const PARAM_NAMES: &'static [&'static str] = &[
        "c_ee", "c_ei", "c_ie", "c_ii", "tau_e", "tau_i",
        "a_e", "b_e", "c_e", "theta_e",
        "a_i", "b_i", "c_i", "theta_i",
        "r_e", "r_i", "k_e", "k_i", "P", "Q", "alpha_e", "alpha_i",
    ];

    fn dfun(
        state: Tensor<B, 2>,
        coupling: Tensor<B, 2>,
        params: &[f32],
    ) -> Tensor<B, 2> {
        let state3 = state.unsqueeze_dim::<3>(0);
        let coupling3 = coupling.unsqueeze_dim::<3>(0);
        let result3 = crate::engine::batch_engine::dfun::wilson_cowan_dfun_batch::<B>(state3, coupling3, params, None);
        result3.squeeze::<2>(0)
    }

    fn clamp(state: &mut Tensor<B, 2>) {
        *state = state.clone().clamp(0.0, 1.0);
    }
}

/// Convenience: create Wilson-Cowan parameters with Python TVB defaults.
pub fn wilson_cowan_default_params() -> Vec<f32> {
    vec![
        12.0, 4.0, 13.0, 11.0, 10.0, 10.0,
        1.2, 2.8, 1.0, 0.0,
        1.0, 4.0, 1.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_wc_dfun_at_zero() {
        // E=0, I=0, c=0 with defaults (shift_sigmoid=True)
        // x_e = 1*(0 - 0 + 0 - 0 + 0) = 0
        // x_i = 1*(0 - 0 + 0 - 0) = 0
        // sig_e_offset = 1/(1+exp(1.2*2.8)) = 1/(1+exp(3.36)) ≈ 1/28.79 ≈ 0.0347
        // sig_e = 1/(1+exp(-1.2*(0-2.8))) = 1/(1+exp(3.36)) ≈ 0.0347
        // s_e = 1*(0.0347 - 0.0347) = 0
        // dE = (-0 + (1-0)*0)/10 = 0
        let state = Tensor::<B, 2>::zeros([1, 2], &Default::default());
        let coupling = Tensor::<B, 2>::zeros([1, 1], &Default::default());
        let params = wilson_cowan_default_params();
        let d = WilsonCowan::dfun(state, coupling, &params);
        let vals = d.into_data();
        let d = vals.as_slice::<f32>().unwrap();

        assert!(d[0].abs() < 1e-5, "dE = {} (expected 0)", d[0]);
        assert!(d[1].abs() < 1e-5, "dI = {} (expected 0)", d[1]);
    }

    #[test]
    fn test_wc_clamp() {
        let mut state = Tensor::<B, 2>::from_floats(
            [[-0.5_f32, 1.5]],
            &Default::default(),
        );
        WilsonCowan::clamp(&mut state);
        let vals = state.into_data();
        let d = vals.as_slice::<f32>().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-6, "E was not clamped: {}", d[0]);
        assert!((d[1] - 1.0).abs() < 1e-6, "I was not clamped: {}", d[1]);
    }
}
