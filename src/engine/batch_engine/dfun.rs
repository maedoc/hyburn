//! Batch-dimension derivative functions (dfun) for all supported models.
//!
//! Each function operates on tensors of shape `[n_sweep, nnodes, nvar]`
//! and returns derivatives of the same shape.

use burn::prelude::Backend;
use burn::tensor::Tensor;
use crate::engine::EngineModel;

/// Batch-dim model dispatch: calls model-specific batch dfun on [n_sweep, nnodes, nvar].
///
/// The `sweep_param` convention: `Option<(param_idx, &Tensor<B, 3>)>` where
/// `param_idx` is the 0-based index into the model's parameter slice (see
/// `model_param_slice` for each model's parameter layout), and the tensor has
/// shape `[n_sweep, 1, 1]` with per-sweep values. Each model dfun extracts the
/// relevant param_idx from the tensor:
///   - G2DO: idx=1 (I_ext), JansenRit: idx=12 (J), WilsonCowan: idx=18 (tau_e),
///   - Mpr: idx=4 (eta), Kuramoto: idx=0 (K), Rww: idx=7 (J_NMDA)
///
/// Returns derivatives as [n_sweep, nnodes, nvar].
pub fn dfun_batch<B: Backend>(
    model: &EngineModel<B>,
    state: Tensor<B, 3>,    // [n_sweep, nnodes, nvar]
    coupling: Tensor<B, 3>, // [n_sweep, nnodes, ncvar]
    params: &[f32],
    sweep_param: Option<(usize, &Tensor<B, 3>)>, // (param_idx, [n_sweep,1,1])
) -> Tensor<B, 3> {
    match model {
        EngineModel::G2do { .. } => {
            let sweep = match sweep_param {
                Some((1, tensor)) => Some(tensor),
                _ => None,
            };
            g2do_dfun_batch::<B>(state, coupling, params, sweep)
        }
        EngineModel::JansenRit { .. } => {
            let sweep = match sweep_param {
                Some((12, tensor)) => Some(tensor),
                _ => None,
            };
            jansen_rit_dfun_batch::<B>(state, coupling, params, sweep)
        }
        EngineModel::WilsonCowan { .. } => {
            let sweep = match sweep_param {
                Some((18, tensor)) => Some(tensor),
                _ => None,
            };
            wilson_cowan_dfun_batch::<B>(state, coupling, params, sweep)
        }
        EngineModel::Mpr { .. } => {
            let sweep = match sweep_param {
                Some((4, tensor)) => Some(tensor),
                _ => None,
            };
            mpr_dfun_batch::<B>(state, coupling, params, sweep)
        }
        EngineModel::Kuramoto { .. } => {
            let sweep = match sweep_param {
                Some((0, tensor)) => Some(tensor),
                _ => None,
            };
            kuramoto_dfun_batch::<B>(state, coupling, params, sweep)
        }
        EngineModel::Rww { .. } => {
            let sweep = match sweep_param {
                Some((7, tensor)) => Some(tensor),
                _ => None,
            };
            rww_dfun_batch::<B>(state, coupling, params, sweep)
        }
        // Fallback: per-point dispatch through 2D dfun (slow but correct)
        _ => dfun_batch_fallback::<B>(model, state, coupling, params, sweep_param),
    }
}

/// Batch clamp: clamps state values in-place on [n_sweep, nnodes, nvar].
pub fn clamp_batch<B: Backend>(model: &EngineModel<B>, state: &mut Tensor<B, 3>) {
    match model {
        EngineModel::WilsonCowan { .. } => {
            // Both E and I clamped to [0, 1] – single clamp is cheaper than narrow+clamp+cat.
            *state = state.clone().clamp(0.0, 1.0);
        }
        EngineModel::Mpr { .. } => {
            let r = state.clone().narrow(2, 0, 1).clamp(0.0, f32::INFINITY);
            let v = state.clone().narrow(2, 1, 1);
            *state = Tensor::cat(vec![r, v], 2);
        }
        EngineModel::Rww { .. } => {
            let s = state.clone().narrow(2, 0, 1).clamp(0.0, 1.0);
            *state = s;
        }
        EngineModel::Kuramoto { .. } => {
            // Phase normalization: wrap θ into [0, 2π) to prevent floating-point
            // precision loss as phases drift to large magnitudes.
            let two_pi = 2.0 * std::f32::consts::PI;
            *state = state.clone() - (state.clone() / two_pi).floor() * two_pi;
        }
        // G2DO, JansenRit, others: no-op clamp
        _ => {}
    }
}

/// Fallback: iterates over sweep points calling the 2D dfun per point.
/// This is slow but supports all model types.
fn dfun_batch_fallback<B: Backend>(
    model: &EngineModel<B>,
    state: Tensor<B, 3>,    // [n_sweep, nnodes, nvar]
    coupling: Tensor<B, 3>, // [n_sweep, nnodes, ncvar]
    _params: &[f32],
    _sweep_param: Option<(usize, &Tensor<B, 3>)>,
) -> Tensor<B, 3> {
    log::warn!("dfun_batch_fallback: using slow sequential iteration over {} sweep points", state.shape().dims[0]);
    let shape = state.shape();
    let n_sweep = shape.dims[0];
    let _nnodes = shape.dims[1];
    let _nvar = shape.dims[2];
    let _ncvar = coupling.shape().dims[2];

    let mut result = Vec::with_capacity(n_sweep);
    for s in 0..n_sweep {
        let s_state = state.clone().narrow(0, s, 1).squeeze::<2>(0); // [nnodes, nvar]
        let s_coupling = coupling.clone().narrow(0, s, 1).squeeze::<2>(0); // [nnodes, ncvar]
        let s_deriv = model.dfun(s_state, s_coupling);
        result.push(s_deriv.unsqueeze_dim::<3>(0)); // [1, nnodes, nvar]
    }
    Tensor::cat(result, 0)
}

// ---------------------------------------------------------------------------
// Batch model dfuns — [n_sweep, nnodes, nvar] tensor shapes
// ---------------------------------------------------------------------------

pub fn g2do_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    i_ext_sweep: Option<&Tensor<B, 3>>, // [n_sweep, 1, 1] per-sweep, or None for scalar
) -> Tensor<B, 3> {
    let tau_g = params[0];
    debug_assert!(tau_g > 0.0, "G2DO tau must be positive, got {}", tau_g);
    let tau_g = tau_g.max(f32::EPSILON); // guard against division by zero
    let a_g = params[2];
    let b_g = params[3];
    let c_g = params[4];
    let d_g = params[5];
    let e_g = params[6];
    let f_g = params[7];
    let g_g = params[8];
    let alpha = params[9];
    let beta = params[10];
    let gamma = params[11];
    let dtau = d_g * tau_g;
    let d_over_tau = d_g / tau_g;

    let v = state.clone().narrow(2, 0, 1);
    let w = state.clone().narrow(2, 1, 1);
    let c_0 = coupling.narrow(2, 0, 1);

    // I_ext: either scalar from params or per-sweep tensor
    let i_ext_term = match i_ext_sweep {
        Some(tensor) => (c_0.clone() + tensor.clone()).mul_scalar(gamma),
        None => c_0.clone().mul_scalar(gamma).add_scalar(gamma * params[1]),
    };

    let v2 = v.clone() * v.clone();
    let v3 = v.clone() * v2.clone();

    // dV = d*tau*(alpha*W + gamma*(I_ext + c0) - f*V^3 + e*V^2 + g*V)
    let dv = (w.clone().mul_scalar(alpha)
        + i_ext_term
        - v3.mul_scalar(f_g)
        + v2.clone().mul_scalar(e_g)
        + v.clone().mul_scalar(g_g))
        .mul_scalar(dtau);

    // dW = d/tau*(a + b*V + c*V^2 - beta*W)
    let dw = (v.mul_scalar(b_g) + v2.mul_scalar(c_g) - w.mul_scalar(beta) + a_g)
        .mul_scalar(d_over_tau);

    Tensor::cat(vec![dv, dw], 2)
}

pub fn jansen_rit_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    mu_sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let a_p = params[0];
    let b_p = params[1];
    let a = params[2];
    let b = params[3];
    let v0 = params[4];
    let nu_max = params[5];
    let r = params[6];
    let j = params[7];
    let a_1 = params[8];
    let a_2 = params[9];
    let a_3 = params[10];
    let a_4 = params[11];
    let mu = params[12];

    let y0 = state.clone().narrow(2, 0, 1);
    let y1 = state.clone().narrow(2, 1, 1);
    let y2 = state.clone().narrow(2, 2, 1);
    let y3 = state.clone().narrow(2, 3, 1);
    let y4 = state.clone().narrow(2, 4, 1);
    let y5 = state.clone().narrow(2, 5, 1);
    let c_0 = coupling.narrow(2, 0, 1);

    let ones = Tensor::<B, 3>::ones(y0.shape(), &state.device());
    let two_nu_max = 2.0 * nu_max;

    let sigm = |v: Tensor<B, 3>| -> Tensor<B, 3> {
        let arg = v.neg().add_scalar(v0).mul_scalar(r);
        let denom = arg.exp().add_scalar(1.0);
        ones.clone().mul_scalar(two_nu_max) / denom
    };

    let sigm_y1_y2 = sigm(y1.clone() - y2.clone());
    let sigm_y0_1 = sigm(y0.clone().mul_scalar(a_1 * j));
    let sigm_y0_3 = sigm(y0.clone().mul_scalar(a_3 * j));

    let dy0 = y3.clone();
    let dy1 = y4.clone();
    let dy2 = y5.clone();
    let dy3 = sigm_y1_y2.mul_scalar(a_p * a)
        - y3.clone().mul_scalar(2.0 * a)
        - y0.mul_scalar(a * a);

    let mu_term = match mu_sweep {
        Some(tensor) => sigm_y0_1.mul_scalar(a_2 * j) + tensor.clone() + c_0,
        None => sigm_y0_1.mul_scalar(a_2 * j).add_scalar(mu) + c_0,
    };
    let dy4 = mu_term.mul_scalar(a_p * a)
        - y4.clone().mul_scalar(2.0 * a)
        - y1.clone().mul_scalar(a * a);

    let dy5 = sigm_y0_3
        .mul_scalar(a_4 * j)
        .mul_scalar(b_p * b)
        - y5.clone().mul_scalar(2.0 * b)
        - y2.clone().mul_scalar(b * b);

    Tensor::cat(vec![dy0, dy1, dy2, dy3, dy4, dy5], 2)
}

pub fn wilson_cowan_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    p_sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let c_ee = params[0]; let c_ei = params[1]; let c_ie = params[2]; let c_ii = params[3];
    let tau_e = params[4]; let tau_i = params[5];
    debug_assert!(tau_e > 0.0, "WilsonCowan tau_e must be positive, got {}", tau_e);
    debug_assert!(tau_i > 0.0, "WilsonCowan tau_i must be positive, got {}", tau_i);
    let tau_e = tau_e.max(f32::EPSILON); // guard against division by zero
    let tau_i = tau_i.max(f32::EPSILON);
    let a_e = params[6]; let b_e = params[7]; let ce = params[8]; let theta_e = params[9];
    let a_i = params[10]; let b_i = params[11]; let ci = params[12]; let theta_i = params[13];
    let r_e = params[14]; let r_i = params[15]; let k_e = params[16]; let k_i = params[17];
    let p = params[18]; let q = params[19]; let alpha_e = params[20]; let alpha_i = params[21];
    let sig_e_offset = 1.0 / (1.0 + (a_e * b_e).exp());
    let sig_i_offset = 1.0 / (1.0 + (a_i * b_i).exp());

    let e = state.clone().narrow(2, 0, 1);
    let i_val = state.clone().narrow(2, 1, 1);
    let c_0 = coupling.narrow(2, 0, 1);

    let x_e_core = e.clone().mul_scalar(c_ee) - i_val.clone().mul_scalar(c_ei) - theta_e + c_0.clone();
    let x_e = match p_sweep {
        Some(tensor) => x_e_core + tensor.clone(),
        None => x_e_core.add_scalar(p),
    };
    let x_e = x_e.mul_scalar(alpha_e);

    let x_i = (e.clone().mul_scalar(c_ie) - i_val.clone().mul_scalar(c_ii) + q - theta_i)
        .mul_scalar(alpha_i);

    let s_e = {
        let exp_arg = x_e.clone().add_scalar(-b_e).mul_scalar(-a_e).exp();
        let sig_e = x_e.zeros_like().add_scalar(1.0) / (exp_arg + x_e.zeros_like().add_scalar(1.0));
        (sig_e - sig_e_offset).mul_scalar(ce)
    };

    let s_i = {
        let exp_arg = x_i.clone().add_scalar(-b_i).mul_scalar(-a_i).exp();
        let sig_i = x_i.zeros_like().add_scalar(1.0) / (exp_arg + x_i.zeros_like().add_scalar(1.0));
        (sig_i - sig_i_offset).mul_scalar(ci)
    };

    let de = (e.clone().neg() + e.clone().mul_scalar(-r_e).add_scalar(k_e) * s_e)
        .mul_scalar(1.0 / tau_e);
    let di = (i_val.clone().neg() + i_val.mul_scalar(-r_i).add_scalar(k_i) * s_i)
        .mul_scalar(1.0 / tau_i);

    Tensor::cat(vec![de, di], 2)
}

pub fn mpr_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    i_ext_sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let tau = params[0]; debug_assert!(tau > 0.0, "MPR tau must be positive, got {}", tau);
    let tau = tau.max(f32::EPSILON); // guard against division by zero
    let delta = params[1]; let eta = params[2];
    let j = params[3]; let i_ext = params[4]; let cr = params[5]; let cv = params[6];
    let r = state.clone().narrow(2, 0, 1);
    let v = state.clone().narrow(2, 1, 1);
    let c_r = coupling.clone().narrow(2, 0, 1);
    let c_v = coupling.narrow(2, 1, 1);
    let pi = std::f32::consts::PI;
    let inv_tau = 1.0 / tau;
    let dr = r.clone().mul(v.clone()).mul_scalar(2.0)
        .add_scalar(delta / (pi * tau)).mul_scalar(inv_tau);
    let v2 = v.clone() * v.clone();
    let r2 = r.clone() * r.clone();
    let dv = match i_ext_sweep {
        Some(tensor) => {
            v2.sub(r2.mul_scalar(pi * pi * tau * tau))
                .add_scalar(eta)
                + tensor.clone()
                + r.clone().mul_scalar(j * tau)
                + c_r.mul_scalar(cr)
                + c_v.mul_scalar(cv)
        }
        None => {
            v2.sub(r2.mul_scalar(pi * pi * tau * tau))
                .add_scalar(eta + i_ext)
                .add(r.clone().mul_scalar(j * tau))
                .add(c_r.mul_scalar(cr))
                .add(c_v.mul_scalar(cv))
        }
    };
    let dv = dv.mul_scalar(inv_tau);
    Tensor::cat(vec![dr, dv], 2)
}

pub fn kuramoto_dfun_batch<B: Backend>(
    _state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    omega_sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let omega = params[0];
    let c_0 = coupling.narrow(2, 0, 1);
    match omega_sweep {
        Some(tensor) => c_0 + tensor.clone(),
        None => c_0.add_scalar(omega),
    }
}

pub fn rww_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    i_o_sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let a = params[0]; let b = params[1]; let d = params[2]; let gamma = params[3];
    let tau_s = params[4]; let w = params[5]; let j_n = params[6]; let i_o = params[7];
    let s = state.clone().narrow(2, 0, 1);
    let c_0 = coupling.narrow(2, 0, 1);
    let x = match i_o_sweep {
        Some(tensor) => s.clone().mul_scalar(w * j_n) + tensor.clone() + c_0.mul_scalar(j_n),
        None => s.clone().mul_scalar(w * j_n).add_scalar(i_o).add(c_0.mul_scalar(j_n)),
    };
    let ax_b = x.clone().mul_scalar(a).add_scalar(-b);
    let neg_d_ax_b = ax_b.clone().mul_scalar(-d);
    let exp_term = neg_d_ax_b.exp();
    let denom = exp_term.neg().add_scalar(1.0);
    let h = ax_b / denom;
    
    s.clone().neg().mul_scalar(1.0 / tau_s)
        .add(s.clone().neg().add_scalar(1.0) * h * gamma)
}

/// Returns true if the model prefers Heun integration over Euler.
/// Currently only G2DO uses Heun; JR, WC, MPR, Kuramoto, RWW use Euler.
pub fn model_prefers_heun<B: Backend>(model: &EngineModel<B>) -> bool {
    matches!(model, EngineModel::G2do { .. })
}

pub fn model_param_slice<B: Backend>(model: &EngineModel<B>) -> Vec<f32> {
    match model {
        EngineModel::G2do { params } => params.clone(),
        EngineModel::Mpr { params } => params.clone(),
        EngineModel::Rww { params } => params.clone(),
        EngineModel::Kuramoto { params } => params.clone(),
        EngineModel::JansenRit { params } => params.clone(),
        EngineModel::WilsonCowan { params } => params.clone(),
        _ => unreachable!(),
    }
}
