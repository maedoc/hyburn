//! Batch-dimension derivative functions (dfun) for all supported models.
//!
//! Each function operates on tensors of shape `[n_sweep, nnodes, nvar]`
//! and returns derivatives of the same shape.

#![allow(unused_variables, clippy::neg_multiply, clippy::clone_on_copy)]

use burn::prelude::Backend;
use burn::tensor::Tensor;
use crate::engine::EngineModel;

pub fn dfun_batch<B: Backend>(
    model: &EngineModel<B>,
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    sweep_param: Option<(usize, &Tensor<B, 3>)>,
) -> Tensor<B, 3> {
    dfun_batch_dispatch(model, state, coupling, params, sweep_param, &[])
}

pub fn dfun_batch_multi<B: Backend>(
    model: &EngineModel<B>,
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    sweep_params: &[(usize, Tensor<B, 3>)],
) -> Tensor<B, 3> {
    dfun_batch_dispatch(model, state, coupling, params, None, sweep_params)
}

fn dfun_batch_dispatch<B: Backend>(
    model: &EngineModel<B>,
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    sweep_param: Option<(usize, &Tensor<B, 3>)>,
    sweep_params: &[(usize, Tensor<B, 3>)],
) -> Tensor<B, 3> {
    if !sweep_params.is_empty() {
        if sweep_params.len() == 1 {
            let (ref pidx, ref tensor) = sweep_params[0];
            return dfun_batch_dispatch(model, state, coupling, params, Some((*pidx, tensor)), &[]);
        }
        return dfun_batch_multi_fallback(model, state, coupling, params, sweep_params);
    }

    match model {
        EngineModel::G2do { .. } => {
            let sweep = match sweep_param { Some((1, t)) => Some(t), _ => None };
            g2do_dfun_batch::<B>(state, coupling, params, sweep)
        }
        EngineModel::JansenRit { .. } => {
            let sweep = match sweep_param { Some((12, t)) => Some(t), _ => None };
            jansen_rit_dfun_batch::<B>(state, coupling, params, sweep)
        }
        EngineModel::WilsonCowan { .. } => {
            let sweep = match sweep_param { Some((18, t)) => Some(t), _ => None };
            wilson_cowan_dfun_batch::<B>(state, coupling, params, sweep)
        }
        EngineModel::Mpr { .. } => {
            let sweep = match sweep_param { Some((4, t)) => Some(t), _ => None };
            mpr_dfun_batch::<B>(state, coupling, params, sweep)
        }
        EngineModel::Kuramoto { .. } => {
            let sweep = match sweep_param { Some((0, t)) => Some(t), _ => None };
            kuramoto_dfun_batch::<B>(state, coupling, params, sweep)
        }
        EngineModel::Rww { .. } => {
            let sweep = match sweep_param { Some((7, t)) => Some(t), _ => None };
            rww_dfun_batch::<B>(state, coupling, params, sweep)
        }
        EngineModel::Linear { .. } => linear_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::SupHopf { .. } => sup_hopf_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::Hopfield { .. } => hopfield_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::CoombesByrne2D { .. } => coombes_byrne2d_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::CoombesByrne { .. } => coombes_byrne_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::GastSD { .. } => gast_sd_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::GastSF { .. } => gast_sf_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::LarterBreakspear { .. } => larter_breakspear_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::Epileptor2D { .. } => epileptor2d_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::Epileptor { .. } => epileptor_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::RwwExcInh { .. } => rww_exc_inh_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::DecoBalancedExcInh { .. } => deco_balanced_exc_inh_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::EpileptorCodim3 { .. } => epileptor_codim3_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::EpileptorCodim3SlowMod { .. } => epileptor_codim3_slowmod_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::EpileptorRS { .. } => epileptor_rs_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::ZetterbergJansen { .. } => zetterberg_jansen_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::ReducedFHN { .. } => reduced_fhn_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::ReducedHR { .. } => reduced_hr_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::DumontGutkin { .. } => dumont_gutkin_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::ZerlautFirst { .. } => zerlaut_first_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::ZerlautSecond { .. } => zerlaut_second_dfun_batch::<B>(state, coupling, params, None),
        EngineModel::KIonEx { .. } => kionex_dfun_batch::<B>(state, coupling, params, None),
        _ => dfun_batch_fallback::<B>(model, state, coupling, params, sweep_param),
    }
}

fn dfun_batch_multi_fallback<B: Backend>(
    model: &EngineModel<B>,
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    sweep_params: &[(usize, Tensor<B, 3>)],
) -> Tensor<B, 3> {
    let n_sweep = state.shape().dims[0];
    log::warn!("dfun_batch_multi_fallback: using slow per-point iteration over {} sweep points with {} swept params", n_sweep, sweep_params.len());

    let overrides: Vec<Vec<(usize, f32)>> = (0..n_sweep).map(|s| {
        sweep_params.iter().map(|(pidx, tensor)| {
            let val_tensor = tensor.clone().narrow(0, s, 1).squeeze::<2>(0).squeeze::<1>(0);
            let (data, _) = crate::io::tensor_to_flat_f32::<B, 1>(val_tensor);
            (*pidx, data[0])
        }).collect()
    }).collect();

    let mut result = Vec::with_capacity(n_sweep);
    for (s, s_overrides) in overrides.iter().enumerate() {
        let s_state = state.clone().narrow(0, s, 1).squeeze::<2>(0);
        let s_coupling = coupling.clone().narrow(0, s, 1).squeeze::<2>(0);
        let mut modified_params = params.to_vec();
        for (pidx, val) in s_overrides {
            if *pidx < modified_params.len() {
                modified_params[*pidx] = *val;
            }
        }
        let mut temp_model = model.clone();
        set_model_params(&mut temp_model, &modified_params);
        let s_deriv = temp_model.dfun(s_state, s_coupling);
        result.push(s_deriv.unsqueeze_dim::<3>(0));
    }
    Tensor::cat(result, 0)
}

fn set_model_params<B: Backend>(model: &mut EngineModel<B>, params: &[f32]) {
    match model {
        EngineModel::G2do { params: p } => { p.copy_from_slice(params); }
        EngineModel::Mpr { params: p } => { p.copy_from_slice(params); }
        EngineModel::Rww { params: p } => { p.copy_from_slice(params); }
        EngineModel::Kuramoto { params: p } => { p.copy_from_slice(params); }
        EngineModel::JansenRit { params: p } => { p.copy_from_slice(params); }
        EngineModel::WilsonCowan { params: p } => { p.copy_from_slice(params); }
        EngineModel::Linear { params: p } => { p.copy_from_slice(params); }
        EngineModel::SupHopf { params: p } => { p.copy_from_slice(params); }
        EngineModel::Hopfield { params: p } => { p.copy_from_slice(params); }
        EngineModel::CoombesByrne2D { params: p } => { p.copy_from_slice(params); }
        EngineModel::CoombesByrne { params: p } => { p.copy_from_slice(params); }
        EngineModel::GastSD { params: p } => { p.copy_from_slice(params); }
        EngineModel::GastSF { params: p } => { p.copy_from_slice(params); }
        EngineModel::LarterBreakspear { params: p } => { p.copy_from_slice(params); }
        EngineModel::Epileptor2D { params: p } => { p.copy_from_slice(params); }
        EngineModel::Epileptor { params: p } => { p.copy_from_slice(params); }
        EngineModel::RwwExcInh { params: p } => { p.copy_from_slice(params); }
        EngineModel::DecoBalancedExcInh { params: p } => { p.copy_from_slice(params); }
        EngineModel::EpileptorCodim3 { params: p } => { p.copy_from_slice(params); }
        EngineModel::EpileptorCodim3SlowMod { params: p } => { p.copy_from_slice(params); }
        EngineModel::EpileptorRS { params: p } => { p.copy_from_slice(params); }
        EngineModel::ZetterbergJansen { params: p } => { p.copy_from_slice(params); }
        EngineModel::ReducedFHN { params: p } => { p.copy_from_slice(params); }
        EngineModel::ReducedHR { params: p } => { p.copy_from_slice(params); }
        EngineModel::DumontGutkin { params: p } => { p.copy_from_slice(params); }
        EngineModel::ZerlautFirst { params: p } => { p.copy_from_slice(params); }
        EngineModel::ZerlautSecond { params: p } => { p.copy_from_slice(params); }
        EngineModel::KIonEx { params: p } => { p.copy_from_slice(params); }
        _ => {}
    }
}

/// Batch clamp: clamps state values in-place on [n_sweep, nnodes, nvar].
pub fn clamp_batch<B: Backend>(model: &EngineModel<B>, state: &mut Tensor<B, 3>) {
    match model {
        EngineModel::WilsonCowan { .. }
        | EngineModel::RwwExcInh { .. }
        | EngineModel::DecoBalancedExcInh { .. } => {
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
            let two_pi = 2.0 * std::f32::consts::PI;
            *state = state.clone() - (state.clone() / two_pi).floor() * two_pi;
        }
        EngineModel::CoombesByrne2D { .. } => {
            let r = state.clone().narrow(2, 0, 1).clamp(0.0, f32::INFINITY);
            let v = state.clone().narrow(2, 1, 1);
            *state = Tensor::cat(vec![r, v], 2);
        }
        EngineModel::CoombesByrne { .. } => {
            let r = state.clone().narrow(2, 0, 1).clamp(0.0, f32::INFINITY);
            let v = state.clone().narrow(2, 1, 1);
            let g = state.clone().narrow(2, 2, 1);
            let q = state.clone().narrow(2, 3, 1);
            *state = Tensor::cat(vec![r, v, g, q], 2);
        }
        EngineModel::GastSD { .. } | EngineModel::GastSF { .. } => {
            let r = state.clone().narrow(2, 0, 1).clamp(0.0, f32::INFINITY);
            let v = state.clone().narrow(2, 1, 1);
            let a = state.clone().narrow(2, 2, 1);
            let b = state.clone().narrow(2, 3, 1);
            *state = Tensor::cat(vec![r, v, a, b], 2);
        }
        EngineModel::DumontGutkin { .. } => {
            let r_e = state.clone().narrow(2, 0, 1).clamp(0.0, f32::INFINITY);
            let v_e = state.clone().narrow(2, 1, 1);
            let s_ee = state.clone().narrow(2, 2, 1);
            let s_ei = state.clone().narrow(2, 3, 1);
            let r_i = state.clone().narrow(2, 4, 1).clamp(0.0, f32::INFINITY);
            let v_i = state.clone().narrow(2, 5, 1);
            let s_ie = state.clone().narrow(2, 6, 1);
            let s_ii = state.clone().narrow(2, 7, 1);
            *state = Tensor::cat(vec![r_e, v_e, s_ee, s_ei, r_i, v_i, s_ie, s_ii], 2);
        }
        EngineModel::ZerlautFirst { .. } => {
            let e = state.clone().narrow(2, 0, 1).clamp(0.0, 1.0);
            let i_val = state.clone().narrow(2, 1, 1).clamp(0.0, 1.0);
            let rest = state.clone().narrow(2, 2, 3);
            *state = Tensor::cat(vec![e, i_val, rest], 2);
        }
        EngineModel::ZerlautSecond { .. } => {
            let e = state.clone().narrow(2, 0, 1).clamp(0.0, 1.0);
            let i_val = state.clone().narrow(2, 1, 1).clamp(0.0, 1.0);
            let rest = state.clone().narrow(2, 2, 6);
            *state = Tensor::cat(vec![e, i_val, rest], 2);
        }
        EngineModel::KIonEx { .. } => {
            let x = state.clone().narrow(2, 0, 1).clamp(0.0, f32::INFINITY);
            let rest = state.clone().narrow(2, 1, 4);
            *state = Tensor::cat(vec![x, rest], 2);
        }
        _ => {}
    }
}

/// Fallback: iterates over sweep points calling the 2D dfun per point.
/// This is slow but supports all model types.
fn dfun_batch_fallback<B: Backend>(
    model: &EngineModel<B>,
    state: Tensor<B, 3>,    // [n_sweep, nnodes, nvar]
    coupling: Tensor<B, 3>, // [n_sweep, nnodes, ncvar]
    params: &[f32],
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

// ---------------------------------------------------------------------------
// New model batch dfuns
// ---------------------------------------------------------------------------

pub fn linear_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let gamma = params[0];
    let x = state.narrow(2, 0, 1);
    let c_0 = coupling.narrow(2, 0, 1);
    x.mul_scalar(gamma) + c_0
}

pub fn sup_hopf_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let a = params[0];
    let omega = params[1];
    let x = state.clone().narrow(2, 0, 1);
    let y = state.clone().narrow(2, 1, 1);
    let c_0 = coupling.clone().narrow(2, 0, 1);
    let c_1 = coupling.narrow(2, 1, 1);
    let x2 = x.clone() * x.clone();
    let y2 = y.clone() * y.clone();
    let amp = x2 + y2.clone();
    let dx = (amp.clone().neg().add_scalar(a)) * x.clone()
        - y.clone().mul_scalar(omega)
        + c_0;
    let dy = (amp.neg().add_scalar(a)) * y
        + x.mul_scalar(omega)
        + c_1;
    Tensor::cat(vec![dx, dy], 2)
}

pub fn hopfield_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let taux = params[0].max(f32::EPSILON);
    let tau_t = params[1].max(f32::EPSILON);
    let dynamic = params[2] > 0.5;
    let x = state.clone().narrow(2, 0, 1);
    let theta = state.clone().narrow(2, 1, 1);
    let c_0 = coupling.clone().narrow(2, 0, 1);
    let c_1 = coupling.narrow(2, 1, 1);
    let dx = (x.clone().neg() + c_0).mul_scalar(1.0 / taux);
    let dtheta = if dynamic {
        (theta.neg() + c_1).mul_scalar(1.0 / tau_t)
    } else {
        theta.zeros_like()
    };
    Tensor::cat(vec![dx, dtheta], 2)
}

pub fn coombes_byrne2d_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let delta = params[0];
    let v_syn = params[1];
    let k = params[2];
    let eta = params[3];
    let pi = std::f32::consts::PI;
    let r = state.clone().narrow(2, 0, 1);
    let v = state.clone().narrow(2, 1, 1);
    let c_r = coupling.clone().narrow(2, 0, 1);
    let _c_v = coupling.narrow(2, 1, 1);
    let g = r.clone().mul_scalar(k * pi);
    let r2 = r.clone() * r.clone();
    let v2 = v.clone() * v.clone();
    let dr = v.clone() * r.clone().mul_scalar(2.0)
        + r.clone().zeros_like().add_scalar(delta / pi)
        - r2.clone().mul_scalar(k * pi)
        + c_r;
    let dv = v2.neg()
        .add_scalar(eta)
        .sub(r2.mul_scalar(pi * pi))
        .add(v.clone().neg().add_scalar(v_syn).mul_scalar(k * pi).mul(r));
    Tensor::cat(vec![dr, dv], 2)
}

pub fn coombes_byrne_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let delta = params[0];
    let alpha = params[1];
    let v_syn = params[2];
    let k = params[3];
    let eta = params[4];
    let pi = std::f32::consts::PI;
    let r = state.clone().narrow(2, 0, 1);
    let v = state.clone().narrow(2, 1, 1);
    let g = state.clone().narrow(2, 2, 1);
    let q = state.clone().narrow(2, 3, 1);
    let c_r = coupling.narrow(2, 0, 1);
    let dr = (v.clone() * r.clone()).mul_scalar(2.0)
        .add_scalar(delta / pi)
        - g.clone() * r.clone()
        + c_r;
    let dv = (v.clone() * v.clone())
        .add_scalar(eta)
        .sub(r.clone() * r.clone().mul_scalar(pi * pi))
        .add((v.clone().neg().add_scalar(v_syn)).mul(g.clone()));
    let dg = q.clone().mul_scalar(alpha);
    let dq = (r.mul_scalar(k * pi).sub(g.clone()).sub(q.mul_scalar(2.0))).mul_scalar(alpha);
    Tensor::cat(vec![dr, dv, dg, dq], 2)
}

pub fn gast_sd_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let tau = params[0].max(f32::EPSILON);
    let tau_a = params[1].max(f32::EPSILON);
    let alpha_p = params[2];
    let i_ext = params[3];
    let delta = params[4];
    let j = params[5];
    let eta = params[6];
    let cr = params[7];
    let cv = params[8];
    let pi = std::f32::consts::PI;
    let inv_tau = 1.0 / tau;
    let r = state.clone().narrow(2, 0, 1);
    let v = state.clone().narrow(2, 1, 1);
    let a = state.clone().narrow(2, 2, 1);
    let b = state.clone().narrow(2, 3, 1);
    let c_r = coupling.clone().narrow(2, 0, 1);
    let c_v = coupling.narrow(2, 1, 1);
    let dr = (r.clone().mul(v.clone()).mul_scalar(2.0)
        .add_scalar(delta / (pi * tau)))
        .mul_scalar(inv_tau);
    let dv = (v.clone() * v.clone()
        .sub(r.clone() * r.clone().mul_scalar(pi * pi * tau * tau))
        .add_scalar(eta + i_ext)
        .add(r.clone().mul_scalar(j * tau).mul(a.clone().neg().add_scalar(1.0)))
        .add(c_r.mul_scalar(cr))
        .add(c_v.mul_scalar(cv)))
        .mul_scalar(inv_tau);
    let da = b.clone().mul_scalar(1.0 / tau_a);
    let db = (b.mul_scalar(-2.0).sub(a.clone()).add(r.mul_scalar(alpha_p)))
        .mul_scalar(1.0 / tau_a);
    Tensor::cat(vec![dr, dv, da, db], 2)
}

pub fn gast_sf_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let tau = params[0].max(f32::EPSILON);
    let tau_a = params[1].max(f32::EPSILON);
    let alpha_p = params[2];
    let i_ext = params[3];
    let delta = params[4];
    let j = params[5];
    let eta = params[6];
    let cr = params[7];
    let cv = params[8];
    let pi = std::f32::consts::PI;
    let inv_tau = 1.0 / tau;
    let r = state.clone().narrow(2, 0, 1);
    let v = state.clone().narrow(2, 1, 1);
    let a = state.clone().narrow(2, 2, 1);
    let b = state.clone().narrow(2, 3, 1);
    let c_r = coupling.clone().narrow(2, 0, 1);
    let c_v = coupling.narrow(2, 1, 1);
    let dr = (r.clone().mul(v.clone()).mul_scalar(2.0)
        .add_scalar(delta / (pi * tau)))
        .mul_scalar(inv_tau);
    let dv = (v.clone() * v.clone()
        .sub(r.clone() * r.clone().mul_scalar(pi * pi * tau * tau))
        .add_scalar(eta + i_ext)
        .add(r.clone().mul_scalar(j * tau))
        .sub(a.clone())
        .add(c_r.mul_scalar(cr))
        .add(c_v.mul_scalar(cv)))
        .mul_scalar(inv_tau);
    let da = b.clone().mul_scalar(1.0 / tau_a);
    let db = (b.mul_scalar(-2.0).sub(a).add(r.mul_scalar(alpha_p)))
        .mul_scalar(1.0 / tau_a);
    Tensor::cat(vec![dr, dv, da, db], 2)
}

pub fn larter_breakspear_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let g_ca = params[0]; let g_k = params[1]; let g_l = params[2]; let g_na = params[3];
    let phi = params[4];
    let v_ca = params[5]; let v_k = params[6]; let v_l = params[7]; let v_na = params[8];
    let t_ca = params[9]; let t_na = params[10]; let t_k = params[11];
    let d_ca = params[12]; let d_na = params[13]; let d_k = params[14];
    let d_v = params[15]; let d_z = params[16];
    let aei = params[17]; let aie = params[18]; let aee = params[19];
    let ane = params[20]; let ani = params[21];
    let b_p = params[22]; let c_p = params[23]; let i_ext = params[24]; let r_nmda = params[25];
    let v_t = params[26]; let z_t = params[27]; let qv_max = params[28]; let qz_max = params[29];
    let t_scale = params[30]; let tau_k = params[31].max(f32::EPSILON);
    let v_state = state.clone().narrow(2, 0, 1);
    let w = state.clone().narrow(2, 1, 1);
    let z = state.clone().narrow(2, 2, 1);
    let c_0 = coupling.narrow(2, 0, 1);
    let half = v_state.clone().zeros_like().add_scalar(0.5);
    let m_ca = half.clone() * ((v_state.clone().add_scalar(-t_ca)).mul_scalar(1.0 / d_ca)).tanh().add(v_state.clone().ones_like());
    let m_na = half.clone() * ((v_state.clone().add_scalar(-t_na)).mul_scalar(1.0 / d_na)).tanh().add(v_state.clone().ones_like());
    let m_k = half * ((v_state.clone().add_scalar(-t_k)).mul_scalar(1.0 / d_k)).tanh().add(v_state.clone().ones_like());
    let qv = v_state.clone().zeros_like().add_scalar(0.5 * qv_max)
        * ((v_state.clone().add_scalar(-v_t)).mul_scalar(1.0 / d_v)).tanh().add(v_state.clone().ones_like());
    let qz = z.clone().zeros_like().add_scalar(0.5 * qz_max)
        * ((z.clone().add_scalar(-z_t)).mul_scalar(1.0 / d_z)).tanh().add(z.clone().ones_like());
    let ca_coupling = c_0.clone().mul_scalar(c_p * r_nmda * aee);
    let local_nmda = qv.clone().mul_scalar((1.0 - c_p) * r_nmda * aee);
    let v_minus_vca = v_state.clone().add_scalar(-v_ca);
    let v_minus_vk = v_state.clone().add_scalar(-v_k);
    let v_minus_vl = v_state.clone().add_scalar(-v_l);
    let v_minus_vna = v_state.clone().add_scalar(-v_na);
    let total_ca = m_ca.clone().zeros_like().add_scalar(g_ca) + local_nmda.clone() + ca_coupling.clone();
    let total_na = m_na.clone().mul_scalar(g_na) + qv.clone().mul_scalar((1.0 - c_p) * aee) + c_0.clone().mul_scalar(c_p * aee);
    let dv = (
        total_ca.neg() * m_ca.clone() * v_minus_vca
        - w.clone().mul_scalar(g_k) * v_minus_vk
        - v_minus_vl.mul_scalar(g_l)
        - total_na * v_minus_vna
        - z.clone().mul_scalar(aie) * qz.clone()
    ).mul_scalar(t_scale)
        .add_scalar(ane * i_ext);
    let dw = (m_k - w).mul_scalar(t_scale * phi * b_p / tau_k);
    let dz = (v_state.clone().mul_scalar(aei) * qv).add_scalar(ani * i_ext).mul_scalar(t_scale * b_p);
    Tensor::cat(vec![dv, dw, dz], 2)
}

fn sigmoid_burn<B: Backend>(v: Tensor<B, 3>, e0: f32, rho_1: f32, rho_2: f32) -> Tensor<B, 3> {
    let two_e0 = 2.0 * e0;
    let ones = v.clone().zeros_like().add_scalar(1.0);
    let arg = v.neg().add_scalar(rho_2).mul_scalar(rho_1);
    let denom = arg.exp().add_scalar(1.0);
    ones.mul_scalar(two_e0) / denom
}

pub fn epileptor2d_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let i_ext = params[0];
    let x0 = params[1];
    let a = params[2];
    let b = params[3];
    let slope = params[4];
    let c = params[5];
    let d = params[6];
    let r = params[7];
    let kvf = params[8];
    let ks = params[9];
    let tt = params[10];
    let modification = params[11] > 0.5;
    let x1 = state.clone().narrow(2, 0, 1);
    let z = state.clone().narrow(2, 1, 1);
    let c_0 = coupling.clone().narrow(2, 0, 1);
    let x1_neg = x1.clone().lower_elem(0.0);
    let x1_sq = x1.clone() * x1.clone();
    let f1_neg = x1_sq.mul_scalar(a) + x1.clone().mul_scalar(d - b);
    let z_shifted = z.clone().add_scalar(-4.0);
    let z_shifted_sq = z_shifted.clone() * z_shifted;
    let f1_pos = z_shifted_sq.mul_scalar(-0.6).add_scalar(-slope) + x1.clone().mul_scalar(d);
    let f1 = f1_pos.mask_where(x1_neg, f1_neg);
    let dx1 = (z.clone().neg().add_scalar(c + i_ext).add(c_0.clone().mul_scalar(kvf)) - f1 * x1.clone())
        .mul_scalar(tt);
    let z_neg = z.clone().lower_elem(0.0);
    let z_01 = z.clone().mul_scalar(-0.1);
    let ydot_z_neg = (z_01.clone() * z_01.clone() * z_01.clone() * z_01.clone() * z_01.clone() * z_01.clone() * z_01).neg();
    let ydot_z = z.clone().zeros_like().mask_where(z_neg, ydot_z_neg);
    let h = if modification {
        let inner = (x1.clone().add_scalar(0.5)).mul_scalar(10.0).neg().exp().add_scalar(1.0);
        x1.clone().add_scalar(3.0) / inner + ydot_z
    } else {
        x1.clone().mul_scalar(4.0).add_scalar(-4.0 * x0) + ydot_z
    };
    let dz = (h - z.clone() + c_0.mul_scalar(ks)).mul_scalar(tt * r);
    Tensor::cat(vec![dx1, dz], 2)
}

pub fn epileptor_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let i_ext = params[0]; let i_ext2 = params[1]; let x0 = params[2];
    let a = params[3]; let b = params[4]; let c_p = params[5]; let d_p = params[6];
    let r = params[7]; let slope = params[8]; let tau = params[9];
    let aa = params[10]; let bb = params[11];
    let kvf = params[12]; let kf = params[13]; let ks = params[14];
    let tt = params[15]; let modification = params[16] > 0.5;
    let x1 = state.clone().narrow(2, 0, 1);
    let y1 = state.clone().narrow(2, 1, 1);
    let z = state.clone().narrow(2, 2, 1);
    let x2 = state.clone().narrow(2, 3, 1);
    let y2 = state.clone().narrow(2, 4, 1);
    let g = state.clone().narrow(2, 5, 1);
    let c_pop1 = coupling.clone().narrow(2, 0, 1);
    let c_pop2 = coupling.narrow(2, 1, 1);
    let x1_neg = x1.clone().lower_elem(0.0);
    let x1_sq = x1.clone() * x1.clone();
    let f1_neg = x1_sq.neg().mul_scalar(a) + x1.clone().mul_scalar(b);
    let z4 = z.clone().add_scalar(-4.0);
    let z4_sq = z4.clone() * z4;
    let f1_pos = x2.clone().neg().add_scalar(slope).add(z4_sq.mul_scalar(0.6));
    let f1 = f1_pos.mask_where(x1_neg, f1_neg);
    let dx1 = (y1.clone() - z.clone() + c_pop1.clone().mul_scalar(kvf) + f1 * x1.clone())
        .add_scalar(i_ext)
        .mul_scalar(tt);
    let dy1 = ((x1.clone() * x1.clone()).mul_scalar(d_p).neg().add_scalar(c_p) - y1.clone()).mul_scalar(tt);
    let z_neg = z.clone().lower_elem(0.0);
    let z_01 = z.clone().mul_scalar(-0.1);
    let ydot_z_neg = (z_01.clone() * z_01.clone() * z_01.clone() * z_01.clone() * z_01.clone() * z_01.clone() * z_01).neg();
    let ydot_z = z.clone().zeros_like().mask_where(z_neg, ydot_z_neg);
    let h = if modification {
        let inner = (x1.clone().add_scalar(0.5)).mul_scalar(10.0).neg().exp().add_scalar(1.0);
        x1.clone().add_scalar(3.0) / inner + ydot_z
    } else {
        x1.clone().mul_scalar(4.0).add_scalar(-4.0 * x0) + ydot_z
    };
    let dz = (h - z.clone() + c_pop1.mul_scalar(ks)).mul_scalar(tt * r);
    let dx2 = (x2.clone() - x2.clone() * x2.clone() * x2.clone() - y2.clone())
        .add_scalar(i_ext2)
        .add(g.clone().mul_scalar(bb))
        .sub(z.clone().add_scalar(-3.5).mul_scalar(0.3))
        .add(c_pop2.clone().mul_scalar(kf))
        .mul_scalar(tt);
    let x2_thresh = x2.clone().add_scalar(0.25).lower_elem(0.0);
    let f2_pos = x2.clone().add_scalar(0.25).mul_scalar(aa);
    let f2 = f2_pos.mask_where(x2_thresh, x2.zeros_like());
    let dy2 = (f2 - y2.clone()).mul_scalar(tt).div_scalar(tau);
    let dg = (g.neg().mul_scalar(0.1).add(x1)).mul_scalar(-0.01 * tt);
    Tensor::cat(vec![dx1, dy1, dz, dx2, dy2, dg], 2)
}

fn rww_exc_inh_transfer<B: Backend>(x: Tensor<B, 3>, a: f32, b: f32, d: f32) -> Tensor<B, 3> {
    let ax_b = x.clone().mul_scalar(a).add_scalar(-b);
    let neg_d_ax_b = ax_b.clone().mul_scalar(-d);
    let exp_term = neg_d_ax_b.exp();
    let denom = exp_term.neg().add_scalar(1.0);
    ax_b / denom
}

pub fn rww_exc_inh_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let a_e = params[0]; let b_e = params[1]; let d_e = params[2];
    let gamma_e = params[3]; let tau_e = params[4].max(f32::EPSILON);
    let w_p = params[5]; let j_n = params[6]; let w_e = params[7];
    let a_i = params[8]; let b_i = params[9]; let d_i = params[10];
    let gamma_i = params[11]; let tau_i = params[12].max(f32::EPSILON);
    let j_i = params[13]; let w_i = params[14];
    let i_o = params[15]; let i_ext = params[16];
    let g = params[17]; let lamda = params[18];
    let s_e = state.clone().narrow(2, 0, 1);
    let s_i = state.clone().narrow(2, 1, 1);
    let c_0 = coupling.narrow(2, 0, 1);
    let cc = c_0.mul_scalar(g * j_n);
    let jn_se = s_e.clone().mul_scalar(j_n);
    let x_e_raw = jn_se.clone().mul_scalar(w_p)
        .sub(s_i.clone().mul_scalar(j_i))
        .add_scalar(w_e * i_o + i_ext)
        .add(cc.clone());
    let h_e = rww_exc_inh_transfer::<B>(x_e_raw, a_e, b_e, d_e);
    let ds_e = s_e.clone().neg().mul_scalar(1.0 / tau_e)
        .add((s_e.neg().add_scalar(1.0) * h_e).mul_scalar(gamma_e));
    let x_i_raw = jn_se.sub(s_i.clone()).add_scalar(w_i * i_o).add(cc.mul_scalar(lamda));
    let h_i = rww_exc_inh_transfer::<B>(x_i_raw, a_i, b_i, d_i);
    let ds_i = s_i.clone().neg().mul_scalar(1.0 / tau_i)
        .add(h_i.mul_scalar(gamma_i));
    Tensor::cat(vec![ds_e, ds_i], 2)
}

pub fn deco_balanced_exc_inh_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let a_e = params[0]; let b_e = params[1]; let d_e = params[2];
    let gamma_e = params[3]; let tau_e = params[4].max(f32::EPSILON);
    let w_p = params[5]; let j_n = params[6]; let w_e = params[7];
    let a_i = params[8]; let b_i = params[9]; let d_i = params[10];
    let gamma_i = params[11]; let tau_i = params[12].max(f32::EPSILON);
    let j_i = params[13]; let w_i = params[14];
    let i_o = params[15]; let i_ext = params[16];
    let g = params[17]; let lamda = params[18]; let m_i = params[19];
    let s_e = state.clone().narrow(2, 0, 1);
    let s_i = state.clone().narrow(2, 1, 1);
    let c_0 = coupling.narrow(2, 0, 1);
    let cc = c_0.mul_scalar(g * j_n);
    let jn_se = s_e.clone().mul_scalar(j_n);
    let x_e_raw = jn_se.clone().mul_scalar(w_p)
        .sub(s_i.clone().mul_scalar(j_i))
        .add_scalar(w_e * i_o + i_ext)
        .add(cc.clone());
    let x_e_adj = x_e_raw.mul_scalar(a_e * m_i).add_scalar(-b_e * m_i);
    let h_e = rww_exc_inh_transfer::<B>(x_e_adj.clone(), 1.0, 0.0, d_e);
    let ds_e = s_e.clone().neg().mul_scalar(1.0 / tau_e)
        .add((s_e.neg().add_scalar(1.0) * h_e).mul_scalar(gamma_e));
    let x_i_raw = jn_se.sub(s_i.clone()).add_scalar(w_i * i_o).add(cc.mul_scalar(lamda));
    let x_i_adj = x_i_raw.mul_scalar(a_i * m_i).add_scalar(-b_i * m_i);
    let h_i = rww_exc_inh_transfer::<B>(x_i_adj.clone(), 1.0, 0.0, d_i);
    let ds_i = s_i.clone().neg().mul_scalar(1.0 / tau_i)
        .add(h_i.mul_scalar(gamma_i));
    Tensor::cat(vec![ds_e, ds_i], 2)
}

pub fn epileptor_codim3_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let mu1_start = params[0]; let mu2_start = params[1]; let nu_start = params[2];
    let mu1_stop = params[3]; let mu2_stop = params[4]; let nu_stop = params[5];
    let b = params[6]; let r_p = params[7]; let c_p = params[8];
    let dstar = params[9]; let ks = params[10];
    let n_branch = params[11] as i32;
    let modification = params[12] > 0.5;
    let start_pt = [mu2_start, -mu1_start, nu_start];
    let stop_pt = [mu2_stop, -mu1_stop, nu_stop];
    let start_norm: f32 = start_pt.iter().map(|v| v * v).sum::<f32>().sqrt();
    let stop_norm: f32 = stop_pt.iter().map(|v| v * v).sum::<f32>().sqrt();
    let a_pt: [f32; 3] = std::array::from_fn(|i| start_pt[i] / start_norm.max(f32::EPSILON));
    let b_pt: [f32; 3] = std::array::from_fn(|i| stop_pt[i] / stop_norm.max(f32::EPSILON));
    let cross: [f32; 3] = [
        a_pt[1] * b_pt[2] - a_pt[2] * b_pt[1],
        a_pt[2] * b_pt[0] - a_pt[0] * b_pt[2],
        a_pt[0] * b_pt[1] - a_pt[1] * b_pt[0],
    ];
    let cross_norm: f32 = cross.iter().map(|v| v * v).sum::<f32>().sqrt();
    let cross_cross_a: [f32; 3] = if cross_norm > f32::EPSILON {
        let cca: [f32; 3] = std::array::from_fn(|i| cross[(i+1)%3] * a_pt[(i+2)%3] - cross[(i+2)%3] * a_pt[(i+1)%3]);
        let cca_norm: f32 = cca.iter().map(|v| v * v).sum::<f32>().sqrt().max(f32::EPSILON);
        std::array::from_fn(|i| cca[i] / cca_norm)
    } else {
        [0.0, 1.0, 0.0]
    };
    let e_vec = a_pt;
    let f_vec = cross_cross_a;
    let x = state.clone().narrow(2, 0, 1);
    let y = state.clone().narrow(2, 1, 1);
    let z = state.clone().narrow(2, 2, 1);
    let c_0 = coupling.narrow(2, 0, 1);
    let cos_z = z.clone().cos();
    let sin_z = z.clone().sin();
    let mu2 = (cos_z.clone().mul_scalar(e_vec[0]) + sin_z.clone().mul_scalar(f_vec[0])).mul_scalar(r_p);
    let mu1 = (cos_z.clone().mul_scalar(e_vec[1]) + sin_z.clone().mul_scalar(f_vec[1])).mul_scalar(-r_p);
    let nu = (cos_z.mul_scalar(e_vec[2]) + sin_z.mul_scalar(f_vec[2])).mul_scalar(r_p);
    let xs = {
        let disc = mu1.clone().mul_scalar(0.25) * mu1.clone()
            - mu2.clone() * mu2.clone() * mu2.clone().mul_scalar(1.0 / 27.0);
        let sqrt_r = mu2.clone().div_scalar(3.0).clamp(f32::EPSILON, f32::INFINITY).sqrt();
        let r_32 = mu2.clone().div_scalar(3.0) * sqrt_r.clone();
        let cos_3theta = mu1.clone().div_scalar(2.0) / r_32.clone().clamp(f32::EPSILON, f32::INFINITY);
        let cos_3theta_c = cos_3theta.clamp(-1.0, 1.0);
        let theta = acos_approx(cos_3theta_c).div_scalar(3.0);
        let two_pi_3 = 2.0 * std::f32::consts::PI / 3.0;
        let xs_trig = match n_branch {
            2 => sqrt_r.clone().mul_scalar(2.0) * (theta.clone() - two_pi_3).cos(),
            3 => sqrt_r.clone().mul_scalar(2.0) * (theta.clone() + two_pi_3).cos(),
            _ => sqrt_r.mul_scalar(2.0) * theta.cos(),
        };
        let sqrt_disc = disc.clone().clamp(0.0, f32::INFINITY).sqrt();
        let half_mu1 = mu1.clone().div_scalar(2.0);
        let pos_arg = half_mu1.clone() + sqrt_disc.clone();
        let neg_arg = half_mu1 - sqrt_disc;
        let third = pos_arg.clone().zeros_like().add_scalar(1.0 / 3.0);
        let cbrt_pos = pos_arg.clone().sign() * pos_arg.abs().powf(third.clone());
        let cbrt_neg = neg_arg.clone().sign() * neg_arg.abs().powf(third);
        let xs_cardano = cbrt_pos + cbrt_neg;
        let disc_neg = disc.lower_elem(0.0);
        xs_cardano.mask_where(disc_neg, xs_trig)
    };
    let dx = y.clone().neg();
    let dy = x.clone() * x.clone() * x.clone()
        - mu2 * x.clone()
        - mu1
        - y.clone() * (nu + x.clone().mul_scalar(b) + x.clone() * x.clone());
    let dist = (x.clone() - xs.clone()) * (x.clone() - xs) + y.clone() * y.clone();
    let dist = dist.sqrt();
    let dz = if modification {
        let z_shifted = z.clone().add_scalar(-0.5);
        (dist.neg().add_scalar(dstar).neg().add((z_shifted.clone() * z_shifted.clone() * z_shifted.clone() * z_shifted.clone() * z_shifted.clone() * z_shifted.clone() * z_shifted).mul_scalar(0.1)).add(c_0.mul_scalar(ks))).mul_scalar(c_p).neg()
    } else {
        (dist.neg().add_scalar(dstar).neg().add(c_0.mul_scalar(ks))).mul_scalar(c_p).neg()
    };
    Tensor::cat(vec![dx, dy, dz], 2)
}

pub fn epileptor_codim3_slowmod_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let b = params[6]; let r_p = params[7]; let c_p = params[8];
    let dstar = params[9]; let ks = params[10];
    let n_branch = params[11] as i32;
    let modification = params[12] > 0.5;
    let c_a = params[25]; let c_b = params[26];
    let g_vec = [params[13], params[14], params[15]];
    let l_vec = [params[16], params[17], params[18]];
    let h_vec = [params[19], params[20], params[21]];
    let m_vec = [params[22], params[23], params[24]];
    let x = state.clone().narrow(2, 0, 1);
    let y = state.clone().narrow(2, 1, 1);
    let z = state.clone().narrow(2, 2, 1);
    let u_a = state.clone().narrow(2, 3, 1);
    let u_b = state.clone().narrow(2, 4, 1);
    let c_0 = coupling.narrow(2, 0, 1);
    let cos_ua = u_a.clone().cos();
    let sin_ua = u_a.clone().sin();
    let cos_ub = u_b.clone().cos();
    let sin_ub = u_b.clone().sin();
    let a_0 = cos_ua.clone().mul_scalar(g_vec[0] * r_p)
        + sin_ua.clone().mul_scalar(h_vec[0] * r_p);
    let a_1 = cos_ua.clone().mul_scalar(g_vec[1] * r_p)
        + sin_ua.clone().mul_scalar(h_vec[1] * r_p);
    let a_2 = cos_ua.mul_scalar(g_vec[2] * r_p)
        + sin_ua.mul_scalar(h_vec[2] * r_p);
    let b_0 = cos_ub.clone().mul_scalar(l_vec[0] * r_p)
        + sin_ub.clone().mul_scalar(m_vec[0] * r_p);
    let b_1 = cos_ub.clone().mul_scalar(l_vec[1] * r_p)
        + sin_ub.clone().mul_scalar(m_vec[1] * r_p);
    let b_2 = cos_ub.mul_scalar(l_vec[2] * r_p)
        + sin_ub.mul_scalar(m_vec[2] * r_p);
    let a_sq = a_0.clone() * a_0.clone() + a_1.clone() * a_1.clone() + a_2.clone() * a_2.clone();
    let a_norm = a_sq.sqrt().clamp(f32::EPSILON, f32::INFINITY);
    let e_0 = a_0.clone() / a_norm.clone();
    let e_1 = a_1.clone() / a_norm.clone();
    let e_2 = a_2.clone() / a_norm;
    let c0 = a_1.clone() * b_2.clone() - a_2.clone() * b_1.clone();
    let c1 = a_2.clone() * b_0.clone() - a_0.clone() * b_2.clone();
    let c2 = a_0.clone() * b_1.clone() - a_1.clone() * b_0.clone();
    let f_0_raw = c1.clone() * a_2.clone() - c2.clone() * a_1.clone();
    let f_1_raw = c2.clone() * a_0.clone() - c0.clone() * a_2.clone();
    let f_2_raw = c0.clone() * a_1.clone() - c1.clone() * a_0.clone();
    let f_sq = f_0_raw.clone() * f_0_raw.clone()
        + f_1_raw.clone() * f_1_raw.clone()
        + f_2_raw.clone() * f_2_raw.clone();
    let f_norm = f_sq.sqrt().clamp(f32::EPSILON, f32::INFINITY);
    let f_0 = f_0_raw / f_norm.clone();
    let f_1 = f_1_raw / f_norm.clone();
    let f_2 = f_2_raw / f_norm;
    let cos_z = z.clone().cos();
    let sin_z = z.clone().sin();
    let mu2 = (cos_z.clone() * e_0 + sin_z.clone() * f_0).mul_scalar(r_p);
    let mu1_raw = (cos_z.clone() * e_1 + sin_z.clone() * f_1).mul_scalar(r_p);
    let nu = (cos_z * e_2 + sin_z * f_2).mul_scalar(r_p);
    let mu1 = mu1_raw.neg();
    let disc = mu1.clone().mul_scalar(0.25) * mu1.clone()
        - mu2.clone() * mu2.clone() * mu2.clone().mul_scalar(1.0 / 27.0);
    let sqrt_r = mu2.clone().div_scalar(3.0).clamp(f32::EPSILON, f32::INFINITY).sqrt();
    let r_32 = mu2.clone().div_scalar(3.0) * sqrt_r.clone();
    let cos_3theta = mu1.clone().div_scalar(2.0) / r_32.clone().clamp(f32::EPSILON, f32::INFINITY);
    let cos_3theta_c = cos_3theta.clamp(-1.0, 1.0);
    let theta = acos_approx(cos_3theta_c).div_scalar(3.0);
    let two_pi_3 = 2.0 * std::f32::consts::PI / 3.0;
    let xs_trig = match n_branch {
        2 => sqrt_r.clone().mul_scalar(2.0) * (theta.clone() - two_pi_3).cos(),
        3 => sqrt_r.clone().mul_scalar(2.0) * (theta.clone() + two_pi_3).cos(),
        _ => sqrt_r.mul_scalar(2.0) * theta.cos(),
    };
    let sqrt_disc = disc.clone().clamp(0.0, f32::INFINITY).sqrt();
    let half_mu1 = mu1.clone().div_scalar(2.0);
    let pos_arg = half_mu1.clone() + sqrt_disc.clone();
    let neg_arg = half_mu1 - sqrt_disc;
    let third = pos_arg.clone().zeros_like().add_scalar(1.0 / 3.0);
    let cbrt_pos = pos_arg.clone().sign() * pos_arg.abs().powf(third.clone());
    let cbrt_neg = neg_arg.clone().sign() * neg_arg.abs().powf(third);
    let xs_cardano = cbrt_pos + cbrt_neg;
    let disc_neg = disc.lower_elem(0.0);
    let xs = xs_cardano.mask_where(disc_neg, xs_trig);
    let dx = y.clone().neg();
    let dy = x.clone() * x.clone() * x.clone()
        - mu2 * x.clone()
        - mu1
        - y.clone() * (nu + x.clone().mul_scalar(b) + x.clone() * x.clone());
    let dist = ((x.clone() - xs.clone()) * (x.clone() - xs) + y.clone() * y.clone()).sqrt();
    let mod_term = if modification {
        let z_shifted = z.clone().add_scalar(-0.5);
        (z_shifted.clone() * z_shifted.clone() * z_shifted.clone() * z_shifted.clone() * z_shifted.clone() * z_shifted.clone() * z_shifted).mul_scalar(0.1)
    } else {
        z.clone().zeros_like()
    };
    let dz = (dist.neg().add_scalar(dstar).neg().add(mod_term).add(c_0.mul_scalar(ks))).mul_scalar(c_p).neg();
    let dua = x.clone().zeros_like().add_scalar(c_a);
    let dub = x.zeros_like().add_scalar(c_b);
    Tensor::cat(vec![dx, dy, dz, dua, dub], 2)
}

pub fn epileptor_rs_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let i_ext = params[0]; let i_ext2 = params[1]; let x0 = params[2];
    let a = params[3]; let b = params[4]; let c_p = params[5]; let d_p = params[6];
    let r = params[7]; let slope = params[8]; let tau = params[9];
    let aa = params[10]; let bb = params[11];
    let kvf = params[12]; let kf = params[13]; let ks = params[14];
    let tt = params[15]; let modification = params[16] > 0.5;
    let tau_rs = params[17].max(f32::EPSILON);
    let i_rs = params[18];
    let a_rs = params[19]; let b_rs = params[20]; let d_rs = params[21];
    let e_rs = params[22]; let f_rs = params[23];
    let alpha_rs = params[24]; let beta_rs = params[25]; let gamma_rs = params[26];
    let k_rs = params[27];
    let x1 = state.clone().narrow(2, 0, 1);
    let y1 = state.clone().narrow(2, 1, 1);
    let z = state.clone().narrow(2, 2, 1);
    let x2 = state.clone().narrow(2, 3, 1);
    let y2 = state.clone().narrow(2, 4, 1);
    let g = state.clone().narrow(2, 5, 1);
    let x_rs = state.clone().narrow(2, 6, 1);
    let y_rs = state.clone().narrow(2, 7, 1);
    let c_pop1 = coupling.clone().narrow(2, 0, 1);
    let c_pop2 = coupling.clone().narrow(2, 1, 1);
    let c_pop3 = coupling.narrow(2, 2, 1);
    let x1_neg = x1.clone().lower_elem(0.0);
    let x1_sq = x1.clone() * x1.clone();
    let f1_neg = x1_sq.neg().mul_scalar(a) + x1.clone().mul_scalar(b);
    let z4 = z.clone().add_scalar(-4.0);
    let z4_sq = z4.clone() * z4;
    let f1_pos = x2.clone().neg().add_scalar(slope).add(z4_sq.mul_scalar(0.6));
    let f1 = f1_pos.mask_where(x1_neg, f1_neg);
    let dx1 = (y1.clone() - z.clone() + c_pop1.clone().mul_scalar(kvf) + f1 * x1.clone())
        .add_scalar(i_ext)
        .mul_scalar(tt);
    let dy1 = (y1.clone().neg().add_scalar(c_p).sub((x1.clone() * x1.clone()).mul_scalar(d_p))).mul_scalar(tt);
    let z_neg = z.clone().lower_elem(0.0);
    let z_01 = z.clone().mul_scalar(-0.1);
    let ydot_z_neg = (z_01.clone() * z_01.clone() * z_01.clone() * z_01.clone() * z_01.clone() * z_01.clone() * z_01).neg();
    let ydot_z = z.clone().zeros_like().mask_where(z_neg, ydot_z_neg);
    let h = if modification {
        let inner = (x1.clone().add_scalar(0.5)).mul_scalar(10.0).neg().exp().add_scalar(1.0);
        x1.clone().add_scalar(3.0) / inner + ydot_z
    } else {
        x1.clone().mul_scalar(4.0).add_scalar(-4.0 * x0) + ydot_z
    };
    let dz = (h - z.clone() + c_pop1.mul_scalar(ks)).mul_scalar(tt * r);
    let dx2 = (x2.clone() - x2.clone() * x2.clone() * x2.clone() - y2.clone())
        .add_scalar(i_ext2)
        .add(g.clone().mul_scalar(bb))
        .sub(z.clone().add_scalar(-3.5).mul_scalar(0.3))
        .add(c_pop2.clone().mul_scalar(kf))
        .mul_scalar(tt);
    let x2_thresh = x2.clone().add_scalar(0.25).lower_elem(0.0);
    let f2_pos = x2.clone().add_scalar(0.25).mul_scalar(aa);
    let f2 = f2_pos.mask_where(x2_thresh, x2.zeros_like());
    let dy2 = (f2 - y2.clone()).mul_scalar(tt).div_scalar(tau);
    let dg = (x1.neg().mul_scalar(0.1).add(g)).mul_scalar(-0.01 * tt);
    let dtau_rs = d_rs * tau_rs;
    let d_over_tau_rs = d_rs / tau_rs;
    let dx_rs = (y_rs.clone().mul_scalar(alpha_rs)
        .sub((x_rs.clone() * x_rs.clone() * x_rs.clone()).mul_scalar(f_rs))
        .add((x_rs.clone() * x_rs.clone()).mul_scalar(e_rs))
        .add_scalar(gamma_rs * i_rs)
        .add(c_pop3.mul_scalar(gamma_rs * k_rs)))
        .mul_scalar(dtau_rs);
    let dy_rs = (x_rs.mul_scalar(b_rs).sub(y_rs.mul_scalar(beta_rs)).add_scalar(a_rs))
        .mul_scalar(d_over_tau_rs);
    Tensor::cat(vec![dx1, dy1, dz, dx2, dy2, dg, dx_rs, dy_rs], 2)
}

pub fn zetterberg_jansen_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let he = params[0]; let hi = params[1];
    let ke = params[2]; let ki = params[3];
    let e0 = params[4]; let rho_2 = params[5]; let rho_1 = params[6];
    let gamma_1 = params[7]; let gamma_2 = params[8];
    let gamma_3 = params[9]; let gamma_4 = params[10]; let gamma_5 = params[11];
    let gamma_1t = params[12]; let gamma_2t = params[13]; let gamma_3t = params[14];
    let p = params[15]; let u = params[16]; let q = params[17];
    let heke = he * ke; let hiki = hi * ki;
    let ke2 = ke * ke; let ki2 = ki * ki;
    let v1 = state.clone().narrow(2, 0, 1);
    let y1 = state.clone().narrow(2, 1, 1);
    let v2 = state.clone().narrow(2, 2, 1);
    let y2 = state.clone().narrow(2, 3, 1);
    let v3 = state.clone().narrow(2, 4, 1);
    let y3 = state.clone().narrow(2, 5, 1);
    let v4 = state.clone().narrow(2, 6, 1);
    let y4 = state.clone().narrow(2, 7, 1);
    let v5 = state.clone().narrow(2, 8, 1);
    let y5 = state.clone().narrow(2, 9, 1);
    let v6 = state.clone().narrow(2, 10, 1);
    let v7 = state.clone().narrow(2, 11, 1);
    let c_0 = coupling.narrow(2, 0, 1);
    let sig_y0 = sigmoid_burn(v1.clone(), e0, rho_1, rho_2);
    let sig_y2_y3 = sigmoid_burn(v2.clone() - v3.clone(), e0, rho_1, rho_2);
    let sig_y4_y5 = sigmoid_burn(v4.clone() - v5.clone(), e0, rho_1, rho_2);
    let dv1 = y1.clone();
    let dy1 = sig_y2_y3.clone().mul_scalar(heke * gamma_1)
        .sub(y1.clone().mul_scalar(2.0 * ke))
        .sub(v1.mul_scalar(ke2));
    let dv2 = y2.clone();
    let dy2 = (sig_y0.clone().mul_scalar(gamma_2t * gamma_2).add_scalar(p).add(c_0.clone()))
        .mul_scalar(heke)
        .sub(y2.clone().mul_scalar(2.0 * ke))
        .sub(v2.mul_scalar(ke2));
    let dv3 = y3.clone();
    let dy3 = (sig_y0.clone().mul_scalar(gamma_3t * gamma_3).add_scalar(q))
        .mul_scalar(hiki)
        .sub(y3.clone().mul_scalar(2.0 * ki))
        .sub(v3.mul_scalar(ki2));
    let dv4 = y4.clone();
    let dy4 = sig_y2_y3.mul_scalar(heke * gamma_4)
        .sub(y4.clone().mul_scalar(2.0 * ke))
        .sub(v4.mul_scalar(ke2));
    let dv5 = y5.clone();
    let dy5 = sig_y0.mul_scalar(hiki * gamma_5)
        .sub(y5.clone().mul_scalar(2.0 * ki))
        .sub(v5.mul_scalar(ki2));
    let dv6 = y2 - y3;
    let dv7 = y4 - y5;
    Tensor::cat(vec![dv1, dy1, dv2, dy2, dv3, dy3, dv4, dy4, dv5, dy5, dv6, dv7], 2)
}

pub fn reduced_fhn_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let tau = params[0];
    let a_p = params[1];
    let b_p = params[2];
    let k11 = params[3];
    let k12 = params[4];
    let k21 = params[5];
    let xi = state.clone().narrow(2, 0, 1);
    let eta = state.clone().narrow(2, 1, 1);
    let alpha = state.clone().narrow(2, 2, 1);
    let beta = state.clone().narrow(2, 3, 1);
    let c_0 = coupling.clone().narrow(2, 0, 1);
    let c_1 = coupling.narrow(2, 1, 1);
    let xi3 = xi.clone() * xi.clone() * xi.clone();
    let alpha3 = alpha.clone() * alpha.clone() * alpha.clone();
    let dxi = (xi.clone().sub(xi3.div_scalar(3.0)).sub(eta.clone())).mul_scalar(tau)
        .add(xi.clone().neg().mul_scalar(k11))
        .sub((alpha.clone() - xi.clone()).mul_scalar(k12))
        .add(c_0.mul_scalar(tau));
    let deta = (xi.clone().sub(eta.mul_scalar(b_p)).add_scalar(a_p)).div_scalar(tau);
    let dalpha = (alpha.clone().sub(alpha3.div_scalar(3.0)).sub(beta.clone())).mul_scalar(tau)
        .add(xi.clone().neg().mul_scalar(k21))
        .add(c_1.mul_scalar(tau));
    let dbeta = (alpha.sub(beta.mul_scalar(b_p)).add_scalar(a_p)).div_scalar(tau);
    Tensor::cat(vec![dxi, deta, dalpha, dbeta], 2)
}

pub fn reduced_hr_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let r_p = params[0];
    let a_p = params[1];
    let b_p = params[2];
    let c_p = params[3];
    let d_p = params[4];
    let s_p = params[5];
    let x0 = params[6];
    let k11 = params[7];
    let k12 = params[8];
    let k21 = params[9];
    let xi = state.clone().narrow(2, 0, 1);
    let eta = state.clone().narrow(2, 1, 1);
    let tau_state = state.clone().narrow(2, 2, 1);
    let alpha = state.clone().narrow(2, 3, 1);
    let beta = state.clone().narrow(2, 4, 1);
    let gamma = state.clone().narrow(2, 5, 1);
    let c_0 = coupling.clone().narrow(2, 0, 1);
    let c_1 = coupling.narrow(2, 1, 1);
    let xi3 = xi.clone() * xi.clone() * xi.clone();
    let alpha3 = alpha.clone() * alpha.clone() * alpha.clone();
    let xi2 = xi.clone() * xi.clone();
    let alpha2 = alpha.clone() * alpha.clone();
    let dxi = eta.clone()
        .sub(xi3.mul_scalar(a_p))
        .add(xi2.clone().mul_scalar(b_p))
        .sub(tau_state.clone())
        .add(xi.clone().neg().mul_scalar(k11))
        .sub((alpha.clone() - xi.clone()).mul_scalar(k12))
        .add(c_0);
    let deta = xi2.neg().mul_scalar(d_p).sub(eta).add_scalar(c_p);
    let dtau = xi.clone().mul_scalar(r_p * s_p).sub(tau_state.mul_scalar(r_p)).add_scalar(-r_p * x0);
    let dalpha = beta.clone()
        .sub(alpha3.mul_scalar(a_p))
        .add(alpha2.clone().mul_scalar(b_p))
        .sub(gamma.clone())
        .add(xi.clone().neg().mul_scalar(k21))
        .add(c_1);
    let dbeta = alpha2.neg().mul_scalar(d_p).sub(beta).add_scalar(c_p);
    let dgamma = alpha.mul_scalar(r_p * s_p).sub(gamma.mul_scalar(r_p)).add_scalar(-r_p * x0);
    Tensor::cat(vec![dxi, deta, dtau, dalpha, dbeta, dgamma], 2)
}

pub fn dumont_gutkin_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let i_e = params[0]; let delta_e = params[1]; let eta_e = params[2]; let tau_e = params[3].max(f32::EPSILON);
    let i_i = params[4]; let delta_i = params[5]; let eta_i = params[6]; let tau_i = params[7].max(f32::EPSILON);
    let tau_s = params[8].max(f32::EPSILON);
    let j_ee = params[9]; let j_ei = params[10]; let j_ie = params[11]; let j_ii = params[12];
    let gamma_p = params[13];
    let pi = std::f32::consts::PI;
    let r_e = state.clone().narrow(2, 0, 1);
    let v_e = state.clone().narrow(2, 1, 1);
    let s_ee = state.clone().narrow(2, 2, 1);
    let s_ei = state.clone().narrow(2, 3, 1);
    let r_i = state.clone().narrow(2, 4, 1);
    let v_i = state.clone().narrow(2, 5, 1);
    let s_ie = state.clone().narrow(2, 6, 1);
    let s_ii = state.clone().narrow(2, 7, 1);
    let c_r_e = coupling.clone().narrow(2, 0, 1);
    let c_v_e = coupling.clone().narrow(2, 1, 1);
    let c_r_i = coupling.clone().narrow(2, 2, 1);
    let c_v_i = coupling.narrow(2, 3, 1);
    let inv_tau_e = 1.0 / tau_e;
    let inv_tau_i = 1.0 / tau_i;
    let inv_tau_s = 1.0 / tau_s;
    let dr_e = (v_e.clone().mul_scalar(2.0) * r_e.clone()).add_scalar(delta_e / (pi * tau_e)).mul_scalar(inv_tau_e);
    let dv_e = (v_e.clone() * v_e
        .sub((r_e.clone() * r_e.clone()).mul_scalar(pi * pi * tau_e * tau_e))
        .add_scalar(eta_e)
        .add(s_ee.clone().mul_scalar(tau_e))
        .sub(s_ei.clone().mul_scalar(tau_e))
        .add_scalar(i_e)
        .add(c_r_e)
        .add(c_v_e.mul_scalar(0.0)))
        .mul_scalar(inv_tau_e);
    let ds_ee = (s_ee.neg().add(r_e.clone().mul_scalar(j_ee)).add(c_r_i.clone())).mul_scalar(inv_tau_s);
    let ds_ei = (s_ei.neg().add(r_i.clone().mul_scalar(j_ei))).mul_scalar(inv_tau_s);
    let dr_i = (v_i.clone().mul_scalar(2.0) * r_i.clone()).add_scalar(delta_i / (pi * tau_i)).mul_scalar(inv_tau_i);
    let dv_i = (v_i.clone() * v_i
        .sub((r_i.clone() * r_i.clone()).mul_scalar(pi * pi * tau_i * tau_i))
        .add_scalar(eta_i)
        .add(s_ie.clone().mul_scalar(tau_i))
        .sub(s_ii.clone().mul_scalar(tau_i))
        .add_scalar(i_i)
        .add(c_r_i.clone().mul_scalar(gamma_p)))
        .mul_scalar(inv_tau_i);
    let ds_ie = (s_ie.neg().add(r_e.mul_scalar(j_ie)).add(c_r_i.mul_scalar(gamma_p))).mul_scalar(inv_tau_s);
    let ds_ii = (s_ii.neg().add(r_i.mul_scalar(j_ii))).mul_scalar(inv_tau_s);
    Tensor::cat(vec![dr_e, dv_e, ds_ee, ds_ei, dr_i, dv_i, ds_ie, ds_ii], 2)
}

fn atan_approx<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let abs_x = x.clone().abs();
    let needs_reduction = abs_x.greater_elem(1.0);
    let x_eff = x.clone().mask_where(needs_reduction.clone(), x.clone().recip());
    let x2 = x_eff.clone() * x_eff.clone();
    let mut theta = x_eff.clone() / x2.mul_scalar(0.28).add_scalar(1.0);
    for _ in 0..12 {
        let s = theta.clone().sin();
        let c = theta.clone().cos();
        theta = theta - (s - x_eff.clone() * c.clone()) * c;
    }
    let reduced = theta.clone().neg().add_scalar(std::f32::consts::FRAC_PI_2);
    theta.mask_where(needs_reduction, reduced)
}

fn acos_approx<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let x_c = x.clamp(-0.9999, 0.9999);
    let ratio = x_c.clone().neg().add_scalar(1.0) / x_c.add_scalar(1.0);
    atan_approx(ratio.sqrt()).mul_scalar(2.0)
}

fn erf_approx<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = x.clone().sign();
    let ax = x.abs();
    let t = (ax.clone().mul_scalar(p).add_scalar(1.0)).recip();
    let t2 = t.clone() * t.clone();
    let t3 = t2.clone() * t.clone();
    let t4 = t3.clone() * t.clone();
    let t5 = t4.clone() * t.clone();
    let y = t5.mul_scalar(a5)
        .add(t4.mul_scalar(a4))
        .add(t3.mul_scalar(a3))
        .add(t2.mul_scalar(a2))
        .add(t.mul_scalar(a1));
    let exp_val = (ax.clone().mul_scalar(-1.0) * ax).exp();
    let y = y.mul(exp_val).neg().add_scalar(1.0);
    sign * y
}

#[allow(clippy::too_many_arguments)]
fn zerlaut_get_fluct_regime_vars<B: Backend>(
    fe: &Tensor<B, 3>, fi: &Tensor<B, 3>,
    fe_ext: &Tensor<B, 3>, fi_ext: &Tensor<B, 3>,
    w: &Tensor<B, 3>, e_l: f32,
    g_l: f32, c_m: f32,
    q_e: f32, tau_e: f32, e_e: f32,
    q_i: f32, tau_i: f32, e_i: f32,
    n_tot: f32, p_con_e: f32, p_con_i: f32,
    g_frac: f32, k_ext_e: f32, k_ext_i: f32,
) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
    let fe_tot = fe.clone().add_scalar(1e-6).mul_scalar((1.0 - g_frac) * p_con_e * n_tot)
        + fe_ext.clone().mul_scalar(k_ext_e);
    let fi_tot = fi.clone().add_scalar(1e-6).mul_scalar(g_frac * p_con_i * n_tot)
        + fi_ext.clone().mul_scalar(k_ext_i);
    let mu_ge = q_e * tau_e;
    let mu_gi = q_i * tau_i;
    let mu_ge_t = fe_tot.clone().mul_scalar(mu_ge);
    let mu_gi_t = fi_tot.clone().mul_scalar(mu_gi);
    let mu_g = mu_ge_t.clone().add(mu_gi_t.clone()).add_scalar(g_l);
    let t_m = mu_g.clone().recip().mul_scalar(c_m);
    let mu_v = (mu_ge_t.mul_scalar(e_e)
        + mu_gi_t.mul_scalar(e_i)
        + w.clone().neg())
        .add_scalar(g_l * e_l)
        / mu_g.clone();
    let u_e = mu_v.clone().neg().add_scalar(e_e).mul_scalar(q_e) / mu_g.clone();
    let u_i = mu_v.clone().neg().add_scalar(e_i).mul_scalar(q_i) / mu_g.clone();
    let ue_tau = u_e.clone().mul_scalar(tau_e);
    let ui_tau = u_i.clone().mul_scalar(tau_i);
    let ue_tau_sq = ue_tau.clone() * ue_tau;
    let ui_tau_sq = ui_tau.clone() * ui_tau;
    let sigma_v_sq = fe_tot.clone() * ue_tau_sq.clone() / (t_m.clone().add_scalar(tau_e).mul_scalar(2.0))
        + fi_tot.clone() * ui_tau_sq.clone() / (t_m.clone().add_scalar(tau_i).mul_scalar(2.0));
    let sigma_v = sigma_v_sq.sqrt().clamp(1e-10, f32::INFINITY);
    let t_v_num = fe_tot.clone() * ue_tau_sq.clone() + fi_tot.clone() * ui_tau_sq.clone();
    let t_v_den = fe_tot * ue_tau_sq / (t_m.clone().add_scalar(tau_e))
        + fi_tot * ui_tau_sq / (t_m.add_scalar(tau_i));
    let t_v = t_v_num / t_v_den.clamp(1e-10, f32::INFINITY);
    (mu_v, sigma_v, t_v)
}

fn zerlaut_threshold_func<B: Backend>(
    mu_v: &Tensor<B, 3>, sigma_v: &Tensor<B, 3>, tvn: &Tensor<B, 3>,
    p: &[f32; 10],
) -> Tensor<B, 3> {
    let v = mu_v.clone().add_scalar(60.0).div_scalar(10.0);
    let s = sigma_v.clone().add_scalar(-4.0).div_scalar(6.0);
    let t = tvn.clone().add_scalar(-0.5);
    let v2 = v.clone() * v.clone();
    let s2 = s.clone() * s.clone();
    let t2 = t.clone() * t.clone();
    let vs = v.clone() * s.clone();
    let vt = v.clone() * t.clone();
    let st = s.clone() * t.clone();
    v2.mul_scalar(p[4])
        .add(s2.mul_scalar(p[5]))
        .add(t2.mul_scalar(p[6]))
        .add(vs.mul_scalar(p[7]))
        .add(vt.mul_scalar(p[8]))
        .add(st.mul_scalar(p[9]))
        .add(v.mul_scalar(p[1]))
        .add(s.mul_scalar(p[2]))
        .add(t.mul_scalar(p[3]))
        .add_scalar(p[0])
}

fn zerlaut_firing_rate<B: Backend>(
    mu_v: &Tensor<B, 3>, sigma_v: &Tensor<B, 3>, t_v: &Tensor<B, 3>,
    v_thre: &Tensor<B, 3>,
) -> Tensor<B, 3> {
    let arg = (v_thre.clone() - mu_v.clone()) / sigma_v.clone().mul_scalar(std::f32::consts::SQRT_2);
    let erfc_val = erf_approx(arg).neg().add_scalar(1.0);
    erfc_val / t_v.clone().mul_scalar(2.0).clamp(f32::EPSILON, f32::INFINITY)
}

#[allow(clippy::too_many_arguments)]
fn zerlaut_tf<B: Backend>(
    fe: &Tensor<B, 3>, fi: &Tensor<B, 3>,
    fe_ext: &Tensor<B, 3>, fi_ext: &Tensor<B, 3>,
    w: &Tensor<B, 3>, e_l: f32,
    g_l: f32, c_m: f32,
    q_e: f32, tau_e: f32, e_e: f32,
    q_i: f32, tau_i: f32, e_i: f32,
    n_tot: f32, p_con_e: f32, p_con_i: f32,
    g_frac: f32, k_ext_e: f32, k_ext_i: f32,
    p: &[f32; 10],
) -> Tensor<B, 3> {
    let (mu_v, sigma_v, t_v) = zerlaut_get_fluct_regime_vars(
        fe, fi, fe_ext, fi_ext, w, e_l,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i);
    let tvn = t_v.clone().mul_scalar(g_l / c_m);
    let v_thre = zerlaut_threshold_func(&mu_v, &sigma_v, &tvn, p);
    let v_thre = v_thre.mul_scalar(1000.0);
    zerlaut_firing_rate(&mu_v, &sigma_v, &t_v, &v_thre)
}

pub fn zerlaut_first_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let g_l = params[0]; let e_l_e = params[1]; let e_l_i = params[2]; let c_m = params[3];
    let b_e = params[4]; let a_e = params[5]; let b_i = params[6]; let a_i = params[7];
    let tau_w_e = params[8]; let tau_w_i = params[9];
    let e_e = params[10]; let e_i = params[11];
    let q_e = params[12]; let q_i = params[13];
    let tau_e = params[14]; let tau_i = params[15];
    let n_tot = params[16];
    let p_con_e = params[17]; let p_con_i = params[18];
    let g_frac = params[19];
    let k_ext_e = params[20]; let k_ext_i = params[21];
    let t_p = params[22];
    let ext_ex_ex = params[23]; let ext_ex_in = params[24];
    let ext_in_ex = params[25]; let ext_in_in = params[26];
    let tau_ou = params[27]; let w_noise = params[28]; let s_i_scale = params[29];
    let p_e = [params[30], params[31], params[32], params[33], params[34],
               params[35], params[36], params[37], params[38], params[39]];
    let p_i = [params[40], params[41], params[42], params[43], params[44],
               params[45], params[46], params[47], params[48], params[49]];
    let e = state.clone().narrow(2, 0, 1);
    let i_val = state.clone().narrow(2, 1, 1);
    let w_e = state.clone().narrow(2, 2, 1);
    let w_i = state.clone().narrow(2, 3, 1);
    let ou = state.clone().narrow(2, 4, 1);
    let c_0 = coupling.narrow(2, 0, 1);
    let fe_ext_raw = c_0 + ou.clone().mul_scalar(w_noise);
    let fe_ext_neg = fe_ext_raw.clone().mul_scalar(k_ext_e).lower_elem(0.0);
    let fe_ext = fe_ext_raw.mask_where(fe_ext_neg, Tensor::zeros_like(&e));
    let fi_ext = Tensor::zeros_like(&e);
    let fe_ext_ex = fe_ext.clone().add_scalar(ext_ex_ex);
    let fi_ext_ex = fi_ext.clone().add_scalar(ext_ex_in);
    let fe_ext_in = fe_ext.clone().add_scalar(ext_in_ex);
    let fi_ext_in = fi_ext.clone().add_scalar(ext_in_in);
    let tf_e = zerlaut_tf(&e, &i_val, &fe_ext_ex, &fi_ext_ex, &w_e, e_l_e,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_e);
    let tf_i = zerlaut_tf(&e, &i_val, &fe_ext_in, &fi_ext_in, &w_i, e_l_i,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_i);
    let de = (tf_e - e.clone()).div_scalar(t_p);
    let di = (tf_i - i_val.clone()).div_scalar(t_p);
    let (mu_v_e, _, _) = zerlaut_get_fluct_regime_vars(
        &e, &i_val, &fe_ext_ex, &fi_ext_ex, &w_e, e_l_e,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i);
    let dw_e = w_e.clone().neg().div_scalar(tau_w_e)
        + e.clone().mul_scalar(b_e)
        + mu_v_e.add_scalar(-e_l_e).mul_scalar(a_e).div_scalar(tau_w_e);
    let (mu_v_i, _, _) = zerlaut_get_fluct_regime_vars(
        &e, &i_val, &fe_ext_in, &fi_ext_in, &w_i, e_l_i,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i);
    let dw_i = w_i.clone().neg().div_scalar(tau_w_i)
        + i_val.clone().mul_scalar(b_i)
        + mu_v_i.add_scalar(-e_l_i).mul_scalar(a_i).div_scalar(tau_w_i);
    let dou = ou.neg().div_scalar(tau_ou);
    Tensor::cat(vec![de, di, dw_e, dw_i, dou], 2)
}

pub fn zerlaut_second_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let g_l = params[0]; let e_l_e = params[1]; let e_l_i = params[2]; let c_m = params[3];
    let b_e = params[4]; let a_e = params[5]; let b_i = params[6]; let a_i = params[7];
    let tau_w_e = params[8]; let tau_w_i = params[9];
    let e_e = params[10]; let e_i = params[11];
    let q_e = params[12]; let q_i = params[13];
    let tau_e = params[14]; let tau_i = params[15];
    let n_tot = params[16];
    let p_con_e = params[17]; let p_con_i = params[18];
    let g_frac = params[19];
    let k_ext_e = params[20]; let k_ext_i = params[21];
    let t_p = params[22];
    let ext_ex_ex = params[23]; let ext_ex_in = params[24];
    let ext_in_ex = params[25]; let ext_in_in = params[26];
    let tau_ou = params[27]; let w_noise = params[28]; let s_i_scale = params[29];
    let p_e = [params[30], params[31], params[32], params[33], params[34],
               params[35], params[36], params[37], params[38], params[39]];
    let p_i = [params[40], params[41], params[42], params[43], params[44],
               params[45], params[46], params[47], params[48], params[49]];
    let n_e = n_tot * (1.0 - g_frac);
    let n_i = n_tot * g_frac;
    let e = state.clone().narrow(2, 0, 1);
    let i_val = state.clone().narrow(2, 1, 1);
    let c_ee = state.clone().narrow(2, 2, 1);
    let c_ei = state.clone().narrow(2, 3, 1);
    let c_ii = state.clone().narrow(2, 4, 1);
    let w_e = state.clone().narrow(2, 5, 1);
    let w_i = state.clone().narrow(2, 6, 1);
    let ou = state.clone().narrow(2, 7, 1);
    let c_0 = coupling.narrow(2, 0, 1);
    let e_input_exc = c_0.clone() + ou.clone().mul_scalar(w_noise).add_scalar(ext_ex_ex);
    let neg_mask_exc = e_input_exc.clone().lower_elem(0.0);
    let zeros_exc = Tensor::zeros_like(&e_input_exc);
    let e_input_exc = e_input_exc.mask_where(neg_mask_exc, zeros_exc);
    let e_input_inh = c_0.mul_scalar(s_i_scale) + ou.clone().mul_scalar(w_noise).add_scalar(ext_in_ex);
    let neg_mask_inh = e_input_inh.clone().lower_elem(0.0);
    let zeros_inh = Tensor::zeros_like(&e_input_inh);
    let e_input_inh = e_input_inh.mask_where(neg_mask_inh, zeros_inh);
    let i_input_exc = i_val.clone().zeros_like().add_scalar(ext_ex_in);
    let i_input_inh = i_val.clone().zeros_like().add_scalar(ext_in_in);
    let tf_e = zerlaut_tf(&e, &i_val, &e_input_exc.clone(), &i_input_exc.clone(), &w_e, e_l_e,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_e);
    let tf_i = zerlaut_tf(&e, &i_val, &e_input_inh.clone(), &i_input_inh.clone(), &w_i, e_l_i,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_i);
    let df: f32 = 1e-4;
    let df_scale: f32 = 1e3;
    let e_p = e.clone().add_scalar(df);
    let e_m = e.clone().add_scalar(-df);
    let i_p = i_val.clone().add_scalar(df);
    let i_m = i_val.clone().add_scalar(-df);
    let tf_e_p0 = zerlaut_tf(&e_p.clone(), &i_val, &e_input_exc, &i_input_exc, &w_e, e_l_e,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_e);
    let tf_e_n0 = zerlaut_tf(&e_m.clone(), &i_val, &e_input_exc, &i_input_exc, &w_e, e_l_e,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_e);
    let tf_e_0p = zerlaut_tf(&e, &i_p.clone(), &e_input_exc, &i_input_exc, &w_e, e_l_e,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_e);
    let tf_e_0n = zerlaut_tf(&e, &i_m.clone(), &e_input_exc, &i_input_exc, &w_e, e_l_e,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_e);
    let tf_e_pp = zerlaut_tf(&e_p.clone(), &i_p.clone(), &e_input_exc, &i_input_exc, &w_e, e_l_e,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_e);
    let tf_e_pn = zerlaut_tf(&e_p, &i_m.clone(), &e_input_exc, &i_input_exc, &w_e, e_l_e,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_e);
    let tf_e_np = zerlaut_tf(&e_m.clone(), &i_p, &e_input_exc, &i_input_exc, &w_e, e_l_e,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_e);
    let tf_e_nn = zerlaut_tf(&e_m, &i_m, &e_input_exc, &i_input_exc, &w_e, e_l_e,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_e);
    let tf_i_p0 = zerlaut_tf(&e.clone().add_scalar(df), &i_val, &e_input_inh, &i_input_inh, &w_i, e_l_i,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_i);
    let tf_i_n0 = zerlaut_tf(&e.clone().add_scalar(-df), &i_val, &e_input_inh, &i_input_inh, &w_i, e_l_i,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_i);
    let tf_i_0p = zerlaut_tf(&e, &i_val.clone().add_scalar(df), &e_input_inh, &i_input_inh, &w_i, e_l_i,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_i);
    let tf_i_0n = zerlaut_tf(&e, &i_val.clone().add_scalar(-df), &e_input_inh, &i_input_inh, &w_i, e_l_i,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_i);
    let tf_i_pp = zerlaut_tf(&e.clone().add_scalar(df), &i_val.clone().add_scalar(df), &e_input_inh, &i_input_inh, &w_i, e_l_i,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_i);
    let tf_i_pn = zerlaut_tf(&e.clone().add_scalar(df), &i_val.clone().add_scalar(-df), &e_input_inh, &i_input_inh, &w_i, e_l_i,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_i);
    let tf_i_np = zerlaut_tf(&e.clone().add_scalar(-df), &i_val.clone().add_scalar(df), &e_input_inh, &i_input_inh, &w_i, e_l_i,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_i);
    let tf_i_nn = zerlaut_tf(&e.clone().add_scalar(-df), &i_val.clone().add_scalar(-df), &e_input_inh, &i_input_inh, &w_i, e_l_i,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i, &p_i);
    let inv_2df = 1.0 / (2.0 * df * df_scale);
    let inv_df_sq = 1.0 / (df * df_scale * df * df_scale);
    let dfe_tf_e = (tf_e_p0.clone() - tf_e_n0.clone()).mul_scalar(inv_2df);
    let dfi_tf_e = (tf_e_0p.clone() - tf_e_0n.clone()).mul_scalar(inv_2df);
    let dfe_tf_i = (tf_i_p0.clone() - tf_i_n0.clone()).mul_scalar(inv_2df);
    let dfi_tf_i = (tf_i_0p.clone() - tf_i_0n.clone()).mul_scalar(inv_2df);
    let d2fefe_e = (tf_e_p0.clone().sub(tf_e.clone().mul_scalar(2.0)).add(tf_e_n0)).mul_scalar(inv_df_sq);
    let d2fifi_e = (tf_e_0p.clone().sub(tf_e.clone().mul_scalar(2.0)).add(tf_e_0n)).mul_scalar(inv_df_sq);
    let d2fefi_e = ((tf_e_pp.clone() - tf_e_np.clone()).mul_scalar(inv_2df)
        - (tf_e_pn.clone() - tf_e_nn.clone()).mul_scalar(inv_2df)).mul_scalar(inv_2df);
    let d2fife_e = ((tf_e_pp - tf_e_pn).mul_scalar(inv_2df)
        - (tf_e_np - tf_e_nn).mul_scalar(inv_2df)).mul_scalar(inv_2df);
    let d2fefe_i = (tf_i_p0.clone().sub(tf_i.clone().mul_scalar(2.0)).add(tf_i_n0)).mul_scalar(inv_df_sq);
    let d2fifi_i = (tf_i_0p.clone().sub(tf_i.clone().mul_scalar(2.0)).add(tf_i_0n)).mul_scalar(inv_df_sq);
    let d2fefi_i = ((tf_i_pp.clone() - tf_i_np.clone()).mul_scalar(inv_2df)
        - (tf_i_pn.clone() - tf_i_nn.clone()).mul_scalar(inv_2df)).mul_scalar(inv_2df);
    let d2fife_i = ((tf_i_pp - tf_i_pn).mul_scalar(inv_2df)
        - (tf_i_np - tf_i_nn).mul_scalar(inv_2df)).mul_scalar(inv_2df);
    let de = (tf_e.clone() - e.clone()
        + c_ee.clone() * d2fefe_e.mul_scalar(0.5)
        + c_ei.clone() * d2fefi_e.mul_scalar(0.5)
        + c_ei.clone() * d2fife_e.mul_scalar(0.5)
        + c_ii.clone() * d2fifi_e.mul_scalar(0.5))
        .div_scalar(t_p);
    let di = (tf_i.clone() - i_val.clone()
        + c_ee.clone() * d2fefe_i.mul_scalar(0.5)
        + c_ei.clone() * d2fefi_i.mul_scalar(0.5)
        + c_ei.clone() * d2fife_i.mul_scalar(0.5)
        + c_ii.clone() * d2fifi_i.mul_scalar(0.5))
        .div_scalar(t_p);
    let dc_ee = (tf_e.clone() * (Tensor::full_like(&tf_e, 1.0 / t_p) - tf_e.clone()).div_scalar(n_e)
        + (tf_e.clone() - e.clone()) * (tf_e.clone() - e.clone())
        + c_ee.clone() * dfe_tf_e.clone().mul_scalar(2.0)
        + c_ei.clone() * dfi_tf_e.clone().mul_scalar(2.0)
        - c_ee.clone().mul_scalar(2.0))
        .div_scalar(t_p);
    let dc_ei = ((tf_e.clone() - e.clone()) * (tf_i.clone() - i_val.clone())
        + c_ee.clone() * dfe_tf_e.clone()
        + c_ei.clone() * dfe_tf_i.clone()
        + c_ei.clone() * dfi_tf_e
        + c_ii.clone() * dfi_tf_i.clone()
        - c_ei.clone().mul_scalar(2.0))
        .div_scalar(t_p);
    let dc_ii = (tf_i.clone() * (Tensor::full_like(&tf_i, 1.0 / t_p) - tf_i.clone()).div_scalar(n_i)
        + (tf_i.clone() - i_val.clone()) * (tf_i - i_val.clone())
        + c_ii.clone() * dfi_tf_i.mul_scalar(2.0)
        + c_ei.clone() * dfe_tf_i.mul_scalar(2.0)
        - c_ii.clone().mul_scalar(2.0))
        .div_scalar(t_p);
    let (mu_v_e, _, _) = zerlaut_get_fluct_regime_vars(
        &e, &i_val, &e_input_exc, &i_input_exc, &w_e, e_l_e,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i);
    let dw_e = w_e.clone().neg().div_scalar(tau_w_e)
        + e.clone().mul_scalar(b_e)
        + mu_v_e.add_scalar(-e_l_e).mul_scalar(a_e).div_scalar(tau_w_e);
    let (mu_v_i, _, _) = zerlaut_get_fluct_regime_vars(
        &e, &i_val, &e_input_inh, &i_input_inh, &w_i, e_l_i,
        g_l, c_m, q_e, tau_e, e_e, q_i, tau_i, e_i,
        n_tot, p_con_e, p_con_i, g_frac, k_ext_e, k_ext_i);
    let dw_i = w_i.clone().neg().div_scalar(tau_w_i)
        + i_val.clone().mul_scalar(b_i)
        + mu_v_i.add_scalar(-e_l_i).mul_scalar(a_i).div_scalar(tau_w_i);
    let dou = ou.neg().div_scalar(tau_ou);
    Tensor::cat(vec![de, di, dc_ee, dc_ei, dc_ii, dw_e, dw_i, dou], 2)
}

pub fn kionex_dfun_batch<B: Backend>(
    state: Tensor<B, 3>,
    coupling: Tensor<B, 3>,
    params: &[f32],
    _sweep: Option<&Tensor<B, 3>>,
) -> Tensor<B, 3> {
    let e_rev = params[0]; let k_bath = params[1];
    let j = params[2]; let eta = params[3]; let delta = params[4];
    let c_minus = params[5]; let r_minus = params[6];
    let c_plus = params[7]; let r_plus = params[8]; let v_star = params[9];
    let c_m = params[10].max(f32::EPSILON);
    let tau_n = params[11].max(f32::EPSILON);
    let gamma_p = params[12];
    let epsilon = params[13];
    let pi = std::f32::consts::PI;
    let x = state.clone().narrow(2, 0, 1);
    let v = state.clone().narrow(2, 1, 1);
    let n = state.clone().narrow(2, 2, 1);
    let d_ki = state.clone().narrow(2, 3, 1);
    let k_g = state.clone().narrow(2, 4, 1);
    let c_0 = coupling.narrow(2, 0, 1);
    let na_o = 138.0;
    let na_i = 16.0;
    let cl_o = 112.0;
    let cl_i = 5.0;
    let k_i0 = 130.0;
    let k_o0 = 4.80;
    let w_i = 2160.0;
    let w_o = 720.0;
    let rt = 26.64;
    let g_na = 40.0;
    let g_k = 22.0;
    let g_kl = 0.12;
    let g_nal = 0.02;
    let g_cl = 7.5;
    let rho = 250.0;
    let c_mna_v = -24.0;
    let d_cmna = 12.0;
    let c_nk_v = -19.0;
    let d_cnk = 18.0;
    let cnap = 21.0;
    let d_cnap = 2.0;
    let ckp = 5.5;
    let d_ckp = 1.0;
    let m_inf = (v.clone().add_scalar(-c_mna_v)).mul_scalar(1.0 / d_cmna).neg().exp().add_scalar(1.0).recip();
    let n_inf = (v.clone().add_scalar(-c_nk_v)).mul_scalar(1.0 / d_cnk).neg().exp().add_scalar(1.0).recip();
    let c_hn = 0.4_f32;
    let d_chn = -8.0_f32;
    let h_na = n.clone().add_scalar(-c_hn).mul_scalar(d_chn).exp().add_scalar(1.0).recip().neg().add_scalar(1.1);
    let beta_vol = w_i / w_o;
    let dna_i = d_ki.clone().neg();
    let dna_o = dna_i.clone().mul_scalar(-beta_vol);
    let dk_o = d_ki.clone().mul_scalar(-beta_vol);
    let k_i = d_ki.clone().neg().add_scalar(k_i0);
    let na_i_val = dna_i.add_scalar(na_i);
    let na_o_val = dna_o.add_scalar(na_o);
    let k_o_val = dk_o.add_scalar(k_o0).add(k_g.clone());
    let nernst_k = (k_o_val.clone().log().clamp(1e-10, f32::INFINITY) - k_i.clone().log().clamp(1e-10, f32::INFINITY)).mul_scalar(rt);
    let i_k = n.clone().mul_scalar(g_k).add_scalar(g_kl) * (v.clone() - nernst_k);
    let nernst_na = (na_o_val.clone().log().clamp(1e-10, f32::INFINITY) - na_i_val.clone().log().clamp(1e-10, f32::INFINITY)).mul_scalar(rt);
    let i_na = (m_inf * h_na).mul_scalar(g_na).add_scalar(g_nal) * (v.clone() - nernst_na);
    let e_cl = (cl_o as f32 / cl_i as f32).ln();
    let i_cl = v.clone().add_scalar(rt * e_cl).mul_scalar(g_cl);
    let pump_na = (na_i_val.clone().add_scalar(-cnap)).neg().exp().add_scalar(1.0).recip();
    let pump_k = (k_o_val.clone().add_scalar(-ckp)).neg().exp().add_scalar(1.0).recip();
    let i_pump = (pump_na * pump_k).mul_scalar(rho);
    let v_dot = (i_na + i_k.clone() + i_cl + i_pump.clone()).mul_scalar(-1.0 / c_m);
    let r_x = x.clone().mul_scalar(r_minus).div_scalar(pi);
    let v_lower = v.clone().lower_elem(v_star);
    let dx_neg = (v.clone().add_scalar(-c_minus)).mul_scalar(2.0 * r_minus) * x.clone();
    let dx_neg = dx_neg.add_scalar(delta) - r_x.clone() * x.clone().mul_scalar(j);
    let dx_pos = (v.clone().add_scalar(-c_plus)).mul_scalar(2.0 * r_plus) * x.clone();
    let dx_pos = dx_pos.add_scalar(delta) - r_x * x.clone().mul_scalar(j);
    let dx = dx_pos.mask_where(v_lower.clone(), dx_neg);
    let dv_neg = v_dot.clone()
        - x.clone() * x.clone().mul_scalar(r_minus)
        .add_scalar(eta)
        .add(c_0.clone().mul_scalar(r_minus / pi) * v.clone().neg().add_scalar(e_rev));
    let dv_pos = v_dot
        - x.clone() * x.clone().mul_scalar(r_plus)
        .add_scalar(eta)
        .add(c_0.mul_scalar(r_minus / pi) * v.clone().neg().add_scalar(e_rev));
    let dv = dv_pos.mask_where(v_lower, dv_neg);
    let dn = (n_inf - n).div_scalar(tau_n);
    let ddki = (i_k - i_pump.mul_scalar(2.0)).mul_scalar(-(gamma_p / w_i));
    let dkg = k_o_val.neg().add_scalar(k_bath).mul_scalar(epsilon);
    Tensor::cat(vec![dx, dv, dn, ddki, dkg], 2)
}

pub fn model_prefers_heun<B: Backend>(model: &EngineModel<B>) -> bool {
    matches!(model, EngineModel::G2do { .. } | EngineModel::Epileptor { .. } | EngineModel::Epileptor2D { .. } | EngineModel::EpileptorRS { .. } | EngineModel::EpileptorCodim3 { .. } | EngineModel::EpileptorCodim3SlowMod { .. } | EngineModel::KIonEx { .. })
}

pub fn model_param_slice<B: Backend>(model: &EngineModel<B>) -> Vec<f32> {
    match model {
        EngineModel::G2do { params } => params.clone(),
        EngineModel::Mpr { params } => params.clone(),
        EngineModel::Rww { params } => params.clone(),
        EngineModel::Kuramoto { params } => params.clone(),
        EngineModel::JansenRit { params } => params.clone(),
        EngineModel::WilsonCowan { params } => params.clone(),
        EngineModel::Linear { params } => params.clone(),
        EngineModel::SupHopf { params } => params.clone(),
        EngineModel::Hopfield { params } => params.clone(),
        EngineModel::CoombesByrne2D { params } => params.clone(),
        EngineModel::CoombesByrne { params } => params.clone(),
        EngineModel::GastSD { params } => params.clone(),
        EngineModel::GastSF { params } => params.clone(),
        EngineModel::LarterBreakspear { params } => params.clone(),
        EngineModel::Epileptor2D { params } => params.clone(),
        EngineModel::Epileptor { params } => params.clone(),
        EngineModel::RwwExcInh { params } => params.clone(),
        EngineModel::DecoBalancedExcInh { params } => params.clone(),
        EngineModel::EpileptorCodim3 { params } => params.clone(),
        EngineModel::EpileptorCodim3SlowMod { params } => params.clone(),
        EngineModel::EpileptorRS { params } => params.clone(),
        EngineModel::ZetterbergJansen { params } => params.clone(),
        EngineModel::ReducedFHN { params } => params.clone(),
        EngineModel::ReducedHR { params } => params.clone(),
        EngineModel::DumontGutkin { params } => params.clone(),
        EngineModel::ZerlautFirst { params } => params.clone(),
        EngineModel::ZerlautSecond { params } => params.clone(),
        EngineModel::KIonEx { params } => params.clone(),
        _ => unreachable!(),
    }
}
