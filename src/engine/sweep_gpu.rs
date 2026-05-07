//! Batch-dim GPU sweep — runs all sweep points simultaneously on GPU.
//!
//! Key idea: stack all sweep points as a leading batch dimension
//! `[n_sweep, nnodes, nvar]` so each Burn tensor operation processes
//! all 1024 points in parallel. This reduces kernel launches from
//! ~30,000 per point (×1024 points = ~30M total) to just ~30,000 total.
//!
//! For the all-to-all scalar coupling used in our benchmark ring network,
//! coupling reduces to a mean across nodes, which broadcasts trivially.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};

/// Result of a batch sweep on GPU.
#[derive(Debug, Clone)]
pub struct BatchGpuSweepResult {
    /// Parameter values used in the sweep.
    pub param_values: Vec<f32>,
    /// Temporal average G2DO, flat: [n_sweep * nnodes * 2]
    pub tavg_g2do: Vec<f32>,
    /// Temporal average JR, flat: [n_sweep * nnodes * 6]
    pub tavg_jr: Vec<f32>,
    /// Temporal average WC, flat: [n_sweep * nnodes * 2]
    pub tavg_wc: Vec<f32>,
    /// Wall-clock time in ms.
    pub elapsed_ms: f64,
}

/// Run a batch sweep of the 3-subnet ring (G2DO → JR → WC) on GPU.
///
/// All sweep points are processed simultaneously as a batch dimension.
/// G2DO uses Heun integration; JR and WC use Euler (matching Numba benchmark).
///
/// * `i_ext_values` - I_ext parameter for each sweep point
/// * `nnodes` - Number of nodes per subnetwork
/// * `n_steps` - Number of integration steps
/// * `dt` - Time step
/// * `coupling_weight` - Scalar coupling weight
pub fn batch_sweep_3subnet<B: Backend>(
    i_ext_values: &[f32],
    nnodes: usize,
    n_steps: usize,
    dt: f32,
    coupling_weight: f32,
    device: &B::Device,
) -> BatchGpuSweepResult {
    use std::time::Instant;

    let n_sweep = i_ext_values.len();
    let inv_nsteps = 1.0f32 / n_steps as f32;

    let g2do_params = crate::model::g2do::g2do_default_params();
    let jr_params = crate::model::jansen_rit::jansen_rit_default_params();
    let wc_params = crate::model::wilson_cowan::wilson_cowan_default_params();

    // --- Initial conditions (seeded LCG PRNG) ---
    let mut lcg_state: u64 = 42;
    let lcg_next = |state: &mut u64| -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((*state >> 33) as u32).min(0x7FFFFFFE);
        x as f32 / 0x7FFFFFFF as f32
    };
    let ic_g2do: Vec<f32> = (0..n_sweep * 2 * nnodes)
        .map(|_| lcg_next(&mut lcg_state) * 0.2 - 0.1)
        .collect();
    let ic_jr: Vec<f32> = (0..n_sweep * 6 * nnodes)
        .map(|_| lcg_next(&mut lcg_state) * 0.02 - 0.01)
        .collect();
    let ic_wc: Vec<f32> = (0..n_sweep * 2 * nnodes)
        .map(|_| lcg_next(&mut lcg_state) * 0.2 + 0.1)
        .collect();

    // State tensors: [n_sweep, nnodes, nvar]
    let mut state_g2do = Tensor::<B, 3>::from_floats(
        TensorData::new::<f32, Vec<usize>>(ic_g2do, vec![n_sweep, nnodes, 2]),
        device,
    );
    let mut state_jr = Tensor::<B, 3>::from_floats(
        TensorData::new::<f32, Vec<usize>>(ic_jr, vec![n_sweep, nnodes, 6]),
        device,
    );
    let mut state_wc = Tensor::<B, 3>::from_floats(
        TensorData::new::<f32, Vec<usize>>(ic_wc, vec![n_sweep, nnodes, 2]),
        device,
    );

    // I_ext per sweep point as [n_sweep, 1, 1] tensor (broadcasts over nnodes+nvar)
    let i_ext_tensor = Tensor::<B, 3>::from_floats(
        TensorData::new::<f32, Vec<usize>>(i_ext_values.to_vec(), vec![n_sweep, 1, 1]),
        device,
    );

    // Temporal average accumulators
    let mut tavg_g2do = Tensor::<B, 3>::zeros([n_sweep, nnodes, 2], device);
    let mut tavg_jr = Tensor::<B, 3>::zeros([n_sweep, nnodes, 6], device);
    let mut tavg_wc = Tensor::<B, 3>::zeros([n_sweep, nnodes, 2], device);

    // Unpack G2DO params
    let tau_g = g2do_params[0];
    let a_g = g2do_params[2];
    let b_g = g2do_params[3];
    let c_g = g2do_params[4];
    let d_g = g2do_params[5];
    let e_g = g2do_params[6];
    let f_g = g2do_params[7];
    let g_g = g2do_params[8];
    let alpha_g = g2do_params[9];
    let beta_g = g2do_params[10];
    let gamma_g = g2do_params[11];
    let dtau = d_g * tau_g;
    let d_over_tau = d_g / tau_g;

    // Unpack JR params
    let a_p = jr_params[0]; // A=3.25
    let b_p = jr_params[1]; // B=22.0
    let a = jr_params[2];   // a=0.1
    let b = jr_params[3];   // b=0.05
    let v0 = jr_params[4];
    let nu_max = jr_params[5];
    let r = jr_params[6];
    let j = jr_params[7];
    let a_1 = jr_params[8];
    let a_2 = jr_params[9];
    let a_3 = jr_params[10];
    let a_4 = jr_params[11];
    let mu = jr_params[12];
    let a_p_a = a_p * a;
    let b_p_b = b_p * b;
    let two_a = 2.0 * a;
    let two_b = 2.0 * b;
    let a_sq = a * a;
    let b_sq = b * b;
    let two_nu_max = 2.0 * nu_max;

    // Unpack WC params
    let c_ee = wc_params[0];
    let c_ei = wc_params[1];
    let c_ie = wc_params[2];
    let c_ii = wc_params[3];
    let tau_e = wc_params[4];
    let tau_i = wc_params[5];
    let a_e = wc_params[6];
    let b_e = wc_params[7];
    let ce = wc_params[8];
    let theta_e = wc_params[9];
    let a_i = wc_params[10];
    let b_i = wc_params[11];
    let ci = wc_params[12];
    let theta_i = wc_params[13];
    let r_e = wc_params[14];
    let r_i = wc_params[15];
    let k_e = wc_params[16];
    let k_i = wc_params[17];
    let p_val = wc_params[18];
    let q_val = wc_params[19];
    let alpha_e = wc_params[20];
    let alpha_i = wc_params[21];
    let inv_tau_e = 1.0 / tau_e;
    let inv_tau_i = 1.0 / tau_i;
    let sig_e_offset = 1.0 / (1.0 + (a_e * b_e).exp());
    let sig_i_offset = 1.0 / (1.0 + (a_i * b_i).exp());

    let w = coupling_weight;
    let w_inv_n = w / nnodes as f32;

    let start = Instant::now();

    for _t in 0..n_steps {
        // --- Mean-field coupling ---
        let wc_e = state_wc.clone().narrow(2, 0, 1);
        let c_g2do = wc_e.mean_dim(1).mul_scalar(w_inv_n).expand([n_sweep, nnodes, 1]);

        let g2do_w = state_g2do.clone().narrow(2, 1, 1);
        let c_jr = g2do_w.mean_dim(1).mul_scalar(w_inv_n).expand([n_sweep, nnodes, 1]);

        let jr_y0 = state_jr.clone().narrow(2, 0, 1);
        let c_wc = jr_y0.mean_dim(1).mul_scalar(w_inv_n).expand([n_sweep, nnodes, 1]);

        // ============================================================
        // G2DO Heun step
        // ============================================================
        {
            let c0_g = c_g2do.narrow(2, 0, 1);

            // k1
            let v = state_g2do.clone().narrow(2, 0, 1);
            let w_val = state_g2do.clone().narrow(2, 1, 1);
            let v2 = v.clone() * v.clone();
            let v3 = v.clone() * v2.clone();

            let dv1 = (w_val.clone().mul_scalar(alpha_g)
                + (c0_g.clone() + i_ext_tensor.clone()).mul_scalar(gamma_g)
                - v3.clone().mul_scalar(f_g)
                + v2.clone().mul_scalar(e_g)
                + v.clone().mul_scalar(g_g))
                .mul_scalar(dtau);
            let dw1 = (v.clone().mul_scalar(b_g)
                + v2.clone().mul_scalar(c_g)
                - w_val.clone().mul_scalar(beta_g)
                + a_g)
                .mul_scalar(d_over_tau);
            let d1 = Tensor::cat(vec![dv1, dw1], 2);

            let predictor = state_g2do.clone() + d1.clone().mul_scalar(dt);

            // k2
            let v_p = predictor.clone().narrow(2, 0, 1);
            let w_p = predictor.clone().narrow(2, 1, 1);
            let v2_p = v_p.clone() * v_p.clone();
            let v3_p = v_p.clone() * v2_p.clone();

            let dv2 = (w_p.clone().mul_scalar(alpha_g)
                + (c0_g + i_ext_tensor.clone()).mul_scalar(gamma_g)
                - v3_p.clone().mul_scalar(f_g)
                + v2_p.clone().mul_scalar(e_g)
                + v_p.clone().mul_scalar(g_g))
                .mul_scalar(dtau);
            let dw2 = (v_p.clone().mul_scalar(b_g)
                + v2_p.mul_scalar(c_g)
                - w_p.mul_scalar(beta_g)
                + a_g)
                .mul_scalar(d_over_tau);
            let d2 = Tensor::cat(vec![dv2, dw2], 2);

            state_g2do = state_g2do + (d1 + d2).mul_scalar(dt * 0.5);
        }
        tavg_g2do = tavg_g2do + state_g2do.clone();

        // ============================================================
        // JansenRit Euler step
        // ============================================================
        {
            let c0_jr = c_jr.narrow(2, 0, 1);

            let y0 = state_jr.clone().narrow(2, 0, 1);
            let y1 = state_jr.clone().narrow(2, 1, 1);
            let y2 = state_jr.clone().narrow(2, 2, 1);
            let y3 = state_jr.clone().narrow(2, 3, 1);
            let y4 = state_jr.clone().narrow(2, 4, 1);
            let y5 = state_jr.clone().narrow(2, 5, 1);

            let ones = Tensor::<B, 3>::ones(y0.shape(), device);

            let sigm_y1_y2 = {
                let arg = (y1.clone() - y2.clone()).neg().add_scalar(v0).mul_scalar(r);
                let denom = arg.exp().add_scalar(1.0);
                ones.clone().mul_scalar(two_nu_max) / denom
            };
            let sigm_y0_1 = {
                let arg = y0.clone().mul_scalar(a_1 * j).neg().add_scalar(v0).mul_scalar(r);
                let denom = arg.exp().add_scalar(1.0);
                ones.clone().mul_scalar(two_nu_max) / denom
            };
            let sigm_y0_3 = {
                let arg = y0.clone().mul_scalar(a_3 * j).neg().add_scalar(v0).mul_scalar(r);
                let denom = arg.exp().add_scalar(1.0);
                ones.clone().mul_scalar(two_nu_max) / denom
            };

            let dy3 = sigm_y1_y2.mul_scalar(a_p_a)
                - y3.clone().mul_scalar(two_a)
                - y0.clone().mul_scalar(a_sq);
            let dy4 = (sigm_y0_1.mul_scalar(a_2 * j).add_scalar(mu) + c0_jr)
                .mul_scalar(a_p_a)
                - y4.clone().mul_scalar(two_a)
                - y1.clone().mul_scalar(a_sq);
            let dy5 = sigm_y0_3
                .mul_scalar(a_4 * j)
                .mul_scalar(b_p_b)
                - y5.clone().mul_scalar(two_b)
                - y2.clone().mul_scalar(b_sq);

            let d_jr = Tensor::cat(vec![y3, y4, y5, dy3, dy4, dy5], 2);
            state_jr = state_jr + d_jr.mul_scalar(dt);
        }
        tavg_jr = tavg_jr + state_jr.clone();

        // ============================================================
        // WilsonCowan Euler step (channel-mask approach for 2-var model)
        // ============================================================
        {
            let c0_wc = c_wc.narrow(2, 0, 1);

            let e = state_wc.clone().narrow(2, 0, 1);
            let i_val = state_wc.clone().narrow(2, 1, 1);

            let x_e = (e.clone().mul_scalar(c_ee)
                - i_val.clone().mul_scalar(c_ei)
                + p_val - theta_e + c0_wc.clone())
                .mul_scalar(alpha_e);
            let x_i = (e.clone().mul_scalar(c_ie)
                - i_val.clone().mul_scalar(c_ii)
                + q_val - theta_i)
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

            let de = (e.clone().neg()
                + e.clone().mul_scalar(-r_e).add_scalar(k_e) * s_e)
                .mul_scalar(inv_tau_e);
            let di = (i_val.clone().neg()
                + i_val.mul_scalar(-r_i).add_scalar(k_i) * s_i)
                .mul_scalar(inv_tau_i);

            let d_wc = Tensor::cat(vec![de, di], 2);
            state_wc = state_wc + d_wc.mul_scalar(dt);

            // Clamp E and I to [0, 1]
            let e_clamped = state_wc.clone().narrow(2, 0, 1).clamp(0.0, 1.0);
            let i_clamped = state_wc.clone().narrow(2, 1, 1).clamp(0.0, 1.0);
            state_wc = Tensor::cat(vec![e_clamped, i_clamped], 2);
        }
        tavg_wc = tavg_wc + state_wc.clone();
    }

    // Average tavg over steps
    tavg_g2do = tavg_g2do.mul_scalar(inv_nsteps);
    tavg_jr = tavg_jr.mul_scalar(inv_nsteps);
    tavg_wc = tavg_wc.mul_scalar(inv_nsteps);

    let elapsed = start.elapsed();

    let tavg_g2do_flat = crate::io::tensor_to_flat_f32::<B, 3>(tavg_g2do).0;
    let tavg_jr_flat = crate::io::tensor_to_flat_f32::<B, 3>(tavg_jr).0;
    let tavg_wc_flat = crate::io::tensor_to_flat_f32::<B, 3>(tavg_wc).0;

    BatchGpuSweepResult {
        param_values: i_ext_values.to_vec(),
        tavg_g2do: tavg_g2do_flat,
        tavg_jr: tavg_jr_flat,
        tavg_wc: tavg_wc_flat,
        elapsed_ms: elapsed.as_millis() as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_batch_sweep_3subnet_ndarray() {
        let device: <B as Backend>::Device = Default::default();
        let i_ext_values: Vec<f32> = vec![-0.5, 0.0, 0.5];
        let result = batch_sweep_3subnet::<B>(&i_ext_values, 4, 100, 0.1, 0.01, &device);
        assert_eq!(result.param_values.len(), 3);
        assert!(result.tavg_g2do.iter().all(|x| x.is_finite()));
        assert!(result.tavg_jr.iter().all(|x| x.is_finite()));
        assert!(result.tavg_wc.iter().all(|x| x.is_finite()));
    }
}