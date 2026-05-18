//! Coupling functions with pre/post split.
//!
//! The coupling pipeline:
//!
//! ```text
//! result = post_with_target( W @ pre(x_j), x_i )
//! ```
//!
//! - `pre()` is applied **per source node**, before the weighted sum.
//! - `W @ pre(x_j)` is the weighted sum over source nodes.
//! - `post_with_target()` is applied after the weighted sum, with optional
//!   access to the target node state `x_i`.
//!
//! For most coupling functions, `x_i` is not needed and `post_with_target()`
//! delegates to `post()`. For **Kuramoto** and **Difference** (non-square),
//! `x_i` is required to compute the classic TVB formulation.
//!
//! ## Classification
//!
//! | Function            | pre(x_j)                              | post / post_with_target          | needs_x_i |
//! |--------------------|---------------------------------------|----------------------------------|-----------|
//! | Linear             | identity                              | a*x + b                          | no        |
//! | Sigmoidal          | identity                              | cmin+(cmax-cmin)/σ               | no        |
//! | ScaledLinear       | identity                              | a*(x - b)                        | no        |
//! | Kuramoto           | [sin(x), cos(x)] (2ch)               | a/N*(cos(x_i)*Σsin - sin(x_i)*Σcos) | yes   |
//! | HyperbolicTangent  | a*(1+tanh((b*x-midpoint)/sigma))     | identity                         | no        |
//! | SigmoidalJansenRit (classic) | cmin+(cmax-cmin)/(1+exp(r*(mid-x))) | a*gx                             | no        |
//! | SigmoidalJansenRit (legacy) | a*(2*e0)/(1+exp(r*(v0-x)))         | identity                         | no        |
//! | PreSigmoidal (static) | h*(q+tanh(g*(p*x-θ)))             | identity                         | no        |
//! | PreSigmoidal (dynamic) | h*(q+tanh(g*(P*x0-x1)))          | identity                         | no        |
//! | Difference         | identity                              | a*(gx - x_i*rowsums)            | yes*      |
//!
//! *Difference with square weight matrices is converted to Linear at construction
//! time (weight preprocessing), so needs_x_i() is false after construction.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use serde::{Deserialize, Serialize};

fn default_sigma() -> f32 { 1.0 }
fn default_cmin() -> f32 { -1.0 }
fn default_sigmoidal_a() -> f32 { 1.0 }
fn default_n_src() -> usize { 1 }
fn default_sjr_r() -> f32 { 0.56 }
fn default_sjr_v0() -> f32 { 6.0 }
fn default_sjr_e0() -> f32 { 0.005 }
fn default_sjr_cmin() -> f32 { 0.0 }
fn default_sjr_cmax() -> f32 { 0.005 }
fn default_sjr_midpoint() -> f32 { 6.0 }
fn default_use_classic() -> bool { false }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CouplingFnConfig {
    Linear { a: f32, b: f32 },
    Sigmoidal { #[serde(default = "default_cmin")] cmin: f32, cmax: f32, midpoint: f32, #[serde(default = "default_sigmoidal_a")] a: f32, #[serde(default = "default_sigma")] sigma: f32 },
    Difference { a: f32, #[serde(skip, default)] rowsums: Option<Vec<f32>> },
    Kuramoto { a: f32, #[serde(default = "default_n_src")] n_src: usize },
    ScaledLinear { a: f32, b: f32 },
    HyperbolicTangent { a: f32, b: f32, #[serde(default)] midpoint: f32, #[serde(default = "default_sigma")] sigma: f32 },
    SigmoidalJansenRit {
        a: f32,
        #[serde(default = "default_use_classic")] use_classic: bool,
        #[serde(default = "default_sjr_cmin")] cmin: f32,
        #[serde(default = "default_sjr_cmax")] cmax: f32,
        #[serde(default = "default_sjr_r")] r: f32,
        #[serde(default = "default_sjr_midpoint")] midpoint: f32,
        #[serde(default = "default_sjr_e0")] e0: f32,
        #[serde(default = "default_sjr_v0")] v0: f32,
    },
    PreSigmoidal { h: f32, q: f32, g: f32, p: f32, theta: f32, #[serde(default)] dynamic: bool, #[serde(default)] global_t: bool },
}

impl CouplingFnConfig {
    pub fn min_src_ncvar(&self) -> usize {
        match self {
            CouplingFnConfig::SigmoidalJansenRit { use_classic: true, .. } => 2,
            CouplingFnConfig::PreSigmoidal { dynamic: true, .. } => 2,
            _ => 1,
        }
    }

    #[inline]
    pub fn needs_two_src_cvar(&self) -> bool {
        matches!(self, CouplingFnConfig::SigmoidalJansenRit { use_classic: true, .. } | CouplingFnConfig::PreSigmoidal { dynamic: true, .. })
    }

    #[inline]
    pub fn needs_x_i(&self) -> bool {
        matches!(self, CouplingFnConfig::Kuramoto { .. } | CouplingFnConfig::Difference { rowsums: Some(_), .. })
    }

    #[inline]
    pub fn pre_channels(&self) -> usize {
        match self {
            CouplingFnConfig::Kuramoto { .. } => 2,
            CouplingFnConfig::SigmoidalJansenRit { use_classic: true, .. } => 1,
            CouplingFnConfig::PreSigmoidal { dynamic: true, .. } => 1,
            _ => 1,
        }
    }

    pub fn set_kuramoto_nsrc(&mut self, n: usize) {
        if let CouplingFnConfig::Kuramoto { n_src, .. } = self {
            *n_src = n;
        }
    }

    pub fn set_difference_rowsums(&mut self, rs: Vec<f32>) {
        if let CouplingFnConfig::Difference { rowsums, .. } = self {
            *rowsums = Some(rs);
        }
    }

    pub fn from_name_and_params(name: &str, params: &[f32]) -> Option<Self> {
        match name {
            "Linear" => {
                let a = params.first().copied().unwrap_or(1.0);
                let b = params.get(1).copied().unwrap_or(0.0);
                Some(CouplingFnConfig::Linear { a, b })
            }
            "Sigmoidal" => {
                if params.len() >= 5 {
                    let cmin = params[0];
                    let cmax = params[1];
                    let midpoint = params[2];
                    let a = params[3];
                    let sigma = params[4];
                    Some(CouplingFnConfig::Sigmoidal { cmin, cmax, midpoint, a, sigma })
                } else {
                    let cmax = params.first().copied().unwrap_or(1.0);
                    let midpoint = params.get(1).copied().unwrap_or(0.0);
                    let sigma = params.get(2).copied().unwrap_or(1.0);
                    Some(CouplingFnConfig::Sigmoidal { cmin: -1.0, cmax, midpoint, a: 1.0, sigma })
                }
            }
            "Difference" => {
                let a = params.first().copied().unwrap_or(1.0);
                Some(CouplingFnConfig::Difference { a, rowsums: None })
            }
            "Kuramoto" => {
                let a = params.first().copied().unwrap_or(1.0);
                Some(CouplingFnConfig::Kuramoto { a, n_src: 1 })
            }
            "ScaledLinear" => {
                let a = params.first().copied().unwrap_or(1.0);
                let b = params.get(1).copied().unwrap_or(0.0);
                Some(CouplingFnConfig::ScaledLinear { a, b })
            }
            "HyperbolicTangent" => {
                let a = params.first().copied().unwrap_or(1.0);
                let b = params.get(1).copied().unwrap_or(1.0);
                let midpoint = params.get(2).copied().unwrap_or(0.0);
                let sigma = params.get(3).copied().unwrap_or(1.0);
                Some(CouplingFnConfig::HyperbolicTangent { a, b, midpoint, sigma })
            }
            "SigmoidalJansenRit" => {
                if params.len() >= 5 {
                    let a = params[0];
                    let cmin = params[1];
                    let cmax = params[2];
                    let r = params[3];
                    let midpoint = params[4];
                    Some(CouplingFnConfig::SigmoidalJansenRit {
                        a,
                        use_classic: true,
                        cmin,
                        cmax,
                        r,
                        midpoint,
                        e0: 0.005,
                        v0: 6.0,
                    })
                } else {
                    let a = params.first().copied().unwrap_or(1.0);
                    let e0 = params.get(1).copied().unwrap_or(0.005);
                    let r = params.get(2).copied().unwrap_or(0.56);
                    let v0 = params.get(3).copied().unwrap_or(6.0);
                    Some(CouplingFnConfig::SigmoidalJansenRit {
                        a,
                        use_classic: false,
                        cmin: 0.0,
                        cmax: 0.005,
                        r,
                        midpoint: 6.0,
                        e0,
                        v0,
                    })
                }
            }
            "PreSigmoidal" => {
                let h = params.first().copied().unwrap_or(1.0);
                let q = params.get(1).copied().unwrap_or(1.0);
                let g = params.get(2).copied().unwrap_or(1.0);
                let p = params.get(3).copied().unwrap_or(1.0);
                let theta = params.get(4).copied().unwrap_or(0.5);
                let dynamic = params.get(5).copied().unwrap_or(0.0) != 0.0;
                let global_t = params.get(6).copied().unwrap_or(0.0) != 0.0;
                Some(CouplingFnConfig::PreSigmoidal { h, q, g, p, theta, dynamic, global_t })
            }
            _ => None,
        }
    }

    pub fn pre<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            CouplingFnConfig::Linear { .. } => x,
            CouplingFnConfig::Sigmoidal { .. } => x,
            CouplingFnConfig::ScaledLinear { .. } => x,
            CouplingFnConfig::Difference { .. } => x,

            CouplingFnConfig::Kuramoto { .. } => {
                let sin_x = x.clone().sin();
                let cos_x = x.cos();
                Tensor::cat(vec![sin_x, cos_x], 1)
            }
            CouplingFnConfig::HyperbolicTangent { a, b, midpoint, sigma } => {
                let inner = x.mul_scalar(*b).add_scalar(-*midpoint).div_scalar(*sigma);
                inner.tanh().add_scalar(1.0).mul_scalar(*a)
            }
            CouplingFnConfig::SigmoidalJansenRit { a, use_classic, cmin, cmax, r, midpoint, e0, v0 } => {
                let dims = x.shape().dims;
                let x0 = x.clone().narrow(1, 0, 1);
                let x1 = if dims[1] >= 2 {
                    x.clone().narrow(1, 1, 1)
                } else {
                    Tensor::zeros([dims[0], 1], &x.device())
                };
                let diff = x0 - x1;
                if *use_classic {
                    let shifted = diff.add_scalar(-*midpoint).mul_scalar(-*r);
                    let denom = shifted.exp().add_scalar(1.0);
                    denom.recip().mul_scalar(cmax - cmin).add_scalar(*cmin)
                } else {
                    let shifted = diff.add_scalar(-*v0).mul_scalar(-*r);
                    let denom = shifted.exp().add_scalar(1.0);
                    denom.recip().mul_scalar(*a * 2.0 * *e0)
                }
            }
            CouplingFnConfig::PreSigmoidal { h, q, g, p, theta, dynamic, global_t } => {
                if *dynamic {
                    let dims = x.shape().dims;
                    let x0 = x.clone().narrow(1, 0, 1);
                    let x1 = x.narrow(1, 1, 1);
                    let threshold = if *global_t {
                        let mean = x1.clone().mean().reshape([1, 1]);
                        mean.expand([dims[0], 1])
                    } else {
                        x1
                    };
                    let inner = x0.mul_scalar(*p) - threshold;
                    inner.mul_scalar(*g).tanh().add_scalar(*q).mul_scalar(*h)
                } else {
                    let inner = x.mul_scalar(*p).add_scalar(-*theta);
                    inner.mul_scalar(*g).tanh().add_scalar(*q).mul_scalar(*h)
                }
            }
        }
    }

    /// CPU-only `pre()` for a single row, avoiding GPU tensor overhead.
    ///
    /// Used in the per-edge delay coupling path where downloading to CPU once
    /// and computing `pre()` on CPU-side data is far cheaper than dispatching
    /// thousands of tiny GPU kernels and synchronizing per edge.
    ///
    /// `row` is a flat slice of length `ncvar`; returns a `Vec<f32>` of length
    /// `ncvar * pre_channels()` (1× for most coupling types, 2× for Kuramoto).
    pub fn pre_cpu(&self, row: &[f32]) -> Vec<f32> {
        match self {
            CouplingFnConfig::Linear { .. }
            | CouplingFnConfig::Sigmoidal { .. }
            | CouplingFnConfig::ScaledLinear { .. }
            | CouplingFnConfig::Difference { .. } => row.to_vec(),

            CouplingFnConfig::Kuramoto { .. } => {
                let mut out = Vec::with_capacity(row.len() * 2);
                for &v in row {
                    out.push(v.sin());
                    out.push(v.cos());
                }
                out
            }
            CouplingFnConfig::HyperbolicTangent { a, b, midpoint, sigma } => {
                row.iter()
                    .map(|&v| {
                        let inner = (v * b - midpoint) / sigma;
                        a * (1.0 + inner.tanh())
                    })
                    .collect()
            }
            CouplingFnConfig::SigmoidalJansenRit { a, use_classic, cmin, cmax, r, midpoint, e0, v0 } => {
                let x0 = row.first().copied().unwrap_or(0.0);
                let x1 = if row.len() >= 2 { row[1] } else { 0.0 };
                let diff = x0 - x1;
                if *use_classic {
                    let shifted = (diff - midpoint) * (-r);
                    let denom = shifted.exp() + 1.0;
                    vec![cmin + (cmax - cmin) / denom]
                } else {
                    let shifted = (diff - v0) * (-r);
                    let denom = shifted.exp() + 1.0;
                    vec![a * 2.0 * e0 / denom]
                }
            }
            CouplingFnConfig::PreSigmoidal { h, q, g, p, theta, dynamic, global_t: _ } => {
                if *dynamic {
                    let x0 = row[0];
                    let x1 = if row.len() >= 2 { row[1] } else { 0.0 };
                    let inner = p * x0 - x1;
                    vec![h * (q + (g * inner).tanh())]
                } else {
                    row.iter()
                        .map(|&v| {
                            let inner = p * v - theta;
                            h * (q + (g * inner).tanh())
                        })
                        .collect()
                }
            }
        }
    }

    pub fn post<B: Backend>(&self, gx: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            CouplingFnConfig::Linear { a, b } => gx.mul_scalar(*a).add_scalar(*b),
            CouplingFnConfig::Sigmoidal { cmin, cmax, midpoint, a, sigma } => {
                let shifted = gx.add_scalar(-*midpoint).div_scalar(*sigma);
                let exponent = shifted.mul_scalar(*a).neg();
                let denom = exponent.exp().add_scalar(1.0);
                denom.recip().mul_scalar(cmax - cmin).add_scalar(*cmin)
            }
            CouplingFnConfig::ScaledLinear { a, b } => gx.add_scalar(-*b).mul_scalar(*a),
            CouplingFnConfig::Difference { a, rowsums: None } => gx.mul_scalar(*a),
            CouplingFnConfig::Difference { a, rowsums: Some(_) } => gx.mul_scalar(*a),
            CouplingFnConfig::Kuramoto { a, n_src } => {
                let ncvar = gx.shape().dims[1] / 2;
                let sin_sum = gx.clone().narrow(1, 0, ncvar);
                sin_sum.mul_scalar(*a / *n_src as f32)
            }
            CouplingFnConfig::HyperbolicTangent { .. } => gx,
            CouplingFnConfig::SigmoidalJansenRit { use_classic: true, a, .. } => gx.mul_scalar(*a),
            CouplingFnConfig::SigmoidalJansenRit { use_classic: false, .. } => gx,
            CouplingFnConfig::PreSigmoidal { .. } => gx,
        }
    }

    /// Apply post-synaptic coupling with optional access to target node state `x_i`.
    ///
    /// For Kuramoto: computes `a/N * (cos(x_i) * Σw·sin(x_j) - sin(x_i) * Σw·cos(x_j))`.
    /// For Difference (non-square, rowsums present): computes `a * (gx - x_i * rowsums)`.
    /// For all others: delegates to `post(gx)`.
    pub fn post_with_target<B: Backend>(
        &self,
        gx: Tensor<B, 2>,
        x_i: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        self.post_with_target_cached(gx, x_i, None)
    }

    pub fn post_with_target_cached<B: Backend>(
        &self,
        gx: Tensor<B, 2>,
        x_i: Option<Tensor<B, 2>>,
        cached_rowsums: Option<&Tensor<B, 2>>,
    ) -> Tensor<B, 2> {
        match self {
            CouplingFnConfig::Kuramoto { a, n_src } => {
                let xi = x_i.expect("Kuramoto coupling requires target state (x_i)");
                let ncvar = xi.shape().dims[1];
                let sin_sum = gx.clone().narrow(1, 0, ncvar);
                let cos_sum = gx.narrow(1, ncvar, ncvar);
                let result = xi.clone().cos().mul(sin_sum) - xi.sin().mul(cos_sum);
                result.mul_scalar(*a / *n_src as f32)
            }
            CouplingFnConfig::Difference { a, rowsums: Some(_) } => {
                if let Some(rowsums_t) = cached_rowsums {
                    let xi = x_i.expect("Difference coupling (non-square) requires target state (x_i)");
                    let result = gx - xi.mul(rowsums_t.clone());
                    result.mul_scalar(*a)
                } else {
                    let xi = x_i.expect("Difference coupling (non-square) requires target state (x_i)");
                    let ntgt = gx.shape().dims[0];
                    let ncvar = gx.shape().dims[1];
                    let dev = gx.device();
                    let rowsums = match self {
                        CouplingFnConfig::Difference { rowsums: Some(rs), .. } => rs,
                        _ => unreachable!(),
                    };
                    let rowsums_t = Tensor::<B, 1>::from_floats(
                        TensorData::new(rowsums.clone(), vec![ntgt]),
                        &dev,
                    ).unsqueeze_dim::<2>(1).expand([ntgt, ncvar]);
                    let result = gx - xi.mul(rowsums_t);
                    result.mul_scalar(*a)
                }
            }
            _ => self.post(gx),
        }
    }

    pub fn pre_3d<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = x.shape().dims;
        let flat = x.reshape([dims[0] * dims[1], dims[2]]);
        let applied = self.pre(flat);
        let out_w = applied.shape().dims[1];
        applied.reshape([dims[0], dims[1], out_w])
    }

    pub fn post_3d<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = x.shape().dims;
        let flat = x.reshape([dims[0] * dims[1], dims[2]]);
        let applied = self.post(flat);
        let out_w = applied.shape().dims[1];
        applied.reshape([dims[0], dims[1], out_w])
    }

    pub fn post_with_target_3d<B: Backend>(
        &self,
        gx: Tensor<B, 3>,
        x_i: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        self.post_with_target_3d_cached(gx, x_i, None)
    }

    pub fn post_with_target_3d_cached<B: Backend>(
        &self,
        gx: Tensor<B, 3>,
        x_i: Option<Tensor<B, 3>>,
        cached_rowsums_3d: Option<&Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let dims = gx.shape().dims;
        let flat_gx = gx.reshape([dims[0] * dims[1], dims[2]]);
        let flat_xi = x_i.map(|xi| {
            let xi_dims = xi.shape().dims;
            xi.reshape([xi_dims[0] * xi_dims[1], xi_dims[2]])
        });
        let cached_rowsums_2d = cached_rowsums_3d.map(|t| {
            let t_dims = t.shape().dims;
            t.clone().reshape([t_dims[0] * t_dims[1], t_dims[2]])
        });
        let applied = self.post_with_target_cached(flat_gx, flat_xi, cached_rowsums_2d.as_ref());
        let out_w = applied.shape().dims[1];
        applied.reshape([dims[0], dims[1], out_w])
    }
}

/// Dense coupling kernel.
///
/// Pipeline: `post_with_target(W @ pre(x_j), x_i)`
///
/// - `weights` has shape `[ntgt, nsrc]`.
/// - `delayed_state` has shape `[nsrc, ncvar]`.
/// - `x_i` is `Some([ntgt, ncvar])` for coupling functions that need target state.
///
/// Returns coupling of shape `[ntgt, ncvar]`.
pub fn dense_coupling<B: Backend>(
    weights: Tensor<B, 2>,
    delayed_state: Tensor<B, 2>,
    coupling_cfg: &CouplingFnConfig,
    x_i: Option<Tensor<B, 2>>,
) -> Tensor<B, 2> {
    dense_coupling_cached(weights, delayed_state, coupling_cfg, x_i, None)
}

pub fn dense_coupling_cached<B: Backend>(
    weights: Tensor<B, 2>,
    delayed_state: Tensor<B, 2>,
    coupling_cfg: &CouplingFnConfig,
    x_i: Option<Tensor<B, 2>>,
    cached_rowsums: Option<&Tensor<B, 2>>,
) -> Tensor<B, 2> {
    debug_assert!(
        weights.shape().dims[1] == delayed_state.shape().dims[0],
        "dense_coupling: weights cols ({}) must match delayed_state rows ({})",
        weights.shape().dims[1], delayed_state.shape().dims[0]
    );
    let pre = coupling_cfg.pre(delayed_state);
    let weighted_sum = weights.matmul(pre);
    coupling_cfg.post_with_target_cached(weighted_sum, x_i, cached_rowsums)
}

/// Read a delayed state slice from a 4-D history buffer.
///
/// `history` has shape `[nvar, nnodes, nmodes, horizon]`.
///
/// Returns tensor of shape `[nvar, nnodes, nmodes]`.
pub fn read_delayed_state<B: Backend>(
    history: &Tensor<B, 4>,
    delay_idx: usize,
) -> Tensor<B, 3> {
    history
        .clone()
        .narrow(3, delay_idx, 1)
        .squeeze::<3>(3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::Backend;
    use burn::backend::ndarray::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    #[test]
    fn test_linear_post() {
        let cfg = CouplingFnConfig::Linear { a: 2.0, b: 0.0 };
        let dev = Default::default();
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]), &dev,
        );
        let pre_result = cfg.pre(x.clone());
        let (data, _) = crate::io::tensor_to_flat_f32(pre_result);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);

        let post_result = cfg.post(x);
        let (data, _) = crate::io::tensor_to_flat_f32(post_result);
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_linear_with_offset() {
        let cfg = CouplingFnConfig::Linear { a: 2.0, b: 1.0 };
        let dev = Default::default();
        let gx = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0], vec![1, 1]), &dev,
        );
        let result = cfg.post(gx);
        let (data, _) = crate::io::tensor_to_flat_f32(result);
        assert!((data[0] - 3.0).abs() < 1e-6, "expected 3.0, got {}", data[0]);
    }

    #[test]
    fn test_sigmoidal_5param_post() {
        let cfg = CouplingFnConfig::Sigmoidal { cmin: -1.0, cmax: 1.0, midpoint: 0.0, a: 1.0, sigma: 1.0 };
        let dev = Default::default();
        let gx = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![0.0], vec![1, 1]), &dev,
        );
        let result = cfg.post(gx);
        let (data, _) = crate::io::tensor_to_flat_f32(result);
        assert!((data[0] - 0.0).abs() < 1e-6, "expected 0.0, got {}", data[0]);
    }

    #[test]
    fn test_sigmoidal_old_3param_compat() {
        let cfg = CouplingFnConfig::from_name_and_params("Sigmoidal", &[1.0, 0.0, 2.0]).unwrap();
        match cfg {
            CouplingFnConfig::Sigmoidal { cmin, cmax, midpoint, a, sigma } => {
                assert_eq!(cmin, -1.0);
                assert_eq!(cmax, 1.0);
                assert_eq!(midpoint, 0.0);
                assert_eq!(a, 1.0);
                assert_eq!(sigma, 2.0);
            }
            _ => panic!("expected Sigmoidal"),
        }
    }

    #[test]
    fn test_sigmoidal_new_5param() {
        let cfg = CouplingFnConfig::from_name_and_params("Sigmoidal", &[-1.0, 1.0, 0.0, 1.0, 230.0]).unwrap();
        match cfg {
            CouplingFnConfig::Sigmoidal { cmin, cmax, midpoint, a, sigma } => {
                assert_eq!(cmin, -1.0);
                assert_eq!(cmax, 1.0);
                assert_eq!(midpoint, 0.0);
                assert_eq!(a, 1.0);
                assert_eq!(sigma, 230.0);
            }
            _ => panic!("expected Sigmoidal"),
        }
    }

    #[test]
    fn test_tanh_pre() {
        let cfg = CouplingFnConfig::HyperbolicTangent { a: 1.0, b: 1.0, midpoint: 0.0, sigma: 1.0 };
        let dev = Default::default();
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![0.0], vec![1, 1]), &dev,
        );
        let result = cfg.pre(x);
        let (data, _) = crate::io::tensor_to_flat_f32(result);
        assert!((data[0] - 1.0).abs() < 1e-6, "expected 1.0, got {}", data[0]);
    }

    #[test]
    fn test_kuramoto_pre_two_channel() {
        let cfg = CouplingFnConfig::Kuramoto { a: 1.0, n_src: 1 };
        let dev = Default::default();
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.5708], vec![1, 1]), &dev,
        );
        let pre_result = cfg.pre(x);
        let (data, shape) = crate::io::tensor_to_flat_f32(pre_result);
        assert_eq!(shape, vec![1, 2], "Kuramoto pre should produce 2 channels");
        assert!((data[0] - 1.0f32).abs() < 1e-4, "sin(pi/2) ≈ 1.0, got {}", data[0]);
        assert!((data[1]).abs() < 1e-4, "cos(pi/2) ≈ 0.0, got {}", data[1]);
    }

    #[test]
    fn test_kuramoto_post_with_target() {
        let cfg = CouplingFnConfig::Kuramoto { a: 1.0, n_src: 2 };
        let dev = Default::default();
        // gx from W @ pre: 2 channels [W@sin, W@cos]
        let gx = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![0.5, 0.3], vec![1, 2]), &dev,
        );
        // x_i = target state (theta_i)
        let x_i = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![0.0], vec![1, 1]), &dev,
        );
        let result = cfg.post_with_target(gx, Some(x_i));
        let (data, shape) = crate::io::tensor_to_flat_f32(result);
        assert_eq!(shape, vec![1, 1]);
        // a/N * (cos(0)*0.5 - sin(0)*0.3) = 1/2 * (1.0*0.5 - 0.0*0.3) = 0.25
        assert!((data[0] - 0.25).abs() < 1e-5, "expected 0.25, got {}", data[0]);
    }

    #[test]
    fn test_kuramoto_post_with_target_identity() {
        // When x_i = x_j, sin(x_j - x_i) = 0 for all edges → coupling should be 0
        let cfg = CouplingFnConfig::Kuramoto { a: 1.0, n_src: 2 };
        let dev = Default::default();
        let theta = 1.23f32;
        // If all source nodes have phase theta, then:
        // W@sin = rowsum * sin(theta), W@cos = rowsum * cos(theta)
        let rowsum = 0.5f32;
        let gx = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![rowsum * theta.sin(), rowsum * theta.cos()],
                vec![1, 2],
            ), &dev,
        );
        let x_i = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![theta], vec![1, 1]), &dev,
        );
        let result = cfg.post_with_target(gx, Some(x_i));
        let (data, _) = crate::io::tensor_to_flat_f32(result);
        // cos(theta)*rowsum*sin(theta) - sin(theta)*rowsum*cos(theta) = 0
        assert!(data[0].abs() < 1e-5, "self-coupling should be ~0, got {}", data[0]);
    }

    #[test]
    fn test_difference_pre_identity() {
        let cfg = CouplingFnConfig::Difference { a: 0.1, rowsums: None };
        let dev = Default::default();
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0, 2.0], vec![1, 2]), &dev,
        );
        let pre_result = cfg.pre(x.clone());
        let (data, _) = crate::io::tensor_to_flat_f32(pre_result);
        assert_eq!(data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_difference_post_with_target() {
        let cfg = CouplingFnConfig::Difference { a: 0.1, rowsums: Some(vec![3.0]) };
        let dev = Default::default();
        let gx = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![5.0], vec![1, 1]), &dev,
        );
        let x_i = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![2.0], vec![1, 1]), &dev,
        );
        let result = cfg.post_with_target(gx, Some(x_i));
        let (data, _) = crate::io::tensor_to_flat_f32(result);
        // a * (gx - x_i * rowsum) = 0.1 * (5.0 - 2.0*3.0) = 0.1 * (-1.0) = -0.1
        assert!((data[0] - (-0.1)).abs() < 1e-6, "expected -0.1, got {}", data[0]);
    }

    #[test]
    fn test_dense_coupling_kuramoto_classic() {
        let dev = Default::default();
        // 2x2 weight matrix, uniform 0.5
        let weights = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![0.0, 0.5, 0.5, 0.0], vec![2, 2]), &dev,
        );
        // source states: theta_0 = 0, theta_1 = pi/2
        let x_j = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![0.0, std::f32::consts::FRAC_PI_2], vec![2, 1]), &dev,
        );
        // target states (same nodes, current state): theta_0 = 0, theta_1 = pi/2
        let x_i = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![0.0, std::f32::consts::FRAC_PI_2], vec![2, 1]), &dev,
        );
        let cfg = CouplingFnConfig::Kuramoto { a: 1.0, n_src: 2 };
        let result = dense_coupling(weights, x_j, &cfg, Some(x_i));
        let (data, shape) = crate::io::tensor_to_flat_f32(result);
        assert_eq!(shape, vec![2, 1]);
        // Target 0: x_i=0, W@sin=[0.5*sin(pi/2)]=[0.5], W@cos=[0.5*cos(pi/2)]=[0.0]
        //   a/N * (cos(0)*0.5 - sin(0)*0.0) = 0.5/2 = 0.25
        assert!((data[0] - 0.25).abs() < 1e-5, "target 0: expected 0.25, got {}", data[0]);
        // Target 1: x_i=pi/2, W@sin=[0.5*sin(0)]=[0.0], W@cos=[0.5*cos(0)]=[0.5]
        //   a/N * (cos(pi/2)*0.0 - sin(pi/2)*0.5) = -0.5/2 = -0.25
        assert!((data[1] - (-0.25)).abs() < 1e-5, "target 1: expected -0.25, got {}", data[1]);
    }

    #[test]
    fn test_dense_coupling_linear_pipeline() {
        let dev = Default::default();
        let weights = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                vec![2, 3],
            ), &dev,
        );
        let delayed_state = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![3, 2],
            ), &dev,
        );

        let cfg = CouplingFnConfig::Linear { a: 2.0, b: 1.0 };
        let result = dense_coupling(weights, delayed_state, &cfg, None);
        let (data, shape) = crate::io::tensor_to_flat_f32(result);
        assert_eq!(shape, vec![2, 2]);
        assert!((data[0] - 3.0).abs() < 1e-5, "expected 3.0, got {}", data[0]);
        assert!((data[1] - 5.0).abs() < 1e-5, "expected 5.0, got {}", data[1]);
        assert!((data[2] - 17.0).abs() < 1e-5, "expected 17.0, got {}", data[2]);
        assert!((data[3] - 21.0).abs() < 1e-5, "expected 21.0, got {}", data[3]);
    }

    #[test]
    fn test_dense_coupling_sigmoidal_pipeline() {
        let dev = Default::default();
        let weights = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![1.0, 1.0],
                vec![1, 2],
            ), &dev,
        );
        let delayed_state = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![0.5, 1.5],
                vec![2, 1],
            ), &dev,
        );

        let cfg = CouplingFnConfig::Sigmoidal { cmin: 0.0, cmax: 1.0, midpoint: 0.0, a: 1.0, sigma: 1.0 };
        let result = dense_coupling(weights, delayed_state, &cfg, None);
        let (data, _) = crate::io::tensor_to_flat_f32(result);

        let expected = 1.0 / (1.0 + (-2.0f32).exp());
        assert!((data[0] - expected).abs() < 1e-5, "expected {}, got {}", expected, data[0]);
    }

    #[test]
    fn test_dense_coupling_tanh_pipeline() {
        let dev = Default::default();
        let weights = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![1.0, 1.0],
                vec![1, 2],
            ), &dev,
        );
        let delayed_state = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![1.0, -1.0],
                vec![2, 1],
            ), &dev,
        );

        let cfg = CouplingFnConfig::HyperbolicTangent { a: 1.0, b: 1.0, midpoint: 0.0, sigma: 1.0 };
        let result = dense_coupling(weights, delayed_state, &cfg, None);
        let (data, _) = crate::io::tensor_to_flat_f32(result);

        let expected = (1.0 + 1.0_f32.tanh()) + (1.0 + (-1.0f32).tanh());
        assert!((data[0] - expected).abs() < 1e-4, "expected {}, got {}", expected, data[0]);
    }

    #[test]
    fn test_presigmoidal_static_p_neq1_theta_neq0() {
        let cfg = CouplingFnConfig::PreSigmoidal { h: 2.0, q: 1.0, g: 3.0, p: 0.5, theta: 2.0, dynamic: false, global_t: false };
        let dev = Default::default();
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![4.0, 5.0, 6.0, 7.0], vec![2, 2]), &dev,
        );
        let result = cfg.pre(x);
        let (data, _) = crate::io::tensor_to_flat_f32(result);

        let expected = |v: f32| -> f32 { 2.0 * (1.0 + (3.0 * (0.5 * v - 2.0)).tanh()) };
        let tol = 1e-5;
        assert!((data[0] - expected(4.0)).abs() < tol, "x=4: got {}, expected {}", data[0], expected(4.0));
        assert!((data[1] - expected(5.0)).abs() < tol, "x=5: got {}, expected {}", data[1], expected(5.0));
        assert!((data[2] - expected(6.0)).abs() < tol, "x=6: got {}, expected {}", data[2], expected(6.0));
        assert!((data[3] - expected(7.0)).abs() < tol, "x=7: got {}, expected {}", data[3], expected(7.0));

        assert!((data[0] - 2.0).abs() < 0.01, "x=4: old formula would give ~3.99, correct gives 2.0, got {}", data[0]);
    }

    #[test]
    fn test_needs_x_i() {
        assert!(!CouplingFnConfig::Linear { a: 1.0, b: 0.0 }.needs_x_i());
        assert!(!CouplingFnConfig::Sigmoidal { cmin: -1.0, cmax: 1.0, midpoint: 0.0, a: 1.0, sigma: 1.0 }.needs_x_i());
        assert!(!CouplingFnConfig::Difference { a: 0.1, rowsums: None }.needs_x_i());
        assert!(CouplingFnConfig::Difference { a: 0.1, rowsums: Some(vec![1.0]) }.needs_x_i());
        assert!(CouplingFnConfig::Kuramoto { a: 1.0, n_src: 4 }.needs_x_i());
        assert!(!CouplingFnConfig::HyperbolicTangent { a: 1.0, b: 1.0, midpoint: 0.0, sigma: 1.0 }.needs_x_i());
    }

    #[test]
    fn test_pre_channels() {
        assert_eq!(CouplingFnConfig::Linear { a: 1.0, b: 0.0 }.pre_channels(), 1);
        assert_eq!(CouplingFnConfig::Kuramoto { a: 1.0, n_src: 4 }.pre_channels(), 2);
        assert_eq!(CouplingFnConfig::Difference { a: 0.1, rowsums: None }.pre_channels(), 1);
        assert_eq!(CouplingFnConfig::SigmoidalJansenRit { a: 1.0, use_classic: true, cmin: 0.0, cmax: 0.01, r: 0.56, midpoint: 6.0, e0: 0.005, v0: 6.0 }.pre_channels(), 1);
        assert_eq!(CouplingFnConfig::PreSigmoidal { h: 1.0, q: 1.0, g: 1.0, p: 1.0, theta: 0.5, dynamic: true, global_t: false }.pre_channels(), 1);
        assert_eq!(CouplingFnConfig::PreSigmoidal { h: 1.0, q: 1.0, g: 1.0, p: 1.0, theta: 0.5, dynamic: false, global_t: false }.pre_channels(), 1);
    }

    #[test]
    fn test_set_kuramoto_nsrc() {
        let mut cfg = CouplingFnConfig::Kuramoto { a: 1.0, n_src: 1 };
        cfg.set_kuramoto_nsrc(4);
        match cfg {
            CouplingFnConfig::Kuramoto { n_src, .. } => assert_eq!(n_src, 4),
            _ => panic!("expected Kuramoto"),
        }
    }

    #[test]
    fn test_coupling_config_serde_roundtrip() {
        let configs = vec![
            CouplingFnConfig::Linear { a: 1.0, b: 0.0 },
            CouplingFnConfig::Sigmoidal { cmin: -1.0, cmax: 1.0, midpoint: 0.0, a: 1.0, sigma: 1.0 },
            CouplingFnConfig::Difference { a: 1.0, rowsums: None },
            CouplingFnConfig::Kuramoto { a: 1.0, n_src: 1 },
            CouplingFnConfig::ScaledLinear { a: 2.0, b: 1.0 },
            CouplingFnConfig::HyperbolicTangent { a: 1.0, b: 1.0, midpoint: 0.0, sigma: 1.0 },
            CouplingFnConfig::SigmoidalJansenRit { a: 1.0, use_classic: false, cmin: 0.0, cmax: 0.005, r: 0.56, midpoint: 6.0, e0: 0.005, v0: 6.0 },
            CouplingFnConfig::PreSigmoidal { h: 1.0, q: 1.0, g: 1.0, p: 1.0, theta: 0.5, dynamic: false, global_t: false },
            CouplingFnConfig::PreSigmoidal { h: 1.0, q: 1.0, g: 1.0, p: 1.0, theta: 0.5, dynamic: true, global_t: true },
        ];
        for cfg in &configs {
            let json = serde_json::to_string(cfg).unwrap();
            let back: CouplingFnConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(json, serde_json::to_string(&back).unwrap());
        }
    }

    #[test]
    fn test_scaled_linear_post() {
        let cfg = CouplingFnConfig::ScaledLinear { a: 3.0, b: 2.0 };
        let dev: <B as Backend>::Device = Default::default();
        let gx = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![5.0], vec![1, 1]), &dev,
        );
        let result = cfg.post(gx);
        let (data, _) = crate::io::tensor_to_flat_f32(result);
        assert!((data[0] - 9.0).abs() < 1e-6, "expected 9.0, got {}", data[0]);
    }

    #[test]
    fn test_scaled_linear_pipeline() {
        let cfg = CouplingFnConfig::ScaledLinear { a: 2.0, b: 1.0 };
        let dev: <B as Backend>::Device = Default::default();
        let weights = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]), &dev,
        );
        let delayed_state = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![3.0, 5.0], vec![2, 1]), &dev,
        );
        let result = dense_coupling(weights, delayed_state, &cfg, None);
        let (data, _) = crate::io::tensor_to_flat_f32(result);
        assert!((data[0] - 4.0).abs() < 1e-5, "expected 4.0, got {}", data[0]);
        assert!((data[1] - 8.0).abs() < 1e-5, "expected 8.0, got {}", data[1]);
    }

    #[test]
    fn test_scaled_linear_pre_identity() {
        let cfg = CouplingFnConfig::ScaledLinear { a: 2.0, b: 1.0 };
        let dev: <B as Backend>::Device = Default::default();
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]), &dev,
        );
        let pre_result = cfg.pre(x);
        let (data, _) = crate::io::tensor_to_flat_f32(pre_result);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_presigmoidal_dynamic() {
        let cfg = CouplingFnConfig::PreSigmoidal { h: 1.0, q: 0.0, g: 1.0, p: 1.0, theta: 0.0, dynamic: true, global_t: false };
        let dev: <B as Backend>::Device = Default::default();
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0, 0.5, 2.0, 0.3], vec![2, 2]), &dev,
        );
        let result = cfg.pre(x);
        let (data, shape) = crate::io::tensor_to_flat_f32(result);
        assert_eq!(shape, vec![2, 1], "dynamic pre should collapse 2 cvars to 1");
        let inner0 = 1.0f32 * 1.0 - 0.5f32;
        let expected0 = 1.0 * (0.0 + (1.0 * inner0).tanh());
        assert!((data[0] - expected0).abs() < 1e-5, "row 0: expected {}, got {}", expected0, data[0]);
        let inner1 = 1.0f32 * 2.0 - 0.3f32;
        let expected1 = 1.0 * (0.0 + (1.0 * inner1).tanh());
        assert!((data[1] - expected1).abs() < 1e-5, "row 1: expected {}, got {}", expected1, data[1]);
    }

    #[test]
    fn test_presigmoidal_dynamic_global_t() {
        let cfg = CouplingFnConfig::PreSigmoidal { h: 1.0, q: 0.0, g: 1.0, p: 1.0, theta: 0.0, dynamic: true, global_t: true };
        let dev: <B as Backend>::Device = Default::default();
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0, 2.0, 0.8, 0.4], vec![2, 2]), &dev,
        );
        let result = cfg.pre(x);
        let (data, shape) = crate::io::tensor_to_flat_f32(result);
        assert_eq!(shape, vec![2, 1], "dynamic global_t pre should collapse 2 cvars to 1");
        let mean_threshold = (2.0f32 + 0.4f32) / 2.0f32;
        let inner0 = 1.0f32 * 1.0 - mean_threshold;
        let expected0 = 1.0 * (0.0 + (1.0 * inner0).tanh());
        assert!((data[0] - expected0).abs() < 1e-5, "row 0: expected {}, got {}", expected0, data[0]);
        let inner1 = 1.0f32 * 0.8 - mean_threshold;
        let expected1 = 1.0 * (0.0 + (1.0 * inner1).tanh());
        assert!((data[1] - expected1).abs() < 1e-5, "row 1: expected {}, got {}", expected1, data[1]);
    }

    #[test]
    fn test_presigmoidal_dynamic_2cvar_collapse() {
        let cfg = CouplingFnConfig::PreSigmoidal { h: 2.0, q: 1.0, g: 3.0, p: 0.5, theta: 0.0, dynamic: true, global_t: false };
        let dev: <B as Backend>::Device = Default::default();
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![4.0, 1.0, 6.0, 2.0], vec![2, 2]), &dev,
        );
        let result = cfg.pre(x);
        let (data, shape) = crate::io::tensor_to_flat_f32(result);
        assert_eq!(shape, vec![2, 1], "dynamic pre should output 1 channel");
        let inner0 = 0.5f32 * 4.0 - 1.0f32;
        let expected0 = 2.0 * (1.0 + (3.0 * inner0).tanh());
        assert!((data[0] - expected0).abs() < 1e-4, "row 0: expected {}, got {}", expected0, data[0]);
        let inner1 = 0.5f32 * 6.0 - 2.0f32;
        let expected1 = 2.0 * (1.0 + (3.0 * inner1).tanh());
        assert!((data[1] - expected1).abs() < 1e-4, "row 1: expected {}, got {}", expected1, data[1]);
    }

    #[test]
    fn test_presigmoidal_dynamic_needs_two_src_cvar() {
        let cfg_dynamic = CouplingFnConfig::PreSigmoidal { h: 1.0, q: 1.0, g: 1.0, p: 1.0, theta: 0.5, dynamic: true, global_t: false };
        let cfg_static = CouplingFnConfig::PreSigmoidal { h: 1.0, q: 1.0, g: 1.0, p: 1.0, theta: 0.5, dynamic: false, global_t: false };
        assert!(cfg_dynamic.needs_two_src_cvar());
        assert!(!cfg_static.needs_two_src_cvar());
        assert_eq!(cfg_dynamic.min_src_ncvar(), 2);
        assert_eq!(cfg_static.min_src_ncvar(), 1);
        assert_eq!(cfg_dynamic.pre_channels(), 1);
        assert_eq!(cfg_static.pre_channels(), 1);
    }

    #[test]
    fn test_sigmoidal_jansen_rit_pre_2cvar() {
        let cfg = CouplingFnConfig::SigmoidalJansenRit { a: 1.0, use_classic: false, cmin: 0.0, cmax: 0.005, r: 0.56, midpoint: 6.0, e0: 0.005, v0: 6.0 };
        let dev: <B as Backend>::Device = Default::default();
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![6.0, 4.0, 6.0, 4.0], vec![2, 2]), &dev,
        );
        let result = cfg.pre(x);
        let (data, _) = crate::io::tensor_to_flat_f32(result);
        assert!(data.iter().all(|v| v.is_finite()));
        assert!(data.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn test_sigmoidal_jansen_rit_pre_single_cvar_fallback() {
        let cfg = CouplingFnConfig::SigmoidalJansenRit { a: 1.0, use_classic: false, cmin: 0.0, cmax: 0.005, r: 0.56, midpoint: 6.0, e0: 0.005, v0: 6.0 };
        let dev: <B as Backend>::Device = Default::default();
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![6.0, 8.0], vec![2, 1]), &dev,
        );
        let result = cfg.pre(x);
        let (data, _) = crate::io::tensor_to_flat_f32(result);
        assert!(data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_sigmoidal_jansen_rit_classic_pre() {
        let cfg = CouplingFnConfig::SigmoidalJansenRit {
            a: 2.0,
            use_classic: true,
            cmin: 0.0,
            cmax: 0.01,
            r: 0.56,
            midpoint: 6.0,
            e0: 0.005,
            v0: 6.0,
        };
        let dev: <B as Backend>::Device = Default::default();
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![8.0, 2.0, 4.0, 0.0], vec![2, 2]), &dev,
        );
        let result = cfg.pre(x);
        let (data, shape) = crate::io::tensor_to_flat_f32(result);
        assert_eq!(shape, vec![2, 1], "classic pre should collapse to 1 channel");
        let diff1 = 8.0f32 - 2.0f32;
        let expected1 = 0.0 + (0.01 - 0.0) / (1.0 + (0.56 * (6.0 - diff1)).exp());
        assert!((data[0] - expected1).abs() < 1e-5, "expected {}, got {}", expected1, data[0]);
        let diff2 = 4.0f32 - 0.0f32;
        let expected2 = 0.0 + (0.01 - 0.0) / (1.0 + (0.56 * (6.0 - diff2)).exp());
        assert!((data[1] - expected2).abs() < 1e-5, "expected {}, got {}", expected2, data[1]);
    }

    #[test]
    fn test_sigmoidal_jansen_rit_classic_post() {
        let cfg = CouplingFnConfig::SigmoidalJansenRit {
            a: 3.0,
            use_classic: true,
            cmin: 0.0,
            cmax: 0.01,
            r: 0.56,
            midpoint: 6.0,
            e0: 0.005,
            v0: 6.0,
        };
        let dev: <B as Backend>::Device = Default::default();
        let gx = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0, 2.0], vec![2, 1]), &dev,
        );
        let result = cfg.post(gx);
        let (data, _) = crate::io::tensor_to_flat_f32(result);
        assert!((data[0] - 3.0).abs() < 1e-5, "classic post: expected 3.0, got {}", data[0]);
        assert!((data[1] - 6.0).abs() < 1e-5, "classic post: expected 6.0, got {}", data[1]);
    }

    #[test]
    fn test_sigmoidal_jansen_rit_legacy_post_is_identity() {
        let cfg = CouplingFnConfig::SigmoidalJansenRit {
            a: 1.0,
            use_classic: false,
            cmin: 0.0,
            cmax: 0.005,
            r: 0.56,
            midpoint: 6.0,
            e0: 0.005,
            v0: 6.0,
        };
        let dev: <B as Backend>::Device = Default::default();
        let gx = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![0.01, 0.02], vec![2, 1]), &dev,
        );
        let result = cfg.post(gx.clone());
        let (data, _) = crate::io::tensor_to_flat_f32(result);
        let (orig, _) = crate::io::tensor_to_flat_f32(gx);
        assert!((data[0] - orig[0]).abs() < 1e-7, "legacy post should be identity");
        assert!((data[1] - orig[1]).abs() < 1e-7, "legacy post should be identity");
    }

    #[test]
    fn test_pre_3d_post_3d_roundtrip() {
        let cfg = CouplingFnConfig::Linear { a: 2.0, b: 1.0 };
        let dev: <B as Backend>::Device = Default::default();
        let x = Tensor::<B, 3>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3, 1]), &dev,
        );
        let pre_result = cfg.pre_3d(x);
        assert_eq!(pre_result.shape().dims, [2, 3, 1]);

        let post_result = cfg.post_3d(pre_result);
        assert_eq!(post_result.shape().dims, [2, 3, 1]);
        let (data, _) = crate::io::tensor_to_flat_f32(post_result);
        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[5] - 13.0).abs() < 1e-5);
    }

    #[test]
    fn test_kuramoto_pre_3d_channel_expansion() {
        let cfg = CouplingFnConfig::Kuramoto { a: 1.0, n_src: 3 };
        let dev: <B as Backend>::Device = Default::default();
        let x = Tensor::<B, 3>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![0.0, 1.5708], vec![1, 2, 1]), &dev,
        );
        let pre_result = cfg.pre_3d(x);
        assert_eq!(pre_result.shape().dims, [1, 2, 2]);
        let (data, _) = crate::io::tensor_to_flat_f32(pre_result);
        assert!(data[0].abs() < 1e-5);
        assert!((data[1] - 1.0).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
        assert!(data[3].abs() < 1e-5);
    }
}


#[cfg(test)]
mod serde_compat_tests {
    use super::*;

    #[test]
    fn test_tanh_old_2param_deserialize() {
        let old_json = r#"{"HyperbolicTangent":{"a":1.0,"b":1.0}}"#;
        let cfg: CouplingFnConfig = serde_json::from_str(old_json).unwrap();
        match cfg {
            CouplingFnConfig::HyperbolicTangent { a, b, midpoint, sigma } => {
                assert_eq!(a, 1.0);
                assert_eq!(b, 1.0);
                assert_eq!(midpoint, 0.0);
                assert_eq!(sigma, 1.0);
            }
            _ => panic!("expected HyperbolicTangent"),
        }
    }

    #[test]
    fn test_sigmoidal_old_3param_deserialize() {
        let cfg = CouplingFnConfig::from_name_and_params("Sigmoidal", &[1.0, 0.0, 2.0]).unwrap();
        match cfg {
            CouplingFnConfig::Sigmoidal { cmin, cmax, midpoint, a, sigma } => {
                assert_eq!(cmin, -1.0);
                assert_eq!(cmax, 1.0);
                assert_eq!(midpoint, 0.0);
                assert_eq!(a, 1.0);
                assert_eq!(sigma, 2.0);
            }
            _ => panic!("expected Sigmoidal"),
        }
    }

    #[test]
    fn test_sigmoidal_json_with_defaults() {
        let json = r#"{"Sigmoidal":{"cmax":1.0,"midpoint":0.0}}"#;
        let cfg: CouplingFnConfig = serde_json::from_str(json).unwrap();
        match cfg {
            CouplingFnConfig::Sigmoidal { cmin, cmax, midpoint, a, sigma } => {
                assert_eq!(cmin, -1.0);
                assert_eq!(cmax, 1.0);
                assert_eq!(midpoint, 0.0);
                assert_eq!(a, 1.0);
                assert_eq!(sigma, 1.0);
            }
            _ => panic!("expected Sigmoidal"),
        }
    }

    #[test]
    fn test_kuramoto_old_json_deserialize() {
        let old_json = r#"{"Kuramoto":{"a":1.0}}"#;
        let cfg: CouplingFnConfig = serde_json::from_str(old_json).unwrap();
        match cfg {
            CouplingFnConfig::Kuramoto { a, n_src } => {
                assert_eq!(a, 1.0);
                assert_eq!(n_src, 1); // default
            }
            _ => panic!("expected Kuramoto"),
        }
    }

    #[test]
    fn test_difference_old_json_deserialize() {
        let old_json = r#"{"Difference":{"a":0.1}}"#;
        let cfg: CouplingFnConfig = serde_json::from_str(old_json).unwrap();
        match cfg {
            CouplingFnConfig::Difference { a, rowsums } => {
                assert_eq!(a, 0.1);
                assert!(rowsums.is_none()); // rowsums skipped in serde
            }
            _ => panic!("expected Difference"),
        }
    }

    #[test]
    fn test_sjr_old_json_deserialize_legacy() {
        let old_json = r#"{"SigmoidalJansenRit":{"a":1.0,"e0":0.005,"r":0.56,"v0":6.0}}"#;
        let cfg: CouplingFnConfig = serde_json::from_str(old_json).unwrap();
        match cfg {
            CouplingFnConfig::SigmoidalJansenRit { a, use_classic, cmin, cmax, r, midpoint, e0, v0 } => {
                assert_eq!(a, 1.0);
                assert!(!use_classic, "old JSON should default use_classic to false");
                assert_eq!(e0, 0.005);
                assert_eq!(r, 0.56);
                assert_eq!(v0, 6.0);
                assert_eq!(cmin, 0.0);
                assert_eq!(cmax, 0.005);
                assert_eq!(midpoint, 6.0);
            }
            _ => panic!("expected SigmoidalJansenRit"),
        }
    }

    #[test]
    fn test_sjr_classic_json_deserialize() {
        let json = r#"{"SigmoidalJansenRit":{"a":2.0,"use_classic":true,"cmin":0.0,"cmax":0.01,"r":0.56,"midpoint":6.0}}"#;
        let cfg: CouplingFnConfig = serde_json::from_str(json).unwrap();
        match cfg {
            CouplingFnConfig::SigmoidalJansenRit { a, use_classic, cmin, cmax, r, midpoint, .. } => {
                assert_eq!(a, 2.0);
                assert!(use_classic);
                assert_eq!(cmin, 0.0);
                assert_eq!(cmax, 0.01);
                assert_eq!(r, 0.56);
                assert_eq!(midpoint, 6.0);
            }
            _ => panic!("expected SigmoidalJansenRit"),
        }
    }

    #[test]
    fn test_sjr_from_params_4param_legacy() {
        let cfg = CouplingFnConfig::from_name_and_params("SigmoidalJansenRit", &[5.0, 0.005, 0.56, 6.0]).unwrap();
        match cfg {
            CouplingFnConfig::SigmoidalJansenRit { a, use_classic, e0, r, v0, .. } => {
                assert!(!use_classic, "4-param should be legacy mode");
                assert_eq!(a, 5.0);
                assert_eq!(e0, 0.005);
                assert_eq!(r, 0.56);
                assert_eq!(v0, 6.0);
            }
            _ => panic!("expected SigmoidalJansenRit"),
        }
    }

    #[test]
    fn test_sjr_from_params_5param_classic() {
        let cfg = CouplingFnConfig::from_name_and_params("SigmoidalJansenRit", &[2.0, 0.0, 0.01, 0.56, 6.0]).unwrap();
        match cfg {
            CouplingFnConfig::SigmoidalJansenRit { a, use_classic, cmin, cmax, r, midpoint, .. } => {
                assert!(use_classic, "5-param should be classic mode");
                assert_eq!(a, 2.0);
                assert_eq!(cmin, 0.0);
                assert_eq!(cmax, 0.01);
                assert_eq!(r, 0.56);
                assert_eq!(midpoint, 6.0);
            }
            _ => panic!("expected SigmoidalJansenRit"),
        }
    }
}
