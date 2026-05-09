//! Coupling functions and dense coupling kernel.
//!
//! Ported from the Python `tvb.simulator.hybrid.coupling` module.
//! Functions: Linear, Sigmoidal, Kuramoto, Difference, ScaledLinear, HyperbolicTangent, etc.

use burn::prelude::Backend;
use burn::tensor::Tensor;

/// Serializable / storable coupling-function configuration.
#[derive(Debug, Clone)]
pub enum CouplingFnConfig {
    Linear { a: f32, b: f32 },
    Sigmoidal { cmax: f32, midpoint: f32, steepness: f32 },
    Difference { a: f32 },
    Kuramoto { a: f32 },
    ScaledLinear { a: f32, b: f32 },
    HyperbolicTangent { a: f32, b: f32 },
    SigmoidalJansenRit { a: f32, e0: f32, r: f32, v0: f32 },
    PreSigmoidal { h: f32, q: f32, g: f32, p: f32, theta: f32 },
}

impl CouplingFnConfig {
    /// Minimum source ncvar required by this coupling function.
    pub fn min_src_ncvar(&self) -> usize {
        match self {
            CouplingFnConfig::Difference { .. } => 2,
            _ => 1,
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
                let cmax = params.first().copied().unwrap_or(1.0);
                let midpoint = params.get(1).copied().unwrap_or(0.0);
                let steepness = params.get(2).copied().unwrap_or(1.0);
                Some(CouplingFnConfig::Sigmoidal { cmax, midpoint, steepness })
            }
            "Difference" => {
                let a = params.first().copied().unwrap_or(1.0);
                Some(CouplingFnConfig::Difference { a })
            }
            "Kuramoto" => {
                let a = params.first().copied().unwrap_or(1.0);
                Some(CouplingFnConfig::Kuramoto { a })
            }
            "ScaledLinear" => {
                let a = params.first().copied().unwrap_or(1.0);
                let b = params.get(1).copied().unwrap_or(0.0);
                Some(CouplingFnConfig::ScaledLinear { a, b })
            }
            "HyperbolicTangent" => {
                let a = params.first().copied().unwrap_or(1.0);
                let b = params.get(1).copied().unwrap_or(1.0);
                Some(CouplingFnConfig::HyperbolicTangent { a, b })
            }
            "SigmoidalJansenRit" => {
                let a = params.first().copied().unwrap_or(1.0);
                let e0 = params.get(1).copied().unwrap_or(0.005);
                let r = params.get(2).copied().unwrap_or(0.56);
                let v0 = params.get(3).copied().unwrap_or(6.0);
                Some(CouplingFnConfig::SigmoidalJansenRit { a, e0, r, v0 })
            }
            "PreSigmoidal" => {
                let h = params.first().copied().unwrap_or(1.0);
                let q = params.get(1).copied().unwrap_or(1.0);
                let g = params.get(2).copied().unwrap_or(1.0);
                let p = params.get(3).copied().unwrap_or(1.0);
                let theta = params.get(4).copied().unwrap_or(0.5);
                Some(CouplingFnConfig::PreSigmoidal { h, q, g, p, theta })
            }
            _ => None,
        }
    }

    /// Apply the configured coupling function to a tensor.
    pub fn apply<B: Backend>(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        match self {
            CouplingFnConfig::Linear { a, b } => Linear { a: *a, b: *b }.apply(x),
            CouplingFnConfig::Sigmoidal { cmax, midpoint, steepness } => {
                Sigmoidal { cmax: *cmax, midpoint: *midpoint, steepness: *steepness }.apply(x)
            }
            CouplingFnConfig::Difference { a } => Difference { a: *a }.apply(x),
            CouplingFnConfig::Kuramoto { a } => Kuramoto { a: *a }.apply(x),
            CouplingFnConfig::ScaledLinear { a, b } => ScaledLinear { a: *a, b: *b }.apply(x),
            CouplingFnConfig::HyperbolicTangent { a, b } => HyperbolicTangent { a: *a, b: *b }.apply(x),
            CouplingFnConfig::SigmoidalJansenRit { a, e0, r, v0 } => {
                SigmoidalJansenRit { a: *a, e0: *e0, r: *r, v0: *v0 }.apply(x)
            }
            CouplingFnConfig::PreSigmoidal { h, q, g, p, theta } => {
                PreSigmoidal { h: *h, q: *q, g: *g, p: *p, theta: *theta }.apply(x)
            }
        }
    }

    pub fn apply_3d<B: Backend>(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = x.shape().dims;
        let flat = x.reshape([dims[0] * dims[1], dims[2]]);
        let applied = self.apply(flat);
        applied.reshape([dims[0], dims[1], dims[2]])
    }

    /// Convert the config enum into a boxed trait object for dynamic dispatch.
    pub fn to_boxed<B: Backend>(&self) -> Box<dyn CouplingFn<B>> {
        match self {
            CouplingFnConfig::Linear { a, b } => Box::new(Linear { a: *a, b: *b }),
            CouplingFnConfig::Sigmoidal { cmax, midpoint, steepness } => {
                Box::new(Sigmoidal { cmax: *cmax, midpoint: *midpoint, steepness: *steepness })
            }
            CouplingFnConfig::Difference { a } => Box::new(Difference { a: *a }),
            CouplingFnConfig::Kuramoto { a } => Box::new(Kuramoto { a: *a }),
            CouplingFnConfig::ScaledLinear { a, b } => Box::new(ScaledLinear { a: *a, b: *b }),
            CouplingFnConfig::HyperbolicTangent { a, b } => Box::new(HyperbolicTangent { a: *a, b: *b }),
            CouplingFnConfig::SigmoidalJansenRit { a, e0, r, v0 } => {
                Box::new(SigmoidalJansenRit { a: *a, e0: *e0, r: *r, v0: *v0 })
            }
            CouplingFnConfig::PreSigmoidal { h, q, g, p, theta } => {
                Box::new(PreSigmoidal { h: *h, q: *q, g: *g, p: *p, theta: *theta })
            }
        }
    }
}

/// Apply a coupling function to a weighted sum of delayed source states.
pub trait CouplingFn<B: Backend> {
    fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2>;
}

/// Linear coupling: `f(x) = a * x + b`
pub struct Linear {
    pub a: f32,
    pub b: f32,
}

impl<B: Backend> CouplingFn<B> for Linear {
    fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.mul_scalar(self.a).add_scalar(self.b)
    }
}

/// Sigmoidal coupling:
/// `f(x) = cmax / (1 + exp(-(x - midpoint) / steepness))`
pub struct Sigmoidal {
    pub cmax: f32,
    pub midpoint: f32,
    pub steepness: f32,
}

impl<B: Backend> CouplingFn<B> for Sigmoidal {
    fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let shifted = x.add_scalar(-self.midpoint);
        let exponent = shifted.div_scalar(self.steepness).neg();
        let denom = exponent.exp().add_scalar(1.0);
        let sigmoid = denom.recip();
        sigmoid.mul_scalar(self.cmax)
    }
}

/// Difference coupling: `f(x) = a * (x1 - x2)` — requires `ncvar >= 2`
pub struct Difference {
    pub a: f32,
}

impl<B: Backend> CouplingFn<B> for Difference {
    fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let dim1 = x.shape().dims[1];
        if dim1 >= 2 {
            let x1 = x.clone().narrow(1, 0, 1);
            let x2 = x.clone().narrow(1, 1, 1);
            (x1 - x2).mul_scalar(self.a)
        } else {
            // ncvar < 2: difference is zero, return zeros
            x.zeros_like()
        }
    }
}

/// Kuramoto coupling (phase): `f(x) = a * sin(x)`
pub struct Kuramoto {
    pub a: f32,
}

impl<B: Backend> CouplingFn<B> for Kuramoto {
    fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.sin().mul_scalar(self.a)
    }
}

/// Scaled linear coupling: `f(x) = a * (x - b)`
pub struct ScaledLinear {
    pub a: f32,
    pub b: f32,
}

impl<B: Backend> CouplingFn<B> for ScaledLinear {
    fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.add_scalar(-self.b).mul_scalar(self.a)
    }
}

/// Hyperbolic tangent coupling: `f(x) = a * tanh(b * x)`
pub struct HyperbolicTangent {
    pub a: f32,
    pub b: f32,
}

impl<B: Backend> CouplingFn<B> for HyperbolicTangent {
    fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.mul_scalar(self.b).tanh().mul_scalar(self.a)
    }
}

pub struct SigmoidalJansenRit {
    pub a: f32,
    pub e0: f32,
    pub r: f32,
    pub v0: f32,
}

impl<B: Backend> CouplingFn<B> for SigmoidalJansenRit {
    fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let shifted = x.add_scalar(-self.v0).mul_scalar(-self.r);
        let denom = shifted.exp().add_scalar(1.0);
        denom.recip().mul_scalar(self.a * 2.0 * self.e0)
    }
}

pub struct PreSigmoidal {
    pub h: f32,
    pub q: f32,
    pub g: f32,
    pub p: f32,
    pub theta: f32,
}

impl<B: Backend> CouplingFn<B> for PreSigmoidal {
    fn apply(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let inner = x.mul_scalar(self.p).add_scalar(-self.theta).mul_scalar(self.g);
        inner.tanh().add_scalar(self.q).mul_scalar(self.h)
    }
}

/// Dense coupling kernel.
///
/// - `weights` has shape `[ntgt, nsrc]`.
/// - `delayed_state` has shape `[nsrc, ncvar]`.
///
/// Returns coupling of shape `[ntgt, ncvar]` by applying `coupling_fn`
/// to each source row and then weighting with `weights` via matrix multiply.
pub fn dense_coupling<B: Backend>(
    weights: Tensor<B, 2>,
    delayed_state: Tensor<B, 2>,
    coupling_fn: &dyn CouplingFn<B>,
) -> Tensor<B, 2> {
    debug_assert!(
        weights.shape().dims[1] == delayed_state.shape().dims[0],
        "dense_coupling: weights cols ({}) must match delayed_state rows ({})",
        weights.shape().dims[1], delayed_state.shape().dims[0]
    );
    let pre = coupling_fn.apply(delayed_state);
    weights.matmul(pre)
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
    use burn::backend::ndarray::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    #[test]
    fn test_linear() {
        let c = Linear { a: 2.0, b: 0.0 };
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            &Default::default(),
        );
        let y = c.apply(x);
        let (data, _) = crate::io::tensor_to_flat_f32(y);
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_linear_with_offset() {
        let c = Linear { a: 2.0, b: 1.0 };
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0], vec![1, 1]),
            &Default::default(),
        );
        let y = c.apply(x);
        let (data, _) = crate::io::tensor_to_flat_f32(y);
        assert!((data[0] - 3.0).abs() < 1e-6, "expected 3.0, got {}", data[0]);
    }

    #[test]
    fn test_sigmoidal() {
        let c = Sigmoidal { cmax: 1.0, midpoint: 0.0, steepness: 1.0 };
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![0.0], vec![1, 1]),
            &Default::default(),
        );
        let y = c.apply(x);
        let (data, _) = crate::io::tensor_to_flat_f32(y);
        assert!((data[0] - 0.5).abs() < 1e-6, "expected 0.5, got {}", data[0]);
    }

    #[test]
    fn test_difference() {
        let c = Difference { a: 1.0 };
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![3.0, 1.0], vec![1, 2]),
            &Default::default(),
        );
        let y = c.apply(x);
        let (data, _) = crate::io::tensor_to_flat_f32(y);
        assert!((data[0] - 2.0).abs() < 1e-6, "expected 2.0, got {}", data[0]);
    }

    #[test]
    fn test_kuramoto() {
        let c = Kuramoto { a: 1.0 };
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![0.0], vec![1, 1]),
            &Default::default(),
        );
        let y = c.apply(x);
        let (data, _) = crate::io::tensor_to_flat_f32(y);
        assert!(data[0].abs() < 1e-6, "expected 0.0, got {}", data[0]);
    }

    #[test]
    fn test_scaled_linear() {
        let c = ScaledLinear { a: 2.0, b: 1.0 };
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![2.0, 3.0], vec![1, 2]),
            &Default::default(),
        );
        let y = c.apply(x);
        let (data, _) = crate::io::tensor_to_flat_f32(y);
        assert!((data[0] - 2.0).abs() < 1e-6, "expected 2.0, got {}", data[0]);
        assert!((data[1] - 4.0).abs() < 1e-6, "expected 4.0, got {}", data[1]);
    }

    #[test]
    fn test_hyperbolic_tangent() {
        let c = HyperbolicTangent { a: 1.0, b: 1.0 };
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![0.0], vec![1, 1]),
            &Default::default(),
        );
        let y = c.apply(x);
        let (data, _) = crate::io::tensor_to_flat_f32(y);
        assert!(data[0].abs() < 1e-6, "expected 0.0, got {}", data[0]);

        let x2 = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0], vec![1, 1]),
            &Default::default(),
        );
        let y2 = c.apply(x2);
        let (data2, _) = crate::io::tensor_to_flat_f32(y2);
        let expected = 1.0_f32.tanh();
        assert!((data2[0] - expected).abs() < 1e-5, "expected {}, got {}", expected, data2[0]);
    }

    #[test]
    fn test_dense_coupling() {
        // weights: [ntgt=2, nsrc=3]
        let weights = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 1.0,
                ],
                vec![2, 3],
            ),
            &Default::default(),
        );
        // delayed_state: [nsrc=3, ncvar=2]
        let delayed_state = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![
                    1.0, 2.0,
                    3.0, 4.0,
                    5.0, 6.0,
                ],
                vec![3, 2],
            ),
            &Default::default(),
        );
        let c = Linear { a: 1.0, b: 0.0 };
        let result = dense_coupling(weights, delayed_state, &c);
        let (data, shape) = crate::io::tensor_to_flat_f32(result);
        assert_eq!(shape, vec![2, 2]);
        // row 0: [1*1, 1*2] = [1, 2]
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        // row 1: [1*3 + 1*5, 1*4 + 1*6] = [8, 10]
        assert!((data[2] - 8.0).abs() < 1e-6);
        assert!((data[3] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaled_linear_config() {
        let cfg = CouplingFnConfig::ScaledLinear { a: 2.0, b: 1.0 };
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![2.0, 3.0], vec![1, 2]),
            &Default::default(),
        );
        let y = cfg.apply(x);
        let (data, _) = crate::io::tensor_to_flat_f32(y);
        assert!((data[0] - 2.0).abs() < 1e-6, "expected 2.0, got {}", data[0]);
        assert!((data[1] - 4.0).abs() < 1e-6, "expected 4.0, got {}", data[1]);
    }

    #[test]
    fn test_hyperbolic_tangent_config() {
        let cfg = CouplingFnConfig::HyperbolicTangent { a: 1.0, b: 1.0 };
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0], vec![1, 1]),
            &Default::default(),
        );
        let y = cfg.apply(x);
        let (data, _) = crate::io::tensor_to_flat_f32(y);
        let expected = 1.0_f32.tanh();
        assert!((data[0] - expected).abs() < 1e-5, "expected {}, got {}", expected, data[0]);
    }

    #[test]
    fn test_sigmoidal_jansen_rit() {
        let a = 1.0f32;
        let e0 = 0.005f32;
        let r = 0.56f32;
        let v0 = 6.0f32;
        let c = SigmoidalJansenRit { a, e0, r, v0 };
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![6.0], vec![1, 1]),
            &Default::default(),
        );
        let y = c.apply(x);
        let (data, _) = crate::io::tensor_to_flat_f32(y);
        let shifted = -r * (6.0f32 - v0);
        let expected = (a * 2.0 * e0) / (shifted.exp() + 1.0);
        assert!((data[0] - expected).abs() < 1e-6, "expected {}, got {}", expected, data[0]);
    }

    #[test]
    fn test_pre_sigmoidal() {
        let h = 1.0f32;
        let q = 1.0f32;
        let g = 1.0f32;
        let p = 1.0f32;
        let theta = 0.5f32;
        let c = PreSigmoidal { h, q, g, p, theta };
        let x = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0], vec![1, 1]),
            &Default::default(),
        );
        let y = c.apply(x);
        let (data, _) = crate::io::tensor_to_flat_f32(y);
        let inner = g * (p * 1.0f32 - theta);
        let expected = h * (inner.tanh() + q);
        assert!((data[0] - expected).abs() < 1e-5, "expected {}, got {}", expected, data[0]);
    }

    #[test]
    fn test_dense_coupling_with_new_functions() {
        let weights = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                vec![2, 3],
            ),
            &Default::default(),
        );
        let delayed_state = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![3, 2],
            ),
            &Default::default(),
        );

        let fns: Vec<Box<dyn CouplingFn<B>>> = vec![
            Box::new(ScaledLinear { a: 1.0, b: 0.0 }),
            Box::new(HyperbolicTangent { a: 1.0, b: 1.0 }),
            Box::new(SigmoidalJansenRit { a: 1.0, e0: 0.005, r: 0.56, v0: 6.0 }),
            Box::new(PreSigmoidal { h: 1.0, q: 0.0, g: 1.0, p: 1.0, theta: 0.0 }),
        ];
        for coupling_fn in &fns {
            let result = dense_coupling(weights.clone(), delayed_state.clone(), coupling_fn.as_ref());
            let (data, shape) = crate::io::tensor_to_flat_f32(result);
            assert_eq!(shape, vec![2, 2]);
            assert!(data.iter().all(|x| x.is_finite()), "non-finite result for coupling function");
        }
    }
}
