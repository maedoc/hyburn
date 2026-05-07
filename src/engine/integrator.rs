//! Integration schemes.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// Supported integrator kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum IntegratorKind {
    #[default]
    Heun,
    Euler,
    EulerStochastic,
    HeunStochastic,
}


impl fmt::Display for IntegratorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegratorKind::Heun => write!(f, "heun"),
            IntegratorKind::Euler => write!(f, "euler"),
            IntegratorKind::EulerStochastic => write!(f, "euler_stochastic"),
            IntegratorKind::HeunStochastic => write!(f, "heun_stochastic"),
        }
    }
}

impl FromStr for IntegratorKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "heun" => Ok(IntegratorKind::Heun),
            "euler" => Ok(IntegratorKind::Euler),
            "euler_stochastic" => Ok(IntegratorKind::EulerStochastic),
            "heun_stochastic" => Ok(IntegratorKind::HeunStochastic),
            _ => Err(format!("unknown integrator kind: {}", s)),
        }
    }
}

/// Forward Euler step with clamping.
///
/// `state_new = state + dt * dfun(state, coupling)`
pub fn euler_step<B: Backend>(
    state: Tensor<B, 2>,
    coupling: Tensor<B, 2>,
    dt: f32,
    dfun: impl Fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
    clamp: impl Fn(&mut Tensor<B, 2>),
) -> Tensor<B, 2> {
    let d = dfun(state.clone(), coupling);
    let mut new_state = state + d.mul_scalar(dt);
    clamp(&mut new_state);
    new_state
}

/// Heun predictor-corrector step with midpoint clamping.
///
/// 1. `k1 = dfun(state, coupling)`
/// 2. `predictor = state + dt * k1`
/// 3. `clamp(predictor)`
/// 4. `k2 = dfun(predictor, coupling)`
/// 5. `corrector = state + dt/2 * (k1 + k2)`
/// 6. `clamp(corrector)`
pub fn heun_step<B: Backend>(
    state: Tensor<B, 2>,
    coupling: Tensor<B, 2>,
    dt: f32,
    dfun: impl Fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
    clamp: impl Fn(&mut Tensor<B, 2>),
) -> Tensor<B, 2> {
    let d0 = dfun(state.clone(), coupling.clone());
    let mut predictor = state.clone() + d0.clone().mul_scalar(dt);
    clamp(&mut predictor);

    let d1 = dfun(predictor, coupling);
    let mut corrector = state + (d0 + d1).mul_scalar(dt * 0.5);
    clamp(&mut corrector);

    corrector
}

/// Generate Gaussian noise tensor with given shape and scale.
fn generate_noise<B: Backend>(shape: [usize; 2], scale: f32, device: &B::Device) -> Tensor<B, 2> {
    let n = shape[0] * shape[1];
    let normal = Normal::new(0.0f64, scale as f64).unwrap();
    let data: Vec<f32> = (0..n).map(|_| normal.sample(&mut thread_rng()) as f32).collect();
    Tensor::from_floats(
        TensorData::new::<f32, Vec<usize>>(data, vec![shape[0], shape[1]]),
        device,
    )
}

/// Stochastic Euler-Maruyama step.
///
/// `state_new = state + dt * dfun(state, coupling) + nsig * sqrt(dt) * Z`
pub fn euler_stochastic_step<B: Backend>(
    state: Tensor<B, 2>,
    coupling: Tensor<B, 2>,
    dt: f32,
    nsig: f32,
    dfun: impl Fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
    clamp: impl Fn(&mut Tensor<B, 2>),
) -> Tensor<B, 2> {
    let dims = state.shape().dims;
    let noise = generate_noise::<B>([dims[0], dims[1]], nsig * dt.sqrt(), &state.device());
    let d = dfun(state.clone(), coupling);
    let mut new_state = state + d.mul_scalar(dt) + noise;
    clamp(&mut new_state);
    new_state
}

/// Stochastic Heun step (weak order 1.0).
///
/// Uses the same noise for predictor and corrector.
pub fn heun_stochastic_step<B: Backend>(
    state: Tensor<B, 2>,
    coupling: Tensor<B, 2>,
    dt: f32,
    nsig: f32,
    dfun: impl Fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
    clamp: impl Fn(&mut Tensor<B, 2>),
) -> Tensor<B, 2> {
    let dims = state.shape().dims;
    let noise = generate_noise::<B>([dims[0], dims[1]], nsig * dt.sqrt(), &state.device());

    let d0 = dfun(state.clone(), coupling.clone());
    let mut predictor = state.clone() + d0.clone().mul_scalar(dt) + noise.clone();
    clamp(&mut predictor);

    let d1 = dfun(predictor, coupling);
    let mut corrector = state + (d0 + d1).mul_scalar(dt * 0.5) + noise;
    clamp(&mut corrector);
    corrector
}
