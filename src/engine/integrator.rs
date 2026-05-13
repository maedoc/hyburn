//! Integration schemes.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use rand::{thread_rng, SeedableRng};
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use std::cell::RefCell;
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
    Rk4,
    Rk4Stochastic,
}


impl fmt::Display for IntegratorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegratorKind::Heun => write!(f, "heun"),
            IntegratorKind::Euler => write!(f, "euler"),
            IntegratorKind::EulerStochastic => write!(f, "euler_stochastic"),
            IntegratorKind::HeunStochastic => write!(f, "heun_stochastic"),
            IntegratorKind::Rk4 => write!(f, "rk4"),
            IntegratorKind::Rk4Stochastic => write!(f, "rk4_stochastic"),
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
            "rk4" => Ok(IntegratorKind::Rk4),
            "rk4_stochastic" => Ok(IntegratorKind::Rk4Stochastic),
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

thread_local! {
    /// Optional seeded RNG for deterministic stochastic tests.
    static SEED_RNG: RefCell<Option<StdRng>> = const { RefCell::new(None) };
}

/// Set a deterministic seed for stochastic integration noise generation.
/// Overrides `thread_rng()` in `generate_noise_per_var` for the current thread.
pub fn set_seed(seed: u64) {
    SEED_RNG.with(|rng| {
        *rng.borrow_mut() = Some(StdRng::seed_from_u64(seed));
    });
}

/// Clear the deterministic seed and revert to `thread_rng()`.
pub fn clear_seed() {
    SEED_RNG.with(|rng| {
        *rng.borrow_mut() = None;
    });
}

/// Generate Gaussian noise tensor with given shape and per-column scales.
///
/// `nsig_slice` has length equal to the number of columns (nvar).
/// Each column `v` gets noise with scale `sqrt(2 * nsig_slice[v]) * sqrt(dt)`,
/// matching TVB's convention where `nsig` is the noise dispersion coefficient D.
pub fn generate_noise_per_var<B: Backend>(
    shape: [usize; 2],
    nsig_slice: &[f32],
    dt: f32,
    device: &B::Device,
) -> Tensor<B, 2> {
    let nrows = shape[0];
    let ncols = shape[1];
    let normal = Normal::new(0.0f64, 1.0).unwrap();
    let sqrt_dt = dt.sqrt();

    // Sample in TVB C-order (v outer, row=n*nmodes+m inner) and place into
    // hyburn's [nrows, ncols] C-order tensor at data[row*ncols + col].
    let mut data = vec![0.0f32; nrows * ncols];
    for col in 0..ncols {
        let scale = (2.0f32 * nsig_slice[col]).sqrt() * sqrt_dt;
        for row in 0..nrows {
            let raw: f64 = SEED_RNG.with(|seeding| {
                match *seeding.borrow_mut() {
                    Some(ref mut rng) => normal.sample(rng),
                    None => normal.sample(&mut thread_rng()),
                }
            });
            data[row * ncols + col] = raw as f32 * scale;
        }
    }
    Tensor::from_floats(
        TensorData::new::<f32, Vec<usize>>(data, vec![nrows, ncols]),
        device,
    )
}

/// Stochastic Euler-Maruyama step with per-variable noise.
///
/// `state_new = state + dt * dfun(state, coupling) + Z`
/// where Z[v, n] ~ N(0, 2 * nsig[v] * dt) per variable column v,
/// matching TVB's convention where `nsig` is the noise dispersion D.
pub fn euler_stochastic_step<B: Backend>(
    state: Tensor<B, 2>,
    coupling: Tensor<B, 2>,
    dt: f32,
    nsig: &[f32],
    dfun: impl Fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
    clamp: impl Fn(&mut Tensor<B, 2>),
) -> Tensor<B, 2> {
    let dims = state.shape().dims;
    let noise = generate_noise_per_var::<B>([dims[0], dims[1]], nsig, dt, &state.device());
    let d = dfun(state.clone(), coupling);
    let mut new_state = state + d.mul_scalar(dt) + noise;
    clamp(&mut new_state);
    new_state
}

/// Stochastic Heun step (weak order 1.0) with per-variable noise.
///
/// Uses the same noise for predictor and corrector.
/// Noise per variable: Z[v, n] ~ N(0, 2 * nsig[v] * dt), matching TVB.
pub fn heun_stochastic_step<B: Backend>(
    state: Tensor<B, 2>,
    coupling: Tensor<B, 2>,
    dt: f32,
    nsig: &[f32],
    dfun: impl Fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
    clamp: impl Fn(&mut Tensor<B, 2>),
) -> Tensor<B, 2> {
    let dims = state.shape().dims;
    let noise = generate_noise_per_var::<B>([dims[0], dims[1]], nsig, dt, &state.device());

    let d0 = dfun(state.clone(), coupling.clone());
    let mut predictor = state.clone() + d0.clone().mul_scalar(dt) + noise.clone();
    clamp(&mut predictor);

    let d1 = dfun(predictor, coupling);
    let mut corrector = state + (d0 + d1).mul_scalar(dt * 0.5) + noise;
    clamp(&mut corrector);
    corrector
}

/// Deterministic RK4 (4th-order Runge-Kutta) step with clamping.
pub fn rk4_step<B: Backend>(
    state: Tensor<B, 2>,
    coupling: Tensor<B, 2>,
    dt: f32,
    dfun: impl Fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
    clamp: impl Fn(&mut Tensor<B, 2>),
) -> Tensor<B, 2> {
    let k1 = dfun(state.clone(), coupling.clone());
    let mut k2_state = state.clone() + k1.clone().mul_scalar(dt / 2.0);
    clamp(&mut k2_state);
    let k2 = dfun(k2_state, coupling.clone());
    let mut k3_state = state.clone() + k2.clone().mul_scalar(dt / 2.0);
    clamp(&mut k3_state);
    let k3 = dfun(k3_state, coupling.clone());
    let mut k4_state = state.clone() + k3.clone().mul_scalar(dt);
    clamp(&mut k4_state);
    let k4 = dfun(k4_state, coupling);
    let mut new_state = state + (k1 + k2.mul_scalar(2.0) + k3.mul_scalar(2.0) + k4).mul_scalar(dt / 6.0);
    clamp(&mut new_state);
    new_state
}

/// Stochastic RK4 step: deterministic RK4 core + per-variable additive noise.
///
/// `state_new = rk4_result + Z`
/// where Z[v, n] ~ N(0, 2 * nsig[v] * dt) per variable column v,
/// matching TVB's convention where `nsig` is the noise dispersion D.
pub fn rk4_stochastic_step<B: Backend>(
    state: Tensor<B, 2>,
    coupling: Tensor<B, 2>,
    dt: f32,
    nsig: &[f32],
    dfun: impl Fn(Tensor<B, 2>, Tensor<B, 2>) -> Tensor<B, 2>,
    clamp: impl Fn(&mut Tensor<B, 2>),
) -> Tensor<B, 2> {
    let mut result = rk4_step(state, coupling, dt, dfun, clamp);
    let dims = result.shape().dims;
    let noise = generate_noise_per_var::<B>([dims[0], dims[1]], nsig, dt, &result.device());
    result = result + noise;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::tensor::{Tensor, TensorData};

    type B = NdArray<f32>;

    fn g2do_dfun(state: Tensor<B, 2>, coupling: Tensor<B, 2>) -> Tensor<B, 2> {
        let tau = state.clone().narrow(1, 0, 1).squeeze::<1>(1);
        let x = state.clone().narrow(1, 1, 1).squeeze::<1>(1);
        let c0 = coupling.clone().narrow(1, 0, 1).squeeze::<1>(1);
        let d_tau = x.clone().mul_scalar(1.0 / 0.001);
        let d_x = (tau.neg().mul_scalar(1.0).add(c0).sub(x.clone().mul_scalar(0.001))).mul_scalar(1.0 / 0.001);
        Tensor::cat(vec![d_tau.unsqueeze_dim::<2>(1), d_x.unsqueeze_dim::<2>(1)], 1)
    }

    fn no_clamp(_s: &mut Tensor<B, 2>) {}

    #[test]
    fn test_rk4_different_from_euler_and_heun() {
        let device: <B as Backend>::Device = Default::default();
        let state = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![0.1_f32, 0.5, 0.2, 0.4], vec![2, 2]),
            &device,
        );
        let coupling = Tensor::<B, 2>::zeros([2, 1], &device);
        let dt = 0.1_f32;

        let euler_result = euler_step(state.clone(), coupling.clone(), dt, g2do_dfun, no_clamp);
        let heun_result = heun_step(state.clone(), coupling.clone(), dt, g2do_dfun, no_clamp);
        let rk4_result = rk4_step(state, coupling, dt, g2do_dfun, no_clamp);

        let (euler_flat, _) = crate::io::tensor_to_flat_f32::<B, 2>(euler_result);
        let (heun_flat, _) = crate::io::tensor_to_flat_f32::<B, 2>(heun_result);
        let (rk4_flat, _) = crate::io::tensor_to_flat_f32::<B, 2>(rk4_result);

        for v in &euler_flat { assert!(v.is_finite(), "Euler produced NaN/Inf"); }
        for v in &heun_flat { assert!(v.is_finite(), "Heun produced NaN/Inf"); }
        for v in &rk4_flat { assert!(v.is_finite(), "RK4 produced NaN/Inf"); }

        let euler_heun_diff: f32 = euler_flat.iter().zip(heun_flat.iter()).map(|(a, b)| (a - b).abs()).sum();
        let euler_rk4_diff: f32 = euler_flat.iter().zip(rk4_flat.iter()).map(|(a, b)| (a - b).abs()).sum();
        let heun_rk4_diff: f32 = heun_flat.iter().zip(rk4_flat.iter()).map(|(a, b)| (a - b).abs()).sum();

        assert!(euler_rk4_diff > 1e-8, "RK4 should differ from Euler");
        assert!(heun_rk4_diff > 1e-8, "RK4 should differ from Heun");
    }

    #[test]
    fn test_rk4_stochastic_produces_different_results() {
        let device: <B as Backend>::Device = Default::default();
        let nsig = vec![0.1_f32, 0.2];
        let dt = 0.1_f32;

        let mut results = Vec::new();
        for _ in 0..2 {
            let state = Tensor::<B, 2>::from_floats(
                TensorData::new::<f32, Vec<usize>>(vec![0.1_f32, 0.5, 0.2, 0.4, 0.3, 0.1, 0.0, 0.2], vec![4, 2]),
                &device,
            );
            let coupling = Tensor::<B, 2>::zeros([4, 1], &device);
            let result = rk4_stochastic_step(state, coupling, dt, &nsig, g2do_dfun, |s| *s = s.clone());
            let (flat, _) = crate::io::tensor_to_flat_f32::<B, 2>(result);
            for v in &flat { assert!(v.is_finite(), "RK4 stochastic produced NaN/Inf"); }
            results.push(flat);
        }

        let diff: f32 = results[0].iter().zip(results[1].iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-6, "RK4 stochastic should differ between runs (stochastic), got diff={}", diff);
    }

    #[test]
    fn test_per_variable_noise_different_scales() {
        let device: <B as Backend>::Device = Default::default();
        let nsig_per_var = vec![0.0_f32, 0.1];

        let noise1 = generate_noise_per_var::<B>([100, 2], &nsig_per_var, 0.1, &device);
        let noise2 = generate_noise_per_var::<B>([100, 2], &nsig_per_var, 0.1, &device);

        let (flat1, _) = crate::io::tensor_to_flat_f32::<B, 2>(noise1);
        let (flat2, _) = crate::io::tensor_to_flat_f32::<B, 2>(noise2);

        let var0_noise: Vec<f32> = flat1.iter().step_by(2).copied().collect();
        let var1_noise: Vec<f32> = flat1.iter().skip(1).step_by(2).copied().collect();

        assert!(var0_noise.iter().all(|x| x.abs() < 1e-10), "Variable 0 should have zero noise (nsig=0)");

        let var1_mean: f32 = var1_noise.iter().copied().sum::<f32>() / var1_noise.len() as f32;
        assert!(var1_mean.abs() < 0.1, "Variable 1 noise should have near-zero mean");

        let diff: f32 = flat1.iter().zip(flat2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01, "Random noise should differ between calls");
    }

    #[test]
    fn test_integrator_kind_from_str() {
        assert_eq!("heun".parse::<IntegratorKind>().unwrap(), IntegratorKind::Heun);
        assert_eq!("euler".parse::<IntegratorKind>().unwrap(), IntegratorKind::Euler);
        assert_eq!("euler_stochastic".parse::<IntegratorKind>().unwrap(), IntegratorKind::EulerStochastic);
        assert_eq!("heun_stochastic".parse::<IntegratorKind>().unwrap(), IntegratorKind::HeunStochastic);
        assert_eq!("rk4".parse::<IntegratorKind>().unwrap(), IntegratorKind::Rk4);
        assert_eq!("rk4_stochastic".parse::<IntegratorKind>().unwrap(), IntegratorKind::Rk4Stochastic);
        assert!("unknown".parse::<IntegratorKind>().is_err());
    }

    #[test]
    fn test_integrator_kind_display() {
        assert_eq!(format!("{}", IntegratorKind::Heun), "heun");
        assert_eq!(format!("{}", IntegratorKind::Euler), "euler");
        assert_eq!(format!("{}", IntegratorKind::EulerStochastic), "euler_stochastic");
        assert_eq!(format!("{}", IntegratorKind::HeunStochastic), "heun_stochastic");
        assert_eq!(format!("{}", IntegratorKind::Rk4), "rk4");
        assert_eq!(format!("{}", IntegratorKind::Rk4Stochastic), "rk4_stochastic");
    }
}
