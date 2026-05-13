//! Coupling function reference tests for hyburn.
//!
//! Hyburn computes coupling as `post_with_target(W @ pre(x_j), x_i)`, matching
//! TVB classic for ALL coupling functions including Kuramoto and Difference.
//!
//! Test strategy:
//! - **Unit tests**: Exact-match coupling function output against reference data
//!   from `ref/coupling_semantics/` (single-step, no simulation, no delay issues).
//! - **E2E tests**: Full simulation with relative tolerance against TVB classic
//!   reference traces from `ref/coupling/` (accounts for 1-step delay difference).
//!
//! Kuramoto and Difference now use correct classic TVB semantics:
//! - Kuramoto: a/N * Σ w_ij * sin(x_j - x_i)  (trig identity via 2-channel pre + post_with_target)
//! - Difference: a * Σ w_ij * (x_j - x_i)      (rowsum preprocessing or diagonal modification)

use burn::backend::ndarray::NdArray;
use burn::tensor::{Tensor, TensorData};
use hyburn::config::{
    InitialStateConfig, NetworkConfig, NsigConfig, ProjectionConfig, SimConfig,
    SubnetworkConfig, WeightsConfig,
};
use hyburn::engine::coupling::{dense_coupling, CouplingFnConfig};
use hyburn::engine::integrator::IntegratorKind;
use hyburn::engine::HybridEngine;
use hyburn::io::read_npy_f32;

type B = NdArray<f32>;

// ==========================================================================
// Helper: load reference .npy from ref/coupling_semantics/
// ==========================================================================

fn load_semantics_npy(name: &str) -> (Vec<f32>, Vec<usize>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("ref")
        .join("coupling_semantics")
        .join(name);
    read_npy_f32(&path).unwrap_or_else(|e| panic!("Failed to load ref/coupling_semantics/{}: {}", name, e))
}

fn load_coupling_npy(name: &str) -> (Vec<f32>, Vec<usize>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("ref")
        .join("coupling")
        .join(name);
    read_npy_f32(&path).unwrap_or_else(|e| panic!("Failed to load ref/coupling/{}: {}", name, e))
}

fn assert_allclose_rel(actual: &[f32], expected: &[f32], rtol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", label);
    let atol_floor = 1e-5f32;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(a.is_finite() && e.is_finite(), "{}: non-finite at {}: actual={}, expected={}", label, i, a, e);
        let abs_diff = (a - e).abs();
        let tol = if e.abs() < atol_floor { atol_floor } else { e.abs() * rtol };
        assert!(abs_diff <= tol, "{}: mismatch at {}: actual={}, expected={}, abs_diff={:.2e}, tol={:.2e}", label, i, a, e, abs_diff, tol);
    }
}

// ==========================================================================
// EXACT-MATCH unit tests: coupling function output on fixed input
// These use reference data from ref/coupling_semantics/ which was computed
// with the CORRECT pipeline: post(W @ pre(x_j))
// ==========================================================================

#[test]
fn test_coupling_unit_linear_b0_exact() {
    // Linear(a=0.004, b=0): post(W @ pre(x_j)) = 0.004 * (W @ x_j)
    let (W_flat, W_shape) = load_semantics_npy("weights.npy");
    let (xj_flat, xj_shape) = load_semantics_npy("x_j.npy");
    let (expected_flat, expected_shape) = load_semantics_npy("Linear_b0_classic.npy");

    let dev = Default::default();
    let W = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(W_flat, W_shape.clone()), &dev,
    );
    let xj = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(xj_flat, xj_shape.clone()), &dev,
    );

    let cfg = CouplingFnConfig::Linear { a: 0.004, b: 0.0 };
    let result = dense_coupling(W, xj, &cfg, None);
    let (actual_flat, actual_shape) = hyburn::io::tensor_to_flat_f32(result);

    assert_eq!(actual_shape, expected_shape, "Linear_b0: shape mismatch");
    assert_allclose_rel(&actual_flat, &expected_flat, 1e-5, "Linear_b0_exact");
}

#[test]
fn test_coupling_unit_linear_bneq0_exact() {
    // Linear(a=0.004, b=0.1): post(W @ pre(x_j)) = 0.004 * (W @ x_j) + 0.1
    let (W_flat, W_shape) = load_semantics_npy("weights.npy");
    let (xj_flat, xj_shape) = load_semantics_npy("x_j.npy");
    let (expected_flat, expected_shape) = load_semantics_npy("Linear_b0.1_classic.npy");

    let dev = Default::default();
    let W = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(W_flat, W_shape.clone()), &dev,
    );
    let xj = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(xj_flat, xj_shape.clone()), &dev,
    );

    let cfg = CouplingFnConfig::Linear { a: 0.004, b: 0.1 };
    let result = dense_coupling(W, xj, &cfg, None);
    let (actual_flat, actual_shape) = hyburn::io::tensor_to_flat_f32(result);

    assert_eq!(actual_shape, expected_shape, "Linear_b0.1: shape mismatch");
    assert_allclose_rel(&actual_flat, &expected_flat, 1e-5, "Linear_b0.1_exact");
}

#[test]
fn test_coupling_unit_sigmoidal_exact() {
    // Sigmoidal(cmin=-1, cmax=1, midpoint=0, a=1, sigma=1)
    let (W_flat, W_shape) = load_semantics_npy("weights.npy");
    let (xj_flat, xj_shape) = load_semantics_npy("x_j.npy");
    let (expected_flat, expected_shape) = load_semantics_npy("Sigmoidal_classic.npy");

    let dev = Default::default();
    let W = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(W_flat, W_shape.clone()), &dev,
    );
    let xj = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(xj_flat, xj_shape.clone()), &dev,
    );

    let cfg = CouplingFnConfig::Sigmoidal { cmin: -1.0, cmax: 1.0, midpoint: 0.0, a: 1.0, sigma: 1.0 };
    let result = dense_coupling(W, xj, &cfg, None);
    let (actual_flat, actual_shape) = hyburn::io::tensor_to_flat_f32(result);

    assert_eq!(actual_shape, expected_shape, "Sigmoidal: shape mismatch");
    assert_allclose_rel(&actual_flat, &expected_flat, 1e-5, "Sigmoidal_exact");
}

#[test]
fn test_coupling_unit_tanh_exact() {
    // HyperbolicTangent(a=1, b=1, midpoint=0, sigma=1)
    let (W_flat, W_shape) = load_semantics_npy("weights.npy");
    let (xj_flat, xj_shape) = load_semantics_npy("x_j.npy");
    let (expected_flat, expected_shape) = load_semantics_npy("Tanh_classic.npy");

    let dev = Default::default();
    let W = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(W_flat, W_shape.clone()), &dev,
    );
    let xj = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(xj_flat, xj_shape.clone()), &dev,
    );

    let cfg = CouplingFnConfig::HyperbolicTangent { a: 1.0, b: 1.0, midpoint: 0.0, sigma: 1.0 };
    let result = dense_coupling(W, xj, &cfg, None);
    let (actual_flat, actual_shape) = hyburn::io::tensor_to_flat_f32(result);

    assert_eq!(actual_shape, expected_shape, "Tanh: shape mismatch");
    assert_allclose_rel(&actual_flat, &expected_flat, 1e-5, "Tanh_exact");
}

#[test]
fn test_coupling_unit_kuramoto_matches_classic() {
    // Kuramoto: a/N * Σ w_ij * sin(x_j - x_i) — classic TVB semantics
    let (W_flat, W_shape) = load_semantics_npy("weights.npy");
    let (xj_flat, xj_shape) = load_semantics_npy("x_j.npy");
    let (xi_flat, xi_shape) = load_semantics_npy("x_i.npy");
    let (expected_flat, expected_shape) = load_semantics_npy("Kuramoto_classic.npy");

    let dev = Default::default();
    let W = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(W_flat, W_shape.clone()), &dev,
    );
    let xj = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(xj_flat, xj_shape.clone()), &dev,
    );
    let xi = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(xi_flat, xi_shape.clone()), &dev,
    );

    let cfg = CouplingFnConfig::Kuramoto { a: 1.0, n_src: 4 };
    let result = dense_coupling(W, xj, &cfg, Some(xi));
    let (actual_flat, actual_shape) = hyburn::io::tensor_to_flat_f32(result);

    assert_eq!(actual_shape, expected_shape, "Kuramoto: shape mismatch");
    assert_allclose_rel(&actual_flat, &expected_flat, 1e-5, "Kuramoto_classic");
}

#[test]
fn test_coupling_unit_difference_matches_classic() {
    // Difference: a * Σ w_ij * (x_j - x_i) — classic TVB semantics
    // For unit test, use rowsums path (non-square preprocessing)
    let (W_flat, W_shape) = load_semantics_npy("weights.npy");
    let (xj_flat, xj_shape) = load_semantics_npy("x_j.npy");
    let (xi_flat, xi_shape) = load_semantics_npy("x_i.npy");
    let (expected_flat, expected_shape) = load_semantics_npy("Difference_classic.npy");

    let dev = Default::default();

    // Compute rowsums from weights before consuming W_flat
    let rowsums: Vec<f32> = (0..W_shape[0])
        .map(|i| {
            (0..W_shape[1])
                .map(|j| W_flat[i * W_shape[1] + j])
                .sum()
        })
        .collect();

    let W = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(W_flat, W_shape.clone()), &dev,
    );
    let xj = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(xj_flat, xj_shape.clone()), &dev,
    );
    let xi = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(xi_flat, xi_shape.clone()), &dev,
    );

    let cfg = CouplingFnConfig::Difference { a: 0.1, rowsums: Some(rowsums) };
    let result = dense_coupling(W, xj, &cfg, Some(xi));
    let (actual_flat, actual_shape) = hyburn::io::tensor_to_flat_f32(result);

    assert_eq!(actual_shape, expected_shape, "Difference: shape mismatch");
    assert_allclose_rel(&actual_flat, &expected_flat, 1e-5, "Difference_classic");
}

#[test]
fn test_coupling_unit_sigmoidal_jr_exact() {
    // SigmoidalJansenRit: pre uses x_j[:,0]-x_j[:,1], no x_i needed → matches classic
    let (W_flat, W_shape) = load_semantics_npy("weights.npy");
    let (xj2_flat, xj2_shape) = load_semantics_npy("x_j_2cvar.npy");
    let (expected_flat, expected_shape) = load_semantics_npy("SigmoidalJansenRit_classic.npy");

    let dev = Default::default();
    let W = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(W_flat, W_shape.clone()), &dev,
    );
    let xj2 = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(xj2_flat, xj2_shape.clone()), &dev,
    );

    let cfg = CouplingFnConfig::SigmoidalJansenRit { a: 1.0, e0: 0.005, r: 0.56, v0: 6.0 };
    let result = dense_coupling(W, xj2, &cfg, None);
    let (actual_flat, actual_shape) = hyburn::io::tensor_to_flat_f32(result);

    // SigmoidalJansenRit reduces ncvar → 1, so result shape is [n_tgt, 1]
    // Reference was saved as [n_tgt] (squeezed). Compare values, not shapes.
    assert_eq!(actual_flat.len(), expected_flat.len(), "SigmoidalJansenRit: value count mismatch");
    assert_allclose_rel(&actual_flat, &expected_flat, 1e-5, "SigmoidalJansenRit_exact");
}

#[test]
fn test_coupling_unit_difference_square_matches_classic() {
    // Difference square matrix: weight preprocessing converts to Linear with modified diagonal.
    // This is only valid when x_j[i] = x_i[i] (self-connection, same nodes), so we use x_j as both.
    let (W_flat, W_shape) = load_semantics_npy("weights.npy");
    let (xj_flat, xj_shape) = load_semantics_npy("x_j.npy");
    let dev = Default::default();

    // Compute classic reference: a * Σ W[i,j] * (x_j[j] - x_j[i]) when x_i = x_j
    let a: f32 = 0.1;
    let mut expected = vec![0.0f32; W_shape[0] * xj_shape[1]];
    for i in 0..W_shape[0] {
        for c in 0..xj_shape[1] {
            let mut sum = 0.0f32;
            for j in 0..W_shape[1] {
                sum += W_flat[i * W_shape[1] + j] * (xj_flat[j * xj_shape[1] + c] - xj_flat[i * xj_shape[1] + c]);
            }
            expected[i * xj_shape[1] + c] = a * sum;
        }
    }

    // Simulate the construction-time weight preprocessing for square Difference:
    // Modify diagonal: W'[i,i] = W[i,i] - rowsum_i, then store as Linear { a, b: 0.0 }
    let mut modified = W_flat.clone();
    for i in 0..W_shape[0] {
        let row_sum: f32 = (0..W_shape[1]).map(|j| W_flat[i * W_shape[1] + j]).sum();
        modified[i * W_shape[1] + i] -= row_sum;
    }
    let W_mod = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(modified, W_shape.clone()), &dev,
    );
    let xj = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(xj_flat, xj_shape.clone()), &dev,
    );

    let cfg = CouplingFnConfig::Linear { a: 0.1, b: 0.0 };
    let result = dense_coupling(W_mod, xj, &cfg, None);
    let (actual_flat, _) = hyburn::io::tensor_to_flat_f32(result);

    assert_allclose_rel(&actual_flat, &expected, 1e-5, "Difference_square_classic");
}

// ==========================================================================
// E2E simulation tests (full simulation with reference comparison)
// Note: TVB classic always has 1-step delay in coupling, while hyburn with
// delay=0 uses current-step state. This causes small numerical differences.
// ==========================================================================

fn run_coupled_sim(
    model: &str,
    nnodes: usize,
    params: Vec<f32>,
    ic: Vec<f32>,
    weights: Vec<Vec<f32>>,
    coupling_fn: &str,
    coupling_params: Vec<f32>,
    cvar_map: &str,
    steps: usize,
    dt: f64,
    sim_length: f64,
) -> (Vec<f32>, Vec<usize>) {
    let cfg = SimConfig {
        sim_length,
        dt,
        integrator: IntegratorKind::Heun,
        nsig: NsigConfig::Scalar(0.0),
        speed: 3.0,
        backend: "ndarray".to_string(),
        network: NetworkConfig {
            subnetworks: vec![SubnetworkConfig {
                model: model.to_string(),
                nnodes,
                nmodes: 1,
                initial_state: InitialStateConfig::Inline(ic),
                params,
            }],
            projections: vec![ProjectionConfig {
                src: 0,
                tgt: 0,
                conn_type: "all_to_all".to_string(),
                weights: WeightsConfig::Dense(weights),
                delays: vec![0u32; nnodes * nnodes],
                tract_lengths: vec![],
                coupling_fn: coupling_fn.to_string(),
                coupling_params,
                cvar_map: cvar_map.to_string(),
            }],
        },
        monitors: vec![],
        stimuli: vec![],
    };
    cfg.validate().unwrap_or_else(|e| panic!("Config validation: {}", e));
    let device = Default::default();
    let mut engine = HybridEngine::<B>::from_config(cfg, device).unwrap();
    engine.run(steps);
    hyburn::io::tensor_to_flat_f32(engine.states[0].clone())
}

fn run_uncoupled_sim(
    model: &str,
    nnodes: usize,
    params: Vec<f32>,
    ic: Vec<f32>,
    steps: usize,
    dt: f64,
    sim_length: f64,
) -> (Vec<f32>, Vec<usize>) {
    let cfg = SimConfig {
        sim_length,
        dt,
        integrator: IntegratorKind::Heun,
        nsig: NsigConfig::Scalar(0.0),
        speed: 3.0,
        backend: "ndarray".to_string(),
        network: NetworkConfig {
            subnetworks: vec![SubnetworkConfig {
                model: model.to_string(),
                nnodes,
                nmodes: 1,
                initial_state: InitialStateConfig::Inline(ic.clone()),
                params: params.clone(),
            }],
            projections: vec![],
        },
        monitors: vec![],
        stimuli: vec![],
    };
    cfg.validate().unwrap_or_else(|e| panic!("Config validation: {}", e));
    let device = Default::default();
    let mut engine = HybridEngine::<B>::from_config(cfg, device).unwrap();
    engine.run(steps);
    hyburn::io::tensor_to_flat_f32(engine.states[0].clone())
}

#[test]
fn test_e2e_weak_coupling_4node() {
    let (expected_flat, expected_shape) = load_coupling_npy("weak_coupling_4node_final.npy");
    let nnodes = 4;
    let mut ic = vec![0.0f32; nnodes];
    ic.extend(vec![0.5f32; nnodes]);
    let (actual_flat, actual_shape) = run_coupled_sim(
        "Generic2dOscillator", nnodes,
        hyburn::model::g2do::g2do_default_params(),
        ic,
        vec![vec![0.01f32; nnodes]; nnodes],
        "Linear", vec![0.001, 0.0], "0:0",
        200, 0.1, 20.0,
    );
    assert_eq!(actual_shape, expected_shape, "Shape mismatch");
    assert_allclose_rel(&actual_flat, &expected_flat, 5e-3, "weak_coupling_4node");
}

#[test]
fn test_e2e_presigmoidal_epileptor() {
    let (expected_flat, expected_shape) = load_coupling_npy("presig_epileptor_final.npy");
    let nnodes = 2;
    let ic = vec![
        -0.5, -0.5, -9.0, -9.0, 3.5, 3.5, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0,
    ];
    let (actual_flat, actual_shape) = run_coupled_sim(
        "Epileptor", nnodes,
        hyburn::model::epileptor::epileptor_default_params(),
        ic,
        vec![vec![0.0f32, 1.0], vec![1.0, 0.0]],
        "PreSigmoidal", vec![1.0, 0.0, 60.0, 1.0, 0.5], "0:0,1:1",
        200, 0.1, 20.0,
    );
    assert_eq!(actual_shape, expected_shape, "Shape mismatch");
    assert_allclose_rel(&actual_flat, &expected_flat, 5e-2, "presig_epileptor");
}

#[test]
fn test_e2e_tanh_g2do() {
    // HyperbolicTangent now matches classic TVB exactly (unit test proves it)
    // E2E test: verify finite output and coupling effect
    let (actual_flat, actual_shape) = run_coupled_sim(
        "Generic2dOscillator", 2,
        hyburn::model::g2do::g2do_default_params(),
        vec![0.0, 0.0, 0.5, 0.5],
        vec![vec![0.0f32, 1.0], vec![1.0, 0.0]],
        "HyperbolicTangent", vec![1.0, 1.0], "0:0",
        200, 0.1, 20.0,
    );
    assert!(actual_flat.iter().all(|v| v.is_finite()), "tanh_g2do: non-finite output");
    assert_eq!(actual_shape, &[2, 2, 1], "tanh_g2do: shape mismatch");

    let (uncoupled_flat, _) = run_uncoupled_sim(
        "Generic2dOscillator", 2,
        hyburn::model::g2do::g2do_default_params(),
        vec![0.0, 0.0, 0.5, 0.5],
        200, 0.1, 20.0,
    );
    let max_diff = actual_flat.iter()
        .zip(uncoupled_flat.iter())
        .map(|(a, u)| (a - u).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff > 1e-3, "tanh_g2do: coupling has no effect (max_diff={})", max_diff);
}

#[test]
fn test_e2e_sigmoidal_jr_produces_finite() {
    let (actual_flat, actual_shape) = run_coupled_sim(
        "JansenRit", 2,
        hyburn::model::jansen_rit::jansen_rit_default_params(),
        vec![0.0f32; 12],
        vec![vec![0.0f32, 1.0], vec![1.0, 0.0]],
        "SigmoidalJansenRit", vec![5.0, 0.28, 0.56, -0.01], "0:0",
        200, 0.1, 20.0,
    );
    assert!(actual_flat.iter().all(|v| v.is_finite()), "sigr_jr: non-finite output");
    assert_eq!(actual_shape, &[6, 2, 1], "sigr_jr: shape mismatch");

    let (uncoupled_flat, _) = run_uncoupled_sim(
        "JansenRit", 2,
        hyburn::model::jansen_rit::jansen_rit_default_params(),
        vec![0.0f32; 12],
        200, 0.1, 20.0,
    );
    let max_diff = actual_flat.iter()
        .zip(uncoupled_flat.iter())
        .map(|(a, u)| (a - u).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff > 1e-3, "sigr_jr: coupling has no effect (max_diff={})", max_diff);
}

#[test]
fn test_e2e_zero_coupling_matches_uncoupled() {
    let (coupled_flat, _) = run_coupled_sim(
        "Generic2dOscillator", 1,
        hyburn::model::g2do::g2do_default_params(),
        vec![0.0, 0.5],
        vec![vec![0.0f32]],
        "Linear", vec![0.01, 0.0], "0:0",
        200, 0.1, 20.0,
    );
    let (uncoupled_flat, _) = run_uncoupled_sim(
        "Generic2dOscillator", 1,
        hyburn::model::g2do::g2do_default_params(),
        vec![0.0, 0.5],
        200, 0.1, 20.0,
    );
    assert_allclose_rel(&coupled_flat, &uncoupled_flat, 1e-6, "zero_coupling_vs_uncoupled");
    assert!(coupled_flat.iter().all(|v| v.is_finite()), "zero_coupling: non-finite output");
}
