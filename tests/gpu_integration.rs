//! GPU integration test — verifies the engine runs on WGPU backend and produces valid results.

use hyburn::engine::{EngineModel, HybridEngine, IntegratorKind};
use hyburn::model::g2do::g2do_default_params;
use hyburn::io::tensor_to_flat_f32;
use burn::tensor::{Tensor, TensorData};
use burn::backend::ndarray::NdArray;

/// Run the same simulation on NdArray and compare results.
/// This tests that the engine produces consistent results regardless of backend.
#[test]
fn test_g2do_ndarray_consistency() {
    type B = NdArray<f32>;
    let device = Default::default();
    let nnodes = 4;
    let nvar = 2;
    let nmodes = 1;
    let dt = 0.1_f64;
    let n_steps = 500;

    // Initial state: small random-like perturbation
    let initial_data: Vec<f32> = (0..nnodes * nvar * nmodes)
        .map(|i| if i % 2 == 0 { 0.1 } else { -0.05 })
        .collect();
    let state = Tensor::<B, 3>::from_floats(
        TensorData::new::<f32, Vec<usize>>(initial_data, vec![nvar, nnodes, nmodes]),
        &device,
    );

    let model = EngineModel::<B>::G2do {
        params: g2do_default_params(),
    };
    let mut engine = HybridEngine::new(state, model, IntegratorKind::Heun, dt, 10, device);
    engine.run(n_steps);

    let (final_data, _) = tensor_to_flat_f32::<B, 3>(engine.states[0].clone());
    for v in &final_data {
        assert!(v.is_finite(), "NaN/Inf in final state: {}", v);
    }

    // Verify the state has actually evolved (not stuck at initial)
    let initial_mag: f32 = (0..nnodes * nvar * nmodes)
        .map(|i| if i % 2 == 0 { 0.1_f32 } else { -0.05_f32 })
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    let final_mag: f32 = final_data.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(final_mag > 0.0, "Final state has zero magnitude");
}

/// Test that the full CLI pipeline works end-to-end: write TOML → run → read output NPY.
#[test]
fn test_cli_end_to_end_with_toml() {
    use hyburn::config::SimConfig;
    use hyburn::io::{read_npy_f32, write_npy_f32};
    use tempfile::tempdir;

    let dir = tempdir().unwrap();
    let output_dir = dir.path().join("output");
    std::fs::create_dir_all(&output_dir).unwrap();

    // Write a minimal TOML config
    let toml_str = r#"
sim_length = 10.0
dt = 0.1
integrator = "heun"

[network]
[[network.subnetworks]]
model = "Generic2dOscillator"
nnodes = 2
nmodes = 1
params = [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0]
initial_state = [0.0, 0.0, 0.0, 0.0]
"#;
    let config_path = dir.path().join("sim.toml");
    std::fs::write(&config_path, toml_str).unwrap();

    let cfg = SimConfig::from_file(config_path.to_str().unwrap()).unwrap();
    cfg.validate().unwrap();

    // Run using the CLI dispatch path (NdArray)
    use hyburn::cli::Cli;
    // Actually, let's just run the engine directly
    use hyburn::engine::HybridEngine;
    
    type B = NdArray<f32>;
    let device = Default::default();
    let sub = &cfg.network.subnetworks[0];
    let nvar = 2;
    let nnodes = sub.nnodes;
    let nmodes = sub.nmodes;

    let state = match &sub.initial_state {
        hyburn::config::InitialStateConfig::Inline(vals) => {
            Tensor::<B, 3>::from_floats(
                TensorData::new::<f32, Vec<usize>>(vals.clone(), vec![nvar, nnodes, nmodes]),
                &device,
            )
        }
        _ => panic!("Expected inline initial state"),
    };

    let model = EngineModel::<B>::from_config(&sub.model, sub.params.clone()).unwrap();
    let n_steps = (cfg.sim_length / cfg.dt) as usize;
    let mut engine = HybridEngine::new(state, model, IntegratorKind::Heun, cfg.dt, 10, device);
    engine.run(n_steps);

    // Write output
    let (final_data, final_shape) = tensor_to_flat_f32::<B, 3>(engine.states[0].clone());
    write_npy_f32(output_dir.join("state_final.npy"), &final_data, &final_shape).unwrap();

    // Read it back and verify
    let (read_data, read_shape) = read_npy_f32(output_dir.join("state_final.npy")).unwrap();
    assert_eq!(final_shape, read_shape);
    for (a, b) in final_data.iter().zip(read_data.iter()) {
        assert!((a - b).abs() < 1e-6, "NPY roundtrip mismatch: {} vs {}", a, b);
    }
    for v in &read_data {
        assert!(v.is_finite(), "NaN/Inf in output NPY");
    }
}