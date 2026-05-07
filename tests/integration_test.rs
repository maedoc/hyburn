//! End-to-end integration test for hyburn.
//!
//! Loads a TOML config from a string, runs a short G2DO simulation,
//! and verifies that the final state contains no NaN or Inf values.

use burn::backend::ndarray::NdArray;
use hyburn::config::SimConfig;
use hyburn::engine::{HybridEngine, ProgressReporter};
use hyburn::io::tensor_to_flat_f32;

#[test]
fn test_g2do_100_steps_end_to_end() {
    let toml_str = r#"
sim_length = 10.0
dt = 0.1
integrator = "heun"
nsig = 0.0

[network]
[[network.subnetworks]]
model = "Generic2dOscillator"
nnodes = 2
params = [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0]
initial_state = [0.0, 0.5, 0.0, 0.5]
"#;

    let cfg: SimConfig = toml::from_str(toml_str).expect("Failed to parse TOML config");
    cfg.validate().expect("Config validation failed");

    type B = NdArray<f32>;
    let device = Default::default();

    let mut engine = HybridEngine::<B>::from_config(cfg.clone(), device)
        .expect("Failed to build engine from config");

    // Attach progress reporter
    engine.progress = Some(ProgressReporter::new(10));

    let n_steps = (cfg.sim_length / cfg.dt) as usize;
    assert_eq!(n_steps, 100, "Expected 100 steps for 10.0ms / 0.1dt");

    engine.run(n_steps);

    // Verify all final states are finite.
    for (i, state) in engine.states.iter().enumerate() {
        let (data, _shape) = tensor_to_flat_f32(state.clone());
        for (j, v) in data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Subnetwork {} state[{}] is non-finite: {}",
                i, j, v
            );
        }
    }

    // Verify trajectory is finite.
    for (i, v) in engine.trajectory.iter().enumerate() {
        assert!(
            v.is_finite(),
            "Trajectory[{}] is non-finite: {}",
            i, v
        );
    }

    // Verify checkpoint round-trip works.
    let dir = tempfile::tempdir().unwrap();
    let ckpt = dir.path().join("test.ckpt");
    engine.checkpoint(ckpt.to_str().unwrap()).expect("Checkpoint failed");

    let mut engine2 = HybridEngine::<B>::from_config(cfg.clone(), device)
        .expect("Failed to build second engine");
    engine2.resume(ckpt.to_str().unwrap()).expect("Resume failed");

    assert_eq!(engine2.step, n_steps);
    let (restored, _) = tensor_to_flat_f32(engine2.states[0].clone());
    let (original, _) = tensor_to_flat_f32(engine.states[0].clone());
    assert_eq!(restored.len(), original.len());
    for (a, b) in restored.iter().zip(original.iter()) {
        assert!((a - b).abs() < 1e-6, "Checkpoint round-trip mismatch: {} vs {}", a, b);
    }
}

#[test]
fn test_euler_stochastic_integration() {
    let toml_str = r#"
sim_length = 5.0
dt = 0.1
integrator = "euler_stochastic"
nsig = 0.01

[network]
[[network.subnetworks]]
model = "Generic2dOscillator"
nnodes = 2
params = [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0]
initial_state = [0.0, 0.5, 0.0, 0.5]
"#;

    let cfg: SimConfig = toml::from_str(toml_str).expect("Failed to parse TOML config");
    cfg.validate().expect("Config validation failed");

    type B = NdArray<f32>;
    let device = Default::default();

    let mut engine = HybridEngine::<B>::from_config(cfg, device)
        .expect("Failed to build engine from config");
    engine.run(50);

    for (i, state) in engine.states.iter().enumerate() {
        let (data, _shape) = tensor_to_flat_f32(state.clone());
        for (j, v) in data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Stochastic subnetwork {} state[{}] is non-finite: {}",
                i, j, v
            );
        }
    }
}
