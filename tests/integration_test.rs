#![cfg(not(target_arch = "wasm32"))]
//! End-to-end integration test for hyburn.
//!
//! Loads a TOML config from a string, runs a short G2DO simulation,
//! and verifies that the final state contains no NaN or Inf values,
//! output shapes match config, and basic numerical properties hold.

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
        let (data, shape) = tensor_to_flat_f32(state.clone());
        for (j, v) in data.iter().enumerate() {
            assert!(
                v.is_finite(),
                "Subnetwork {} state[{}] is non-finite: {}",
                i, j, v
            );
        }
        // Verify state shape matches config: [nvar, nnodes, nmodes]
        let sub = &cfg.network.subnetworks[i];
        assert_eq!(shape[0], sub.nnodes, "Subnetwork {} nnodes mismatch", i);
    }

    // Verify trajectory is finite and has correct length.
    assert!(!engine.trajectory.is_empty(), "Trajectory should not be empty");
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

#[test]
fn test_config_validation_rejects_invalid_dt() {
    use hyburn::engine::integrator::IntegratorKind;
    use hyburn::config::{NetworkConfig, SimConfig};

    let cfg = SimConfig {
        sim_length: 100.0,
        dt: 0.0, // invalid: must be positive
        network: NetworkConfig { subnetworks: vec![], projections: vec![] },
        integrator: IntegratorKind::Heun,
        monitors: vec![],
        stimuli: vec![],
        nsig: crate::config::NsigConfig::Scalar(0.0),
        speed: 3.0,
        backend: "ndarray".to_string(),
    };
    assert!(cfg.validate().is_err(), "dt=0 should fail validation");
}

#[test]
fn test_config_validation_rejects_unknown_model() {
    use hyburn::config::{SimConfig, NetworkConfig, SubnetworkConfig, InitialStateConfig};
    use hyburn::engine::integrator::IntegratorKind;

    let cfg = SimConfig {
        sim_length: 100.0,
        dt: 0.1,
        network: NetworkConfig {
            subnetworks: vec![SubnetworkConfig {
                model: "FakeModel".to_string(),
                nnodes: 2,
                nmodes: 1,
                initial_state: InitialStateConfig::Inline(vec![0.0; 4]),
                params: vec![1.0; 12],
            }],
            projections: vec![],
        },
        integrator: IntegratorKind::Heun,
        monitors: vec![],
        stimuli: vec![],
        nsig: crate::config::NsigConfig::Scalar(0.0),
        speed: 3.0,
        backend: "ndarray".to_string(),
    };
    let err = cfg.validate().unwrap_err();
    let msg = format!("{}", err);
    assert!(msg.contains("Unknown model"), "Should mention unknown model, got: {}", msg);
}

#[test]
fn test_all_models_run_without_crash() {
    // Quick smoke test: every model runs 10 steps without panicking.
    let models: Vec<(&str, usize, usize)> = vec![
        ("Generic2dOscillator", 2, 12),
        ("MontbrioPazoRoxin", 2, 7),
        ("ReducedWongWang", 1, 8),
        ("Kuramoto", 1, 1),
        ("JansenRit", 6, 13),
        ("WilsonCowan", 2, 22),
    ];

    type B = NdArray<f32>;
    let device = Default::default();

    for (name, nvar, nparams) in models {
        let default_params: Vec<f32> = match name {
            "Generic2dOscillator" => hyburn::model::g2do::g2do_default_params(),
            "MontbrioPazoRoxin" => hyburn::model::mpr::mpr_default_params(),
            "Kuramoto" => hyburn::model::kuramoto_model::kuramoto_default_params(),
            _ => vec![0.1; nparams], // fallback: small positive params
        };

        let init = vec![0.0f32; nvar * 2]; // 2 nodes
        let cfg = SimConfig {
            sim_length: 1.0,
            dt: 0.1,
            network: hyburn::config::NetworkConfig {
                subnetworks: vec![hyburn::config::SubnetworkConfig {
                    model: name.to_string(),
                    nnodes: 2,
                    nmodes: 1,
                    initial_state: hyburn::config::InitialStateConfig::Inline(init),
                    params: default_params,
                }],
                projections: vec![],
            },
            integrator: hyburn::engine::integrator::IntegratorKind::Euler,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };
        cfg.validate().unwrap_or_else(|e| panic!("{} config should validate: {}", name, e));

        let mut engine = HybridEngine::<B>::from_config(cfg, device)
            .unwrap_or_else(|e| panic!("{} engine creation failed: {}", name, e));
        engine.run(10);

        // Verify finite final state
        for (si, state) in engine.states.iter().enumerate() {
            let (data, _) = tensor_to_flat_f32(state.clone());
            for (j, v) in data.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "{} subnetwork {} state[{}] non-finite: {}",
                    name, si, j, v
                );
            }
        }
    }
}
