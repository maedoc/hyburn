//! WASM-bindgen tests for the WebEngine API.
//!
//! Run with: wasm-pack test --node --no-default-features --features wasm

use wasm_bindgen_test::*;
use wasm_bindgen::JsValue;

use crate::wasm::{WebEngine, validate_config_json, model_registry_json};

wasm_bindgen_test_configure!(run_in_browser);

fn g2do_config_json() -> String {
    serde_json::json!({
        "sim_length": 100.0,
        "dt": 0.1,
        "integrator": "euler",
        "nsig": 0.0,
        "network": {
            "subnetworks": [{
                "model": "Generic2dOscillator",
                "nnodes": 2,
                "nmodes": 1,
                "params": [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                "initial_state": [0.0, 0.5, 0.0, 0.5]
            }],
            "projections": []
        }
    }).to_string()
}

#[wasm_bindgen_test]
fn test_create_engine_from_json() {
    let engine = WebEngine::from_json(&g2do_config_json())
        .expect("Failed to create engine from JSON");

    let info = engine.info();
    assert_eq!(info.nvar(), 2, "G2DO should have 2 variables");
    assert_eq!(info.nnodes(), 2, "Should have 2 nodes");
    assert_eq!(info.nmodes(), 1, "Should have 1 mode");
}

#[wasm_bindgen_test]
fn test_step_and_trajectory() {
    let mut engine = WebEngine::from_json(&g2do_config_json())
        .expect("Failed to create engine");

    assert_eq!(engine.current_step(), 0);
    assert_eq!(engine.trajectory_len(), 0);

    engine.step();
    assert_eq!(engine.current_step(), 1);
    assert!(engine.trajectory_len() > 0, "Trajectory should have data after stepping");
}

#[wasm_bindgen_test]
fn test_step_n() {
    let mut engine = WebEngine::from_json(&g2do_config_json())
        .expect("Failed to create engine");

    engine.step_n(100);
    assert_eq!(engine.current_step(), 100);
    assert!(engine.trajectory_len() > 0);
}

#[wasm_bindgen_test]
fn test_trajectory_data() {
    let mut engine = WebEngine::from_json(&g2do_config_json())
        .expect("Failed to create engine");

    engine.step_n(10);
    let traj = engine.trajectory();
    // G2DO: 2 vars × 2 nodes × 1 mode = 4 values per step
    // 10 steps × 4 = 40 values
    assert_eq!(traj.length(), 40, "Trajectory should have 40 f32 values");
}

#[wasm_bindgen_test]
fn test_current_state() {
    let mut engine = WebEngine::from_json(&g2do_config_json())
        .expect("Failed to create engine");

    engine.step_n(10);
    let state = engine.current_state();
    assert_eq!(state.length(), 4, "State should have 4 f32 values (2var × 2nodes × 1mode)");
}

#[wasm_bindgen_test]
fn test_validate_config() {
    // Valid config
    let err = validate_config_json(&g2do_config_json());
    assert_eq!(err, "", "Valid config should return empty error string");

    // Invalid config: negative dt
    let bad_json = serde_json::json!({
        "sim_length": 100.0,
        "dt": -0.1,
        "network": {
            "subnetworks": [{
                "model": "Generic2dOscillator",
                "nnodes": 2,
                "nmodes": 1,
                "params": [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                "initial_state": [0.0, 0.5, 0.0, 0.5]
            }],
            "projections": []
        }
    }).to_string();
    let err = validate_config_json(&bad_json);
    assert_ne!(err, "", "Invalid config should return non-empty error string");
}

#[wasm_bindgen_test]
fn test_invalid_model() {
    let bad_json = serde_json::json!({
        "sim_length": 100.0,
        "dt": 0.1,
        "network": {
            "subnetworks": [{
                "model": "NonexistentModel",
                "nnodes": 2,
                "nmodes": 1,
                "params": [1.0],
                "initial_state": [0.0, 0.0]
            }],
            "projections": []
        }
    }).to_string();

    let result = WebEngine::from_json(&bad_json);
    assert!(result.is_err(), "Unknown model should fail");
}

#[wasm_bindgen_test]
fn test_model_registry() {
    let json = model_registry_json();
    let registry: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
    assert!(registry.len() >= 6, "Should have at least 6 models in registry");
    assert!(registry.iter().any(|m| m["name"] == "Generic2dOscillator"));
}

#[wasm_bindgen_test]
fn test_engine_dt() {
    let engine = WebEngine::from_json(&g2do_config_json())
        .expect("Failed to create engine");
    assert!((engine.dt() - 0.1).abs() < 1e-10, "dt should be 0.1");
}

#[wasm_bindgen_test]
fn test_engine_nsig() {
    let engine = WebEngine::from_json(&g2do_config_json())
        .expect("Failed to create engine");
    assert_eq!(engine.nsig(), 0.0, "nsig should be 0.0");
}

#[wasm_bindgen_test]
fn test_engine_integrator() {
    let engine = WebEngine::from_json(&g2do_config_json())
        .expect("Failed to create engine");
    assert_eq!(engine.integrator(), "euler", "Integrator should be euler");
}
