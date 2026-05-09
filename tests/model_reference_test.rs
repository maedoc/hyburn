#![cfg(not(target_arch = "wasm32"))]
//! Reference validation tests for new neural mass models.
//!
//! For the 6 newly added models (epileptor_codim3, epileptor_codim3_slowmod,
//! epileptor_rs, zerlaut_first, zerlaut_second, kionex), we compare against
//! TVB-generated reference traces.
//!
//! For all 22 models, we run smoke tests that verify finite output.

use burn::backend::ndarray::NdArray;
use hyburn::config::{InitialStateConfig, NetworkConfig, SimConfig, SubnetworkConfig};
use hyburn::engine::integrator::IntegratorKind;
use hyburn::engine::HybridEngine;
use hyburn::io::read_npy_f32;

type B = NdArray<f32>;

fn load_fixture(name: &str) -> (Vec<f32>, Vec<usize>) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    read_npy_f32(&path).unwrap_or_else(|e| panic!("Failed to load fixture {}: {}", name, e))
}

fn assert_allclose(actual: &[f32], expected: &[f32], rtol: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", label);
    let atol = 1e-4f32;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if !a.is_finite() || !e.is_finite() {
            panic!("{}: non-finite at index {}: actual={}, expected={}", label, i, a, e);
        }
        let abs_diff = (a - e).abs();
        if e.abs() < atol {
            assert!(abs_diff < atol, "{}: mismatch at {}: actual={}, expected={}", label, i, a, e);
        } else {
            let rel_err = abs_diff / e.abs();
            assert!(rel_err <= rtol, "{}: mismatch at {}: actual={}, expected={}, rel_err={:.4}", label, i, a, e, rel_err);
        }
    }
}

fn run_model(model: &str, nnodes: usize, params: Vec<f32>, initial_state: Vec<f32>,
             dt: f64, integrator: IntegratorKind, n_steps: usize) -> Vec<f32> {
    let cfg = SimConfig {
        sim_length: dt * n_steps as f64,
        dt,
        integrator,
        nsig: 0.0,
        backend: "ndarray".to_string(),
        network: NetworkConfig {
            subnetworks: vec![SubnetworkConfig {
                model: model.to_string(),
                nnodes,
                nmodes: 1,
                initial_state: InitialStateConfig::Inline(initial_state),
                params,
            }],
            projections: vec![],
        },
        monitors: vec![],
        stimuli: vec![],
    };
    cfg.validate().unwrap_or_else(|e| panic!("{} config validation: {}", model, e));

    let device = Default::default();
    let mut engine = HybridEngine::<B>::from_config(cfg, device)
        .unwrap_or_else(|e| panic!("{} engine creation: {}", model, e));
    engine.run(n_steps);

    let (actual, _shape) = hyburn::io::tensor_to_flat_f32(engine.states[0].clone());
    assert!(actual.iter().all(|v| v.is_finite()), "{}: non-finite final state", model);
    actual
}

fn compare_final_state(model: &str, actual: &[f32], fixture_name: &str, rtol: f32) {
    let (expected_flat, expected_shape) = load_fixture(fixture_name);
    let nnodes = expected_shape[0];
    let step_len = expected_shape[1];
    let expected_final = &expected_flat[(nnodes - 1) * step_len..nnodes * step_len];
    assert_allclose(actual, expected_final, rtol, &format!("{}_final", model));
}

// ---------------------------------------------------------------------------
// Reference trace tests for the 6 newly-added models
// ---------------------------------------------------------------------------

#[test]
fn test_ref_epileptor_codim3() {
    let actual = run_model(
        "EpileptorCodim3", 2,
        hyburn::model::epileptor_codim3::epileptor_codim3_default_params(),
        vec![0.5, 0.5, 0.0, 0.0, 0.1, 0.1],
        0.1, IntegratorKind::Heun, 100,
    );
    compare_final_state("EpileptorCodim3", &actual, "epileptor_codim3_ref.npy", 0.70);
}

#[test]
fn test_ref_epileptor_codim3_slowmod() {
    let actual = run_model(
        "EpileptorCodim3SlowMod", 2,
        hyburn::model::epileptor_codim3_slowmod::epileptor_codim3_slowmod_default_params(),
        vec![0.5, 0.5, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0],
        0.1, IntegratorKind::Heun, 100,
    );
    compare_final_state("EpileptorCodim3SlowMod", &actual, "epileptor_codim3_slowmod_ref.npy", 0.60);
}

#[test]
fn test_ref_epileptor_rs() {
    let actual = run_model(
        "EpileptorRestingState", 2,
        hyburn::model::epileptor_rs::epileptor_rs_default_params(),
        vec![-1.6, -1.6, -12.5, -12.5, 3.8, 3.8, -1.0, -1.0, 0.005, 0.005, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        0.1, IntegratorKind::Heun, 100,
    );
    compare_final_state("EpileptorRestingState", &actual, "epileptor_rs_ref.npy", 0.05);
}

#[test]
fn test_ref_zerlaut_first() {
    let actual = run_model(
        "ZerlautAdaptationFirstOrder", 2,
        hyburn::model::zerlaut_first::zerlaut_first_default_params(),
        vec![0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        0.1, IntegratorKind::Euler, 100,
    );
    compare_final_state("ZerlautAdaptationFirstOrder", &actual, "zerlaut_first_ref.npy", 0.10);
}

#[test]
fn test_ref_zerlaut_second() {
    let actual = run_model(
        "ZerlautAdaptationSecondOrder", 2,
        hyburn::model::zerlaut_second::zerlaut_second_default_params(),
        vec![0.01, 0.01, 0.01, 0.01, 0.0001, 0.0001, 0.0, 0.0, 0.0001, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        0.1, IntegratorKind::Euler, 100,
    );
    compare_final_state("ZerlautAdaptationSecondOrder", &actual, "zerlaut_second_ref.npy", 0.30);
}

#[test]
fn test_ref_kionex() {
    // KIonEx is stiff: uses dt=0.01 with Heun, 1000 steps
    // Reference trace records every 10th step → 100 samples
    let actual = run_model(
        "KIonEx", 2,
        hyburn::model::kionex::kionex_default_params(),
        vec![0.1, 0.1, -65.0, -65.0, 0.5, 0.5, -0.5, -0.5, -2.0, -2.0],
        0.01, IntegratorKind::Heun, 1000,
    );
    compare_final_state("KIonEx", &actual, "kionex_ref.npy", 0.10);
}

// ---------------------------------------------------------------------------
// Smoke tests: verify all 22 new models run without crash and produce finite output
// ---------------------------------------------------------------------------

#[test]
fn test_all_22_new_models_smoke() {
    let models: Vec<(&str, usize, Vec<f32>)> = vec![
        ("Linear", 1, hyburn::model::linear::linear_default_params()),
        ("SupHopf", 2, hyburn::model::sup_hopf::sup_hopf_default_params()),
        ("Hopfield", 2, hyburn::model::hopfield::hopfield_default_params()),
        ("CoombesByrne2D", 2, hyburn::model::coombes_byrne2d::coombes_byrne2d_default_params()),
        ("CoombesByrne", 4, hyburn::model::coombes_byrne::coombes_byrne_default_params()),
        ("GastSchmidtKnoscheSD", 4, hyburn::model::gast_schmidt_knosche_sd::gast_sd_default_params()),
        ("GastSchmidtKnoscheSF", 4, hyburn::model::gast_schmidt_knosche_sf::gast_sf_default_params()),
        ("LarterBreakspear", 3, hyburn::model::larter_breakspear::larter_breakspear_default_params()),
        ("Epileptor2D", 2, hyburn::model::epileptor2d::epileptor2d_default_params()),
        ("Epileptor", 6, hyburn::model::epileptor::epileptor_default_params()),
        ("ReducedWongWangExcInh", 2, hyburn::model::rww_exc_inh::rww_exc_inh_default_params()),
        ("DecoBalancedExcInh", 2, hyburn::model::deco_balanced_exc_inh::deco_balanced_exc_inh_default_params()),
        ("EpileptorCodim3", 3, hyburn::model::epileptor_codim3::epileptor_codim3_default_params()),
        ("EpileptorCodim3SlowMod", 5, hyburn::model::epileptor_codim3_slowmod::epileptor_codim3_slowmod_default_params()),
        ("EpileptorRestingState", 8, hyburn::model::epileptor_rs::epileptor_rs_default_params()),
        ("ZetterbergJansen", 12, hyburn::model::zetterberg_jansen::zetterberg_jansen_default_params()),
        ("ReducedSetFitzHughNagumo", 4, hyburn::model::reduced_fhn::reduced_fhn_default_params()),
        ("ReducedSetHindmarshRose", 6, hyburn::model::reduced_hr::reduced_hr_default_params()),
        ("DumontGutkin", 8, hyburn::model::dumont_gutkin::dumont_gutkin_default_params()),
        ("ZerlautAdaptationFirstOrder", 5, hyburn::model::zerlaut_first::zerlaut_first_default_params()),
        ("ZerlautAdaptationSecondOrder", 8, hyburn::model::zerlaut_second::zerlaut_second_default_params()),
        ("KIonEx", 5, hyburn::model::kionex::kionex_default_params()),
    ];

    let device: <B as burn::prelude::Backend>::Device = Default::default();
    let nnodes = 2usize;

    for (name, nvar, params) in models {
        let init = vec![0.01f32; nvar * nnodes];
        let cfg = SimConfig {
            sim_length: 1.0,
            dt: 0.1,
            backend: "ndarray".to_string(),
            network: NetworkConfig {
                subnetworks: vec![SubnetworkConfig {
                    model: name.to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(init),
                    params,
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Euler,
            monitors: vec![],
            stimuli: vec![],
            nsig: 0.0,
        };
        cfg.validate().unwrap_or_else(|e| panic!("{} config: {}", name, e));

        let mut engine = HybridEngine::<B>::from_config(cfg, device)
            .unwrap_or_else(|e| panic!("{} engine: {}", name, e));
        engine.run(10);

        for (si, state) in engine.states.iter().enumerate() {
            let (data, _) = hyburn::io::tensor_to_flat_f32(state.clone());
            for (j, v) in data.iter().enumerate() {
                assert!(v.is_finite(), "{} sub{} state[{}] non-finite: {}", name, si, j, v);
            }
        }
    }
}
