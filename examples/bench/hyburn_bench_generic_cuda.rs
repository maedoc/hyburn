#!/usr/bin/env rust
//! Quick CUDA benchmark: hardcoded vs generic BatchHybridEngine

#[cfg(feature = "cuda")]
use hyburn::config::*;
#[cfg(feature = "cuda")]
use hyburn::engine::IntegratorKind;
#[cfg(feature = "cuda")]
use hyburn::engine::batch_engine::{BatchHybridEngine, SweepParam};
#[cfg(feature = "cuda")]
use hyburn::engine::sweep_gpu;
#[cfg(feature = "cuda")]
use hyburn::model::{g2do, jansen_rit, wilson_cowan};

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This binary requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    let n_sweep_vals = [256, 512, 1024];
    let n_steps = 1000;
    let dt: f32 = 0.1;
    let w: f32 = 0.01;

    type B = burn_cuda::Cuda<f32, i32>;
    let device = burn_cuda::CudaDevice::default();

    // Warmup
    let warmup = vec![-0.5f32, 0.0, 0.5];
    let _ = sweep_gpu::batch_sweep_3subnet::<B>(&warmup, 4, 10, dt, w, &device);
    let mut we = BatchHybridEngine::<B>::from_config(make_config(76, w), 3, device.clone()).unwrap();
    let _ = we.run_sweep(&SweepParam { sub_idx: 0, param_idx: 1 }, &warmup, 10);

    println!("========================================================");
    println!("  CUDA Batch Sweep: Hardcoded vs Generic Engine");
    println!("========================================================");
    println!("Network: G2DO → JR → WC ring, 1000 steps, dt=0.1");
    println!();

    for &nnodes in &[76, 164] {
        for &n_sweep in &n_sweep_vals {
            let i_ext: Vec<f32> = (0..n_sweep)
                .map(|i| -0.5 + i as f32 / (n_sweep - 1).max(1) as f32)
                .collect();

            // Hardcoded
            let r = sweep_gpu::batch_sweep_3subnet::<B>(&i_ext, nnodes, n_steps, dt, w, &device);
            let hardcoded_ms = r.elapsed_ms;

            // Generic
            let mut engine = BatchHybridEngine::<B>::from_config(make_config(nnodes, w), n_sweep, device.clone()).unwrap();
            engine.hybrid_integrator = true;  // Match hardcoded: Heun for G2DO, Euler for JR/WC
            let r2 = engine.run_sweep(&SweepParam { sub_idx: 0, param_idx: 1 }, &i_ext, n_steps);
            let generic_ms = r2.elapsed_ms;

            let ratio = generic_ms / hardcoded_ms;
            println!("{} nodes × {} pts: hardcoded={:.0}ms ({:.2}ms/pt)  generic={:.0}ms ({:.2}ms/pt)  ratio={:.2}x",
                nnodes, n_sweep, hardcoded_ms, hardcoded_ms/n_sweep as f64,
                generic_ms, generic_ms/n_sweep as f64, ratio);
        }
        println!();
    }

    // Single-subnet models
    for (model_name, model, nvar) in [
        ("G2DO", "Generic2dOscillator", 2),
        ("JR", "JansenRit", 6),
        ("WC", "WilsonCowan", 2),
        ("MPR", "MontbrioPazoRoxin", 2),
    ] {
        let n_sweep = 256;
        let i_ext: Vec<f32> = (0..n_sweep)
            .map(|i| -0.5 + i as f32 / (n_sweep - 1) as f32)
            .collect();

        let config = SimConfig {
            sim_length: 10.0,
            dt: 0.1,
            network: NetworkConfig {
                subnetworks: vec![SubnetworkConfig {
                    model: model.to_string(),
                    nnodes: 76, nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.0f32; nvar * 76]),
                    params: match model {
                        "Generic2dOscillator" => g2do::g2do_default_params(),
                        "JansenRit" => jansen_rit::jansen_rit_default_params(),
                        "WilsonCowan" => wilson_cowan::wilson_cowan_default_params(),
                        "MontbrioPazoRoxin" => vec![1.0, 1.0, -5.0, 15.0, 0.0, 1.0, 0.0],
                        _ => vec![],
                    },
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Euler,
            monitors: vec![],
            stimuli: vec![],
            nsig: NsigConfig::Scalar(0.0),
            backend: "ndarray".to_string(),
        };

        let mut engine = BatchHybridEngine::<B>::from_config(config, n_sweep, device.clone()).unwrap();
        let param = SweepParam { sub_idx: 0, param_idx: if model == "MontbrioPazoRoxin" { 2 } else { 1 } };
        let r = engine.run_sweep(&param, &i_ext, n_steps);
        println!("{} single-model: {:.0}ms ({:.2}ms/pt)", model_name, r.elapsed_ms, r.elapsed_ms / n_sweep as f64);
    }

    println!();
    println!("========================================================");
    println!("  Reference: Numba CUDA = 0.40 ms/pt (76 nodes, 1024 pts)");
    println!("========================================================");
}

#[cfg(feature = "cuda")]
fn make_config(nnodes: usize, w: f32) -> SimConfig {
    SimConfig {
        sim_length: 100.0, dt: 0.1,
        network: NetworkConfig {
            subnetworks: vec![
                SubnetworkConfig { model: "Generic2dOscillator".to_string(), nnodes, nmodes: 1, initial_state: InitialStateConfig::Inline(vec![0.0f32; 2*nnodes]), params: g2do::g2do_default_params() },
                SubnetworkConfig { model: "JansenRit".to_string(), nnodes, nmodes: 1, initial_state: InitialStateConfig::Inline(vec![0.0f32; 6*nnodes]), params: jansen_rit::jansen_rit_default_params() },
                SubnetworkConfig { model: "WilsonCowan".to_string(), nnodes, nmodes: 1, initial_state: InitialStateConfig::Inline(vec![0.0f32; 2*nnodes]), params: wilson_cowan::wilson_cowan_default_params() },
            ],
            projections: vec![
                ProjectionConfig { src: 0, tgt: 1, conn_type: "all_to_all".to_string(), weights: WeightsConfig::Scalar(w), coupling_fn: "Linear".to_string(), coupling_params: vec![1.0], cvar_map: "0:0".to_string(), delays: vec![0] },
                ProjectionConfig { src: 1, tgt: 2, conn_type: "all_to_all".to_string(), weights: WeightsConfig::Scalar(w), coupling_fn: "Linear".to_string(), coupling_params: vec![1.0], cvar_map: "0:0".to_string(), delays: vec![0] },
                ProjectionConfig { src: 2, tgt: 0, conn_type: "all_to_all".to_string(), weights: WeightsConfig::Scalar(w), coupling_fn: "Linear".to_string(), coupling_params: vec![1.0], cvar_map: "0:0".to_string(), delays: vec![0] },
            ],
        },
        integrator: IntegratorKind::Heun, monitors: vec![], stimuli: vec![], nsig: NsigConfig::Scalar(0.0),
    }
}