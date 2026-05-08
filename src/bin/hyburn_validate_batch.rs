#!/usr/bin/env rust
//! Validate batch sweep results against Numba CUDA reference.
//!
//! Runs a small sweep (5 points, 100 steps) on NdArray and compares
//! output statistics (mean, std) against expected ranges from the
//! Numba CUDA benchmark.

use burn::backend::ndarray::NdArray;
use burn::prelude::Backend;

fn main() {
    type B = NdArray<f32>;
    let device: <B as Backend>::Device = Default::default();

    let i_ext_values: Vec<f32> = vec![-0.5, -0.25, 0.0, 0.25, 0.5];
    let nnodes = 76;
    let n_steps = 100;
    let dt: f32 = 0.1;
    let w: f32 = 0.01;

    let result = hyburn::engine::sweep_gpu::batch_sweep_3subnet::<B>(
        &i_ext_values, nnodes, n_steps, dt, w, &device,
    );

    println!("=== Batch Sweep Validation ===");
    println!("Sweep points: {}, nnodes: {}, steps: {}", i_ext_values.len(), nnodes, n_steps);
    println!();

    // Check G2DO output (tavg): shape should be [n_sweep * nnodes * 2]
    println!("G2DO tavg: {} values", result.tavg_g2do.len());
    let g2do_mean: f32 = result.tavg_g2do.iter().copied().sum::<f32>() / result.tavg_g2do.len() as f32;
    let g2do_std = {
        let m = g2do_mean;
        let var: f32 = result.tavg_g2do.iter().map(|x| (x - m).powi(2)).sum::<f32>() / result.tavg_g2do.len() as f32;
        var.sqrt()
    };
    println!("  mean: {:.6}, std: {:.6}", g2do_mean, g2do_std);
    println!("  min: {:.6}, max: {:.6}", 
        result.tavg_g2do.iter().copied().fold(f32::INFINITY, f32::min),
        result.tavg_g2do.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    println!("  NaN: {}", result.tavg_g2do.iter().filter(|x| x.is_nan()).count());
    println!();

    // Check JR output
    println!("JR tavg: {} values", result.tavg_jr.len());
    let jr_mean: f32 = result.tavg_jr.iter().copied().sum::<f32>() / result.tavg_jr.len() as f32;
    let jr_std = {
        let m = jr_mean;
        let var: f32 = result.tavg_jr.iter().map(|x| (x - m).powi(2)).sum::<f32>() / result.tavg_jr.len() as f32;
        var.sqrt()
    };
    println!("  mean: {:.6}, std: {:.6}", jr_mean, jr_std);
    println!("  NaN: {}", result.tavg_jr.iter().filter(|x| x.is_nan()).count());
    println!();

    // Check WC output
    println!("WC tavg: {} values", result.tavg_wc.len());
    let wc_mean: f32 = result.tavg_wc.iter().copied().sum::<f32>() / result.tavg_wc.len() as f32;
    let wc_std = {
        let m = wc_mean;
        let var: f32 = result.tavg_wc.iter().map(|x| (x - m).powi(2)).sum::<f32>() / result.tavg_wc.len() as f32;
        var.sqrt()
    };
    println!("  mean: {:.6}, std: {:.6}", wc_mean, wc_std);
    println!("  NaN: {}", result.tavg_wc.iter().filter(|x| x.is_nan()).count());
    println!();

    // Run the same sweep using the existing serial HybridEngine for comparison
    use hyburn::config::{InitialStateConfig, NetworkConfig, ProjectionConfig, SimConfig, SubnetworkConfig, WeightsConfig};
    use hyburn::model::{g2do::g2do_default_params, jansen_rit::jansen_rit_default_params, wilson_cowan::wilson_cowan_default_params};

    let cfg = SimConfig {
        sim_length: 100.0,
        dt: 0.1,
        network: NetworkConfig {
            subnetworks: vec![
                SubnetworkConfig {
                    model: "Generic2dOscillator".to_string(), nnodes, nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.0f32; 2 * nnodes]),
                    params: g2do_default_params(),
                },
                SubnetworkConfig {
                    model: "JansenRit".to_string(), nnodes, nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.0f32; 6 * nnodes]),
                    params: jansen_rit_default_params(),
                },
                SubnetworkConfig {
                    model: "WilsonCowan".to_string(), nnodes, nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.0f32; 2 * nnodes]),
                    params: wilson_cowan_default_params(),
                },
            ],
            projections: vec![
                ProjectionConfig { src: 0, tgt: 1, conn_type: "all_to_all".to_string(), weights: WeightsConfig::Scalar(0.01), delays: vec![], coupling_fn: "Linear".to_string(), coupling_params: vec![0.01], cvar_map: "0:0".to_string() },
                ProjectionConfig { src: 1, tgt: 2, conn_type: "all_to_all".to_string(), weights: WeightsConfig::Scalar(0.01), delays: vec![], coupling_fn: "Linear".to_string(), coupling_params: vec![0.01], cvar_map: "0:0".to_string() },
                ProjectionConfig { src: 2, tgt: 0, conn_type: "all_to_all".to_string(), weights: WeightsConfig::Scalar(0.01), delays: vec![], coupling_fn: "Linear".to_string(), coupling_params: vec![0.01], cvar_map: "0:0".to_string() },
            ],
        },
        integrator: hyburn::engine::IntegratorKind::Heun,
        monitors: vec![],
        stimuli: vec![],
        nsig: 0.0,
            backend: "ndarray".to_string(),
    };

    println!("=== Serial HybridEngine Comparison (I_ext = 0.0) ===");
    let mut c = cfg.clone();
    c.network.subnetworks[0].params[1] = 0.0;
    let mut engine = hyburn::engine::HybridEngine::<B>::from_config(c, device).unwrap();
    engine.run(n_steps);

    // Final state statistics
    for (i, state) in engine.states.iter().enumerate() {
        let (data, _) = hyburn::io::tensor_to_flat_f32::<B, 3>(state.clone());
        let mean: f32 = data.iter().copied().sum::<f32>() / data.len() as f32;
        let std = {
            let m = mean;
            let var: f32 = data.iter().map(|x| (x - m).powi(2)).sum::<f32>() / data.len() as f32;
            var.sqrt()
        };
        let sub_name = match i { 0 => "G2DO", 1 => "JR", 2 => "WC", _ => "?" };
        println!("  {} state: mean={:.6}, std={:.6}, NaN={}", sub_name, mean, std,
            data.iter().filter(|x| x.is_nan()).count());
    }
}