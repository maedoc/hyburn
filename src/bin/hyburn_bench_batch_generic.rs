#!/usr/bin/env rust
//! Batch-dim generic sweep benchmark: NdArray vs WGPU via BatchHybridEngine.
//!
//! Uses `rayon_batch_sweep` with NdArray baseline, then WGPU for comparison.
//! The same 3-subnet ring (G2DO → JR → WC) as the Python TVB benchmark.

use std::time::Instant;

use hyburn::config::{
    InitialStateConfig, NetworkConfig, NsigConfig, ProjectionConfig, SimConfig,
    SubnetworkConfig, WeightsConfig,
};
use hyburn::engine::batch_engine::SweepParam;
use hyburn::engine::rayon_batch_sweep;
use hyburn::model::g2do::g2do_default_params;
use hyburn::model::jansen_rit::jansen_rit_default_params;
use hyburn::model::wilson_cowan::wilson_cowan_default_params;

fn make_3subnet_config(nnodes: usize) -> SimConfig {
    let g2do_params = g2do_default_params();
    let g2do_init = vec![0.0f32; 2 * nnodes];
    let jr_params = jansen_rit_default_params();
    let jr_init = vec![0.0f32; 6 * nnodes];
    let wc_params = wilson_cowan_default_params();
    let wc_init = vec![0.0f32; 2 * nnodes];

    SimConfig {
        sim_length: 1000.0,
        dt: 0.1,
        network: NetworkConfig {
            subnetworks: vec![
                SubnetworkConfig {
                    model: "Generic2dOscillator".to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(g2do_init),
                    params: g2do_params,
                },
                SubnetworkConfig {
                    model: "JansenRit".to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(jr_init),
                    params: jr_params,
                },
                SubnetworkConfig {
                    model: "WilsonCowan".to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(wc_init),
                    params: wc_params,
                },
            ],
            projections: vec![
                ProjectionConfig {
                    src: 0, tgt: 1,
                    conn_type: "all_to_all".to_string(),
                    weights: WeightsConfig::Scalar(0.01),
                    delays: vec![],
                    tract_lengths: vec![],
                    coupling_fn: "Linear".to_string(),
                    coupling_params: vec![0.01],
                    cvar_map: "0:0".to_string(),
                },
                ProjectionConfig {
                    src: 1, tgt: 2,
                    conn_type: "all_to_all".to_string(),
                    weights: WeightsConfig::Scalar(0.01),
                    delays: vec![],
                    tract_lengths: vec![],
                    coupling_fn: "Linear".to_string(),
                    coupling_params: vec![0.01],
                    cvar_map: "0:0".to_string(),
                },
                ProjectionConfig {
                    src: 2, tgt: 0,
                    conn_type: "all_to_all".to_string(),
                    weights: WeightsConfig::Scalar(0.01),
                    delays: vec![],
                    tract_lengths: vec![],
                    coupling_fn: "Linear".to_string(),
                    coupling_params: vec![0.01],
                    cvar_map: "0:0".to_string(),
                },
            ],
        },
        integrator: hyburn::engine::integrator::IntegratorKind::Heun,
        monitors: vec![],
        stimuli: vec![],
        nsig: NsigConfig::Scalar(0.0),
        noise_mode: Default::default(),
        speed: 3.0,
        backend: "ndarray".to_string(),
    }
}

fn main() {
    println!("============================================================");
    println!("  Hyburn Batch-Dim Generic Sweep — NdArray vs WGPU");
    println!("============================================================");
    println!("Network: G2DO → JansenRit → WilsonCowan (coupled ring)");
    println!("Sweep: 1024 points × 1000 steps (dt=0.1), Heun integrator");
    println!("Rayon shards across threads");
    println!();

    let nnodes = 76;
    let n_sweep = 1024;
    let n_steps = 1000;
    let sweep_param = SweepParam { sub_idx: 0, param_idx: 1 }; // G2DO I_ext

    let config = make_3subnet_config(nnodes);
    let param_values: Vec<f32> = (-512..512).map(|i| i as f32 * 6.0 / 1024.0).collect();

    // --- NdArray baseline ---
    {
        use burn::backend::ndarray::NdArray;
        type B = NdArray<f32>;
        let device: <B as burn::prelude::Backend>::Device = Default::default();

        // Warmup
        let warmup_vals: Vec<f32> = (-5..=5).map(|i| i as f32 * 6.0 / 1024.0).collect();
        let _ = rayon_batch_sweep::<B>(
            config.clone(), sweep_param.clone(),
            &warmup_vals, n_steps, None, device.clone(),
        );

        let t0 = Instant::now();
        let result = rayon_batch_sweep::<B>(
            config.clone(), sweep_param.clone(),
            &param_values, n_steps, None, device.clone(),
        );
        let elapsed = t0.elapsed();
        let ms_point = elapsed.as_millis() as f64 / n_sweep as f64;
        let g2do_mean = result.tavg[0].iter().step_by(2).sum::<f32>()
            / (result.tavg[0].len() / 2) as f32;
        let nan_count = result.tavg[0].iter().filter(|x| x.is_nan()).count();

        println!("--- NdArray (rayon, {} threads) ---", rayon::current_num_threads());
        println!("  Elapsed:    {:>8.0} ms  ({:.2} ms/point)", elapsed.as_millis(), ms_point);
        println!("  G2DO V mean: {:.6}, NaNs: {}", g2do_mean, nan_count);
        println!();
    }

    // --- WGPU ---
    #[cfg(feature = "wgpu")]
    {
        use burn_wgpu::Wgpu;
        type BW = Wgpu<f32, i32>;
        let device = burn_wgpu::WgpuDevice::default();

        // Warmup
        let warmup_vals: Vec<f32> = (-5..=5).map(|i| i as f32 * 6.0 / 1024.0).collect();
        let _ = rayon_batch_sweep::<BW>(
            config.clone(), sweep_param.clone(),
            &warmup_vals, n_steps, None, device.clone(),
        );

        let t0 = Instant::now();
        let result = rayon_batch_sweep::<BW>(
            config.clone(), sweep_param.clone(),
            &param_values, n_steps, None, device.clone(),
        );
        let elapsed = t0.elapsed();
        let ms_point = elapsed.as_millis() as f64 / n_sweep as f64;
        let g2do_mean = result.tavg[0].iter().step_by(2).sum::<f32>()
            / (result.tavg[0].len() / 2) as f32;
        let nan_count = result.tavg[0].iter().filter(|x| x.is_nan()).count();

        println!("--- WGPU (Metal, rayonsharded) ---");
        println!("  Elapsed:    {:>8.0} ms  ({:.2} ms/point)", elapsed.as_millis(), ms_point);
        println!("  G2DO V mean: {:.6}, NaNs: {}", g2do_mean, nan_count);
        println!();
    }
    #[cfg(not(feature = "wgpu"))]
    {
        println!("--- WGPU: not compiled (build with --features wgpu) ---");
    }

    println!("Done.");
}