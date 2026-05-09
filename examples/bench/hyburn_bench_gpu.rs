#!/usr/bin/env rust
//! Benchmark: Burn GPU backends (WGPU/CUDA) for 3-subnet sweep.
//!
//! Compares: NdArray Rayon vs WGPU serial vs CUDA serial
//!
//! Usage:
//!   cargo run --release --bin hyburn-bench-gpu --features wgpu
//!   cargo run --release --bin hyburn-bench-gpu --features cuda
//!   cargo run --release --bin hyburn-bench-gpu --features wgpu,cuda

use std::time::Instant;

use burn::prelude::Backend;

use hyburn::config::{
    InitialStateConfig, NetworkConfig, NsigConfig, ProjectionConfig, SimConfig,
    SubnetworkConfig, WeightsConfig,
};
use hyburn::model::g2do::g2do_default_params;
use hyburn::model::jansen_rit::jansen_rit_default_params;
use hyburn::model::wilson_cowan::wilson_cowan_default_params;

fn make_3subnet_config(nnodes: usize) -> SimConfig {
    SimConfig {
        sim_length: 1000.0,
        dt: 0.1,
        network: NetworkConfig {
            subnetworks: vec![
                SubnetworkConfig {
                    model: "Generic2dOscillator".to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.0f32; 2 * nnodes]),
                    params: g2do_default_params(),
                },
                SubnetworkConfig {
                    model: "JansenRit".to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.0f32; 6 * nnodes]),
                    params: jansen_rit_default_params(),
                },
                SubnetworkConfig {
                    model: "WilsonCowan".to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.0f32; 2 * nnodes]),
                    params: wilson_cowan_default_params(),
                },
            ],
            projections: vec![
                ProjectionConfig {
                    src: 0, tgt: 1,
                    conn_type: "all_to_all".to_string(),
                    weights: WeightsConfig::Scalar(0.01),
                    delays: vec![],
                    coupling_fn: "Linear".to_string(),
                    coupling_params: vec![0.01],
                    cvar_map: "0:0".to_string(),
                },
                ProjectionConfig {
                    src: 1, tgt: 2,
                    conn_type: "all_to_all".to_string(),
                    weights: WeightsConfig::Scalar(0.01),
                    delays: vec![],
                    coupling_fn: "Linear".to_string(),
                    coupling_params: vec![0.01],
                    cvar_map: "0:0".to_string(),
                },
                ProjectionConfig {
                    src: 2, tgt: 0,
                    conn_type: "all_to_all".to_string(),
                    weights: WeightsConfig::Scalar(0.01),
                    delays: vec![],
                    coupling_fn: "Linear".to_string(),
                    coupling_params: vec![0.01],
                    cvar_map: "0:0".to_string(),
                },
            ],
        },
        integrator: hyburn::engine::IntegratorKind::Heun,
        monitors: vec![],
        stimuli: vec![],
        nsig: NsigConfig::Scalar(0.0),
            backend: "ndarray".to_string(),
    }
}

fn sweep_values(n: usize) -> Vec<f32> {
    (0..n).map(|i| -0.5f32 + i as f32 * (1.0f32 / (n - 1).max(1) as f32)).collect()
}

/// Serial sweep on any backend
#[allow(dead_code)]
fn run_serial_sweep<B: Backend>(cfg: &SimConfig, n_sweep: usize, n_steps: usize, device: B::Device) -> std::time::Duration {
    let values = sweep_values(n_sweep);
    let start = Instant::now();
    for &v in &values {
        let mut c = cfg.clone();
        c.network.subnetworks[0].params[1] = v;
        let mut engine = hyburn::engine::HybridEngine::<B>::from_config(c, device.clone()).unwrap();
        engine.run(n_steps);
    }
    start.elapsed()
}

fn main() {
    let n_sweep = 1024;
    let n_steps = 1000;

    // ===== NdArray baseline =====
    let cfg76 = make_3subnet_config(76);
    let cfg164 = make_3subnet_config(164);

    {
        use burn::backend::ndarray::NdArray;
        type B = NdArray<f32>;
        let n_cores = rayon::current_num_threads();

        println!("============================================================");
        println!("  hyburn GPU Backend Comparison");
        println!("============================================================");
        println!();
        println!("Network: G2DO → JansenRit → WilsonCowan (coupled ring)");
        println!("Sweep: {} points, {} steps, dt=0.1ms", n_sweep, n_steps);
        println!("CPU cores: {}", n_cores);
        println!();

        // Warmup
        let device: <B as Backend>::Device = Default::default();
        let mut c = cfg76.clone();
        c.network.subnetworks[0].params[1] = 0.0;
        let state = burn::tensor::Tensor::<B, 3>::from_data(
            burn::tensor::TensorData::new::<f32, Vec<usize>>(vec![0.0f32; 2*4], vec![2, 4, 1]),
            &device,
        );
        let mut engine = hyburn::engine::HybridEngine::<B>::new(
            state,
            hyburn::engine::EngineModel::G2do { params: g2do_default_params() },
            hyburn::engine::IntegratorKind::Heun, 0.1, 1, device.clone(),
        );
        engine.run(5);

        // Rayon parallel sweep (NdArray)
        println!("--- NdArray Rayon ---");
        let device: <B as Backend>::Device = Default::default();
        for nodes in [76, 164] {
            let cfg = if nodes == 76 { cfg76.clone() } else { cfg164.clone() };
            let values = sweep_values(n_sweep);
            let start = Instant::now();
            let _results = hyburn::engine::sweep::parallel_sweep_from_config::<B>(
                cfg, "subnetworks[0].params[1]", values, n_steps, device.clone(),
            );
            let t = start.elapsed();
            println!("  {} nodes: {:>10.1} ms  ({:.2} ms/point)", nodes, t.as_millis(), t.as_millis() as f64 / n_sweep as f64);
        }
    }

    // ===== WGPU backend =====
    #[cfg(feature = "wgpu")]
    {
        type B = burn_wgpu::Wgpu<f32, i32>;
        println!();
        println!("--- WGPU serial ---");
        let device = burn_wgpu::WgpuDevice::default();

        // Warmup
        {
            let mut c = cfg76.clone();
            c.network.subnetworks[0].params[1] = 0.0;
            let mut engine = hyburn::engine::HybridEngine::<B>::from_config(c, device.clone()).unwrap();
            engine.run(5);
        }

        for nodes in [76, 164] {
            let cfg = if nodes == 76 { cfg76.clone() } else { cfg164.clone() };
            let t = run_serial_sweep::<B>(&cfg, n_sweep, n_steps, device.clone());
            println!("  {} nodes: {:>10.1} ms  ({:.2} ms/point)", nodes, t.as_millis(), t.as_millis() as f64 / n_sweep as f64);
        }
    }

    // ===== CUDA backend =====
    #[cfg(feature = "cuda")]
    {
        type B = burn_cuda::Cuda<f32, i32>;
        println!();
        println!("--- CUDA serial ---");
        let device = burn_cuda::CudaDevice::default();

        // Warmup
        {
            let mut c = cfg76.clone();
            c.network.subnetworks[0].params[1] = 0.0;
            let mut engine = hyburn::engine::HybridEngine::<B>::from_config(c, device.clone()).unwrap();
            engine.run(5);
        }

        for nodes in [76, 164] {
            let cfg = if nodes == 76 { cfg76.clone() } else { cfg164.clone() };
            let t = run_serial_sweep::<B>(&cfg, n_sweep, n_steps, device.clone());
            println!("  {} nodes: {:>10.1} ms  ({:.2} ms/point)", nodes, t.as_millis(), t.as_millis() as f64 / n_sweep as f64);
        }
    }

    println!();
    println!("============================================================");
    println!("  Done");
    println!("============================================================");
}