#!/usr/bin/env rust
//! Mini benchmark: single sweep point on each backend to measure per-step cost.

use std::time::Instant;
use burn::prelude::Backend;
use hyburn::config::{InitialStateConfig, NetworkConfig, NsigConfig, ProjectionConfig, SimConfig, SubnetworkConfig, WeightsConfig};
use hyburn::engine::integrator::IntegratorKind;
use hyburn::model::g2do::g2do_default_params;
use hyburn::model::jansen_rit::jansen_rit_default_params;
use hyburn::model::wilson_cowan::wilson_cowan_default_params;

fn make_3subnet_config(nnodes: usize) -> SimConfig {
    SimConfig {
        sim_length: 1000.0, dt: 0.1,
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
        integrator: IntegratorKind::Heun, monitors: vec![], stimuli: vec![], nsig: NsigConfig::Scalar(0.0),
    }
}

fn main() {
    let n_steps = 1000;
    let n_points = 20;  // Small sweep for quick benchmark

    // NdArray
    {
        use burn::backend::ndarray::NdArray;
        type B = NdArray<f32>;
        let device: <B as Backend>::Device = Default::default();
        let cfg = make_3subnet_config(76);

        let mut c = cfg.clone();
        c.network.subnetworks[0].params[1] = 0.0;
        let mut engine = hyburn::engine::HybridEngine::<B>::from_config(c, device.clone()).unwrap();
        engine.run(5); // warmup

        println!("--- NdArray: {} points × {} steps × 76 nodes ---", n_points, n_steps);
        let start = Instant::now();
        for i in 0..n_points {
            let mut c = cfg.clone();
            c.network.subnetworks[0].params[1] = -0.5 + i as f32 / (n_points - 1) as f32;
            let mut engine = hyburn::engine::HybridEngine::<B>::from_config(c, device.clone()).unwrap();
            engine.run(n_steps);
        }
        let t = start.elapsed();
        println!("  NdArray serial: {:>8.1} ms  ({:.2} ms/point)", t.as_millis(), t.as_millis() as f64 / n_points as f64);
    }

    // WGPU
    #[cfg(feature = "wgpu")]
    {
        type B = burn_wgpu::Wgpu<f32, i32>;
        let device = burn_wgpu::WgpuDevice::default();
        let cfg = make_3subnet_config(76);

        println!("--- WGPU: {} points × {} steps × 76 nodes ---", n_points, n_steps);
        let mut c = cfg.clone();
        c.network.subnetworks[0].params[1] = 0.0;
        let mut engine = hyburn::engine::HybridEngine::<B>::from_config(c, device.clone()).unwrap();
        engine.run(5); // warmup (compile kernels)

        let start = Instant::now();
        for i in 0..n_points {
            let mut c = cfg.clone();
            c.network.subnetworks[0].params[1] = -0.5 + i as f32 / (n_points - 1) as f32;
            let mut engine = hyburn::engine::HybridEngine::<B>::from_config(c, device.clone()).unwrap();
            engine.run(n_steps);
        }
        let t = start.elapsed();
        println!("  WGPU serial:   {:>8.1} ms  ({:.2} ms/point)", t.as_millis(), t.as_millis() as f64 / n_points as f64);
    }

    // CUDA
    #[cfg(feature = "cuda")]
    {
        type B = burn_cuda::Cuda<f32, i32>;
        let device = burn_cuda::CudaDevice::default();
        let cfg = make_3subnet_config(76);

        println!("--- CUDA: {} points × {} steps × 76 nodes ---", n_points, n_steps);
        let mut c = cfg.clone();
        c.network.subnetworks[0].params[1] = 0.0;
        let mut engine = hyburn::engine::HybridEngine::<B>::from_config(c, device.clone()).unwrap();
        engine.run(5); // warmup

        let start = Instant::now();
        for i in 0..n_points {
            let mut c = cfg.clone();
            c.network.subnetworks[0].params[1] = -0.5 + i as f32 / (n_points - 1) as f32;
            let mut engine = hyburn::engine::HybridEngine::<B>::from_config(c, device.clone()).unwrap();
            engine.run(n_steps);
        }
        let t = start.elapsed();
        println!("  CUDA serial:   {:>8.1} ms  ({:.2} ms/point)", t.as_millis(), t.as_millis() as f64 / n_points as f64);
    }

    println!();
    println!("NOTE: GPU serial sweep is slow because each step requires many tiny");
    println!("kernel dispatches. Numba CUDA avoids this by compiling the entire");
    println!("inner loop into a single GPU kernel. Burn GPU backends need a");
    println!("batch-dim approach or custom WGSL/CubeCL kernels to compete.");
}
