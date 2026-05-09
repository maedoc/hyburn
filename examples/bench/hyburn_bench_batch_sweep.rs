#!/usr/bin/env rust
//! Benchmark: Batch-dim GPU sweep for 3-subnet ring model.
//!
//! Compares Numba CUDA baseline vs:
//!   1. NdArray serial (baseline)
//!   2. NdArray batch (batch-dim via Burn NdArray backend)
//!   3. WGPU batch (batch-dim via Burn WGPU backend)
//!   4. CUDA batch (batch-dim via Burn CUDA backend)
//!
//! Usage:
//!   cargo run --release --bin hyburn-bench-batch-sweep
//!   cargo run --release --bin hyburn-bench-batch-sweep --features wgpu
//!   cargo run --release --bin hyburn-bench-batch-sweep --features cuda
//!   cargo run --release --bin hyburn-bench-batch-sweep --features wgpu,cuda


fn main() {
    let n_sweep = 1024;
    let n_steps = 1000;
    let dt: f32 = 0.1;
    let w: f32 = 0.01;

    let i_ext_values: Vec<f32> = (0..n_sweep)
        .map(|i| -0.5 + i as f32 * (1.0f32 / (n_sweep - 1).max(1) as f32))
        .collect();

    println!("============================================================");
    println!("  Hyburn Batch-Dim GPU Sweep Benchmark");
    println!("============================================================");
    println!("Network: G2DO → JansenRit → WilsonCowan (coupled ring)");
    println!("Coupling: All-to-all scalar, weight={}", w);
    println!("Sweep: {} points × {} steps (dt={})", n_sweep, n_steps, dt);
    println!();

    // ===== NdArray batch =====
    {
        use burn::backend::ndarray::NdArray;
        type B = NdArray<f32>;
        let device: <B as burn::prelude::Backend>::Device = Default::default();

        for &nnodes in &[76, 164] {
            println!("--- NdArray hardcoded, {} nodes ---", nnodes);
            // Warmup
            let warmup_vals = vec![-0.5f32, 0.0, 0.5];
            let _ = hyburn::engine::sweep_gpu::batch_sweep_3subnet::<B>(
                &warmup_vals, 4, 10, dt, w, &device,
            );

            let result = hyburn::engine::sweep_gpu::batch_sweep_3subnet::<B>(
                &i_ext_values, nnodes, n_steps, dt, w, &device,
            );
            let per_point = result.elapsed_ms / n_sweep as f64;
            println!("  NdArray hardcoded: {:>8.1} ms  ({:.2} ms/point)", result.elapsed_ms, per_point);
            // Sanity check
            let g2do_mean: f32 = result.tavg_g2do.iter().copied().sum::<f32>() / result.tavg_g2do.len() as f32;
            let nan_count = result.tavg_g2do.iter().chain(result.tavg_jr.iter()).chain(result.tavg_wc.iter())
                .filter(|x| x.is_nan()).count();
            println!("  G2DO V mean: {:.4}, NaN count: {}", g2do_mean, nan_count);
            println!();
        }
    }

    // ===== NdArray generic BatchHybridEngine =====
    {
        use burn::backend::ndarray::NdArray;
        type B = NdArray<f32>;
        use hyburn::config::{InitialStateConfig, NetworkConfig, NsigConfig, ProjectionConfig, SimConfig, SubnetworkConfig, WeightsConfig};
        use hyburn::engine::IntegratorKind;
        use hyburn::engine::batch_engine::{BatchHybridEngine, SweepParam};
        let device: <B as burn::prelude::Backend>::Device = Default::default();

        for &nnodes in &[76, 164] {
            println!("--- NdArray generic BatchHybridEngine, {} nodes ---", nnodes);

            let config = SimConfig {
                sim_length: n_steps as f64 * dt as f64,
                dt: dt as f64,
                network: NetworkConfig {
                    subnetworks: vec![
                        SubnetworkConfig {
                            model: "Generic2dOscillator".to_string(),
                            nnodes,
                            nmodes: 1,
                            initial_state: InitialStateConfig::Inline(vec![0.0f32; 2 * nnodes]),
                            params: hyburn::model::g2do::g2do_default_params(),
                        },
                        SubnetworkConfig {
                            model: "JansenRit".to_string(),
                            nnodes,
                            nmodes: 1,
                            initial_state: InitialStateConfig::Inline(vec![0.0f32; 6 * nnodes]),
                            params: hyburn::model::jansen_rit::jansen_rit_default_params(),
                        },
                        SubnetworkConfig {
                            model: "WilsonCowan".to_string(),
                            nnodes,
                            nmodes: 1,
                            initial_state: InitialStateConfig::Inline(vec![0.0f32; 2 * nnodes]),
                            params: hyburn::model::wilson_cowan::wilson_cowan_default_params(),
                        },
                    ],
                    projections: vec![
                        ProjectionConfig {
                            src: 0, tgt: 1,
                            conn_type: "all_to_all".to_string(),
                            weights: WeightsConfig::Scalar(w),
                            coupling_fn: "Linear".to_string(),
                            coupling_params: vec![1.0],
                            cvar_map: "0:0".to_string(),
                            delays: vec![0],
                            tract_lengths: vec![],
                        },
                        ProjectionConfig {
                            src: 1, tgt: 2,
                            conn_type: "all_to_all".to_string(),
                            weights: WeightsConfig::Scalar(w),
                            coupling_fn: "Linear".to_string(),
                            coupling_params: vec![1.0],
                            cvar_map: "0:0".to_string(),
                            delays: vec![0],
                            tract_lengths: vec![],
                        },
                        ProjectionConfig {
                            src: 2, tgt: 0,
                            conn_type: "all_to_all".to_string(),
                            weights: WeightsConfig::Scalar(w),
                            coupling_fn: "Linear".to_string(),
                            coupling_params: vec![1.0],
                            cvar_map: "0:0".to_string(),
                            delays: vec![0],
                            tract_lengths: vec![],
                        },
                    ],
                },
                integrator: IntegratorKind::Heun,
                monitors: vec![],
                stimuli: vec![],
                nsig: NsigConfig::Scalar(0.0),
                speed: 3.0,
            backend: "ndarray".to_string(),
            };

            // Warmup
            let mut warmup_engine = BatchHybridEngine::<B>::from_config(config.clone(), 3, device.clone()).unwrap();
            let _ = warmup_engine.run_sweep(
                &SweepParam { sub_idx: 0, param_idx: 1 },
                &[-0.5f32, 0.0, 0.5],
                10,
            );

            let mut engine = BatchHybridEngine::<B>::from_config(config, n_sweep, device).unwrap();
            let result = engine.run_sweep(
                &SweepParam { sub_idx: 0, param_idx: 1 },
                &i_ext_values,
                n_steps,
            );
            let per_point = result.elapsed_ms / n_sweep as f64;
            println!("  NdArray generic:   {:>8.1} ms  ({:.2} ms/point)", result.elapsed_ms, per_point);

            let g2do_tavg = &result.tavg[0];
            let g2do_mean: f32 = g2do_tavg.iter().copied().sum::<f32>() / g2do_tavg.len() as f32;
            let nan_count = result.tavg.iter().map(|v| v.iter().filter(|x: &&f32| x.is_nan()).count()).sum::<usize>();
            println!("  G2DO V mean: {:.4}, NaN count: {}", g2do_mean, nan_count);
            println!();
        }
    }

    // ===== WGPU batch =====
    #[cfg(feature = "wgpu")]
    {
        type B = burn_wgpu::Wgpu<f32, i32>;
        let device = burn_wgpu::WgpuDevice::default();

        for &nnodes in &[76, 164] {
            println!("--- WGPU batch, {} nodes ---", nnodes);
            // Warmup (compile kernels)
            let warmup_vals = vec![-0.5f32, 0.0, 0.5];
            let _ = hyburn::engine::sweep_gpu::batch_sweep_3subnet::<B>(
                &warmup_vals, 4, 10, dt, w, &device,
            );

            let result = hyburn::engine::sweep_gpu::batch_sweep_3subnet::<B>(
                &i_ext_values, nnodes, n_steps, dt, w, &device,
            );
            let per_point = result.elapsed_ms / n_sweep as f64;
            println!("  WGPU batch: {:>8.1} ms  ({:.2} ms/point)", result.elapsed_ms, per_point);
            let nan_count = result.tavg_g2do.iter().chain(result.tavg_jr.iter()).chain(result.tavg_wc.iter())
                .filter(|x| x.is_nan()).count();
            println!("  NaN count: {}", nan_count);
            println!();
        }
    }

    // ===== CUDA batch =====
    #[cfg(feature = "cuda")]
    {
        type B = burn_cuda::Cuda<f32, i32>;
        let device = burn_cuda::CudaDevice::default();

        for &nnodes in &[76, 164] {
            println!("--- CUDA hardcoded, {} nodes ---", nnodes);
            let warmup_vals = vec![-0.5f32, 0.0, 0.5];
            let _ = hyburn::engine::sweep_gpu::batch_sweep_3subnet::<B>(
                &warmup_vals, 4, 10, dt, w, &device,
            );
            let result = hyburn::engine::sweep_gpu::batch_sweep_3subnet::<B>(
                &i_ext_values, nnodes, n_steps, dt, w, &device,
            );
            let per_point = result.elapsed_ms / n_sweep as f64;
            println!("  CUDA hardcoded: {:>8.1} ms  ({:.2} ms/point)", result.elapsed_ms, per_point);
            let nan_count = result.tavg_g2do.iter().chain(result.tavg_jr.iter()).chain(result.tavg_wc.iter())
                .filter(|x| x.is_nan()).count();
            println!("  NaN count: {}", nan_count);
            println!();
        }

        // Generic BatchHybridEngine on CUDA
        {
            use hyburn::config::{InitialStateConfig, NetworkConfig, NsigConfig, ProjectionConfig, SimConfig, SubnetworkConfig, WeightsConfig};
            use hyburn::engine::IntegratorKind;
            use hyburn::engine::batch_engine::{BatchHybridEngine, SweepParam};

            for &nnodes in &[76, 164] {
                println!("--- CUDA generic BatchHybridEngine, {} nodes ---", nnodes);

                let config = SimConfig {
                    sim_length: n_steps as f64 * dt as f64,
                    dt: dt as f64,
                    network: NetworkConfig {
                        subnetworks: vec![
                            SubnetworkConfig {
                                model: "Generic2dOscillator".to_string(),
                                nnodes,
                                nmodes: 1,
                                initial_state: InitialStateConfig::Inline(vec![0.0f32; 2 * nnodes]),
                                params: hyburn::model::g2do::g2do_default_params(),
                            },
                            SubnetworkConfig {
                                model: "JansenRit".to_string(),
                                nnodes,
                                nmodes: 1,
                                initial_state: InitialStateConfig::Inline(vec![0.0f32; 6 * nnodes]),
                                params: hyburn::model::jansen_rit::jansen_rit_default_params(),
                            },
                            SubnetworkConfig {
                                model: "WilsonCowan".to_string(),
                                nnodes,
                                nmodes: 1,
                                initial_state: InitialStateConfig::Inline(vec![0.0f32; 2 * nnodes]),
                                params: hyburn::model::wilson_cowan::wilson_cowan_default_params(),
                            },
                        ],
                        projections: vec![
                            ProjectionConfig {
                                src: 0, tgt: 1,
                                conn_type: "all_to_all".to_string(),
                                weights: WeightsConfig::Scalar(w),
                                coupling_fn: "Linear".to_string(),
                                coupling_params: vec![1.0],
                                cvar_map: "0:0".to_string(),
                                delays: vec![0],
                                tract_lengths: vec![],
                            },
                            ProjectionConfig {
                                src: 1, tgt: 2,
                                conn_type: "all_to_all".to_string(),
                                weights: WeightsConfig::Scalar(w),
                                coupling_fn: "Linear".to_string(),
                                coupling_params: vec![1.0],
                                cvar_map: "0:0".to_string(),
                                delays: vec![0],
                                tract_lengths: vec![],
                            },
                            ProjectionConfig {
                                src: 2, tgt: 0,
                                conn_type: "all_to_all".to_string(),
                                weights: WeightsConfig::Scalar(w),
                                coupling_fn: "Linear".to_string(),
                                coupling_params: vec![1.0],
                                cvar_map: "0:0".to_string(),
                                delays: vec![0],
                                tract_lengths: vec![],
                            },
                        ],
                    },
                    integrator: IntegratorKind::Heun,
                    monitors: vec![],
                    stimuli: vec![],
                    nsig: NsigConfig::Scalar(0.0),
                    speed: 3.0,
            backend: "ndarray".to_string(),
                };

                let mut warmup_engine = BatchHybridEngine::<B>::from_config(config.clone(), 3, device.clone()).unwrap();
                let _ = warmup_engine.run_sweep(
                    &SweepParam { sub_idx: 0, param_idx: 1 },
                    &[-0.5f32, 0.0, 0.5],
                    10,
                );

                let mut engine = BatchHybridEngine::<B>::from_config(config, n_sweep, device.clone()).unwrap();
                let result = engine.run_sweep(
                    &SweepParam { sub_idx: 0, param_idx: 1 },
                    &i_ext_values,
                    n_steps,
                );
                let per_point = result.elapsed_ms / n_sweep as f64;
                println!("  CUDA generic:     {:>8.1} ms  ({:.2} ms/point)", result.elapsed_ms, per_point);

                let g2do_tavg = &result.tavg[0];
                let g2do_mean: f32 = g2do_tavg.iter().copied().sum::<f32>() / g2do_tavg.len() as f32;
                let nan_count = result.tavg.iter().map(|v| v.iter().filter(|x: &&f32| x.is_nan()).count()).sum::<usize>();
                println!("  G2DO V mean: {:.4}, NaN count: {}", g2do_mean, nan_count);
                println!();
            }
        }
    }

    println!("============================================================");
    println!("  Done");
    println!("============================================================");
}