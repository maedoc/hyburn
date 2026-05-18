#!/usr/bin/env rust
//! Realistic 3-subnetwork parameter sweep benchmark.
//!
//! Models: Generic2dOscillator → JansenRit → WilsonCowan (fully coupled ring)
//! Matching Python TVB benchmark for direct comparison.
//!
//! Usage:
//!   cargo run --release --bin hyburn-bench-realistic

use std::time::Instant;

use burn::backend::ndarray::NdArray;

use hyburn::config::{
    InitialStateConfig, NetworkConfig, NsigConfig, ProjectionConfig, SimConfig,
    SubnetworkConfig, WeightsConfig,
};
use hyburn::engine::integrator::IntegratorKind;
use hyburn::engine::sweep::parallel_sweep_from_config;
use hyburn::model::g2do::g2do_default_params;
use hyburn::model::jansen_rit::jansen_rit_default_params;
use hyburn::model::wilson_cowan::wilson_cowan_default_params;

type B = NdArray<f32>;

fn make_3subnet_config(nnodes: usize) -> SimConfig {
    // Subnet 0: Generic2dOscillator (2 vars, 1 cvar)
    let g2do_params = g2do_default_params();
    let g2do_init = vec![0.0f32; 2 * nnodes];

    // Subnet 1: JansenRit (6 vars, 1 cvar)
    let jr_params = jansen_rit_default_params();
    let jr_init = vec![0.0f32; 6 * nnodes];

    // Subnet 2: WilsonCowan (2 vars, 1 cvar)
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
                // G2DO → JansenRit (Linear on cvar 0→0)
                ProjectionConfig {
                    src: 0,
                    tgt: 1,
                    conn_type: "all_to_all".to_string(),
                    weights: WeightsConfig::Scalar(0.01),
                    delays: vec![],
                    tract_lengths: vec![],
                    coupling_fn: "Linear".to_string(),
                    coupling_params: vec![0.01],
                    cvar_map: "0:0".to_string(),
                },
                // JansenRit → WilsonCowan (Linear on cvar 0→0)
                ProjectionConfig {
                    src: 1,
                    tgt: 2,
                    conn_type: "all_to_all".to_string(),
                    weights: WeightsConfig::Scalar(0.01),
                    delays: vec![],
                    tract_lengths: vec![],
                    coupling_fn: "Linear".to_string(),
                    coupling_params: vec![0.01],
                    cvar_map: "0:0".to_string(),
                },
                // WilsonCowan → G2DO (Linear on cvar 0→0)
                ProjectionConfig {
                    src: 2,
                    tgt: 0,
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
        integrator: IntegratorKind::Heun,
        monitors: vec![],
        stimuli: vec![],
        nsig: NsigConfig::Scalar(0.0),
        noise_mode: Default::default(),
        speed: 3.0,
        backend: "ndarray".to_string(),
    }
}

fn sweep_values(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| -0.5f32 + i as f32 * (1.0f32 / (n - 1).max(1) as f32))
        .collect()
}

/// Serial sweep
fn run_serial_sweep(cfg: &SimConfig, n_sweep: usize, n_steps: usize) -> std::time::Duration {
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
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

/// Parallel sweep via Rayon
fn run_parallel_sweep(cfg: &SimConfig, n_sweep: usize, n_steps: usize) -> std::time::Duration {
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let values = sweep_values(n_sweep);
    let start = Instant::now();
    let _results = parallel_sweep_from_config::<B>(
        cfg.clone(),
        "subnetworks[0].params[1]",
        values,
        n_steps,
        device,
    );
    start.elapsed()
}

fn main() {
    let n_cores = rayon::current_num_threads();
    let n_sweep = 1024;
    let n_steps = 1000;

    println!("============================================================");
    println!("  hyburn Realistic 3-Subnetwork Sweep Benchmark");
    println!("============================================================");
    println!();
    println!("Network: G2DO → JansenRit → WilsonCowan (coupled ring)");
    println!("Coupling: All-to-all Linear, weight=0.01");
    println!("Sweep:   G2DO I_ext (params[1]) over {} points", n_sweep);
    println!("Steps:   {} per sweep point (dt=0.1ms)", n_steps);
    println!("Cores:   {}", n_cores);
    println!("Integrator: Heun (deterministic)");
    println!();

    // Warmup (small network, few points)
    {
        let warmup_cfg = make_3subnet_config(2);
        let warmup_vals = sweep_values(4);
        let _ = parallel_sweep_from_config::<B>(
            warmup_cfg,
            "subnetworks[0].params[1]",
            warmup_vals,
            10,
            Default::default(),
        );
    }

    for &nnodes in &[76, 164] {
        let cfg = make_3subnet_config(nnodes);
        let total_state_vars = 2 * nnodes + 6 * nnodes + 2 * nnodes; // G2DO + JR + WC
        let total_nodes = 3 * nnodes;

        println!("--- {} nodes/subnet ({} total nodes, {} total state vars) ---",
                 nnodes, total_nodes, total_state_vars);

        // Serial
        let serial_t = run_serial_sweep(&cfg, n_sweep, n_steps);
        println!(
            "  Serial:  {:>10.1} ms  ({:.2} ms/point)",
            serial_t.as_millis(),
            serial_t.as_millis() as f64 / n_sweep as f64,
        );

        // Parallel (Rayon)
        let parallel_t = run_parallel_sweep(&cfg, n_sweep, n_steps);
        let speedup = serial_t.as_secs_f64() / parallel_t.as_secs_f64();
        println!(
            "  Rayon:   {:>10.1} ms  ({:.2} ms/point)  {:.1}× speedup",
            parallel_t.as_millis(),
            parallel_t.as_millis() as f64 / n_sweep as f64,
            speedup,
        );
        println!();
    }

    println!("============================================================");
    println!("  Done");
    println!("============================================================");
}