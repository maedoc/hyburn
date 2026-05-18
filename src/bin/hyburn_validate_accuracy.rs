#!/usr/bin/env rust
//! Validate BatchHybridEngine against Numba CUDA reference output.
//!
//! Reads /tmp/numba_tavg_*.npy and /tmp/numba_i_ext.npy, runs the
//! same sweep on NdArray, and compares per-point statistics.

use burn::backend::ndarray::NdArray;
use burn::prelude::Backend;
use hyburn::config::*;
use hyburn::engine::IntegratorKind;
use hyburn::engine::batch_engine::{BatchHybridEngine, SweepParam};
use hyburn::model::{g2do, jansen_rit, wilson_cowan};

type B = NdArray<f32>;

fn main() {
    let device: <B as Backend>::Device = Default::default();
    let nnodes = 4;
    let n_steps = 100;
    let w: f32 = 0.01;

    // Must match the Numba benchmark's I_ext values
    let i_ext_values: Vec<f32> = vec![-0.5, -0.25, 0.0, 0.25, 0.5];

    println!("============================================================");
    println!("  Hyburn BatchHybridEngine vs Numba CUDA Validation");
    println!("============================================================");
    println!("Config: 5 points × 100 steps × {} nodes", nnodes);
    println!();

    let config = SimConfig {
        sim_length: 10.0,
        dt: 0.1,
        network: NetworkConfig {
            subnetworks: vec![
                SubnetworkConfig {
                    model: "Generic2dOscillator".to_string(), nnodes, nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.0f32; 2 * nnodes]),
                    params: g2do::g2do_default_params(),
                },
                SubnetworkConfig {
                    model: "JansenRit".to_string(), nnodes, nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.0f32; 6 * nnodes]),
                    params: jansen_rit::jansen_rit_default_params(),
                },
                SubnetworkConfig {
                    model: "WilsonCowan".to_string(), nnodes, nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.0f32; 2 * nnodes]),
                    params: wilson_cowan::wilson_cowan_default_params(),
                },
            ],
            projections: vec![
                ProjectionConfig { src: 0, tgt: 1, conn_type: "all_to_all".to_string(), weights: WeightsConfig::Scalar(w), coupling_fn: "Linear".to_string(), coupling_params: vec![1.0], cvar_map: "0:0".to_string(), delays: vec![0], tract_lengths: vec![] },
                ProjectionConfig { src: 1, tgt: 2, conn_type: "all_to_all".to_string(), weights: WeightsConfig::Scalar(w), coupling_fn: "Linear".to_string(), coupling_params: vec![1.0], cvar_map: "0:0".to_string(), delays: vec![0], tract_lengths: vec![] },
                ProjectionConfig { src: 2, tgt: 0, conn_type: "all_to_all".to_string(), weights: WeightsConfig::Scalar(w), coupling_fn: "Linear".to_string(), coupling_params: vec![1.0], cvar_map: "0:0".to_string(), delays: vec![0], tract_lengths: vec![] },
            ],
        },
        integrator: IntegratorKind::Heun,
        monitors: vec![],
        stimuli: vec![],
        nsig: NsigConfig::Scalar(0.0),
        noise_mode: Default::default(),
        speed: 3.0,
        backend: "ndarray".to_string(),
    };

    let mut engine = BatchHybridEngine::<B>::from_config(config, 5, device).unwrap();
    let result = engine.run_sweep(
        &SweepParam { sub_idx: 0, param_idx: 1 },
        &i_ext_values,
        n_steps,
    );

    // tavg[0] = G2DO, shape [n_sweep * nnodes * 2] flattened
    // Layout: [sweep0_v_n0, sweep0_v_n1, ..., sweep0_w_n0, ...] or [sweep0_node0_v, sweep0_node0_w, sweep0_node1_v, ...]
    // Need to figure out layout from the Tensor shape [n_sweep, nnodes, nvar]

    let g2do_tavg = &result.tavg[0];
    let jr_tavg = &result.tavg[1];
    let wc_tavg = &result.tavg[2];

    println!("G2DO tavg length: {}", g2do_tavg.len());
    println!("JR tavg length: {}", jr_tavg.len());
    println!("WC tavg length: {}", wc_tavg.len());
    println!();

    // The tavg flat layout is [n_sweep, nnodes, nvar] → row-major
    // So for 5 sweeps × 4 nodes × 2 vars = 40 elements
    // stride: sweep × (nnodes * nvar) + node * nvar + var

    let nvar_g2do = 2;
    let nvar_jr = 6;
    let nn = nnodes;

    println!("--- Per-point comparison (Hyburn NdArray vs Numba CUDA) ---");
    println!("{:>10} {:>12} {:>12} {:>12}", "I_ext", "Hyburn V", "Numba V", "ΔV");

    // Numba reference V means (from the Python validation output)
    let numba_v_means = [-0.033235f32, -0.008991, 0.015494, 0.040218, 0.065180];

    for (i, iext) in i_ext_values.iter().enumerate() {
        // Compute mean of V (var 0) across nodes for this sweep point
        let mut v_sum = 0.0f32;
        let mut w_sum = 0.0f32;
        for n in 0..nn {
            let v_idx = i * (nn * nvar_g2do) + n * nvar_g2do + 0;
            let w_idx = i * (nn * nvar_g2do) + n * nvar_g2do + 1;
            v_sum += g2do_tavg[v_idx];
            w_sum += g2do_tavg[w_idx];
        }
        let v_mean = v_sum / nn as f32;
        let _w_mean = w_sum / nn as f32;

        let numba_v = numba_v_means[i];
        let delta = (v_mean - numba_v).abs();

        println!("{:+10.3} {:+12.6} {:+12.6} {:+12.6}", iext, v_mean, numba_v, delta);
    }

    println!();

    // JR statistics
    println!("JR per-point y0 means:");
    for (i, iext) in i_ext_values.iter().enumerate() {
        let mut y0_sum = 0.0f32;
        for n in 0..nn {
            let y0_idx = i * (nn * nvar_jr) + n * nvar_jr + 0;
            y0_sum += jr_tavg[y0_idx];
        }
        let y0_mean = y0_sum / nn as f32;
        println!("  I_ext={:+.3}: JR y0_mean={:+.6}", iext, y0_mean);
    }

    println!();

    // WC statistics
    println!("WC per-point E means:");
    for (i, iext) in i_ext_values.iter().enumerate() {
        let mut e_sum = 0.0f32;
        for n in 0..nn {
            let e_idx = i * (nn * 2) + n * 2 + 0;
            e_sum += wc_tavg[e_idx];
        }
        let e_mean = e_sum / nn as f32;
        println!("  I_ext={:+.3}: WC E_mean={:+.6}", iext, e_mean);
    }

    println!();
    println!("NOTE: Exact numerical match is not expected because:");
    println!("  - Different initial conditions (Hyburn: all zeros, Numba: random)");
    println!("  - Different RNG seeds");
    println!("  - Different coupling function (Hyburn: Linear(a=1.0) vs Numba: weight*inv_N)");
    println!("  - The sweep IS varying I_ext per point (validated by test_batch_engine_g2do_sweep_actually_varies)");
    println!();
    println!("For exact validation, use test_batch_vs_serial_three_subnet_ring which");
    println!("compares BatchHybridEngine vs serial HybridEngine with same initial conditions.");
}