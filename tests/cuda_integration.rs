#![cfg(feature = "cuda")]

//! CUDA/GPU integration test — runs the engine on the CUDA backend.

use hyburn::engine::{EngineModel, HybridEngine, IntegratorKind};
use hyburn::model::g2do::g2do_default_params;
use hyburn::io::tensor_to_flat_f32;
use burn_cuda::Cuda;
use burn::tensor::{Tensor, TensorData};
use burn::backend::ndarray::NdArray;

type B = Cuda<f32, i32>;

#[test]
fn test_g2do_cuda_backend() {
    let device = burn_cuda::CudaDevice::default();
    let nnodes = 4;
    let nvar = 2;
    let nmodes = 1;
    let dt = 0.1_f64;
    let n_steps = 500;

    let initial_data: Vec<f32> = (0..nnodes * nvar * nmodes)
        .map(|i| if i % 2 == 0 { 0.1 } else { -0.05 })
        .collect();
    let state = Tensor::<B, 3>::from_floats(
        TensorData::new::<f32, Vec<usize>>(initial_data, vec![nvar, nnodes, nmodes]),
        &device,
    );

    let model = EngineModel::<B>::G2do {
        params: g2do_default_params(),
    };
    let mut engine = HybridEngine::new(state, model, IntegratorKind::Heun, dt, 10, device);
    engine.run(n_steps);

    let (final_data, _) = tensor_to_flat_f32::<B, 3>(engine.states[0].clone());
    for v in &final_data {
        assert!(v.is_finite(), "NaN/Inf in CUDA final state: {}", v);
    }

    let final_mag: f32 = final_data.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(final_mag > 0.0, "Final state has zero magnitude on CUDA");
}

#[test]
fn test_g2do_cuda_matches_ndarray() {
    type NB = NdArray<f32>;

    let nnodes = 2;
    let nvar = 2;
    let nmodes = 1;
    let dt = 0.1_f64;
    let n_steps = 100;

    let initial_data: Vec<f32> = vec![0.1, -0.05, 0.2, -0.1];

    // NdArray run
    let nd_device = Default::default();
    let nd_state = Tensor::<NB, 3>::from_floats(
        TensorData::new::<f32, Vec<usize>>(initial_data.clone(), vec![nvar, nnodes, nmodes]),
        &nd_device,
    );
    let nd_model = EngineModel::<NB>::G2do { params: g2do_default_params() };
    let mut nd_engine = HybridEngine::new(nd_state, nd_model, IntegratorKind::Heun, dt, 1, nd_device);
    nd_engine.run(n_steps);
    let (nd_final, _) = tensor_to_flat_f32::<NB, 3>(nd_engine.states[0].clone());

    // CUDA run
    let cuda_device = burn_cuda::CudaDevice::default();
    let cuda_state = Tensor::<B, 3>::from_floats(
        TensorData::new::<f32, Vec<usize>>(initial_data, vec![nvar, nnodes, nmodes]),
        &cuda_device,
    );
    let cuda_model = EngineModel::<B>::G2do { params: g2do_default_params() };
    let mut cuda_engine = HybridEngine::new(cuda_state, cuda_model, IntegratorKind::Heun, dt, 1, cuda_device);
    cuda_engine.run(n_steps);
    let (cuda_final, _) = tensor_to_flat_f32::<B, 3>(cuda_engine.states[0].clone());

    // Compare
    assert_eq!(nd_final.len(), cuda_final.len(), "Output length mismatch");
    for (i, (a, b)) in nd_final.iter().zip(cuda_final.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-4,
            "NdArray vs CUDA mismatch at index {}: NdArray={}, CUDA={}, diff={}",
            i, a, b, diff
        );
    }
}