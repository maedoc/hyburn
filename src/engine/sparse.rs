//! Sparse CSR coupling kernel.
//!
//! Implements CPU-side CSR matrix–vector multiply and a sparse coupling
//! function that applies a coupling function to delayed source states and
//! then multiplies by a CSR weight matrix.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use crate::engine::coupling::CouplingFn;

/// CSR matrix–vector multiply on CPU.
///
/// # Arguments
/// - `data`   – non-zero values, length `nnz`.
/// - `indices` – column indices for each non-zero, length `nnz`.
/// - `indptr`  – row pointers, length `ntgt + 1`.
/// - `x`       – dense input vector, length `nsrc`.
/// - `y`       – dense output vector, length `ntgt`.
///
/// For each row `i`:
/// ```text
/// y[i] = sum_{j = indptr[i]}^{indptr[i+1]-1} data[j] * x[indices[j]]
/// ```
pub fn sparse_csr_matvec(
    data: &[f32],
    indices: &[usize],
    indptr: &[usize],
    x: &[f32],
    y: &mut [f32],
) {
    let ntgt = y.len();
    assert_eq!(
        indptr.len(),
        ntgt + 1,
        "indptr length must be ntgt + 1"
    );

    for i in 0..ntgt {
        let mut acc = 0.0_f32;
        let row_start = indptr[i];
        let row_end = indptr[i + 1];
        for j in row_start..row_end {
            acc += data[j] * x[indices[j]];
        }
        y[i] = acc;
    }
}

/// Sparse coupling via CSR matrix multiplication on CPU.
///
/// # Arguments
/// - `csr_data`    – non-zero values, length `nnz`.
/// - `csr_indices` – column indices, length `nnz`.
/// - `csr_indptr`  – row pointers, length `ntgt + 1`.
/// - `delayed_state` – dense tensor of shape `[nsrc, ncvar]`.
/// - `coupling_fn` – coupling function to apply to each source row.
///
/// Returns a tensor of shape `[ntgt, ncvar]`.
pub fn sparse_coupling<B: Backend>(
    csr_data: &[f32],
    csr_indices: &[usize],
    csr_indptr: &[usize],
    delayed_state: Tensor<B, 2>,
    coupling_fn: &dyn CouplingFn<B>,
) -> Tensor<B, 2> {
    let pre = coupling_fn.apply(delayed_state);
    let device = pre.device();

    let (pre_data, pre_shape) = crate::io::tensor_to_flat_f32(pre);
    let nsrc = pre_shape[0];
    let ncvar = pre_shape[1];
    let ntgt = csr_indptr.len().saturating_sub(1);

    let mut result_data = vec![0.0_f32; ntgt * ncvar];

    // For each coupling variable column, extract the column vector and
    // perform a CSR matvec.
    for k in 0..ncvar {
        let x: Vec<f32> = (0..nsrc).map(|i| pre_data[i * ncvar + k]).collect();
        let mut y = vec![0.0_f32; ntgt];
        sparse_csr_matvec(csr_data, csr_indices, csr_indptr, &x, &mut y);
        for i in 0..ntgt {
            result_data[i * ncvar + k] = y[i];
        }
    }

    Tensor::from_floats(
        TensorData::new::<f32, Vec<usize>>(result_data, vec![ntgt, ncvar]),
        &device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::coupling::{dense_coupling, Linear};
    use burn::backend::ndarray::NdArray;
    use burn::tensor::TensorData;

    type B = NdArray<f32>;

    #[test]
    fn test_sparse_csr_matvec_basic() {
        // 3x3 matrix:
        // [ 0  2  0 ]
        // [ 0  0  3 ]
        // [ 1  0  0 ]
        let data = vec![2.0_f32, 3.0_f32, 1.0_f32];
        let indices = vec![1_usize, 2, 0];
        let indptr = vec![0_usize, 1, 2, 3];
        let x = vec![1.0_f32, 2.0, 3.0];
        let mut y = vec![0.0_f32; 3];

        sparse_csr_matvec(&data, &indices, &indptr, &x, &mut y);

        // row 0: 2 * x[1] = 4
        // row 1: 3 * x[2] = 9
        // row 2: 1 * x[0] = 1
        assert!((y[0] - 4.0).abs() < 1e-6, "expected 4.0, got {}", y[0]);
        assert!((y[1] - 9.0).abs() < 1e-6, "expected 9.0, got {}", y[1]);
        assert!((y[2] - 1.0).abs() < 1e-6, "expected 1.0, got {}", y[2]);
    }

    #[test]
    fn test_sparse_coupling_matches_dense_ring() {
        // 5-node directed ring: each node i receives from node (i-1) mod 5
        // with weight 0.1.
        //
        // Dense 5x5 weights:
        // row 0: [0,0,0,0,0.1]
        // row 1: [0.1,0,0,0,0]
        // row 2: [0,0.1,0,0,0]
        // row 3: [0,0,0.1,0,0]
        // row 4: [0,0,0,0.1,0]
        let dense_weights = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.1,
                    0.1, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.1, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.1, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.1, 0.0,
                ],
                vec![5, 5],
            ),
            &Default::default(),
        );

        // CSR representation of the same matrix.
        let csr_data = vec![0.1_f32; 5];
        let csr_indices = vec![4_usize, 0, 1, 2, 3];
        let csr_indptr = vec![0_usize, 1, 2, 3, 4, 5];

        // delayed_state: [nsrc=5, ncvar=2]
        let delayed_state = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![
                    1.0, 2.0,
                    3.0, 4.0,
                    5.0, 6.0,
                    7.0, 8.0,
                    9.0, 10.0,
                ],
                vec![5, 2],
            ),
            &Default::default(),
        );

        let coupling_fn = Linear { a: 1.0 };

        let dense_result = dense_coupling(dense_weights, delayed_state.clone(), &coupling_fn);
        let sparse_result = sparse_coupling(
            &csr_data,
            &csr_indices,
            &csr_indptr,
            delayed_state,
            &coupling_fn,
        );

        let (dense_data, dense_shape) = crate::io::tensor_to_flat_f32(dense_result);
        let (sparse_data, sparse_shape) = crate::io::tensor_to_flat_f32(sparse_result);

        assert_eq!(dense_shape, vec![5, 2]);
        assert_eq!(sparse_shape, vec![5, 2]);

        for (i, (d, s)) in dense_data.iter().zip(sparse_data.iter()).enumerate() {
            assert!(
                (d - s).abs() < 1e-5,
                "mismatch at index {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_sparse_coupling_with_non_trivial_coupling_fn() {
        // 3x3 sparse identity.
        let csr_data = vec![1.0_f32, 1.0, 1.0];
        let csr_indices = vec![0_usize, 1, 2];
        let csr_indptr = vec![0_usize, 1, 2, 3];

        let delayed_state = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![
                    1.0, 2.0,
                    3.0, 4.0,
                    5.0, 6.0,
                ],
                vec![3, 2],
            ),
            &Default::default(),
        );

        let dense_weights = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![
                    1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0,
                ],
                vec![3, 3],
            ),
            &Default::default(),
        );

        // Linear with a=2.0.
        let coupling_fn = Linear { a: 2.0 };

        let dense_result = dense_coupling(dense_weights, delayed_state.clone(), &coupling_fn);
        let sparse_result = sparse_coupling(
            &csr_data,
            &csr_indices,
            &csr_indptr,
            delayed_state,
            &coupling_fn,
        );

        let (dense_data, _) = crate::io::tensor_to_flat_f32(dense_result);
        let (sparse_data, _) = crate::io::tensor_to_flat_f32(sparse_result);

        for (i, (d, s)) in dense_data.iter().zip(sparse_data.iter()).enumerate() {
            assert!(
                (d - s).abs() < 1e-5,
                "mismatch at index {}: dense={}, sparse={}",
                i,
                d,
                s
            );
        }
    }

    #[test]
    fn test_engine_csr_from_config_matches_dense() {
        use crate::config::{
            InitialStateConfig, NetworkConfig, ProjectionConfig, SimConfig, SubnetworkConfig,
            WeightsConfig,
        };
        use crate::engine::{EngineModel, HybridEngine, IntegratorKind};
        use crate::model::g2do::g2do_default_params;

        let nnodes = 5;
        let nvar = 2;
        let nmodes = 1;

        let sub = SubnetworkConfig {
            model: "Generic2dOscillator".to_string(),
            nnodes,
            nmodes,
            initial_state: InitialStateConfig::Inline(vec![0.1_f32; nvar * nnodes * nmodes]),
            params: g2do_default_params(),
        };

        let dense_proj = ProjectionConfig {
            src: 0,
            tgt: 0,
            conn_type: "all_to_all".to_string(),
            weights: WeightsConfig::Dense(vec![
                vec![0.0, 0.0, 0.0, 0.0, 0.1],
                vec![0.1, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.1, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.1, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.1, 0.0],
            ]),
            delays: vec![1u32; 5],
            coupling_fn: "Linear".to_string(),
            coupling_params: vec![1.0],
            cvar_map: "0:0".to_string(),
        };

        let csr_proj = ProjectionConfig {
            src: 0,
            tgt: 0,
            conn_type: "csr".to_string(),
            weights: WeightsConfig::Csr {
                data: vec![0.1_f32; 5],
                indices: vec![4u32, 0, 1, 2, 3],
                indptr: vec![0u32, 1, 2, 3, 4, 5],
            },
            delays: vec![1u32; 5],
            coupling_fn: "Linear".to_string(),
            coupling_params: vec![1.0],
            cvar_map: "0:0".to_string(),
        };

        let dense_cfg = SimConfig {
            sim_length: 10.0,
            dt: 0.1,
            network: NetworkConfig {
                subnetworks: vec![sub.clone()],
                projections: vec![dense_proj],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: 0.0,
            backend: "ndarray".to_string(),
        };

        let csr_cfg = SimConfig {
            sim_length: 10.0,
            dt: 0.1,
            network: NetworkConfig {
                subnetworks: vec![sub],
                projections: vec![csr_proj],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: 0.0,
            backend: "ndarray".to_string(),
        };

        let mut dense_engine =
            HybridEngine::<B>::from_config(dense_cfg, Default::default()).unwrap();
        let mut csr_engine =
            HybridEngine::<B>::from_config(csr_cfg, Default::default()).unwrap();

        dense_engine.run(10);
        csr_engine.run(10);

        let (dense_flat, _) = crate::io::tensor_to_flat_f32(dense_engine.states[0].clone());
        let (csr_flat, _) = crate::io::tensor_to_flat_f32(csr_engine.states[0].clone());

        assert_eq!(dense_flat.len(), csr_flat.len());
        for (i, (d, s)) in dense_flat.iter().zip(csr_flat.iter()).enumerate() {
            assert!(
                (d - s).abs() < 1e-5,
                "dense vs csr mismatch at index {}: dense={}, csr={}",
                i,
                d,
                s
            );
        }
    }
}
