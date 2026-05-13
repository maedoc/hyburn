//! Autotuning strategy selection for coupling computation.
//!
//! Picks between dense, sparse CPU, and tiled sparse GPU strategies
//! based on the network size (and in future, empirical micro-benchmarks).

use std::time::Instant;

/// Available coupling execution strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CouplingStrategy {
    /// Dense matrix multiplication (optimal for small networks).
    Dense,
    /// Sparse CSR matvec on CPU (optimal for medium networks).
    SparseCSR,
    /// Tiled CSR kernel on GPU (optimal for very large networks).
    TiledCSR,
}

/// Result of an autotuning run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutotuneResult {
    /// Recommended block / tile size for the kernel.
    pub optimal_block_size: usize,
    /// Fastest strategy found.
    pub optimal_strategy: CouplingStrategy,
    /// Measured time for the best strategy (nanoseconds).
    pub benchmark_time_ns: u128,
}

/// Select the best coupling strategy for a given number of nodes.
///
/// The thresholds are chosen heuristically based on typical CPU/GPU
/// crossover points for sparse matrix–vector products:
///
/// | Range               | Strategy  | Rationale                          |
/// |---------------------|-----------|------------------------------------|
/// | `N < 500`           | Dense     | Dense matmul overhead is fine.     |
/// | `500 ≤ N < 2000`    | SparseCSR | CPU sparse avoids O(N²) fill.      |
/// | `N ≥ 2000`          | TiledCSR  | GPU kernel wins at large scale.    |
///
/// # Example
/// ```
/// use hyburn::engine::autotune::{select_strategy, CouplingStrategy};
/// assert_eq!(select_strategy(100), CouplingStrategy::Dense);
/// assert_eq!(select_strategy(1000), CouplingStrategy::SparseCSR);
/// assert_eq!(select_strategy(5000), CouplingStrategy::TiledCSR);
/// ```
pub fn select_strategy(nnodes: usize) -> CouplingStrategy {
    if nnodes < 500 {
        CouplingStrategy::Dense
    } else if nnodes < 2000 {
        CouplingStrategy::SparseCSR
    } else {
        CouplingStrategy::TiledCSR
    }
}

/// Benchmark a coupling strategy for `n_steps` on a synthetic network.
///
/// Returns elapsed time in nanoseconds.
pub fn benchmark_coupling(nnodes: usize, strategy: CouplingStrategy, n_steps: usize) -> u128 {
    use burn::backend::ndarray::NdArray;
    use burn::tensor::{Tensor, TensorData};
    use crate::engine::coupling::{dense_coupling, CouplingFnConfig};
    use crate::engine::sparse::sparse_coupling;

    type B = NdArray<f32>;
    let device = Default::default();

    // Synthetic dense weights and delayed state.
    let weights_data: Vec<f32> = (0..nnodes * nnodes).map(|_| rand::random::<f32>()).collect();
    let weights = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(weights_data, vec![nnodes, nnodes]),
        &device,
    );

    let state_data: Vec<f32> = (0..nnodes * 2).map(|_| rand::random::<f32>()).collect();
    let delayed_state = Tensor::<B, 2>::from_floats(
        TensorData::new::<f32, Vec<usize>>(state_data, vec![nnodes, 2]),
        &device,
    );

    let coupling_cfg = CouplingFnConfig::Linear { a: 1.0, b: 0.0 };

    // Build a sparse CSR equivalent with ~10% density.
    let mut sparse_data = Vec::new();
    let mut sparse_indices = Vec::new();
    let mut sparse_indptr = vec![0_usize];
    let density = 0.1_f32;
    for i in 0..nnodes {
        let mut row_nnz = 0;
        for j in 0..nnodes {
            if rand::random::<f32>() < density || i == j {
                sparse_data.push(rand::random::<f32>());
                sparse_indices.push(j);
                row_nnz += 1;
            }
        }
        sparse_indptr.push(sparse_indptr.last().unwrap() + row_nnz);
    }

    let start = Instant::now();
    match strategy {
        CouplingStrategy::Dense => {
            for _ in 0..n_steps {
                let _ = dense_coupling(weights.clone(), delayed_state.clone(), &coupling_cfg, None);
            }
        }
        CouplingStrategy::SparseCSR => {
            for _ in 0..n_steps {
                let _ = sparse_coupling(
                    &sparse_data,
                    &sparse_indices,
                    &sparse_indptr,
                    delayed_state.clone(),
                    &coupling_cfg,
                    None,
                );
            }
        }
        CouplingStrategy::TiledCSR => {
            // TiledCSR not yet implemented as a distinct kernel; benchmark CSR as proxy.
            for _ in 0..n_steps {
                let _ = sparse_coupling(
                    &sparse_data,
                    &sparse_indices,
                    &sparse_indptr,
                    delayed_state.clone(),
                    &coupling_cfg,
                    None,
                );
            }
        }
    }
    start.elapsed().as_nanos()
}

/// Autotune coupling for a given network size by benchmarking Dense vs SparseCSR.
///
/// Returns an [`AutotuneResult`] with the faster strategy.
pub fn autotune_coupling(nnodes: usize) -> AutotuneResult {
    // Scale benchmark steps inversely with network size to keep runtime reasonable.
    let n_steps = if nnodes < 100 {
        1000
    } else if nnodes < 1000 {
        100
    } else {
        10
    };

    let dense_time = benchmark_coupling(nnodes, CouplingStrategy::Dense, n_steps);
    let sparse_time = benchmark_coupling(nnodes, CouplingStrategy::SparseCSR, n_steps);

    let (optimal_strategy, benchmark_time_ns) = if dense_time <= sparse_time {
        (CouplingStrategy::Dense, dense_time)
    } else {
        (CouplingStrategy::SparseCSR, sparse_time)
    };

    // Heuristic block size: smaller tiles for large networks.
    let optimal_block_size = if nnodes < 256 {
        nnodes
    } else if nnodes < 1024 {
        128
    } else {
        256
    };

    AutotuneResult {
        optimal_block_size,
        optimal_strategy,
        benchmark_time_ns,
    }
}

/// Stub for future GPU autotuning.
///
/// When Burn/CubeCL exposes kernel-benchmarking hooks, this struct
/// will run micro-benchmarks against synthetic weight matrices and
/// return the empirically fastest [`CouplingStrategy`].
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuAutotuner;

impl GpuAutotuner {
    /// Create a new GPU autotuner stub.
    pub fn new() -> Self {
        Self
    }

    /// Run (future) micro-benchmarks to find the best strategy.
    ///
    /// Currently a stub; always returns [`CouplingStrategy::TiledCSR`].
    pub fn autotune(&self, _nnodes: usize) -> CouplingStrategy {
        CouplingStrategy::TiledCSR
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_strategy_thresholds() {
        assert_eq!(select_strategy(0), CouplingStrategy::Dense);
        assert_eq!(select_strategy(1), CouplingStrategy::Dense);
        assert_eq!(select_strategy(499), CouplingStrategy::Dense);
        assert_eq!(select_strategy(500), CouplingStrategy::SparseCSR);
        assert_eq!(select_strategy(1000), CouplingStrategy::SparseCSR);
        assert_eq!(select_strategy(1999), CouplingStrategy::SparseCSR);
        assert_eq!(select_strategy(2000), CouplingStrategy::TiledCSR);
        assert_eq!(select_strategy(10000), CouplingStrategy::TiledCSR);
    }

    #[test]
    fn test_gpu_autotuner_stub() {
        let tuner = GpuAutotuner::new();
        assert_eq!(tuner.autotune(100), CouplingStrategy::TiledCSR);
        assert_eq!(tuner.autotune(5000), CouplingStrategy::TiledCSR);
    }

    #[test]
    fn test_benchmark_coupling_dense_runs() {
        let t = benchmark_coupling(10, CouplingStrategy::Dense, 10);
        assert!(t > 0, "benchmark should take some time");
    }

    #[test]
    fn test_autotune_coupling_produces_result() {
        let result = autotune_coupling(50);
        assert!(result.benchmark_time_ns > 0);
        // Strategy should be one of the benchmarked variants.
        assert!(
            matches!(
                result.optimal_strategy,
                CouplingStrategy::Dense | CouplingStrategy::SparseCSR
            ),
            "unexpected strategy: {:?}",
            result.optimal_strategy
        );
    }
}
