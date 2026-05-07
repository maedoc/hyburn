//! Parallel parameter sweep using Rayon multicore parallelism.
//!
//! Runs multiple `HybridEngine` instances in parallel across CPU cores,
//! each with different parameter values. This gives approximately N_cores×
//! speedup for parameter sweeps on multicore machines.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};

use rayon::prelude::*;

use crate::config::SimConfig;
use crate::engine::{EngineModel, HybridEngine, IntegratorKind};
use crate::model::g2do::g2do_default_params;
use crate::sbi::features::{extract_features_with, FeatureSet};

/// Result of a single sweep point simulation.
#[derive(Debug, Clone)]
pub struct SweepResult {
    /// Parameter value at this sweep point.
    pub param_value: f32,
    /// Final state tensor data for each subnetwork.
    pub final_states: Vec<Vec<f32>>,
    /// Extracted features from the trajectory.
    pub features: Vec<f32>,
    /// Full trajectory (if requested).
    pub trajectory: Option<Vec<f32>>,
}

/// Configuration for a parameter sweep.
pub struct SweepConfig {
    /// Parameter index to vary (e.g., 1 for I_ext in G2DO).
    pub param_idx: usize,
    /// Parameter values to sweep over.
    pub param_values: Vec<f32>,
    /// Number of simulation steps per sweep point.
    pub n_steps: usize,
    /// Number of nodes per subnetwork.
    pub nnodes: usize,
    /// Integration time step.
    pub dt: f64,
    /// Whether to keep full trajectory in results.
    pub keep_trajectory: bool,
    /// Feature extraction method for trajectory summarization.
    pub feature_set: FeatureSet,
}

impl Default for SweepConfig {
    fn default() -> Self {
        Self {
            param_idx: 1, // I_ext
            param_values: Vec::new(),
            n_steps: 500,
            nnodes: 2,
            dt: 0.1,
            keep_trajectory: false,
            feature_set: FeatureSet::Classic,
        }
    }
}

/// Run a parameter sweep using Rayon parallelism.
///
/// Each sweep point creates its own `HybridEngine` and runs `n_steps` steps.
/// The `param_idx` parameter in the default G2DO params is replaced with each
/// value from `param_values`.
///
/// Returns results in the same order as `param_values`.
pub fn parallel_sweep<B: Backend>(config: &SweepConfig, device: B::Device) -> Vec<SweepResult> {
    let _nvar = 2;
    let _nmodes = 1;
    let _nvar_g2do = 2;

    config
        .param_values
        .par_iter()
        .map(|&param_value| {
            run_single_point::<B>(
                config.param_idx,
                param_value,
                config.n_steps,
                config.nnodes,
                config.dt,
                config.keep_trajectory,
                config.feature_set.clone(),
                device.clone(),
            )
        })
        .collect()
}

/// Configuration for a single sweep-point simulation.
#[derive(Debug, Clone)]
pub struct SinglePointConfig {
    /// Index of the G2DO parameter to vary.
    pub param_idx: usize,
    /// Parameter value for this sweep point.
    pub param_value: f32,
    /// Number of simulation steps.
    pub n_steps: usize,
    /// Number of network nodes.
    pub nnodes: usize,
    /// Time step (ms).
    pub dt: f64,
    /// Whether to keep the full state trajectory.
    pub keep_trajectory: bool,
    /// Feature set to extract after the run.
    pub feature_set: FeatureSet,
}

/// Run a single sweep point using a config struct.
fn run_single_point_with_config<B: Backend>(
    config: &SinglePointConfig,
    device: B::Device,
) -> SweepResult {
    let mut params = g2do_default_params();
    params[config.param_idx] = config.param_value;

    let nvar = 2;
    let nmodes = 1;
    let initial_data = vec![0.0f32; nvar * config.nnodes * nmodes];
    let state = Tensor::<B, 3>::from_data(
        TensorData::new::<f32, Vec<usize>>(initial_data, vec![nvar, config.nnodes, nmodes]),
        &device,
    );

    let model = EngineModel::<B>::G2do { params };
    let mut engine = HybridEngine::new(state, model, IntegratorKind::Heun, config.dt, 1, device);

    engine.run(config.n_steps);

    // Extract final states
    let final_states: Vec<Vec<f32>> = engine
        .states
        .iter()
        .map(|s| {
            let (data, _) = crate::io::tensor_to_flat_f32::<B, 3>(s.clone());
            data
        })
        .collect();

    // Extract features
    let nvar = engine.subnetworks[0].nvar;
    let nnodes = engine.subnetworks[0].nnodes;
    let nmodes = engine.subnetworks[0].nmodes;
    let features = if !engine.trajectory.is_empty() {
        extract_features_with(
            &engine.trajectory,
            &[config.n_steps, nvar, nnodes, nmodes],
            &config.feature_set,
        )
    } else {
        Vec::new()
    };

    let trajectory = if config.keep_trajectory {
        Some(engine.trajectory.clone())
    } else {
        None
    };

    SweepResult {
        param_value: config.param_value,
        final_states,
        features,
        trajectory,
    }
}

/// Backward-compatible wrapper around [`run_single_point_with_config`].
#[allow(clippy::too_many_arguments)]
fn run_single_point<B: Backend>(
    param_idx: usize,
    param_value: f32,
    n_steps: usize,
    nnodes: usize,
    dt: f64,
    keep_trajectory: bool,
    feature_set: FeatureSet,
    device: B::Device,
) -> SweepResult {
    run_single_point_with_config::<B>(
        &SinglePointConfig {
            param_idx,
            param_value,
            n_steps,
            nnodes,
            dt,
            keep_trajectory,
            feature_set,
        },
        device,
    )
}

/// Run a sequential (non-parallel) sweep for comparison.
pub fn serial_sweep<B: Backend>(config: &SweepConfig, device: B::Device) -> Vec<SweepResult> {
    config
        .param_values
        .iter()
        .map(|&param_value| {
            run_single_point::<B>(
                config.param_idx,
                param_value,
                config.n_steps,
                config.nnodes,
                config.dt,
                config.keep_trajectory,
                config.feature_set.clone(),
                device.clone(),
            )
        })
        .collect()
}

/// Run a parallel sweep using a SimConfig (for coupled networks, projections, etc.).
pub fn parallel_sweep_from_config<B: Backend>(
    base_config: SimConfig,
    param_name: &str,
    param_values: Vec<f32>,
    n_steps: usize,
    device: B::Device,
) -> Vec<SweepResult> {
    param_values
        .par_iter()
        .map(|&value| {
            let mut cfg = base_config.clone();
            // Apply sweep value
            if let Err(e) = apply_sweep_value(&mut cfg, param_name, value) {
                log::warn!("Sweep value {} failed: {}", value, e);
            }

            let mut engine = HybridEngine::<B>::from_config(cfg, device.clone())
                .expect("from_config failed");
            engine.run(n_steps);

            let trajectory = engine.trajectory.clone();
            let final_states: Vec<Vec<f32>> = engine
                .states
                .iter()
                .map(|s| {
                    let (data, _) = crate::io::tensor_to_flat_f32::<B, 3>(s.clone());
                    data
                })
                .collect();

            SweepResult {
                param_value: value,
                final_states,
                features: Vec::new(),
                trajectory: Some(trajectory),
            }
        })
        .collect()
}

/// Apply a sweep value to a SimConfig parameter.
fn apply_sweep_value(cfg: &mut SimConfig, name: &str, value: f32) -> Result<(), String> {
    if name == "dt" {
        cfg.dt = value as f64;
    } else if name == "nsig" {
        cfg.nsig = value;
    } else if name.starts_with("subnetworks[") && name.contains("].params[") {
        let start = name.find('[').ok_or_else(|| format!("malformed sweep param name '{}': missing '['", name))? + 1;
        let end = name.find(']').ok_or_else(|| format!("malformed sweep param name '{}': missing ']'", name))?;
        if end <= start {
            return Err(format!("malformed sweep param name '{}': empty subnetwork index", name));
        }
        let sub_idx: usize = name[start..end].parse().map_err(|e| format!("invalid subnetwork index in '{}': {}", name, e))?;
        let pstart = name.rfind('[').ok_or_else(|| format!("malformed sweep param name '{}': missing second '['", name))? + 1;
        let pend = name.rfind(']').ok_or_else(|| format!("malformed sweep param name '{}': missing second ']'", name))?;
        if pend <= pstart {
            return Err(format!("malformed sweep param name '{}': empty param index", name));
        }
        let param_idx: usize = name[pstart..pend].parse().map_err(|e| format!("invalid param index in '{}': {}", name, e))?;
        if sub_idx >= cfg.network.subnetworks.len() {
            return Err(format!("subnetwork index {} out of range", sub_idx));
        }
        let params = &mut cfg.network.subnetworks[sub_idx].params;
        if param_idx >= params.len() {
            return Err(format!("param index {} out of range for subnetwork {}", param_idx, sub_idx));
        }
        params[param_idx] = value;
    } else {
        return Err(format!("unsupported sweep parameter_name: {}", name));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_parallel_sweep_matches_serial() {
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let config = SweepConfig {
            param_idx: 1,
            param_values: vec![-0.3, 0.0, 0.3],
            n_steps: 100,
            nnodes: 2,
            dt: 0.1,
            keep_trajectory: false,
            feature_set: FeatureSet::Classic,
        };

        let par_results = parallel_sweep::<B>(&config, device.clone());
        let ser_results = serial_sweep::<B>(&config, device);

        assert_eq!(par_results.len(), ser_results.len());
        for (p, s) in par_results.iter().zip(ser_results.iter()) {
            assert_eq!(p.param_value, s.param_value);
            // Features should match (deterministic simulation)
            assert_eq!(p.features.len(), s.features.len());
            for (i, (pf, sf)) in p.features.iter().zip(s.features.iter()).enumerate() {
                assert!(
                    (pf - sf).abs() < 1e-5,
                    "Feature {} mismatch: parallel={}, serial={}",
                    i, pf, sf
                );
            }
        }
    }

    #[test]
    fn test_parallel_sweep_produces_different_trajectories() {
        let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
        let config = SweepConfig {
            param_idx: 1,
            param_values: vec![-0.5, 0.5],
            n_steps: 200,
            nnodes: 2,
            dt: 0.1,
            keep_trajectory: false,
            feature_set: FeatureSet::Classic,
        };

        let results = parallel_sweep::<B>(&config, device);
        assert_eq!(results.len(), 2);

        // Different I_ext values should produce different features
        let diff: f32 = results[0]
            .features
            .iter()
            .zip(results[1].features.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01, "Different I_ext should produce different features");
    }
}