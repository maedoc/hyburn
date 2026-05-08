//! CrossCoder → Simulation pipeline.
//!
//! High-level orchestration:
//! 1. Sample latent codes `z ~ MVN(cohort_prior)`.
//! 2. Decode `z` through a CrossCoder view decoder to get a synthetic SC matrix.
//! 3. Inject the SC matrix into a `SimConfig` as projection weights.
//! 4. Run the simulation for `n_sweep` samples.
//! 5. Extract summary features from each simulation trajectory.
//! 6. Build a `(params, features)` dataset ready for MAF training.
//!
//! This bridges the generative latent model (CrossCoder) with the
//! forward-simulation SBI workflow.

use crate::config::{SimConfig, WeightsConfig};
use crate::engine::HybridEngine;
use crate::sbi::crosscoder::CrossCoder;
use crate::sbi::crosscoder_cohort::MvnPrior;
use crate::sbi::features::{extract_features_with, FeatureSet};
use crate::sbi::train::train_maf_with_data_and_log;
use crate::sbi::MafConfig;
use crate::sbi::MAF;

/// Type alias for the result of training a MAF from synthetic cohort data.
pub type MafTrainingResult = (MAF<Autodiff<NdArray<f32>>>, Vec<(usize, f32)>);

/// Configuration for the CrossCoder → simulation → features pipeline.
#[derive(Debug, Clone)]
pub struct CrossCoderPipelineConfig<'a, B: Backend> {
    /// Trained CrossCoder model.
    pub model: &'a CrossCoder<B>,
    /// MVN prior fitted over the cohort latents.
    pub prior: &'a MvnPrior,
    /// View index to decode through.
    pub view_idx: usize,
    /// Simulation template configuration.
    pub template: &'a SimConfig,
    /// Number of nodes (determines SC matrix size).
    pub nnodes: usize,
    /// Feature set to extract from trajectories.
    pub feature_set: &'a FeatureSet,
    /// Number of synthetic samples to generate.
    pub n_samples: usize,
    /// Burn compute device.
    pub device: &'a B::Device,
    /// Optional RNG seed.
    pub seed: Option<u64>,
}

use burn::backend::autodiff::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::tensor::{backend::Backend, Tensor, TensorData};

/// Generate synthetic SC matrices by sampling from the latent MVN prior
/// and decoding through the CrossCoder.
///
/// Each decoded matrix is returned as a flat `[nnodes * nnodes]` Vec.
pub fn generate_synthetic_sc_matrices<B: Backend>(
    model: &CrossCoder<B>,
    prior: &MvnPrior,
    view_idx: usize,
    n_samples: usize,
    device: &B::Device,
    seed: Option<u64>,
) -> Vec<Vec<f32>> {
    let latent_codes = prior.sample(n_samples, seed);
    let latent_dim = prior.latent_dim;

    let mut matrices = Vec::with_capacity(n_samples);
    for s in 0..n_samples {
        let z_flat = &latent_codes[s * latent_dim..(s + 1) * latent_dim];
        let z_tensor = Tensor::<B, 2>::from_data(
            TensorData::new::<f32, Vec<usize>>(z_flat.to_vec(), vec![1, latent_dim]),
            device,
        );
        let decoded = model.views[view_idx].decode(z_tensor);
        let (flat, _shape) = crate::io::tensor_to_flat_f32(decoded);
        matrices.push(flat);
    }
    matrices
}

/// Build a `SimConfig` from a template, injecting a decoded SC matrix
/// as the weights of projection 0.
///
/// `sc_matrix` is flat `[nnodes, nnodes]` row-major.
pub fn build_sim_config_with_sc(
    template: &SimConfig,
    sc_matrix: &[f32],
    nnodes: usize,
) -> SimConfig {
    let mut cfg = template.clone();

    // Convert flat matrix to 2D Vec for WeightsConfig::Dense
    let mut dense = Vec::with_capacity(nnodes);
    for i in 0..nnodes {
        let mut row = Vec::with_capacity(nnodes);
        for j in 0..nnodes {
            row.push(sc_matrix[i * nnodes + j]);
        }
        dense.push(row);
    }

    if !cfg.network.projections.is_empty() {
        cfg.network.projections[0].weights = WeightsConfig::Dense(dense);
    } else {
        // If no projections exist, create one (src 0, tgt 0)
        cfg.network.projections.push(crate::config::ProjectionConfig {
            src: 0,
            tgt: 0,
            conn_type: "all_to_all".to_string(),
            weights: WeightsConfig::Dense(dense),
            delays: Vec::new(),
            coupling_fn: "Linear".to_string(),
            coupling_params: vec![1.0],
            cvar_map: "0:0".to_string(),
        });
    }
    cfg
}

/// Run the full CrossCoder → Simulation → Features pipeline for `n_samples`.
///
/// For each sample:
/// 1. Sample `z` from the MVN prior.
/// 2. Decode `z` to an SC matrix.
/// 3. Run a simulation using the injected SC matrix.
/// 4. Extract summary features from the trajectory.
///
/// Returns `(all_params, all_features)` where `all_params` contains the
/// flattened decoded SC matrices and `all_features` contains the extracted
/// feature vectors.
pub fn run_crosscoder_simulation_pipeline_with_config<B: Backend>(
    config: CrossCoderPipelineConfig<B>,
) -> (Vec<f32>, Vec<f32>) {
    let CrossCoderPipelineConfig {
        model,
        prior,
        view_idx,
        template,
        nnodes,
        feature_set,
        n_samples,
        device,
        seed,
    } = config;

    let sc_matrices = generate_synthetic_sc_matrices(
        model, prior, view_idx, n_samples, device, seed,
    );

    let mut all_params = Vec::new();
    let mut all_features = Vec::new();

    for sc in &sc_matrices {
        let sim_cfg = build_sim_config_with_sc(template, sc, nnodes);
        let mut engine = match HybridEngine::<B>::from_config(sim_cfg, device.clone()) {
            Ok(e) => e,
            Err(err) => {
                log::warn!("Simulation config failed: {}, skipping sample", err);
                continue;
            }
        };

        let n_steps = (template.sim_length / template.dt) as usize;
        engine.run(n_steps);

        let nvar = engine.subnetworks[0].nvar;
        let nmodes = engine.subnetworks[0].nmodes;
        let traj = &engine.trajectory;
        let shape = vec![n_steps, nvar, nnodes, nmodes];

        let feats = extract_features_with(traj, &shape, feature_set);
        all_params.extend_from_slice(sc);
        all_features.extend_from_slice(&feats);
    }

    (all_params, all_features)
}

/// Backward-compatible wrapper around [`run_crosscoder_simulation_pipeline_with_config`].
#[allow(clippy::too_many_arguments)]
pub fn run_crosscoder_simulation_pipeline<B: Backend>(
    model: &CrossCoder<B>,
    prior: &MvnPrior,
    view_idx: usize,
    template: &SimConfig,
    nnodes: usize,
    feature_set: &FeatureSet,
    n_samples: usize,
    device: &B::Device,
    seed: Option<u64>,
) -> (Vec<f32>, Vec<f32>) {
    run_crosscoder_simulation_pipeline_with_config(CrossCoderPipelineConfig {
        model,
        prior,
        view_idx,
        template,
        nnodes,
        feature_set,
        n_samples,
        device,
        seed,
    })
}

/// End-to-end pipeline: generate synthetic cohort → simulate → extract
/// features → train MAF.
///
/// `maf_config` must have `param_dim == nnodes * nnodes` (flattened SC matrix).
/// Returns the trained MAF and its loss history.
pub fn train_maf_from_synthetic_cohort_with_config<B: Backend>(
    sim_config: CrossCoderPipelineConfig<B>,
    maf_config: &MafConfig,
    n_epochs: usize,
    batch_size: usize,
) -> MafTrainingResult {
    let nnodes = sim_config.nnodes;
    let param_dim = nnodes * nnodes;
    let (params, features) = run_crosscoder_simulation_pipeline_with_config(sim_config);

    let n_samples = params.len() / param_dim;
    log::info!(
        "Synthetic cohort: {} samples, param_dim={}, feature_dim={}",
        n_samples,
        param_dim,
        features.len() / n_samples,
    );

    train_maf_with_data_and_log(
        maf_config,
        params,
        features,
        n_epochs,
        batch_size,
    )
}

/// Backward-compatible wrapper around [`train_maf_from_synthetic_cohort_with_config`].
#[allow(clippy::too_many_arguments)]
pub fn train_maf_from_synthetic_cohort<B: Backend>(
    model: &CrossCoder<B>,
    prior: &MvnPrior,
    view_idx: usize,
    template: &SimConfig,
    nnodes: usize,
    feature_set: &FeatureSet,
    n_synth: usize,
    maf_config: &MafConfig,
    n_epochs: usize,
    batch_size: usize,
    device: &B::Device,
    seed: Option<u64>,
) -> MafTrainingResult {
    train_maf_from_synthetic_cohort_with_config(
        CrossCoderPipelineConfig {
            model,
            prior,
            view_idx,
            template,
            nnodes,
            feature_set,
            n_samples: n_synth,
            device,
            seed,
        },
        maf_config,
        n_epochs,
        batch_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use crate::sbi::crosscoder::CrossCoder;
    use crate::sbi::crosscoder_cohort::fit_mvn_over_latents;
    use crate::engine::IntegratorKind;

    type B = NdArray<f32>;

    #[test]
    fn test_generate_synthetic_sc() {
        let device = Default::default();
        let cc = CrossCoder::<B>::new(&device, &[9], 4, 1.0);
        let latents: Vec<f32> = (0..40).map(|i| (i as f32 * 0.1).sin()).collect();
        let (mean, cov) = fit_mvn_over_latents(&latents, 10, 4);
        let prior = MvnPrior::from_mean_cov(mean, cov, 4);
        let scs = generate_synthetic_sc_matrices(&cc, &prior, 0, 5, &device, Some(123)
        );
        assert_eq!(scs.len(), 5);
        assert_eq!(scs[0].len(), 9);
    }

    #[test]
    fn test_build_sim_config_with_sc() {
        let template = SimConfig {
            sim_length: 100.0,
            dt: 0.1,
            network: crate::config::NetworkConfig {
                subnetworks: vec![crate::config::SubnetworkConfig {
                    model: "Generic2dOscillator".to_string(),
                    nnodes: 2,
                    nmodes: 1,
                    initial_state: crate::config::InitialStateConfig::Inline(vec![0.0; 4]),
                    params: crate::model::g2do::g2do_default_params(),
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: 0.0,
            backend: "ndarray".to_string(),
        };
        let sc = vec![0.1, 0.2, 0.3, 0.4];
        let cfg = build_sim_config_with_sc(&template, &sc, 2);
        assert_eq!(cfg.network.projections.len(), 1);
    }
}
