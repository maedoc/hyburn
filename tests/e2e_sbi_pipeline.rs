#![cfg(not(target_arch = "wasm32"))]
use burn::backend::autodiff::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::tensor::{Tensor, TensorData};

use hyburn::engine::{EngineModel, HybridEngine, IntegratorKind};
use hyburn::model::g2do::g2do_default_params;
use hyburn::sbi::{extract_features_with, normalize_features, FeatureSet, MafConfig, SbiDiagnostics};

// Feature normalization is now handled by hyburn::sbi::normalize_features
// This local helper is kept for reference
#[allow(dead_code)]
fn _normalize_features_local(features: &mut [f32], n_samples: usize, feature_dim: usize) {
    let mut means = vec![0.0f32; feature_dim];
    let mut stds = vec![0.0f32; feature_dim];

    for d in 0..feature_dim {
        let mut sum = 0.0f32;
        for i in 0..n_samples {
            sum += features[i * feature_dim + d];
        }
        means[d] = sum / n_samples as f32;

        let mut sum_sq = 0.0f32;
        for i in 0..n_samples {
            let diff = features[i * feature_dim + d] - means[d];
            sum_sq += diff * diff;
        }
        stds[d] = (sum_sq / n_samples as f32).sqrt().max(1e-8);
    }

    for i in 0..n_samples {
        for d in 0..feature_dim {
            features[i * feature_dim + d] = (features[i * feature_dim + d] - means[d]) / stds[d];
        }
    }
}

fn run_sweep_and_train_with_features(feature_set: &FeatureSet) {
    type B = NdArray<f32>;
    let device = Default::default();

    let n_sweep = 20;
    let n_steps = 500;
    let nnodes = 2;
    let nmodes = 1;
    let nvar = 2;

    let mut all_params: Vec<f32> = Vec::with_capacity(n_sweep);
    let mut all_features: Vec<f32> = Vec::new();

    // 1. Sweep over I_ext (params[1]) from -0.5 to 0.5
    for i in 0..n_sweep {
        let i_ext = -0.5f32 + i as f32 * (1.0f32 / 19.0f32);

        let mut params = g2do_default_params();
        params[1] = i_ext;

        let initial_data = vec![0.0f32; nvar * nnodes * nmodes];
        let state = Tensor::<B, 3>::from_data(
            TensorData::new::<f32, Vec<usize>>(
                initial_data,
                vec![nvar, nnodes, nmodes],
            ),
            &device,
        );

        let model = EngineModel::<B>::G2do { params };
        let mut engine = HybridEngine::new(state, model, IntegratorKind::Euler, 0.1, 1, device.clone());
        engine.run(n_steps);

        let features = extract_features_with(&engine.trajectory,
            &[n_steps, nvar, nnodes, nmodes],
            feature_set,
        );

        all_params.push(i_ext);
        all_features.extend_from_slice(&features);
    }

    let feature_dim = all_features.len() / n_sweep;

    // Normalize features for catch22 (essential for numerical stability)
    let (normalized_features, _means, _stds) = match feature_set {
        FeatureSet::Classic => (all_features.clone(), vec![], vec![]),
        FeatureSet::Catch22 | FeatureSet::Catch24 => {
            normalize_features(&all_features, n_sweep, feature_dim)
        }
    };

    // 2. Train MAF — scale hidden units for catch22's larger feature space
    let hidden_units = match feature_set {
        FeatureSet::Classic => 16,
        FeatureSet::Catch22 => 128,
        FeatureSet::Catch24 => 128,
    };

    let maf_config = MafConfig {
        param_dim: 1,
        feature_dim,
        hidden_units,
        n_flows: 4,
        learning_rate: 1e-3,
        feature_set: format!("{:?}", feature_set).to_lowercase(),
    };

    let maf = hyburn::sbi::train_maf_with_data(
        &maf_config,
        all_params.clone(),
        normalized_features.clone(),
        300,
        5,
    );

    // 3. Validate using SBI diagnostics (z-score + shrinkage)
    let n_post_samples = 100;
    let device_ad = Default::default();

    // Prior: uniform I_ext in [-0.5, 0.5] → mean=0, std=1/sqrt(12) ≈ 0.2887
    let prior_mean = 0.0f32;
    let prior_std = (1.0f32 / 12.0f32).sqrt(); // uniform [-0.5, 0.5] std ≈ 0.2887

    let mut all_posterior_samples: Vec<f32> = Vec::new();
    let mut all_true_params: Vec<f32> = Vec::new();

    for (i, &true_i_ext) in all_params.iter().enumerate() {
        let f_start = i * feature_dim;
        let features_slice = &normalized_features[f_start..f_start + feature_dim];

        let context = Tensor::<Autodiff<NdArray<f32>>, 2>::from_data(
            TensorData::new::<f32, Vec<usize>>(
                features_slice.to_vec(),
                vec![1, feature_dim],
            ),
            &device_ad,
        );

        let samples = maf.inverse_sample(context, n_post_samples);
        let data = samples.into_data();
        let slice = data.as_slice::<f32>().unwrap();

        all_posterior_samples.extend_from_slice(slice);
        all_true_params.push(true_i_ext);

        // Per-point validation
        let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
        let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / slice.len() as f32;
        let std = var.sqrt();

        assert!(
            (mean - true_i_ext).abs() < 0.4,
            "Posterior mean {:.3} too far from true I_ext {:.3} at sweep index {}",
            mean,
            true_i_ext,
            i
        );
        assert!(
            std < 0.6,
            "Posterior std {:.3} too large at sweep index {}",
            std,
            i
        );
    }

    // 4. Compute and validate SBI diagnostics
    let diagnostics = SbiDiagnostics::from_samples(
        &all_posterior_samples,
        &all_true_params,
        &[prior_mean],
        &[prior_std],
        n_post_samples,
        1, // param_dim = 1 (only I_ext)
    );

    eprintln!("{}", diagnostics.report());

    assert!(
        diagnostics.is_well_calibrated(),
        "SBI diagnostics failed: {}",
        diagnostics.report()
    );
}

#[test]
fn test_e2e_sbi_sweep_train_validate_classic() {
    run_sweep_and_train_with_features(&FeatureSet::Classic);
}

#[test]
fn test_e2e_sbi_sweep_train_validate_catch22() {
    run_sweep_and_train_with_features(&FeatureSet::Catch22);
}
