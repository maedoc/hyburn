//! Feature extraction from simulation trajectories.
//!
//! Supports multiple feature sets:
//! - **Classic**: mean, variance, lag-1 autocorrelation (original hyburn)
//! - **Catch22**: all 22 hctsa-derived dynamical features
//! - **Catch24**: catch22 + mean + standard deviation

pub mod catch22;
pub mod classic;
pub mod fc;
pub mod spectral;
pub mod temporal;

use serde::{Deserialize, Serialize};

/// Feature domain taxonomy matching vbi/vbjax conventions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum FeatureDomain {
    /// Time-domain summary statistics (mean, variance, autocorrelation, etc.)
    Temporal,
    /// Frequency-domain features (PSD, spectral moments, band power)
    Spectral,
    /// Functional-connectivity and network-level features
    Connectivity,
    /// Distributional statistics (skewness, kurtosis, burstiness)
    Statistical,
    /// Information-theoretic quantities (entropy, MI, TE)
    InformationTheoretic,
    /// Hidden-Markov-model derived features
    Hmm,
    /// Spans multiple domains (e.g. a combined feature set)
    MultiDomain,
}

impl FeatureDomain {
    /// Human-readable domain name.
    pub fn name(&self) -> &'static str {
        match self {
            FeatureDomain::Temporal => "temporal",
            FeatureDomain::Spectral => "spectral",
            FeatureDomain::Connectivity => "connectivity",
            FeatureDomain::Statistical => "statistical",
            FeatureDomain::InformationTheoretic => "information_theoretic",
            FeatureDomain::Hmm => "hmm",
            FeatureDomain::MultiDomain => "multi_domain",
        }
    }
}

/// Available feature extraction methods.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FeatureSet {
    /// Original 3 features per variable: mean, variance, lag-1 autocorrelation
    #[default]
    Classic,
    /// 22 dynamical features from the catch22 set
    Catch22,
    /// 22 catch22 features + mean + standard deviation
    Catch24,
    /// Functional-connectivity summary statistics (7 features per variable)
    Fc,
    /// Spectral features: EEG band-power fractions + spectral moments (9 per variable)
    Spectral,
    /// Temporal and statistical summary features (7 per variable)
    TemporalStat,
    /// Multi-domain combined feature set (concatenation of multiple feature sets).
    Combined(Vec<FeatureSet>),
}

impl FeatureSet {
    /// Number of features per state variable for this feature set.
    pub fn features_per_var(&self) -> usize {
        match self {
            FeatureSet::Classic => 3,
            FeatureSet::Catch22 => 22,
            FeatureSet::Catch24 => 24,
            FeatureSet::Fc => 7,
            FeatureSet::Spectral => spectral::SPECTRAL_FEATURE_COUNT,
            FeatureSet::TemporalStat => temporal::TEMPORAL_STAT_FEATURE_COUNT,
            FeatureSet::Combined(parts) => parts.iter().map(|p| p.features_per_var()).sum(),
        }
    }

    /// Human-readable names for each feature per variable.
    pub fn feature_names(&self) -> Vec<&'static str> {
        match self {
            FeatureSet::Classic => vec!["mean", "var", "lag1_ac"],
            FeatureSet::Catch22 => catch22::CATCH22_NAMES.to_vec(),
            FeatureSet::Catch24 => catch22::CATCH24_NAMES.to_vec(),
            FeatureSet::Fc => vec![
                "fc_mean",
                "fc_std",
                "fc_min",
                "fc_max",
                "fc_median",
                "fc_homotopic_mean",
                "fc_homotopic_std",
            ],
            FeatureSet::Spectral => spectral::SPECTRAL_FEATURE_NAMES.to_vec(),
            FeatureSet::TemporalStat => temporal::TEMPORAL_STAT_FEATURE_NAMES.to_vec(),
            FeatureSet::Combined(parts) => {
                let mut names = Vec::new();
                for p in parts {
                    names.extend_from_slice(&p.feature_names());
                }
                names
            }
        }
    }

    /// Total feature dimension given number of state variables, nodes, and modes.
    ///
    /// For Classic and Catch22/Catch24, features are averaged across nodes/modes,
    /// so the dimension is `nvar * features_per_var()`.
    pub fn feature_dim(&self, nvar: usize, _nnodes: usize, _nmodes: usize) -> usize {
        // Currently all feature sets average across spatial dimensions
        nvar * self.features_per_var()
    }

    /// Feature domain classification matching the vbi taxonomy.
    pub fn domain(&self) -> FeatureDomain {
        match self {
            FeatureSet::Classic | FeatureSet::Catch22 | FeatureSet::Catch24 => {
                FeatureDomain::Temporal
            }
            FeatureSet::Fc => FeatureDomain::Connectivity,
            FeatureSet::Spectral => FeatureDomain::Spectral,
            FeatureSet::TemporalStat => FeatureDomain::Statistical,
            FeatureSet::Combined(_) => FeatureDomain::MultiDomain,
        }
    }
}

/// Parse a feature set name string into a `FeatureSet` enum.
///
/// Supports: "classic", "catch22", "catch24", "fc", "spectral",
/// "temporal_stat", and "combined:classic,catch22" prefix notation.
pub fn parse_feature_set(s: &str) -> Option<FeatureSet> {
    let s = s.trim().to_lowercase();
    match s.as_str() {
        "classic" => Some(FeatureSet::Classic),
        "catch22" => Some(FeatureSet::Catch22),
        "catch24" => Some(FeatureSet::Catch24),
        "fc" => Some(FeatureSet::Fc),
        "spectral" => Some(FeatureSet::Spectral),
        "temporal_stat" => Some(FeatureSet::TemporalStat),
        other if other.starts_with("combined:") => {
            let parts: Vec<FeatureSet> = other["combined:".len()..]
                .split(',')
                .filter_map(|p| parse_feature_set(p.trim()))
                .collect();
            if parts.is_empty() { None } else { Some(FeatureSet::Combined(parts)) }
        }
        _ => None,
    }
}

/// Extract features from a flat simulation trajectory.
///
/// `trajectory` is a flat `[n_steps, nvar, nnodes, nmodes]` array.
/// This is the backward-compatible entry point using Classic features.
pub fn extract_features(trajectory: &[f32], shape: &[usize]) -> Vec<f32> {
    extract_features_with(trajectory, shape, &FeatureSet::Classic)
}

/// Extract features from a flat simulation trajectory using the specified feature set.
///
/// `trajectory` is a flat `[n_steps, nvar, nnodes, nmodes]` array.
/// For each state variable, the time series is extracted and features are computed,
/// then averaged over nodes and modes.
pub fn extract_features_with(
    trajectory: &[f32],
    shape: &[usize],
    feature_set: &FeatureSet,
) -> Vec<f32> {
    assert_eq!(
        shape.len(),
        4,
        "expected trajectory shape [n_steps, nvar, nnodes, nmodes]"
    );
    let n_steps = shape[0];
    let nvar = shape[1];
    let nnodes = shape[2];
    let nmodes = shape[3];

    match feature_set {
        FeatureSet::Classic => classic::extract_features_classic(trajectory, shape),
        FeatureSet::Catch22 | FeatureSet::Catch24 => {
            let include_mean_std = matches!(feature_set, FeatureSet::Catch24);
            let mut features = Vec::with_capacity(feature_set.feature_dim(nvar, nnodes, nmodes));

            for var in 0..nvar {
                // Collect all (node, mode) time series for this variable
                let mut per_series_features = Vec::new();

                for n in 0..nnodes {
                    for m in 0..nmodes {
                        let mut series = Vec::with_capacity(n_steps);
                        for t in 0..n_steps {
                            let idx = ((t * nvar + var) * nnodes + n) * nmodes + m;
                            series.push(trajectory[idx] as f64);
                        }

                        let feat = match catch22::catch22_features(&series) {
                            Ok(mut f) => {
                                if include_mean_std {
                                    // Prepend mean and std (computed on raw, not z-scored data)
                                    let mean = series.iter().sum::<f64>() / series.len() as f64;
                                    let variance =
                                        series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                                            / series.len() as f64;
                                    let std = variance.sqrt();
                                    let mut full = vec![mean, std];
                                    full.append(&mut f);
                                    full
                                } else {
                                    f
                                }
                            }
                            Err(catch22::FeatureError::Constant) => {
                                // Constant series: mean/std are valid, dynamical features are NaN
                                let n = if include_mean_std { 24 } else { 22 };
                                let mut f = vec![f64::NAN; n];
                                if include_mean_std {
                                    // Mean and std are still meaningful for constant series
                                    let mean = series.iter().sum::<f64>() / series.len() as f64;
                                    f[0] = mean;
                                    f[1] = 0.0; // std is zero for constant series
                                }
                                f
                            }
                            Err(_) => {
                                // Other errors (too short, non-finite): all NaN
                                let n = if include_mean_std { 24 } else { 22 };
                                vec![f64::NAN; n]
                            }
                        };
                        per_series_features.push(feat);
                    }
                }

                // Average features across nodes and modes
                let n_series = nnodes * nmodes;
                let n_features = if include_mean_std { 24 } else { 22 };
                for f_idx in 0..n_features {
                    let sum: f64 = per_series_features
                        .iter()
                        .map(|s: &Vec<f64>| s[f_idx])
                        .sum();
                    features.push((sum / n_series as f64) as f32);
                }
            }

            features
        }
        FeatureSet::Fc => {
            let mut features = Vec::with_capacity(feature_set.feature_dim(nvar, nnodes, nmodes));
            for var in 0..nvar {
                // Average over nmodes to get [n_steps, nnodes]
                let mut node_series = vec![0.0f32; n_steps * nnodes];
                for t in 0..n_steps {
                    for n in 0..nnodes {
                        let mut sum = 0.0f32;
                        for m in 0..nmodes {
                            let idx = ((t * nvar + var) * nnodes + n) * nmodes + m;
                            sum += trajectory[idx];
                        }
                        node_series[t * nnodes + n] = sum / nmodes as f32;
                    }
                }
                let fc = fc::fc_matrix(&node_series, n_steps, nnodes);
                let stats = fc::fc_stats(&fc, nnodes);
                let (h_mean, h_std, _) = fc::homotopic_fc(&fc, nnodes);
                features.extend_from_slice(&stats);
                features.push(h_mean);
                features.push(h_std);
            }
            features
        }
        FeatureSet::Spectral => {
            // Default sampling rate: 1000 Hz (dt = 1.0 ms).  Callers requiring
            // a different fs can use `spectral::extract_spectral_features` directly.
            spectral::extract_spectral_features(trajectory, shape, 1000.0)
        }
        FeatureSet::TemporalStat => {
            temporal::extract_temporal_stat_features(trajectory, shape)
        }
        FeatureSet::Combined(parts) => {
            let nvar = shape[1];
            let mut per_var: Vec<Vec<f32>> = vec![Vec::new(); nvar];
            for part in parts.iter() {
                let sub = extract_features_with(trajectory, shape, part);
                let fpvar = part.features_per_var();
                for (v, per_var_v) in per_var.iter_mut().enumerate().take(nvar) {
                    let start = v * fpvar;
                    let end = start + fpvar;
                    per_var_v.extend_from_slice(&sub[start..end]);
                }
            }
            per_var.into_iter().flatten().collect()
        }
    }
}

// Re-export the main entry points
pub use classic::extract_features_classic;

/// Normalize features to zero mean and unit variance per feature dimension.
///
/// `features` is a flat array of shape `[n_samples, feature_dim]` in row-major order.
/// Returns a new normalized array and the (means, stds) used for normalization.
///
/// This is essential for catch22 features which have vastly different scales
/// (e.g., SB_BinaryStats can reach 468 while others are ~0-1).
pub fn normalize_features(features: &[f32], n_samples: usize, feature_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut normalized = features.to_vec();
    let mut means = vec![0.0f32; feature_dim];
    let mut stds = vec![0.0f32; feature_dim];

    // Compute means
    for d in 0..feature_dim {
        let mut sum = 0.0f32;
        for i in 0..n_samples {
            sum += features[i * feature_dim + d];
        }
        means[d] = sum / n_samples as f32;
    }

    // Compute stds
    for d in 0..feature_dim {
        let mut sum_sq = 0.0f32;
        for i in 0..n_samples {
            let diff = features[i * feature_dim + d] - means[d];
            sum_sq += diff * diff;
        }
        stds[d] = (sum_sq / n_samples as f32).sqrt().max(1e-8);
    }

    // Normalize
    for i in 0..n_samples {
        for d in 0..feature_dim {
            normalized[i * feature_dim + d] = (features[i * feature_dim + d] - means[d]) / stds[d];
        }
    }

    (normalized, means, stds)
}

/// Apply saved normalization to new features (for inference).
pub fn apply_normalization(features: &[f32], means: &[f32], stds: &[f32]) -> Vec<f32> {
    let feature_dim = means.len();
    let n_samples = features.len() / feature_dim;
    let mut normalized = features.to_vec();

    for i in 0..n_samples {
        for d in 0..feature_dim {
            normalized[i * feature_dim + d] = (features[i * feature_dim + d] - means[d]) / stds[d];
        }
    }

    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_set_domain() {
        assert_eq!(FeatureSet::Classic.domain(), FeatureDomain::Temporal);
        assert_eq!(FeatureSet::Catch22.domain(), FeatureDomain::Temporal);
        assert_eq!(FeatureSet::Catch24.domain(), FeatureDomain::Temporal);
        assert_eq!(FeatureSet::Fc.domain(), FeatureDomain::Connectivity);
    }

    #[test]
    fn test_feature_set_fc_names() {
        let names = FeatureSet::Fc.feature_names();
        assert_eq!(names.len(), 7);
        assert_eq!(names[0], "fc_mean");
        assert_eq!(names[5], "fc_homotopic_mean");
    }

    #[test]
    fn test_feature_set_fc_dim() {
        assert_eq!(FeatureSet::Fc.feature_dim(2, 4, 1), 14); // 2 vars * 7 features each
    }

    #[test]
    fn test_feature_set_temporalstat_names() {
        let names = FeatureSet::TemporalStat.feature_names();
        assert_eq!(names.len(), 7);
        assert_eq!(names[0], "abs_energy");
        assert_eq!(names[6], "burstiness");
    }

    #[test]
    fn test_feature_set_temporalstat_domain() {
        assert_eq!(FeatureSet::TemporalStat.domain(), FeatureDomain::Statistical);
    }

    #[test]
    fn test_extract_features_temporalstat() {
        let n_steps = 32;
        let nvar = 1;
        let nnodes = 2;
        let nmodes = 1;
        let trajectory: Vec<f32> = (0..n_steps * nvar * nnodes * nmodes)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let features = extract_features_with(
            &trajectory,
            &[n_steps, nvar, nnodes, nmodes],
            &FeatureSet::TemporalStat,
        );
        assert_eq!(features.len(), 7);
        assert!(features.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_extract_features_fc() {
        let trajectory: Vec<f32> = (0..15).map(|i| (i as f32 * 0.1).sin()).collect();
        let features = extract_features_with(
            &trajectory,
            &[5, 1, 3, 1],
            &FeatureSet::Fc,
        );
        assert_eq!(features.len(), 7, "Fc should produce 7 features per variable");
        assert_eq!(FeatureSet::Fc.feature_names().len(), 7);
        assert_eq!(FeatureSet::Fc.features_per_var(), 7);
        assert!(features[0].is_finite(), "fc_mean should be finite");
    }

    #[test]
    fn test_combined_features() {
        let n_steps = 32;
        let nvar = 2;
        let nnodes = 3;
        let nmodes = 1;
        let trajectory: Vec<f32> = (0..n_steps * nvar * nnodes * nmodes)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let combined = FeatureSet::Combined(vec![
            FeatureSet::Classic,
            FeatureSet::Fc,
        ]);
        let features = extract_features_with(
            &trajectory,
            &[n_steps, nvar, nnodes, nmodes],
            &combined,
        );
        let expected_dim = nvar * (3 + 7);
        assert_eq!(features.len(), expected_dim, "combined classic+fc should yield 10 features per var");
        assert_eq!(combined.features_per_var(), 10);
        assert_eq!(combined.domain(), FeatureDomain::MultiDomain);
        assert_eq!(combined.feature_names().len(), 10);
        // All features should be finite
        assert!(features.iter().all(|v| v.is_finite()));
    }
}