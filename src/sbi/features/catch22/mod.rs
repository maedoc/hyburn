//! Pure Rust implementation of the catch22 time-series feature set.
//!
//! Reference: Lubba et al. (2019) "catch22: CAnonical Time-series CHaracteristics"
//! doi: 10.1002/cplx.21501
//!
//! All 22 features are computed on z-scored input data.
//! Internal computation uses f64 for numerical stability.
//!
//! # Feature Categories and Expected Ranges
//!
//! Features are organized into categories based on what aspect of dynamics they capture.
//! Approximate ranges are for typical z-scored time series of length 100-1000:
//!
//! ## Distributional (DN_*)
//! - `DN_OutlierInclude_*`: Timing of outliers, normalized by series length. Range: ~[0, 2]
//! - `DN_HistogramMode_*`: Mode of z-scored distribution. Range: ~[-2, 2]
//!
//! ## Correlation (CO_*)
//! - `CO_f1ecac`: Lag where AC drops below 1/e. Range: ~[1, N/10] (can be large for monotonic series)
//! - `CO_FirstMin_ac`: First AC minimum lag. Range: ~[0, N]
//! - `CO_HistogramAMI_even_2_5`: Auto-mutual information at lag 2. Range: ~[0, 3]
//! - `CO_trev_1_num`: Time-reversibility statistic. Range: ~[-1, 1]
//! - `CO_Embed2_Dist_*`: Embedding distance metric. Range: ~[0, 10]
//!
//! ## Forecasting (FC_*)
//! - `FC_LocalSimple_*`: Local forecasting error statistics. Range: ~[0, 2]
//!
//! ## Information (IN_*)
//! - `IN_AutoMutualInfoStats_*`: Auto-mutual info timescale. Range: ~[1, N/4]
//!
//! ## Heart Rate Variability (MD_*)
//! - `MD_hrv_classic_pnn40`: Fraction of large successive differences. Range: [0, 1]
//!
//! ## Symbolic Binary (SB_*)
//! - `SB_BinaryStats_*`: Statistics on binary-transformed series. Range: ~[0, N] (longest stretch)
//! - `SB_MotifThree_*`: Motif pattern entropy. Range: [0, ~1.1] (max is log(3))
//! - `SB_TransitionMatrix_*`: Transition matrix covariance. Range: ~[0, 1]
//!
//! ## Fluctuation Analysis (SC_*)
//! - `SC_FluctAnal_*`: Proportion of first scaling regime. Range: [0, 1]
//!
//! ## Spectral (SP_*)
//! - `SP_Summaries_welch_rect_*`: Power spectral density features. Area: ~[0, 1], Centroid: ~[0, 0.5] (normalized freq)
//!
//! ## Periodicity (PD_*)
//! - `PD_PeriodicityWang_*`: Detected period lag. Range: [0, N/3] (0 means no periodicity)
//!
//! # Numerical Stability
//!
//! For neural network training, features should be normalized to zero mean and unit variance
//! across samples. Use `sbi::normalize_features()` before training MAF/MADE models.
//!
//! Constant series are rejected with `FeatureError::Constant` since dynamical features
//! require temporal variation to be meaningful.

pub mod stats;
pub mod fft;
pub mod histogram;
pub mod helpers;
pub mod spline;
pub mod dn_outlier;
pub mod dn_histogram;
pub mod co_autocorr;
pub mod fc_local;
pub mod in_automutual;
pub mod md_hrv;
pub mod sb_binary;
pub mod sb_motif;
pub mod sc_fluct;
pub mod sp_spectral;
pub mod sb_transition;
pub mod pd_periodicity;

/// Canonical names for the 22 catch22 features (in computation order).
pub static CATCH22_NAMES: [&str; 22] = [
    "DN_OutlierInclude_n_001_mdrmd",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "FC_LocalSimple_mean1_tauresrat",
    "FC_LocalSimple_mean3_stderr",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_diff_longstretch0",
    "SB_BinaryStats_mean_longstretch1",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
];

/// Canonical names for the 24 catch24 features (catch22 + mean + std).
pub static CATCH24_NAMES: [&str; 24] = [
    "DN_Mean",
    "DN_Spread_Std",
    "DN_OutlierInclude_n_001_mdrmd",
    "DN_OutlierInclude_p_001_mdrmd",
    "DN_HistogramMode_5",
    "DN_HistogramMode_10",
    "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "CO_f1ecac",
    "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5",
    "CO_trev_1_num",
    "FC_LocalSimple_mean1_tauresrat",
    "FC_LocalSimple_mean3_stderr",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi",
    "MD_hrv_classic_pnn40",
    "SB_BinaryStats_diff_longstretch0",
    "SB_BinaryStats_mean_longstretch1",
    "SB_MotifThree_quantile_hh",
    "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1",
    "SP_Summaries_welch_rect_area_5_1",
    "SP_Summaries_welch_rect_centroid",
    "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01",
];

/// Error type for feature computation failures.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureError {
    /// Time series is too short (minimum 10 data points required)
    TooShort { len: usize, min: usize },
    /// Time series contains NaN or Inf values
    ContainsNonFinite { index: usize },
    /// Time series is constant (zero standard deviation)
    Constant,
}

impl std::fmt::Display for FeatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeatureError::TooShort { len, min } => {
                write!(f, "time series too short: {} points (minimum {})", len, min)
            }
            FeatureError::ContainsNonFinite { index } => {
                write!(f, "non-finite value at index {}", index)
            }
            FeatureError::Constant => write!(f, "constant time series (zero std)"),
        }
    }
}

impl std::error::Error for FeatureError {}

/// Validate a time series for catch22 computation.
///
/// Checks: minimum length (10 points), no NaN/Inf, non-constant.
fn quality_check(y: &[f64]) -> Result<(), FeatureError> {
    const MIN_LEN: usize = 10;
    if y.len() < MIN_LEN {
        return Err(FeatureError::TooShort {
            len: y.len(),
            min: MIN_LEN,
        });
    }
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(FeatureError::ContainsNonFinite { index: i });
        }
    }
    let std_dev = stats::stddev(y);
    if std_dev < f64::EPSILON {
        return Err(FeatureError::Constant);
    }
    Ok(())
}

/// Compute all 22 catch22 features on a time series.
///
/// The input is z-scored internally before feature computation.
/// Returns a Vec of 22 feature values in the canonical order (see CATCH22_NAMES).
pub fn catch22_features(y: &[f64]) -> Result<Vec<f64>, FeatureError> {
    quality_check(y)?;

    // Z-score the input once for all dynamical features
    let y_z = stats::zscore(y);

    // Compute all 22 features on the z-scored data
    Ok(vec![
        dn_outlier::dn_outlier_include_n(&y_z),
        dn_outlier::dn_outlier_include_p(&y_z),
        dn_histogram::dn_histogram_mode_5(&y_z),
        dn_histogram::dn_histogram_mode_10(&y_z),
        co_autocorr::co_embed2_dist_tau_d_expfit_meandiff(&y_z),
        co_autocorr::co_f1ecac(&y_z),
        co_autocorr::co_first_min_ac(&y_z) as f64,
        co_autocorr::co_histogram_ami_even_2_5(&y_z),
        co_autocorr::co_trev_1_num(&y_z),
        fc_local::fc_local_simple_mean1_tauresrat(&y_z),
        fc_local::fc_local_simple_mean3_stderr(&y_z),
        in_automutual::in_auto_mutual_info_stats_40_gaussian_fmmi(&y_z),
        md_hrv::md_hrv_classic_pnn40(&y_z),
        sb_binary::sb_binary_stats_diff_longstretch0(&y_z),
        sb_binary::sb_binary_stats_mean_longstretch1(&y_z),
        sb_motif::sb_motif_three_quantile_hh(&y_z),
        sc_fluct::sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1(&y_z),
        sc_fluct::sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1(&y_z),
        sp_spectral::sp_summaries_welch_rect_area_5_1(&y_z),
        sp_spectral::sp_summaries_welch_rect_centroid(&y_z),
        sb_transition::sb_transition_matrix_3ac_sumdiagcov(&y_z),
        pd_periodicity::pd_periodicity_wang_th0_01(&y_z) as f64,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_check_too_short() {
        let y = vec![1.0, 2.0, 3.0];
        assert!(matches!(
            catch22_features(&y),
            Err(FeatureError::TooShort { .. })
        ));
    }

    #[test]
    fn test_quality_check_constant() {
        let y = vec![5.0; 20];
        assert!(matches!(
            catch22_features(&y),
            Err(FeatureError::Constant)
        ));
    }

    #[test]
    fn test_quality_check_nan() {
        let mut y: Vec<f64> = (0..20).map(|i| i as f64).collect();
        y[5] = f64::NAN;
        assert!(matches!(
            catch22_features(&y),
            Err(FeatureError::ContainsNonFinite { .. })
        ));
    }

    #[test]
    fn test_catch22_sin_wave() {
        // Simple sinusoid — should compute without errors
        let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let features = catch22_features(&y).expect("catch22 on sin should succeed");
        assert_eq!(features.len(), 22);
        // All features should be finite
        for (i, &f) in features.iter().enumerate() {
            assert!(
                f.is_finite(),
                "feature {} ({}) = {} is not finite",
                i,
                CATCH22_NAMES[i],
                f
            );
        }
    }

    #[test]
    fn test_quality_check_inf() {
        let mut y: Vec<f64> = (0..20).map(|i| i as f64).collect();
        y[5] = f64::INFINITY;
        assert!(matches!(
            catch22_features(&y),
            Err(FeatureError::ContainsNonFinite { .. })
        ));
    }

    #[test]
    fn test_quality_check_empty() {
        let y: Vec<f64> = vec![];
        assert!(matches!(
            catch22_features(&y),
            Err(FeatureError::TooShort { .. })
        ));
    }

    #[test]
    fn test_quality_check_single_element() {
        let y = vec![1.0];
        assert!(matches!(
            catch22_features(&y),
            Err(FeatureError::TooShort { .. })
        ));
    }

    #[test]
    fn test_quality_check_min_valid_length() {
        // Exactly MIN_LEN (10) points should work
        let y: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let result = catch22_features(&y);
        assert!(result.is_ok(), "min valid length should work");
    }

    #[test]
    fn test_catch22_random_walk() {
        // Random walk — should compute without errors
        let mut y = vec![0.0; 200];
        for i in 1..200 {
            y[i] = y[i - 1] + 0.1 * ((i as f64 * 0.3).sin());
        }
        let features = catch22_features(&y).expect("catch22 on random walk should succeed");
        assert_eq!(features.len(), 22);
        for (i, &f) in features.iter().enumerate() {
            assert!(
                f.is_finite(),
                "feature {} ({}) = {} is not finite",
                i,
                CATCH22_NAMES[i],
                f
            );
        }
    }
}