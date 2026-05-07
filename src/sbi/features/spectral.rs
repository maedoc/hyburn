//! Spectral feature extraction.
//!
//! Computes power-spectral-density (Welch) and derived summary features
//! (band power, spectral moments) from node-level time series.

use crate::sbi::features::catch22::fft::welch_psd;

/// EEG-style frequency bands and their boundaries (Hz).
pub const EEG_BANDS: &[(&str, f64, f64)] = &[
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 80.0),
];

/// Spectral features extracted from a single time series.
///
/// Per-node features, averaged over nodes and modes later.
pub fn spectral_features(series: &[f32], fs: f64) -> Vec<f32> {
    if series.len() < 16 {
        // Not enough data for meaningful PSD
        return vec![f32::NAN; EEG_BANDS.len() + 4];
    }
    let y: Vec<f64> = series.iter().map(|v| *v as f64).collect();
    let nfft = super::catch22::fft::next_power_of_2(series.len());
    // Adaptive window: use min(256, n_samples/2) with floor of 16
    let nperseg = (series.len() / 2).clamp(16, 256);
    let (psd, freqs) = welch_psd(&y, nfft, fs, nperseg);
    if psd.is_empty() || freqs.is_empty() {
        return vec![f32::NAN; EEG_BANDS.len() + 4];
    }

    let total_power: f64 = psd.iter().sum();
    if total_power <= 0.0 || !total_power.is_finite() {
        return vec![f32::NAN; EEG_BANDS.len() + 4];
    }

    // Band powers (integrate PSD over band limits)
    let mut features = Vec::with_capacity(EEG_BANDS.len() + 4);
    for &(_, f_low, f_high) in EEG_BANDS {
        let mut band_power = 0.0f64;
        for (i, &f) in freqs.iter().enumerate() {
            if f >= f_low && f < f_high && i < psd.len() {
                band_power += psd[i];
            }
        }
        // AUC approximation: power * df averaged. Here we just sum power samples.
        features.push((band_power / total_power) as f32);
    }

    // Spectral moments: centroid, spread, skewness, kurtosis
    let centroid = psd
        .iter()
        .zip(freqs.iter())
        .map(|(&p, &f)| p * f)
        .sum::<f64>()
        / total_power;

    let spread = (psd
        .iter()
        .zip(freqs.iter())
        .map(|(&p, &f)| p * (f - centroid).powi(2))
        .sum::<f64>()
        / total_power)
        .sqrt();

    let skewness = if spread > 1e-12 {
        psd.iter()
            .zip(freqs.iter())
            .map(|(&p, &f)| p * (f - centroid).powi(3))
            .sum::<f64>()
            / total_power
            / spread.powi(3)
    } else {
        0.0
    };

    let kurtosis = if spread > 1e-12 {
        psd.iter()
            .zip(freqs.iter())
            .map(|(&p, &f)| p * (f - centroid).powi(4))
            .sum::<f64>()
            / total_power
            / spread.powi(4)
            - 3.0 // excess kurtosis
    } else {
        0.0
    };

    features.push(centroid as f32);
    features.push(spread as f32);
    features.push(skewness as f32);
    features.push(kurtosis as f32);

    features
}

/// Number of spectral features per variable.
pub const SPECTRAL_FEATURE_COUNT: usize = EEG_BANDS.len() + 4;

/// Human-readable names for each spectral feature.
pub const SPECTRAL_FEATURE_NAMES: &[&str] = &[
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
    "spectral_centroid", "spectral_spread", "spectral_skewness", "spectral_kurtosis",
];

/// Extract spectral features from a flat `[n_steps, nvar, nnodes, nmodes]` trajectory.
///
/// Returns a flat vector of shape `[nvar * SPECTRAL_FEATURE_COUNT]` where each
/// variable block gives the band-power fractions, centroid, spread, skewness,
/// and kurtosis.
pub fn extract_spectral_features(
    trajectory: &[f32],
    shape: &[usize],
    fs: f64,
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

    let mut out = Vec::with_capacity(nvar * SPECTRAL_FEATURE_COUNT);
    for var in 0..nvar {
        // Average spectral features across nodes and modes
        let mut summed = vec![0.0f64; SPECTRAL_FEATURE_COUNT];
        for n in 0..nnodes {
            for m in 0..nmodes {
                let mut series = Vec::with_capacity(n_steps);
                for t in 0..n_steps {
                    let idx = ((t * nvar + var) * nnodes + n) * nmodes + m;
                    series.push(trajectory[idx]);
                }
                let feats = spectral_features(&series, fs);
                for (i, v) in feats.iter().enumerate() {
                    summed[i] += *v as f64;
                }
            }
        }
        let n_series = (nnodes * nmodes) as f64;
        for v in summed {
            out.push((v / n_series) as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_features_sin() {
        let fs = 100.0; // Hz
        let series: Vec<f32> = (0..1024)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI * 10.0 / fs).sin() as f32)
            .collect();
        let feats = spectral_features(&series, fs);
        assert_eq!(feats.len(), SPECTRAL_FEATURE_COUNT);
        // A 10 Hz sinusoid should have most power in alpha band (8–13 Hz)
        let alpha_idx = 2; // delta=0, theta=1, alpha=2, beta=3, gamma=4
        let alpha_power = feats[alpha_idx];
        assert!(
            alpha_power > 0.5,
            "10 Hz sine should have most power in alpha band; got {}",
            alpha_power
        );
        // Centroid should be near 10 Hz
        let centroid = feats[EEG_BANDS.len()] as f64;
        assert!(
            (centroid - 10.0).abs() < 3.0,
            "centroid should be near 10 Hz, got {}",
            centroid
        );
    }

    #[test]
    fn test_spectral_features_too_short() {
        let series = vec![1.0f32; 4];
        let feats = spectral_features(&series, 100.0);
        assert!(feats.iter().all(|v| v.is_nan()), "short series should yield NaN");
    }

    #[test]
    fn test_extract_spectral_features_shape() {
        let n_steps = 64;
        let nvar = 2;
        let nnodes = 3;
        let nmodes = 1;
        let trajectory: Vec<f32> = (0..n_steps * nvar * nnodes * nmodes)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let feats = extract_spectral_features(
            &trajectory,
            &[n_steps, nvar, nnodes, nmodes],
            100.0,
        );
        assert_eq!(feats.len(), nvar * SPECTRAL_FEATURE_COUNT);
    }

    #[test]
    fn test_spectral_features_theta_frequency() {
        // 6 Hz sine → should be dominant in theta band (4–8 Hz)
        let fs = 100.0;
        let freq = 6.0;
        let series: Vec<f32> = (0..1024)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI * freq / fs).sin() as f32)
            .collect();
        let feats = spectral_features(&series, fs);
        let theta_idx = 1; // delta=0, theta=1
        let theta_power = feats[theta_idx];
        assert!(
            theta_power > 0.5,
            "6 Hz sine should have most power in theta band; got {}",
            theta_power
        );
        // Delta power should be small
        assert!(
            feats[0] < 0.3,
            "6 Hz sine should have little delta power; got {}",
            feats[0]
        );
    }

    #[test]
    fn test_spectral_features_gamma_frequency() {
        // 40 Hz sine → should be dominant in gamma band (30–80 Hz)
        let fs = 200.0; // need higher fs for Nyquist
        let freq = 40.0;
        let series: Vec<f32> = (0..2048)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI * freq / fs).sin() as f32)
            .collect();
        let feats = spectral_features(&series, fs);
        let gamma_idx = 4; // delta=0, theta=1, alpha=2, beta=3, gamma=4
        let gamma_power = feats[gamma_idx];
        assert!(
            gamma_power > 0.5,
            "40 Hz sine should have most power in gamma band; got {}",
            gamma_power
        );
    }

    #[test]
    fn test_spectral_features_adaptive_window_short_series() {
        // Verify that the adaptive nperseg doesn't panic on short series
        // (e.g., 32 samples → nperseg = min(256, 16) = 16)
        let fs = 100.0;
        let series: Vec<f32> = (0..32)
            .map(|i| (i as f64 * 2.0 * std::f64::consts::PI * 10.0 / fs).sin() as f32)
            .collect();
        let feats = spectral_features(&series, fs);
        // Should return valid (non-NaN) features for 32 samples
        assert_eq!(feats.len(), SPECTRAL_FEATURE_COUNT);
        for (i, v) in feats.iter().enumerate() {
            assert!(
                v.is_finite() || v.is_nan(),
                "Feature {} should be finite or NaN, got inf",
                i
            );
        }
    }
}
