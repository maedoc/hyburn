//! Temporal and simple statistical feature extraction.
//!
//! Computes per-node time-series summary statistics:
//!   * absolute energy  – Σ x²
//!   * average power    – Σ x² / n
//!   * temporal centroid – Σ t·|x| / Σ |x|
//!   * zero-crossing rate  – count of sign changes / (n-1)
//!   * skewness          – 3rd standardized moment
//!   * kurtosis (excess) – 4th standardized moment minus 3
//!   * burstiness        – (σ - μ) / (σ + μ)  [Goh & Barabási]

/// Extract the 7 temporal / statistical features from a single time series.
pub fn temporal_stat_features(series: &[f32]) -> Vec<f32> {
    let n = series.len();
    if n == 0 {
        return vec![f32::NAN; 7];
    }

    let mean = series.iter().copied().sum::<f32>() / n as f32;

    // Variance, skewness, kurtosis accumulators
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    let mut m4 = 0.0f64;
    let mut abs_energy = 0.0f64;
    let mut abs_sum = 0.0f64;
    let mut weighted_t = 0.0f64;

    for (t, &x) in series.iter().enumerate() {
        let d = x as f64 - mean as f64;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
        let x64 = x as f64;
        abs_energy += x64 * x64;
        let abs_x = x64.abs();
        abs_sum += abs_x;
        weighted_t += t as f64 * abs_x;
    }

    let variance = m2 / n as f64;
    let std = variance.sqrt();
    let skewness = if std > 1e-12 {
        (m3 / n as f64) / std.powi(3)
    } else {
        0.0
    };
    let kurtosis = if std > 1e-12 {
        (m4 / n as f64) / std.powi(4) - 3.0
    } else {
        0.0
    };

    let centroid = if abs_sum > 1e-12 {
        weighted_t / abs_sum
    } else {
        0.0
    };

    // Zero-crossing rate
    let mut zc = 0usize;
    for i in 1..n {
        if series[i - 1] * series[i] < 0.0 {
            zc += 1;
        }
    }
    let zcr = if n > 1 {
        zc as f64 / (n - 1) as f64
    } else {
        0.0
    };

    let avg_power = abs_energy / n as f64;

    let burstiness = if std + mean as f64 > 1e-12 {
        (std - mean as f64) / (std + mean as f64)
    } else {
        0.0
    };

    vec![
        abs_energy as f32,
        avg_power as f32,
        centroid as f32,
        zcr as f32,
        skewness as f32,
        kurtosis as f32,
        burstiness as f32,
    ]
}

/// Number of temporal/statistical features per variable.
pub const TEMPORAL_STAT_FEATURE_COUNT: usize = 7;

/// Human-readable names for each temporal/statistical feature.
pub const TEMPORAL_STAT_FEATURE_NAMES: &[&str] = &[
    "abs_energy",
    "average_power",
    "temporal_centroid",
    "zero_crossing_rate",
    "skewness",
    "kurtosis",
    "burstiness",
];

/// Extract temporal/statistical features from a flat `[n_steps, nvar, nnodes, nmodes]` trajectory.
///
/// Features are averaged across nodes and modes, yielding `nvar * 7` values.
pub fn extract_temporal_stat_features(trajectory: &[f32], shape: &[usize]) -> Vec<f32> {
    assert_eq!(
        shape.len(),
        4,
        "expected trajectory shape [n_steps, nvar, nnodes, nmodes]"
    );
    let n_steps = shape[0];
    let nvar = shape[1];
    let nnodes = shape[2];
    let nmodes = shape[3];

    let mut out = Vec::with_capacity(nvar * TEMPORAL_STAT_FEATURE_COUNT);
    for var in 0..nvar {
        let mut summed = vec![0.0f64; TEMPORAL_STAT_FEATURE_COUNT];
        for n in 0..nnodes {
            for m in 0..nmodes {
                let mut series = Vec::with_capacity(n_steps);
                for t in 0..n_steps {
                    let idx = ((t * nvar + var) * nnodes + n) * nmodes + m;
                    series.push(trajectory[idx]);
                }
                let feats = temporal_stat_features(&series);
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
    fn test_temporal_stat_constant() {
        let series = vec![2.0f32; 10];
        let f = temporal_stat_features(&series);
        assert_eq!(f.len(), 7);
        assert!((f[0] - 40.0).abs() < 1e-5, "abs_energy = {}", f[0]); // 10 * 4
        assert!((f[1] - 4.0).abs() < 1e-5, "avg_power = {}", f[1]);
        assert_eq!(f[3], 0.0, "no zero crossings for constant");
        assert!(f[4].abs() < 1e-5, "skewness = {}", f[4]);
        assert!(f[5].abs() < 1e-5, "kurtosis = {}", f[5]);
        // Burstiness for constant positive series = (0 - mean) / (0 + mean) = -1
        assert!((f[6] + 1.0).abs() < 1e-5, "burstiness for constant should be -1, got {}", f[6]);
    }

    #[test]
    fn test_temporal_stat_sin() {
        let series: Vec<f32> = (0..100)
            .map(|i| (i as f64 * 0.1).sin() as f32)
            .collect();
        let f = temporal_stat_features(&series);
        assert_eq!(f.len(), 7);
        // Zero crossings ≈ 2 per period, period ≈ 63 samples → ~3 crossings total for 100 samples
        assert!(f[3] > 0.0, "zcr should be > 0");
        // For near-zero-mean oscillatory data burstiness should be positive
        assert!(f[6] > 0.0, "burstiness of near-zero-mean sine should be positive, got {}", f[6]);
    }

    #[test]
    fn test_extract_temporal_stat_shape() {
        let n_steps = 32;
        let nvar = 2;
        let nnodes = 3;
        let nmodes = 1;
        let trajectory: Vec<f32> = (0..n_steps * nvar * nnodes * nmodes)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let feats = extract_temporal_stat_features(
            &trajectory,
            &[n_steps, nvar, nnodes, nmodes],
        );
        assert_eq!(feats.len(), nvar * TEMPORAL_STAT_FEATURE_COUNT);
    }
}
