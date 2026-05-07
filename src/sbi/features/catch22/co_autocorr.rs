//! Autocorrelation-based features (CO_*).
//!
//! Features:
//! - CO_f1ecac: Lag where AC first drops below 1/e
//! - CO_FirstMin_ac: First local minimum of AC
//! - CO_Embed2_Dist_tau_d_expfit_meandiff: Embedding distance vs exponential
//! - CO_HistogramAMI_even_2_5: Auto-mutual information via histogram
//! - CO_trev_1_num: Time-reversibility

use super::stats;
use super::fft;
use super::histogram;
use super::helpers;

/// Time-reversibility statistic: mean of cubed successive differences.
pub fn co_trev_1_num(y: &[f64]) -> f64 {
    if y.len() < 2 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    let n = y.len() - 1;
    let mut sum = 0.0;
    for i in 0..n {
        let d = y[i + 1] - y[i];
        sum += d.powi(3);
    }
    sum / n as f64
}

/// Compute the 1/e crossing timescale of the autocorrelation function.
///
/// Finds the lag where the normalized AC first drops below 1/e ≈ 0.3679,
/// with linear interpolation between bounding lags.
pub fn co_f1ecac(y: &[f64]) -> f64 {
    if y.len() < 2 {
        return 0.0;
    }
    for &v in y {
        if !v.is_finite() {
            return 0.0;
        }
    }

    let ac = fft::fft_autocorrelation(y);
    let threshold = 1.0 / std::f64::consts::E; // ≈ 0.3679

    for i in 1..ac.len() {
        if ac[i] < threshold {
            // Linear interpolation to find exact crossing
            if i > 0 && (ac[i - 1] - ac[i]).abs() > f64::EPSILON {
                let frac = (ac[i - 1] - threshold) / (ac[i - 1] - ac[i]);
                return (i as f64 - 1.0) + frac;
            }
            return i as f64;
        }
    }

    // Never dropped below 1/e, return last lag
    ac.len() as f64
}

/// Find the first local minimum of the autocorrelation function.
pub fn co_first_min_ac(y: &[f64]) -> usize {
    if y.len() < 3 {
        return 0;
    }
    for &v in y {
        if !v.is_finite() {
            return 0;
        }
    }

    let ac = fft::fft_autocorrelation(y);

    match helpers::first_local_min(&ac) {
        Some(idx) => idx,
        None => y.len(), // No minimum found
    }
}

/// Two-dimensional embedding distance compared to exponential fit.
///
/// Computes distances in a 2D delay embedding at tau = first AC zero crossing,
/// then compares the histogram of distances to an exponential distribution.
pub fn co_embed2_dist_tau_d_expfit_meandiff(y: &[f64]) -> f64 {
    if y.len() < 3 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    // Find tau as first zero crossing of AC (maximum tau = N/10)
    let tau_max = (y.len() / 10).max(1);
    let ac = fft::fft_autocorrelation(y);
    let mut tau = 1;
    for i in 1..=tau_min(tau_max, ac.len()) {
        if i < ac.len() && ac[i] < 0.0 {
            tau = i;
            break;
        }
    }

    // Compute 2D embedding distances
    let n = y.len() - tau - 1;
    if n < 1 {
        return f64::NAN;
    }

    let mut distances: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let dx = y[i + 1] - y[i];
        let dy = y[i + tau + 1] - y[i + tau];
        distances.push(dx.hypot(dy));
    }

    // Compare histogram to exponential distribution
    let (counts, edges) = histogram::histcounts_auto(&distances);
    if counts.is_empty() {
        return f64::NAN;
    }

    let centers = histogram::bin_centers(&edges);
    let total: usize = counts.iter().sum();
    if total == 0 {
        return f64::NAN;
    }

    // Mean of distances for exponential parameter
    let mean_dist = stats::mean(&distances);
    if mean_dist.abs() < f64::EPSILON {
        return f64::NAN;
    }

    // Compare histogram proportions to exponential PDF
    let mut meandiff = 0.0;
    let mut n_valid = 0;
    for (i, &count) in counts.iter().enumerate() {
        if i < centers.len() {
            let p_hist = count as f64 / total as f64;
            let p_exp = (1.0 / mean_dist) * (-centers[i] / mean_dist).exp();
            if p_exp.is_finite() && p_exp > 0.0 {
                meandiff += (p_hist - p_exp).abs();
                n_valid += 1;
            }
        }
    }

    if n_valid > 0 {
        meandiff / n_valid as f64
    } else {
        f64::NAN
    }
}

fn tau_min(a: usize, b: usize) -> usize {
    a.min(b)
}

/// Auto-mutual information via histogram at lag 2 with 5 bins.
///
/// Computes the AMI between y[t] and y[t+2] using a 5x5 histogram.
pub fn co_histogram_ami_even_2_5(y: &[f64]) -> f64 {
    if y.len() < 3 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    let tau = 2;
    let n_bins = 5;
    let n = y.len() - tau;

    // Create joint histogram of (y[t], y[t+tau])
    let x = &y[..n];
    let z = &y[tau..];

    // Bin edges based on the full series
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_width = if (y_max - y_min).abs() > f64::EPSILON {
        (y_max - y_min) / n_bins as f64
    } else {
        1.0
    };

    // Build joint histogram
    let mut joint_counts = vec![0usize; n_bins * n_bins];
    let mut x_counts = vec![0usize; n_bins];
    let mut z_counts = vec![0usize; n_bins];

    for i in 0..n {
        let xi = bin_index(x[i], y_min, bin_width, n_bins);
        let zi = bin_index(z[i], y_min, bin_width, n_bins);
        joint_counts[xi * n_bins + zi] += 1;
        x_counts[xi] += 1;
        z_counts[zi] += 1;
    }

    // Compute AMI
    let total = n as f64;
    let mut ami = 0.0;

    for i in 0..n_bins {
        for j in 0..n_bins {
            let pxy = joint_counts[i * n_bins + j] as f64 / total;
            let px = x_counts[i] as f64 / total;
            let py = z_counts[j] as f64 / total;

            if pxy > 0.0 && px > 0.0 && py > 0.0 {
                ami += pxy * (pxy / (px * py)).ln();
            }
        }
    }

    ami
}

fn bin_index(val: f64, min: f64, width: f64, n_bins: usize) -> usize {
    if width.abs() < f64::EPSILON {
        return 0;
    }
    let idx = ((val - min) / width) as usize;
    idx.min(n_bins - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_co_trev_symmetric() {
        // Symmetric differences: alternating up/down
        let y = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let t = co_trev_1_num(&y);
        // Differences: -2, 2, -2, 2 → cubes: -8, 8, -8, 8 → mean ≈ 0
        assert!(t.abs() < 1e-10, "trev = {}", t);
    }

    #[test]
    fn test_co_f1ecac_sin() {
        let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = co_f1ecac(&y);
        assert!(ts.is_finite(), "timescale = {}", ts);
        // For a sinusoid, the 1/e crossing should be relatively small
        assert!(ts > 0.0, "timescale = {}", ts);
    }

    #[test]
    fn test_co_first_min_ac() {
        // Sinusoid should have a clear first minimum
        let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let min_lag = co_first_min_ac(&y);
        assert!(min_lag > 0, "first min lag = {}", min_lag);
        assert!(min_lag < 200, "first min lag = {}", min_lag);
    }

    #[test]
    fn test_co_histogram_ami() {
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let ami = co_histogram_ami_even_2_5(&y);
        assert!(ami.is_finite(), "AMI = {}", ami);
        assert!(ami >= 0.0, "AMI should be non-negative: {}", ami);
    }

    #[test]
    fn test_co_embed2_dist() {
        let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let d = co_embed2_dist_tau_d_expfit_meandiff(&y);
        assert!(d.is_finite(), "embedding dist = {}", d);
    }
}