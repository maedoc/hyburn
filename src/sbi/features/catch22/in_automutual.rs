//! IN_AutoMutualInfoStats: Gaussian AMI first minimum.
//!
//! Approximates auto-mutual information using Gaussian approximation
//! AMI(lag) = -0.5 * ln(1 - AC(lag)^2), and finds the first local minimum.

use super::fft;

/// Gaussian AMI first minimum up to lag 40 (or size/2).
///
/// Returns the lag of the first local minimum of AMI(lag).
/// If no minimum is found, returns the maximum lag.
pub fn in_auto_mutual_info_stats_40_gaussian_fmmi(y: &[f64]) -> f64 {
    if y.len() < 3 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    let ac = fft::fft_autocorrelation(y);
    let max_lag = (40usize).min(y.len() / 2).min(ac.len() - 1);

    if max_lag < 2 {
        return 0.0;
    }

    // Compute AMI for each lag
    let mut ami: Vec<f64> = Vec::with_capacity(max_lag + 1);
    ami.push(0.0); // AMI(0) = 0 conceptually
    for &ac_lag in ac.iter().take(max_lag + 1).skip(1) {
        let r2 = ac_lag * ac_lag;
        if r2 >= 1.0 {
            ami.push(f64::INFINITY);
        } else {
            ami.push(-0.5 * (1.0 - r2).ln());
        }
    }

    // Find first local minimum
    for i in 1..ami.len() - 1 {
        if ami[i] < ami[i - 1] && ami[i] < ami[i + 1] {
            return i as f64;
        }
    }

    // No minimum found, return max lag
    max_lag as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ami_timescale() {
        let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let ts = in_auto_mutual_info_stats_40_gaussian_fmmi(&y);
        assert!(ts.is_finite(), "AMI timescale = {}", ts);
        assert!(ts >= 0.0, "timescale should be non-negative: {}", ts);
    }
}