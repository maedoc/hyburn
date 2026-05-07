//! SB_TransitionMatrix: Symbolic transition matrix covariance.
//!
//! Downsamples by tau (first AC zero crossing), coarse-grains to 3 symbols,
//! builds a 3x3 transition matrix, and returns sum of diagonal covariances.

use super::fft;
use super::helpers;

/// Symbolic transition matrix: sum of diagonal covariances.
///
/// Downsample by tau (first AC zero crossing), coarse-grain to 3 symbols,
/// build transition matrix, compute column covariance, return sum of diagonal.
pub fn sb_transition_matrix_3ac_sumdiagcov(y: &[f64]) -> f64 {
    if y.len() < 3 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    let n = y.len();

    // Find tau (first zero crossing of AC)
    let ac = fft::fft_autocorrelation(y);
    let mut tau = 1;
    if let Some((i, _)) = ac.iter().enumerate().skip(1).find(|&(_, &val)| val <= 0.0) {
        tau = i;
    }
    tau = tau.max(1);

    // Downsample by tau
    let y_ds: Vec<f64> = (0..n)
        .step_by(tau)
        .map(|i| y[i])
        .collect();

    if y_ds.len() < 3 {
        return f64::NAN;
    }

    // Coarse-grain to 3 symbols using quantile boundaries
    let mut sorted = y_ds.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q33 = helpers::quantile_sorted(&sorted, 1.0 / 3.0);
    let q67 = helpers::quantile_sorted(&sorted, 2.0 / 3.0);

    let symbols: Vec<usize> = y_ds.iter().map(|&v| {
        if v <= q33 { 0 } else if v <= q67 { 1 } else { 2 }
    }).collect();

    // Build 3x3 transition matrix
    let mut trans = [0usize; 9];
    for i in 0..symbols.len() - 1 {
        trans[symbols[i] * 3 + symbols[i + 1]] += 1;
    }

    // Normalize rows to get probabilities
    let mut trans_prob = [0.0f64; 9];
    for row in 0..3 {
        let row_sum: usize = (0..3).map(|col| trans[row * 3 + col]).sum();
        if row_sum > 0 {
            for col in 0..3 {
                trans_prob[row * 3 + col] = trans[row * 3 + col] as f64 / row_sum as f64;
            }
        }
    }

    // Compute covariance matrix of the 3 columns
    // Column j contains: P(symbol=j | previous was 0), P(symbol=j | previous was 1), P(symbol=j | previous was 2)
    // But we have a single observation per symbol, so compute column means and cov
    let n_rows = 3.0;
    let mut col_means = [0.0; 3];
    for j in 0..3 {
        let mut sum = 0.0;
        for i in 0..3 {
            sum += trans_prob[i * 3 + j];
        }
        col_means[j] = sum / n_rows;
    }

    // Covariance matrix
    let mut cov = [0.0f64; 9];
    for j1 in 0..3 {
        for j2 in 0..3 {
            let mut sum = 0.0;
            for i in 0..3 {
                let d1 = trans_prob[i * 3 + j1] - col_means[j1];
                let d2 = trans_prob[i * 3 + j2] - col_means[j2];
                sum += d1 * d2;
            }
            cov[j1 * 3 + j2] = sum / (n_rows - 1.0);
        }
    }

    // Sum of diagonal covariances
    cov[0] + cov[4] + cov[8]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transition_matrix() {
        let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let cov = sb_transition_matrix_3ac_sumdiagcov(&y);
        assert!(cov.is_finite(), "sum diag cov = {}", cov);
    }
}