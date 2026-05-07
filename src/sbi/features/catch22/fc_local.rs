//! FC_LocalSimple: Local simple forecasting features.
//!
//! Features:
//! - FC_LocalSimple_mean1_tauresrat: forecast residual timescale ratio
//! - FC_LocalSimple_mean3_stderr: forecast error standard deviation

use super::helpers;

/// Local simple forecast with train_length=1, timescale ratio.
///
/// Naïve predictor: y_hat[i+1] = y[i] (1-step persistence).
/// Computes residuals, then returns the ratio of residual AC zero-crossing
/// to original AC zero-crossing.
pub fn fc_local_simple_mean1_tauresrat(y: &[f64]) -> f64 {
    if y.len() < 4 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    // Naïve forecast: residual = y[i+1] - y[i]
    let residuals: Vec<f64> = helpers::diff(y);

    // Check constant residual
    let res_std = super::stats::stddev(&residuals);
    if res_std.abs() < f64::EPSILON {
        return 0.0;
    }

    // First zero crossing of residual AC
    let res_tau = helpers::first_zero_crossing_ac(&residuals);

    // First zero crossing of original AC
    let orig_tau = helpers::first_zero_crossing_ac(y);

    if orig_tau == 0 {
        return f64::NAN;
    }

    res_tau as f64 / orig_tau as f64
}

/// Local simple forecast with train_length=3, residual standard error.
///
/// Predictor: mean of previous 3 points. Returns std(residuals).
pub fn fc_local_simple_mean3_stderr(y: &[f64]) -> f64 {
    if y.len() < 4 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    let train_len = 3;
    if y.len() <= train_len {
        return f64::NAN;
    }

    let n = y.len() - train_len;
    let mut residuals = Vec::with_capacity(n);

    for i in 0..n {
        let pred: f64 = y[i..i + train_len].iter().sum::<f64>() / train_len as f64;
        residuals.push(y[i + train_len] - pred);
    }

    super::stats::stddev(&residuals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean3_stderr() {
        let y: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let se = fc_local_simple_mean3_stderr(&y);
        assert!(se.is_finite(), "stderr = {}", se);
    }

    #[test]
    fn test_mean1_tauresrat() {
        let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.05).sin()).collect();
        let r = fc_local_simple_mean1_tauresrat(&y);
        assert!(r.is_finite(), "tauresrat = {}", r);
    }
}