//! SC_FluctAnal: Fluctuation analysis features (R/S and DFA).
//!
//! Computes two-regime log-log fits of fluctuation functions.
//! Returns the proportion of the first linear regime.

use super::stats;
use super::helpers;

/// Rescaled Range (R/S) analysis with two-regime fit.
///
/// Returns the proportion of the first regime: (first_regime_length) / n_tau.
pub fn sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1(y: &[f64]) -> f64 {
    fluct_anal(y, FluctMethod::RescaledRange)
}

/// Detrended Fluctuation Analysis (DFA) with two-regime fit.
///
/// Returns the proportion of the first regime.
pub fn sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1(y: &[f64]) -> f64 {
    fluct_anal(y, FluctMethod::Dfa)
}

#[derive(Clone, Copy)]
enum FluctMethod {
    RescaledRange,
    Dfa,
}

fn fluct_anal(y: &[f64], method: FluctMethod) -> f64 {
    if y.len() < 5 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    let n = y.len();

    // Cumulative sum of the series
    let cs = helpers::cumsum(y);

    // Generate 50 log-spaced window sizes from 5 to n/2
    let tau_min = 5usize;
    let tau_max = n / 2;

    if tau_max < tau_min {
        return f64::NAN;
    }

    let n_tau_points = 50usize.min(tau_max - tau_min + 1);
    if n_tau_points < 3 {
        return f64::NAN;
    }

    // Log-spaced tau values
    let log_min = (tau_min as f64).ln();
    let log_max = (tau_max as f64).ln();
    let mut taus: Vec<usize> = Vec::with_capacity(n_tau_points);

    for i in 0..n_tau_points {
        let log_tau = log_min + (i as f64 / (n_tau_points - 1).max(1) as f64) * (log_max - log_min);
        let tau = log_tau.exp().round() as usize;
        let tau_clamped = tau.max(tau_min).min(tau_max);
        // Avoid duplicates
        if taus.is_empty() || *taus.last().unwrap() != tau_clamped {
            taus.push(tau_clamped);
        }
    }

    if taus.len() < 3 {
        return f64::NAN;
    }

    // Compute fluctuation function F(tau) for each tau
    let mut log_tau_vals: Vec<f64> = Vec::with_capacity(taus.len());
    let mut log_f_vals: Vec<f64> = Vec::with_capacity(taus.len());

    for &tau in &taus {
        let num_buffers = n / tau;
        if num_buffers < 1 {
            continue;
        }

        let mut f_sum = 0.0;

        for buf in 0..num_buffers {
            let start = buf * tau;
            let end = start + tau;

            match method {
                FluctMethod::RescaledRange => {
                    // Detrend by subtracting linear fit
                    let x: Vec<f64> = (0..tau).map(|t| t as f64).collect();
                    let segment: Vec<f64> = cs[start..end].to_vec();
                    let (slope, intercept) = stats::linreg(&x, &segment);
                    let detrended: Vec<f64> = segment
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| v - (slope * i as f64 + intercept))
                        .collect();

                    // Range = max - min of detrended
                    let d_min = detrended.iter().cloned().fold(f64::INFINITY, f64::min);
                    let d_max = detrended.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    f_sum += (d_max - d_min).powi(2);
                }
                FluctMethod::Dfa => {
                    // DFA: linear detrend, then RMS of residuals
                    let x: Vec<f64> = (0..tau).map(|t| t as f64).collect();
                    let segment: Vec<f64> = cs[start..end].to_vec();
                    let (slope, intercept) = stats::linreg(&x, &segment);
                    let mut residual_sq_sum = 0.0;
                    for (t, &seg_t) in segment.iter().enumerate().take(tau) {
                        let fitted = slope * t as f64 + intercept;
                        let residual = seg_t - fitted;
                        residual_sq_sum += residual * residual;
                    }
                    f_sum += residual_sq_sum / tau as f64;
                }
            }
        }

        let f_tau = if num_buffers > 0 {
            (f_sum / num_buffers as f64).sqrt()
        } else {
            continue;
        };

        if f_tau > 0.0 {
            log_tau_vals.push((tau as f64).ln());
            log_f_vals.push(f_tau.ln());
        }
    }

    if log_tau_vals.len() < 3 {
        return f64::NAN;
    }

    // Two-regime linear fit: find the best break point
    // For simplicity, try all possible break points and pick the one
    // that minimizes total squared error
    let n_pts = log_tau_vals.len();
    let mut best_err = f64::INFINITY;
    let mut best_break = n_pts / 2;

    for break_idx in 1..n_pts - 1 {
        // Need at least 2 points on each side
        if break_idx < 2 || break_idx > n_pts - 2 {
            continue;
        }

        let (s1, i1) = stats::linreg(&log_tau_vals[..=break_idx], &log_f_vals[..=break_idx]);
        let (s2, i2) = stats::linreg(&log_tau_vals[break_idx..], &log_f_vals[break_idx..]);

        // Compute total squared error
        let mut err = 0.0;
        for j in 0..=break_idx {
            let pred = s1 * log_tau_vals[j] + i1;
            err += (log_f_vals[j] - pred).powi(2);
        }
        for j in break_idx..n_pts {
            let pred = s2 * log_tau_vals[j] + i2;
            err += (log_f_vals[j] - pred).powi(2);
        }

        if err < best_err {
            best_err = err;
            best_break = break_idx;
        }
    }

    // Return proportion of first regime
    (best_break + 1) as f64 / n_pts as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rs_range() {
        // Random walk-like data
        let mut y = vec![0.0; 200];
        for i in 1..200 {
            y[i] = y[i - 1] + 0.1 * ((i as f64 * 0.3).sin());
        }
        let prop = sc_fluct_anal_2_rsrangefit_50_1_logi_prop_r1(&y);
        assert!(prop.is_finite(), "rs_range prop = {}", prop);
        assert!(prop > 0.0 && prop <= 1.0, "rs_range prop = {}", prop);
    }

    #[test]
    fn test_dfa() {
        let mut y = vec![0.0; 200];
        for i in 1..200 {
            y[i] = y[i - 1] + 0.1 * ((i as f64 * 0.3).sin());
        }
        let prop = sc_fluct_anal_2_dfa_50_1_2_logi_prop_r1(&y);
        assert!(prop.is_finite(), "dfa prop = {}", prop);
        assert!(prop > 0.0 && prop <= 1.0, "dfa prop = {}", prop);
    }
}