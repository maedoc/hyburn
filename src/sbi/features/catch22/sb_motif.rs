//! SB_MotifThree: Symbolic coarse-graining entropy.
//!
//! Coarse-grains to 3 symbols (via quantiles), then computes
//! entropy of length-2 motif (2-letter word) distributions.

use super::helpers;

/// Symbolic motif entropy with 3 quantile symbols.
///
/// Coarse-grain to {0, 1, 2} using quantile boundaries (33rd, 67th percentile).
/// Compute all length-2 motifs and their entropy.
pub fn sb_motif_three_quantile_hh(y: &[f64]) -> f64 {
    if y.len() < 3 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    let n = y.len();

    // Quantile boundaries
    let mut sorted = y.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q33 = helpers::quantile_sorted(&sorted, 0.3333);
    let q67 = helpers::quantile_sorted(&sorted, 0.6667);

    // Symbolize: 0 if <= q33, 1 if <= q67, 2 otherwise
    let symbols: Vec<usize> = y.iter().map(|&v| {
        if v <= q33 { 0 } else if v <= q67 { 1 } else { 2 }
    }).collect();

    // Build 3x3 transition matrix (count of symbol pairs)
    let mut trans = [0usize; 9];
    for i in 0..n - 1 {
        let row = symbols[i];
        let col = symbols[i + 1];
        trans[row * 3 + col] += 1;
    }

    // Compute entropy of each row (conditional distribution)
    let mut total_entropy = 0.0;
    let mut n_valid_rows = 0;

    for row in 0..3 {
        let row_sum: usize = (0..3).map(|col| trans[row * 3 + col]).sum();
        if row_sum == 0 {
            continue;
        }

        let mut row_entropy = 0.0;
        for col in 0..3 {
            let p = trans[row * 3 + col] as f64 / row_sum as f64;
            if p > 0.0 {
                row_entropy -= p * p.ln();
            }
        }
        total_entropy += row_entropy;
        n_valid_rows += 1;
    }

    if n_valid_rows > 0 {
        total_entropy / n_valid_rows as f64
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motif_entropy() {
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let h = sb_motif_three_quantile_hh(&y);
        assert!(h.is_finite(), "entropy = {}", h);
        assert!(h >= 0.0, "entropy should be non-negative: {}", h);
    }
}