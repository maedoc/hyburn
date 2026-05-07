//! DN_HistogramMode: histogram mode with 5 and 10 bins.
//!
//! Returns the midpoint of the most frequently occupied bin (or average of
//! multiple maxima in case of ties).

use super::histogram;

/// Compute histogram mode with 5 bins (z-scored input assumed).
pub fn dn_histogram_mode_5(y: &[f64]) -> f64 {
    histogram_mode(y, 5)
}

/// Compute histogram mode with 10 bins (z-scored input assumed).
pub fn dn_histogram_mode_10(y: &[f64]) -> f64 {
    histogram_mode(y, 10)
}

fn histogram_mode(y: &[f64], n_bins: usize) -> f64 {
    if y.is_empty() {
        return f64::NAN;
    }

    let (counts, edges) = histogram::histcounts_even(y, n_bins);
    if counts.is_empty() {
        return f64::NAN;
    }

    // Find maximum count
    let max_count = *counts.iter().max().unwrap_or(&0);

    // Average the midpoints of all bins that achieve the maximum count
    let mut mode_sum = 0.0;
    let mut mode_count = 0;
    for (i, &c) in counts.iter().enumerate() {
        if c == max_count {
            let midpoint = (edges[i] + edges[i + 1]) / 2.0;
            mode_sum += midpoint;
            mode_count += 1;
        }
    }

    if mode_count > 0 {
        mode_sum / mode_count as f64
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_mode_5() {
        // Uniform data around 0 — mode should be near center bin
        let y: Vec<f64> = (-50..50).map(|i| i as f64 / 50.0).collect();
        let mode = dn_histogram_mode_5(&y);
        // Should be finite
        assert!(mode.is_finite(), "mode = {}", mode);
    }

    #[test]
    fn test_histogram_mode_10() {
        let y: Vec<f64> = (-50..50).map(|i| i as f64 / 50.0).collect();
        let mode = dn_histogram_mode_10(&y);
        assert!(mode.is_finite(), "mode = {}", mode);
    }
}