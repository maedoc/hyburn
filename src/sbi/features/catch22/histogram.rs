//! Histogram computation utilities for catch22 features.

/// Compute histogram bin counts with evenly-spaced bins.
///
/// Returns `(counts, edges)` where `edges` has `n_bins + 1` edges.
pub fn histcounts_even(y: &[f64], n_bins: usize) -> (Vec<usize>, Vec<f64>) {
    if y.is_empty() || n_bins == 0 {
        return (vec![], vec![]);
    }

    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut counts = vec![0usize; n_bins];
    let bin_width = if (y_max - y_min).abs() < f64::EPSILON {
        1.0 // uniform for constant data
    } else {
        (y_max - y_min) / n_bins as f64
    };

    let mut edges = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        edges.push(y_min + i as f64 * bin_width);
    }

    for &val in y {
        let bin_idx = if bin_width > f64::EPSILON {
            let idx = ((val - y_min) / bin_width) as usize;
            // Handle edge case where val = y_max
            if idx >= n_bins {
                n_bins - 1
            } else {
                idx
            }
        } else {
            0
        };
        counts[bin_idx] += 1;
    }

    (counts, edges)
}

/// Compute histogram bin counts with a specified number of bins,
/// using a reasonable automatic bin width (Scott's rule).
///
/// Returns `(counts, edges)` where `edges` has `num_bins(y)` + 1 edges.
pub fn histcounts_auto(y: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let n_bins = num_bins_auto(y);
    histcounts_even(y, n_bins)
}

/// Compute the number of bins using Scott's rule.
///
/// Scott's rule: bin_width = 3.5 * std / n^(1/3)
/// Number of bins = (max - min) / bin_width
pub fn num_bins_auto(y: &[f64]) -> usize {
    if y.len() < 2 {
        return 1;
    }
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (y_max - y_min).abs() < f64::EPSILON {
        return 1;
    }

    let n = y.len() as f64;
    let s = super::stats::stddev(y);
    if s.abs() < f64::EPSILON {
        return 1;
    }

    let bin_width = 3.5 * s / n.powf(1.0 / 3.0);
    if bin_width.abs() < f64::EPSILON {
        return 1;
    }

    let n_bins = ((y_max - y_min) / bin_width).ceil() as usize;
    n_bins.max(1).min(y.len())
}

/// Compute bin centers from edges.
pub fn bin_centers(edges: &[f64]) -> Vec<f64> {
    (0..edges.len() - 1)
        .map(|i| (edges[i] + edges[i + 1]) / 2.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histcounts_even() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let (counts, edges) = histcounts_even(&y, 5);
        assert_eq!(counts.len(), 5);
        assert_eq!(edges.len(), 6);
        assert_eq!(counts.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_histcounts_even_constant() {
        let y = vec![5.0; 10];
        let (counts, edges) = histcounts_even(&y, 3);
        // All in one bin for constant data
        assert_eq!(counts.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_num_bins_auto() {
        // Normally distributed data
        let y: Vec<f64> = (0..1000).map(|i| (i as f64 % 10.0)).collect();
        let n_bins = num_bins_auto(&y);
        assert!(n_bins >= 1);
        assert!(n_bins <= 1000);
    }
}