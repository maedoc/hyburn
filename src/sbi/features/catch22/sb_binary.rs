//! SB_BinaryStats: Binary run-length features.
//!
//! Features:
//! - SB_BinaryStats_diff_longstretch0: longest run of decreasing values
//! - SB_BinaryStats_mean_longstretch1: longest run above mean

/// Longest stretch of consecutive decreases (binary stat on differences).
///
/// Binarize diff < 0 as 0, diff >= 0 as 1. Find longest run of 0s.
pub fn sb_binary_stats_diff_longstretch0(y: &[f64]) -> f64 {
    if y.len() < 2 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    let mut max_stretch = 0usize;
    let mut current_stretch = 0usize;

    for i in 0..y.len() - 1 {
        if y[i + 1] - y[i] < 0.0 {
            current_stretch += 1;
            if current_stretch > max_stretch {
                max_stretch = current_stretch;
            }
        } else {
            current_stretch = 0;
        }
    }

    max_stretch as f64
}

/// Longest stretch of consecutive values above the mean.
///
/// Binarize y > mean as 1. Find longest run of 1s.
pub fn sb_binary_stats_mean_longstretch1(y: &[f64]) -> f64 {
    if y.is_empty() {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    let m = super::stats::mean(y);
    let mut max_stretch = 0usize;
    let mut current_stretch = 0usize;

    for &v in y {
        if v > m {
            current_stretch += 1;
            if current_stretch > max_stretch {
                max_stretch = current_stretch;
            }
        } else {
            current_stretch = 0;
        }
    }

    max_stretch as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_longstretch0_decreasing() {
        let y = vec![10.0, 9.0, 8.0, 7.0, 6.0, 10.0, 9.0, 8.0];
        let s = sb_binary_stats_diff_longstretch0(&y);
        assert!((s - 4.0).abs() < 1e-10, "stretch = {}", s);
    }

    #[test]
    fn test_mean_longstretch1() {
        // Values [0, 2, 3, 0, 2, 3, 0], mean = ~1.43
        let y = vec![0.0, 2.0, 3.0, 0.0, 2.0, 3.0, 0.0];
        let s = sb_binary_stats_mean_longstretch1(&y);
        assert!((s - 2.0).abs() < 1e-10, "stretch = {}", s);
    }
}