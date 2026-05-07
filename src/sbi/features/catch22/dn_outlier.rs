//! DN_OutlierInclude: Incremental threshold outlier timing.

/// Outlier timing for negative deviations.
///
/// At each threshold thr = j * 0.01 (j = 1, 2, ...), find indices where y <= -thr.
/// Compute normalized timing measures and return the median across valid thresholds.
pub fn dn_outlier_include_n(y: &[f64]) -> f64 {
    outlier_include(y, -1.0)
}

/// Outlier timing for positive deviations.
pub fn dn_outlier_include_p(y: &[f64]) -> f64 {
    outlier_include(y, 1.0)
}

fn outlier_include(y: &[f64], sign: f64) -> f64 {
    if y.len() < 3 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    let n = y.len();
    let y_abs_max = y.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

    if y_abs_max.abs() < f64::EPSILON {
        return 0.0; // constant series
    }

    let mut ms_dti4_values: Vec<f64> = Vec::new();

    let mut thr = 0.01;
    while thr <= y_abs_max {
        // Find indices where sign * y[i] >= thr
        let mut outlier_indices: Vec<usize> = Vec::new();
        for (i, &val) in y.iter().enumerate().take(n) {
            if sign * val >= thr {
                outlier_indices.push(i);
            }
        }

        if outlier_indices.is_empty() {
            thr += 0.01;
            continue;
        }

        // msDti1: mean time between successive outliers
        let ms_dti1 = if outlier_indices.len() > 1 {
            let mut sum_gap = 0.0;
            for j in 0..outlier_indices.len() - 1 {
                sum_gap += (outlier_indices[j + 1] - outlier_indices[j]) as f64;
            }
            sum_gap / (outlier_indices.len() - 1) as f64
        } else {
            f64::INFINITY
        };

        // msDti3: fraction of points above threshold
        let ms_dti3 = outlier_indices.len() as f64 / n as f64 * 100.0; // percentage

        // msDti4: median index normalized by (size/2 - 1)
        let median_idx = if outlier_indices.len().is_multiple_of(2) {
            let mid = outlier_indices.len() / 2;
            (outlier_indices[mid - 1] + outlier_indices[mid]) as f64 / 2.0
        } else {
            outlier_indices[outlier_indices.len() / 2] as f64
        };
        let normalizer = n as f64 / 2.0 - 1.0;
        let ms_dti4 = if normalizer.abs() > f64::EPSILON {
            median_idx / normalizer
        } else {
            0.0
        };

        // Only include if msDti3 > 2% and msDti1 is finite
        if ms_dti3 > 2.0 && ms_dti1.is_finite() && ms_dti1 > 0.0 {
            ms_dti4_values.push(ms_dti4);
        }

        thr += 0.01;
    }

    if ms_dti4_values.is_empty() {
        return 0.0;
    }

    // Return median of accumulated ms_dti4 values
    ms_dti4_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = ms_dti4_values.len();
    if len.is_multiple_of(2) {
        (ms_dti4_values[len / 2 - 1] + ms_dti4_values[len / 2]) / 2.0
    } else {
        ms_dti4_values[len / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outlier_include_sin() {
        let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let n = dn_outlier_include_n(&y);
        let p = dn_outlier_include_p(&y);
        // Both should be finite for a sinusoid
        assert!(n.is_finite(), "outlier_n = {}", n);
        assert!(p.is_finite(), "outlier_p = {}", p);
    }

    #[test]
    fn test_outlier_include_constant() {
        let y = vec![0.0; 20];
        let n = dn_outlier_include_n(&y);
        assert!((n - 0.0).abs() < 1e-10, "constant outlier_n = {}", n);
    }
}