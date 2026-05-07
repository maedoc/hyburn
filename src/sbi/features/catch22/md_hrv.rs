//! MD_hrv: Heart-rate variability metric (pNN40).
//!
//! Computes the proportion of successive absolute differences
//! (scaled by 1000) that exceed 40.

/// Compute pNN40: proportion of |Dy| * 1000 > 40.
///
/// Dy[i] = y[i+1] - y[i]. Returns the fraction where |Dy[i] * 1000| > 40.
pub fn md_hrv_classic_pnn40(y: &[f64]) -> f64 {
    if y.len() < 2 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    let n = y.len() - 1;
    let mut count_above = 0;

    for i in 0..n {
        let dy = (y[i + 1] - y[i]).abs() * 1000.0;
        if dy > 40.0 {
            count_above += 1;
        }
    }

    count_above as f64 / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pnn40_constant() {
        let y = vec![5.0; 20];
        let p = md_hrv_classic_pnn40(&y);
        assert!((p - 0.0).abs() < 1e-10, "pnn40 = {}", p);
    }

    #[test]
    fn test_pnn40_large_diffs() {
        // Differences of 0.1 * 1000 = 100 > 40
        let y: Vec<f64> = (0..20).step_by(2).flat_map(|i| vec![i as f64, i as f64 + 0.1]).collect();
        let p = md_hrv_classic_pnn40(&y);
        assert!(p > 0.0, "pnn40 = {}", p);
    }
}