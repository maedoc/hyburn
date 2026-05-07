//! Helper functions for catch22 feature computation.

/// Compute the quantile of a sorted array using linear interpolation.
///
/// `q` should be in [0, 1]. The input array must be sorted.
pub fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Compute the quantile of an unsorted array.
pub fn quantile(y: &[f64], q: f64) -> f64 {
    let mut sorted = y.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    quantile_sorted(&sorted, q)
}

/// Compute Shannon entropy: -sum(p * ln(p)) where p = counts / total.
pub fn f_entropy(counts: &[f64], total: f64) -> f64 {
    if total.abs() < f64::EPSILON {
        return 0.0;
    }
    let mut ent = 0.0;
    for &c in counts {
        if c > 0.0 {
            let p = c / total;
            ent -= p * p.ln();
        }
    }
    ent
}

/// Sort a slice in place and return it.
pub fn sort(y: &mut [f64]) {
    y.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Return a sorted copy.
pub fn sorted(y: &[f64]) -> Vec<f64> {
    let mut s = y.to_vec();
    sort(&mut s);
    s
}

/// Compute cumulative sum.
pub fn cumsum(y: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(y.len());
    let mut sum = 0.0;
    for &v in y {
        sum += v;
        result.push(sum);
    }
    result
}

/// Compute first differences: y[i+1] - y[i].
pub fn diff(y: &[f64]) -> Vec<f64> {
    (0..y.len().saturating_sub(1))
        .map(|i| y[i + 1] - y[i])
        .collect()
}

/// Find the index of the first local minimum.
pub fn first_local_min(y: &[f64]) -> Option<usize> {
    if y.len() < 3 {
        return None;
    }
    (1..y.len() - 1).find(|&i| y[i] < y[i - 1] && y[i] < y[i + 1])
}

/// Compute the first zero crossing index of the autocorrelation function.
/// Returns the index where AC first becomes negative.
pub fn first_zero_crossing_ac(y: &[f64]) -> usize {
    let ac = crate::sbi::features::catch22::fft::fft_autocorrelation(y);
    ac.iter()
        .enumerate()
        .skip(1)
        .find(|&(_, &val)| val < 0.0)
        .map(|(i, _)| i)
        .unwrap_or(y.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((quantile(&y, 0.0) - 1.0).abs() < 1e-10);
        assert!((quantile(&y, 1.0) - 5.0).abs() < 1e-10);
        assert!((quantile(&y, 0.5) - 3.0).abs() < 1e-10);
        assert!((quantile(&y, 0.25) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_entropy() {
        // Uniform distribution: entropy = ln(n_bins)
        let counts = vec![5.0, 5.0, 5.0];
        let ent = f_entropy(&counts, 15.0);
        assert!((ent - 3.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_diff() {
        let y = vec![1.0, 3.0, 6.0, 10.0];
        let d = diff(&y);
        assert_eq!(d, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cumsum() {
        let y = vec![1.0, 2.0, 3.0];
        let cs = cumsum(&y);
        assert_eq!(cs, vec![1.0, 3.0, 6.0]);
    }

    #[test]
    fn test_first_local_min() {
        let y = vec![5.0, 3.0, 1.0, 2.0, 4.0];
        assert_eq!(first_local_min(&y), Some(2));
    }
}