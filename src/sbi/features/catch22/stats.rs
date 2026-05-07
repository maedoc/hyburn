//! Basic statistics utilities for catch22 feature computation.

/// Compute the arithmetic mean.
pub fn mean(y: &[f64]) -> f64 {
    y.iter().sum::<f64>() / y.len() as f64
}

/// Compute the sample variance (population version: divide by N).
pub fn variance(y: &[f64]) -> f64 {
    let m = mean(y);
    y.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / y.len() as f64
}

/// Compute the population standard deviation.
pub fn stddev(y: &[f64]) -> f64 {
    variance(y).sqrt()
}

/// Z-score normalize: (x - mean) / std.
///
/// Returns a new vector. If std is near-zero, returns a zero vector.
pub fn zscore(y: &[f64]) -> Vec<f64> {
    let m = mean(y);
    let s = stddev(y);
    if s.abs() < f64::EPSILON {
        return vec![0.0; y.len()];
    }
    y.iter().map(|&x| (x - m) / s).collect()
}

/// Compute the Pearson autocorrelation at a given lag using the FFT.
///
/// This computes the full normalized autocorrelation function and returns
/// the value at the specified lag.
pub fn autocorrelation_lag(y: &[f64], lag: usize) -> f64 {
    let ac = crate::sbi::features::catch22::fft::fft_autocorrelation(y);
    if lag < ac.len() {
        ac[lag]
    } else {
        0.0
    }
}

/// Compute the unnormalized autocovariance at a given lag.
pub fn autocovariance_lag(y: &[f64], lag: usize) -> f64 {
    let m = mean(y);
    let n = y.len();
    if lag >= n {
        return 0.0;
    }
    let mut cov = 0.0;
    for i in 0..(n - lag) {
        cov += (y[i] - m) * (y[i + lag] - m);
    }
    cov / n as f64
}

/// Simple ordinary least squares linear regression.
///
/// Fits y = m*x + b. Returns (slope, intercept).
pub fn linreg(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len().min(y.len()) as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for i in 0..x.len().min(y.len()) {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxx += dx * dx;
        sxy += dx * dy;
    }
    if sxx.abs() < f64::EPSILON {
        return (0.0, my);
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    (slope, intercept)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 1e-10);
        assert!((mean(&[0.0; 10]) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance() {
        let v = variance(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((v - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_stddev() {
        let s = stddev(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((s - 2.0f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_zscore() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z = zscore(&y);
        // z-scored: mean should be ~0, std ~1
        assert!(mean(&z).abs() < 1e-10);
        assert!((stddev(&z) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zscore_constant() {
        let y = vec![5.0; 10];
        let z = zscore(&y);
        assert!(z.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_linreg() {
        // y = 2x + 1
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let (slope, intercept) = linreg(&x, &y);
        assert!((slope - 2.0).abs() < 1e-10);
        assert!((intercept - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_empty() {
        // This would panic - behavior undefined for empty
        // Just test that we don't crash on very small arrays
        let y = vec![1.0];
        assert!((mean(&y) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_single() {
        let y = vec![5.0];
        assert!((variance(&y)).abs() < 1e-10);
    }

    #[test]
    fn test_linreg_perfect_vertical() {
        // Vertical line: x = constant
        let x = vec![1.0; 5];
        let y: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let (slope, intercept) = linreg(&x, &y);
        // Slope should be 0 (no variation in x)
        assert!(slope.abs() < 1e-10, "slope should be 0 for constant x: {}", slope);
        // Intercept should be mean of y
        let expected_intercept = mean(&y);
        assert!((intercept - expected_intercept).abs() < 1e-10);
    }

    #[test]
    fn test_autocovariance_lag_zero() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cov0 = autocovariance_lag(&y, 0);
        let var = variance(&y);
        assert!((cov0 - var).abs() < 1e-10, "lag-0 autocov should equal variance");
    }
}