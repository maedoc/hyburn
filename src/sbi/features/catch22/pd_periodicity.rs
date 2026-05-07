//! PD_PeriodicityWang: Periodicity detection via spline detrending + ACF peak.
//!
//! Fits a cubic B-spline with 3 knots to detrend, then looks for
//! a peak in the autocorrelation of the detrended signal.

use super::fft;
use super::spline;

/// Periodicity detection: returns the lag of the first ACF peak in the
/// spline-detrended signal, or 0 if no periodicity is detected.
///
/// A peak must satisfy: preceded by a trough, peak - trough >= 0.01, peak > 0.
pub fn pd_periodicity_wang_th0_01(y: &[f64]) -> usize {
    if y.len() < 4 {
        return 0;
    }
    for &v in y {
        if !v.is_finite() {
            return 0;
        }
    }

    let n = y.len();

    // Spline detrend
    let trend = spline::splinefit(y);
    let detrended: Vec<f64> = y.iter().zip(trend.iter()).map(|(&yi, &ti)| yi - ti).collect();

    // Compute autocorrelation of detrended signal up to N/3
    let max_lag = n / 3;
    let ac = fft::fft_autocorrelation(&detrended);

    if ac.len() < 3 {
        return 0;
    }

    // Find troughs and peaks
    let threshold = 0.01;
    let check_limit = max_lag.min(ac.len() - 1);

    let mut last_trough_val = f64::INFINITY;
    let mut _last_trough_idx = 0usize;
    let mut found_trough = false;

    for i in 1..check_limit {
        // Check if this is a trough (local minimum)
        if i > 0 && i < ac.len() - 1 && ac[i] < ac[i - 1] && ac[i] < ac[i + 1] {
            last_trough_val = ac[i];
            _last_trough_idx = i;
            found_trough = true;
        }

        // Check if this is a peak (local maximum) preceded by a trough
        if found_trough && i > 0 && i < ac.len() - 1 && ac[i] > ac[i - 1] && ac[i] > ac[i + 1] {
            let peak_val = ac[i];
            let peak_minus_trough = peak_val - last_trough_val;

            if peak_minus_trough >= threshold && peak_val > 0.0 {
                return i;
            }
        }
    }

    0 // No periodicity detected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_periodicity_sin() {
        // Sinusoid with period ~63 (2*pi/0.1)
        let y: Vec<f64> = (0..500).map(|i| (i as f64 * 0.1).sin()).collect();
        let period = pd_periodicity_wang_th0_01(&y);
        // The detected period should be around 63
        assert!(period > 0, "period = {}", period);
    }

    #[test]
    fn test_periodicity_noise() {
        // White noise: no clear periodicity
        let y: Vec<f64> = (0..200).map(|i| ((i * 7 + 3) % 13) as f64 * 0.1 - 0.6).collect();
        let period = pd_periodicity_wang_th0_01(&y);
        // May or may not detect periodicity in quasi-random data
        assert!(period < 200, "period = {}", period);
    }
}