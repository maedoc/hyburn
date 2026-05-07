//! SP_Summaries: Welch periodogram spectral features.
//!
//! Features:
//! - SP_Summaries_welch_rect_area_5_1: low-frequency power area
//! - SP_Summaries_welch_rect_centroid: spectral centroid frequency

use super::fft;

/// Welch periodogram with rectangular window: integrated area from DC to 5th bin.
pub fn sp_summaries_welch_rect_area_5_1(y: &[f64]) -> f64 {
    if y.len() < 5 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    let n = y.len();
    let fs = 1.0; // Assume sampling rate = 1
    let window_width = n.min(128);
    let nfft = window_width;

    let (psd, _freqs) = fft::welch_psd(y, nfft, fs, window_width);

    if psd.len() < 5 {
        return f64::NAN;
    }

    // Area of first 5 frequency bins as proportion of total
    let area_5: f64 = psd[..5.min(psd.len())].iter().sum();
    let total_area: f64 = psd.iter().sum();

    if total_area.abs() < f64::EPSILON {
        return f64::NAN;
    }

    area_5 / total_area
}

/// Welch periodogram with rectangular window: spectral centroid.
///
/// Returns the angular frequency where cumulative power first exceeds
/// half the total power.
pub fn sp_summaries_welch_rect_centroid(y: &[f64]) -> f64 {
    if y.len() < 5 {
        return f64::NAN;
    }
    for &v in y {
        if !v.is_finite() {
            return f64::NAN;
        }
    }

    let n = y.len();
    let fs = 1.0; // Assume sampling rate = 1
    let window_width = n.min(128);
    let nfft = window_width;

    let (psd, freqs) = fft::welch_psd(y, nfft, fs, window_width);

    let total_power: f64 = psd.iter().sum();
    if total_power.abs() < f64::EPSILON {
        return f64::NAN;
    }

    let half_power = total_power / 2.0;
    let mut cumsum = 0.0;
    for (i, &p) in psd.iter().enumerate() {
        cumsum += p;
        if cumsum >= half_power {
            // Return angular frequency
            let f = if i < freqs.len() { freqs[i] } else { 0.0 };
            return 2.0 * std::f64::consts::PI * f;
        }
    }

    // Fallback: return last frequency
    if let Some(&f) = freqs.last() {
        2.0 * std::f64::consts::PI * f
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welch_area() {
        let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let area = sp_summaries_welch_rect_area_5_1(&y);
        assert!(area.is_finite(), "area = {}", area);
        assert!(area > 0.0 && area <= 1.0, "area = {}", area);
    }

    #[test]
    fn test_welch_centroid() {
        let y: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let centroid = sp_summaries_welch_rect_centroid(&y);
        assert!(centroid.is_finite(), "centroid = {}", centroid);
        assert!(centroid > 0.0, "centroid = {}", centroid);
    }
}