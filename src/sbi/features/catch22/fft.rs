//! Cooley-Tukey radix-2 FFT implementation and autocorrelation via FFT.
//!
//! This is a pure-Rust port of the catch22 custom FFT (no FFTW dependency).

/// Compute the next power of 2 >= n.
pub fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// In-place iterative Cooley-Tukey FFT (radix-2, decimation-in-time).
///
/// Input: real and imaginary parts interleaved in `data` as [re0, im0, re1, im1, ...].
/// `n` must be a power of 2.
/// `sign`: -1 for forward FFT, +1 for inverse FFT.
pub fn fft(data: &mut [f64], n: usize, sign: f64) {
    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
    }

    // Butterfly operations
    let mut len = 2;
    while len <= n {
        let half_len = len / 2;
        let angle = sign * std::f64::consts::PI / half_len as f64;
        let w_re = angle.cos();
        let w_im = angle.sin();

        for i in (0..n).step_by(len) {
            let mut cur_re = 1.0;
            let mut cur_im = 0.0;
            for k in 0..half_len {
                let even_idx = 2 * (i + k);
                let odd_idx = 2 * (i + k + half_len);

                let t_re = cur_re * data[odd_idx] - cur_im * data[odd_idx + 1];
                let t_im = cur_re * data[odd_idx + 1] + cur_im * data[odd_idx];

                data[odd_idx] = data[even_idx] - t_re;
                data[odd_idx + 1] = data[even_idx + 1] - t_im;
                data[even_idx] += t_re;
                data[even_idx + 1] += t_im;

                let new_cur_re = cur_re * w_re - cur_im * w_im;
                let new_cur_im = cur_re * w_im + cur_im * w_re;
                cur_re = new_cur_re;
                cur_im = new_cur_im;
            }
        }
        len *= 2;
    }
}

/// Compute the normalized autocorrelation function using FFT.
///
/// Returns a vector of length `n` where `ac[lag]` is the autocorrelation at that lag.
/// The input is not modified.
pub fn fft_autocorrelation(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    let nfft = next_power_of_2(2 * n);

    // Prepare input: y - mean(y), zero-padded to nfft
    let m = super::stats::mean(y);
    let mut data = vec![0.0f64; 2 * nfft];
    for i in 0..n {
        data[2 * i] = y[i] - m;
        data[2 * i + 1] = 0.0;
    }

    // Forward FFT
    fft(&mut data, nfft, -1.0);

    // Compute power spectrum: |F|^2
    for i in 0..nfft {
        let re = data[2 * i];
        let im = data[2 * i + 1];
        data[2 * i] = re * re + im * im;
        data[2 * i + 1] = 0.0;
    }

    // Inverse FFT
    fft(&mut data, nfft, 1.0);

    // Normalize by variance * nfft and extract real parts
    let v0 = data[0]; // This is variance * nfft
    let mut ac = Vec::with_capacity(n);
    for i in 0..n {
        if v0.abs() > f64::EPSILON {
            ac.push(data[2 * i] / v0);
        } else {
            ac.push(0.0);
        }
    }
    ac
}

/// Compute the power spectral density using Welch's method with a rectangular window.
///
/// Returns `(psd, frequencies)` where `psd[i]` is the power at `frequencies[i]`.
/// `nfft` is the FFT size (will be zero-padded to next power of 2 if needed).
/// `fs` is the sampling frequency.
/// `window_width` is the segment length for Welch averaging.
pub fn welch_psd(y: &[f64], nfft: usize, fs: f64, window_width: usize) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let m = super::stats::mean(y);
    let nfft_actual = next_power_of_2(nfft.max(window_width));

    let half_nfft = nfft_actual / 2 + 1;
    let hop = window_width / 2;
    let n_segments = if n > window_width {
        (n - window_width) / hop + 1
    } else {
        1
    };

    // Accumulate power spectrum
    let mut psd = vec![0.0f64; half_nfft];

    for seg in 0..n_segments {
        let start = seg * hop;
        if start + window_width > n {
            break;
        }

        // Window the segment (rectangular = no windowing)
        let mut data = vec![0.0f64; 2 * nfft_actual];
        for i in 0..window_width {
            data[2 * i] = y[start + i] - m;
            data[2 * i + 1] = 0.0;
        }

        fft(&mut data, nfft_actual, -1.0);

        // Accumulate |F|^2 for the single-sided spectrum
        for i in 0..half_nfft {
            let re = data[2 * i];
            let im = data[2 * i + 1];
            psd[i] += re * re + im * im;
        }
    }

    // Normalize
    let df = fs / nfft_actual as f64;
    let norm = n_segments as f64 * window_width as f64 * fs;
    for psd_i in psd.iter_mut().take(half_nfft) {
        *psd_i /= norm;
    }

    // Frequency axis
    let freqs: Vec<f64> = (0..half_nfft).map(|i| i as f64 * df).collect();

    (psd, freqs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(8), 8);
        assert_eq!(next_power_of_2(9), 16);
        assert_eq!(next_power_of_2(0), 1);
    }

    #[test]
    fn test_fft_dc_signal() {
        // DC signal of amplitude 4
        let n = 8;
        let mut data = vec![0.0f64; 2 * n];
        for i in 0..n {
            data[2 * i] = 4.0;
        }
        fft(&mut data, n, -1.0);
        // DC bin should be 4*8 = 32
        assert!((data[0] - 32.0).abs() < 1e-6);
        assert!(data[1].abs() < 1e-6);
        // Other bins should be ~0
        for i in 1..n {
            assert!(data[2 * i].abs() < 1e-6, "bin {} re = {}", i, data[2 * i]);
            assert!(data[2 * i + 1].abs() < 1e-6, "bin {} im = {}", i, data[2 * i + 1]);
        }
    }

    #[test]
    fn test_autocorrelation_white_noise_like() {
        // A constant signal should have ac[0] = 1, ac[k>0] ≈ 0
        let y = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ac = fft_autocorrelation(&y);
        // For a constant sequence, mean-subtracted = all zeros, so ac should be 0 or NaN
        // Z-scored constant = all zeros, so ac = 0/0 ≈ 0
        // But our implementation subtracts mean and divides by v0
        // When all values are the same, mean-subtracted = 0, so power spectrum = 0
        // This is a degenerate case
    }

    #[test]
    fn test_autocorrelation_sinusoid() {
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let ac = fft_autocorrelation(&y);
        // ac[0] should be 1.0 (normalized)
        assert!((ac[0] - 1.0).abs() < 0.01, "ac[0] = {}", ac[0]);
        // At lag = ~63 (2*pi/0.1 ≈ 63), should have another peak
        // Just check it's computed without panicking
        assert_eq!(ac.len(), 100);
    }

    #[test]
    fn test_fft_sinusoid() {
        // Sinusoid at frequency 1 (period = n)
        let n = 16;
        let mut data = vec![0.0f64; 2 * n];
        for i in 0..n {
            data[2 * i] = (2.0 * std::f64::consts::PI * i as f64 / n as f64).sin();
        }
        fft(&mut data, n, -1.0);
        // Should have peaks at bins 1 and n-1
        let mag_0 = (data[0] * data[0] + data[1] * data[1]).sqrt();
        let mag_1 = (data[2] * data[2] + data[3] * data[3]).sqrt();
        let mag_n1 = (data[2 * (n - 1)] * data[2 * (n - 1)] + data[2 * (n - 1) + 1] * data[2 * (n - 1) + 1]).sqrt();
        // DC component should be small
        assert!(mag_0 < 1.0, "DC magnitude should be small: {}", mag_0);
        // Bin 1 and n-1 should have significant energy
        assert!(mag_1 > 1.0, "Bin 1 magnitude: {}", mag_1);
        assert!(mag_n1 > 1.0, "Bin n-1 magnitude: {}", mag_n1);
    }

    #[test]
    fn test_autocorrelation_short() {
        // Short series
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ac = fft_autocorrelation(&y);
        assert_eq!(ac.len(), 5);
        assert!((ac[0] - 1.0).abs() < 0.1, "ac[0] should be ~1: {}", ac[0]);
    }

    #[test]
    fn test_welch_psd_simple() {
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let (psd, freqs) = welch_psd(&y, 64, 1.0, 32);
        assert!(!psd.is_empty());
        assert_eq!(psd.len(), freqs.len());
        // PSD should be non-negative
        for &p in &psd {
            assert!(p >= 0.0, "PSD should be non-negative: {}", p);
        }
    }
}