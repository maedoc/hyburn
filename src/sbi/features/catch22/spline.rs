//! Cubic B-spline fitting for periodicity detection.
//!
//! Implements least-squares cubic B-spline basis fit with 3 knots.

use super::stats;

/// Fit a cubic B-spline with 3 interior knots to the data via least squares.
///
/// Returns the fitted values (smooth trend).
pub fn splinefit(y: &[f64]) -> Vec<f64> {
    let n = y.len();
    if n < 5 {
        return y.to_vec();
    }

    // 3 interior knots placed at 25%, 50%, 75% of the data range
    let t1 = n as f64 * 0.25;
    let t2 = n as f64 * 0.50;
    let t3 = n as f64 * 0.75;

    // Build design matrix: cubic B-spline basis functions
    // We use a simplified approach: fit a piecewise polynomial
    // with basis [1, t, t^2, t^3, (t-t1)^3_+, (t-t2)^3_+, (t-t3)^3_+]
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

    // Design matrix: 7 basis functions
    let n_basis = 7;
    let mut design = vec![0.0f64; n * n_basis];

    for i in 0..n {
        let t = x[i];
        design[i * n_basis] = 1.0;
        design[i * n_basis + 1] = t;
        design[i * n_basis + 2] = t * t;
        design[i * n_basis + 3] = t * t * t;
        design[i * n_basis + 4] = if t > t1 { (t - t1).powi(3) } else { 0.0 };
        design[i * n_basis + 5] = if t > t2 { (t - t2).powi(3) } else { 0.0 };
        design[i * n_basis + 6] = if t > t3 { (t - t3).powi(3) } else { 0.0 };
    }

    // Solve via normal equations: (A^T A) c = A^T y
    let mut ata = vec![0.0f64; n_basis * n_basis];
    let mut aty = vec![0.0f64; n_basis];

    for i in 0..n_basis {
        for j in 0..n_basis {
            let mut sum = 0.0;
            for k in 0..n {
                sum += design[k * n_basis + i] * design[k * n_basis + j];
            }
            ata[i * n_basis + j] = sum;
        }
        let mut sum = 0.0;
        for k in 0..n {
            sum += design[k * n_basis + i] * y[k];
        }
        aty[i] = sum;
    }

    // Solve using Gaussian elimination with partial pivoting
    let mut augmented = vec![0.0f64; n_basis * (n_basis + 1)];
    for i in 0..n_basis {
        for j in 0..n_basis {
            augmented[i * (n_basis + 1) + j] = ata[i * n_basis + j];
        }
        augmented[i * (n_basis + 1) + n_basis] = aty[i];
    }

    // Forward elimination
    for col in 0..n_basis {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = augmented[col * (n_basis + 1) + col].abs();
        for row in (col + 1)..n_basis {
            let val = augmented[row * (n_basis + 1) + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        // Swap rows
        if max_row != col {
            for j in 0..=n_basis {
                augmented.swap(col * (n_basis + 1) + j, max_row * (n_basis + 1) + j);
            }
        }

        let pivot = augmented[col * (n_basis + 1) + col];
        if pivot.abs() < 1e-12 {
            // Singular matrix; return mean as trend
            let m = stats::mean(y);
            return vec![m; n];
        }

        for row in (col + 1)..n_basis {
            let factor = augmented[row * (n_basis + 1) + col] / pivot;
            for j in col..=n_basis {
                augmented[row * (n_basis + 1) + j] -= factor * augmented[col * (n_basis + 1) + j];
            }
        }
    }

    // Back substitution
    let mut coeffs = vec![0.0f64; n_basis];
    for i in (0..n_basis).rev() {
        let mut sum = augmented[i * (n_basis + 1) + n_basis];
        for j in (i + 1)..n_basis {
            sum -= augmented[i * (n_basis + 1) + j] * coeffs[j];
        }
        coeffs[i] = sum / augmented[i * (n_basis + 1) + i];
    }

    // Compute fitted values
    let mut fitted = vec![0.0f64; n];
    for i in 0..n {
        let t = x[i];
        fitted[i] = coeffs[0]
            + coeffs[1] * t
            + coeffs[2] * t * t
            + coeffs[3] * t * t * t
            + if t > t1 {
                coeffs[4] * (t - t1).powi(3)
            } else {
                0.0
            }
            + if t > t2 {
                coeffs[5] * (t - t2).powi(3)
            } else {
                0.0
            }
            + if t > t3 {
                coeffs[6] * (t - t3).powi(3)
            } else {
                0.0
            };
    }

    fitted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splinefit_trend() {
        // Linear trend: y = 2x + 5
        let y: Vec<f64> = (0..50).map(|i| 2.0 * i as f64 + 5.0).collect();
        let fitted = splinefit(&y);
        // The spline should capture the linear trend closely
        for (i, (&y_orig, &y_fit)) in y.iter().zip(fitted.iter()).enumerate() {
            assert!(
                (y_orig - y_fit).abs() < 5.0,
                "index {}: original = {}, fitted = {}",
                i,
                y_orig,
                y_fit
            );
        }
    }

    #[test]
    fn test_splinefit_short() {
        let y = vec![1.0, 2.0, 3.0];
        let fitted = splinefit(&y);
        assert_eq!(fitted, y); // Returns copy for short series
    }
}