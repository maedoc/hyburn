//! Functional connectivity (FC) feature extraction.
//!
//! Computes static and dynamic FC matrices from trajectories, along with
//! summary statistics that capture network-level coupling patterns.

/// Compute the Pearson correlation matrix from a node-level time-series.
///
/// `trajectory` has shape `[n_steps, nnodes]` in row-major order.
/// Returns the upper-triangular (flat) FC matrix of shape `[nnodes, nnodes]`
/// as a flat `Vec<f32>`.
pub fn fc_matrix(trajectory: &[f32], n_steps: usize, nnodes: usize) -> Vec<f32> {
    assert_eq!(trajectory.len(), n_steps * nnodes);

    // Compute per-node means
    let mut means = vec![0.0f64; nnodes];
    for t in 0..n_steps {
        for n in 0..nnodes {
            means[n] += trajectory[t * nnodes + n] as f64;
        }
    }
    for means_n in means.iter_mut().take(nnodes) {
        *means_n /= n_steps as f64;
    }

    // Compute per-node stds
    let mut stds = vec![0.0f64; nnodes];
    for t in 0..n_steps {
        for n in 0..nnodes {
            let d = trajectory[t * nnodes + n] as f64 - means[n];
            stds[n] += d * d;
        }
    }
    for stds_n in stds.iter_mut().take(nnodes) {
        *stds_n = (*stds_n / (n_steps as f64)).sqrt();
        if *stds_n < 1e-12 {
            *stds_n = 1.0; // avoid divide-by-zero for constant series
        }
    }

    // Compute Pearson correlation matrix
    let mut corr = vec![0.0f32; nnodes * nnodes];
    for i in 0..nnodes {
        for j in 0..=i {
            let mut cov = 0.0f64;
            for t in 0..n_steps {
                let di = trajectory[t * nnodes + i] as f64 - means[i];
                let dj = trajectory[t * nnodes + j] as f64 - means[j];
                cov += di * dj;
            }
            cov /= n_steps as f64;
            let r = (cov / (stds[i] * stds[j])).clamp(-1.0, 1.0) as f32;
            corr[i * nnodes + j] = r;
            if i != j {
                corr[j * nnodes + i] = r;
            }
        }
    }
    corr
}

/// Statistics of the upper-triangular (off-diagonal) FC values.
///
/// Returns: `[mean, std, min, max, median]`.
pub fn fc_stats(fc: &[f32], nnodes: usize) -> Vec<f32> {
    let mut vals = Vec::with_capacity(nnodes * (nnodes - 1) / 2);
    for i in 0..nnodes {
        for j in (i + 1)..nnodes {
            vals.push(fc[i * nnodes + j]);
        }
    }
    if vals.is_empty() {
        return vec![f32::NAN; 5];
    }

    let mean = vals.iter().copied().sum::<f32>() / vals.len() as f32;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    let std = var.sqrt();
    let min = vals.iter().copied().fold(f32::INFINITY, |a, b| a.min(b));
    let max = vals.iter().copied().fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let mut sorted = vals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];

    vec![mean, std, min, max, median]
}

/// Dynamic functional connectivity (FCD) matrix using a sliding window.
///
/// `trajectory` has shape `[n_steps, nnodes]`.
/// `window_len` is the window size in steps.
/// `stride` is the step between consecutive windows.
///
/// Returns the FCD matrix as a flat `[n_windows, n_windows]` array.
pub fn fcd_matrix(
    trajectory: &[f32],
    n_steps: usize,
    nnodes: usize,
    window_len: usize,
    stride: usize,
) -> Vec<f32> {
    assert!(window_len > 1 && window_len <= n_steps, "window_len must be in [2, n_steps]");
    assert!(stride > 0, "stride must be > 0");

    // Number of windows
    let n_windows = ((n_steps.saturating_sub(window_len)) / stride) + 1;
    if n_windows <= 1 {
        return Vec::new();
    }

    // Compute FC per window → list of flat FC vectors
    let mut window_fcs: Vec<Vec<f32>> = Vec::with_capacity(n_windows);
    for w in 0..n_windows {
        let start = w * stride;
        let end = (start + window_len).min(n_steps);
        let actual_len = end - start;
        if actual_len < 2 {
            continue;
        }
        let window_slice = &trajectory[start * nnodes..end * nnodes];
        let fc = fc_matrix(window_slice, actual_len, nnodes);
        window_fcs.push(fc);
    }

    let nw = window_fcs.len();
    if nw <= 1 {
        return Vec::new();
    }

    // FCD = Pearson correlation between window-FC upper triangles
    let mut fcd = vec![0.0f32; nw * nw];
    for i in 0..nw {
        fcd[i * nw + i] = 1.0;
        for j in (i + 1)..nw {
            let r = upper_triangle_correlation(&window_fcs[i], &window_fcs[j], nnodes,
            );
            fcd[i * nw + j] = r;
            fcd[j * nw + i] = r;
        }
    }
    fcd
}

/// Correlation between the upper-triangular elements of two FC matrices.
fn upper_triangle_correlation(a: &[f32], b: &[f32], nnodes: usize) -> f32 {
    let n = nnodes * (nnodes - 1) / 2;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    for i in 0..nnodes {
        for j in (i + 1)..nnodes {
            let val_a = a[i * nnodes + j] as f64;
            let val_b = b[i * nnodes + j] as f64;
            sum_a += val_a;
            sum_b += val_b;
        }
    }
    let n_f64 = n as f64;
    let mean_a = sum_a / n_f64;
    let mean_b = sum_b / n_f64;

    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;
    for i in 0..nnodes {
        for j in (i + 1)..nnodes {
            let da = a[i * nnodes + j] as f64 - mean_a;
            let db = b[i * nnodes + j] as f64 - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }
    }

    if var_a <= 0.0 || var_b <= 0.0 {
        0.0
    } else {
        (cov / (var_a.sqrt() * var_b.sqrt())).clamp(-1.0, 1.0) as f32
    }
}

/// Statistics of the upper-triangular (off-diagonal) FCD values.
pub fn fcd_stats(fcd: &[f32], n_windows: usize) -> Vec<f32> {
    if n_windows <= 1 {
        return vec![f32::NAN; 5];
    }
    fc_stats(fcd, n_windows)
}

/// Homotopic correlation: average FC(r, l) for symmetric node pairs.
///
/// Assumes nodes are ordered such that node `i` pairs with node `nnodes-1-i`.
/// Returns the mean and std of homotopic correlations.
pub fn homotopic_fc(fc: &[f32], nnodes: usize) -> (f32, f32, usize) {
    let n_pairs = nnodes / 2;
    if n_pairs == 0 {
        return (f32::NAN, f32::NAN, 0);
    }

    let mut vals = Vec::with_capacity(n_pairs);
    for i in 0..n_pairs {
        let j = nnodes - 1 - i;
        vals.push(fc[i * nnodes + j]);
    }

    let mean = vals.iter().copied().sum::<f32>() / vals.len() as f32;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    (mean, var.sqrt(), n_pairs)
}

/// Flatten the upper-triangular elements of a symmetric matrix (excluding diagonal).
pub fn upper_triangle_flat(mat: &[f32], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            out.push(mat[i * n + j]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fc_matrix_identity() {
        // Two perfectly correlated nodes
        let trajectory = vec![
            1.0, 1.0,
            2.0, 2.0,
            3.0, 3.0,
        ];
        let fc = fc_matrix(&trajectory, 3, 2);
        assert!((fc[0] - 1.0).abs() < 1e-5, "diagonal should be 1");
        assert!((fc[1] - 1.0).abs() < 1e-5, "off-diagonal should be 1 for perfectly correlated");
        assert!((fc[2] - 1.0).abs() < 1e-5, "symmetric element");
        assert!((fc[3] - 1.0).abs() < 1e-5, "diagonal should be 1");
    }

    #[test]
    fn test_fc_matrix_anti_correlated() {
        // Two perfectly anti-correlated nodes
        let trajectory = vec![
            1.0, -1.0,
            2.0, -2.0,
            3.0, -3.0,
        ];
        let fc = fc_matrix(&trajectory, 3, 2);
        assert!((fc[1] + 1.0).abs() < 1e-5, "off-diagonal should be -1 for anti-correlated");
    }

    #[test]
    fn test_fc_stats() {
        let fc = vec![
            1.0, 0.5, -0.3,
            0.5, 1.0, 0.8,
            -0.3, 0.8, 1.0,
        ];
        let stats = fc_stats(&fc, 3);
        assert_eq!(stats.len(), 5);
        // Off-diagonal values: 0.5, -0.3, 0.8
        assert!((stats[0] - (0.5 - 0.3 + 0.8) / 3.0).abs() < 1e-5, "mean");
        assert!(stats[1] >= 0.0, "std should be non-negative");
        assert!((stats[2] + 0.3).abs() < 1e-5, "min");
        assert!((stats[3] - 0.8).abs() < 1e-5, "max");
    }

    #[test]
    fn test_fcd_matrix() {
        // 10 steps, 3 nodes
        let trajectory: Vec<f32> = (0..30).map(|i| (i as f32).sin()).collect();
        let fcd = fcd_matrix(&trajectory, 10, 3, 4, 2);
        let nw = fcd.len().isqrt(); // approximate sqrt
        assert!(nw >= 2, "should have at least 2 windows (10-4)/2+1 = 4");
        assert!((fcd[0] - 1.0).abs() < 1e-5, "diagonal should be 1");
    }

    #[test]
    fn test_homotopic_fc() {
        // 4 nodes, symmetric pairs: (0,3), (1,2)
        let fc = vec![
            1.0, 0.1, 0.2, 0.9,
            0.1, 1.0, 0.8, 0.3,
            0.2, 0.8, 1.0, 0.4,
            0.9, 0.3, 0.4, 1.0,
        ];
        let (mean, std, n_pairs) = homotopic_fc(&fc, 4);
        assert_eq!(n_pairs, 2);
        // Values: fc[0][3] = 0.9, fc[1][2] = 0.8
        assert!((mean - 0.85).abs() < 1e-5, "mean homotopic");
        let expected_std = ((0.05f32 * 0.05 + 0.05 * 0.05) / 2.0).sqrt();
        assert!((std - expected_std).abs() < 1e-5, "std homotopic");
    }
}
