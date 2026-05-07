//! Classic feature extraction: mean, variance, lag-1 autocorrelation.
//!
//! This is the original hyburn feature extractor, computing 3 summary
//! statistics per state variable, averaged over nodes and modes.

/// Extract classic summary statistics from a flat simulation trajectory.
///
/// `trajectory` is a flat `[n_steps, nvar, nnodes, nmodes]` array.
/// Returns a feature vector containing, for each state variable:
///   - mean across time, nodes and modes
///   - variance across time, nodes and modes
///   - lag-1 autocorrelation coefficient
pub fn extract_features_classic(trajectory: &[f32], shape: &[usize]) -> Vec<f32> {
    assert_eq!(
        shape.len(),
        4,
        "expected trajectory shape [n_steps, nvar, nnodes, nmodes]"
    );
    let n_steps = shape[0];
    let nvar = shape[1];
    let nnodes = shape[2];
    let nmodes = shape[3];
    let expected_len = n_steps * nvar * nnodes * nmodes;
    assert_eq!(
        trajectory.len(),
        expected_len,
        "trajectory length mismatch"
    );

    let mut features = Vec::with_capacity(nvar * 3);

    for var in 0..nvar {
        let mut mean_sum = 0.0f32;
        let mut var_sum = 0.0f32;
        let mut ac_sum = 0.0f32;

        for n in 0..nnodes {
            for m in 0..nmodes {
                let mut m1 = 0.0f32;
                for t in 0..n_steps {
                    let idx = ((t * nvar + var) * nnodes + n) * nmodes + m;
                    m1 += trajectory[idx];
                }
                m1 /= n_steps as f32;

                let mut v = 0.0f32;
                let mut c = 0.0f32;
                for t in 0..n_steps {
                    let idx = ((t * nvar + var) * nnodes + n) * nmodes + m;
                    let d = trajectory[idx] - m1;
                    v += d * d;
                    if t > 0 {
                        let idx_prev = (((t - 1) * nvar + var) * nnodes + n) * nmodes + m;
                        let d_prev = trajectory[idx_prev] - m1;
                        c += d_prev * d;
                    }
                }

                mean_sum += m1;
                if n_steps > 0 {
                    var_sum += v / n_steps as f32;
                }
                ac_sum += if v > 1e-12 { c / v } else { 0.0 };
            }
        }

        let n_spatial = (nnodes * nmodes) as f32;
        features.push(mean_sum / n_spatial);
        features.push(var_sum / n_spatial);
        features.push(ac_sum / n_spatial);
    }

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_features_constant() {
        let trajectory = vec![2.0f32; 4 * 2 * 2 * 1]; // 4 steps, 2 vars, 2 nodes, 1 mode
        let features = extract_features_classic(&trajectory, &[4, 2, 2, 1]);
        assert_eq!(features.len(), 6);
        // Means
        assert!((features[0] - 2.0).abs() < 1e-6);
        assert!((features[3] - 2.0).abs() < 1e-6);
        // Variances
        assert!(features[1].abs() < 1e-6);
        assert!(features[4].abs() < 1e-6);
        // Autocorrelations for constant signal should be ~0
        assert!(features[2].abs() < 1e-6, "ac0 = {}", features[2]);
        assert!(features[5].abs() < 1e-6, "ac1 = {}", features[5]);
    }

    #[test]
    fn test_extract_features_ramp() {
        // 3 steps, 1 var, 1 node, 1 mode: values [0, 1, 2]
        let trajectory = vec![0.0f32, 1.0, 2.0];
        let features = extract_features_classic(&trajectory, &[3, 1, 1, 1]);
        assert_eq!(features.len(), 3);
        assert!((features[0] - 1.0).abs() < 1e-6); // mean
        assert!((features[1] - 2.0f32 / 3.0).abs() < 1e-5); // var
        assert!(features[2].abs() < 1e-6, "ac = {}", features[2]);
    }
}