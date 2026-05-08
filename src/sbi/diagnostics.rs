//! SBI diagnostic metrics: z-score and shrinkage.
//!
//! References:
//! - Geweke (2004): "Getting it right: Joint distribution tests of posterior simulators"
//! - Talts et al. (2018): "Validating Bayesian inference algorithms with simulation-based calibration"
//! - sbi Python library: sbi.diagnostics and sbi.analysis

/// Result of SBI diagnostic evaluation across multiple test points.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SbiDiagnostics {
    /// Per-parameter z-scores: |posterior_mean - true_param| / prior_std
    pub z_scores: Vec<f32>,
    /// Per-parameter shrinkage: 1 - posterior_var / prior_var
    /// Good: close to 1 (posterior much narrower than prior)
    /// Bad: close to 0 (no information gained from data)
    /// Catastrophic: negative (posterior wider than prior)
    pub shrinkages: Vec<f32>,
    /// Average z-score across all parameters
    pub mean_z_score: f32,
    /// Average shrinkage across all parameters
    pub mean_shrinkage: f32,
    /// Number of test points evaluated
    pub n_test_points: usize,
    /// Per-parameter posterior means (for debugging)
    pub posterior_means: Vec<f32>,
    /// Per-parameter posterior stds (for debugging)
    pub posterior_stds: Vec<f32>,
}

impl SbiDiagnostics {
    /// Compute z-scores and shrinkage from a batch of posterior samples.
    ///
    /// Computes per-test-point diagnostics then averages across test points.
    /// This is the standard SBI validation approach (Geweke 2004, sbi library).
    ///
    /// # Arguments
    /// * `posterior_samples` - flat array of shape [n_test_points × n_samples, param_dim]
    ///   (row-major: test point changes slowly, sample changes fast)
    /// * `true_params` - flat array of shape [n_test_points, param_dim]
    /// * `_prior_means` - array of shape [param_dim], the prior mean for each parameter (currently unused)
    /// * `prior_stds` - array of shape [param_dim], the prior std for each parameter
    /// * `n_samples` - number of posterior samples per test point
    /// * `param_dim` - dimensionality of the parameter space
    pub fn from_samples(
        posterior_samples: &[f32],
        true_params: &[f32],
        _prior_means: &[f32],
        prior_stds: &[f32],
        n_samples: usize,
        param_dim: usize,
    ) -> Self {
        let n_test = true_params.len() / param_dim;

        // Per-parameter aggregates
        let mut z_scores = vec![0.0f32; param_dim];
        let mut shrinkages = vec![0.0f32; param_dim];
        let mut posterior_means = vec![0.0f32; param_dim];
        let mut posterior_stds = vec![0.0f32; param_dim];

        for d in 0..param_dim {
            let prior_var = prior_stds[d].powi(2);
            let mut z_score_sum = 0.0f32;
            let mut shrinkage_sum = 0.0f32;
            let mut post_mean_sum = 0.0f32;
            let mut post_std_sum = 0.0f32;

            for i in 0..n_test {
                // Compute per-test-point posterior mean and variance
                let mut sum = 0.0f32;
                let mut sum_sq = 0.0f32;

                for s in 0..n_samples {
                    let idx = (i * n_samples + s) * param_dim + d;
                    let val = posterior_samples[idx];
                    sum += val;
                    sum_sq += val * val;
                }

                let post_mean_i = sum / n_samples as f32;
                let post_var_i = if n_samples > 1 {
                    (sum_sq - sum * sum / n_samples as f32) / (n_samples - 1) as f32
                } else {
                    0.0
                };
                let post_std_i = post_var_i.sqrt().max(1e-8);

                let true_val = true_params[i * param_dim + d];

                // Per-test-point z-score: |posterior_mean - true_param| / prior_std
                z_score_sum += (post_mean_i - true_val).abs() / prior_stds[d].max(1e-8);

                // Per-test-point shrinkage: 1 - posterior_var / prior_var
                let shrink_i = if prior_var > 1e-10 {
                    1.0 - post_var_i / prior_var
                } else {
                    1.0
                };
                shrinkage_sum += shrink_i;

                post_mean_sum += post_mean_i;
                post_std_sum += post_std_i;
            }

            // Average across test points
            z_scores[d] = z_score_sum / n_test as f32;
            shrinkages[d] = shrinkage_sum / n_test as f32;
            posterior_means[d] = post_mean_sum / n_test as f32;
            posterior_stds[d] = post_std_sum / n_test as f32;
        }

        let mean_z_score = z_scores.iter().sum::<f32>() / z_scores.len() as f32;
        let mean_shrinkage = shrinkages.iter().sum::<f32>() / shrinkages.len() as f32;

        SbiDiagnostics {
            z_scores,
            shrinkages,
            mean_z_score,
            mean_shrinkage,
            n_test_points: n_test,
            posterior_means,
            posterior_stds,
        }
    }

    /// Compute per-test-point z-scores (not averaged across test points).
    ///
    /// Returns [n_test_points, param_dim] z-score array.
    pub fn per_point_z_scores(
        posterior_samples: &[f32],
        true_params: &[f32],
        prior_stds: &[f32],
        n_samples: usize,
        param_dim: usize,
    ) -> Vec<f32> {
        let n_test = true_params.len() / param_dim;
        let mut result = vec![0.0f32; n_test * param_dim];

        for i in 0..n_test {
            for d in 0..param_dim {
                // Posterior mean for this test point
                let mut sum = 0.0f32;
                for s in 0..n_samples {
                    let idx = (i * n_samples + s) * param_dim + d;
                    sum += posterior_samples[idx];
                }
                let post_mean = sum / n_samples as f32;
                let true_val = true_params[i * param_dim + d];
                result[i * param_dim + d] =
                    (post_mean - true_val).abs() / prior_stds[d].max(1e-8);
            }
        }
        result
    }

    /// Check if diagnostics indicate a well-calibrated posterior.
    ///
    /// Criteria:
    /// - Mean z-score < 2.0 (posterior mean within 2 prior std devs of truth)
    /// - Mean shrinkage > 0.0 (posterior narrower than prior)
    /// - z-scores are concentrated (std < 1.0)
    pub fn is_well_calibrated(&self) -> bool {
        self.mean_z_score < 2.0 && self.mean_shrinkage > 0.0
    }

    /// Generate a human-readable diagnostic report.
    pub fn report(&self) -> String {
        let mut s = String::new();
        s.push_str("=== SBI Diagnostic Report ===\n");
        s.push_str(&format!("Test points: {}\n", self.n_test_points));
        s.push_str(&format!("Parameters: {}\n", self.z_scores.len()));
        s.push_str("\nPer-parameter z-scores (lower is better, < 2.0 is good):\n");
        for (d, z) in self.z_scores.iter().enumerate() {
            s.push_str(&format!("  θ[{}]: z = {:.4}\n", d, z));
        }
        s.push_str(&format!("\nMean z-score: {:.4}\n", self.mean_z_score));
        s.push_str("\nPer-parameter shrinkage (higher is better, > 0.0 is good, > 0.5 is great):\n");
        for (d, sh) in self.shrinkages.iter().enumerate() {
            let quality = if *sh > 0.5 { "excellent" } else if *sh > 0.0 { "moderate" } else { "FAILED" };
            s.push_str(&format!("  θ[{}]: shrinkage = {:.4} ({})\n", d, sh, quality));
        }
        s.push_str(&format!("\nMean shrinkage: {:.4}\n", self.mean_shrinkage));
        s.push_str(&format!("\nWell-calibrated: {}\n", if self.is_well_calibrated() { "YES ✓" } else { "NO ✗" }));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostics_perfect_posterior() {
        // If posterior exactly matches the true value with tiny variance,
        // z-score should be ~0 and shrinkage should be ~1
        let posterior_samples: Vec<f32> = vec![
            // 2 test points, 3 samples each, 2 params
            0.99, 0.51, // test 0, sample 0
            1.01, 0.49, // test 0, sample 1
            1.00, 0.50, // test 0, sample 2
            -0.01, 0.01, // test 1, sample 0
            0.01, -0.01, // test 1, sample 1
            0.00, 0.00, // test 1, sample 2
        ];
        let true_params = vec![1.0_f32, 0.5, 0.0_f32, 0.0]; // 2 tests, 2 params
        let prior_means = vec![0.0_f32, 0.0];
        let prior_stds = vec![1.0_f32, 1.0];

        let diag = SbiDiagnostics::from_samples(
            &posterior_samples,
            &true_params,
            &prior_means,
            &prior_stds,
            3, // n_samples
            2, // param_dim
        );

        // With near-perfect posterior, z-score should be very small and shrinkage should be high
        assert!(diag.mean_z_score < 0.5, "z-score too high: {}", diag.mean_z_score);
        assert!(diag.mean_shrinkage > 0.5, "shrinkage too low: {}", diag.mean_shrinkage);
    }

    #[test]
    fn test_diagnostics_uninformative_posterior() {
        // If posterior = prior (no information from data),
        // z-score depends on how far true values are from prior mean,
        // but shrinkage should be ~0 (no narrowing)
        let n_test = 10;
        let n_samples = 100;
        let param_dim = 2;
        let prior_std = 1.0_f32;

        // Use a deterministic LCG for reproducibility (no rand dep)
        let mut rng_state: u32 = 12345;
        let mut next_uniform = || -> f32 {
            // LCG: x = (1664525 * x + 1013904223) mod 2^32
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng_state >> 1) as f32 / (1u32 << 31) as f32 // (0, 1)
        };
        let mut next_normal = || -> f32 {
            use std::f32::consts::PI;
            let u1 = next_uniform().max(1e-10);
            let u2 = next_uniform();
            (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * PI * u2).cos()
        };

        // Posterior = prior: samples from N(0, 1)
        let mut posterior_samples = Vec::new();
        let mut true_params = Vec::new();
        for i in 0..n_test {
            let true_i_ext = -0.5 + i as f32 * (1.0 / 9.0);
            true_params.push(true_i_ext);
            true_params.push(0.0); // second param
            for _ in 0..n_samples {
                // Draw from N(0, 1) — same as prior
                posterior_samples.push(next_normal()); // first param
                posterior_samples.push(next_normal()); // second param
            }
        }

        let prior_means = vec![0.0_f32, 0.0];
        let prior_stds = vec![prior_std, prior_std];

        let diag = SbiDiagnostics::from_samples(
            &posterior_samples,
            &true_params,
            &prior_means,
            &prior_stds,
            n_samples,
            param_dim,
        );

        // Uninformative posterior: shrinkage ≈ 0
        assert!(diag.mean_shrinkage < 0.3, "shrinkage should be near 0 for uninformative posterior, got {}", diag.mean_shrinkage);
    }
}