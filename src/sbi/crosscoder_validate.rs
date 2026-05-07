//! Validation test: compare Hyburn CrossCoder against vbjax reference outputs.
//!
//! Loads Python-generated cohort NPY files, trains a Hyburn CrossCoder with
//! matching hyperparameters, and checks that reconstruction MSE and latent
//! structure are within reasonable bounds of the vbjax reference.

#[cfg(test)]
mod tests {
    use burn::backend::autodiff::Autodiff;
    use burn::backend::ndarray::NdArray;
    use burn::tensor::{Tensor, TensorData};

    use crate::io::read_npy_f32;
    use crate::sbi::crosscoder::{CrossCoder, CrossCoderConfig};
    use crate::sbi::crosscoder_train::train_crosscoder;
    use crate::sbi::crosscoder_cohort::fit_mvn_over_latents;

    type B = Autodiff<NdArray<f32>>;

    /// Pearson correlation coefficient between two equal-length vectors.
    fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len());
        let n = x.len() as f32;
        let mean_x = x.iter().sum::<f32>() / n;
        let mean_y = y.iter().sum::<f32>() / n;

        let mut num = 0.0f32;
        let mut den_x = 0.0f32;
        let mut den_y = 0.0f32;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }

        let den = den_x.sqrt() * den_y.sqrt();
        if den < 1e-12 {
            0.0
        } else {
            num / den
        }
    }

    /// Mean squared error between two equal-length vectors.
    fn compute_mse(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let n = a.len() as f32;
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / n
    }

    #[test]
    fn test_crosscoder_matches_vbjax_reference() {
        let device = Default::default();

        // 1. Load cohort data from Python-generated NPY files
        let (view1_data, view1_shape) = read_npy_f32("tests/validate_output/cohort_view1.npy").unwrap();
        let (view2_data, view2_shape) = read_npy_f32("tests/validate_output/cohort_view2.npy").unwrap();

        assert_eq!(view1_shape, vec![50, 45]);
        assert_eq!(view2_shape, vec![50, 105]);

        let n_samples = view1_shape[0];
        let input_dims = vec![view1_shape[1], view2_shape[1]];
        let latent_dim = 4;

        // 2. Train Hyburn CrossCoder with same hyperparameters as vbjax
        let cfg = CrossCoderConfig {
            input_dims: input_dims.clone(),
            latent_dim,
            learning_rate: 3e-4,
            n_epochs: 2000,
            batch_size: 64,
            beta: 1e-3,
            grad_clip: 5.0,
        };

        let data = vec![view1_data.clone(), view2_data.clone()];
        let shapes = vec![(n_samples, view1_shape[1]), (n_samples, view2_shape[1])];

        let (model, loss_history) =
            train_crosscoder(&data, &shapes, &cfg, Some(2000), Some(64));

        // Check training converged
        let initial_loss = loss_history.first().unwrap().1;
        let final_loss = loss_history.last().unwrap().1;
        println!("Hyburn initial loss: {}", initial_loss);
        println!("Hyburn final loss: {}", final_loss);
        assert!(
            final_loss < initial_loss * 0.5,
            "Training did not converge: initial={}, final={}",
            initial_loss,
            final_loss
        );

        // 3. Encode all views and compute consensus latent codes
        let view1_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(view1_data.clone(), vec![n_samples, view1_shape[1]]),
            &device,
        );
        let view2_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(view2_data.clone(), vec![n_samples, view2_shape[1]]),
            &device,
        );

        let (mu1, _) = model.views[0].encode(view1_tensor);
        let (mu2, _) = model.views[1].encode(view2_tensor);

        // Consensus: average μ from both views (same as vbjax)
        let consensus = (mu1.clone() + mu2.clone()).mul_scalar(0.5f32);
        let (consensus_flat, _) = crate::io::tensor_to_flat_f32(consensus);

        // 4. Compare latent distribution to vbjax reference
        let (vbjax_latents, vbjax_latent_shape) =
            read_npy_f32("tests/validate_output/vbjax_latents.npy").unwrap();
        assert_eq!(vbjax_latent_shape, vec![50, 4]);

        // Use Pearson correlation per latent dimension as a relaxed check.
        // Latent spaces may differ by rotation / scaling / permutation.
        // We therefore compute a full pairwise correlation matrix and match
        // each vbjax dimension to its best hyburn counterpart, then assert
        // on the mean of those best-match correlations.
        let mut corr_matrix = vec![vec![0.0f32; latent_dim]; latent_dim];
        for i in 0..latent_dim {
            let hyburn_i: Vec<f32> = (0..n_samples)
                .map(|s| consensus_flat[s * latent_dim + i])
                .collect();
            for j in 0..latent_dim {
                let vbjax_j: Vec<f32> = (0..n_samples)
                    .map(|s| vbjax_latents[s * latent_dim + j])
                    .collect();
                corr_matrix[i][j] = pearson_correlation(&hyburn_i, &vbjax_j
                ).abs();
            }
        }

        // Greedy best-match: for each hyburn dim, find the best unmatched vbjax dim
        let mut matched_vbjax = vec![false; latent_dim];
        let mut best_corrs = Vec::with_capacity(latent_dim);
        for _ in 0..latent_dim {
            let mut best_val = 0.0f32;
            let (mut best_i, mut best_j) = (0, 0);
            for i in 0..latent_dim {
                for j in 0..latent_dim {
                    if !matched_vbjax[j] && corr_matrix[i][j] > best_val {
                        best_val = corr_matrix[i][j];
                        best_i = i;
                        best_j = j;
                    }
                }
            }
            matched_vbjax[best_j] = true;
            best_corrs.push(best_val);
            println!(
                "Latent match hyburn_dim{} -> vbjax_dim{}: corr = {}",
                best_i, best_j, best_val
            );
        }

        let mean_best_corr = best_corrs.iter().sum::<f32>() / latent_dim as f32;
        println!("Mean best-match correlation: {}", mean_best_corr);
        assert!(
            mean_best_corr > 0.35,
            "Mean best-match correlation too low: {}",
            mean_best_corr
        );

        // Also compare latent covariance structure
        let (hy_mean, hy_cov) = fit_mvn_over_latents(&consensus_flat, n_samples, latent_dim);
        println!("Hyburn latent mean: {:?}", hy_mean);
        println!(
            "Hyburn latent cov diagonal: {:?}",
            (0..latent_dim)
                .map(|d| hy_cov[d * latent_dim + d])
                .collect::<Vec<_>>()
        );

        // 5. Compare reconstruction MSE to vbjax reference
        let vbjax_summary: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string("tests/validate_output/summary.json").unwrap(),
        )
        .unwrap();

        let vbjax_recon_v1 = vbjax_summary["reconstruction_mse"]["view1_to_view1"]
            .as_f64()
            .unwrap();
        let vbjax_recon_v2 = vbjax_summary["reconstruction_mse"]["view2_to_view2"]
            .as_f64()
            .unwrap();

        // Decode through Hyburn
        let dec1 = model.views[0].decode(mu1);
        let dec2 = model.views[1].decode(mu2);

        let hy_recon_v1 = crate::io::tensor_to_flat_f32(dec1).0;
        let hy_recon_v2 = crate::io::tensor_to_flat_f32(dec2).0;

        let v1_mse = compute_mse(&view1_data, &hy_recon_v1);
        let v2_mse = compute_mse(&view2_data, &hy_recon_v2);

        println!("Hyburn recon MSE v1: {}, vbjax: {}", v1_mse, vbjax_recon_v1);
        println!("Hyburn recon MSE v2: {}, vbjax: {}", v2_mse, vbjax_recon_v2);

        // Hyburn MSE should be within ~10× of vbjax (same architecture,
        // but different init/optimizer and training stochasticity).
        assert!(
            v1_mse < vbjax_recon_v1 as f32 * 10.0,
            "View1 recon MSE too high: {} vs vbjax {}",
            v1_mse,
            vbjax_recon_v1
        );
        assert!(
            v2_mse < vbjax_recon_v2 as f32 * 10.0,
            "View2 recon MSE too high: {} vs vbjax {}",
            v2_mse,
            vbjax_recon_v2
        );
    }
}
