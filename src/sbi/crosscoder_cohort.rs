//! Cohort data loading, encoding, and MVN prior fitting for CrossCoder.
//!
//! Workflow:
//! 1. Load cohort connectomes from NPY files (one per view).
//! 2. Encode all samples through the trained CrossCoder → per-view μ latents.
//! 3. Average latents across views to get a consensus latent per subject.
//! 4. Fit a full-covariance MVN(mean, cov) over the consensus latents.
//! 5. Sample new latents from the MVN and decode to generate synthetic data.

use crate::error::Result;
#[cfg(not(target_arch = "wasm32"))]
use crate::io::read_npy_f32;
use crate::sbi::crosscoder::CrossCoder;
use burn::tensor::{backend::Backend, Tensor, TensorData};

/// Cohort data per view and the shape of each view.
pub type CohortData = (Vec<Vec<f32>>, Vec<(usize, usize)>);

/// Load a cohort of multi-view connectomes from NPY files.
///
/// `paths[i]` is the NPY file path for view `i`.
/// Each file must be `[n_samples, input_dim_i]` in row-major order.
///
/// Returns `(data_vecs, shapes)` where `shapes[i] = (n_samples, input_dim_i)`.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_cohort_from_npy(paths: &[impl AsRef<str>]) -> Result<CohortData> {
    let mut data_vecs = Vec::with_capacity(paths.len());
    let mut shapes = Vec::with_capacity(paths.len());
    let mut n_samples = None;

    for path in paths {
        let path_str = path.as_ref();
        let (data, shape) = read_npy_f32(path_str)?;
        if shape.len() != 2 {
            return Err(crate::error::SimulationError::InvalidConfig(format!(
                "Cohort NPY {} must be 2D, got {}D",
                path_str, shape.len()
            )));
        }
        let ns = shape[0];
        let dim = shape[1];
        match n_samples {
            Some(expected) if expected != ns => {
                return Err(crate::error::SimulationError::InvalidConfig(format!(
                    "Cohort NPY {} has {} samples, expected {}",
                    path_str, ns, expected
                )));
            }
            None => n_samples = Some(ns),
            _ => {}
        }
        data_vecs.push(data);
        shapes.push((ns, dim));
    }

    Ok((data_vecs, shapes))
}

/// Fit a full-covariance MVN over encoded latents.
///
/// `latents_flat` is a flat `[n_samples, latent_dim]` row-major array.
///
/// Returns `(mean_vec, cov_flat)` where `cov_flat` is row-major `[latent_dim, latent_dim]`.
pub fn fit_mvn_over_latents(latents_flat: &[f32], n_samples: usize, latent_dim: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(latents_flat.len(), n_samples * latent_dim);

    // Compute mean
    let mut mean = vec![0.0f64; latent_dim];
    for chunk in latents_flat.chunks_exact(latent_dim) {
        for (mean_d, &val) in mean.iter_mut().zip(chunk.iter()) {
            *mean_d += val as f64;
        }
    }
    for mean_d in mean.iter_mut().take(latent_dim) {
        *mean_d /= n_samples as f64;
    }

    // Compute covariance
    let mut cov = vec![0.0f64; latent_dim * latent_dim];
    for s in 0..n_samples {
        for i in 0..latent_dim {
            let di = latents_flat[s * latent_dim + i] as f64 - mean[i];
            for j in 0..latent_dim {
                let dj = latents_flat[s * latent_dim + j] as f64 - mean[j];
                cov[i * latent_dim + j] += di * dj;
            }
        }
    }
    for i in 0..latent_dim {
        for j in 0..latent_dim {
            cov[i * latent_dim + j] /= (n_samples - 1) as f64; // sample covariance
        }
    }

    // Add small regularisation to diagonal for positive definiteness
    for i in 0..latent_dim {
        cov[i * latent_dim + i] += 1e-6;
    }

    let mean_f32: Vec<f32> = mean.iter().map(|v| *v as f32).collect();
    let cov_f32: Vec<f32> = cov.iter().map(|v| *v as f32).collect();
    (mean_f32, cov_f32)
}

/// Full-covariance MVN prior for latent sampling.
#[derive(Debug, Clone)]
pub struct MvnPrior {
    pub mean: Vec<f32>,
    pub cov: Vec<f32>,
    pub latent_dim: usize,
    /// Cholesky factor L (lower triangular) such that cov = L * L^T.
    pub chol: Vec<f32>,
}

impl MvnPrior {
    pub fn from_mean_cov(mean: Vec<f32>, cov: Vec<f32>, latent_dim: usize) -> Self {
        let chol = cholesky_decompose(&cov, latent_dim);
        Self {
            mean,
            cov,
            latent_dim,
            chol,
        }
    }

    /// Sample `n` latent vectors from the MVN.
    pub fn sample(&self, n: usize, seed: Option<u64>,
    ) -> Vec<f32> {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::{Distribution, Normal};

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let std_normal = Normal::new(0.0, 1.0).unwrap();

        let mut samples = Vec::with_capacity(n * self.latent_dim);
        for _ in 0..n {
            // z = L * ε + μ
            let mut eps = vec![0.0f64; self.latent_dim];
            for eps_i in eps.iter_mut().take(self.latent_dim) {
                *eps_i = std_normal.sample(&mut rng);
            }
            for (i, &mean_i) in self.mean.iter().enumerate().take(self.latent_dim) {
                let mut sum = 0.0f64;
                for (j, &eps_j) in eps.iter().enumerate().take(i + 1) {
                    sum += self.chol[i * self.latent_dim + j] as f64 * eps_j;
                }
                samples.push((sum + mean_i as f64) as f32);
            }
        }
        samples
    }
}

/// Simple Cholesky decomposition for a symmetric positive-definite matrix.
/// Returns the lower-triangular factor L in row-major order.
fn cholesky_decompose(cov: &[f32], n: usize) -> Vec<f32> {
    let mut l = vec![0.0f64; n * n];
    for j in 0..n {
        let mut sum = 0.0f64;
        for k in 0..j {
            sum += l[j * n + k].powi(2);
        }
        let diag = cov[j * n + j] as f64 - sum;
        if diag <= 0.0 {
            // numerical issue; regularise and retry
            l[j * n + j] = 1e-6f64.sqrt();
        } else {
            l[j * n + j] = diag.sqrt();
        }
        for i in (j + 1)..n {
            let mut sum = 0.0f64;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            let denom = l[j * n + j];
            if denom.abs() > 1e-12 {
                l[i * n + j] = (cov[i * n + j] as f64 - sum) / denom;
            } else {
                l[i * n + j] = 0.0;
            }
        }
    }
    l.iter().map(|v| *v as f32).collect()
}

/// Encode a full cohort through the CrossCoder and average latents across views.
///
/// `data[i]` is `[n_samples, input_dim_i]` flat for view `i`.
///
/// Returns consensus latents as `[n_samples * latent_dim]` flat vector.
pub fn encode_cohort<B: Backend>(
    model: &CrossCoder<B>,
    data: &[Vec<f32>],
    shapes: &[(usize, usize)],
    device: &B::Device,
) -> Vec<f32> {
    let n_samples = shapes[0].0;
    let input_dims: Vec<usize> = shapes.iter().map(|s| s.1).collect();
    let mut tensors = Vec::with_capacity(input_dims.len());
    for (v, &dim) in input_dims.iter().enumerate() {
        let t = Tensor::<B, 2>::from_data(
            TensorData::new::<f32, Vec<usize>>(data[v].clone(), vec![n_samples, dim]),
            device,
        );
        tensors.push(t);
    }

    let mus = model.encode_all(&tensors);
    let n_views = mus.len();
    let latent_dim = model.latent_dim;

    // Average across views (consensus latent)
    let mut consensus = vec![0.0f64; n_samples * latent_dim];
    let mut mu_data_all = Vec::with_capacity(n_views);
    for mu in mus {
        let flat = mu.into_data().as_slice::<f32>().unwrap().to_vec();
        for s in 0..n_samples {
            for d in 0..latent_dim {
                consensus[s * latent_dim + d] += flat[s * latent_dim + d] as f64;
            }
        }
        mu_data_all.push(flat);
    }
    for v in consensus.iter_mut() {
        *v /= n_views as f64;
    }

    consensus.iter().map(|v| *v as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use crate::sbi::crosscoder::CrossCoder;

    type B = NdArray<f32>;

    #[test]
    fn test_fit_mvn_and_sample() {
        let n_samples = 100;
        let latent_dim = 3;
        // Synthetic latents: N(μ=[1,2,3], diag(0.1))
        let mut latents = Vec::with_capacity(n_samples * latent_dim);
        for _ in 0..n_samples {
            latents.push(1.0 + rand::random::<f32>() * 0.2 - 0.1);
            latents.push(2.0 + rand::random::<f32>() * 0.2 - 0.1);
            latents.push(3.0 + rand::random::<f32>() * 0.2 - 0.1);
        }

        let (mean, cov) = fit_mvn_over_latents(&latents, n_samples, latent_dim);
        assert_eq!(mean.len(), 3);
        assert_eq!(cov.len(), 9);

        let prior = MvnPrior::from_mean_cov(mean, cov, latent_dim);
        let samples = prior.sample(50, Some(42));
        assert_eq!(samples.len(), 50 * 3);
    }

    #[test]
    fn test_cholesky_identity() {
        // Cov = I
        let cov = vec![
            1.0f32, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let l = cholesky_decompose(&cov, 3);
        // L should be identity
        assert!((l[0] - 1.0).abs() < 1e-4);
        assert!((l[4] - 1.0).abs() < 1e-4);
        assert!((l[8] - 1.0).abs() < 1e-4);
        assert!(l[1].abs() < 1e-4);
        assert!(l[2].abs() < 1e-4);
        assert!(l[5].abs() < 1e-4);
    }

    #[test]
    fn test_encode_cohort_shape() {
        let device = Default::default();
        let cc = CrossCoder::<B>::new(&device, &[4, 6], 3, 1.0);
        let data_a = vec![0.0f32; 10 * 4];
        let data_b = vec![0.0f32; 10 * 6];
        let latents = encode_cohort(&cc, &[ data_a, data_b ],
            &[(10, 4), (10, 6)], &device);
        assert_eq!(latents.len(), 10 * 3);
    }
}
