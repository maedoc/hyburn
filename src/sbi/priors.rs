//! Prior distribution types for SBI parameter sampling.

use serde::{Deserialize, Serialize};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

/// A single parameter's prior bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamPrior {
    pub name: String,
    pub min: f32,
    pub max: f32,
}

impl ParamPrior {
    pub fn new(name: impl Into<String>, min: f32, max: f32) -> Self {
        Self { name: name.into(), min, max }
    }

    pub fn mean(&self) -> f32 {
        (self.min + self.max) / 2.0
    }

    pub fn std(&self) -> f32 {
        // For uniform: std = (max - min) / sqrt(12)
        (self.max - self.min) / 12.0f32.sqrt()
    }

    pub fn variance(&self) -> f32 {
        let range = self.max - self.min;
        range * range / 12.0
    }
}

/// Prior distribution configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorDistribution {
    /// Uniform distribution between bounds per parameter.
    BoxUniform(Vec<ParamPrior>),
    /// Samples loaded from an NPY file: shape [n_samples, n_params].
    SamplesFromNpy { path: String },
    /// Multivariate normal: user provides mean vector and diagonal stds.
    MultivariateNormal {
        means: Vec<f32>,
        stds: Vec<f32>,
    },
}

impl PriorDistribution {
    /// Dimensionality of the parameter space.
    pub fn param_dim(&self) -> usize {
        match self {
            PriorDistribution::BoxUniform(priors) => priors.len(),
            PriorDistribution::SamplesFromNpy { .. } => {
                // We'll read this when sampling
                0
            }
            PriorDistribution::MultivariateNormal { means, .. } => means.len(),
        }
    }

    /// Sample `n` parameter vectors from the prior.
    /// Returns flat Vec<f32> of shape [n, param_dim] row-major.
    pub fn sample(&self, n: usize, seed: Option<u64>) -> anyhow::Result<(Vec<f32>, Vec<ParamPrior>)> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        match self {
            PriorDistribution::BoxUniform(priors) => {
                let mut samples = Vec::with_capacity(n * priors.len());
                for _ in 0..n {
                    for p in priors {
                        let dist = Uniform::new(p.min, p.max);
                        samples.push(dist.sample(&mut rng));
                    }
                }
                Ok((samples, priors.clone()))
            }
            PriorDistribution::SamplesFromNpy { path } => {
                let (data, shape) = crate::io::read_npy_f32(path)?;
                let n_file = shape[0];
                let param_dim = shape[1];
                let mut samples = Vec::with_capacity(n * param_dim);
                let priors: Vec<ParamPrior> = (0..param_dim)
                    .map(|i| ParamPrior::new(format!("param_{}", i), f32::NEG_INFINITY, f32::INFINITY))
                    .collect();
                if n > n_file {
                    anyhow::bail!("Requested {} samples but NPY file only has {}", n, n_file);
                }
                for i in 0..n {
                    let start = i * param_dim;
                    samples.extend_from_slice(&data[start..start + param_dim]);
                }
                Ok((samples, priors))
            }
            PriorDistribution::MultivariateNormal { means, stds } => {
                let mut samples = Vec::with_capacity(n * means.len());
                for _ in 0..n {
                    for (mean, std) in means.iter().zip(stds.iter()) {
                        let dist = Normal::new(*mean as f64, *std as f64)
                            .map_err(|e| anyhow::anyhow!("Invalid normal distribution: {}", e))?;
                        samples.push(dist.sample(&mut rng) as f32);
                    }
                }
                let priors: Vec<ParamPrior> = means.iter().zip(stds.iter())
                    .enumerate()
                    .map(|(i, (m, s))| ParamPrior::new(format!("param_{}", i), m - 3.0 * s, m + 3.0 * s))
                    .collect();
                Ok((samples, priors))
            }
        }
    }
}

/// Configuration for SBI priors, optionally loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorConfig {
    #[serde(flatten)]
    pub distribution: PriorDistribution,
    /// Random seed for reproducible sampling.
    #[serde(default)]
    pub seed: Option<u64>,
}

impl PriorConfig {
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let cfg: Self = toml::from_str(&content)?;
        Ok(cfg)
    }

    /// Dimensionality of the parameter space.
    pub fn param_dim(&self) -> usize {
        self.distribution.param_dim()
    }

    /// Sample `n` parameter vectors from the prior.
    pub fn sample(&self, n: usize) -> anyhow::Result<(Vec<f32>, Vec<ParamPrior>)> {
        self.distribution.sample(n, self.seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_uniform_stats() {
        let p = ParamPrior::new("test", 0.0, 1.0);
        assert!((p.mean() - 0.5).abs() < 1e-6);
        assert!((p.std() - (1.0f32 / 12.0f32.sqrt())).abs() < 1e-6);
    }

    #[test]
    fn test_sample_uniform() {
        let prior = PriorDistribution::BoxUniform(vec![
            ParamPrior::new("a", 0.0, 1.0),
            ParamPrior::new("b", -2.0, 2.0),
        ]);
        let (samples, priors) = prior.sample(100, Some(42)).unwrap();
        assert_eq!(samples.len(), 200);
        assert_eq!(priors.len(), 2);
        // Check bounds
        for i in 0..100 {
            assert!(samples[i * 2 + 0] >= 0.0 && samples[i * 2 + 0] <= 1.0);
            assert!(samples[i * 2 + 1] >= -2.0 && samples[i * 2 + 1] <= 2.0);
        }
    }

    #[test]
    fn test_mvn_sample() {
        let prior = PriorDistribution::MultivariateNormal {
            means: vec![0.0, 5.0],
            stds: vec![1.0, 0.5],
        };
        let (samples, _priors) = prior.sample(50, Some(123)).unwrap();
        assert_eq!(samples.len(), 100);
        // Just smoke test
    }
}

#[cfg(test)]
mod toml_format_tests {
    use super::*;

    #[test]
    fn test_prior_toml_boxuniform() {
        let toml = r#"
BoxUniform = [
    { name = "subnetworks[0].params[1]", min = -0.5, max = 0.5 },
    { name = "subnetworks[0].params[0]", min = 0.5, max = 2.0 },
]
seed = 42
"#;
        let cfg: PriorConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.seed, Some(42));
        match cfg.distribution {
            PriorDistribution::BoxUniform(p) => {
                assert_eq!(p.len(), 2);
                assert_eq!(p[0].name, "subnetworks[0].params[1]");
                assert_eq!(p[0].min, -0.5);
            }
            _ => panic!("Expected BoxUniform"),
        }
    }

    #[test]
    fn test_prior_toml_mvn() {
        let toml = r#"
MultivariateNormal = { means = [0.0, 5.0], stds = [1.0, 0.5] }
seed = 123
"#;
        let cfg: PriorConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.seed, Some(123));
        match cfg.distribution {
            PriorDistribution::MultivariateNormal { means, stds } => {
                assert_eq!(means, vec![0.0, 5.0]);
                assert_eq!(stds, vec![1.0, 0.5]);
            }
            _ => panic!("Expected MVN"),
        }
    }
}
