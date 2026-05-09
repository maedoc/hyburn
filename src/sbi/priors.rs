//! Prior distribution types for SBI parameter sampling.

use serde::{Deserialize, Serialize};
use rand::{Rng, SeedableRng};
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
        (self.max - self.min) / 12.0f32.sqrt()
    }

    pub fn variance(&self) -> f32 {
        let range = self.max - self.min;
        range * range / 12.0
    }
}

/// Sampling method for BoxUniform priors.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SamplingMethod {
    #[default]
    Lhs,
    Uniform,
    Sobol,
    Halton,
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
            PriorDistribution::SamplesFromNpy { .. } => 0,
            PriorDistribution::MultivariateNormal { means, .. } => means.len(),
        }
    }

    /// Sample `n` parameter vectors from the prior using the given method.
    /// Returns flat Vec<f32> of shape [n, param_dim] row-major.
    pub fn sample(&self, n: usize, seed: Option<u64>) -> anyhow::Result<(Vec<f32>, Vec<ParamPrior>)> {
        self.sample_with_method(n, seed, SamplingMethod::default())
    }

    /// Sample `n` parameter vectors from the prior using a specific method.
    pub fn sample_with_method(
        &self,
        n: usize,
        seed: Option<u64>,
        method: SamplingMethod,
    ) -> anyhow::Result<(Vec<f32>, Vec<ParamPrior>)> {
        let rng_seed = seed.unwrap_or_else(|| {
            use rand::RngCore;
            StdRng::from_entropy().next_u64()
        });

        match self {
            PriorDistribution::BoxUniform(priors) => {
                let ranges: Vec<(f32, f32)> = priors.iter().map(|p| (p.min, p.max)).collect();
                let n_dims = priors.len();

                let samples: Vec<Vec<f32>> = match method {
                    SamplingMethod::Uniform => uniform_samples(n, &ranges, rng_seed),
                    SamplingMethod::Lhs => latin_hypercube(n, n_dims, &ranges, rng_seed),
                    SamplingMethod::Halton => halton_samples(n, n_dims, &ranges, rng_seed),
                    SamplingMethod::Sobol => latin_hypercube(n, n_dims, &ranges, rng_seed),
                };

                let flat: Vec<f32> = samples.into_iter().flatten().collect();
                Ok((flat, priors.clone()))
            }
            PriorDistribution::SamplesFromNpy { path } => {
                #[cfg(not(target_arch = "wasm32"))]
                {
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
                #[cfg(target_arch = "wasm32")]
                {
                    let _ = path;
                    anyhow::bail!("SamplesFromNpy is not supported in WASM");
                }
            }
            PriorDistribution::MultivariateNormal { means, stds } => {
                let mut rng = StdRng::seed_from_u64(rng_seed);
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

/// Generate uniform random samples within bounds.
fn uniform_samples(n_samples: usize, ranges: &[(f32, f32)], seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_samples)
        .map(|_| {
            ranges
                .iter()
                .map(|&(lo, hi)| {
                    let dist = Uniform::new(lo, hi);
                    dist.sample(&mut rng)
                })
                .collect()
        })
        .collect()
}

/// Generate Latin Hypercube samples within bounds.
pub fn latin_hypercube(
    n_samples: usize,
    n_dims: usize,
    ranges: &[(f32, f32)],
    seed: u64,
) -> Vec<Vec<f32>> {
    use rand::seq::SliceRandom;

    let mut rng = StdRng::seed_from_u64(seed);
    let bin_width = 1.0 / n_samples as f32;

    let mut samples = vec![vec![0.0f32; n_dims]; n_samples];

    for d in 0..n_dims {
        let mut dim_samples: Vec<f32> = (0..n_samples)
            .map(|i| {
                let lo = i as f32 * bin_width;
                lo + rng.gen_range(0.0f32..bin_width)
            })
            .collect();
        dim_samples.shuffle(&mut rng);

        for i in 0..n_samples {
            let (lo, hi) = ranges[d];
            samples[i][d] = lo + dim_samples[i] * (hi - lo);
        }
    }

    samples
}

/// Generate Halton quasi-random samples within bounds.
pub fn halton_samples(
    n_samples: usize,
    n_dims: usize,
    ranges: &[(f32, f32)],
    seed: u64,
) -> Vec<Vec<f32>> {
    let primes: [usize; 20] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71];

    (0..n_samples)
        .map(|i| {
            (0..n_dims)
                .map(|d| {
                    let base = primes[d % primes.len()];
                    let u = halton(i + seed as usize + 1, base);
                    let (lo, hi) = ranges[d];
                    lo + u * (hi - lo)
                })
                .collect()
        })
        .collect()
}

fn halton(index: usize, base: usize) -> f32 {
    let mut result = 0.0f64;
    let mut f = 1.0 / base as f64;
    let mut i = index;
    while i > 0 {
        result += f * (i % base) as f64;
        i /= base;
        f /= base as f64;
    }
    result as f32
}

/// Resolve a parameter range from a model given a name like "subnetworks[0].params[1]".
/// Returns `None` if the name cannot be parsed or the model/param index is out of range.
pub fn resolve_param_range(name: &str, models: &[crate::engine::construction::EngineModel<burn::backend::ndarray::NdArray<f32>>]) -> Option<(f32, f32)> {
    let (sub_idx, param_idx) = parse_subnetwork_param(name)?;
    let model = models.get(sub_idx)?;
    let ranges = model.param_ranges();
    ranges.get(param_idx).copied()
}

/// Parse "subnetworks[K].params[P]" format into (K, P).
fn parse_subnetwork_param(name: &str) -> Option<(usize, usize)> {
    let name = name.trim();
    if !name.starts_with("subnetworks[") {
        return None;
    }
    let rest = &name["subnetworks[".len()..];
    let close_bracket = rest.find(']')?;
    let sub_idx: usize = rest[..close_bracket].parse().ok()?;
    let after = rest[close_bracket + 1..].trim();
    if !after.starts_with(".params[") {
        return None;
    }
    let inner = &after[".params[".len()..];
    let close2 = inner.find(']')?;
    let param_idx: usize = inner[..close2].parse().ok()?;
    Some((sub_idx, param_idx))
}

/// Configuration for SBI priors, optionally loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorConfig {
    #[serde(flatten)]
    pub distribution: PriorDistribution,
    /// Random seed for reproducible sampling.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Sampling method for BoxUniform priors (default: LHS).
    #[serde(default)]
    pub sampling: SamplingMethod,
}

impl PriorConfig {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let cfg: Self = toml::from_str(&content)?;
        Ok(cfg)
    }

    pub fn from_toml_str(s: &str) -> anyhow::Result<Self> {
        let cfg: Self = toml::from_str(s)?;
        Ok(cfg)
    }

    pub fn param_dim(&self) -> usize {
        self.distribution.param_dim()
    }

    pub fn sample(&self, n: usize) -> anyhow::Result<(Vec<f32>, Vec<ParamPrior>)> {
        self.distribution.sample_with_method(n, self.seed, self.sampling)
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
        for i in 0..100 {
            assert!(samples[i * 2 + 0] >= 0.0 && samples[i * 2 + 0] <= 1.0);
            assert!(samples[i * 2 + 1] >= -2.0 && samples[i * 2 + 1] <= 2.0);
        }
    }

    #[test]
    fn test_sample_lhs() {
        let prior = PriorDistribution::BoxUniform(vec![
            ParamPrior::new("a", 0.0, 1.0),
            ParamPrior::new("b", -2.0, 2.0),
        ]);
        let (samples, _) = prior.sample_with_method(50, Some(42), SamplingMethod::Lhs).unwrap();
        assert_eq!(samples.len(), 100);
        for i in 0..50 {
            assert!(samples[i * 2 + 0] >= 0.0 && samples[i * 2 + 0] <= 1.0);
            assert!(samples[i * 2 + 1] >= -2.0 && samples[i * 2 + 1] <= 2.0);
        }
    }

    #[test]
    fn test_lhs_stratification() {
        let n = 10;
        let ranges = vec![(0.0f32, 1.0f32), (-1.0f32, 1.0f32)];
        let samples = latin_hypercube(n, 2, &ranges, 42);
        assert_eq!(samples.len(), n);

        for d in 0..2 {
            let bin_width = 1.0 / n as f32;
            let mut bins = vec![false; n];
            for i in 0..n {
                let (lo, hi) = ranges[d];
                let unit = (samples[i][d] - lo) / (hi - lo);
                let bin = (unit / bin_width).floor() as usize;
                assert!(bin < n, "bin {} >= n {}", bin, n);
                bins[bin] = true;
            }
            assert!(bins.iter().all(|&b| b), "LHS dim {} not fully stratified", d);
        }
    }

    #[test]
    fn test_halton_samples() {
        let ranges = vec![(0.0f32, 1.0f32), (-2.0f32, 2.0f32)];
        let samples = halton_samples(100, 2, &ranges, 0);
        assert_eq!(samples.len(), 100);
        for s in &samples {
            assert!(s[0] >= 0.0 && s[0] <= 1.0, "halton dim0 out of bounds: {}", s[0]);
            assert!(s[1] >= -2.0 && s[1] <= 2.0, "halton dim1 out of bounds: {}", s[1]);
        }
    }

    #[test]
    fn test_halton_base2() {
        let h0 = halton(1, 2);
        assert!((h0 - 0.5).abs() < 1e-6, "halton(1,2) = {}, expected 0.5", h0);
        let h1 = halton(2, 2);
        assert!((h1 - 0.25).abs() < 1e-6, "halton(2,2) = {}, expected 0.25", h1);
        let h2 = halton(3, 2);
        assert!((h2 - 0.75).abs() < 1e-6, "halton(3,2) = {}, expected 0.75", h2);
    }

    #[test]
    fn test_mvn_sample() {
        let prior = PriorDistribution::MultivariateNormal {
            means: vec![0.0, 5.0],
            stds: vec![1.0, 0.5],
        };
        let (samples, _priors) = prior.sample(50, Some(123)).unwrap();
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_sampling_method_default() {
        assert_eq!(SamplingMethod::default(), SamplingMethod::Lhs);
    }

    #[test]
    fn test_parse_subnetwork_param() {
        assert_eq!(parse_subnetwork_param("subnetworks[0].params[1]"), Some((0, 1)));
        assert_eq!(parse_subnetwork_param("subnetworks[2].params[5]"), Some((2, 5)));
        assert_eq!(parse_subnetwork_param("bad_name"), None);
        assert_eq!(parse_subnetwork_param("subnetworks[0]"), None);
    }

    #[test]
    fn test_model_param_ranges_count() {
        use burn::backend::ndarray::NdArray;
        use crate::model::NeuralMassModel;
        use crate::model::g2do::Generic2dOscillator;
        use crate::model::mpr::MontbrioPazoRoxin;
        use crate::model::rww::ReducedWongWang;
        use crate::model::kuramoto_model::Kuramoto;
        use crate::model::jansen_rit::JansenRit;
        use crate::model::wilson_cowan::WilsonCowan;
        use crate::model::linear::Linear;
        use crate::model::sup_hopf::SupHopf;
        use crate::model::hopfield::Hopfield;
        use crate::model::coombes_byrne2d::CoombesByrne2D;
        use crate::model::coombes_byrne::CoombesByrne;
        use crate::model::gast_schmidt_knosche_sd::GastSchmidtKnoscheSD;
        use crate::model::gast_schmidt_knosche_sf::GastSchmidtKnoscheSF;
        use crate::model::larter_breakspear::LarterBreakspear;
        use crate::model::epileptor2d::Epileptor2D;
        use crate::model::epileptor::Epileptor;
        use crate::model::rww_exc_inh::ReducedWongWangExcInh;
        use crate::model::deco_balanced_exc_inh::DecoBalancedExcInh;
        use crate::model::epileptor_codim3::EpileptorCodim3;
        use crate::model::epileptor_codim3_slowmod::EpileptorCodim3SlowMod;
        use crate::model::epileptor_rs::EpileptorRestingState;
        use crate::model::zetterberg_jansen::ZetterbergJansen;
        use crate::model::reduced_fhn::ReducedSetFitzHughNagumo;
        use crate::model::reduced_hr::ReducedSetHindmarshRose;
        use crate::model::dumont_gutkin::DumontGutkin;
        use crate::model::zerlaut_first::ZerlautAdaptationFirstOrder;
        use crate::model::zerlaut_second::ZerlautAdaptationSecondOrder;
        use crate::model::kionex::KIonEx;

        type B = NdArray<f32>;

        assert_eq!(<Generic2dOscillator as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <Generic2dOscillator as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<MontbrioPazoRoxin as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <MontbrioPazoRoxin as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<ReducedWongWang as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <ReducedWongWang as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<Kuramoto as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <Kuramoto as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<JansenRit as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <JansenRit as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<WilsonCowan as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <WilsonCowan as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<Linear as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <Linear as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<SupHopf as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <SupHopf as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<Hopfield as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <Hopfield as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<CoombesByrne2D as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <CoombesByrne2D as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<CoombesByrne as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <CoombesByrne as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<GastSchmidtKnoscheSD as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <GastSchmidtKnoscheSD as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<GastSchmidtKnoscheSF as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <GastSchmidtKnoscheSF as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<LarterBreakspear as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <LarterBreakspear as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<Epileptor2D as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <Epileptor2D as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<Epileptor as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <Epileptor as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<ReducedWongWangExcInh as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <ReducedWongWangExcInh as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<DecoBalancedExcInh as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <DecoBalancedExcInh as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<EpileptorCodim3 as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <EpileptorCodim3 as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<EpileptorCodim3SlowMod as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <EpileptorCodim3SlowMod as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<EpileptorRestingState as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <EpileptorRestingState as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<ZetterbergJansen as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <ZetterbergJansen as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<ReducedSetFitzHughNagumo as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <ReducedSetFitzHughNagumo as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<ReducedSetHindmarshRose as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <ReducedSetHindmarshRose as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<DumontGutkin as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <DumontGutkin as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<ZerlautAdaptationFirstOrder as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <ZerlautAdaptationFirstOrder as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<ZerlautAdaptationSecondOrder as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <ZerlautAdaptationSecondOrder as NeuralMassModel<B>>::PARAM_NAMES.len());
        assert_eq!(<KIonEx as NeuralMassModel<B>>::PARAM_RANGES.len(),
                   <KIonEx as NeuralMassModel<B>>::PARAM_NAMES.len());
    }

    #[test]
    fn test_model_svar_ranges_count() {
        use burn::backend::ndarray::NdArray;
        use crate::model::NeuralMassModel;
        use crate::model::g2do::Generic2dOscillator;
        use crate::model::mpr::MontbrioPazoRoxin;
        use crate::model::rww::ReducedWongWang;
        use crate::model::kuramoto_model::Kuramoto;
        use crate::model::jansen_rit::JansenRit;
        use crate::model::wilson_cowan::WilsonCowan;
        use crate::model::linear::Linear;
        use crate::model::sup_hopf::SupHopf;
        use crate::model::hopfield::Hopfield;
        use crate::model::coombes_byrne2d::CoombesByrne2D;
        use crate::model::coombes_byrne::CoombesByrne;
        use crate::model::gast_schmidt_knosche_sd::GastSchmidtKnoscheSD;
        use crate::model::gast_schmidt_knosche_sf::GastSchmidtKnoscheSF;
        use crate::model::larter_breakspear::LarterBreakspear;
        use crate::model::epileptor2d::Epileptor2D;
        use crate::model::epileptor::Epileptor;
        use crate::model::rww_exc_inh::ReducedWongWangExcInh;
        use crate::model::deco_balanced_exc_inh::DecoBalancedExcInh;
        use crate::model::epileptor_codim3::EpileptorCodim3;
        use crate::model::epileptor_codim3_slowmod::EpileptorCodim3SlowMod;
        use crate::model::epileptor_rs::EpileptorRestingState;
        use crate::model::zetterberg_jansen::ZetterbergJansen;
        use crate::model::reduced_fhn::ReducedSetFitzHughNagumo;
        use crate::model::reduced_hr::ReducedSetHindmarshRose;
        use crate::model::dumont_gutkin::DumontGutkin;
        use crate::model::zerlaut_first::ZerlautAdaptationFirstOrder;
        use crate::model::zerlaut_second::ZerlautAdaptationSecondOrder;
        use crate::model::kionex::KIonEx;

        type B = NdArray<f32>;

        assert_eq!(<Generic2dOscillator as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <Generic2dOscillator as NeuralMassModel<B>>::NVAR);
        assert_eq!(<MontbrioPazoRoxin as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <MontbrioPazoRoxin as NeuralMassModel<B>>::NVAR);
        assert_eq!(<ReducedWongWang as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <ReducedWongWang as NeuralMassModel<B>>::NVAR);
        assert_eq!(<Kuramoto as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <Kuramoto as NeuralMassModel<B>>::NVAR);
        assert_eq!(<JansenRit as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <JansenRit as NeuralMassModel<B>>::NVAR);
        assert_eq!(<WilsonCowan as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <WilsonCowan as NeuralMassModel<B>>::NVAR);
        assert_eq!(<Linear as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <Linear as NeuralMassModel<B>>::NVAR);
        assert_eq!(<SupHopf as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <SupHopf as NeuralMassModel<B>>::NVAR);
        assert_eq!(<Hopfield as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <Hopfield as NeuralMassModel<B>>::NVAR);
        assert_eq!(<CoombesByrne2D as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <CoombesByrne2D as NeuralMassModel<B>>::NVAR);
        assert_eq!(<CoombesByrne as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <CoombesByrne as NeuralMassModel<B>>::NVAR);
        assert_eq!(<GastSchmidtKnoscheSD as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <GastSchmidtKnoscheSD as NeuralMassModel<B>>::NVAR);
        assert_eq!(<GastSchmidtKnoscheSF as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <GastSchmidtKnoscheSF as NeuralMassModel<B>>::NVAR);
        assert_eq!(<LarterBreakspear as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <LarterBreakspear as NeuralMassModel<B>>::NVAR);
        assert_eq!(<Epileptor2D as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <Epileptor2D as NeuralMassModel<B>>::NVAR);
        assert_eq!(<Epileptor as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <Epileptor as NeuralMassModel<B>>::NVAR);
        assert_eq!(<ReducedWongWangExcInh as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <ReducedWongWangExcInh as NeuralMassModel<B>>::NVAR);
        assert_eq!(<DecoBalancedExcInh as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <DecoBalancedExcInh as NeuralMassModel<B>>::NVAR);
        assert_eq!(<EpileptorCodim3 as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <EpileptorCodim3 as NeuralMassModel<B>>::NVAR);
        assert_eq!(<EpileptorCodim3SlowMod as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <EpileptorCodim3SlowMod as NeuralMassModel<B>>::NVAR);
        assert_eq!(<EpileptorRestingState as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <EpileptorRestingState as NeuralMassModel<B>>::NVAR);
        assert_eq!(<ZetterbergJansen as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <ZetterbergJansen as NeuralMassModel<B>>::NVAR);
        assert_eq!(<ReducedSetFitzHughNagumo as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <ReducedSetFitzHughNagumo as NeuralMassModel<B>>::NVAR);
        assert_eq!(<ReducedSetHindmarshRose as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <ReducedSetHindmarshRose as NeuralMassModel<B>>::NVAR);
        assert_eq!(<DumontGutkin as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <DumontGutkin as NeuralMassModel<B>>::NVAR);
        assert_eq!(<ZerlautAdaptationFirstOrder as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <ZerlautAdaptationFirstOrder as NeuralMassModel<B>>::NVAR);
        assert_eq!(<ZerlautAdaptationSecondOrder as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <ZerlautAdaptationSecondOrder as NeuralMassModel<B>>::NVAR);
        assert_eq!(<KIonEx as NeuralMassModel<B>>::SVAR_RANGES.len(),
                   <KIonEx as NeuralMassModel<B>>::NVAR);
    }

    #[test]
    fn test_engine_model_param_ranges_dispatch() {
        use burn::backend::ndarray::NdArray;
        use crate::engine::construction::EngineModel;
        use crate::model::g2do::g2do_default_params;

        type B = NdArray<f32>;

        let model: EngineModel<B> = EngineModel::G2do { params: g2do_default_params() };
        let ranges = model.param_ranges();
        assert_eq!(ranges.len(), 12);
        assert!((ranges[0].0 - 0.01).abs() < 1e-6);
        assert!((ranges[0].1 - 100.0).abs() < 1e-6);

        let sv = model.svar_ranges();
        assert_eq!(sv.len(), 2);
        let st = model.stvar();
        assert_eq!(st, &[0, 1]);
    }

    #[test]
    fn test_prior_config_sampling_field() {
        let toml = r#"
BoxUniform = [
    { name = "subnetworks[0].params[1]", min = -0.5, max = 0.5 },
]
seed = 42
sampling = "halton"
"#;
        let cfg: PriorConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.sampling, SamplingMethod::Halton);
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
