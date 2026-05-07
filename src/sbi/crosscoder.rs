//! Linear Variational Cross-Coder (CrossCoder) in Burn.
//!
//! A multi-view encoder/decoder where each view `i` has a separate linear
//! encoder that produces a shared latent representation via the
//! reparameterisation trick, and a separate linear decoder that reconstructs
//! every view from that latent code.
//!
//! The training objective is:
//! ```text
//! L = Σ_i Σ_j MSE(decode_j( z_i ), x_j) + β · KL( q(z|x_i) || N(0,I) )
//! ```
//! where `z_i = μ_i + σ_i · ε` and `ε ~ N(0,1)`.
//!
//! All layers are **strictly linear** (no non-linearity).

use burn::{
    module::Module,
    nn::{Initializer, Linear, LinearConfig},
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{backend::Backend, Tensor},
};

/// File extension for CrossCoder checkpoints.
pub const CROSSCODER_CKPT_EXT: &str = ".cc.bin";

/// Output of [`CrossCoder::forward`]: reconstructed views, means, and log-variances per view.
pub type CrossCoderOutput<B> = (Vec<Vec<Tensor<B, 2>>>, Vec<Tensor<B, 2>>, Vec<Tensor<B, 2>>);

/// Per-view encoder/decoder pair.
#[derive(Module, Debug)]
pub struct CrossCoderView<B: Backend> {
    /// Linear encoder: input_dim → 2*latent_dim (μ || logσ²)
    pub encoder: Linear<B>,
    /// Linear decoder: latent_dim → input_dim
    pub decoder: Linear<B>,
    /// Dimension of the input view
    pub input_dim: usize,
}

impl<B: Backend> CrossCoderView<B> {
    pub fn new(device: &B::Device, input_dim: usize, latent_dim: usize) -> Self {
        let encoder = LinearConfig::new(input_dim, 2 * latent_dim)
            .with_bias(true)
            .with_initializer(Initializer::XavierUniform { gain: 2.0f64.sqrt() })
            .init(device);

        let decoder = LinearConfig::new(latent_dim, input_dim)
            .with_bias(true)
            .with_initializer(Initializer::XavierUniform { gain: 2.0f64.sqrt() })
            .init(device);

        Self {
            encoder,
            decoder,
            input_dim,
        }
    }

    /// Encode a batch of inputs to (μ, logσ²).
    ///
    /// `x`: `[batch, input_dim]`
    /// Returns: μ `[batch, latent_dim]`, logσ² `[batch, latent_dim]`
    pub fn encode(&self, x: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let out = self.encoder.forward(x); // [batch, 2*latent_dim]
        let latent_dim = out.dims()[1] / 2;
        let mu = out.clone().narrow(1, 0, latent_dim); // first half
        let logvar = out.narrow(1, latent_dim, latent_dim); // second half
        (mu, logvar)
    }

    /// Decode a latent code back to input space.
    ///
    /// `z`: `[batch, latent_dim]`
    /// Returns: `[batch, input_dim]`
    pub fn decode(&self, z: Tensor<B, 2>) -> Tensor<B, 2> {
        self.decoder.forward(z)
    }

    /// Deterministic encoding (returns only μ, no sampling).
    ///
    /// `x`: `[batch, input_dim]`
    /// Returns: μ `[batch, latent_dim]`
    pub fn encode_deterministic(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let out = self.encoder.forward(x); // [batch, 2*latent_dim]
        let latent_dim = out.dims()[1] / 2;
        out.narrow(1, 0, latent_dim) // first half = μ
    }
}

/// Multi-view Linear Variational CrossCoder.
#[derive(Module, Debug)]
pub struct CrossCoder<B: Backend> {
    pub views: Vec<CrossCoderView<B>>,
    pub latent_dim: usize,
    pub beta: f64,
}

impl<B: Backend> CrossCoder<B> {
    /// Create a new CrossCoder.
    ///
    /// `input_dims[i]` is the dimension of view `i`.
    pub fn new(
        device: &B::Device,
        input_dims: &[usize],
        latent_dim: usize,
        beta: f64,
    ) -> Self {
        let views = input_dims
            .iter()
            .map(|&d| CrossCoderView::new(device, d, latent_dim))
            .collect();
        Self {
            views,
            latent_dim,
            beta,
        }
    }

    /// Forward pass: encode every view, sample latents, decode every view
    /// from every latent.
    ///
    /// `inputs[i]` is a `[batch, input_dim_i]` tensor for view `i`.
    ///
    /// Returns:
    /// - `reconstructed[i][j]`: view `j` reconstructed from latent of view `i`
    /// - `mus[i]`: μ_i `[batch, latent_dim]`
    /// - `logvars[i]`: logσ²_i `[batch, latent_dim]`
    pub fn forward(
        &self,
        inputs: &[Tensor<B, 2>],
    ) -> CrossCoderOutput<B> {
        let n_views = self.views.len();
        let batch_size = inputs[0].dims()[0];
        let device = inputs[0].device();

        let mut mus = Vec::with_capacity(n_views);
        let mut logvars = Vec::with_capacity(n_views);
        let mut latents = Vec::with_capacity(n_views);

        for (view, input_i) in self.views.iter().zip(inputs.iter()) {
            let (mu, logvar) = view.encode(input_i.clone());
            let std = logvar.clone().exp().powf_scalar(0.5f32);
            let eps = Tensor::<B, 2>::random(
                [batch_size, self.latent_dim],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            );
            let z = mu.clone() + std * eps;
            mus.push(mu);
            logvars.push(logvar);
            latents.push(z);
        }

        let mut reconstructed: Vec<Vec<Tensor<B, 2>>> = Vec::with_capacity(n_views);
        for latent in &latents {
            let mut row = Vec::with_capacity(n_views);
            for view in &self.views {
                let xhat = view.decode(latent.clone());
                row.push(xhat);
            }
            reconstructed.push(row);
        }

        (reconstructed, mus, logvars)
    }

    /// Deterministic encode-all: returns μ latents for every view.
    ///
    /// `inputs[i]`: `[batch, input_dim_i]` for view `i`.
    /// Returns: `mus[i]`: `[batch, latent_dim]`
    pub fn encode_all(
        &self,
        inputs: &[Tensor<B, 2>],
    ) -> Vec<Tensor<B, 2>> {
        let n_views = self.views.len();
        let mut mus = Vec::with_capacity(n_views);
        for (view, input_i) in self.views.iter().zip(inputs.iter()) {
            mus.push(view.encode_deterministic(input_i.clone()));
        }
        mus
    }

    /// Compute the full CrossCoder loss.
    ///
    /// `inputs[i]`: `[batch, input_dim_i]` for view `i`.
    ///
    /// Returns a scalar tensor containing the loss value.
    /// Return a copy of this model with a different β value.
    ///
    /// Weights are cloned; only the scalar hyperparameter changes.
    pub fn with_beta(&self, beta: f64) -> Self {
        Self {
            views: self.views.clone(),
            latent_dim: self.latent_dim,
            beta,
        }
    }

    pub fn loss(
        &self,
        inputs: &[Tensor<B, 2>],
    ) -> Tensor<B, 1> {
        let (reconstructed, mus, logvars) = self.forward(inputs);
        let _n_views = self.views.len();
        let mut recon_loss = Tensor::<B, 1>::zeros([1], &inputs[0].device());

        for row in &reconstructed {
            for (j, xhat) in row.iter().enumerate() {
                let diff = xhat.clone() - inputs[j].clone();
                let mse = diff.powf_scalar(2.0).mean();
                recon_loss = recon_loss + mse;
            }
        }

        let mut kl_loss = Tensor::<B, 1>::zeros([1], &inputs[0].device());
        for (mu, logvar) in mus.iter().zip(logvars.iter()) {
            let mu = mu.clone();
            let logvar = logvar.clone();
            // KL(q||N) = -0.5 * Σ (1 + logσ² - μ² - σ²)
            let sigma_sq = logvar.clone().exp();
            let term = (mu.clone().powf_scalar(2.0) + sigma_sq - logvar - 1.0).sum_dim(1);
            kl_loss = kl_loss + term.mean();
        }
        kl_loss = kl_loss * 0.5f32;

        recon_loss + kl_loss * self.beta as f32
    }

    /// Save full model weights to a binary checkpoint file.
    ///
    /// Uses Burn's `BinFileRecorder`. The file can be reloaded with
    /// [`load`](Self::load) into a freshly-constructed model of the same
    /// architecture.
    pub fn save(&self, path: &str) {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        self.clone()
            .save_file(path, &recorder)
            .expect("Failed to save CrossCoder checkpoint");
    }
}

/// Load a CrossCoder from a binary checkpoint file.
///
/// Constructs a fresh model with the given architecture and loads saved
/// encoder/decoder weights into it.
pub fn load_crosscoder<B: Backend>(
    path: &str,
    device: &B::Device,
    input_dims: &[usize],
    latent_dim: usize,
    beta: f64,
) -> CrossCoder<B> {
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model = CrossCoder::new(device, input_dims, latent_dim, beta);
    model
        .load_file(path, &recorder, device)
        .expect("Failed to load CrossCoder checkpoint file")
}

/// Configuration for a CrossCoder model and its training hyperparameters.
#[derive(Debug, Clone)]
pub struct CrossCoderConfig {
    pub input_dims: Vec<usize>,
    pub latent_dim: usize,
    pub beta: f64,
    pub learning_rate: f64,
    pub grad_clip: f64,
    pub batch_size: usize,
    pub n_epochs: usize,
}

impl Default for CrossCoderConfig {
    fn default() -> Self {
        Self {
            input_dims: vec![68, 68], // example: two 68-region connectome views
            latent_dim: 16,
            beta: 1.0,
            learning_rate: 1e-3,
            grad_clip: 5.0,
            batch_size: 32,
            n_epochs: 100,
        }
    }
}

impl CrossCoderConfig {
    /// Total number of trainable parameters (approximate).
    pub fn n_params(&self) -> usize {
        self.input_dims
            .iter()
            .map(|d| {
                // encoder: (d * 2*latent) + 2*latent biases
                let enc = d * 2 * self.latent_dim + 2 * self.latent_dim;
                // decoder: (latent * d) + d biases
                let dec = self.latent_dim * d + d;
                enc + dec
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_crosscoder_forward_shape() {
        let device = Default::default();
        let cc = CrossCoder::<B>::new(&device,
            &[4, 6], // two views with dims 4 and 6
            3,         // latent dim = 3
            1.0,
        );

        let x0 = Tensor::<B, 2>::zeros([8, 4], &device);
        let x1 = Tensor::<B, 2>::zeros([8, 6], &device);

        let (recon, mus, logvars) = cc.forward(&[ x0, x1 ]);

        // recon[i][j]: view j reconstructed from latent of view i
        assert_eq!(recon.len(), 2);
        assert_eq!(recon[0].len(), 2);
        assert_eq!(recon[0][0].dims(), [8, 4]); // decode view 0 from latent 0
        assert_eq!(recon[0][1].dims(), [8, 6]); // decode view 1 from latent 0
        assert_eq!(recon[1][0].dims(), [8, 4]); // decode view 0 from latent 1
        assert_eq!(recon[1][1].dims(), [8, 6]); // decode view 1 from latent 1

        assert_eq!(mus[0].dims(), [8, 3]);
        assert_eq!(logvars[0].dims(), [8, 3]);
        assert_eq!(mus[1].dims(), [8, 3]);
        assert_eq!(logvars[1].dims(), [8, 3]);
    }

    #[test]
    fn test_crosscoder_loss_finite() {
        let device = Default::default();
        let cc = CrossCoder::<B>::new(&device, &[4, 6], 3, 1.0);
        let x0 = Tensor::<B, 2>::zeros([8, 4], &device);
        let x1 = Tensor::<B, 2>::zeros([8, 6], &device);

        let loss = cc.loss(&[ x0, x1 ]);
        assert_eq!(loss.dims(), [1]);
        let val = loss.into_data().as_slice::<f32>().unwrap()[0];
        assert!(val.is_finite(), "loss should be finite, got {}", val);
    }

    #[test]
    fn test_crosscoder_config() {
        let cfg = CrossCoderConfig::default();
        assert_eq!(cfg.input_dims, vec![68, 68]);
        assert_eq!(cfg.latent_dim, 16);
        assert!(cfg.n_params() > 0);
    }

    #[test]
    fn test_crosscoder_save_load_roundtrip() {
        let device = Default::default();
        let cc = CrossCoder::<B>::new(&device, &[4, 6], 3, 1.0);

        let tmpdir = tempfile::tempdir().unwrap();
        let path = tmpdir.path().join("test.cc.bin").to_str().unwrap().to_string();

        cc.save(&path);
        assert!(std::path::Path::new(&path).exists(), "checkpoint file should exist");

        let cc2 = load_crosscoder::<B>(&path, &device, &[4, 6], 3, 1.0);

        // Verify architecture matches
        assert_eq!(cc2.views.len(), 2);
        assert_eq!(cc2.latent_dim, 3);
        assert_eq!(cc2.views[0].input_dim, 4);
        assert_eq!(cc2.views[1].input_dim, 6);
    }
}
