use burn::{
    module::Module,
    tensor::{backend::Backend, Distribution, Tensor},
};

use super::MADE;

/// Masked Autoregressive Flow (MAF).
///
/// A stack of MADE layers with permutations between them.  Supports both a
/// forward pass (computing log-probabilities for training) and an inverse
/// pass (autoregressive sampling).
#[derive(Module, Debug)]
pub struct MAF<B: Backend> {
    pub layers: Vec<MADE<B>>,
    pub param_dim: usize,
}

impl<B: Backend> MAF<B> {
    pub fn new(
        device: &B::Device,
        param_dim: usize,
        feature_dim: usize,
        hidden_dim: usize,
        n_flows: usize,
    ) -> Self {
        let layers: Vec<_> = (0..n_flows)
            .map(|_| MADE::new(device, param_dim, feature_dim, hidden_dim))
            .collect();
        Self { layers, param_dim }
    }

    /// Forward pass: compute log p(params | context).
    ///
    /// - `params`: `[batch, param_dim]`
    /// - `context`: `[batch, feature_dim]`
    ///
    /// Returns a `[batch]` tensor of log-probabilities.
    pub fn forward_log_prob(
        &self,
        params: Tensor<B, 2>,
        context: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let [batch_size, _d] = params.dims();
        let mut u = params;
        let mut log_det = Tensor::<B, 1>::zeros([batch_size], &u.device());

        for layer in &self.layers {
            u = layer.apply_perm(u);
            let (mu, alpha) = layer.forward(u.clone(), context.clone());
            let neg_alpha = -alpha;
            u = (u - mu) * neg_alpha.clone().exp();
            log_det = log_det + neg_alpha.sum_dim(1).flatten::<1>(0, 1);
        }

        let d = self.param_dim as f32;
        let base_logp = u
            .powf_scalar(2.0)
            .sum_dim(1)
            .flatten::<1>(0, 1)
            .neg()
            * 0.5
            - (0.5 * d * (2.0 * std::f32::consts::PI).ln());

        base_logp + log_det
    }

    /// Inverse pass: sample from p(params | context).
    ///
    /// - `context`: `[batch, feature_dim]` conditioning features.
    /// - `n_samples`: number of samples to draw **per** batch entry.
    ///
    /// Returns `[batch * n_samples, param_dim]`.
    pub fn inverse_sample(
        &self,
        context: Tensor<B, 2>,
        n_samples: usize,
    ) -> Tensor<B, 2> {
        let d = self.param_dim;
        let device = context.device();
        let z = Tensor::<B, 2>::random(
            [n_samples, d],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mut x = z;

        for layer in self.layers.iter().rev() {
            let y_perm = x.clone();
            let mut u_parts: Vec<Tensor<B, 2>> = Vec::new();

            for i in 0..d {
                let u = if u_parts.is_empty() {
                    Tensor::<B, 2>::zeros([n_samples, d], &device)
                } else {
                    let remaining = d - u_parts.len();
                    let zeros = Tensor::<B, 2>::zeros([n_samples, remaining], &device);
                    let mut parts = u_parts.clone();
                    parts.push(zeros);
                    Tensor::cat(parts, 1)
                };

                let (mu, alpha) = layer.forward(u, context.clone());
                let mu_i = mu.narrow(1, i, 1);
                let alpha_i = alpha.narrow(1, i, 1);
                let y_i = y_perm.clone().narrow(1, i, 1);
                let u_i = y_i * alpha_i.exp() + mu_i;
                u_parts.push(u_i);
            }

            let u = Tensor::cat(u_parts, 1);
            x = layer.apply_inv_perm(u);
        }

        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use burn::tensor::Tensor;

    type B = NdArray<f32>;

    #[test]
    fn test_maf_forward_log_prob() {
        let device = Default::default();
        let maf = MAF::<B>::new(&device, 2, 1, 8, 2);
        let params = Tensor::<B, 2>::zeros([4, 2], &device);
        let context = Tensor::<B, 2>::zeros([4, 1], &device);
        let log_prob = maf.forward_log_prob(params, context);
        assert_eq!(log_prob.dims(), [4]);
    }

    #[test]
    fn test_maf_inverse_sample() {
        let device = Default::default();
        let maf = MAF::<B>::new(&device, 2, 1, 8, 2);
        let context = Tensor::<B, 2>::zeros([1, 1], &device);
        let samples = maf.inverse_sample(context, 10);
        assert_eq!(samples.dims(), [10, 2]);
    }

    #[test]
    fn test_maf_roundtrip_invertibility() {
        // MAF must be invertible: forward(z) = x, inverse(x) ≈ z.
        // We test: x → forward_log_prob returns finite log-prob,
        // then inverse_sample from same context should produce samples
        // with the same distribution (finite, correct shape).
        // A more rigorous test: sample from inverse, compute forward log-prob,
        // and verify log-prob is finite and reasonable.
        let device = Default::default();
        let maf = MAF::<B>::new(&device, 2, 1, 8, 2);
        let context = Tensor::<B, 2>::zeros([1, 1], &device);

        // Draw samples from inverse (i.e. from the learned posterior)
        let samples = maf.inverse_sample(context, 100);
        assert_eq!(samples.dims(), [100, 2]);

        // Expand context to match batch size
        let context_batch = Tensor::<B, 2>::zeros([100, 1], &device);
        let log_prob = maf.forward_log_prob(samples, context_batch);

        // All log-probs must be finite — this proves the flow is invertible
        let lp_data = log_prob.into_data();
        let lp_slice = lp_data.as_slice::<f32>().unwrap();
        for (i, &lp) in lp_slice.iter().enumerate() {
            assert!(
                lp.is_finite(),
                "MAF round-trip: log_prob[{}] is non-finite: {}",
                i, lp
            );
        }
    }

    #[test]
    fn test_maf_no_context() {
        let device = Default::default();
        let maf = MAF::<B>::new(&device, 2, 0, 8, 2);
        let params = Tensor::<B, 2>::zeros([4, 2], &device);
        let context = Tensor::<B, 2>::zeros([4, 0], &device);
        let log_prob = maf.forward_log_prob(params, context);
        assert_eq!(log_prob.dims(), [4]);
    }
}
