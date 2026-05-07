use burn::{
    module::Module,
    nn::{Initializer, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor, TensorData},
};
use rand::seq::SliceRandom;

/// Masked Autoencoder for Distribution Estimation (MADE).
///
/// A single masked dense block used inside a MAF flow.  The binary masks
/// `mask1` and `mask2` enforce the autoregressive property so that
/// dimension *i* of the output only depends on dimensions < *i* of the
/// input.
#[derive(Module, Debug)]
pub struct MADE<B: Backend> {
    pub linear_y: Linear<B>,
    pub linear_c: Linear<B>,
    pub linear_out: Linear<B>,
    pub linear_out_c: Linear<B>,
    pub mask1: Tensor<B, 2>,
    pub mask2: Tensor<B, 2>,
    pub perm_matrix: Tensor<B, 2>,
    pub inv_perm_matrix: Tensor<B, 2>,
    pub param_dim: usize,
    pub feature_dim: usize,
    pub hidden_dim: usize,
}

impl<B: Backend> MADE<B> {
    /// Create a new MADE layer.
    pub fn new(
        device: &B::Device,
        param_dim: usize,
        feature_dim: usize,
        hidden_dim: usize,
    ) -> Self {
        let linear_y = LinearConfig::new(param_dim, hidden_dim)
            .with_bias(false)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.01,
            })
            .init(device);

        let linear_c = LinearConfig::new(feature_dim, hidden_dim)
            .with_bias(false)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.01,
            })
            .init(device);

        let linear_out = LinearConfig::new(hidden_dim, 2 * param_dim)
            .with_bias(true)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.01,
            })
            .init(device);

        let linear_out_c = LinearConfig::new(feature_dim, 2 * param_dim)
            .with_bias(false)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.01,
            })
            .init(device);

        let (mask1, mask2) = build_masks(param_dim, hidden_dim, device);

        let mut perm: Vec<usize> = (0..param_dim).collect();
        perm.shuffle(&mut rand::thread_rng());

        let mut inv_perm = vec![0; param_dim];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }

        let perm_matrix = build_perm_matrix(&perm, device);
        let inv_perm_matrix = build_perm_matrix(&inv_perm, device);

        Self {
            linear_y,
            linear_c,
            linear_out,
            linear_out_c,
            mask1,
            mask2,
            perm_matrix,
            inv_perm_matrix,
            param_dim,
            feature_dim,
            hidden_dim,
        }
    }

    /// Forward pass: given `y` (parameters) and `context` (features), return
    /// `(mu, alpha)` each with shape `[batch, param_dim]`.
    pub fn forward(
        &self,
        y: Tensor<B, 2>,
        context: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Hidden layer.
        let w1y_masked = self.linear_y.weight.val() * self.mask1.clone();
        let mut h1 = y.matmul(w1y_masked);

        if self.feature_dim > 0 {
            h1 = h1 + self.linear_c.forward(context.clone());
        }

        let h = h1.tanh();

        // Output layer.
        let w2_masked = self.linear_out.weight.val() * self.mask2.clone();
        let mut out = h.matmul(w2_masked);

        if self.feature_dim > 0 {
            out = out + self.linear_out_c.forward(context);
        }

        if let Some(ref b) = self.linear_out.bias {
            let bias = b.val().unsqueeze();
            out = out + bias;
        }

        // Split into mu and alpha.
        let mu = out.clone().narrow(1, 0, self.param_dim);
        let alpha = out
            .narrow(1, self.param_dim, self.param_dim)
            .clamp(-7.0, 7.0);

        (mu, alpha)
    }

    /// Permute the input dimensions.
    pub fn apply_perm(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.perm_matrix.clone())
    }

    /// Apply the inverse permutation.
    pub fn apply_inv_perm(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.inv_perm_matrix.clone())
    }
}

/// Build the two MADE masks.
fn build_masks<B: Backend>(
    param_dim: usize,
    hidden_dim: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 2>) {
    let m_in: Vec<usize> = (1..=param_dim).collect();
    let m_h: Vec<usize> = if param_dim > 1 {
        (0..hidden_dim)
            .map(|_| rand::random::<usize>() % (param_dim - 1) + 1)
            .collect()
    } else {
        vec![1; hidden_dim]
    };

    // mask1 for `linear_y` weight [D, H].
    let mut m1_data = vec![0.0f32; param_dim * hidden_dim];
    for (j, &m_h_j) in m_h.iter().enumerate().take(hidden_dim) {
        for (i, &m_in_i) in m_in.iter().enumerate().take(param_dim) {
            let idx = i * hidden_dim + j;
            if m_in_i <= m_h_j {
                m1_data[idx] = 1.0;
            }
        }
    }

    // mask2 for `linear_out` weight [H, 2*D].
    let mut m2_data = vec![0.0f32; hidden_dim * 2 * param_dim];
    for j in 0..hidden_dim {
        for d in 0..param_dim {
            let m_out = d + 1;
            let val = if m_h[j] < m_out {
                1.0
            } else {
                0.0
            };
            m2_data[j * 2 * param_dim + d] = val;
            m2_data[j * 2 * param_dim + param_dim + d] = val;
        }
    }

    let mask1 = Tensor::<B, 2>::from_data(
        TensorData::new::<f32, Vec<usize>>(m1_data, vec![param_dim, hidden_dim]),
        device,
    );
    let mask2 = Tensor::<B, 2>::from_data(
        TensorData::new::<f32, Vec<usize>>(m2_data, vec![hidden_dim, 2 * param_dim]),
        device,
    );

    (mask1, mask2)
}

/// Build a permutation matrix `P` such that `y @ P` permutes the columns.
fn build_perm_matrix<B: Backend>(
    perm: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let d = perm.len();
    let mut data = vec![0.0f32; d * d];
    for i in 0..d {
        data[perm[i] * d + i] = 1.0;
    }
    Tensor::<B, 2>::from_data(
        TensorData::new::<f32, Vec<usize>>(data, vec![d, d]),
        device,
    )
}
