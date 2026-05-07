//! Simple in-memory dataset for SBI training.

/// A dataset holding (parameters, features) pairs as flat vectors.
pub struct SbiDataset {
    pub params: Vec<f32>,
    pub features: Vec<f32>,
    pub param_dim: usize,
    pub feature_dim: usize,
    pub n_samples: usize,
}

impl SbiDataset {
    pub fn new(params: Vec<f32>, features: Vec<f32>, param_dim: usize, feature_dim: usize) -> Self {
        assert_eq!(params.len() % param_dim, 0, "params length must be a multiple of param_dim");
        assert_eq!(features.len() % feature_dim, 0, "features length must be a multiple of feature_dim");
        let n_params = params.len() / param_dim;
        let n_features = features.len() / feature_dim;
        assert_eq!(n_params, n_features, "params and features must have same number of samples");
        Self {
            params,
            features,
            param_dim,
            feature_dim,
            n_samples: n_params,
        }
    }

    pub fn get_batch(&self, indices: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let batch_size = indices.len();
        let mut params_batch = Vec::with_capacity(batch_size * self.param_dim);
        let mut features_batch = Vec::with_capacity(batch_size * self.feature_dim);
        for &i in indices {
            let p_start = i * self.param_dim;
            let f_start = i * self.feature_dim;
            params_batch.extend_from_slice(&self.params[p_start..p_start + self.param_dim]);
            features_batch.extend_from_slice(&self.features[f_start..f_start + self.feature_dim]);
        }
        (params_batch, features_batch)
    }
}
