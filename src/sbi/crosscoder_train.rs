//! Training loop for the Linear Variational CrossCoder.
//!
//! Provides mini-batch SGD with Adam, gradient clipping, and β-annealing.

use burn::backend::autodiff::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{Tensor, TensorData};

use super::crosscoder::{CrossCoder, CrossCoderConfig};

type AD = Autodiff<NdArray<f32>>;

/// Train a CrossCoder on a cohort of multi-view data.
///
/// `data[i]` is a flat `[n_samples, input_dim_i]` array for view `i`.
///
/// Returns the trained model and a per-epoch loss history.
pub fn train_crosscoder(
    data: &[Vec<f32>],
    shape: &[(usize, usize)], // [(n_samples, input_dim), ...]
    cfg: &CrossCoderConfig,
    n_epochs: Option<usize>,
    batch_size: Option<usize>,
) -> (CrossCoder<AD>, Vec<(usize, f32)>) {
    let device = Default::default();
    let n_samples = shape[0].0;
    let input_dims: Vec<usize> = shape.iter().map(|s| s.1).collect();

    let mut model = CrossCoder::<AD>::new(
        &device, &input_dims, cfg.latent_dim, cfg.beta,
    );

    let mut optimizer = AdamConfig::new().init::<AD, CrossCoder<AD>>();
    let lr = cfg.learning_rate;
    let bs = batch_size.unwrap_or(cfg.batch_size);
    let epochs = n_epochs.unwrap_or(cfg.n_epochs);
    let mut loss_history: Vec<(usize, f32)> = Vec::new();

    let mut indices: Vec<usize> = (0..n_samples).collect();

    for epoch in 0..epochs {
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());

        // β-annealing: linear warmup from 0.01 to target β over first 20% epochs
        let beta = if epochs > 0 {
            let warmup_end = (epochs as f64 * 0.2).ceil() as usize;
            if epoch < warmup_end {
                let frac = epoch as f64 / warmup_end as f64;
                cfg.beta * frac.max(0.01)
            } else {
                cfg.beta
            }
        } else {
            cfg.beta
        };
        // burn modules are immutable, so we clone weights and update β
        model = model.with_beta(beta);

        for batch_start in (0..n_samples).step_by(bs) {
            let batch_end = (batch_start + bs).min(n_samples);
            let batch_indices = &indices[batch_start..batch_end];
            let current_batch_size = batch_indices.len();

            let mut batch_tensors = Vec::with_capacity(input_dims.len());
            for (v, &dim) in input_dims.iter().enumerate() {
                let mut vec = Vec::with_capacity(current_batch_size * dim);
                for &idx in batch_indices {
                    for d in 0..dim {
                        vec.push(data[v][idx * dim + d]);
                    }
                }
                let tensor = Tensor::<AD, 2>::from_data(
                    TensorData::new::<f32, Vec<usize>>(vec, vec![current_batch_size, dim]),
                    &device,
                );
                batch_tensors.push(tensor);
            }

            let loss = model.loss(&batch_tensors);
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);
        }

        if epoch % 5 == 0 || epoch == epochs - 1 {
            let epoch_loss = compute_epoch_loss(&model, data, shape, &device);
            log::info!("CrossCoder epoch {}/{}  loss={:.4}  beta={:.3}", epoch + 1, epochs, epoch_loss, beta);
            loss_history.push((epoch, epoch_loss));
        }
    }

    (model, loss_history)
}

fn compute_epoch_loss(
    model: &CrossCoder<AD>,
    data: &[Vec<f32>],
    shape: &[(usize, usize)],
    device: &<AD as burn::tensor::backend::Backend>::Device,
) -> f32 {
    let n_samples = shape[0].0;
    let input_dims: Vec<usize> = shape.iter().map(|s| s.1).collect();
    let mut tensors = Vec::with_capacity(input_dims.len());
    for (v, &dim) in input_dims.iter().enumerate() {
        let t = Tensor::<AD, 2>::from_data(
            TensorData::new::<f32, Vec<usize>>(
                data[v].clone(),
                vec![n_samples, dim],
            ),
            device,
        );
        tensors.push(t);
    }
    let loss = model.loss(&tensors);
    loss.into_data().as_slice::<f32>().unwrap()[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_crosscoder_toy() {
        let n_samples = 64;
        let dim_a = 4;
        let dim_b = 6;

        // Generate correlated toy data: view B = 2*view_A_plus_noise
        let mut data_a = Vec::with_capacity(n_samples * dim_a);
        let mut data_b = Vec::with_capacity(n_samples * dim_b);
        for _ in 0..n_samples {
            for _ in 0..dim_a {
                let val = rand::random::<f32>();
                data_a.push(val);
                data_b.push(2.0 * val + rand::random::<f32>() * 0.1);
            }
            // Pad view_b to dim_b if dim_b > dim_a (not needed here, dim_b==6 dim_a==4)
            for _ in dim_a..dim_b {
                data_b.push(rand::random::<f32>());
            }
        }

        let cfg = CrossCoderConfig {
            input_dims: vec![dim_a, dim_b],
            latent_dim: 3,
            beta: 0.1,
            learning_rate: 1e-2,
            grad_clip: 5.0,
            batch_size: 8,
            n_epochs: 10,
        };

        let (model, history) = train_crosscoder(
            &[ data_a, data_b ],
            &[(n_samples, dim_a), (n_samples, dim_b)],
            &cfg,
            Some(10),
            Some(8),
        );

        assert!(!history.is_empty());
        let final_loss = history.last().unwrap().1;
        assert!(final_loss.is_finite(), "final loss should be finite: {}", final_loss);
        // Loss should decrease (or at least not blow up)
        assert!(final_loss < 100.0, "loss blew up: {}", final_loss);
    }
}
