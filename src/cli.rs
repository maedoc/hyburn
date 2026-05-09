//! CLI definition using clap derive.

use clap::{Parser, Subcommand};
use crate::sbi::{PriorConfig, PriorDistribution, ParamPrior};

#[derive(Parser, Debug)]
#[command(name = "hyburn", version, about = "Burn-based GPU hybrid neural mass simulator")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run a simulation.
    Run {
        /// Path to TOML config file.
        #[arg(short, long, default_value = "sim.toml")]
        config: String,

        /// Output directory for results.
        #[arg(short, long, default_value = "output")]
        output: String,

        /// Backend: "ndarray" (CPU), "wgpu" (GPU), or "cuda" (NVIDIA GPU).
        #[arg(short, long, default_value = "ndarray")]
        backend: String,

        /// Checkpoint file path (optional). Saves state every N steps.
        #[arg(long)]
        checkpoint: Option<String>,

        /// Resume from a checkpoint file.
        #[arg(long)]
        resume: Option<String>,

        /// Progress reporting interval in steps (0 = disabled).
        #[arg(long, default_value = "0")]
        progress: usize,

        /// Path to sweep configuration TOML for parameter sweep mode.
        #[arg(long)]
        sweep: Option<String>,
    },

    /// Run benchmarks.
    Benchmark {
        /// Path to TOML config file.
        #[arg(short, long)]
        config: String,

        /// Number of warmup steps.
        #[arg(long, default_value = "10")]
        warmup: usize,

        /// Number of timed steps.
        #[arg(long, default_value = "100")]
        steps: usize,
    },

    /// Train an SBI model (MAF-based conditional density estimator).
    TrainSbi {
        /// Path to TOML config file.
        #[arg(short, long)]
        config: String,
    },

    /// Run inference with a trained SBI model.
    Infer {
        /// Path to saved model record.
        #[arg(short, long)]
        model: String,

        /// Path to feature data (.npy).
        #[arg(short, long)]
        features: String,

        /// Number of posterior samples.
        #[arg(short, long, default_value = "1000")]
        n_samples: usize,
    },

    /// Generate a self-contained HTML SBI diagnostic report.
    SbiReport {
        /// Path to TOML config file.
        #[arg(short, long, default_value = "sim.toml")]
        config: String,

        /// Output HTML path.
        #[arg(short, long, default_value = "sbi-report.html")]
        output: String,

        /// Compute backend (NdArray only; WGPU autodiff not yet available).
        #[arg(short, long, default_value = "ndarray")]
        backend: String,

        /// Number of sweep points.
        #[arg(long, default_value = "20")]
        n_sweep: usize,

        /// Number of simulation steps per sweep point.
        #[arg(long, default_value = "500")]
        steps: usize,

        /// Number of posterior samples per test point.
        #[arg(long, default_value = "100")]
        n_post_samples: usize,

        /// Training epochs.
        #[arg(long, default_value = "200")]
        epochs: usize,

        /// Training batch size.
        #[arg(long, default_value = "5")]
        batch_size: usize,

        /// Path to prior config TOML (optional).
        #[arg(long)]
        prior: Option<String>,
    },

    /// Run autotuning for optimal kernel parameters.
    Autotune {
        /// Path to TOML config file.
        #[arg(short, long)]
        config: String,
    },

    /// End-to-end pipeline: sweep → features → train → infer.
    Pipeline {
        /// Path to simulation config TOML.
        #[arg(short, long, default_value = "sim.toml")]
        config: String,

        /// Path to pipeline config TOML.
        #[arg(short, long, default_value = "pipeline.toml")]
        pipeline: String,

        /// Output directory for results.
        #[arg(short, long, default_value = "pipeline_output")]
        output: String,

        /// Backend: "ndarray", "wgpu", or "cuda".
        #[arg(short, long, default_value = "ndarray")]
        backend: String,
    },
}

impl Cli {
    /// Dispatch to the appropriate command handler.
    pub fn run(self) -> anyhow::Result<()> {
        match self.command {
            Command::Run {
                config,
                output,
                backend,
                checkpoint,
                resume,
                progress,
                sweep,
            } => run_cmd(
                &config,
                &output,
                &backend,
                checkpoint.as_deref(),
                resume.as_deref(),
                progress,
                sweep.as_deref(),
            ),
            Command::Benchmark { .. } => {
                anyhow::bail!(
                    "Benchmark subcommand is not yet implemented. \
                     Use one of the standalone benchmark binaries: \
                     `hyburn-bench-batch-sweep`, `hyburn-bench-generic-cuda`, \
                     `hyburn-validate-batch`, or `hyburn-validate-accuracy`. \
                     These are compiled as separate binaries in `src/bin/`."
                );
            }
            Command::TrainSbi { config } => {
                let cfg = crate::sbi::MafConfig::from_file(config.as_str())?;
                log::info!("Training MAF with config: {:?}", cfg);
                crate::sbi::train_maf(&cfg)?;
                Ok(())
            }
            Command::Infer {
                model,
                features,
                n_samples,
            } => {
                crate::sbi::infer_maf(model.as_str(), features.as_str(), n_samples)?;
                Ok(())
            }
            Command::SbiReport {
                config,
                output,
                backend,
                n_sweep,
                steps,
                n_post_samples,
                epochs,
                batch_size,
                prior,
            } => sbi_report_cmd(&config, &output, &backend, n_sweep, steps, n_post_samples, epochs, batch_size, prior.as_deref(),
            ),
            Command::Autotune { config } => autotune_cmd(&config),
            Command::Pipeline { config, pipeline, output, backend } => {
                pipeline_cmd(&config, &pipeline, &output, &backend)
            }
        }
    }
}

/// Normalise backend name, logging a warning and falling back to
/// `ndarray` if the requested backend is unavailable.
pub fn select_backend(backend: &str) -> &'static str {
    match backend {
        "ndarray" | "nd" | "cpu" => "ndarray",
        #[cfg(feature = "wgpu")]
        "wgpu" => "wgpu",
        #[cfg(feature = "cuda")]
        "cuda" => "cuda",
        _ => {
            log::warn!("Backend '{}' unavailable, falling back to NdArray", backend);
            "ndarray"
        }
    }
}

fn autotune_cmd(config: &str) -> anyhow::Result<()> {
    use crate::config::SimConfig;
    let cfg = SimConfig::from_file(config)?;
    cfg.validate()?;
    let nnodes = cfg
        .network
        .subnetworks
        .iter()
        .map(|s| s.nnodes)
        .max()
        .unwrap_or(0);
    if nnodes == 0 {
        anyhow::bail!("No subnetworks found for autotuning");
    }
    let result = crate::engine::autotune::autotune_coupling(nnodes);
    log::info!("Autotune complete for {} nodes: {:?}", nnodes, result);
    println!("AutotuneResult {{");
    println!("  optimal_strategy: {:?},", result.optimal_strategy);
    println!("  optimal_block_size: {},", result.optimal_block_size);
    println!("  benchmark_time_ns: {},", result.benchmark_time_ns);
    println!("}}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Pipeline command: sweep → features → train
// ---------------------------------------------------------------------------

/// Legacy pipeline config for backward compatibility.
#[derive(Debug, Clone, serde::Deserialize)]
struct LegacyPipelineConfig {
    pub sweep_param: String,
    pub n_sweep: usize,
    pub sweep_min: f32,
    pub sweep_max: f32,
    #[serde(default = "default_n_steps")]
    pub n_steps: usize,
    #[serde(default = "default_hidden")]
    pub hidden_units: usize,
    #[serde(default = "default_n_flows")]
    pub n_flows: usize,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
}

/// Pipeline configuration (TOML-based).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PipelineConfig {
    /// Prior distribution for parameters.
    pub prior: PriorConfig,
    /// Number of simulation steps per sweep point.
    #[serde(default = "default_n_steps")]
    pub n_steps: usize,
    /// MAF hidden layer size.
    #[serde(default = "default_hidden")]
    pub hidden_units: usize,
    /// Number of MAF flow layers.
    #[serde(default = "default_n_flows")]
    pub n_flows: usize,
    /// Training epochs.
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    /// Training batch size.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Learning rate.
    #[serde(default = "default_lr")]
    pub learning_rate: f64,
    /// Number of simulation samples for SBI.
    #[serde(default = "default_n_samples")]
    pub n_samples: usize,
}

fn default_n_steps() -> usize { 1000 }
fn default_hidden() -> usize { 64 }
fn default_n_flows() -> usize { 4 }
fn default_epochs() -> usize { 200 }
fn default_batch_size() -> usize { 32 }
fn default_lr() -> f64 { 1e-3 }
fn default_n_samples() -> usize { 100 }

impl PipelineConfig {
    /// Load from a TOML file. Supports both new PriorConfig-based and legacy sweep formats.
    pub fn from_file(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        match toml::from_str::<Self>(&content) {
            Ok(cfg) => Ok(cfg),
            Err(_) => {
                let legacy: LegacyPipelineConfig = toml::from_str(&content)
                    .map_err(|e| anyhow::anyhow!("Failed to parse pipeline config: {}", e))?;
                log::warn!("Legacy pipeline config detected (sweep_param). Converting to PriorConfig.");
                let prior = PriorDistribution::BoxUniform(vec![
                    ParamPrior::new(legacy.sweep_param.clone(), legacy.sweep_min, legacy.sweep_max)
                ]);
                Ok(PipelineConfig {
                    prior: PriorConfig { distribution: prior, seed: Some(42), sampling: Default::default() },
                    n_steps: legacy.n_steps,
                    hidden_units: legacy.hidden_units,
                    n_flows: legacy.n_flows,
                    epochs: legacy.epochs,
                    batch_size: legacy.batch_size,
                    learning_rate: legacy.learning_rate,
                    n_samples: legacy.n_sweep,
                })
            }
        }
    }
}

fn pipeline_cmd(
    config: &str,
    pipeline_config: &str,
    output: &str,
    backend: &str,
) -> anyhow::Result<()> {
    use crate::config::SimConfig;
    use crate::sbi::{MafConfig, extract_features_with, FeatureSet};

    let sim_cfg = SimConfig::from_file(config)?;
    sim_cfg.validate()?;
    let pipe_cfg = PipelineConfig::from_file(pipeline_config)?;
    let n_steps = pipe_cfg.n_steps;

    std::fs::create_dir_all(output)?;

    let backend_id = select_backend(backend);

    let (samples, priors) = pipe_cfg.prior.sample(pipe_cfg.n_samples)?;
    let param_dim = priors.len();
    let n_samples = samples.len() / param_dim;

    let _prior_means: Vec<f32> = priors.iter().map(|p| p.mean()).collect();
    let _prior_stds: Vec<f32> = priors.iter().map(|p| p.std()).collect();

    log::info!("Pipeline: {} samples × {} steps, param_dim={}, backend={}", n_samples, n_steps, param_dim, backend_id);

    let use_batch = param_dim == 1
        && matches!(pipe_cfg.prior.distribution, PriorDistribution::BoxUniform(_))
        && priors[0].name.starts_with("subnetworks[");

    if use_batch {
        let values = samples;
        let (sub_idx, param_idx) = parse_sweep_param(&priors[0].name, &sim_cfg)?;

        let sweep_result = match backend_id {
            "ndarray" => {
                use burn::backend::ndarray::NdArray;
                run_pipeline_sweep::<NdArray<f32>>(
                    sim_cfg, &values, sub_idx, param_idx, n_steps, Default::default(),
                )
            }
            #[cfg(feature = "wgpu")]
            "wgpu" => {
                use burn_wgpu::Wgpu;
                let device = burn_wgpu::WgpuDevice::default();
                run_pipeline_sweep::<Wgpu<f32, i32>>(
                    sim_cfg, &values, sub_idx, param_idx, n_steps, device,
                )
            }
            #[cfg(feature = "cuda")]
            "cuda" => {
                use burn_cuda::Cuda;
                let device = burn_cuda::CudaDevice::default();
                run_pipeline_sweep::<Cuda<f32, i32>>(
                    sim_cfg, &values, sub_idx, param_idx, n_steps, device,
                )
            }
            _ => unreachable!("select_backend normalises to a known backend"),
        }?;

        let result = sweep_result.result;
        let nvar = sweep_result.nvar;
        let nnodes = sweep_result.nnodes;

        log::info!("Sweep done: {:.1}ms ({:.2}ms/pt)", result.elapsed_ms, result.elapsed_ms / n_samples as f64);

        let trajectories = result.trajectories.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Pipeline requires trajectory recording — enable run_sweep_with_trajectory"))?;

        let traj = &trajectories[0];
        let points_per_step = nvar * nnodes;
        let sweep_stride = n_steps * points_per_step;

        let mut all_params = Vec::with_capacity(n_samples);
        let mut all_features = Vec::with_capacity(n_samples);
        for (i, &param_val) in values.iter().enumerate() {
            let start = i * sweep_stride;
            let traj_slice = &traj[start..start + sweep_stride];
            let features = extract_features_with(traj_slice, &[n_steps, nvar, nnodes, 1], &FeatureSet::Classic);
            all_params.push(param_val);
            all_features.extend(features);
        }

        let feature_dim = all_features.len().checked_div(n_samples).unwrap_or(0);
        log::info!("Extracted features: {} points, feature_dim={}", n_samples, feature_dim);

        let maf_config = MafConfig {
            param_dim: 1,
            feature_dim,
            hidden_units: pipe_cfg.hidden_units,
            n_flows: pipe_cfg.n_flows,
            learning_rate: pipe_cfg.learning_rate,
            feature_set: "classic".to_string(),
        };

        log::info!("Training MAF: {} epochs, batch_size={}", pipe_cfg.epochs, pipe_cfg.batch_size);
        let (_maf, loss_history) = train_maf_with_data_and_log_wrap(
            &maf_config,
            all_params,
            all_features,
            pipe_cfg.epochs,
            pipe_cfg.batch_size,
        );

        let final_loss = loss_history.last().map(|(_, l)| *l).unwrap_or(f32::NAN);
        log::info!("MAF training complete. Final loss: {:.4}", final_loss);

        let loss_path = format!("{}/loss_history.csv", output);
        std::fs::write(&loss_path, loss_history.iter()
            .map(|(epoch, loss)| format!("{},{}", epoch, loss))
            .collect::<Vec<String>>()
            .join("
"))?;

        for (i, tavg) in result.tavg.iter().enumerate() {
            let path = format!("{}/tavg_sub{}.npy", output, i);
            crate::io::write_npy_f32(&path, tavg, &[n_samples, nvar, nnodes])?;
        }

        log::info!("Pipeline complete. Results in {}", output);
        Ok(())
        } else {
            match backend_id {
                "ndarray" => {
                    use burn::backend::ndarray::NdArray;
                    let (all_params, all_features) = run_pipeline_prior_multi_par::<NdArray<f32>>(
                        sim_cfg, &samples, &priors, n_steps, Default::default(),
                    )?;
                    finish_pipeline_maf(all_params, all_features, n_samples, &pipe_cfg, output)
                }
                #[cfg(feature = "wgpu")]
                "wgpu" => {
                    use burn_wgpu::Wgpu;
                    let (all_params, all_features) = run_pipeline_prior_multi::<Wgpu<f32, i32>>(
                        sim_cfg, &samples, &priors, n_steps, burn_wgpu::WgpuDevice::default(),
                    )?;
                    finish_pipeline_maf(all_params, all_features, n_samples, &pipe_cfg, output)
                }
                #[cfg(feature = "cuda")]
                "cuda" => {
                    use burn_cuda::Cuda;
                    let (all_params, all_features) = run_pipeline_prior_multi::<Cuda<f32, i32>>(
                        sim_cfg, &samples, &priors, n_steps, burn_cuda::CudaDevice::default(),
                    )?;
                    finish_pipeline_maf(all_params, all_features, n_samples, &pipe_cfg, output)
                }
                _ => unreachable!("select_backend normalises to a known backend"),
            }
        }
}

fn finish_pipeline_maf(
    all_params: Vec<f32>,
    all_features: Vec<f32>,
    n_samples: usize,
    pipe_cfg: &PipelineConfig,
    output: &str,
) -> anyhow::Result<()> {
    use crate::sbi::MafConfig;

    let feature_dim = all_features.len().checked_div(n_samples).unwrap_or(0);
    let param_dim = all_params.len().checked_div(n_samples).unwrap_or(0);
    log::info!("Extracted features: {} points, feature_dim={}", n_samples, feature_dim);

    let maf_config = MafConfig {
        param_dim,
        feature_dim,
        hidden_units: pipe_cfg.hidden_units,
        n_flows: pipe_cfg.n_flows,
        learning_rate: pipe_cfg.learning_rate,
        feature_set: "classic".to_string(),
    };

    log::info!("Training MAF: {} epochs, batch_size={}", pipe_cfg.epochs, pipe_cfg.batch_size);
    let (_maf, loss_history) = train_maf_with_data_and_log_wrap(
        &maf_config,
        all_params,
        all_features,
        pipe_cfg.epochs,
        pipe_cfg.batch_size,
    );

    let final_loss = loss_history.last().map(|(_, l)| *l).unwrap_or(f32::NAN);
    log::info!("MAF training complete. Final loss: {:.4}", final_loss);

    let loss_path = format!("{}/loss_history.csv", output);
    std::fs::write(&loss_path, loss_history.iter()
        .map(|(epoch, loss)| format!("{},{}", epoch, loss))
        .collect::<Vec<String>>()
        .join("
"))?;

    log::info!("Pipeline complete. Results in {}", output);
    Ok(())
}

#[allow(dead_code)]
fn run_pipeline_prior_multi<B: burn::prelude::Backend>(
    sim_cfg: crate::config::SimConfig,
    samples: &[f32],
    priors: &[ParamPrior],
    n_steps: usize,
    device: B::Device,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    use crate::engine::HybridEngine;
    use crate::sbi::extract_features_with;
    use crate::sbi::FeatureSet;

    let param_dim = priors.len();
    let n_samples = samples.len() / param_dim;

    let (nvar, _, _) = crate::config::lookup_model(&sim_cfg.network.subnetworks[0].model)
        .unwrap_or((2, 1, 12));
    let nnodes = sim_cfg.network.subnetworks[0].nnodes;
    let nmodes = sim_cfg.network.subnetworks[0].nmodes;

    let per_sub_stride = nvar * nnodes * nmodes;
    let total_step_stride: usize = sim_cfg.network.subnetworks.iter()
        .map(|s| {
            let (nv, _, _) = crate::config::lookup_model(&s.model).unwrap_or((2, 1, 12));
            nv * s.nnodes * s.nmodes
        })
        .sum();

    let mut all_params = Vec::with_capacity(n_samples * param_dim);
    let mut all_features = Vec::new();

    for i in 0..n_samples {
        let mut cfg = sim_cfg.clone();
        for (j, p) in priors.iter().enumerate() {
            let val = samples[i * param_dim + j];
            apply_sweep_value(&mut cfg, &p.name, val)?;
        }

        let mut engine = HybridEngine::<B>::from_config(cfg, device.clone())?;
        engine.run(n_steps);

        let features = if sim_cfg.network.subnetworks.len() == 1 {
            extract_features_with(&engine.trajectory, &[n_steps, nvar, nnodes, nmodes], &FeatureSet::Classic)
        } else {
            let mut sub0_traj = Vec::with_capacity(n_steps * per_sub_stride);
            for t in 0..n_steps {
                let step_start = t * total_step_stride;
                sub0_traj.extend_from_slice(&engine.trajectory[step_start..step_start + per_sub_stride]);
            }
            extract_features_with(&sub0_traj, &[n_steps, nvar, nnodes, nmodes], &FeatureSet::Classic)
        };
        all_features.extend(features);

        for j in 0..param_dim {
            all_params.push(samples[i * param_dim + j]);
        }

        if (i + 1) % 10 == 0 || i + 1 == n_samples {
            log::info!("Sample {}/{} complete", i + 1, n_samples);
        }
    }

    Ok((all_params, all_features))
}

fn run_pipeline_prior_multi_par<B: burn::prelude::Backend>(
    sim_cfg: crate::config::SimConfig,
    samples: &[f32],
    priors: &[ParamPrior],
    n_steps: usize,
    device: B::Device,
) -> anyhow::Result<(Vec<f32>, Vec<f32>)>
where
    B::Device: Send + Sync,
{
    use crate::engine::HybridEngine;
    use crate::sbi::extract_features_with;
    use crate::sbi::FeatureSet;
    use rayon::prelude::*;

    let param_dim = priors.len();
    let n_samples = samples.len() / param_dim;

    let (nvar, _, _) = crate::config::lookup_model(&sim_cfg.network.subnetworks[0].model)
        .unwrap_or((2, 1, 12));
    let nnodes = sim_cfg.network.subnetworks[0].nnodes;
    let nmodes = sim_cfg.network.subnetworks[0].nmodes;

    let per_sub_stride = nvar * nnodes * nmodes;
    let total_step_stride: usize = sim_cfg.network.subnetworks.iter()
        .map(|s| {
            let (nv, _, _) = crate::config::lookup_model(&s.model).unwrap_or((2, 1, 12));
            nv * s.nnodes * s.nmodes
        })
        .sum();

    let results: Vec<(Vec<f32>, Vec<f32>)> = (0..n_samples)
        .into_par_iter()
        .map(|i| -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
            let mut cfg = sim_cfg.clone();
            for (j, p) in priors.iter().enumerate() {
                let val = samples[i * param_dim + j];
                apply_sweep_value(&mut cfg, &p.name, val)?;
            }

            let mut engine = HybridEngine::<B>::from_config(cfg, device.clone())?;
            engine.run(n_steps);

            let features = if sim_cfg.network.subnetworks.len() == 1 {
                extract_features_with(&engine.trajectory, &[n_steps, nvar, nnodes, nmodes], &FeatureSet::Classic)
            } else {
                let mut sub0_traj = Vec::with_capacity(n_steps * per_sub_stride);
                for t in 0..n_steps {
                    let step_start = t * total_step_stride;
                    sub0_traj.extend_from_slice(&engine.trajectory[step_start..step_start + per_sub_stride]);
                }
                extract_features_with(&sub0_traj, &[n_steps, nvar, nnodes, nmodes], &FeatureSet::Classic)
            };

            let mut sample_vec = Vec::with_capacity(param_dim);
            for j in 0..param_dim {
                sample_vec.push(samples[i * param_dim + j]);
            }

            Ok((sample_vec, features))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut all_params = Vec::with_capacity(n_samples * param_dim);
    let mut all_features = Vec::new();
    for (sample_vec, features) in results {
        all_params.extend(sample_vec);
        all_features.extend(features);
    }

    Ok((all_params, all_features))
}

/// Run batch sweep with trajectory recording, returning (result, nvar, nnodes).
struct PipelineSweepResult {
    result: crate::engine::batch_engine::BatchSweepResult,
    nvar: usize,
    nnodes: usize,
}

fn run_pipeline_sweep<B: burn::prelude::Backend>(
    cfg: crate::config::SimConfig,
    values: &[f32],
    sub_idx: usize,
    param_idx: usize,
    n_steps: usize,
    device: B::Device,
) -> anyhow::Result<PipelineSweepResult> {
    use crate::engine::batch_engine::{BatchHybridEngine, SweepParam};

    let n_sweep = values.len();
    let mut engine = BatchHybridEngine::<B>::from_config(cfg, n_sweep, device)
        .map_err(|e| anyhow::anyhow!("BatchHybridEngine init failed: {}", e))?;

    let nvar = engine.models[0].nvar();
    let nnodes = engine.states[0].shape().dims[1];

    let result = engine.run_sweep_with_trajectory(
        &SweepParam { sub_idx, param_idx },
        values,
        n_steps,
    );

    Ok(PipelineSweepResult { result, nvar, nnodes })
}

type AdBackend = burn::backend::autodiff::Autodiff<burn::backend::ndarray::NdArray<f32>>;

/// Wrapper around train_maf_with_data_and_log that uses simple flat params (1D).
fn train_maf_with_data_and_log_wrap(
    maf_config: &crate::sbi::MafConfig,
    flat_params: Vec<f32>,
    features: Vec<f32>,
    n_epochs: usize,
    batch_size: usize,
) -> (crate::sbi::MAF<AdBackend>, Vec<(usize, f32)>) {
    use crate::sbi::train_maf_with_data_and_log;
    train_maf_with_data_and_log(maf_config, flat_params, features, n_epochs, batch_size)
}

fn run_cmd(
    config: &str,
    output: &str,
    backend: &str,
    checkpoint: Option<&str>,
    resume: Option<&str>,
    progress_interval: usize,
    sweep: Option<&str>,
) -> anyhow::Result<()> {
    use crate::config::SimConfig;

    let cfg = SimConfig::from_file(config)?;

    if let Some(sweep_path) = sweep {
        run_sweep(
            cfg,
            output,
            backend,
            checkpoint,
            resume,
            progress_interval,
            sweep_path,
        )
    } else {
        dispatch_backend(backend, cfg, output, checkpoint, resume, progress_interval)
    }
}

fn dispatch_backend(
    backend: &str,
    cfg: crate::config::SimConfig,
    output: &str,
    checkpoint: Option<&str>,
    resume: Option<&str>,
    progress_interval: usize,
) -> anyhow::Result<()> {
    let backend = select_backend(backend);
    match backend {
        "ndarray" => {
            use burn::backend::ndarray::NdArray;
            run_simulation::<NdArray<f32>>(
                cfg,
                output,
                checkpoint,
                resume,
                progress_interval,
                Default::default(),
            )
        }
        #[cfg(feature = "wgpu")]
        "wgpu" => {
            use burn_wgpu::Wgpu;
            let device = burn_wgpu::WgpuDevice::default();
            run_simulation::<Wgpu<f32, i32>>(
                cfg,
                output,
                checkpoint,
                resume,
                progress_interval,
                device,
            )
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            use burn_cuda::Cuda;
            let device = burn_cuda::CudaDevice::default();
            run_simulation::<Cuda<f32, i32>>(
                cfg,
                output,
                checkpoint,
                resume,
                progress_interval,
                device,
            )
        }
        _ => unreachable!("select_backend normalises to a known backend"),
    }
}

fn run_sweep(
    cfg: crate::config::SimConfig,
    output: &str,
    backend: &str,
    _checkpoint: Option<&str>,
    _resume: Option<&str>,
    _progress_interval: usize,
    sweep_path: &str,
) -> anyhow::Result<()> {
    use crate::config::SweepConfig;

    if _checkpoint.is_some() {
        log::warn!("Checkpoint ignored in sweep mode");
    }
    if _resume.is_some() {
        log::warn!("Resume ignored in sweep mode");
    }

    let sweep_cfg = SweepConfig::from_file(sweep_path)?;
    let values = sweep_cfg.generate_values();

    if values.is_empty() {
        anyhow::bail!("Sweep config has no values");
    }

    // Parse sweep parameter name to (sub_idx, param_idx)
    let (sub_idx, param_idx) = parse_sweep_param(&sweep_cfg.parameter_name, &cfg)?;
    let n_steps = (cfg.sim_length / cfg.dt) as usize;
    let backend_id = select_backend(backend);

    match backend_id {
        "ndarray" => {
            use burn::backend::ndarray::NdArray;
            run_batch_sweep::<NdArray<f32>>(
                cfg, &values, sub_idx, param_idx, n_steps, output, Default::default(),
            )
        }
        #[cfg(feature = "wgpu")]
        "wgpu" => {
            use burn_wgpu::Wgpu;
            let device = burn_wgpu::WgpuDevice::default();
            run_batch_sweep::<Wgpu<f32, i32>>(
                cfg, &values, sub_idx, param_idx, n_steps, output, device,
            )
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            use burn_cuda::Cuda;
            let device = burn_cuda::CudaDevice::default();
            run_batch_sweep::<Cuda<f32, i32>>(
                cfg, &values, sub_idx, param_idx, n_steps, output, device,
            )
        }
        _ => unreachable!("select_backend normalises to a known backend"),
    }
}

/// Run a batch sweep using `BatchHybridEngine` and save results.
fn run_batch_sweep<B: burn::prelude::Backend>(
    cfg: crate::config::SimConfig,
    values: &[f32],
    sub_idx: usize,
    param_idx: usize,
    n_steps: usize,
    output: &str,
    device: B::Device,
) -> anyhow::Result<()> {
    use crate::engine::batch_engine::{BatchHybridEngine, SweepParam};
    use crate::io::write_npy_f32;

    std::fs::create_dir_all(output)?;

    let n_sweep = values.len();
    let mut engine = BatchHybridEngine::<B>::from_config(cfg, n_sweep, device)
        .map_err(|e| anyhow::anyhow!("BatchHybridEngine init failed: {}", e))?;
    let result = engine.run_sweep(
        &SweepParam { sub_idx, param_idx },
        values,
        n_steps,
    );

    log::info!(
        "Batch sweep complete: {} points × {} steps in {:.1} ms ({:.2} ms/point)",
        n_sweep, n_steps, result.elapsed_ms, result.elapsed_ms / n_sweep as f64,
    );

    // Save temporal averages per subnetwork
    for (i, tavg) in result.tavg.iter().enumerate() {
        let nnodes = engine.states.get(i).map(|s| s.shape().dims[1]).unwrap_or(0);
        let nvar = engine.models.get(i).map(|m| m.nvar()).unwrap_or(0);
        let n_sweep = result.n_sweep;
        write_npy_f32(
            format!("{}/tavg_sub{}.npy", output, i),
            tavg,
            &[n_sweep, nvar, nnodes],
        )?;
    }

    Ok(())
}

/// Parse sweep parameter name (e.g., "subnetworks[0].params[1]") to (sub_idx, param_idx).
fn parse_sweep_param(name: &str, cfg: &crate::config::SimConfig) -> anyhow::Result<(usize, usize)> {
    if name == "dt" || name == "nsig" {
        anyhow::bail!("Batch sweep does not support sweeping {} (model parameter sweeps only)", name);
    }
    if name.starts_with("subnetworks[") && name.contains("].params[") {
        let start = name.find('[').unwrap() + 1;
        let end = name.find(']').unwrap();
        let sub_idx: usize = name[start..end]
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid subnetwork index: {}", e))?;
        let pstart = name.rfind('[').unwrap() + 1;
        let pend = name.rfind(']').unwrap();
        let param_idx: usize = name[pstart..pend]
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid param index: {}", e))?;
        if sub_idx >= cfg.network.subnetworks.len() {
            anyhow::bail!("subnetwork index {} out of range", sub_idx);
        }
        Ok((sub_idx, param_idx))
    } else {
        anyhow::bail!("unsupported sweep parameter_name: {}. Use format: subnetworks[N].params[M]", name);
    }
}

/// Apply a sweep value to a SimConfig parameter.
fn apply_sweep_value(
    cfg: &mut crate::config::SimConfig,
    name: &str,
    value: f32,
) -> anyhow::Result<()> {
    if name == "dt" {
        cfg.dt = value as f64;
    } else if name == "nsig" {
        cfg.nsig = crate::config::NsigConfig::Scalar(value);
    } else if name.starts_with("subnetworks[") && name.contains("].params[") {
        let start = name.find('[').unwrap() + 1;
        let end = name.find(']').unwrap();
        let sub_idx: usize = name[start..end]
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid subnetwork index: {}", e))?;
        let pstart = name.rfind('[').unwrap() + 1;
        let pend = name.rfind(']').unwrap();
        let param_idx: usize = name[pstart..pend]
            .parse()
            .map_err(|e| anyhow::anyhow!("invalid param index: {}", e))?;
        if sub_idx >= cfg.network.subnetworks.len() {
            anyhow::bail!("subnetwork index {} out of range", sub_idx);
        }
        let params = &mut cfg.network.subnetworks[sub_idx].params;
        if param_idx >= params.len() {
            anyhow::bail!(
                "param index {} out of range for subnetwork {}",
                param_idx,
                sub_idx
            );
        }
        params[param_idx] = value;
    } else {
        anyhow::bail!("unsupported sweep parameter_name: {}", name);
    }
    Ok(())
}

fn run_simulation<B: burn::prelude::Backend>(
    cfg: crate::config::SimConfig,
    output: &str,
    checkpoint: Option<&str>,
    resume: Option<&str>,
    progress_interval: usize,
    device: B::Device,
) -> anyhow::Result<()> {
    use crate::engine::{HybridEngine, ProgressReporter};
    use crate::io::{tensor_to_flat_f32, write_npy_f32};

    cfg.validate()?;
    std::fs::create_dir_all(output)?;

    let mut engine = HybridEngine::<B>::from_config(cfg.clone(), device)?;

    if let Some(ckpt) = resume {
        log::info!("Resuming from checkpoint: {}", ckpt);
        engine.resume(ckpt)?;
    }

    if progress_interval > 0 {
        engine.progress = Some(ProgressReporter::new(progress_interval));
    }

    let n_steps = (cfg.sim_length / cfg.dt) as usize;
    engine.run(n_steps);

    if let Some(ckpt) = checkpoint {
        engine.checkpoint(ckpt)?;
    }

    for (i, (_sub, state)) in engine.subnetworks.iter().zip(&engine.states).enumerate() {
        let (final_data, final_shape) = tensor_to_flat_f32(state.clone());
        write_npy_f32(
            format!("{}/state_final_sub{}.npy", output, i),
            &final_data,
            &final_shape,
        )?;
    }

    if !engine.trajectory.is_empty() {
        if engine.subnetworks.len() == 1 {
            let sub = &engine.subnetworks[0];
            let traj_shape = vec![n_steps, sub.nvar, sub.nnodes, sub.nmodes];
            write_npy_f32(
                format!("{}/state_traj.npy", output),
                &engine.trajectory,
                &traj_shape,
            )?;
        } else {
            write_npy_f32(
                format!("{}/state_traj.npy", output),
                &engine.trajectory,
                &[engine.trajectory.len()],
            )?;
        }
    }

    log::info!("Simulation complete. {} steps written to {}", n_steps, output);
    Ok(())
}

/// Configuration for the SBI report command.
#[derive(Debug, Clone)]
pub struct SbiReportConfig<'a> {
    /// Path to the simulation configuration file.
    pub config: &'a str,
    /// Output directory for the report.
    pub output: &'a str,
    /// Compute backend name.
    pub backend: &'a str,
    /// Number of sweep points.
    pub n_sweep: usize,
    /// Number of simulation steps.
    pub steps: usize,
    /// Number of posterior samples.
    pub n_post_samples: usize,
    /// Number of training epochs.
    pub epochs: usize,
    /// Training batch size.
    pub batch_size: usize,
    /// Optional path to a prior configuration file.
    pub prior_path: Option<&'a str>,
}

fn sbi_report_cmd_with_config(config: SbiReportConfig) -> anyhow::Result<()> {
    use crate::report::{ReportConfig, generate_report};

    let priors = if let Some(path) = config.prior_path {
        let prior_cfg = PriorConfig::from_file(path)?;
        match prior_cfg.distribution {
            PriorDistribution::BoxUniform(p) => p,
            PriorDistribution::MultivariateNormal { means, stds } => {
                means.iter().zip(stds.iter())
                    .enumerate()
                    .map(|(i, (m, s))| ParamPrior::new(format!("param_{}", i), m - 3.0 * s, m + 3.0 * s))
                    .collect()
            }
            PriorDistribution::SamplesFromNpy { path: npy_path } => {
                let (_, shape) = crate::io::read_npy_f32(&npy_path)?;
                let param_dim = shape[1];
                (0..param_dim)
                    .map(|i| ParamPrior::new(format!("param_{}", i), f32::NEG_INFINITY, f32::INFINITY))
                    .collect()
            }
        }
    } else {
        vec![ParamPrior::new("I_ext", -0.5, 0.5)]
    };

    let report_cfg = ReportConfig {
        config_path: config.config.to_string(),
        backend: select_backend(config.backend).to_string(),
        n_sweep: config.n_sweep,
        n_steps: config.steps,
        n_post_samples: config.n_post_samples,
        n_epochs: config.epochs,
        batch_size: config.batch_size,
        output_path: config.output.to_string(),
        priors,
        ..Default::default()
    };

    let html = generate_report(report_cfg)?;
    println!("SBI report written to {} ({:.1} KB)", config.output, html.len() as f32 / 1024.0);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sbi_report_cmd(
    config: &str,
    output: &str,
    backend: &str,
    n_sweep: usize,
    steps: usize,
    n_post_samples: usize,
    epochs: usize,
    batch_size: usize,
    prior_path: Option<&str>,
) -> anyhow::Result<()> {
    sbi_report_cmd_with_config(SbiReportConfig {
        config,
        output,
        backend,
        n_sweep,
        steps,
        n_post_samples,
        epochs,
        batch_size,
        prior_path,
    })
}

