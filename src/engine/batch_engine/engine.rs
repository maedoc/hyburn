use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use std::time::Instant;

use crate::config::{InitialStateConfig, NetworkConfig, SimConfig, WeightsConfig};
use crate::engine::{EngineModel, IntegratorKind};
use crate::engine::parse_cvar_map;
use crate::error::SimulationError;

use super::dfun::{dfun_batch, clamp_batch, model_prefers_heun, model_param_slice};
use super::projection::{PrecomputedProjection, ProjectionWeightKind};

/// Result of a generic batch sweep.
#[derive(Debug, Clone)]
pub struct BatchSweepResult {
    /// Sweep parameter values.
    pub param_values: Vec<f32>,
    /// Temporal averages per subnetwork, shape [n_sweep, nvar, nnodes] each.
    pub tavg: Vec<Vec<f32>>,
    /// Per-subnetwork final states, shape [n_sweep, nvar, nnodes] each.
    pub final_states: Vec<Vec<f32>>,
    /// Optional per-subnetwork trajectories: flat [n_steps * n_sweep * nvar * nnodes].
    pub trajectories: Option<Vec<Vec<f32>>>,
    /// Optional BOLD data per monitor: each entry is [n_sweep, n_bold_samples, nnodes].
    pub bold: Option<Vec<Vec<f32>>>,
    /// Wall-clock time in milliseconds.
    pub elapsed_ms: f64,
    /// Number of sweep points.
    pub n_sweep: usize,
}

/// Configuration for which parameter to sweep.
#[derive(Debug, Clone)]
pub struct SweepParam {
    /// Subnetwork index (e.g., 0 for first subnetwork).
    pub sub_idx: usize,
    /// Parameter index within that subnetwork's params.
    pub param_idx: usize,
}

/// A batch-dim hybrid simulation engine that processes all sweep points in parallel.
///
/// Unlike [`crate::engine::HybridEngine`] which runs one point at a time,
/// `BatchHybridEngine` stacks all `[n_sweep]` parameter values into batched
/// tensors and processes the entire simulation on GPU without CPU sync per step.
///
/// Supports any `SimConfig` with:
/// - G2DO, JansenRit, WilsonCowan, Mpr, Kuramoto, Rww models (batch-native)
/// - Other models (per-point 2D fallback, slower)
/// - All-to-all scalar coupling (batch-native)
/// - Dense weight matrix coupling (batch-native via matmul)
/// - Delayed coupling (not yet supported)
pub struct BatchHybridEngine<B: Backend> {
    /// Model descriptors per subnetwork.
    pub models: Vec<EngineModel<B>>,
    /// Batched states: [n_sweep, nnodes, nvar] per subnetwork.
    pub states: Vec<Tensor<B, 3>>,
    /// History buffers for delayed coupling: [n_sweep, nnodes, nvar, horizon] per subnetwork.
    pub histories: Vec<Tensor<B, 4>>,
    /// Current integration step.
    pub step: usize,
    /// Number of sweep points.
    pub n_sweep: usize,
    /// Network configuration (for projection info).
    pub network: NetworkConfig,
    /// Integration time step.
    pub dt: f32,
    /// Integrator type (from config).
    pub integrator: IntegratorKind,
    /// When true, use Heun for G2DO-like oscillators and Euler for other models.
    /// This matches the Numba CUDA benchmark behavior.
    /// When false, use the config's integrator for all models uniformly.
    pub hybrid_integrator: bool,
    /// Noise amplitude per variable for stochastic integration.
    pub nsig_vec: Vec<f32>,
    /// Device for tensor allocation.
    pub device: B::Device,
    /// Pre-computed projections with cached weight tensors.
    pub(crate) precomputed_projections: Vec<PrecomputedProjection<B>>,
    /// BOLD monitors for neural input tracking.
    pub bold_monitors: Vec<crate::engine::bold_monitor::BoldMonitor>,
    /// Target subnetwork index for each BOLD monitor.
    pub bold_targets: Vec<usize>,
}

impl<B: Backend> BatchHybridEngine<B> {
    /// Create a batch engine from a SimConfig.
    ///
    /// `n_sweep` copies of the network are created, each with its own state.
    /// The `sweep_param` indicates which parameter varies across sweep points.
    pub fn from_config(
        base_config: SimConfig,
        n_sweep: usize,
        device: B::Device,
    ) -> crate::error::Result<Self> {
        let mut models = Vec::new();
        let mut states = Vec::new();

        for sub_cfg in &base_config.network.subnetworks {
            let model = EngineModel::<B>::from_config(&sub_cfg.model, sub_cfg.params.clone())?;

            // Initial state: [n_sweep, nnodes, nvar]
            let nvar = model.nvar();
            let nnodes = sub_cfg.nnodes;
            let nmodes = sub_cfg.nmodes;

            let initial_flat = match &sub_cfg.initial_state {
                InitialStateConfig::Inline(vals) => {
                    // Replicate the same initial state for all sweep points
                    let single: Vec<f32> = vals.to_vec();
                    let mut batch = Vec::with_capacity(n_sweep * single.len());
                    for _ in 0..n_sweep {
                        batch.extend_from_slice(&single);
                    }
                    batch
                }
                InitialStateConfig::NpyPath(path) => {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let (data, shape) = crate::io::read_npy_f32(path)?;
                        // NPY can be 2D [nvar, nnodes] or 3D [nvar, nnodes, nmodes]
                        // We need flat order: [nnodes*nmodes, nvar] in row-major
                        let single: Vec<f32> = if shape.len() == 2 {
                            // [nvar, nnodes] → transpose to [nnodes, nvar] → flatten
                            let nvar_s = shape[0];
                            let nnodes_s = shape[1];
                            let mut transposed = vec![0.0f32; nnodes_s * nvar_s];
                            for n in 0..nnodes_s {
                                for v in 0..nvar_s {
                                    transposed[n * nvar_s + v] = data[v * nnodes_s + n];
                                }
                            }
                            if nmodes > 1 {
                                // Replicate across modes
                                let mut replicated = Vec::with_capacity(nnodes * nmodes * nvar_s);
                                for _m in 0..nmodes {
                                    replicated.extend_from_slice(&transposed);
                                }
                                replicated
                            } else {
                                transposed
                            }
                        } else if shape.len() == 3 {
                            // [nvar, nnodes, nmodes] → [nnodes*nmodes, nvar] flatten
                            let nvar_s = shape[0];
                            let nnodes_s = shape[1];
                            let nmodes_s = shape[2];
                            let mut transposed = vec![0.0f32; nnodes_s * nmodes_s * nvar_s];
                            for m in 0..nmodes_s {
                                for n in 0..nnodes_s {
                                    for v in 0..nvar_s {
                                        transposed[(n * nmodes_s + m) * nvar_s + v] =
                                            data[(v * nnodes_s * n) * nmodes_s + m];
                                    }
                                }
                            }
                            transposed
                        } else {
                            return Err(SimulationError::InvalidConfig(format!(
                                "Initial state NPY has {} dims, expected 2 or 3",
                                shape.len()
                            )));
                        };
                        // Replicate across sweep points
                        let mut batch = Vec::with_capacity(n_sweep * single.len());
                        for _ in 0..n_sweep {
                            batch.extend_from_slice(&single);
                        }
                        batch
                    }
                    #[cfg(target_arch = "wasm32")]
                    {
                        let _ = path;
                        return Err(SimulationError::BackendError("NPY file loading not supported in WASM".into()));
                    }
                }
                InitialStateConfig::Memory { data, shape } => {
                    // In-memory initial state: replicate across sweep points
                    let single = data.clone();
                    let _ = shape; // shape is implied by data layout matching the engine's expectations
                    let mut batch = Vec::with_capacity(n_sweep * single.len());
                    for _ in 0..n_sweep {
                        batch.extend_from_slice(&single);
                    }
                    batch
                }
            };

            // Reshape from flat to [n_sweep, nnodes*nmodes, nvar]
            let state = Tensor::<B, 3>::from_floats(
                TensorData::new::<f32, Vec<usize>>(initial_flat, vec![n_sweep, nnodes * nmodes, nvar]),
                &device,
            );

            models.push(model);
            states.push(state);
        }

        let max_delay = base_config.network.projections.iter()
            .map(|p| p.delays.iter().copied().max().unwrap_or(0) as usize)
            .max()
            .unwrap_or(0);
        let horizon = if max_delay == 0 { 0 } else { max_delay + 1 };

        let mut histories = Vec::new();
        if horizon > 0 {
            for state in &states {
                let state_slice = state.clone().unsqueeze_dim::<4>(3); // [n_sweep, nnodes*nmodes, nvar, 1]
                let rest = Tensor::<B, 4>::zeros(
                    [state.shape().dims[0], state.shape().dims[1], state.shape().dims[2], horizon - 1],
                    &device,
                );
                let h = Tensor::cat(vec![state_slice, rest], 3);
                histories.push(h);
            }
        }

        let mut precomputed_projections: Vec<PrecomputedProjection<B>> = Vec::new();
        for proj in &base_config.network.projections {
            let src_nnodes = base_config.network.subnetworks[proj.src].nnodes
                * base_config.network.subnetworks[proj.src].nmodes;
            let tgt_nnodes = base_config.network.subnetworks[proj.tgt].nnodes
                * base_config.network.subnetworks[proj.tgt].nmodes;

            let weight_kind = match &proj.weights {
                WeightsConfig::Scalar(w) => {
                    let effective_weight = if let Some(a) = proj.coupling_params.first() {
                        *w * *a
                    } else {
                        *w
                    };
                    ProjectionWeightKind::Scalar { weight: effective_weight }
                }
                WeightsConfig::Dense(mat) => {
                    let flat: Vec<f32> = mat.iter().flatten().copied().collect();
                    let weights = Tensor::<B, 2>::from_floats(
                        TensorData::new::<f32, Vec<usize>>(flat, vec![tgt_nnodes, src_nnodes]),
                        &device,
                    );
                    let weights = if let Some(a) = proj.coupling_params.first() {
                        weights.mul_scalar(*a)
                    } else {
                        weights
                    };
                    let weights_t = weights.swap_dims(0, 1);
                    ProjectionWeightKind::Dense { weights: weights_t }
                }
                WeightsConfig::Csr { data, indices, indptr } => {
                    let csr_indices_usize: Vec<usize> =
                        indices.iter().map(|&i| i as usize).collect();
                    let csr_indptr_usize: Vec<usize> =
                        indptr.iter().map(|&i| i as usize).collect();
                    let mut dense_weights = vec![0.0f32; tgt_nnodes * src_nnodes];
                    for tgt in 0..tgt_nnodes {
                        let start = csr_indptr_usize[tgt];
                        let end = csr_indptr_usize[tgt + 1];
                        for idx in start..end {
                            let src = csr_indices_usize[idx];
                            dense_weights[tgt * src_nnodes + src] = data[idx];
                        }
                    }
                    let weights = Tensor::<B, 2>::from_floats(
                        TensorData::new::<f32, Vec<usize>>(
                            dense_weights,
                            vec![tgt_nnodes, src_nnodes],
                        ),
                        &device,
                    );
                    let weights = if let Some(a) = proj.coupling_params.first() {
                        weights.mul_scalar(*a)
                    } else {
                        weights
                    };
                    let weights_t = weights.swap_dims(0, 1);
                    ProjectionWeightKind::Csr { weights: weights_t }
                }
            };

            let cvar_map_parsed = parse_cvar_map(&proj.cvar_map);
            // Validate cvar_map against model ncvar
            let src_ncvar = models[proj.src].ncvar();
            let tgt_ncvar = models[proj.tgt].ncvar();
            for &(s, t) in &cvar_map_parsed {
                if s >= src_ncvar {
                    return Err(SimulationError::InvalidConfig(format!(
                        "Projection {} cvar_map src index {} >= src_ncvar {}",
                        precomputed_projections.len(), s, src_ncvar
                    )));
                }
                if t >= tgt_ncvar {
                    return Err(SimulationError::InvalidConfig(format!(
                        "Projection {} cvar_map tgt index {} >= tgt_ncvar {}",
                        precomputed_projections.len(), t, tgt_ncvar
                    )));
                }
            }

            precomputed_projections.push(PrecomputedProjection {
                src: proj.src,
                tgt: proj.tgt,
                delay: proj.delays.first().copied().unwrap_or(0),
                cvar_map: cvar_map_parsed,
                weight_kind,
            });
        }

        let mut bold_monitors = Vec::new();
        let mut bold_targets = Vec::new();
        for mon_cfg in &base_config.monitors {
            let mon_type = mon_cfg.monitor_type.to_ascii_lowercase();
            if mon_type == "bold" {
                let target = 0usize;
                if target >= models.len() {
                    continue;
                }
                let bold_period = mon_cfg.bold_period.unwrap_or_else(|| {
                    let period_ms = mon_cfg.period.unwrap_or(2000.0);
                    (period_ms / base_config.dt).max(1.0).round() as usize
                });
                let tr = mon_cfg.tr.unwrap_or(2.0);
                let nnodes = base_config.network.subnetworks[target].nnodes
                    * base_config.network.subnetworks[target].nmodes;
                if nnodes == 0 {
                    continue;
                }
                let bm = crate::engine::bold_monitor::BoldMonitor::new(
                    target,
                    nnodes,
                    bold_period,
                    tr,
                    base_config.dt,
                    None,
                );
                bold_monitors.push(bm);
                bold_targets.push(target);
            }
        }

        let first_nvar = models.first().map(|m| m.nvar()).unwrap_or(1);
        let nsig_vec = base_config.nsig.to_vec(first_nvar);

        Ok(Self {
            models,
            states,
            histories,
            step: 0,
            n_sweep,
            network: base_config.network,
            dt: base_config.dt as f32,
            integrator: base_config.integrator,
            hybrid_integrator: false,
            nsig_vec,
            device,
            precomputed_projections,
            bold_monitors,
            bold_targets,
        })
    }

    /// Run the simulation for `n_steps`, sweeping `param` across the batch.
    ///
    /// The sweep parameter is applied per-sweep-point before starting.
    /// All points share identical initial conditions except for the swept parameter.
    pub fn run_sweep(
        &mut self,
        param: &SweepParam,
        param_values: &[f32],
        n_steps: usize,
    ) -> BatchSweepResult {
        self.run_sweep_internal(param, param_values, n_steps, false)
    }

    /// Run the simulation and optionally record full per-step trajectories.
    pub fn run_sweep_with_trajectory(
        &mut self,
        param: &SweepParam,
        param_values: &[f32],
        n_steps: usize,
    ) -> BatchSweepResult {
        self.run_sweep_internal(param, param_values, n_steps, true)
    }

    fn run_sweep_internal(
        &mut self,
        param: &SweepParam,
        param_values: &[f32],
        n_steps: usize,
        record_trajectory: bool,
    ) -> BatchSweepResult {
        let start = Instant::now();

        // Initialize temporal average accumulators
        let mut tavg: Vec<Tensor<B, 3>> = self.states.iter().map(|s| {
            let shape = s.shape();
            Tensor::<B, 3>::zeros([shape.dims[0], shape.dims[1], shape.dims[2]], &self.device)
        }).collect();

        let inv_nsteps = 1.0f32 / n_steps as f32;
        let n_sweep = param_values.len();

        // Optional trajectory recording
        let mut trajectories_per_sub: Vec<Vec<f32>> = if record_trajectory {
            self.models.iter().map(|_| Vec::new()).collect()
        } else {
            vec![]
        };

        // Pre-allocate coupling zero tensors (avoid allocating inside the loop)
        let zero_couplings: Vec<Tensor<B, 3>> = self.states.iter().enumerate().map(|(i, s)| {
            let shape = s.shape();
            let ncvar = self.models[i].ncvar();
            Tensor::<B, 3>::zeros([shape.dims[0], shape.dims[1], ncvar], &self.device)
        }).collect();

        // Pre-allocate param slices outside the loop (params don't change per step)
        let all_params: Vec<Vec<f32>> = self.models.iter().map(|m| model_param_slice(m)).collect();

        // Per-sweep parameter tensor [n_sweep, 1, 1] for batch-native models
        let sweep_tensor = Tensor::<B, 3>::from_floats(
            TensorData::new::<f32, Vec<usize>>(param_values.to_vec(), vec![n_sweep, 1, 1]),
            &self.device,
        );

        // Pre-compute per-projection coupling info
        let projections = &self.precomputed_projections;

        for _t in 0..n_steps {
            // 0. Save current states into history buffers (ring buffer)
            for i in 0..self.models.len() {
                if let Some(h) = self.histories.get(i) {
                    let state_slice = self.states[i].clone().unsqueeze_dim::<4>(3);
                    let new_h = Tensor::cat(vec![h.clone(), state_slice], 3);
                    let horizon = h.shape().dims[3];
                    let new_len = new_h.shape().dims[3] - horizon;
                    let new_h = new_h.narrow(3, new_len, horizon);
                    self.histories[i] = new_h;
                }
            }

            // 1. Compute coupling for each target subnetwork
            let mut couplings: Vec<Option<Tensor<B, 3>>> = vec![None; self.models.len()];
            let mut cvar_cache: std::collections::HashMap<(usize, u32), Tensor<B, 3>> = Default::default();

            for proj in projections {
                let src_idx = proj.src;
                let tgt_idx = proj.tgt;
                let src_model = &self.models[src_idx];
                let src_state = &self.states[src_idx];
                let ncvar = src_model.ncvar();

                // Extract coupling variables: [n_sweep, nnodes, ncvar]
                let key = (src_idx, proj.delay);
                let cvar_state = match cvar_cache.get(&key) {
                    Some(cached) => cached.clone(),
                    None => {
                        let cvar = if proj.delay == 0 || self.step == 0 {
                            src_state.clone().narrow(2, 0, ncvar.min(src_model.nvar()))
                        } else {
                            let raw_delay = proj.delay as usize;
                            if raw_delay <= self.step {
                                let h = &self.histories[src_idx];
                                let horizon = h.shape().dims[3];
                                let slot = (self.step - raw_delay + horizon) % horizon;
                                let delayed = h.clone().narrow(3, slot, 1).squeeze::<3>(3);
                                delayed.narrow(2, 0, ncvar.min(src_model.nvar()))
                            } else {
                                Tensor::<B, 3>::zeros(
                                    [n_sweep, src_state.shape().dims[1], ncvar.min(src_model.nvar())],
                                    &self.device,
                                )
                            }
                        };
                        cvar_cache.insert(key, cvar.clone());
                        cvar
                    }
                };

                // Compute coupling based on pre-computed weight kind
                let coupled = match &proj.weight_kind {
                    ProjectionWeightKind::Scalar { weight } => {
                        // All-to-all: mean over nodes × weight, expanded for dfun
                        let mean = cvar_state.mean_dim(1); // [n_sweep, 1, ncvar]
                        mean
                            .mul_scalar(*weight)
                            .expand([n_sweep, src_state.shape().dims[1], ncvar.min(src_model.nvar())])
                    }
                    ProjectionWeightKind::Dense { weights } => {
                        // Dense weights: batch matmul across sweep dimension.
                        // weights stored transposed as [src_nnodes, tgt_nnodes].
                        let src_nnodes = weights.shape().dims[0];
                        let tgt_nnodes = weights.shape().dims[1];
                        let cvar_t = cvar_state.swap_dims(1, 2); // [n_sweep, ncvar, src_nnodes]
                        let weights_3d = weights
                            .clone()
                            .unsqueeze_dim::<3>(0)
                            .expand([n_sweep, src_nnodes, tgt_nnodes]);
                        let result = cvar_t.matmul(weights_3d); // [n_sweep, ncvar, tgt_nnodes]
                        result.swap_dims(1, 2) // [n_sweep, tgt_nnodes, ncvar]
                    }
                    ProjectionWeightKind::Csr { weights } => {
                        // CSR converted to dense during pre-computation.
                        // weights stored transposed as [src_nnodes, tgt_nnodes].
                        let src_nnodes = weights.shape().dims[0];
                        let tgt_nnodes = weights.shape().dims[1];
                        let cvar_t = cvar_state.swap_dims(1, 2); // [n_sweep, ncvar, src_nnodes]
                        let weights_3d = weights
                            .clone()
                            .unsqueeze_dim::<3>(0)
                            .expand([n_sweep, src_nnodes, tgt_nnodes]);
                        let result = cvar_t.matmul(weights_3d); // [n_sweep, ncvar, tgt_nnodes]
                        result.swap_dims(1, 2) // [n_sweep, tgt_nnodes, ncvar]
                    }
                };

                // Remap cvars to target using cvar_map
                let tgt_model = &self.models[tgt_idx];
                let tgt_ncvar = tgt_model.ncvar();
                let remapped = if tgt_ncvar == ncvar
                    && proj.cvar_map.len() == 1
                    && proj.cvar_map[0] == (0, 0)
                {
                    coupled
                } else {
                    // Scatter source cvars into target cvar layout per cvar_map
                    let src_ncvar = ncvar;
                    let n_batch = coupled.shape().dims[0];
                    let n_nodes = coupled.shape().dims[1];
                    let mut tgt_data = vec![0.0f32; n_batch * n_nodes * tgt_ncvar];
                    let src_data = crate::io::tensor_to_flat_f32(coupled.clone()).0;
                    for &(s, t) in &proj.cvar_map {
                        if s < src_ncvar && t < tgt_ncvar {
                            for b in 0..n_batch {
                                for n in 0..n_nodes {
                                    tgt_data[(b * n_nodes + n) * tgt_ncvar + t] +=
                                        src_data[(b * n_nodes + n) * src_ncvar + s];
                                }
                            }
                        }
                    }
                    Tensor::<B, 3>::from_floats(
                        TensorData::new::<f32, Vec<usize>>(tgt_data, vec![n_batch, n_nodes, tgt_ncvar]),
                        &self.device,
                    )
                };

                match &mut couplings[tgt_idx] {
                    Some(existing) => {
                        *existing = existing.clone() + remapped;
                    }
                    None => {
                        couplings[tgt_idx] = Some(remapped);
                    }
                }
            }

            // 2. Integrate each subnetwork
            for i in 0..self.models.len() {
                let model = &self.models[i];

                let coupling = match &couplings[i] {
                    Some(c) => c.clone(),
                    None => zero_couplings[i].clone(),
                };

                // Use pre-allocated params (sweep param already set at index)
                let params = &all_params[i];
                let sweep_param = if i == param.sub_idx {
                    Some((param.param_idx, &sweep_tensor))
                } else {
                    None
                };

                // dfun
                let deriv = dfun_batch(model, self.states[i].clone(), coupling.clone(), params, sweep_param);

                // Integrate: use Heun for oscillatory models (G2DO), Euler for others
                // when hybrid_integrator is enabled; otherwise use config's integrator uniformly
                let use_heun = if self.hybrid_integrator {
                    model_prefers_heun(model)
                } else {
                    matches!(self.integrator, IntegratorKind::Heun)
                };

                if use_heun {
                    let mut predictor = self.states[i].clone() + deriv.clone().mul_scalar(self.dt);
                    clamp_batch(model, &mut predictor);
                    let deriv2 = dfun_batch(model, predictor, coupling, params, sweep_param);
                    self.states[i] = self.states[i].clone() + (deriv + deriv2).mul_scalar(self.dt * 0.5);
                } else {
                    match self.integrator {
                        IntegratorKind::Euler => {
                            self.states[i] = self.states[i].clone() + deriv.mul_scalar(self.dt);
                        }
                        IntegratorKind::Rk4 => {
                            let k1 = deriv;
                            let mut k2_state = self.states[i].clone() + k1.clone().mul_scalar(self.dt / 2.0);
                            clamp_batch(model, &mut k2_state);
                            let k2 = dfun_batch(model, k2_state, coupling.clone(), params, sweep_param);
                            let mut k3_state = self.states[i].clone() + k2.clone().mul_scalar(self.dt / 2.0);
                            clamp_batch(model, &mut k3_state);
                            let k3 = dfun_batch(model, k3_state, coupling.clone(), params, sweep_param);
                            let mut k4_state = self.states[i].clone() + k3.clone().mul_scalar(self.dt);
                            clamp_batch(model, &mut k4_state);
                            let k4 = dfun_batch(model, k4_state, coupling, params, sweep_param);
                            self.states[i] = self.states[i].clone() + (k1 + k2.mul_scalar(2.0) + k3.mul_scalar(2.0) + k4).mul_scalar(self.dt / 6.0);
                        }
                        IntegratorKind::EulerStochastic => {
                            let dims = self.states[i].shape().dims;
                            let noise = crate::engine::integrator::generate_noise_per_var::<B>(
                                [dims[1], dims[2]], &self.nsig_vec, self.dt, &self.device,
                            );
                            // Expand noise from [nnodes*nmodes, nvar] to [n_sweep, nnodes*nmodes, nvar]
                            let noise_3d = noise.unsqueeze_dim::<3>(0).expand([dims[0], dims[1], dims[2]]);
                            self.states[i] = self.states[i].clone() + deriv.mul_scalar(self.dt) + noise_3d;
                        }
                        IntegratorKind::HeunStochastic => {
                            let dims = self.states[i].shape().dims;
                            let noise = crate::engine::integrator::generate_noise_per_var::<B>(
                                [dims[1], dims[2]], &self.nsig_vec, self.dt, &self.device,
                            );
                            let noise_3d = noise.unsqueeze_dim::<3>(0).expand([dims[0], dims[1], dims[2]]);
                            let mut predictor = self.states[i].clone() + deriv.clone().mul_scalar(self.dt) + noise_3d.clone();
                            clamp_batch(model, &mut predictor);
                            let deriv2 = dfun_batch(model, predictor, coupling, params, sweep_param);
                            self.states[i] = self.states[i].clone() + (deriv + deriv2).mul_scalar(self.dt * 0.5) + noise_3d;
                        }
                        IntegratorKind::Rk4Stochastic => {
                            let k1 = deriv;
                            let mut k2_state = self.states[i].clone() + k1.clone().mul_scalar(self.dt / 2.0);
                            clamp_batch(model, &mut k2_state);
                            let k2 = dfun_batch(model, k2_state, coupling.clone(), params, sweep_param);
                            let mut k3_state = self.states[i].clone() + k2.clone().mul_scalar(self.dt / 2.0);
                            clamp_batch(model, &mut k3_state);
                            let k3 = dfun_batch(model, k3_state, coupling.clone(), params, sweep_param);
                            let mut k4_state = self.states[i].clone() + k3.clone().mul_scalar(self.dt);
                            clamp_batch(model, &mut k4_state);
                            let k4 = dfun_batch(model, k4_state, coupling, params, sweep_param);
                            let deterministic = self.states[i].clone() + (k1 + k2.mul_scalar(2.0) + k3.mul_scalar(2.0) + k4).mul_scalar(self.dt / 6.0);
                            let dims = deterministic.shape().dims;
                            let noise = crate::engine::integrator::generate_noise_per_var::<B>(
                                [dims[1], dims[2]], &self.nsig_vec, self.dt, &self.device,
                            );
                            let noise_3d = noise.unsqueeze_dim::<3>(0).expand([dims[0], dims[1], dims[2]]);
                            self.states[i] = deterministic + noise_3d;
                        }
                        _ => {
                            // Heun is handled above in use_heun branch
                            self.states[i] = self.states[i].clone() + deriv.mul_scalar(self.dt);
                        }
                    }
                }

                clamp_batch(model, &mut self.states[i]);

                // Accumulate temporal average
                tavg[i] = tavg[i].clone() + self.states[i].clone();  // TODO: use in-place add when Burn supports it

                // Append to trajectory if recording
                if record_trajectory && i < trajectories_per_sub.len() {
                    let (flat, _) = crate::io::tensor_to_flat_f32::<B, 3>(self.states[i].clone());
                    trajectories_per_sub[i].extend_from_slice(&flat);
                }
            }

            // 3. Accumulate BOLD neural input
            for (mi, monitor) in self.bold_monitors.iter_mut().enumerate() {
                let target = self.bold_targets[mi];
                let state = &self.states[target];
                let nnodes = self.network.subnetworks[target].nnodes;
                let nmodes = self.network.subnetworks[target].nmodes;
                // State shape: [n_sweep, nnodes*nmodes, nvar]
                // Extract var0, average over modes and sweep points
                let var0 = state.clone().narrow(2, 0, 1) // [n_sweep, nnodes*nmodes, 1]
                    .reshape([self.n_sweep, nnodes, nmodes]) // [n_sweep, nnodes, nmodes]
                    .mean_dim(2) // [n_sweep, nnodes, 1]
                    .squeeze::<2>(2) // [n_sweep, nnodes]
                    .mean_dim(0) // [1, nnodes]
                    .squeeze::<1>(0); // [nnodes]
                let (flat, _shape) = crate::io::tensor_to_flat_f32::<B, 1>(var0);
                monitor.accumulate(&flat);
            }

            self.step += 1;
        }

        // Flush BOLD monitors and collect data
        let bold_data: Option<Vec<Vec<f32>>> = if self.bold_monitors.is_empty() {
            None
        } else {
            Some(self.bold_monitors.iter_mut().map(|m| m.flush()).collect())
        };

        // Average and collect results
        let mut tavg_results = Vec::new();
        let mut final_state_results = Vec::new();

        for (i, tavg_i) in tavg.iter().enumerate().take(self.models.len()) {
            let avg = tavg_i.clone().mul_scalar(inv_nsteps);
            let (tavg_data, _) = crate::io::tensor_to_flat_f32::<B, 3>(avg);
            let (state_data, _) = crate::io::tensor_to_flat_f32::<B, 3>(self.states[i].clone());
            tavg_results.push(tavg_data);
            final_state_results.push(state_data);
        }

        let elapsed = start.elapsed();

        let trajectories = if record_trajectory {
            Some(trajectories_per_sub)
        } else {
            None
        };

        BatchSweepResult {
            param_values: param_values.to_vec(),
            tavg: tavg_results,
            final_states: final_state_results,
            trajectories,
            bold: bold_data,
            elapsed_ms: elapsed.as_millis() as f64,
            n_sweep,
        }
    }
}

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Shard a large batch sweep across Rayon threads.
///
/// Splits `param_values` into `n_threads` shards (default: number of CPUs),
/// runs a `BatchHybridEngine` per shard in parallel, then concatenates results.
///
/// This lets you use the full generic `BatchHybridEngine` with NdArray while
/// still saturating CPU cores. On a GPU backend `B` it is usually better to
/// keep everything in one big batch because GPU kernel launches are already
/// parallel.
#[cfg(feature = "parallel")]
pub fn rayon_batch_sweep<B: Backend>(
    base_config: SimConfig,
    param: SweepParam,
    param_values: &[f32],
    n_steps: usize,
    n_threads: Option<usize>,
    device: B::Device,
) -> BatchSweepResult
where
    B::Device: Send + Clone,
{
    let n_threads = n_threads.unwrap_or_else(rayon::current_num_threads);
    let shard_size = param_values.len().div_ceil(n_threads);

    let start = std::time::Instant::now();

    let shard_results: Vec<BatchSweepResult> = (0..param_values.len())
        .step_by(shard_size)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|start_idx| {
            let end_idx = (start_idx + shard_size).min(param_values.len());
            let shard_values = &param_values[start_idx..end_idx];
            let mut engine = BatchHybridEngine::<B>::from_config(base_config.clone(), shard_values.len(), device.clone())
                .expect("from_config should succeed");
            let mut sweep_param = param.clone();
            sweep_param.sub_idx = param.sub_idx; // keep same sub_idx / param_idx
            engine.run_sweep(
                &sweep_param,
                shard_values,
                n_steps,
            )
        })
        .collect();

    // Concatenate results across shards
    let mut all_tavg: Vec<Vec<f32>> = Vec::new();
    let mut all_final: Vec<Vec<f32>> = Vec::new();
    let n_subs = shard_results[0].tavg.len();
    for sub in 0..n_subs {
        let mut tavg_sub = Vec::with_capacity(param_values.len() * shard_results[0].tavg[sub].len() / shard_results[0].n_sweep);
        let mut final_sub = Vec::with_capacity(param_values.len() * shard_results[0].final_states[sub].len() / shard_results[0].n_sweep);
        for shard in &shard_results {
            tavg_sub.extend_from_slice(&shard.tavg[sub]);
            final_sub.extend_from_slice(&shard.final_states[sub]);
        }
        all_tavg.push(tavg_sub);
        all_final.push(final_sub);
    }

    let mut all_params = Vec::with_capacity(param_values.len());
    for shard in &shard_results {
        all_params.extend_from_slice(&shard.param_values);
    }

    let elapsed_ms = start.elapsed().as_millis() as f64;
    BatchSweepResult {
        param_values: all_params,
        tavg: all_tavg,
        final_states: all_final,
        trajectories: None,
        bold: None,
        elapsed_ms,
        n_sweep: param_values.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_batch_engine_g2do_single_subnet() {
        let device: <B as Backend>::Device = Default::default();
        let config = SimConfig {
            sim_length: 100.0,
            dt: 0.1,
            network: NetworkConfig {
                subnetworks: vec![crate::config::SubnetworkConfig {
                    model: "Generic2dOscillator".to_string(),
                    nnodes: 4,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.0; 8]),
                    params: crate::model::g2do::g2do_default_params(),
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut engine = BatchHybridEngine::<B>::from_config(config, 3, device).unwrap();
        let result = engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 1 },
            &[-0.5f32, 0.0, 0.5],
            100,
        );

        assert_eq!(result.n_sweep, 3);
        assert!(result.tavg[0].iter().all(|x| x.is_finite()));
        assert!(result.final_states[0].iter().all(|x| x.is_finite()));

        // Verify that different sweep points produce different outputs
        let tavg_pt0 = &result.tavg[0][0..8];
        let tavg_pt1 = &result.tavg[0][8..16];
        let tavg_pt2 = &result.tavg[0][16..24];
        let diff_01: f32 = tavg_pt0.iter().zip(tavg_pt1.iter()).map(|(a, b)| (a - b).abs()).sum();
        let diff_12: f32 = tavg_pt1.iter().zip(tavg_pt2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff_01 > 1e-6, "sweep points 0 and 1 should differ, got diff={}", diff_01);
        assert!(diff_12 > 1e-6, "sweep points 1 and 2 should differ, got diff={}", diff_12);
    }

    #[test]
    fn test_batch_engine_g2do_sweep_actually_varies() {
        // Dedicated test: with I_ext = -0.5 vs 0.0 vs 0.5, tavg should be measurably different
        let device: <B as Backend>::Device = Default::default();
        let config = SimConfig {
            sim_length: 50.0,
            dt: 0.1,
            network: NetworkConfig {
                subnetworks: vec![crate::config::SubnetworkConfig {
                    model: "Generic2dOscillator".to_string(),
                    nnodes: 2,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.1, 0.1, 0.1, 0.1]),
                    params: crate::model::g2do::g2do_default_params(),
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Euler,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut engine = BatchHybridEngine::<B>::from_config(config, 2, device).unwrap();
        let result = engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 1 },
            &[-1.0f32, 1.0],
            50,
        );

        assert_eq!(result.n_sweep, 2);
        let nvar = 2;
        let nnodes = 2;
        let per_point = nnodes * nvar;
        let pt0 = &result.tavg[0][0..per_point];
        let pt1 = &result.tavg[0][per_point..2 * per_point];
        let diff: f32 = pt0.iter().zip(pt1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "sweep points should differ significantly, got diff={}", diff);
    }

    #[test]
    fn test_burn_batch_matmul_3d_2d() {
        let device = Default::default();
        let cvar_state = Tensor::<B, 3>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                    7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ],
                vec![2, 3, 2],
            ),
            &device,
        );
        let weights = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                vec![2, 3],
            ),
            &device,
        );
        let cvar_t = cvar_state.swap_dims(1, 2); // [2, 2, 3]
        let weights_t = weights.swap_dims(0, 1); // [3, 2]
        let weights_3d = weights_t.unsqueeze_dim::<3>(0).expand([2, 3, 2]); // [n_sweep, nsrc, ntgt]
        let result = cvar_t.matmul(weights_3d); // [2, 2, 2]
        let result = result.swap_dims(1, 2); // [2, 2, 2] = [n_sweep, ntgt, ncvar]
        let (flat, shape) = crate::io::tensor_to_flat_f32(result);
        assert_eq!(shape, vec![2, 2, 2]);
        // sweep 0: tgt0 gets from src0 -> [1,2]; tgt1 gets from src1 -> [3,4]
        assert!((flat[0] - 1.0).abs() < 1e-6);
        assert!((flat[1] - 2.0).abs() < 1e-6);
        assert!((flat[2] - 3.0).abs() < 1e-6);
        assert!((flat[3] - 4.0).abs() < 1e-6);
        // sweep 1: tgt0 gets from src0 -> [7,8]; tgt1 gets from src1 -> [9,10]
        assert!((flat[4] - 7.0).abs() < 1e-6);
        assert!((flat[5] - 8.0).abs() < 1e-6);
        assert!((flat[6] - 9.0).abs() < 1e-6);
        assert!((flat[7] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_burn_compact_coupling_broadcast() {
        let device = Default::default();

        // Compact coupling [n_sweep, 1, ncvar] — like scalar all-to-all without expand
        let coupling = Tensor::<B, 3>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![2.0f32, 3.0, 10.0, 20.0], vec![2, 1, 2]),
            &device,
        );

        // State narrow [n_sweep, nnodes, nvar]
        let v = Tensor::<B, 3>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0f32; 12], vec![2, 3, 2]),
            &device,
        );

        // c_0 = coupling.narrow(2, 0, 1) → [n_sweep, 1, 1]
        let c_0 = coupling.narrow(2, 0, 1);
        assert_eq!(c_0.shape().dims, [2, 1, 1]);

        // Addition: [2,3,2] + [2,1,1] should broadcast
        let result = v.clone() + c_0.clone();
        let (data, shape) = crate::io::tensor_to_flat_f32::<B, 3>(result);
        assert_eq!(shape, vec![2, 3, 2]);
        // sweep 0: all 6 elements should be 1.0 + 2.0 = 3.0
        for i in 0..6 {
            assert!((data[i] - 3.0).abs() < 1e-5, "point 0 elem {}: got {}", i, data[i]);
        }
        // sweep 1: all 6 elements should be 1.0 + 10.0 = 11.0
        for i in 6..12 {
            assert!((data[i] - 11.0).abs() < 1e-5, "point 1 elem {}: got {}", i, data[i]);
        }

        // Multiplication: [2,3,2] * [2,1,1] should broadcast
        let result2 = v * c_0;
        let (data2, _) = crate::io::tensor_to_flat_f32::<B, 3>(result2);
        assert!((data2[0] - 2.0).abs() < 1e-5);
        assert!((data2[6] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_burn_coupling_accumulator_broadcast() {
        let device = Default::default();

        // Test: compact + expanded coupling accumulator
        // Simulates first projection storing compact, second adding expanded
        let compact = Tensor::<B, 3>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 1, 2]),
            &device,
        ); // [n_sweep, 1, ncvar]
        let expanded = Tensor::<B, 3>::from_floats(
            TensorData::new::<f32, Vec<usize>>(vec![10.0f32; 12], vec![2, 3, 2]),
            &device,
        ); // [n_sweep, nnodes, ncvar]

        // compact + expanded should broadcast to [2, 3, 2]
        let sum = compact + expanded;
        let (data, shape) = crate::io::tensor_to_flat_f32::<B, 3>(sum);
        assert_eq!(shape, vec![2, 3, 2]);
        // sweep 0, node 0: [1+10, 2+10] = [11, 12]; node 1: [1+10, 2+10] = [11, 12]
        assert!((data[0] - 11.0).abs() < 1e-5, "got {}", data[0]);
        assert!((data[1] - 12.0).abs() < 1e-5, "got {}", data[1]);
        // sweep 1, node 0: [3+10, 4+10] = [13, 14]
        assert!((data[6] - 13.0).abs() < 1e-5, "got {}", data[6]);
        assert!((data[7] - 14.0).abs() < 1e-5, "got {}", data[7]);
    }

    #[test]
    fn test_batch_engine_dense_coupling_shape() {
        let device: <B as Backend>::Device = Default::default();
        let config = SimConfig {
            sim_length: 10.0,
            dt: 0.1,
            network: NetworkConfig {
                subnetworks: vec![
                    crate::config::SubnetworkConfig {
                        model: "Generic2dOscillator".to_string(),
                        nnodes: 2,
                        nmodes: 1,
                        initial_state: InitialStateConfig::Inline(vec![0.1, 0.1, 0.2, 0.2]),
                        params: crate::model::g2do::g2do_default_params(),
                    },
                    crate::config::SubnetworkConfig {
                        model: "Generic2dOscillator".to_string(),
                        nnodes: 2,
                        nmodes: 1,
                        initial_state: InitialStateConfig::Inline(vec![0.1, 0.1, 0.2, 0.2]),
                        params: crate::model::g2do::g2do_default_params(),
                    },
                ],
                projections: vec![crate::config::ProjectionConfig {
                    src: 0,
                    tgt: 1,
                    conn_type: "all_to_all".to_string(),
                    weights: WeightsConfig::Dense(vec![
                        vec![1.0, 0.0],
                        vec![0.0, 1.0],
                    ]),
                    delays: vec![0],
                    tract_lengths: vec![],
                    coupling_fn: "Linear".to_string(),
                    coupling_params: vec![1.0],
                    cvar_map: "0:0".to_string(),
                }],
            },
            integrator: IntegratorKind::Euler,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut engine = BatchHybridEngine::<B>::from_config(config, 2, device).unwrap();
        let result = engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 1 },
            &[0.0f32, 0.5],
            10,
        );
        assert_eq!(result.n_sweep, 2);
        assert!(result.tavg[0].iter().all(|x| x.is_finite()));
        assert!(result.tavg[1].iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_batch_vs_serial_g2do_single_subnet() {
        use crate::engine::HybridEngine;
        use crate::io::tensor_to_flat_f32;

        let device: <B as Backend>::Device = Default::default();
        let n_steps = 50usize;
        let dt = 0.1f64;
        let sweep_val = 0.2f32;

        let mut params = crate::model::g2do::g2do_default_params();
        params[1] = sweep_val;

        // Serial layout: [nvar=2, nnodes=2, nmodes=1]
        // flat order: v0_n0, v0_n1, v1_n0, v1_n1
        let initial_state_serial = vec![0.1f32, 0.2, 0.3, 0.4];
        // Batch layout: [n_sweep=1, nnodes=2, nvar=2]
        // flat order: n0_v0, n0_v1, n1_v0, n1_v1
        // To match serial, we transpose:
        let initial_state_batch = vec![0.1f32, 0.3, 0.2, 0.4];

        let mut config_serial = SimConfig {
            sim_length: n_steps as f64 * dt,
            dt,
            network: NetworkConfig {
                subnetworks: vec![crate::config::SubnetworkConfig {
                    model: "Generic2dOscillator".to_string(),
                    nnodes: 2,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(initial_state_serial.clone()),
                    params: params.clone(),
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut config_batch = config_serial.clone();
        config_batch.network.subnetworks[0].initial_state = InitialStateConfig::Inline(initial_state_batch);

        let mut batch_engine = BatchHybridEngine::<B>::from_config(config_batch, 1, device.clone()).unwrap();
        let _ = batch_engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 1 },
            &[sweep_val],
            n_steps,
        );
        let batch_state_tensor = batch_engine.states[0].clone();
        let batch_s0 = batch_state_tensor.narrow(0, 0, 1).squeeze::<2>(0);
        let (batch_flat, _) = tensor_to_flat_f32(batch_s0);

        let mut serial_engine = HybridEngine::<B>::from_config(config_serial, device).unwrap();
        serial_engine.run(n_steps);
        let serial_state_2d = serial_engine.states[0].clone().permute([1, 0, 2]).squeeze::<2>(2);
        let (serial_flat, _) = tensor_to_flat_f32(serial_state_2d);

        assert_eq!(batch_flat.len(), serial_flat.len());
        for (i, (b, s)) in batch_flat.iter().zip(serial_flat.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-4,
                "Mismatch at index {}: batch={}, serial={}",
                i, b, s
            );
        }
    }

    #[test]
    fn test_batch_vs_serial_wc_single_subnet() {
        use crate::engine::HybridEngine;
        use crate::io::tensor_to_flat_f32;

        let device: <B as Backend>::Device = Default::default();
        let n_steps = 30usize;
        let dt = 0.1f64;
        let sweep_val = 0.15f32;

        let mut params = crate::model::wilson_cowan::wilson_cowan_default_params();
        params[18] = sweep_val;

        let initial_state_serial = vec![0.1f32, 0.2, 0.3, 0.4];
        let initial_state_batch = vec![0.1f32, 0.3, 0.2, 0.4];

        let mut config_serial = SimConfig {
            sim_length: n_steps as f64 * dt,
            dt,
            network: NetworkConfig {
                subnetworks: vec![crate::config::SubnetworkConfig {
                    model: "WilsonCowan".to_string(),
                    nnodes: 2,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(initial_state_serial.clone()),
                    params: params.clone(),
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut config_batch = config_serial.clone();
        config_batch.network.subnetworks[0].initial_state = InitialStateConfig::Inline(initial_state_batch);

        let mut batch_engine = BatchHybridEngine::<B>::from_config(config_batch, 1, device.clone()).unwrap();
        let _ = batch_engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 18 },
            &[sweep_val],
            n_steps,
        );
        let batch_state_tensor = batch_engine.states[0].clone();
        let batch_s0 = batch_state_tensor.narrow(0, 0, 1).squeeze::<2>(0);
        let (batch_flat, _) = tensor_to_flat_f32(batch_s0);

        let mut serial_engine = HybridEngine::<B>::from_config(config_serial, device).unwrap();
        serial_engine.run(n_steps);
        let serial_state_2d = serial_engine.states[0].clone().permute([1, 0, 2]).squeeze::<2>(2);
        let (serial_flat, _) = tensor_to_flat_f32(serial_state_2d);

        assert_eq!(batch_flat.len(), serial_flat.len());
        for (i, (b, s)) in batch_flat.iter().zip(serial_flat.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-4,
                "WC mismatch at index {}: batch={}, serial={}",
                i, b, s
            );
        }
    }

    #[test]
    fn test_batch_vs_serial_jr_single_subnet() {
        use crate::engine::HybridEngine;
        use crate::io::tensor_to_flat_f32;

        let device: <B as Backend>::Device = Default::default();
        let n_steps = 30usize;
        let dt = 0.1f64;
        let sweep_val = 0.25f32;

        let mut params = crate::model::jansen_rit::jansen_rit_default_params();
        params[12] = sweep_val; // mu

        let nnodes = 2;
        let nvar = 6;
        let initial_state_serial: Vec<f32> = (0..nnodes * nvar).map(|i| (i as f32) * 0.01).collect();
        let mut initial_state_batch = vec![0.0f32; nnodes * nvar];
        for n in 0..nnodes {
            for v in 0..nvar {
                initial_state_batch[n * nvar + v] = initial_state_serial[v * nnodes + n];
            }
        }

        let mut config_serial = SimConfig {
            sim_length: n_steps as f64 * dt,
            dt,
            network: NetworkConfig {
                subnetworks: vec![crate::config::SubnetworkConfig {
                    model: "JansenRit".to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(initial_state_serial.clone()),
                    params: params.clone(),
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut config_batch = config_serial.clone();
        config_batch.network.subnetworks[0].initial_state = InitialStateConfig::Inline(initial_state_batch);

        let mut batch_engine = BatchHybridEngine::<B>::from_config(config_batch, 1, device.clone()).unwrap();
        let _ = batch_engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 12 },
            &[sweep_val],
            n_steps,
        );
        let batch_s0 = batch_engine.states[0].clone().narrow(0, 0, 1).squeeze::<2>(0);
        let (batch_flat, _) = tensor_to_flat_f32(batch_s0);

        let mut serial_engine = HybridEngine::<B>::from_config(config_serial, device).unwrap();
        serial_engine.run(n_steps);
        let serial_state_2d = serial_engine.states[0].clone().permute([1, 0, 2]).squeeze::<2>(2);
        let (serial_flat, _) = tensor_to_flat_f32(serial_state_2d);

        assert_eq!(batch_flat.len(), serial_flat.len());
        for (i, (b, s)) in batch_flat.iter().zip(serial_flat.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-4,
                "JR mismatch at index {}: batch={}, serial={}",
                i, b, s
            );
        }
    }

    #[test]
    fn test_batch_vs_serial_mpr_single_subnet() {
        use crate::engine::HybridEngine;
        use crate::io::tensor_to_flat_f32;

        let device: <B as Backend>::Device = Default::default();
        let n_steps = 30usize;
        let dt = 0.1f64;
        let sweep_val = 0.5f32;

        let mut params = crate::model::mpr::mpr_default_params();
        params[4] = sweep_val; // I_ext

        let nnodes = 2;
        let nvar = 2;
        let initial_state_serial: Vec<f32> = (0..nnodes * nvar).map(|i| (i as f32) * 0.01).collect();
        let mut initial_state_batch = vec![0.0f32; nnodes * nvar];
        for n in 0..nnodes {
            for v in 0..nvar {
                initial_state_batch[n * nvar + v] = initial_state_serial[v * nnodes + n];
            }
        }

        let mut config_serial = SimConfig {
            sim_length: n_steps as f64 * dt,
            dt,
            network: NetworkConfig {
                subnetworks: vec![crate::config::SubnetworkConfig {
                    model: "MontbrioPazoRoxin".to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(initial_state_serial.clone()),
                    params: params.clone(),
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut config_batch = config_serial.clone();
        config_batch.network.subnetworks[0].initial_state = InitialStateConfig::Inline(initial_state_batch);

        let mut batch_engine = BatchHybridEngine::<B>::from_config(config_batch, 1, device.clone()).unwrap();
        let _ = batch_engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 4 },
            &[sweep_val],
            n_steps,
        );
        let batch_s0 = batch_engine.states[0].clone().narrow(0, 0, 1).squeeze::<2>(0);
        let (batch_flat, _) = tensor_to_flat_f32(batch_s0);

        let mut serial_engine = HybridEngine::<B>::from_config(config_serial, device).unwrap();
        serial_engine.run(n_steps);
        let serial_state_2d = serial_engine.states[0].clone().permute([1, 0, 2]).squeeze::<2>(2);
        let (serial_flat, _) = tensor_to_flat_f32(serial_state_2d);

        assert_eq!(batch_flat.len(), serial_flat.len());
        for (i, (b, s)) in batch_flat.iter().zip(serial_flat.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-4,
                "MPR mismatch at index {}: batch={}, serial={}",
                i, b, s
            );
        }
    }

    #[test]
    fn test_batch_vs_serial_kuramoto_single_subnet() {
        use crate::engine::HybridEngine;
        use crate::io::tensor_to_flat_f32;

        let device: <B as Backend>::Device = Default::default();
        let n_steps = 30usize;
        let dt = 0.1f64;
        let sweep_val = 1.0f32;

        let mut params = crate::model::kuramoto_model::kuramoto_default_params();
        params[0] = sweep_val; // omega

        let nnodes = 2;
        let nvar = 1;
        let initial_state_serial: Vec<f32> = (0..nnodes * nvar).map(|i| (i as f32) * 0.1).collect();
        // nvar=1 means serial and batch layouts are identical
        let initial_state_batch = initial_state_serial.clone();

        let mut config_serial = SimConfig {
            sim_length: n_steps as f64 * dt,
            dt,
            network: NetworkConfig {
                subnetworks: vec![crate::config::SubnetworkConfig {
                    model: "Kuramoto".to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(initial_state_serial.clone()),
                    params: params.clone(),
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut config_batch = config_serial.clone();
        config_batch.network.subnetworks[0].initial_state = InitialStateConfig::Inline(initial_state_batch);

        let mut batch_engine = BatchHybridEngine::<B>::from_config(config_batch, 1, device.clone()).unwrap();
        let _ = batch_engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 0 },
            &[sweep_val],
            n_steps,
        );
        let batch_s0 = batch_engine.states[0].clone().narrow(0, 0, 1).squeeze::<2>(0);
        let (batch_flat, _) = tensor_to_flat_f32(batch_s0);

        let mut serial_engine = HybridEngine::<B>::from_config(config_serial, device).unwrap();
        serial_engine.run(n_steps);
        let serial_state_2d = serial_engine.states[0].clone().permute([1, 0, 2]).squeeze::<2>(2);
        let (serial_flat, _) = tensor_to_flat_f32(serial_state_2d);

        assert_eq!(batch_flat.len(), serial_flat.len());
        for (i, (b, s)) in batch_flat.iter().zip(serial_flat.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-4,
                "Kuramoto mismatch at index {}: batch={}, serial={}",
                i, b, s
            );
        }
    }

    #[test]
    fn test_batch_vs_serial_rww_single_subnet() {
        use crate::engine::HybridEngine;
        use crate::io::tensor_to_flat_f32;

        let device: <B as Backend>::Device = Default::default();
        let n_steps = 30usize;
        let dt = 0.1f64;
        let sweep_val = 0.15f32;

        let mut params = crate::model::rww::rww_default_params();
        params[7] = sweep_val; // I_o

        let nnodes = 2;
        let nvar = 1;
        let initial_state_serial: Vec<f32> = (0..nnodes * nvar).map(|i| (i as f32) * 0.1).collect();
        let initial_state_batch = initial_state_serial.clone();

        let mut config_serial = SimConfig {
            sim_length: n_steps as f64 * dt,
            dt,
            network: NetworkConfig {
                subnetworks: vec![crate::config::SubnetworkConfig {
                    model: "ReducedWongWang".to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(initial_state_serial.clone()),
                    params: params.clone(),
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut config_batch = config_serial.clone();
        config_batch.network.subnetworks[0].initial_state = InitialStateConfig::Inline(initial_state_batch);

        let mut batch_engine = BatchHybridEngine::<B>::from_config(config_batch, 1, device.clone()).unwrap();
        let _ = batch_engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 7 },
            &[sweep_val],
            n_steps,
        );
        let batch_s0 = batch_engine.states[0].clone().narrow(0, 0, 1).squeeze::<2>(0);
        let (batch_flat, _) = tensor_to_flat_f32(batch_s0);

        let mut serial_engine = HybridEngine::<B>::from_config(config_serial, device).unwrap();
        serial_engine.run(n_steps);
        let serial_state_2d = serial_engine.states[0].clone().permute([1, 0, 2]).squeeze::<2>(2);
        let (serial_flat, _) = tensor_to_flat_f32(serial_state_2d);

        assert_eq!(batch_flat.len(), serial_flat.len());
        for (i, (b, s)) in batch_flat.iter().zip(serial_flat.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-4,
                "RWW mismatch at index {}: batch={}, serial={}",
                i, b, s
            );
        }
    }

    #[test]
    fn test_batch_vs_serial_two_subnet_ring() {
        // Validate BatchHybridEngine against serial HybridEngine for a 2-subnet
        // ring with scalar all-to-all coupling.
        use crate::engine::HybridEngine;
        use crate::io::tensor_to_flat_f32;

        let device: <B as Backend>::Device = Default::default();
        let n_steps = 30usize;
        let dt = 0.1f64;
        let w = 0.1f32;
        let sweep_val = 0.2f32;

        let mut g2do_params = crate::model::g2do::g2do_default_params();
        g2do_params[1] = sweep_val;
        let wc_params = crate::model::wilson_cowan::wilson_cowan_default_params();

        let nnodes = 2;
        let g2do_nvar = 2;
        let wc_nvar = 2;

        let g2do_serial: Vec<f32> = (0..nnodes * g2do_nvar).map(|i| (i as f32) * 0.01).collect();
        let mut g2do_batch = vec![0.0f32; nnodes * g2do_nvar];
        for n in 0..nnodes {
            for v in 0..g2do_nvar {
                g2do_batch[n * g2do_nvar + v] = g2do_serial[v * nnodes + n];
            }
        }

        let wc_serial: Vec<f32> = (0..nnodes * wc_nvar).map(|i| 0.1 + (i as f32) * 0.01).collect();
        let mut wc_batch = vec![0.0f32; nnodes * wc_nvar];
        for n in 0..nnodes {
            for v in 0..wc_nvar {
                wc_batch[n * wc_nvar + v] = wc_serial[v * nnodes + n];
            }
        }

        let mut config_serial = SimConfig {
            sim_length: n_steps as f64 * dt,
            dt,
            network: NetworkConfig {
                subnetworks: vec![
                    crate::config::SubnetworkConfig {
                        model: "Generic2dOscillator".to_string(),
                        nnodes,
                        nmodes: 1,
                        initial_state: InitialStateConfig::Inline(g2do_serial.clone()),
                        params: g2do_params.clone(),
                    },
                    crate::config::SubnetworkConfig {
                        model: "WilsonCowan".to_string(),
                        nnodes,
                        nmodes: 1,
                        initial_state: InitialStateConfig::Inline(wc_serial.clone()),
                        params: wc_params.clone(),
                    },
                ],
                projections: vec![
                    crate::config::ProjectionConfig {
                        src: 0,
                        tgt: 1,
                        conn_type: "all_to_all".to_string(),
                        weights: WeightsConfig::Scalar(w),
                        delays: vec![0],
                        tract_lengths: vec![],
                        coupling_fn: "Linear".to_string(),
                        coupling_params: vec![1.0],
                        cvar_map: "0:0".to_string(),
                    },
                    crate::config::ProjectionConfig {
                        src: 1,
                        tgt: 0,
                        conn_type: "all_to_all".to_string(),
                        weights: WeightsConfig::Scalar(w),
                        coupling_fn: "Linear".to_string(),
                        coupling_params: vec![1.0],
                        cvar_map: "0:0".to_string(),
                        delays: vec![0],
                        tract_lengths: vec![],
                    },
                ],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut config_batch = config_serial.clone();
        config_batch.network.subnetworks[0].initial_state = InitialStateConfig::Inline(g2do_batch);
        config_batch.network.subnetworks[1].initial_state = InitialStateConfig::Inline(wc_batch);

        let mut batch_engine = BatchHybridEngine::<B>::from_config(config_batch, 1, device.clone()).unwrap();
        let _ = batch_engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 1 },
            &[sweep_val],
            n_steps,
        );

        let mut serial_engine = HybridEngine::<B>::from_config(config_serial, device).unwrap();
        serial_engine.run(n_steps);

        // Compare subnetwork 0 (G2DO)
        let batch_s0 = batch_engine.states[0].clone().narrow(0, 0, 1).squeeze::<2>(0);
        let (batch_flat_0, _) = tensor_to_flat_f32(batch_s0);
        let serial_state_0 = serial_engine.states[0].clone().permute([1, 0, 2]).squeeze::<2>(2);
        let (serial_flat_0, _) = tensor_to_flat_f32(serial_state_0);
        for (i, (b, s)) in batch_flat_0.iter().zip(serial_flat_0.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-4,
                "2-subnet G2DO mismatch at index {}: batch={}, serial={}",
                i, b, s
            );
        }

        // Compare subnetwork 1 (WC)
        let batch_s1 = batch_engine.states[1].clone().narrow(0, 0, 1).squeeze::<2>(0);
        let (batch_flat_1, _) = tensor_to_flat_f32(batch_s1);
        let serial_state_1 = serial_engine.states[1].clone().permute([1, 0, 2]).squeeze::<2>(2);
        let (serial_flat_1, _) = tensor_to_flat_f32(serial_state_1);
        for (i, (b, s)) in batch_flat_1.iter().zip(serial_flat_1.iter()).enumerate() {
            assert!(
                (b - s).abs() < 1e-4,
                "2-subnet WC mismatch at index {}: batch={}, serial={}",
                i, b, s
            );
        }
    }

    #[test]
    fn test_batch_vs_serial_three_subnet_ring() {
        // Validate BatchHybridEngine against serial HybridEngine for a 3-subnet
        // ring (G2DO -> JR -> WC) with scalar all-to-all coupling.
        // This is the same topology as the Numba CUDA benchmark.
        use crate::engine::HybridEngine;
        use crate::io::tensor_to_flat_f32;

        let device: <B as Backend>::Device = Default::default();
        let n_steps = 30usize;
        let dt = 0.1f64;
        let w = 0.1f32;
        let sweep_val = 0.2f32;

        let mut g2do_params = crate::model::g2do::g2do_default_params();
        g2do_params[1] = sweep_val;
        let jr_params = crate::model::jansen_rit::jansen_rit_default_params();
        let wc_params = crate::model::wilson_cowan::wilson_cowan_default_params();

        let nnodes = 2;
        let g2do_nvar = 2;
        let jr_nvar = 6;
        let wc_nvar = 2;

        let g2do_serial: Vec<f32> = (0..nnodes * g2do_nvar).map(|i| (i as f32) * 0.01).collect();
        let mut g2do_batch = vec![0.0f32; nnodes * g2do_nvar];
        for n in 0..nnodes { for v in 0..g2do_nvar { g2do_batch[n * g2do_nvar + v] = g2do_serial[v * nnodes + n]; } }

        let jr_serial: Vec<f32> = (0..nnodes * jr_nvar).map(|i| (i as f32) * 0.001).collect();
        let mut jr_batch = vec![0.0f32; nnodes * jr_nvar];
        for n in 0..nnodes { for v in 0..jr_nvar { jr_batch[n * jr_nvar + v] = jr_serial[v * nnodes + n]; } }

        let wc_serial: Vec<f32> = (0..nnodes * wc_nvar).map(|i| 0.1 + (i as f32) * 0.01).collect();
        let mut wc_batch = vec![0.0f32; nnodes * wc_nvar];
        for n in 0..nnodes { for v in 0..wc_nvar { wc_batch[n * wc_nvar + v] = wc_serial[v * nnodes + n]; } }

        let mut config_serial = SimConfig {
            sim_length: n_steps as f64 * dt,
            dt,
            network: NetworkConfig {
                subnetworks: vec![
                    crate::config::SubnetworkConfig {
                        model: "Generic2dOscillator".to_string(), nnodes, nmodes: 1,
                        initial_state: InitialStateConfig::Inline(g2do_serial.clone()),
                        params: g2do_params.clone(),
                    },
                    crate::config::SubnetworkConfig {
                        model: "JansenRit".to_string(), nnodes, nmodes: 1,
                        initial_state: InitialStateConfig::Inline(jr_serial.clone()),
                        params: jr_params.clone(),
                    },
                    crate::config::SubnetworkConfig {
                        model: "WilsonCowan".to_string(), nnodes, nmodes: 1,
                        initial_state: InitialStateConfig::Inline(wc_serial.clone()),
                        params: wc_params.clone(),
                    },
                ],
                projections: vec![
                    crate::config::ProjectionConfig {
                        src: 0, tgt: 1, conn_type: "all_to_all".to_string(),
                        weights: WeightsConfig::Scalar(w), coupling_fn: "Linear".to_string(),
                        coupling_params: vec![1.0], cvar_map: "0:0".to_string(), delays: vec![0], tract_lengths: vec![],
                    },
                    crate::config::ProjectionConfig {
                        src: 1, tgt: 2, conn_type: "all_to_all".to_string(),
                        weights: WeightsConfig::Scalar(w), coupling_fn: "Linear".to_string(),
                        coupling_params: vec![1.0], cvar_map: "0:0".to_string(), delays: vec![0], tract_lengths: vec![],
                    },
                    crate::config::ProjectionConfig {
                        src: 2, tgt: 0, conn_type: "all_to_all".to_string(),
                        weights: WeightsConfig::Scalar(w), coupling_fn: "Linear".to_string(),
                        coupling_params: vec![1.0], cvar_map: "0:0".to_string(), delays: vec![0], tract_lengths: vec![],
                    },
                ],
            },
            integrator: IntegratorKind::Euler,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut config_batch = config_serial.clone();
        config_batch.network.subnetworks[0].initial_state = InitialStateConfig::Inline(g2do_batch);
        config_batch.network.subnetworks[1].initial_state = InitialStateConfig::Inline(jr_batch);
        config_batch.network.subnetworks[2].initial_state = InitialStateConfig::Inline(wc_batch);

        let mut batch_engine = BatchHybridEngine::<B>::from_config(config_batch, 1, device.clone()).unwrap();
        let _ = batch_engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 1 },
            &[sweep_val], n_steps,
        );

        let mut serial_engine = HybridEngine::<B>::from_config(config_serial, device).unwrap();
        serial_engine.run(n_steps);

        for sub_idx in 0..3 {
            let batch_s = batch_engine.states[sub_idx].clone().narrow(0, 0, 1).squeeze::<2>(0);
            let (batch_flat, _) = tensor_to_flat_f32(batch_s);
            let serial_state = serial_engine.states[sub_idx].clone().permute([1, 0, 2]).squeeze::<2>(2);
            let (serial_flat, _) = tensor_to_flat_f32(serial_state);
            assert_eq!(batch_flat.len(), serial_flat.len());
            for (i, (b, s)) in batch_flat.iter().zip(serial_flat.iter()).enumerate() {
                assert!(
                    (b - s).abs() < 1e-3,
                    "3-subnet sub={} mismatch at index {}: batch={}, serial={}",
                    sub_idx, i, b, s
                );
            }
        }
    }

    #[test]
    fn bench_generic_engine_3subnet_small() {
        use std::time::Instant;
        let n_sweep = 8usize;
        let n_steps = 50usize;
        let nnodes = 4usize;
        let dt = 0.1f64;
        let w = 0.01f32;

        let i_ext_values: Vec<f32> = (0..n_sweep)
            .map(|i| -0.5 + i as f32 * (1.0f32 / (n_sweep - 1).max(1) as f32))
            .collect();

        let mut lcg_state: u64 = 42;
        let lcg_next = |state: &mut u64| -> f32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let x = ((*state >> 33) as u32).min(0x7FFFFFFE);
            x as f32 / 0x7FFFFFFF as f32
        };
        let ic_g2do: Vec<f32> = (0..nnodes * 2)
            .map(|_| lcg_next(&mut lcg_state) * 0.2 - 0.1)
            .collect();
        let ic_jr: Vec<f32> = (0..nnodes * 6)
            .map(|_| lcg_next(&mut lcg_state) * 0.02 - 0.01)
            .collect();
        let ic_wc: Vec<f32> = (0..nnodes * 2)
            .map(|_| lcg_next(&mut lcg_state) * 0.2 + 0.1)
            .collect();

        let config = SimConfig {
            sim_length: n_steps as f64 * dt,
            dt,
            network: NetworkConfig {
                subnetworks: vec![
                    crate::config::SubnetworkConfig {
                        model: "Generic2dOscillator".to_string(),
                        nnodes,
                        nmodes: 1,
                        initial_state: InitialStateConfig::Inline(ic_g2do),
                        params: crate::model::g2do::g2do_default_params(),
                    },
                    crate::config::SubnetworkConfig {
                        model: "JansenRit".to_string(),
                        nnodes,
                        nmodes: 1,
                        initial_state: InitialStateConfig::Inline(ic_jr),
                        params: crate::model::jansen_rit::jansen_rit_default_params(),
                    },
                    crate::config::SubnetworkConfig {
                        model: "WilsonCowan".to_string(),
                        nnodes,
                        nmodes: 1,
                        initial_state: InitialStateConfig::Inline(ic_wc),
                        params: crate::model::wilson_cowan::wilson_cowan_default_params(),
                    },
                ],
                projections: vec![
                    crate::config::ProjectionConfig {
                        src: 0, tgt: 1,
                        conn_type: "all_to_all".to_string(),
                        weights: WeightsConfig::Scalar(w),
                        coupling_fn: "Linear".to_string(),
                        coupling_params: vec![1.0],
                        cvar_map: "0:0".to_string(),
                        delays: vec![0],
                        tract_lengths: vec![],
                    },
                    crate::config::ProjectionConfig {
                        src: 1, tgt: 2,
                        conn_type: "all_to_all".to_string(),
                        weights: WeightsConfig::Scalar(w),
                        coupling_fn: "Linear".to_string(),
                        coupling_params: vec![1.0],
                        cvar_map: "0:0".to_string(),
                        delays: vec![0],
                        tract_lengths: vec![],
                    },
                    crate::config::ProjectionConfig {
                        src: 2, tgt: 0,
                        conn_type: "all_to_all".to_string(),
                        weights: WeightsConfig::Scalar(w),
                        coupling_fn: "Linear".to_string(),
                        coupling_params: vec![1.0],
                        cvar_map: "0:0".to_string(),
                        delays: vec![0],
                        tract_lengths: vec![],
                    },
                ],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let device: <B as Backend>::Device = Default::default();
        let mut engine = BatchHybridEngine::<B>::from_config(config, n_sweep, device).unwrap();
        let start = Instant::now();
        let result = engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 1 },
            &i_ext_values,
            n_steps,
        );
        let elapsed_ms = start.elapsed().as_millis() as f64;

        assert_eq!(result.n_sweep, n_sweep);
        assert!(result.tavg[0].iter().all(|x| x.is_finite()));
        assert!(result.tavg[1].iter().all(|x| x.is_finite()));
        assert!(result.tavg[2].iter().all(|x| x.is_finite()));

        let per_sweep = elapsed_ms / n_sweep as f64;
        let per_step_per_sweep = elapsed_ms / (n_sweep * n_steps) as f64;
        assert!(elapsed_ms < 1000.0, "generic engine too slow: {}ms", elapsed_ms);

        eprintln!(
            "Generic 3-subnet: n_sweep={}, n_steps={}, nnodes={}, elapsed={:.2}ms, {:.3} ms/sweep, {:.5} ms/step/sweep",
            n_sweep, n_steps, nnodes, elapsed_ms, per_sweep, per_step_per_sweep
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn bench_rayon_vs_single_g2do() {
        use std::time::Instant;
        let n_sweep = 32usize;
        let n_steps = 100usize;
        let nnodes = 8usize;
        let dt = 0.1f64;

        let sweep_values: Vec<f32> = (0..n_sweep)
            .map(|i| -0.5 + i as f32 * (1.0f32 / (n_sweep - 1).max(1) as f32))
            .collect();

        let initial_state = vec![0.1f32; 2 * nnodes];
        let config = SimConfig {
            sim_length: n_steps as f64 * dt,
            dt,
            network: NetworkConfig {
                subnetworks: vec![crate::config::SubnetworkConfig {
                    model: "Generic2dOscillator".to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(initial_state),
                    params: crate::model::g2do::g2do_default_params(),
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let device: <B as Backend>::Device = Default::default();

        // Single-threaded batch
        let mut engine_single = BatchHybridEngine::<B>::from_config(config.clone(), n_sweep, device.clone()).unwrap();
        let start = Instant::now();
        let _single_result = engine_single.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 1 },
            &sweep_values,
            n_steps,
        );
        let single_ms = start.elapsed().as_millis() as f64;

        // Rayon batch (2 threads so test is deterministic even on small CI runners)
        let start = Instant::now();
        let _rayon_result = rayon_batch_sweep::<B>(
            config,
            SweepParam { sub_idx: 0, param_idx: 1 },
            &sweep_values,
            n_steps,
            Some(2),
            device,
        );
        let rayon_ms = start.elapsed().as_millis() as f64;

        eprintln!(
            "G2DO NdArray n_sweep={} n_steps={} nnodes={}: single={:.1}ms ({:.2}ms/sweep), rayon={:.1}ms ({:.2}ms/sweep), speedup={:.2}x",
            n_sweep, n_steps, nnodes,
            single_ms, single_ms / n_sweep as f64,
            rayon_ms, rayon_ms / n_sweep as f64,
            single_ms / rayon_ms.max(1.0)
        );

        // Sanity: rayon should not be orders of magnitude slower
        assert!(rayon_ms < single_ms * 5.0, "rayon unexpectedly slow: {}ms vs {}ms", rayon_ms, single_ms);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn bench_rayon_batch_vs_parallel_sweep() {
        use std::time::Instant;
        use crate::engine::sweep::{parallel_sweep, SweepConfig};
        let n_sweep = 16usize;
        let n_steps = 50usize;
        let nnodes = 4usize;
        let dt = 0.1f64;

        let sweep_values: Vec<f32> = (0..n_sweep)
            .map(|i| -0.5 + i as f32 * (1.0f32 / (n_sweep - 1).max(1) as f32))
            .collect();

        let initial_state = vec![0.1f32; 2 * nnodes];
        let base_config = SimConfig {
            sim_length: n_steps as f64 * dt,
            dt,
            network: NetworkConfig {
                subnetworks: vec![crate::config::SubnetworkConfig {
                    model: "Generic2dOscillator".to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(initial_state.clone()),
                    params: crate::model::g2do::g2do_default_params(),
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let device: <B as Backend>::Device = Default::default();

        // Rayon batch sweep (our new approach — shard param values, batch inside each shard)
        let rayon_start = Instant::now();
        let rayon_result = rayon_batch_sweep::<B>(
            base_config.clone(),
            SweepParam { sub_idx: 0, param_idx: 1 },
            &sweep_values,
            n_steps,
            None, // use all cores
            device.clone(),
        );
        let rayon_us = rayon_start.elapsed().as_micros() as f64;

        // Existing parallel_sweep (one HybridEngine per point, all in one batch)
        let sweep_config = SweepConfig {
            param_idx: 1,
            param_values: sweep_values.clone(),
            n_steps,
            nnodes,
            dt,
            keep_trajectory: false,
            feature_set: crate::sbi::features::FeatureSet::Classic,
        };
        let parallel_start = Instant::now();
        let _parallel_result = parallel_sweep::<B>(
            &sweep_config,
            device,
        );
        let parallel_us = parallel_start.elapsed().as_micros() as f64;

        let rayon_ms = rayon_us / 1000.0;
        let parallel_ms = parallel_us / 1000.0;

        eprintln!(
            "G2DO sweep n_sweep={} n_steps={} nnodes={}: rayon_batch={:.2}ms ({:.3}ms/pt), parallel_sweep={:.2}ms ({:.3}ms/pt), ratio={:.2}x",
            n_sweep, n_steps, nnodes,
            rayon_ms, rayon_ms / n_sweep as f64,
            parallel_ms, parallel_ms / n_sweep as f64,
            parallel_ms / rayon_ms.max(0.001)
        );

        // Sanity: our batch approach should be competitive
        assert!(rayon_result.n_sweep == n_sweep);
    }

    #[test]
    fn bench_single_subnet_models_generic() {
        use std::time::Instant;
        let n_sweep = 128usize;
        let n_steps = 100usize;
        let nnodes = 16usize;
        let dt = 0.1f64;

        let sweep_values: Vec<f32> = (0..n_sweep)
            .map(|i| -0.5 + i as f32 * (1.0f32 / (n_sweep - 1).max(1) as f32))
            .collect();

        let g2do_init = vec![0.1f32; 2 * nnodes];
        let jr_init = vec![0.01f32; 6 * nnodes];
        let wc_init = vec![0.1f32; 2 * nnodes];

        for (name, model_name, initial_state, default_params) in [
            ("G2DO", "Generic2dOscillator", g2do_init, crate::model::g2do::g2do_default_params as fn() -> Vec<f32>),
            ("JR", "JansenRit", jr_init, crate::model::jansen_rit::jansen_rit_default_params as fn() -> Vec<f32>),
            ("WC", "WilsonCowan", wc_init, crate::model::wilson_cowan::wilson_cowan_default_params as fn() -> Vec<f32>),
        ] {
            let nvar: usize = match model_name {
                "Generic2dOscillator" => 2,
                "JansenRit" => 6,
                "WilsonCowan" => 2,
                _ => 2,
            };
            let config = SimConfig {
                sim_length: n_steps as f64 * dt,
                dt,
                network: NetworkConfig {
                    subnetworks: vec![crate::config::SubnetworkConfig {
                        model: model_name.to_string(),
                        nnodes,
                        nmodes: 1,
                        initial_state: InitialStateConfig::Inline(initial_state.clone()),
                        params: default_params(),
                    }],
                    projections: vec![],
                },
                integrator: IntegratorKind::Heun,
                monitors: vec![],
                stimuli: vec![],
                nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
            };

            let device: <B as Backend>::Device = Default::default();
            let mut engine = BatchHybridEngine::<B>::from_config(config, n_sweep, device).unwrap();
            let start = Instant::now();
            let result = engine.run_sweep(
                &SweepParam { sub_idx: 0, param_idx: 1 },
                &sweep_values,
                n_steps,
            );
            let elapsed_ms = start.elapsed().as_millis() as f64;

            assert!(result.tavg[0].iter().all(|x| x.is_finite()), "{} NaN in tavg", name);
            assert_eq!(result.n_sweep, n_sweep);

            let per_sweep_ms = elapsed_ms / n_sweep as f64;
            let per_step_per_sweep_ms = elapsed_ms / (n_sweep * n_steps) as f64;
            eprintln!(
                "Single-subnet {}: n_sweep={}, n_steps={}, nnodes={}, nvar={}, elapsed={:.2}ms, {:.3}ms/sweep, {:.5}ms/step/sweep",
                name, n_sweep, n_steps, nnodes, nvar, elapsed_ms, per_sweep_ms, per_step_per_sweep_ms
            );
        }
    }

    #[test]
    fn test_batch_engine_delay_coupling() {
        // Validate that delay coupling extracts the correct past state.
        let device: <B as Backend>::Device = Default::default();
        let n_steps = 10usize;
        let dt = 0.1f64;

        let mut g2do_params = crate::model::g2do::g2do_default_params();
        g2do_params[1] = 0.0f32;

        let nnodes = 2;
        let nvar = 2;

        let initial_state = vec![0.1f32, 0.2, 0.3, 0.4];

        let mut config = SimConfig {
            sim_length: n_steps as f64 * dt,
            dt,
            network: NetworkConfig {
                subnetworks: vec![
                    crate::config::SubnetworkConfig {
                        model: "Generic2dOscillator".to_string(),
                        nnodes,
                        nmodes: 1,
                        initial_state: InitialStateConfig::Inline(initial_state.clone()),
                        params: g2do_params.clone(),
                    },
                    crate::config::SubnetworkConfig {
                        model: "Generic2dOscillator".to_string(),
                        nnodes,
                        nmodes: 1,
                        initial_state: InitialStateConfig::Inline(initial_state.clone()),
                        params: g2do_params.clone(),
                    },
                ],
                projections: vec![
                    crate::config::ProjectionConfig {
                        src: 0, tgt: 1,
                        conn_type: "all_to_all".to_string(),
                        weights: WeightsConfig::Scalar(0.1),
                        coupling_fn: "Linear".to_string(),
                        coupling_params: vec![1.0],
                        cvar_map: "0:0".to_string(),
                        delays: vec![3], // 3-step delay
                        tract_lengths: vec![],
                    },
                ],
            },
            integrator: IntegratorKind::Euler,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut engine = BatchHybridEngine::<B>::from_config(config.clone(), 1, device.clone()).unwrap();
        let result_delay = engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 1 },
            &[0.0f32],
            n_steps,
        );

        // Compare with no-delay version
        config.network.projections[0].delays = vec![0];
        let mut engine_no_delay = BatchHybridEngine::<B>::from_config(config, 1, device).unwrap();
        let result_no_delay = engine_no_delay.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 1 },
            &[0.0f32],
            n_steps,
        );

        // Delay and no-delay should produce different results (not identical)
        let diff: f32 = result_delay.tavg[1].iter().zip(result_no_delay.tavg[1].iter())
            .map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-8, "delay coupling should change output, diff={}", diff);

        // Both should be finite
        assert!(result_delay.tavg[1].iter().all(|x| x.is_finite()));
        assert!(result_no_delay.tavg[1].iter().all(|x| x.is_finite()));

        // History should have been allocated and be non-empty
        assert_eq!(engine.histories.len(), 2);
        assert!(engine.histories[0].shape().dims[3] >= 4); // horizon >= delay+1
    }

    #[test]
    fn test_batch_engine_trajectory_recording() {
        let device: <B as Backend>::Device = Default::default();
        let n_steps = 10usize;
        let config = SimConfig {
            sim_length: n_steps as f64 * 0.1,
            dt: 0.1,
            network: NetworkConfig {
                subnetworks: vec![crate::config::SubnetworkConfig {
                    model: "Generic2dOscillator".to_string(),
                    nnodes: 2,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.1f32, 0.2, 0.3, 0.4]),
                    params: crate::model::g2do::g2do_default_params(),
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Euler,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut engine_with_traj = BatchHybridEngine::<B>::from_config(config.clone(), 2, device.clone()).unwrap();
        let result_with = engine_with_traj.run_sweep_with_trajectory(
            &SweepParam { sub_idx: 0, param_idx: 1 },
            &[-0.5f32, 0.5],
            n_steps,
        );
        let traj_with = result_with.trajectories.as_ref().expect("trajectory should exist");
        let traj_sub0 = &traj_with[0];
        let expected_per_step = 2 * 2 * 2; // n_sweep * nnodes * nvar
        assert_eq!(traj_sub0.len(), n_steps * expected_per_step, "traj length mismatch");

        let mut engine_without = BatchHybridEngine::<B>::from_config(config, 2, device).unwrap();
        let result_without = engine_without.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 1 },
            &[-0.5f32, 0.5],
            n_steps,
        );
        assert!(result_without.trajectories.is_none());

        assert_eq!(result_with.tavg[0].len(), result_without.tavg[0].len());
        for (w, wo) in result_with.tavg[0].iter().zip(result_without.tavg[0].iter()) {
            assert!((w - wo).abs() < 1e-6, "tavg mismatch with traj: {} vs {}", w, wo);
        }
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_rayon_batch_matches_single_batch() {
        let device: <B as Backend>::Device = Default::default();
        let n_steps = 30usize;
        let sweep_values = vec![-0.5f32, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
        let nnodes = 4;
        let config = SimConfig {
            sim_length: n_steps as f64 * 0.1,
            dt: 0.1,
            network: NetworkConfig {
                subnetworks: vec![crate::config::SubnetworkConfig {
                    model: "Generic2dOscillator".to_string(),
                    nnodes,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.1f32; 2 * nnodes]),
                    params: crate::model::g2do::g2do_default_params(),
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: crate::config::NsigConfig::Scalar(0.0),
            speed: 3.0,
            backend: "ndarray".to_string(),
        };

        let mut single_engine = BatchHybridEngine::<B>::from_config(config.clone(), sweep_values.len(), device.clone()).unwrap();
        let single_result = single_engine.run_sweep(
            &SweepParam { sub_idx: 0, param_idx: 1 },
            &sweep_values,
            n_steps,
        );

        let rayon_result = rayon_batch_sweep::<B>(
            config,
            SweepParam { sub_idx: 0, param_idx: 1 },
            &sweep_values,
            n_steps,
            Some(2),
            device,
        );

        assert_eq!(single_result.n_sweep, rayon_result.n_sweep);
        assert_eq!(single_result.tavg.len(), rayon_result.tavg.len());
        for (sub_idx, (single_tavg, rayon_tavg)) in single_result.tavg.iter().zip(rayon_result.tavg.iter()).enumerate() {
            assert_eq!(single_tavg.len(), rayon_tavg.len());
            for (i, (s, r)) in single_tavg.iter().zip(rayon_tavg.iter()).enumerate() {
                assert!(
(s - r).abs() < 1e-4, "Rayon mismatch at sub {} idx {}: single={} rayon={}", sub_idx, i, s, r);
            }
        }
    }
}
