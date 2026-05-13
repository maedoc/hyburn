//! Engine runtime — run(), step(), checkpoint(), resume().
//!
//! Contains the hot-path integration loop and checkpoint serialization.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
use crate::io::tensor_to_flat_f32;
use crate::error::{Result, SimulationError};

use super::construction::{HybridEngine, IntegratorKind, CKPT_MAGIC, CKPT_VERSION};
use super::monitor::Monitor;
use super::sparse::sparse_coupling;
use super::coupling::{CouplingFnConfig, dense_coupling};

impl<B: Backend> HybridEngine<B> {

    /// Run the simulation for `n_steps`.
    pub fn run(&mut self, n_steps: usize) {
        for _ in 0..n_steps {
            self.step();
        }
    }

    /// Flush any partial BOLD accumulator windows and write each monitor's
    /// down-sampled time-series to its configured output path (or a default
    /// path derived from `output_dir`).
    pub fn flush_bold_monitors(&mut self) {
        for monitor in self.bold_monitors.iter_mut() {
            let _ = monitor.flush();
        }
    }

    /// Flush all monitors (BOLD, sensor projection, spatial average) and
    /// return collected data per monitor type.
    pub fn flush_all_monitors(&mut self) {
        self.flush_bold_monitors();
        for monitor in self.sensor_monitors.iter_mut() {
            let _ = Monitor::<B>::flush(monitor);
        }
        for monitor in self.spatial_monitors.iter_mut() {
            let _ = Monitor::<B>::flush(monitor);
        }
    }

    /// Execute one integration step.
    pub fn step(&mut self) {
        let n_subs = self.subnetworks.len();
        let dt = self.dt as f32;

        // Report progress.
        if let Some(ref pr) = self.progress {
            pr.report(self.step);
        }

        // 1. Save current states into history buffers (ring buffer).
        for i in 0..n_subs {
            let h = &mut self.histories[i];
            let horizon = h.shape().dims[3];
            if horizon == 0 {
                continue;
            }
            let idx = self.step % horizon;

            let before = if idx > 0 {
                Some(h.clone().narrow(3, 0, idx))
            } else {
                None
            };
            let state_slice = self.states[i].clone().unsqueeze_dim::<4>(3);
            let after = if idx + 1 < horizon {
                Some(h.clone().narrow(3, idx + 1, horizon - idx - 1))
            } else {
                None
            };

            let mut parts: Vec<Tensor<B, 4>> = Vec::new();
            if let Some(b) = before {
                parts.push(b);
            }
            parts.push(state_slice);
            if let Some(a) = after {
                parts.push(a);
            }
            *h = Tensor::cat(parts, 3);
        }

        // 2. Compute incoming coupling for each target subnetwork.
        let mut couplings: Vec<Option<Tensor<B, 2>>> = vec![None; n_subs];
        for proj in &self.projections {
            let src_state = &self.states[proj.src];
            let src_sub = &self.subnetworks[proj.src];
            let tgt_sub = &self.subnetworks[proj.tgt];
            let needs_xi = proj.coupling_cfg.needs_x_i();

            let mut mode_couplings = Vec::new();
            for mode in 0..src_sub.nmodes {
                let mode_state = src_state.clone().narrow(2, mode, 1).squeeze::<2>(2); // [nvar, nnodes]
                let ncvar_extract = src_sub.ncvar.min(src_sub.nvar);
                let cvars = mode_state.narrow(0, 0, ncvar_extract); // [ncvar, nnodes]
                let cvars_t = cvars.permute([1, 0]); // [nnodes, ncvar]

                // Extract target cvars (x_i) when coupling function needs them
                let x_i: Option<Tensor<B, 2>> = if needs_xi {
                    let tgt_state = &self.states[proj.tgt];
                    let tgt_ncvar_extract = tgt_sub.ncvar.min(tgt_sub.nvar);
                    let tgt_mode_state = tgt_state.clone().narrow(2, mode, 1).squeeze::<2>(2);
                    let tgt_cvars = tgt_mode_state.narrow(0, 0, tgt_ncvar_extract);
                    Some(tgt_cvars.permute([1, 0])) // [ntgt_nodes, ncvar]
                } else {
                    None
                };

                let mode_coup = if proj.is_sparse {
                    let csr_data = proj.csr_data.as_ref()
                        .expect("CSR projection missing data (is_sparse=true but no csr_data)");
                    let csr_indices = proj.csr_indices.as_ref()
                        .expect("CSR projection missing indices (is_sparse=true but no csr_indices)");
                    let csr_indptr = proj.csr_indptr.as_ref()
                        .expect("CSR projection missing indptr (is_sparse=true but no csr_indptr)");

                    let has_per_edge_delays = proj.csr_idelays.as_ref()
                        .map(|d| d.len() == csr_data.len() && !d.is_empty())
                        .unwrap_or(false);

                    if has_per_edge_delays {
                        let ntgt = csr_indptr.len() - 1;
                        let pre_ncvar = if let CouplingFnConfig::Kuramoto { .. } = proj.coupling_cfg {
                            ncvar_extract * 2
                        } else {
                            ncvar_extract
                        };
                        let mut weighted_sum = vec![0.0f32; ntgt * pre_ncvar];
                        let h = &self.histories[proj.src];
                        let horizon = h.shape().dims[3];

                        for tgt in 0..ntgt {
                            for edge_idx in csr_indptr[tgt]..csr_indptr[tgt + 1] {
                                let src_node = csr_indices[edge_idx];
                                let weight = csr_data[edge_idx];
                                let edge_delay = proj.csr_idelays.as_ref()
                                    .and_then(|d| d.get(edge_idx).copied())
                                    .unwrap_or(0);

                                let src_row = if edge_delay == 0 || self.step == 0 {
                                    cvars_t.clone().narrow(0, src_node, 1) // [1, ncvar], GPU tensor
                                } else {
                                    let raw_delay = edge_delay as usize;
                                    if raw_delay <= self.step {
                                        let slot = (self.step - raw_delay + horizon) % horizon;
                                        let delayed_state = h.clone().narrow(3, slot, 1).squeeze::<3>(3);
                                        let delayed_mode_state = delayed_state.narrow(2, mode, 1).squeeze::<2>(2);
                                        let delayed_cvars = delayed_mode_state.narrow(0, 0, ncvar_extract);
                                        delayed_cvars.permute([1, 0]).narrow(0, src_node, 1) // [1, ncvar], GPU tensor
                                    } else {
                                        Tensor::<B, 2>::zeros([1, ncvar_extract], &self.device) // [1, ncvar], GPU tensor
                                    }
                                }; // src_row is [1, ncvar] — a GPU tensor, no CPU copy

                                // Apply pre() per-edge (before weighting)
                                let pre_result = proj.coupling_cfg.pre(src_row);
                                let pre_data = crate::io::tensor_to_flat_f32(pre_result).0;

                                for cvar in 0..pre_ncvar {
                                    weighted_sum[tgt * pre_ncvar + cvar] += weight * pre_data[cvar];
                                }
                            }
                        }

                        // Apply post_with_target() to the weighted sum
                        let weighted_sum_tensor = Tensor::<B, 2>::from_floats(
                            TensorData::new::<f32, Vec<usize>>(weighted_sum, vec![ntgt, pre_ncvar]),
                            &self.device,
                        );
                        proj.coupling_cfg.post_with_target(weighted_sum_tensor, x_i)
                    } else {
                        let delayed_cvars = if let Some(delay) = proj.delays.first().copied() {
                            if delay == 0 || self.step == 0 {
                                cvars_t.clone()
                            } else {
                                let h = &self.histories[proj.src];
                                let horizon = h.shape().dims[3];
                                let raw_delay = delay as usize;
                                if raw_delay <= self.step {
                                    let idx = (self.step - raw_delay) % horizon;
                                    let delayed_state = h.clone().narrow(3, idx, 1).squeeze::<3>(3);
                                    let delayed_mode_state = delayed_state.narrow(2, mode, 1).squeeze::<2>(2);
                                    let delayed_cvars_2d = delayed_mode_state.narrow(0, 0, ncvar_extract);
                                    delayed_cvars_2d.permute([1, 0])
                                } else {
                                    Tensor::<B, 2>::zeros([src_sub.nnodes, ncvar_extract], &self.device)
                                }
                            }
                        } else {
                            cvars_t.clone()
                        };

                        sparse_coupling(
                            csr_data,
                            csr_indices,
                            csr_indptr,
                            delayed_cvars,
                            &proj.coupling_cfg,
                            x_i,
                        )
                    }
                } else {
                    let delayed_cvars = if let Some(delay) = proj.delays.first().copied() {
                        if delay == 0 || self.step == 0 {
                            cvars_t.clone()
                        } else {
                            let h = &self.histories[proj.src];
                            let horizon = h.shape().dims[3];
                            let raw_delay = delay as usize;
                            if raw_delay <= self.step {
                                let idx = (self.step - raw_delay) % horizon;
                                let delayed_state = h.clone().narrow(3, idx, 1).squeeze::<3>(3);
                                let delayed_mode_state = delayed_state.narrow(2, mode, 1).squeeze::<2>(2);
                                let delayed_cvars_2d = delayed_mode_state.narrow(0, 0, ncvar_extract);
                                delayed_cvars_2d.permute([1, 0])
                            } else {
                                Tensor::<B, 2>::zeros([src_sub.nnodes, ncvar_extract], &self.device)
                            }
                        }
                    } else {
                        cvars_t.clone()
                    };

                    // Pipeline: pre(x_j) → W @ pre(x_j) → post_with_target(gx, x_i)
                    dense_coupling(proj.weights.clone(), delayed_cvars, &proj.coupling_cfg, x_i)
                };
                mode_couplings.push(mode_coup);
            }

            if mode_couplings.is_empty() {
                continue;
            }

            let raw_coupling = Tensor::cat(mode_couplings, 0); // [ntgt_nodes*nmodes, src_ncvar]

            // Remap cvars: src_ncvar → tgt_ncvar using cvar_map
            let tgt_sub = &self.subnetworks[proj.tgt];
            let tgt_ncvar = tgt_sub.ncvar;
            let ntgt_rows = raw_coupling.shape().dims[0];

            let proj_coupling = if tgt_ncvar == src_sub.ncvar && proj.cvar_map.len() == 1 && proj.cvar_map[0] == (0, 0) {
                // Fast path: 1:1 mapping with matching ncvar
                raw_coupling
            } else {
                // General path: remap via cvar_map
                // Read source coupling, scatter into target cvar layout
                let src_data = crate::io::tensor_to_flat_f32(raw_coupling).0;
                let mut tgt_data = match &couplings[proj.tgt] {
                    Some(existing) => crate::io::tensor_to_flat_f32(existing.clone()).0,
                    None => vec![0.0f32; ntgt_rows * tgt_ncvar],
                };
                let src_ncvar = src_sub.ncvar;
                for &(s, t) in &proj.cvar_map {
                    if s < src_ncvar && t < tgt_ncvar {
                        for row in 0..ntgt_rows {
                            tgt_data[row * tgt_ncvar + t] += src_data[row * src_ncvar + s];
                        }
                    }
                }
                Tensor::<B, 2>::from_floats(
                    TensorData::new::<f32, Vec<usize>>(tgt_data, vec![ntgt_rows, tgt_ncvar]),
                    &self.device,
                )
            };

            match &mut couplings[proj.tgt] {
                Some(existing) => {
                    *existing = existing.clone() + proj_coupling;
                }
                None => {
                    couplings[proj.tgt] = Some(proj_coupling);
                }
            }
        }

        // 2b. Apply stimulus to target subnetwork couplings.
        for stim in &self.stimuli {
            if stim.target >= n_subs {
                continue; // invalid target index, skip
            }
            let stim_val = stim.apply(self.step, self.dt);
            if stim_val == 0.0 {
                continue;
            }
            let sub = &self.subnetworks[stim.target];
            let nn = sub.nnodes * sub.nmodes;
            let dev = &self.device;
            match &mut couplings[stim.target] {
                Some(existing) => {
                    if sub.ncvar >= 1 {
                        let c0 = existing.clone().narrow(1, 0, 1).add_scalar(stim_val);
                        if sub.ncvar > 1 {
                            let rest = existing.clone().narrow(1, 1, sub.ncvar - 1);
                            *existing = Tensor::cat(vec![c0, rest], 1);
                        } else {
                            *existing = c0;
                        }
                    } else {
                        // ncvar == 0, nothing to add
                    }
                }
                None => {
                    let mut stim_data = vec![0.0f32; nn * sub.ncvar];
                    for r in 0..nn {
                        stim_data[r * sub.ncvar] = stim_val;
                    }
                    let stim_tensor = Tensor::<B, 2>::from_floats(
                        TensorData::new::<f32, Vec<usize>>(stim_data, vec![nn, sub.ncvar]),
                        dev,
                    );
                    couplings[stim.target] = Some(stim_tensor);
                }
            }
        }

        // 3. Integrate each subnetwork.
        for (i, sub) in self.subnetworks.iter().enumerate().take(n_subs) {
            let state = self.states[i].clone();

            // Flatten: [nvar, nnodes, nmodes] → [nnodes*nmodes, nvar]
            let state_2d = state
                .permute([1, 2, 0])
                .reshape([sub.nnodes * sub.nmodes, sub.nvar]);

            let coupling = match &couplings[i] {
                Some(c) => c.clone(),
                None => Tensor::<B, 2>::zeros([sub.nnodes * sub.nmodes, sub.ncvar], &self.device),
            };

            let new_state_2d = match self.integrator {
                IntegratorKind::Heun => super::integrator::heun_step(
                    state_2d,
                    coupling,
                    dt,
                    |s, c| sub.dfun(s, c),
                    |s| sub.clamp(s),
                ),
                IntegratorKind::Euler => super::integrator::euler_step(
                    state_2d,
                    coupling,
                    dt,
                    |s, c| sub.dfun(s, c),
                    |s| sub.clamp(s),
                ),
                IntegratorKind::EulerStochastic => super::integrator::euler_stochastic_step(
                    state_2d,
                    coupling,
                    dt,
                    &self.nsig,
                    |s, c| sub.dfun(s, c),
                    |s| sub.clamp(s),
                ),
                IntegratorKind::HeunStochastic => super::integrator::heun_stochastic_step(
                    state_2d,
                    coupling,
                    dt,
                    &self.nsig,
                    |s, c| sub.dfun(s, c),
                    |s| sub.clamp(s),
                ),
                IntegratorKind::Rk4 => super::integrator::rk4_step(
                    state_2d,
                    coupling,
                    dt,
                    |s, c| sub.dfun(s, c),
                    |s| sub.clamp(s),
                ),
                IntegratorKind::Rk4Stochastic => super::integrator::rk4_stochastic_step(
                    state_2d,
                    coupling,
                    dt,
                    &self.nsig,
                    |s, c| sub.dfun(s, c),
                    |s| sub.clamp(s),
                ),
            };

            // Reshape back: [nnodes*nmodes, nvar] → [nvar, nnodes, nmodes]
            self.states[i] = new_state_2d
                .reshape([sub.nnodes, sub.nmodes, sub.nvar])
                .permute([2, 0, 1]);
        }

        // 4. Record trajectory.
        for state in &self.states {
            let (flat, _shape) = tensor_to_flat_f32(state.clone());
            self.trajectory.extend_from_slice(&flat);
        }

        // 4b. Record sensor projection and spatial average monitors.
        for (mi, monitor) in self.sensor_monitors.iter_mut().enumerate() {
            let target = self.sensor_monitor_targets[mi];
            if target >= n_subs {
                continue;
            }
            Monitor::<B>::record(monitor, &self.states[target], self.step, self.step as f64 * self.dt);
        }
        for (mi, monitor) in self.spatial_monitors.iter_mut().enumerate() {
            let target = self.spatial_monitor_targets[mi];
            if target >= n_subs {
                continue;
            }
            Monitor::<B>::record(monitor, &self.states[target], self.step, self.step as f64 * self.dt);
        }

        // 5. Accumulate BOLD neural input (GPU-path: accumulate on device, sync only when period elapses)
        if !self.bold_monitors.is_empty() {
            // Initialize accumulators if needed (first step with BOLD monitors)
            if self.bold_accumulators.len() != self.bold_monitors.len() {
                self.bold_accumulators = self.bold_monitors.iter().map(|m| {
                    Some(Tensor::<B, 1>::zeros([m.nnodes], &self.device))
                }).collect();
                self.bold_accumulator_counts = vec![0; self.bold_monitors.len()];
            }

            for (mi, monitor) in self.bold_monitors.iter_mut().enumerate() {
                let target = monitor.target_subnetwork;
                if target >= n_subs {
                    continue;
                }
                let sub = &self.subnetworks[target];
                if sub.nvar == 0 || sub.nnodes == 0 {
                    continue;
                }
                let state = &self.states[target];
                // Extract var0 averaged over modes, shape [nnodes]
                let var0 = state.clone().narrow(0, 0, 1) // [1, nnodes, nmodes]
                    .squeeze::<2>(0)                      // [nnodes, nmodes]
                    .mean_dim(1)                          // [nnodes, 1]
                    .squeeze::<1>(1);                     // [nnodes]
                // Accumulate on GPU
                if let Some(ref mut acc) = self.bold_accumulators[mi] {
                    *acc = acc.clone() + var0;
                }
                self.bold_accumulator_counts[mi] += 1;

                // Check if this monitor needs flushing
                let count = self.bold_accumulator_counts[mi];
                if count >= monitor.bold_period {
                    let count_f = count as f32;
                    // Transfer accumulator to CPU, divide by count, pass to BOLD monitor
                    if let Some(ref acc) = self.bold_accumulators[mi] {
                        let avg = acc.clone().div_scalar(count_f);
                        let (flat, _shape) = crate::io::tensor_to_flat_f32::<B, 1>(avg);
                        monitor.accumulate(&flat);
                    }
                    // Reset this monitor's accumulator and counter
                    self.bold_accumulators[mi] = Some(Tensor::<B, 1>::zeros([monitor.nnodes], &self.device));
                    self.bold_accumulator_counts[mi] = 0;
                }
            }
        }

        self.step += 1;
    }

    /// Save the current engine state to a checkpoint `.bin` file.
    ///
    /// Not available in WASM builds (no filesystem access).
    ///
    /// The binary format (v2):
    /// - 8 bytes magic (`HYBURNCK`)
    /// - 8 bytes version (u64 LE) = 2
    /// - 8 bytes step (u64 LE)
    /// - 8 bytes dt (f64 LE)
    /// - 1 byte integrator kind + 7 bytes padding
    /// - 4 bytes nsig_len (u32 LE) + 4 bytes padding
    /// - nsig_len * 4 bytes nsig_vec (f32 LE each)
    /// - 8 bytes number of subnetworks (u64 LE)
    /// - Per subnetwork: 8 bytes nvar, 8 bytes nnodes, 8 bytes nmodes (u64 LE)
    /// - Concatenated flat f32 LE state data for all subnetworks.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn checkpoint(&self, path: &str) -> Result<()> {
        use std::io::Write;

        let mut f = std::fs::File::create(path)?;

        // Header
        f.write_all(CKPT_MAGIC)?;
        f.write_all(&CKPT_VERSION.to_le_bytes())?;
        f.write_all(&(self.step as u64).to_le_bytes())?;
        f.write_all(&self.dt.to_le_bytes())?;

        let integrator_byte: u8 = match self.integrator {
            IntegratorKind::Heun => 1,
            IntegratorKind::Euler => 2,
            IntegratorKind::EulerStochastic => 3,
            IntegratorKind::HeunStochastic => 4,
            IntegratorKind::Rk4 => 5,
            IntegratorKind::Rk4Stochastic => 6,
        };
        f.write_all(&[integrator_byte, 0, 0, 0, 0, 0, 0, 0])?;
        let nsig_len = self.nsig.len() as u32;
        f.write_all(&nsig_len.to_le_bytes())?;
        f.write_all(&[0u8; 4])?; // padding
        for &val in &self.nsig {
            f.write_all(&val.to_le_bytes())?;
        }

        let n_subs = self.subnetworks.len() as u64;
        f.write_all(&n_subs.to_le_bytes())?;

        // Per-subnetwork metadata
        for sub in &self.subnetworks {
            f.write_all(&(sub.nvar as u64).to_le_bytes())?;
            f.write_all(&(sub.nnodes as u64).to_le_bytes())?;
            f.write_all(&(sub.nmodes as u64).to_le_bytes())?;
        }

        // Flat state data
        for state in &self.states {
            let (flat, _shape) = tensor_to_flat_f32(state.clone());
            for val in flat {
                f.write_all(&val.to_le_bytes())?;
            }
        }

        log::info!("Checkpoint saved to {} at step {}", path, self.step);
        Ok(())
    }

    /// Resume engine state from a checkpoint file.
    ///
    /// Not available in WASM builds (no filesystem access).
    ///
    /// Supports both v1 (scalar nsig) and v2 (per-variable nsig) formats.
    /// Verifies that the subnetwork shapes match the checkpoint metadata,
    /// then restores `step`, `dt`, `integrator`, `nsig`, and all states.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn resume(&mut self, path: &str) -> Result<()> {
        use std::io::Read;

        let mut buf = Vec::new();
        let mut f = std::fs::File::open(path)?;
        f.read_to_end(&mut buf)?;

        let mut offset = 8usize;

        macro_rules! read_u64 {
            () => {{
                let val = u64::from_le_bytes([
                    buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3],
                    buf[offset + 4], buf[offset + 5], buf[offset + 6], buf[offset + 7],
                ]);
                offset += 8;
                val
            }};
        }

        macro_rules! read_u32 {
            () => {{
                let val = u32::from_le_bytes([
                    buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3],
                ]);
                offset += 4;
                val
            }};
        }

        macro_rules! read_f64 {
            () => {{
                let val = f64::from_le_bytes([
                    buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3],
                    buf[offset + 4], buf[offset + 5], buf[offset + 6], buf[offset + 7],
                ]);
                offset += 8;
                val
            }};
        }

        macro_rules! read_f32 {
            () => {{
                let val = f32::from_le_bytes([
                    buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3],
                ]);
                offset += 4;
                val
            }};
        }

        macro_rules! check_remaining {
            ($n_bytes:expr) => {{
                if offset + $n_bytes > buf.len() {
                    return Err(SimulationError::InvalidState(
                        "Checkpoint data truncated: expected more bytes".into()
                    ));
                }
            }};
        }

        // Magic
        if buf.len() < 8 || &buf[0..8] != CKPT_MAGIC {
            return Err(SimulationError::InvalidState("Invalid checkpoint file (bad magic)".into()));
        }

        let version = read_u64!();

        let step = read_u64!() as usize;
        let dt = read_f64!();
        let integrator_byte = read_u64!() as u8;

        // Read nsig based on version
        let nsig_vec = if version == 1 {
            // v1: scalar nsig (f32) + 4 bytes padding
            let scalar_nsig = read_f32!();
            offset += 4; // padding
            vec![scalar_nsig]
        } else if version == 2 {
            // v2: nsig_len (u32) + 4 bytes padding + nsig_len * f32 values
            let nsig_len = read_u32!() as usize;
            offset += 4; // padding
            check_remaining!(nsig_len * 4);
            let mut v = Vec::with_capacity(nsig_len);
            for _ in 0..nsig_len {
                v.push(read_f32!());
            }
            v
        } else {
            return Err(SimulationError::InvalidState(format!(
                "Unsupported checkpoint version: {} (expected 1 or 2)",
                version
            )));
        };

        // v1 check was done above for version; for v2 we already accepted it
        if version != CKPT_VERSION && version != 1 {
            return Err(SimulationError::InvalidState(format!(
                "Unsupported checkpoint version: {} (expected {})",
                version, CKPT_VERSION
            )));
        }

        let n_subs = read_u64!() as usize;

        if n_subs != self.subnetworks.len() {
            return Err(SimulationError::InvalidState(format!(
                "Checkpoint has {} subnetworks, expected {}",
                n_subs, self.subnetworks.len()
            )));
        }

        let integrator = match integrator_byte {
            1 => IntegratorKind::Heun,
            2 => IntegratorKind::Euler,
            3 => IntegratorKind::EulerStochastic,
            4 => IntegratorKind::HeunStochastic,
            5 => IntegratorKind::Rk4,
            6 => IntegratorKind::Rk4Stochastic,
            _ => return Err(SimulationError::InvalidState(format!(
                "Unknown integrator kind in checkpoint: {}", integrator_byte
            ))),
        };

        let mut shapes = Vec::with_capacity(n_subs);
        check_remaining!(n_subs * 3 * 8); // 3 u64 per subnetwork: nvar, nnodes, nmodes
        for _ in 0..n_subs {
            let nvar = read_u64!() as usize;
            let nnodes = read_u64!() as usize;
            let nmodes = read_u64!() as usize;
            shapes.push((nvar, nnodes, nmodes));
        }

        // Verify shapes and read data
        let mut new_states = Vec::with_capacity(n_subs);
        for (i, sub) in self.subnetworks.iter().enumerate() {
            let (nvar, nnodes, nmodes) = shapes[i];
            if nvar != sub.nvar || nnodes != sub.nnodes || nmodes != sub.nmodes {
                return Err(SimulationError::InvalidState(format!(
                    "Checkpoint shape mismatch for subnetwork {}: expected (nvar={},nnodes={},nmodes={}), got ({},{},{})",
                    i, sub.nvar, sub.nnodes, sub.nmodes, nvar, nnodes, nmodes
                )));
            }

            let n_elements = nvar * nnodes * nmodes;
            check_remaining!(n_elements * 4); // f32 = 4 bytes per element
            let mut flat = vec![0.0f32; n_elements];
            for flat_j in flat.iter_mut().take(n_elements) {
                *flat_j = read_f32!();
            }

            let tensor = Tensor::<B, 3>::from_floats(
                TensorData::new::<f32, Vec<usize>>(flat, vec![nvar, nnodes, nmodes]),
                &self.device,
            );
            new_states.push(tensor);
        }

        self.states = new_states;
        self.step = step;
        self.dt = dt;
        self.integrator = integrator;
        self.nsig = nsig_vec;

        log::info!("Resumed from checkpoint {} at step {}", path, self.step);
        Ok(())
    }
}
