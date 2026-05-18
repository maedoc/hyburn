//! Monitors for recording simulation state.
//!
//! Provides monitors that capture the 3-D state tensor `[nvar, nnodes, nmodes]`
//! at each step, including subsampling, averaging, coupling, projection,
//! and a simplified BOLD haemodynamic model.
//!
//! GPU-accumulating monitors (`SpatialAverageMonitor`, `SensorProjectionMonitor`)
//! defer the GPU→CPU sync to `flush()` or period boundaries, avoiding a full
//! pipeline stall on every `record()` call.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};

/// Trait for all monitors.
///
/// Monitors observe the simulation state and accumulate data.
/// `flush` returns the accumulated data and clears the internal buffer.
pub trait Monitor<B: Backend> {
    /// Record the state at the given simulation step and time.
    fn record(&mut self, state: &Tensor<B, 3>, step: usize, t: f64);
    /// Finalize any partial accumulation and return all accumulated data.
    fn flush(&mut self) -> Vec<f32>;
    /// Sampling period (in steps). A period of 1 means every step.
    fn period(&self) -> usize;
}

// ============================================================
// RawMonitor
// ============================================================

/// Raw monitor — records every simulation step verbatim.
///
/// State shape per step: `[nvar, nnodes, nmodes]`.
pub struct RawMonitor {
    /// Flattened time-series data.
    pub data: Vec<f32>,
    /// Number of time steps recorded so far.
    pub nsteps: usize,
    nvar: usize,
    nnodes: usize,
    nmodes: usize,
}

impl RawMonitor {
    pub fn new(nvar: usize, nnodes: usize, nmodes: usize) -> Self {
        Self {
            data: Vec::new(),
            nsteps: 0,
            nvar,
            nnodes,
            nmodes,
        }
    }

    pub fn push<Bk: Backend>(&mut self, state: Tensor<Bk, 3>) {
        let (flat, shape) = crate::io::tensor_to_flat_f32(state);
        debug_assert_eq!(shape, vec![self.nvar, self.nnodes, self.nmodes]);
        self.data.extend_from_slice(&flat);
        self.nsteps += 1;
    }

    /// Return the full trajectory shape: `[nsteps, nvar, nnodes, nmodes]`
    pub fn trajectory_shape(&self) -> [usize; 4] {
        [self.nsteps, self.nvar, self.nnodes, self.nmodes]
    }

    /// Return accumulated data and clear the buffer.
    pub fn flush(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.data)
    }
}

impl<B: Backend> Monitor<B> for RawMonitor {
    fn record(&mut self, state: &Tensor<B, 3>, _step: usize, _t: f64) {
        self.push(state.clone());
    }

    fn flush(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.data)
    }

    fn period(&self) -> usize {
        1
    }
}

// ============================================================
// TemporalAverageMonitor
// ============================================================

/// Temporal-average monitor — accumulates state over `period` steps,
/// then stores the mean and resets.
///
/// This is useful for downsampling the output to a coarser time grid.
pub struct TemporalAverageMonitor {
    accumulator: Vec<f32>,
    count: usize,
    period: usize,
    /// Downsampled time-series data.
    pub data: Vec<f32>,
    /// Number of averaged snapshots stored so far.
    pub nobs: usize,
    nvar: usize,
    nnodes: usize,
    nmodes: usize,
}

impl TemporalAverageMonitor {
    pub fn new(nvar: usize, nnodes: usize, nmodes: usize, period: usize) -> Self {
        assert!(period > 0, "period must be > 0");
        Self {
            accumulator: vec![0.0; nvar * nnodes * nmodes],
            count: 0,
            period,
            data: Vec::new(),
            nobs: 0,
            nvar,
            nnodes,
            nmodes,
        }
    }

    pub fn push<Bk: Backend>(&mut self, _step: usize, state: Tensor<Bk, 3>) {
        let (flat, shape) = crate::io::tensor_to_flat_f32(state);
        debug_assert_eq!(shape, vec![self.nvar, self.nnodes, self.nmodes]);
        debug_assert_eq!(flat.len(), self.accumulator.len());

        for (a, v) in self.accumulator.iter_mut().zip(flat.iter()) {
            *a += *v;
        }
        self.count += 1;

        if self.count >= self.period {
            let inv = 1.0 / self.count as f32;
            for a in self.accumulator.iter_mut() {
                self.data.push(*a * inv);
                *a = 0.0;
            }
            self.count = 0;
            self.nobs += 1;
        }
    }

    /// Finalize any remaining partial accumulation and push to data.
    pub fn finalize(&mut self) {
        if self.count == 0 {
            return;
        }
        let inv = 1.0 / self.count as f32;
        for a in self.accumulator.iter_mut() {
            self.data.push(*a * inv);
            *a = 0.0;
        }
        self.count = 0;
        self.nobs += 1;
    }

    /// Finalize any remaining partial accumulation and return all data.
    pub fn flush(&mut self) -> Vec<f32> {
        self.finalize();
        std::mem::take(&mut self.data)
    }

    /// Return the downsampled trajectory shape: `[nobs, nvar, nnodes, nmodes]`
    pub fn trajectory_shape(&self) -> [usize; 4] {
        [self.nobs, self.nvar, self.nnodes, self.nmodes]
    }
}

impl<B: Backend> Monitor<B> for TemporalAverageMonitor {
    fn record(&mut self, state: &Tensor<B, 3>, step: usize, _t: f64) {
        self.push(step, state.clone());
    }

    fn flush(&mut self) -> Vec<f32> {
        self.finalize();
        std::mem::take(&mut self.data)
    }

    fn period(&self) -> usize {
        self.period
    }
}

// ============================================================
// SubSampleMonitor
// ============================================================

/// Sub-sample monitor — records every `period`-th simulation step verbatim.
pub struct SubSampleMonitor {
    /// Flattened time-series data.
    pub data: Vec<f32>,
    /// Number of recorded snapshots.
    pub nsteps: usize,
    nvar: usize,
    nnodes: usize,
    nmodes: usize,
    period: usize,
}

impl SubSampleMonitor {
    pub fn new(nvar: usize, nnodes: usize, nmodes: usize, period: usize) -> Self {
        assert!(period > 0, "period must be > 0");
        Self {
            data: Vec::new(),
            nsteps: 0,
            nvar,
            nnodes,
            nmodes,
            period,
        }
    }

    pub fn trajectory_shape(&self) -> [usize; 4] {
        [self.nsteps, self.nvar, self.nnodes, self.nmodes]
    }

    /// Return accumulated data and clear the buffer.
    pub fn flush(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.data)
    }
}

impl<B: Backend> Monitor<B> for SubSampleMonitor {
    fn record(&mut self, state: &Tensor<B, 3>, step: usize, _t: f64) {
        if step.is_multiple_of(self.period) {
            let (flat, shape) = crate::io::tensor_to_flat_f32(state.clone());
            debug_assert_eq!(shape, vec![self.nvar, self.nnodes, self.nmodes]);
            self.data.extend_from_slice(&flat);
            self.nsteps += 1;
        }
    }

    fn flush(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.data)
    }

    fn period(&self) -> usize {
        self.period
    }
}

// ============================================================
// GlobalAverageMonitor
// ============================================================

/// Global-average monitor — averages over the spatial (`nnodes`) dimension.
///
/// Output per recorded step has conceptual shape `[nvar, 1, nmodes]`.
///
/// On GPU backends, `record()` accumulates the per-step average into a
/// device-side tensor; the GPU→CPU sync only happens when a period
/// boundary is reached or `flush()` is called.
pub struct GlobalAverageMonitor<B: Backend> {
    pub data: Vec<f32>,
    pub nsteps: usize,
    nvar: usize,
    _nnodes: usize,
    nmodes: usize,
    period: usize,
    accumulator: Option<Tensor<B, 1>>,
    accumulator_count: usize,
    _inv_nnodes: f32,
}

impl<B: Backend> GlobalAverageMonitor<B> {
    pub fn new(nvar: usize, nnodes: usize, nmodes: usize) -> Self {
        Self {
            data: Vec::new(),
            nsteps: 0,
            nvar,
            _nnodes: nnodes,
            nmodes,
            period: 1,
            accumulator: None,
            accumulator_count: 0,
            _inv_nnodes: 1.0 / nnodes as f32,
        }
    }

    pub fn with_period(nvar: usize, nnodes: usize, nmodes: usize, period: usize) -> Self {
        Self {
            data: Vec::new(),
            nsteps: 0,
            nvar,
            _nnodes: nnodes,
            nmodes,
            period,
            accumulator: None,
            accumulator_count: 0,
            _inv_nnodes: 1.0 / nnodes as f32,
        }
    }

    pub fn trajectory_shape(&self) -> [usize; 4] {
        [self.nsteps, self.nvar, 1, self.nmodes]
    }

    fn drain_accumulator(&mut self) {
        if self.accumulator_count == 0 {
            return;
        }
        if let Some(ref acc) = self.accumulator {
            let avg = acc.clone().div_scalar(self.accumulator_count as f32);
            let (flat, _shape) = crate::io::tensor_to_flat_f32::<B, 1>(avg);
            self.data.extend_from_slice(&flat);
        }
        self.accumulator = None;
        self.accumulator_count = 0;
    }
}

impl<B: Backend> Monitor<B> for GlobalAverageMonitor<B> {
    fn record(&mut self, state: &Tensor<B, 3>, _step: usize, _t: f64) {
        // state shape: [nvar, nnodes, nmodes]
        // Average over nnodes (dim 1): result shape [nvar, 1, nmodes] → squeeze → [nvar, nmodes]
        // Then flatten to [nvar * nmodes] for accumulation.
        let avg = state.clone()
            .mean_dim(1)
            .squeeze::<2>(1)
            .reshape([self.nvar * self.nmodes]);

        if self.accumulator.is_none() {
            self.accumulator = Some(avg);
        } else if let Some(ref mut acc) = self.accumulator {
            *acc = acc.clone() + avg;
        }
        self.accumulator_count += 1;

        if self.accumulator_count >= self.period {
            self.drain_accumulator();
            self.nsteps += 1;
        }
    }

    fn flush(&mut self) -> Vec<f32> {
        self.drain_accumulator();
        std::mem::take(&mut self.data)
    }

    fn period(&self) -> usize {
        self.period
    }
}

// ============================================================
// AfferentCouplingMonitor
// ============================================================

/// Afferent coupling monitor — records the coupling input arriving at each node.
///
/// Because the [`Monitor`] trait only receives the state tensor, the caller must
/// supply the coupling vector **before** calling `record` via
/// [`set_coupling`](Self::set_coupling).
/// If no coupling was set for a step, that step is silently skipped.
pub struct AfferentCouplingMonitor {
    /// Flattened coupling time-series data.
    pub data: Vec<f32>,
    /// Number of recorded snapshots.
    pub nsteps: usize,
    nnodes: usize,
    ncvar: usize,
    current_coupling: Option<Vec<f32>>,
}

impl AfferentCouplingMonitor {
    pub fn new(nnodes: usize, ncvar: usize) -> Self {
        Self {
            data: Vec::new(),
            nsteps: 0,
            nnodes,
            ncvar,
            current_coupling: None,
        }
    }

    /// Provide the coupling vector for the upcoming `record` call.
    ///
    /// `coupling` is typically a flattened tensor of shape
    /// `[nnodes * nmodes, ncvar]`.
    pub fn set_coupling(&mut self, coupling: Vec<f32>) {
        self.current_coupling = Some(coupling);
    }

    pub fn trajectory_shape(&self, nmodes: usize) -> [usize; 3] {
        [self.nsteps, self.nnodes * nmodes, self.ncvar]
    }

    /// Return accumulated data and clear the buffer.
    pub fn flush(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.data)
    }
}

impl<B: Backend> Monitor<B> for AfferentCouplingMonitor {
    fn record(&mut self, _state: &Tensor<B, 3>, _step: usize, _t: f64) {
        if let Some(coupling) = self.current_coupling.take() {
            self.data.extend_from_slice(&coupling);
            self.nsteps += 1;
        }
    }

    fn flush(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.data)
    }

    fn period(&self) -> usize {
        1
    }
}

// ============================================================
// ProjectionMonitor
// ============================================================

/// Projection monitor — records the projection activity between subnetworks.
///
/// Similar to [`AfferentCouplingMonitor`], the actual projection data must be
/// injected before `record` via [`set_activity`](Self::set_activity).
pub struct ProjectionMonitor {
    /// Flattened projection-activity time-series data.
    pub data: Vec<f32>,
    /// Number of recorded snapshots.
    pub nsteps: usize,
    current: Option<Vec<f32>>,
}

impl ProjectionMonitor {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            nsteps: 0,
            current: None,
        }
    }

    /// Provide the projection activity for the upcoming `record` call.
    pub fn set_activity(&mut self, activity: Vec<f32>) {
        self.current = Some(activity);
    }

    /// Return accumulated data and clear the buffer.
    pub fn flush(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.data)
    }
}

impl Default for ProjectionMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Monitor<B> for ProjectionMonitor {
    fn record(&mut self, _state: &Tensor<B, 3>, _step: usize, _t: f64) {
        if let Some(activity) = self.current.take() {
            self.data.extend_from_slice(&activity);
            self.nsteps += 1;
        }
    }

    fn flush(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.data)
    }

    fn period(&self) -> usize {
        1
    }
}

// ============================================================
// SensorProjectionMonitor
// ============================================================

/// Sensor projection monitor — projects neural activity through a gain
/// matrix to produce sensor-space signals (EEG/MEG/iEEG).
///
/// For each recorded step, the monitor:
/// 1. Extracts variables of interest (`voi`) from the state tensor
///    `[nvar, nnodes, nmodes]`, averaging over modes.
/// 2. Multiplies the per-variable node activity by the gain matrix
///    `[n_sensors, n_regions]` to get sensor-space signals.
/// 3. Accumulates over `period` steps and stores temporal averages.
///
/// On GPU backends, steps 1–2 are performed as device-side tensor
/// operations and the result is accumulated in a `Tensor<B, 1>`.
/// The GPU→CPU sync only occurs at period boundaries or in `flush()`.
///
/// Output per averaged time point: `[n_sensors * n_voi]` values, ordered
/// as `[sensors_for_voi0..., sensors_for_voi1...]`.
pub struct SensorProjectionMonitor<B: Backend> {
    /// Gain matrix `[n_sensors, n_regions]` as a device-side tensor.
    gain_tensor: Tensor<B, 2>,
    /// Indices of variables of interest (0-based into nvar dimension).
    pub voi: Vec<usize>,
    /// Temporal averaging period in simulation steps.
    pub period: usize,
    /// GPU-side accumulator for the current temporal-averaging window.
    /// Shape: `[n_sensors * nvar]` (all variables; voi selected at flush).
    accumulator: Option<Tensor<B, 1>>,
    /// Number of steps accumulated in the current window.
    accumulator_count: usize,
    /// Recorded data (temporal averages, voi-selected, CPU-side).
    pub data: Vec<f32>,
    /// Number of sensors.
    pub n_sensors: usize,
    /// Number of regions (nodes).
    pub n_regions: usize,
    /// Number of state variables.
    nvar: usize,
    _nmodes: usize,
    _device: B::Device,
    _inv_nmodes: f32,
}

impl<B: Backend> SensorProjectionMonitor<B> {
    pub fn new(
        gain: Vec<Vec<f32>>,
        voi: Vec<usize>,
        period: usize,
        nvar: usize,
        n_regions: usize,
        nmodes: usize,
        device: &B::Device,
    ) -> Self {
        assert!(period > 0, "period must be > 0");
        let n_sensors = gain.len();
        for row in &gain {
            assert_eq!(row.len(), n_regions, "gain row length {} != n_regions {}", row.len(), n_regions);
        }
        for &vi in &voi {
            assert!(vi < nvar, "voi index {} exceeds nvar {} (max allowed: {})", vi, nvar, nvar - 1);
        }

        // Flatten gain [n_sensors][n_regions] → row-major [n_sensors * n_regions]
        let gain_flat: Vec<f32> = gain.iter().flat_map(|r| r.iter().copied()).collect();
        let gain_tensor = Tensor::<B, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(gain_flat, vec![n_sensors, n_regions]),
            device,
        );

        let inv_nmodes = 1.0 / nmodes.max(1) as f32;

        Self {
            gain_tensor,
            voi,
            period,
            accumulator: None,
            accumulator_count: 0,
            data: Vec::new(),
            n_sensors,
            n_regions,
            nvar,
            _nmodes: nmodes,
            _device: device.clone(),
            _inv_nmodes: inv_nmodes,
        }
    }

    /// Compute sensor signal on GPU from a state tensor `[nvar, nnodes, nmodes]`.
    ///
    /// Returns a 1D tensor of shape `[n_sensors * nvar]`, ordered as
    /// `[sensors_for_var0, sensors_for_var1, ...]` (row-major of `[nvar, n_sensors]`).
    fn compute_sensor_signal_gpu(&self, state: &Tensor<B, 3>) -> Tensor<B, 1> {
        // 1. Average over modes: [nvar, nnodes, nmodes] → [nvar, nnodes]
        let state_mean = state.clone().mean_dim(2).squeeze::<2>(2);

        // 2. Transpose to [nnodes, nvar] for matmul with gain
        let activity = state_mean.permute([1, 0]); // [n_regions, nvar]

        // 3. gain_tensor: [n_sensors, n_regions] @ activity: [n_regions, nvar]
        //    = [n_sensors, nvar]
        let sensor_all = self.gain_tensor.clone().matmul(activity); // [n_sensors, nvar]

        // 4. Transpose to [nvar, n_sensors] so flattening gives
        //    [sensors_for_var0, sensors_for_var1, ...] matching the expected voi-first order
        let sensor_voi_order = sensor_all.permute([1, 0]); // [nvar, n_sensors]

        sensor_voi_order.reshape([self.nvar * self.n_sensors])
    }

    fn drain_accumulator(&mut self) {
        if self.accumulator_count == 0 {
            return;
        }
        if let Some(ref acc) = self.accumulator {
            let avg = acc.clone().div_scalar(self.accumulator_count as f32);
            let (flat, _shape) = crate::io::tensor_to_flat_f32::<B, 1>(avg);

            // flat is [nvar, n_sensors] row-major, i.e.
            // [var0_sensor0, ..., var0_sensorN, var1_sensor0, ..., varK_sensorN]
            // Extract voi indices: for voi vi, the slice is [vi * n_sensors .. (vi+1) * n_sensors]
            let n_sensors = self.n_sensors;
            for &vi in &self.voi {
                let start = vi * n_sensors;
                let end = start + n_sensors;
                if end <= flat.len() {
                    self.data.extend_from_slice(&flat[start..end]);
                }
            }
        }
        self.accumulator = None;
        self.accumulator_count = 0;
    }
}

impl<B: Backend> Monitor<B> for SensorProjectionMonitor<B> {
    fn record(&mut self, state: &Tensor<B, 3>, _step: usize, _t: f64) {
        let signal = self.compute_sensor_signal_gpu(state);

        if self.accumulator.is_none() {
            self.accumulator = Some(signal);
        } else if let Some(ref mut acc) = self.accumulator {
            *acc = acc.clone() + signal;
        }
        self.accumulator_count += 1;

        if self.accumulator_count >= self.period {
            self.drain_accumulator();
        }
    }

    fn flush(&mut self) -> Vec<f32> {
        self.drain_accumulator();
        std::mem::take(&mut self.data)
    }

    fn period(&self) -> usize {
        self.period
    }
}

// ============================================================
// SpatialAverageMonitor
// ============================================================

/// Spatial average monitor — computes a weighted spatial average of selected
/// state variables over nodes using a mask vector.
///
/// For each recorded step and each variable of interest `vi`:
/// `output[vi] = sum_n(mask[n] * state[vi,n,:]) / sum_n(mask[n])`
/// where modes are averaged out.
///
/// On GPU backends, the spatial average is computed as a device-side tensor
/// operation and accumulated in a `Tensor<B, 1>`. The GPU→CPU sync only
/// occurs at period boundaries or in `flush()`.
///
/// If `voi` is empty, all state variables are monitored.
/// If `mask` is all ones, this reduces to a uniform spatial average.
pub struct SpatialAverageMonitor<B: Backend> {
    /// Weighting mask `[n_regions]` as a device-side tensor.
    mask_tensor: Tensor<B, 1>,
    /// Temporal averaging period in simulation steps.
    pub period: usize,
    /// GPU-side accumulator for the current temporal-averaging window: `[nvar]`.
    accumulator: Option<Tensor<B, 1>>,
    /// Number of steps accumulated in the current window.
    accumulator_count: usize,
    /// Recorded data (temporal averages, voi-selected, CPU-side).
    pub data: Vec<f32>,
    /// Number of state variables.
    _nvar: usize,
    _n_regions: usize,
    _nmodes: usize,
    mask_sum: f32,
    pub voi: Vec<usize>,
    _device: B::Device,
}

impl<B: Backend> SpatialAverageMonitor<B> {
    pub fn new(
        mask: Vec<f32>,
        voi: Vec<usize>,
        period: usize,
        nvar: usize,
        n_regions: usize,
        nmodes: usize,
        device: &B::Device,
    ) -> Self {
        assert!(period > 0, "period must be > 0");
        let mask_sum: f32 = mask.iter().sum();
        assert!(mask_sum.abs() > 1e-12, "spatial mask sum must be non-zero (got {})", mask_sum);
        for &vi in &voi {
            assert!(vi < nvar, "voi index {} exceeds nvar {} (max allowed: {})", vi, nvar, nvar - 1);
        }
        let voi = if voi.is_empty() { (0..nvar).collect() } else { voi };

        let mask_tensor = Tensor::<B, 1>::from_floats(
            TensorData::new::<f32, Vec<usize>>(mask, vec![n_regions]),
            device,
        );

        Self {
            mask_tensor,
            period,
            accumulator: None,
            accumulator_count: 0,
            data: Vec::new(),
            _nvar: nvar,
            _n_regions: n_regions,
            _nmodes: nmodes,
            mask_sum,
            voi,
            _device: device.clone(),
        }
    }

    /// Compute spatial averages on GPU from a state tensor `[nvar, nnodes, nmodes]`.
    ///
    /// Returns a 1D tensor of shape `[nvar]` where each element is the
    /// mask-weighted spatial average of that variable (modes averaged out).
    fn compute_spatial_average_gpu(&self, state: &Tensor<B, 3>) -> Tensor<B, 1> {
        // 1. Average over modes: [nvar, nnodes, nmodes] → [nvar, nnodes]
        let state_mean = state.clone().mean_dim(2).squeeze::<2>(2);

        // 2. Multiply by mask: mask_tensor [n_regions] → reshape to [1, n_regions]
        //    Broadcast multiply: [nvar, n_regions] * [1, n_regions] → [nvar, n_regions]
        let mask_2d = self.mask_tensor.clone().unsqueeze_dim::<2>(0); // [1, n_regions]
        let masked = state_mean.mul(mask_2d); // [nvar, n_regions]

        // 3. Sum over nodes: [nvar, n_regions] → [nvar, 1] → squeeze → [nvar]
        let spatial_sum = masked.sum_dim(1).squeeze::<1>(1); // [nvar]

        // 4. Normalize by mask_sum
        spatial_sum.div_scalar(self.mask_sum)
    }

    fn drain_accumulator(&mut self) {
        if self.accumulator_count == 0 {
            return;
        }
        if let Some(ref acc) = self.accumulator {
            let avg = acc.clone().div_scalar(self.accumulator_count as f32);
            let (flat, _shape) = crate::io::tensor_to_flat_f32::<B, 1>(avg);

            // flat is [nvar], select voi indices
            for &vi in &self.voi {
                if vi < flat.len() {
                    self.data.push(flat[vi]);
                }
            }
        }
        self.accumulator = None;
        self.accumulator_count = 0;
    }
}

impl<B: Backend> Monitor<B> for SpatialAverageMonitor<B> {
    fn record(&mut self, state: &Tensor<B, 3>, _step: usize, _t: f64) {
        let avg = self.compute_spatial_average_gpu(state);

        if self.accumulator.is_none() {
            self.accumulator = Some(avg);
        } else if let Some(ref mut acc) = self.accumulator {
            *acc = acc.clone() + avg;
        }
        self.accumulator_count += 1;

        if self.accumulator_count >= self.period {
            self.drain_accumulator();
        }
    }

    fn flush(&mut self) -> Vec<f32> {
        self.drain_accumulator();
        std::mem::take(&mut self.data)
    }

    fn period(&self) -> usize {
        self.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_state(data: Vec<f32>, shape: Vec<usize>) -> Tensor<B, 3> {
        Tensor::from_floats(
            TensorData::new::<f32, Vec<usize>>(data, shape),
            &Default::default(),
        )
    }

    #[test]
    fn test_raw_monitor() {
        let mut m = RawMonitor::new(2, 1, 1);
        let s1 = make_state(vec![1.0, 2.0], vec![2, 1, 1]);
        let s2 = make_state(vec![3.0, 4.0], vec![2, 1, 1]);
        m.push(s1);
        m.push(s2);
        assert_eq!(m.nsteps, 2);
        assert_eq!(m.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m.trajectory_shape(), [2, 2, 1, 1]);
    }

    #[test]
    fn test_temporal_average_monitor() {
        let mut m = TemporalAverageMonitor::new(2, 1, 1, 2);
        let s1 = make_state(vec![1.0, 2.0], vec![2, 1, 1]);
        let s2 = make_state(vec![3.0, 4.0], vec![2, 1, 1]);
        let s3 = make_state(vec![5.0, 6.0], vec![2, 1, 1]);

        m.push(0, s1);
        m.push(1, s2);
        // after 2 steps, average = [(1+3)/2, (2+4)/2] = [2.0, 3.0]
        assert_eq!(m.nobs, 1);
        assert_eq!(m.data, vec![2.0, 3.0]);

        m.push(2, s3);
        // not yet flushed, accumulator holds [5.0, 6.0]
        assert_eq!(m.nobs, 1);

        let flushed = m.flush();
        assert_eq!(m.nobs, 2);
        assert_eq!(flushed, vec![2.0, 3.0, 5.0, 6.0]);
        assert_eq!(m.trajectory_shape(), [2, 2, 1, 1]);
    }

    #[test]
    fn test_temporal_average_period_3() {
        let mut m = TemporalAverageMonitor::new(1, 1, 1, 3);
        for i in 0..6 {
            let s = make_state(vec![(i + 1) as f32], vec![1, 1, 1]);
            m.push(i, s);
        }
        // obs 0: (1+2+3)/3 = 2.0
        // obs 1: (4+5+6)/3 = 5.0
        assert_eq!(m.nobs, 2);
        assert_eq!(m.data, vec![2.0, 5.0]);
    }

    #[test]
    fn test_subsample_monitor() {
        let mut m = SubSampleMonitor::new(2, 1, 1, 3);
        let states: Vec<Tensor<B, 3>> = (0..7)
            .map(|i| make_state(vec![i as f32, (i + 10) as f32], vec![2, 1, 1]))
            .collect();

        for (step, state) in states.iter().enumerate() {
            m.record(state, step, 0.0);
        }
        // steps 0, 3, 6 are recorded -> 3 snapshots
        assert_eq!(m.nsteps, 3);
        assert_eq!(
            m.data,
            vec![
                0.0, 10.0, // step 0
                3.0, 13.0, // step 3
                6.0, 16.0, // step 6
            ]
        );
        assert_eq!(m.trajectory_shape(), [3, 2, 1, 1]);
    }

    #[test]
    fn test_global_average_monitor() {
        let mut m = GlobalAverageMonitor::<B>::new(2, 3, 1);
        // state [nvar=2, nnodes=3, nmodes=1]:
        // var0: [1, 2, 3]
        // var1: [4, 5, 6]
        let s = make_state(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3, 1]);
        m.record(&s, 0, 0.0);
        // averages: var0 -> 2.0, var1 -> 5.0
        let flushed = m.flush();
        assert_eq!(flushed.len(), 2);
        assert!((flushed[0] - 2.0).abs() < 1e-5);
        assert!((flushed[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_global_average_monitor_with_modes() {
        let mut m = GlobalAverageMonitor::<B>::new(2, 2, 3);
        // state [nvar=2, nnodes=2, nmodes=3]:
        // var0: [[1,2,3],[4,5,6]]
        // var1: [[7,8,9],[10,11,12]]
        let s = make_state(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![2, 2, 3],
        );
        m.record(&s, 0, 0.0);
        // var0 node averages: node0=(1+2+3)/2.5 not right... it's mean over nnodes
        // For [nvar, nnodes, nmodes]: mean over nnodes (dim 1)
        // var0 node0 mean_m = 2.0, node1 mean_m = 5.0 → mean over nnodes = 3.5
        // Wait, mean_dim(1) averages over nnodes dimension giving [nvar, 1, nmodes]
        // var0: [[1,2,3],[4,5,6]] → mean over nnodes → [[2.5, 3.5, 4.5]] → [nvar=2,1,nmodes=3]
        // var1: [[7,8,9],[10,11,12]] → mean over nnodes → [[8.5, 9.5, 10.5]]
        // Flatten: [2.5, 3.5, 4.5, 8.5, 9.5, 10.5]
        let flushed = m.flush();
        assert_eq!(flushed.len(), 6);
        assert!((flushed[0] - 2.5).abs() < 1e-4);
        assert!((flushed[1] - 3.5).abs() < 1e-4);
        assert!((flushed[2] - 4.5).abs() < 1e-4);
        assert!((flushed[3] - 8.5).abs() < 1e-4);
        assert!((flushed[4] - 9.5).abs() < 1e-4);
        assert!((flushed[5] - 10.5).abs() < 1e-4);
    }

    #[test]
    fn test_afferent_coupling_monitor() {
        let mut m = AfferentCouplingMonitor::new(2, 2);
        let s = make_state(vec![0.0; 4], vec![1, 2, 2]);
        // no coupling set -> skip
        m.record(&s, 0, 0.0);
        assert_eq!(m.nsteps, 0);
        assert!(m.data.is_empty());

        m.set_coupling(vec![0.5, 0.6, 0.7, 0.8]);
        m.record(&s, 1, 0.1);
        assert_eq!(m.nsteps, 1);
        assert_eq!(m.data, vec![0.5, 0.6, 0.7, 0.8]);
    }

    #[test]
    fn test_projection_monitor() {
        let mut m = ProjectionMonitor::new();
        let s = make_state(vec![0.0; 4], vec![1, 2, 2]);
        m.set_activity(vec![1.0, 2.0]);
        m.record(&s, 0, 0.0);
        assert_eq!(m.nsteps, 1);
        assert_eq!(m.data, vec![1.0, 2.0]);
    }

    #[test]
    fn test_sensor_projection_monitor() {
        let dev: <B as Backend>::Device = Default::default();
        // 2 sensors, 3 regions, nvar=2, nmodes=1, voi=[0]
        // Gain: sensor 0 sums all regions, sensor 1 takes difference
        let gain = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, -1.0, 0.0],
        ];
        let voi = vec![0];
        let mut m = SensorProjectionMonitor::<B>::new(gain, voi, 1, 2, 3, 1, &dev);

        // state: [nvar=2, nnodes=3, nmodes=1]
        // var0 = [1, 2, 3], var1 = [4, 5, 6]
        let s = make_state(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3, 1]);
        <SensorProjectionMonitor<B> as Monitor<B>>::record(&mut m, &s, 0, 0.0);

        // sensor0 (var0): 1*1 + 1*2 + 1*3 = 6.0
        // sensor1 (var0): 1*1 + (-1)*2 + 0*3 = -1.0
        let flushed = <SensorProjectionMonitor<B> as Monitor<B>>::flush(&mut m);
        assert_eq!(flushed.len(), 2);
        assert!((flushed[0] - 6.0).abs() < 1e-4, "sensor0 = {}", flushed[0]);
        assert!((flushed[1] - (-1.0)).abs() < 1e-4, "sensor1 = {}", flushed[1]);
    }

    #[test]
    fn test_sensor_projection_monitor_multiple_voi() {
        let dev: <B as Backend>::Device = Default::default();
        // 1 sensor, 2 regions, nvar=2, nmodes=1, voi=[0, 1]
        let gain = vec![vec![1.0, 1.0]];
        let voi = vec![0, 1];
        let mut m = SensorProjectionMonitor::<B>::new(gain, voi, 2, 2, 2, 1, &dev);

        // state step 0: var0=[1,2], var1=[3,4]
        let s0 = make_state(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2, 1]);
        // state step 1: var0=[5,6], var1=[7,8]
        let s1 = make_state(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2, 1]);

        <SensorProjectionMonitor<B> as Monitor<B>>::record(&mut m, &s0, 0, 0.0);
        <SensorProjectionMonitor<B> as Monitor<B>>::record(&mut m, &s1, 1, 0.1);

        // After period=2, temporal average:
        // voi=0: sensor = mean((1+2), (5+6)) = mean(3, 11) = 7.0
        // voi=1: sensor = mean((3+4), (7+8)) = mean(7, 15) = 11.0
        let flushed = <SensorProjectionMonitor<B> as Monitor<B>>::flush(&mut m);
        assert_eq!(flushed.len(), 2);
        assert!((flushed[0] - 7.0).abs() < 1e-4, "voi0 sensor = {}", flushed[0]);
        assert!((flushed[1] - 11.0).abs() < 1e-4, "voi1 sensor = {}", flushed[1]);
    }

    #[test]
    fn test_sensor_projection_monitor_with_modes() {
        let dev: <B as Backend>::Device = Default::default();
        // 1 sensor, 2 regions, nvar=1, nmodes=2, voi=[0]
        let gain = vec![vec![1.0, 1.0]];
        let voi = vec![0];
        let mut m = SensorProjectionMonitor::<B>::new(gain, voi, 1, 1, 2, 2, &dev);

        // state: [nvar=1, nnodes=2, nmodes=2]
        // var0: node0=[10,20], node1=[30,40]
        // After mode averaging: node0=15, node1=35
        // sensor: 1*15 + 1*35 = 50
        let s = make_state(vec![10.0, 20.0, 30.0, 40.0], vec![1, 2, 2]);
        <SensorProjectionMonitor<B> as Monitor<B>>::record(&mut m, &s, 0, 0.0);

        let flushed = <SensorProjectionMonitor<B> as Monitor<B>>::flush(&mut m);
        assert_eq!(flushed.len(), 1);
        assert!((flushed[0] - 50.0).abs() < 1e-4, "sensor = {}", flushed[0]);
    }

    #[test]
    fn test_spatial_average_monitor() {
        let dev: <B as Backend>::Device = Default::default();
        let mask = vec![1.0, 1.0, 1.0];
        let mut m = SpatialAverageMonitor::<B>::new(mask, vec![], 1, 2, 3, 1, &dev);

        // state: var0=[1,2,3], var1=[4,5,6]
        let s = make_state(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3, 1]);
        <SpatialAverageMonitor<B> as Monitor<B>>::record(&mut m, &s, 0, 0.0);

        let flushed = <SpatialAverageMonitor<B> as Monitor<B>>::flush(&mut m);
        // var0 avg = (1+2+3)/3 = 2.0, var1 avg = (4+5+6)/3 = 5.0
        assert_eq!(flushed.len(), 2);
        assert!((flushed[0] - 2.0).abs() < 1e-5);
        assert!((flushed[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_spatial_average_monitor_weighted() {
        let dev: <B as Backend>::Device = Default::default();
        // Weighted mask, nvar=1, n_regions=2, nmodes=1
        let mask = vec![2.0, 3.0];
        let mut m = SpatialAverageMonitor::<B>::new(mask, vec![], 1, 1, 2, 1, &dev);

        // state: var0=[4, 6]
        let s = make_state(vec![4.0, 6.0], vec![1, 2, 1]);
        <SpatialAverageMonitor::<B> as Monitor<B>>::record(&mut m, &s, 0, 0.0);

        let flushed = <SpatialAverageMonitor::<B> as Monitor<B>>::flush(&mut m);
        // var0 avg = (2*4 + 3*6)/(2+3) = (8+18)/5 = 26/5 = 5.2
        assert_eq!(flushed.len(), 1);
        assert!((flushed[0] - 5.2).abs() < 1e-5);
    }

    #[test]
    fn test_spatial_average_monitor_with_modes() {
        let dev: <B as Backend>::Device = Default::default();
        // Uniform mask, nvar=1, n_regions=2, nmodes=2
        let mask = vec![1.0, 1.0];
        let mut m = SpatialAverageMonitor::<B>::new(mask, vec![], 2, 1, 2, 2, &dev);

        // state: [nvar=1, nnodes=2, nmodes=2]
        // var0: node0=[10,20], node1=[30,40]
        let s = make_state(vec![10.0, 20.0, 30.0, 40.0], vec![1, 2, 2]);
        <SpatialAverageMonitor::<B> as Monitor<B>>::record(&mut m, &s, 0, 0.0);

        let flushed = <SpatialAverageMonitor::<B> as Monitor<B>>::flush(&mut m);
        // var0: node0 avg over modes = (10+20)/2=15, node1 avg = (30+40)/2=35
        // spatial avg = (15+35)/2 = 25.0
        assert_eq!(flushed.len(), 1);
        assert!((flushed[0] - 25.0).abs() < 1e-5);
    }

    #[test]
    fn test_spatial_average_monitor_temporal_period() {
        let dev: <B as Backend>::Device = Default::default();
        // Uniform mask, nvar=1, n_regions=2, nmodes=1, period=2
        let mask = vec![1.0, 1.0];
        let mut m = SpatialAverageMonitor::<B>::new(mask, vec![], 2, 1, 2, 1, &dev);

        let s1 = make_state(vec![2.0, 4.0], vec![1, 2, 1]); // avg = 3.0
        let s2 = make_state(vec![4.0, 8.0], vec![1, 2, 1]); // avg = 6.0
        <SpatialAverageMonitor::<B> as Monitor<B>>::record(&mut m, &s1, 0, 0.0);
        <SpatialAverageMonitor::<B> as Monitor<B>>::record(&mut m, &s2, 1, 0.1);

        // After period=2: temporal average = (3.0 + 6.0) / 2 = 4.5
        let flushed = <SpatialAverageMonitor::<B> as Monitor<B>>::flush(&mut m);
        assert_eq!(flushed.len(), 1);
        assert!((flushed[0] - 4.5).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "spatial mask sum must be non-zero")]
    fn test_spatial_average_monitor_zero_mask_panics() {
        let dev: <B as Backend>::Device = Default::default();
        SpatialAverageMonitor::<B>::new(vec![0.0, 0.0], vec![], 1, 1, 2, 1, &dev);
    }

    #[test]
    #[should_panic(expected = "voi index")]
    fn test_spatial_average_monitor_voi_out_of_bounds() {
        let dev: <B as Backend>::Device = Default::default();
        SpatialAverageMonitor::<B>::new(vec![1.0, 1.0], vec![5], 1, 2, 2, 1, &dev);
    }

    #[test]
    fn test_monitor_trait_object_safety() {
        // Verify that monitors can be used as trait objects.
        let dev: <B as Backend>::Device = Default::default();
        let gain = vec![vec![1.0]];
        let mask = vec![1.0];
        let mut monitors: Vec<Box<dyn Monitor<B>>> = vec![
            Box::new(RawMonitor::new(1, 1, 1)),
            Box::new(TemporalAverageMonitor::new(1, 1, 1, 2)),
            Box::new(SubSampleMonitor::new(1, 1, 1, 2)),
            Box::new(GlobalAverageMonitor::<B>::new(1, 1, 1)),
            Box::new(AfferentCouplingMonitor::new(1, 1)),
            Box::new(ProjectionMonitor::new()),
            Box::new(SensorProjectionMonitor::<B>::new(gain, vec![0], 1, 1, 1, 1, &dev)),
            Box::new(SpatialAverageMonitor::<B>::new(mask, vec![], 1, 1, 1, 1, &dev)),
        ];
        let s = make_state(vec![1.0], vec![1, 1, 1]);
        for (step, monitor) in monitors.iter_mut().enumerate() {
            monitor.record(&s, step, step as f64 * 0.1);
        }
        for monitor in monitors.iter_mut() {
            let d = monitor.flush();
            let _ = d.len();
        }
    }
}