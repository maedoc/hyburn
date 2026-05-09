//! Monitors for recording simulation state.
//!
//! Provides monitors that capture the 3-D state tensor `[nvar, nnodes, nmodes]`
//! at each step, including subsampling, averaging, coupling, projection,
//! and a simplified BOLD haemodynamic model.

use burn::prelude::Backend;
use burn::tensor::Tensor;

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
pub struct GlobalAverageMonitor {
    /// Averaged time-series data.
    pub data: Vec<f32>,
    /// Number of recorded snapshots.
    pub nsteps: usize,
    nvar: usize,
    nnodes: usize,
    nmodes: usize,
}

impl GlobalAverageMonitor {
    pub fn new(nvar: usize, nnodes: usize, nmodes: usize) -> Self {
        Self {
            data: Vec::new(),
            nsteps: 0,
            nvar,
            nnodes,
            nmodes,
        }
    }

    pub fn trajectory_shape(&self) -> [usize; 4] {
        [self.nsteps, self.nvar, 1, self.nmodes]
    }

    /// Return accumulated data and clear the buffer.
    pub fn flush(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.data)
    }
}

impl<B: Backend> Monitor<B> for GlobalAverageMonitor {
    fn record(&mut self, state: &Tensor<B, 3>, _step: usize, _t: f64) {
        let (flat, shape) = crate::io::tensor_to_flat_f32(state.clone());
        debug_assert_eq!(shape, vec![self.nvar, self.nnodes, self.nmodes]);

        for var in 0..self.nvar {
            for mode in 0..self.nmodes {
                let mut sum = 0.0f32;
                for node in 0..self.nnodes {
                    let idx = var * self.nnodes * self.nmodes + node * self.nmodes + mode;
                    sum += flat[idx];
                }
                self.data.push(sum / self.nnodes as f32);
            }
        }
        self.nsteps += 1;
    }

    fn flush(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.data)
    }

    fn period(&self) -> usize {
        1
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
/// 1. Extracts the specified variables of interest (`voi`) from the state
///    tensor `[nvar, nnodes, nmodes]`, averaging over modes.
/// 2. Multiplies each extracted variable's per-node activity by the gain
///    matrix `[n_sensors, n_regions]` to get sensor-space signals.
/// 3. Accumulates over `period` steps and stores temporal averages.
///
/// Output per averaged time point: `[n_sensors * n_voi]` values, ordered
/// as `[sensors_for_voi0..., sensors_for_voi1...]`.
pub struct SensorProjectionMonitor {
    /// Gain matrix `[n_sensors][n_regions]`.
    pub gain: Vec<Vec<f32>>,
    /// Indices of variables of interest (0-based into nvar dimension).
    pub voi: Vec<usize>,
    /// Temporal averaging period in simulation steps.
    pub period: usize,
    /// Accumulator for temporal averaging.
    accumulator: Vec<f32>,
    /// Number of steps accumulated in the current window.
    accumulator_count: usize,
    /// Recorded data (temporal averages).
    pub data: Vec<f32>,
    /// Number of sensors.
    pub n_sensors: usize,
    /// Number of regions (nodes).
    pub n_regions: usize,
    /// Number of state variables.
    nvar: usize,
    /// Number of modes.
    nmodes: usize,
}

impl SensorProjectionMonitor {
    pub fn new(
        gain: Vec<Vec<f32>>,
        voi: Vec<usize>,
        period: usize,
        nvar: usize,
        n_regions: usize,
        nmodes: usize,
    ) -> Self {
        assert!(period > 0, "period must be > 0");
        let n_sensors = gain.len();
        let n_voi = voi.len().max(1);
        let output_size = n_sensors * n_voi;
        Self {
            gain,
            voi,
            period,
            accumulator: vec![0.0; output_size],
            accumulator_count: 0,
            data: Vec::new(),
            n_sensors,
            n_regions,
            nvar,
            nmodes,
        }
    }

    /// Compute sensor signals from a flat state vector `[nvar * n_regions * nmodes]`.
    fn compute_sensor_signal(&self, flat: &[f32]) -> Vec<f32> {
        let n_voi = self.voi.len();
        let mut signal = Vec::with_capacity(self.n_sensors * n_voi);

        for &vi in &self.voi {
            // Extract variable vi, averaged over modes: node_activity[n] = mean_m(flat[vi * n_regions * nmodes + n * nmodes + m])
            let node_activity: Vec<f32> = (0..self.n_regions).map(|n| {
                let sum: f32 = (0..self.nmodes).map(|m| {
                    flat[vi * self.n_regions * self.nmodes + n * self.nmodes + m]
                }).sum();
                sum / self.nmodes as f32
            }).collect();

            // Multiply: sensor = gain * node_activity
            for s in 0..self.n_sensors {
                let val: f32 = self.gain[s].iter()
                    .zip(node_activity.iter())
                    .map(|(g, a)| g * a)
                    .sum();
                signal.push(val);
            }
        }
        signal
    }
}

impl<B: Backend> Monitor<B> for SensorProjectionMonitor {
    fn record(&mut self, state: &Tensor<B, 3>, _step: usize, _t: f64) {
        let (flat, shape) = crate::io::tensor_to_flat_f32(state.clone());
        debug_assert_eq!(shape, vec![self.nvar, self.n_regions, self.nmodes]);

        let signal = self.compute_sensor_signal(&flat);
        for (a, v) in self.accumulator.iter_mut().zip(signal.iter()) {
            *a += *v;
        }
        self.accumulator_count += 1;

        if self.accumulator_count >= self.period {
            let inv = 1.0 / self.accumulator_count as f32;
            for a in self.accumulator.iter_mut() {
                self.data.push(*a * inv);
                *a = 0.0;
            }
            self.accumulator_count = 0;
        }
    }

    fn flush(&mut self) -> Vec<f32> {
        if self.accumulator_count > 0 {
            let inv = 1.0 / self.accumulator_count as f32;
            for a in self.accumulator.iter_mut() {
                self.data.push(*a * inv);
                *a = 0.0;
            }
            self.accumulator_count = 0;
        }
        std::mem::take(&mut self.data)
    }

    fn period(&self) -> usize {
        self.period
    }
}

// ============================================================
// SpatialAverageMonitor
// ============================================================

/// Spatial average monitor — computes a weighted spatial average of each
/// state variable over nodes using a mask vector.
///
/// For each recorded step and each state variable `v`:
/// `output[v] = sum_n(mask[n] * state[v,n,:]) / sum_n(mask[n])`
/// where modes are averaged out.
///
/// If `mask` is all ones, this reduces to a uniform spatial average.
pub struct SpatialAverageMonitor {
    /// Weighting mask `[n_regions]`.
    pub mask: Vec<f32>,
    /// Temporal averaging period in simulation steps.
    pub period: usize,
    /// Accumulator for temporal averaging.
    accumulator: Vec<f32>,
    /// Number of steps accumulated in the current window.
    accumulator_count: usize,
    /// Recorded data (temporal averages).
    pub data: Vec<f32>,
    /// Number of state variables.
    nvar: usize,
    /// Number of regions (nodes).
    n_regions: usize,
    /// Number of modes.
    nmodes: usize,
    /// Pre-computed sum of mask weights for normalization.
    mask_sum: f32,
}

impl SpatialAverageMonitor {
    pub fn new(mask: Vec<f32>, period: usize, nvar: usize, n_regions: usize, nmodes: usize) -> Self {
        assert!(period > 0, "period must be > 0");
        let mask_sum: f32 = mask.iter().sum();
        assert!(mask_sum.abs() > 1e-12, "spatial mask sum must be non-zero (got {})", mask_sum);
        Self {
            mask,
            period,
            accumulator: vec![0.0; nvar],
            accumulator_count: 0,
            data: Vec::new(),
            nvar,
            n_regions,
            nmodes,
            mask_sum,
        }
    }

    /// Compute spatial averages from a flat state vector `[nvar * n_regions * nmodes]`.
    fn compute_spatial_average(&self, flat: &[f32]) -> Vec<f32> {
        let mut avg = vec![0.0f32; self.nvar];
        let inv_mask = 1.0 / self.mask_sum;
        let inv_nmodes = 1.0 / self.nmodes as f32;

        #[allow(clippy::needless_range_loop)]
        for v in 0..self.nvar {
            let mut sum = 0.0f32;
            for n in 0..self.n_regions {
                let mut mode_sum = 0.0f32;
                for m in 0..self.nmodes {
                    let idx = v * self.n_regions * self.nmodes + n * self.nmodes + m;
                    mode_sum += flat[idx];
                }
                sum += self.mask[n] * mode_sum * inv_nmodes;
            }
            avg[v] = sum * inv_mask;
        }
        avg
    }
}

impl<B: Backend> Monitor<B> for SpatialAverageMonitor {
    fn record(&mut self, state: &Tensor<B, 3>, _step: usize, _t: f64) {
        let (flat, shape) = crate::io::tensor_to_flat_f32(state.clone());
        debug_assert_eq!(shape, vec![self.nvar, self.n_regions, self.nmodes]);

        let avg = self.compute_spatial_average(&flat);
        for (a, v) in self.accumulator.iter_mut().zip(avg.iter()) {
            *a += *v;
        }
        self.accumulator_count += 1;

        if self.accumulator_count >= self.period {
            let inv = 1.0 / self.accumulator_count as f32;
            for a in self.accumulator.iter_mut() {
                self.data.push(*a * inv);
                *a = 0.0;
            }
            self.accumulator_count = 0;
        }
    }

    fn flush(&mut self) -> Vec<f32> {
        if self.accumulator_count > 0 {
            let inv = 1.0 / self.accumulator_count as f32;
            for a in self.accumulator.iter_mut() {
                self.data.push(*a * inv);
                *a = 0.0;
            }
            self.accumulator_count = 0;
        }
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
    use burn::tensor::TensorData;

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
        let mut m = GlobalAverageMonitor::new(2, 3, 1);
        // state [nvar=2, nnodes=3, nmodes=1]:
        // var0: [1, 2, 3]
        // var1: [4, 5, 6]
        let s = make_state(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3, 1]);
        m.record(&s, 0, 0.0);
        // averages: var0 -> 2.0, var1 -> 5.0
        assert_eq!(m.nsteps, 1);
        assert_eq!(m.data, vec![2.0, 5.0]);
        assert_eq!(m.trajectory_shape(), [1, 2, 1, 1]);
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
        // 2 sensors, 3 regions, nvar=2, nmodes=1, voi=[0]
        // Gain: sensor 0 sums all regions, sensor 1 takes difference
        let gain = vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, -1.0, 0.0],
        ];
        let voi = vec![0];
        let mut m = SensorProjectionMonitor::new(gain, voi, 1, 2, 3, 1);

        // state: [nvar=2, nnodes=3, nmodes=1]
        // var0 = [1, 2, 3], var1 = [4, 5, 6]
        let s = make_state(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3, 1]);
        <SensorProjectionMonitor as Monitor<B>>::record(&mut m, &s, 0, 0.0);

        // sensor0: 1*1 + 1*2 + 1*3 = 6.0
        // sensor1: 1*1 + (-1)*2 + 0*3 = -1.0
        let flushed = <SensorProjectionMonitor as Monitor<B>>::flush(&mut m);
        assert_eq!(flushed.len(), 2);
        assert!((flushed[0] - 6.0).abs() < 1e-5, "sensor0 = {}", flushed[0]);
        assert!((flushed[1] - (-1.0)).abs() < 1e-5, "sensor1 = {}", flushed[1]);
    }

    #[test]
    fn test_sensor_projection_monitor_multiple_voi() {
        // 1 sensor, 2 regions, nvar=2, nmodes=1, voi=[0, 1]
        let gain = vec![vec![1.0, 1.0]];
        let voi = vec![0, 1];
        let mut m = SensorProjectionMonitor::new(gain, voi, 2, 2, 2, 1);

        // state step 0: var0=[1,2], var1=[3,4]
        let s0 = make_state(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2, 1]);
        // state step 1: var0=[5,6], var1=[7,8]
        let s1 = make_state(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2, 1]);

        <SensorProjectionMonitor as Monitor<B>>::record(&mut m, &s0, 0, 0.0);
        <SensorProjectionMonitor as Monitor<B>>::record(&mut m, &s1, 1, 0.1);

        // After period=2, temporal average:
        // voi=0: sensor = mean((1+2), (5+6)) = mean(3, 11) = 7.0
        // voi=1: sensor = mean((3+4), (7+8)) = mean(7, 15) = 11.0
        let flushed = <SensorProjectionMonitor as Monitor<B>>::flush(&mut m);
        assert_eq!(flushed.len(), 2);
        assert!((flushed[0] - 7.0).abs() < 1e-5, "voi0 sensor = {}", flushed[0]);
        assert!((flushed[1] - 11.0).abs() < 1e-5, "voi1 sensor = {}", flushed[1]);
    }

    #[test]
    fn test_spatial_average_monitor() {
        // Uniform mask (all ones), nvar=2, n_regions=3, nmodes=1
        let mask = vec![1.0, 1.0, 1.0];
        let mut m = SpatialAverageMonitor::new(mask, 1, 2, 3, 1);

        // state: var0=[1,2,3], var1=[4,5,6]
        let s = make_state(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3, 1]);
        <SpatialAverageMonitor as Monitor<B>>::record(&mut m, &s, 0, 0.0);

        let flushed = <SpatialAverageMonitor as Monitor<B>>::flush(&mut m);
        // var0 avg = (1+2+3)/3 = 2.0, var1 avg = (4+5+6)/3 = 5.0
        assert_eq!(flushed.len(), 2);
        assert!((flushed[0] - 2.0).abs() < 1e-5);
        assert!((flushed[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_spatial_average_monitor_weighted() {
        // Weighted mask, nvar=1, n_regions=2, nmodes=1
        let mask = vec![2.0, 3.0];
        let mut m = SpatialAverageMonitor::new(mask, 1, 1, 2, 1);

        // state: var0=[4, 6]
        let s = make_state(vec![4.0, 6.0], vec![1, 2, 1]);
        <SpatialAverageMonitor as Monitor<B>>::record(&mut m, &s, 0, 0.0);

        let flushed = <SpatialAverageMonitor as Monitor<B>>::flush(&mut m);
        // var0 avg = (2*4 + 3*6)/(2+3) = (8+18)/5 = 26/5 = 5.2
        assert_eq!(flushed.len(), 1);
        assert!((flushed[0] - 5.2).abs() < 1e-5);
    }

    #[test]
    fn test_spatial_average_monitor_with_modes() {
        // Uniform mask, nvar=1, n_regions=2, nmodes=2
        let mask = vec![1.0, 1.0];
        let mut m = SpatialAverageMonitor::new(mask, 2, 1, 2, 2);

        // state: [nvar=1, nnodes=2, nmodes=2]
        // var0: node0=[10,20], node1=[30,40]
        let s = make_state(vec![10.0, 20.0, 30.0, 40.0], vec![1, 2, 2]);
        <SpatialAverageMonitor as Monitor<B>>::record(&mut m, &s, 0, 0.0);

        let flushed = <SpatialAverageMonitor as Monitor<B>>::flush(&mut m);
        // var0: node0 avg over modes = (10+20)/2=15, node1 avg = (30+40)/2=35
        // spatial avg = (15+35)/2 = 25.0
        assert_eq!(flushed.len(), 1);
        assert!((flushed[0] - 25.0).abs() < 1e-5);
    }

    #[test]
    #[should_panic(expected = "spatial mask sum must be non-zero")]
    fn test_spatial_average_monitor_zero_mask_panics() {
        SpatialAverageMonitor::new(vec![0.0, 0.0], 1, 1, 2, 1);
    }

    #[test]
    fn test_monitor_trait_object_safety() {
        // Verify that monitors can be used as trait objects.
        let gain = vec![vec![1.0]];
        let mask = vec![1.0];
        let mut monitors: Vec<Box<dyn Monitor<B>>> = vec![
            Box::new(RawMonitor::new(1, 1, 1)),
            Box::new(TemporalAverageMonitor::new(1, 1, 1, 2)),
            Box::new(SubSampleMonitor::new(1, 1, 1, 2)),
            Box::new(GlobalAverageMonitor::new(1, 1, 1)),
            Box::new(AfferentCouplingMonitor::new(1, 1)),
            Box::new(ProjectionMonitor::new()),
            Box::new(SensorProjectionMonitor::new(gain, vec![0], 1, 1, 1, 1)),
            Box::new(SpatialAverageMonitor::new(mask, 1, 1, 1, 1)),
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
