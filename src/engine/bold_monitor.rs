//! BOLD monitor that wraps the Balloon–Windkessel model.
//!
//! Accumulates neural input over `bold_period` neural steps, advances the
//! BW ODEs, and optionally down-samples the resulting BOLD time-series to
//! a user-specified TR (repetition time) before writing to `.npy`.

use crate::engine::bold::{BoldModel, BoldParameters};
use crate::error::{Result, SimulationError};
use crate::io::write_npy_f32;

/// A per-subnetwork BOLD monitor.
///
/// Construction is engine-agnostic; the caller (e.g. `HybridEngine`) supplies
/// the instantaneous per-node neural input every neural step via
/// [`accumulate`](Self::accumulate).
pub struct BoldMonitor {
    pub model: BoldModel,
    pub target_subnetwork: usize,
    /// Number of neural steps between BW integrations.
    pub bold_period: usize,
    /// Repetition time in **seconds**.
    pub tr: f64,
    /// Neural integration step size in **ms**.
    pub dt_neural: f64,
    /// Number of nodes in the target subnetwork.
    pub nnodes: usize,
    /// Accumulator for neural input over the current `bold_period` window.
    accumulator: Vec<f32>,
    /// How many neural steps have been accumulated so far.
    accumulator_count: usize,
    /// Absolute time (in ms) of each stored BOLD sample.
    pub times: Vec<f64>,
    /// Flat BOLD signal data in row-major order: `[ntimes, nnodes]`.
    pub data: Vec<f32>,
    /// Optional path to write NPY on finalisation.
    pub output_path: Option<String>,
}

impl BoldMonitor {
    pub fn new(
        target_subnetwork: usize,
        nnodes: usize,
        bold_period: usize,
        tr: f64,
        dt_neural: f64,
        params: Option<BoldParameters>,
    ) -> Self {
        let model = match params {
            Some(p) => BoldModel::with_params(nnodes, p),
            None => BoldModel::new(nnodes),
        };
        Self {
            model,
            target_subnetwork,
            bold_period: bold_period.max(1),
            tr,
            dt_neural,
            nnodes,
            accumulator: vec![0.0f32; nnodes],
            accumulator_count: 0,
            times: Vec::new(),
            data: Vec::new(),
            output_path: None,
        }
    }

    /// Feed instantaneous per-node neural input for one neural step.
    ///
    /// When `bold_period` steps have been accumulated the monitor averages the
    /// input and advances the BW model by `bold_period * dt_neural` ms.
    pub fn accumulate(&mut self, neural_input: &[f32]) {
        assert_eq!(
            neural_input.len(),
            self.nnodes,
            "neural_input length ({}) != nnodes ({}) for BOLD monitor",
            neural_input.len(),
            self.nnodes
        );

        for (i, &val) in neural_input.iter().enumerate().take(self.nnodes) {
            self.accumulator[i] += val;
        }
        self.accumulator_count += 1;

        if self.accumulator_count >= self.bold_period {
            let avg: Vec<f32> = self
                .accumulator
                .iter()
                .map(|v| *v / self.accumulator_count as f32)
                .collect();
            let dt_bold = self.dt_neural * self.accumulator_count as f64; // ms

            // Advance BW ODEs by the accumulated window
            self.model.step(&avg, dt_bold / 1000.0); // convert ms → s for BW ODEs

            // Store signal
            let sig = self.model.signal();
            let t = self.times.len() as f64 * dt_bold + dt_bold; // end-of-window time in ms
            self.times.push(t);
            self.data.extend_from_slice(&sig);

            // Reset accumulator
            self.accumulator.fill(0.0);
            self.accumulator_count = 0;
        }
    }

    /// Number of BOLD samples recorded so far.
    pub fn n_samples(&self) -> usize {
        self.data.len() / self.nnodes.max(1)
    }

    /// Flattened shape `[n_samples, nnodes]`.
    pub fn data_shape(&self) -> [usize; 2] {
        [self.n_samples(), self.nnodes]
    }

    /// Down-sample the internal BOLD time-series to the user-supplied TR.
    ///
    /// Returns `(target_times_ms, flat_data, shape)` where `shape` is
    /// `[ntarget, nnodes]`.
    pub fn downsample_to_tr(&self,
    ) -> (Vec<f64>, Vec<f32>, Vec<usize>) {
        if self.times.is_empty() {
            return (Vec::new(), Vec::new(), vec![0, self.nnodes]);
        }

        let dt_bold = self.dt_neural * self.bold_period as f64; // ms
        let tr_ms = self.tr * 1000.0; // ms

        // If TR matches the native BOLD sampling, return as-is
        if tr_ms <= 0.0 || (tr_ms - dt_bold).abs() < 1e-6 {
            return (
                self.times.clone(),
                self.data.clone(),
                vec![self.n_samples(), self.nnodes],
            );
        }

        let max_t = *self.times.last().unwrap();
        let n_out = if max_t >= tr_ms {
            (max_t / tr_ms).floor() as usize
        } else {
            0
        };

        let mut target_times = Vec::with_capacity(n_out);
        let mut out_data = Vec::with_capacity(n_out * self.nnodes);

        let nnodes = self.nnodes;
        let n_samples = self.n_samples();

        for j in 0..n_out {
            let t_target = (j + 1) as f64 * tr_ms; // sample at integer multiples of TR
            target_times.push(t_target);

            if t_target <= self.times[0] {
                // Before first sample → nearest
                for n in 0..nnodes {
                    out_data.push(self.data[n]);
                }
                continue;
            }

            // Find bracketing index i such that times[i] <= t_target <= times[i+1]
            let mut idx = 0;
            for k in 1..n_samples {
                if self.times[k] > t_target {
                    idx = k - 1;
                    break;
                }
                idx = k;
            }

            if idx + 1 >= n_samples {
                // Past last sample → nearest
                let base = idx * nnodes;
                for n in 0..nnodes {
                    out_data.push(self.data[base + n]);
                }
                continue;
            }

            let t0 = self.times[idx];
            let t1 = self.times[idx + 1];
            let base0 = idx * nnodes;
            let base1 = (idx + 1) * nnodes;
            let w = if (t1 - t0).abs() < 1e-12 {
                0.0
            } else {
                ((t_target - t0) / (t1 - t0)).clamp(0.0, 1.0)
            };

            for n in 0..nnodes {
                let v0 = self.data[base0 + n];
                let v1 = self.data[base1 + n];
                out_data.push(v0 + (v1 - v0) * w as f32);
            }
        }

        (target_times, out_data, vec![n_out, nnodes])
    }

    /// Finalise any partial accumulation, down-sample to TR, and return
    /// the final flat BOLD data.
    pub fn flush(&mut self) -> Vec<f32> {
        if self.accumulator_count > 0 {
            // Consume remaining partial window
            let avg: Vec<f32> = self
                .accumulator
                .iter()
                .map(|v| *v / self.accumulator_count as f32)
                .collect();
            let dt_bold = self.dt_neural * self.accumulator_count as f64;
            self.model.step(&avg, dt_bold / 1000.0);
            let sig = self.model.signal();
            let t = self.times.len() as f64 * (self.dt_neural * self.bold_period as f64) + dt_bold;
            self.times.push(t);
            self.data.extend_from_slice(&sig);
            self.accumulator.fill(0.0);
            self.accumulator_count = 0;
        }

        let (_ts, data, _shape) = self.downsample_to_tr();
        data
    }

    /// Write the current BOLD time-series (down-sampled to TR) to an `.npy` file.
    pub fn write_npy(&self,
        path: &str,
    ) -> Result<()> {
        let (_ts, data, shape) = self.downsample_to_tr();
        if shape[0] == 0 {
            return Err(SimulationError::InvalidState(
                "No BOLD data to write".into(),
            ));
        }
        write_npy_f32(path, &data, &shape)
    }

    /// Convenience: write to [`output_path`](Self::output_path) if set.
    pub fn write_output(&self,
    ) -> Result<()> {
        match &self.output_path {
            Some(path) => self.write_npy(path),
            None => Err(SimulationError::InvalidConfig(
                "No output_path set for BoldMonitor".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bold_monitor_output_shape() {
        let mut mon = BoldMonitor::new(0, 3, 5, 0.01, 2.0, None);
        // simulate 20 neural steps at dt=2.0 ms → bold_period=5 → 4 BOLD samples
        for step in 0..20 {
            let input = vec![(step as f32).sin(); 3];
            mon.accumulate(&input);
        }
        let shape = mon.data_shape();
        assert_eq!(shape[0], 4, "expected 4 BOLD samples");
        assert_eq!(shape[1], 3, "expected 3 nodes");

        // After flush/downsample, shape should still be [4,3] because TR matches dt_bold (10 ms)
        let flushed = mon.flush();
        assert_eq!(flushed.len(), 4 * 3);
    }

    #[test]
    fn test_bold_monitor_tr_downsample() {
        // dt=1.0 ms, bold_period=10 → BOLD every 10 ms
        // TR=0.025 s = 25 ms → output every 2.5 BOLD samples → should yield coarse grid
        let mut mon = BoldMonitor::new(0, 2, 10, 0.025, 1.0, None);
        for step in 0..100 {
            let input = vec![(step as f32 * 0.1).sin(); 2];
            mon.accumulate(&input);
        }
        let (ts, data, shape) = mon.downsample_to_tr();
        let n_out = shape[0];
        assert!(!ts.is_empty());
        assert_eq!(data.len(), n_out * 2);
        // Should be fewer or equal points than raw BOLD samples
        assert!(n_out <= mon.n_samples());
    }
}
