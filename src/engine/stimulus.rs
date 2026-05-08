//! Stimulus injection for targeted subnetwork perturbation.
//!
//! Supports common temporal patterns:
//! - `impulse`: single delta at onset time
//! - `step`: constant amplitude during an interval
//! - `sinusoid`: sinusoidal oscillation
//! - `pulse_train`: periodic rectangular pulses
//!
//! The stimulus is applied to the first coupling variable (cvar 0) of the target subnetwork
//! at every integration step.

use crate::config::StimulusConfig;
use crate::error::SimulationError;

/// Parsed stimulus pattern applied during simulation steps.
#[derive(Debug, Clone)]
pub struct StimulusApplier {
    /// Target subnetwork index.
    pub target: usize,
    /// Temporal pattern name (impulse, step, sinusoid, pulse_train).
    pub pattern: String,
    /// Pattern-dependent parameters.
    pub params: Vec<f32>,
}

impl StimulusApplier {
    /// Build an applier from a config entry.
    pub fn from_config(cfg: &StimulusConfig) -> Result<Self, SimulationError> {
        let pattern = cfg.temporal.to_lowercase();
        let params = cfg.params.clone();
        // Validate param counts
        match pattern.as_str() {
            "impulse" => {
                if params.len() < 2 {
                    return Err(SimulationError::InvalidConfig("temporal=impulse requires [onset_ms, amplitude]".into()));
                }
            }
            "step" => {
                if params.len() < 3 {
                    return Err(SimulationError::InvalidConfig("temporal=step requires [onset_ms, offset_ms, amplitude]".into()));
                }
            }
            "sinusoid" => {
                if params.len() < 3 {
                    return Err(SimulationError::InvalidConfig("temporal=sinusoid requires [onset_ms, amplitude, frequency_hz, phase_rad?]".into()));
                }
            }
            "pulse_train" => {
                if params.len() < 4 {
                    return Err(SimulationError::InvalidConfig("temporal=pulse_train requires [onset_ms, amplitude, period_ms, width_ms]".into()));
                }
            }
            _ => {
                return Err(SimulationError::InvalidConfig(format!("Unknown temporal pattern: {}", cfg.temporal)));
            }
        }
        Ok(Self { target: cfg.target, pattern, params })
    }

    /// Compute stimulus value at the given simulation step.
    ///
    /// `t_ms = step as f64 * dt_ms`
    pub fn apply(&self, step: usize, dt_ms: f64) -> f32 {
        let t = step as f64 * dt_ms;
        match self.pattern.as_str() {
            "impulse" => impulse_value(t, &self.params),
            "step" => step_value(t, &self.params),
            "sinusoid" => sinusoid_value(t, &self.params),
            "pulse_train" => pulse_train_value(t, &self.params),
            _ => 0.0,
        }
    }
}

fn impulse_value(t: f64, params: &[f32]) -> f32 {
    // params: [onset_ms, amplitude]
    let onset = params[0] as f64;
    let amplitude = params[1];
    if (t - onset).abs() < 1e-6 || (t > onset && t <= onset + 1e-6) {
        amplitude
    } else {
        0.0
    }
}

fn step_value(t: f64, params: &[f32]) -> f32 {
    // params: [onset_ms, offset_ms, amplitude]
    let onset = params[0] as f64;
    let offset = params[1] as f64;
    let amplitude = params[2];
    if t >= onset && t < offset {
        amplitude
    } else {
        0.0
    }
}

fn sinusoid_value(t: f64, params: &[f32]) -> f32 {
    // params: [onset_ms, amplitude, frequency_hz, phase_rad (opt)]
    let onset = params[0] as f64;
    let amplitude = params[1];
    let freq = params[2];
    let phase = params.get(3).copied().unwrap_or(0.0);
    if t < onset {
        0.0
    } else {
        let raw = (2.0 * std::f64::consts::PI * freq as f64 * (t - onset) + phase as f64).sin();
        amplitude * raw as f32
    }
}

fn pulse_train_value(t: f64, params: &[f32]) -> f32 {
    // params: [onset_ms, amplitude, period_ms, width_ms]
    let onset = params[0] as f64;
    let amplitude = params[1];
    let period = params[2] as f64;
    let width = params[3] as f64;
    if t < onset || period <= 0.0 || width <= 0.0 {
        return 0.0;
    }
    let elapsed = t - onset;
    let within_cycle = elapsed % period;
    if within_cycle < width {
        amplitude
    } else {
        0.0
    }
}

impl Default for StimulusApplier {
    fn default() -> Self {
        Self {
            target: 0,
            pattern: "step".to_string(),
            params: vec![0.0, 1000.0, 1.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_impulse_at_onset() {
        let s = StimulusApplier {
            target: 0,
            pattern: "impulse".to_string(),
            params: vec![5.0, 2.0],
        };
        assert!((s.apply(5, 1.0) - 2.0).abs() < 1e-5);
        assert_eq!(s.apply(4, 1.0), 0.0);
        assert_eq!(s.apply(6, 1.0), 0.0);
    }

    #[test]
    fn test_step_duration() {
        let s = StimulusApplier {
            target: 0,
            pattern: "step".to_string(),
            params: vec![2.0, 5.0, 10.0],
        };
        // dt = 1.0, t = step * dt
        assert_eq!(s.apply(1, 1.0), 0.0); // t=1
        assert!((s.apply(2, 1.0) - 10.0).abs() < 1e-5); // t=2
        assert!((s.apply(4, 1.0) - 10.0).abs() < 1e-5); // t=4
        assert_eq!(s.apply(5, 1.0), 0.0); // t=5
    }

    #[test]
    fn test_sinusoid() {
        let s = StimulusApplier {
            target: 0,
            pattern: "sinusoid".to_string(),
            params: vec![0.0, 1.0, 0.25], // freq = 0.25 Hz => period 4 ms
        };
        // t=0 -> sin(0) = 0
        assert!((s.apply(0, 1.0)).abs() < 1e-5);
        // t=1 -> sin(2*pi*0.25*1) = sin(pi/2) = 1
        assert!((s.apply(1, 1.0) - 1.0).abs() < 1e-4);
        // t=2 -> sin(pi) = 0
        assert!((s.apply(2, 1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_pulse_train() {
        let s = StimulusApplier {
            target: 0,
            pattern: "pulse_train".to_string(),
            params: vec![0.0, 3.0, 5.0, 2.0], // period=5, width=2
        };
        assert!((s.apply(0, 1.0) - 3.0).abs() < 1e-5); // t=0, within pulse
        assert!((s.apply(1, 1.0) - 3.0).abs() < 1e-5); // t=1, within pulse
        assert_eq!(s.apply(2, 1.0), 0.0); // t=2, after pulse
        assert!((s.apply(5, 1.0) - 3.0).abs() < 1e-5); // t=5, next period start
    }
}
