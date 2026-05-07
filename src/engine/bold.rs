//! Balloon–Windkessel haemodynamic model (BOLD signal).
//!
//! Implements the canonical Friston et al. (2000, 2003) BW equations as
//! used by TVB’s `BalloonModel` / `Bold` monitor.  State integration is
//! performed with simple forward-Euler in plain Rust `Vec`s (BOLD is orders
//! of magnitude slower than the neural integration, so the overhead is
//! negligible and avoids tensor gymnastics).

/// Per-node BOLD state variables.
#[derive(Debug, Clone)]
pub struct BoldState {
    /// Vasodilatory signal
    pub s: Vec<f32>,
    /// Blood inflow
    pub f: Vec<f32>,
    /// Blood volume
    pub v: Vec<f32>,
    /// Deoxyhaemoglobin content
    pub q: Vec<f32>,
}

impl BoldState {
    pub fn new(nnodes: usize) -> Self {
        let mut s = vec![0.0f32; nnodes];
        let f = vec![1.0f32; nnodes];
        let v = vec![1.0f32; nnodes];
        let q = vec![1.0f32; nnodes];
        // Clamp to non-negative per TVB convention
        for si in &mut s {
            *si = si.max(0.0);
        }
        Self { s, f, v, q }
    }
}

/// Physical / biophysical parameters for the BW model.
///
/// Defaults are the classical TVB values (Friston CBM, not the vbjax
/// short-parameter set).
#[derive(Debug, Clone)]
pub struct BoldParameters {
    /// Signal decay time constant τ_s  (s)
    pub tau_s: f32,
    /// Flow-dependent elimination time constant τ_f  (s)
    pub tau_f: f32,
    /// Haemodynamic transit time τ_0  (s)
    pub tau_0: f32,
    /// Grubb's stiffness exponent α
    pub alpha: f32,
    /// Resting oxygen extraction fraction E0
    pub e0: f32,
    /// Resting blood volume fraction V0
    pub v0: f32,
    /// BOLD coefficient k1
    pub k1: f32,
    /// BOLD coefficient k2
    pub k2: f32,
    /// BOLD coefficient k3
    pub k3: f32,
}

impl Default for BoldParameters {
    fn default() -> Self {
        Self {
            tau_s: 1.54,
            tau_f: 0.50,
            tau_0: 1.02,
            alpha: 0.32,
            e0: 0.34,
            v0: 4.0,
            k1: 7.0,
            k2: 2.0,
            k3: 2.0,
        }
    }
}

/// Balloon–Windkessel ODE integrator.
pub struct BoldModel {
    pub state: BoldState,
    pub params: BoldParameters,
    pub nnodes: usize,
}

impl BoldModel {
    pub fn new(nnodes: usize) -> Self {
        Self {
            state: BoldState::new(nnodes),
            params: BoldParameters::default(),
            nnodes,
        }
    }

    pub fn with_params(nnodes: usize, params: BoldParameters) -> Self {
        Self {
            state: BoldState::new(nnodes),
            params,
            nnodes,
        }
    }

    /// Advance the BW ODEs by one macro-step of size `dt` (seconds).
    ///
    /// `neural_input` must have length `nnodes` and already be temporally
    /// averaged over the desired BOLD period.
    pub fn step(&mut self, neural_input: &[f32], dt: f64) {
        assert_eq!(
            neural_input.len(),
            self.nnodes,
            "neural_input length must equal nnodes"
        );
        let dt_f = dt as f32;
        let p = &self.params;
        let kappa = 1.0 / p.tau_s;
        let gamma = 1.0 / p.tau_f;
        let tau_inv = 1.0 / p.tau_0;
        let alpha = p.alpha;
        let e0 = p.e0;

        for (i, &z_raw) in neural_input.iter().enumerate().take(self.nnodes) {
            let z = z_raw.max(0.0);
            let s = self.state.s[i];
            let f = self.state.f[i];
            let v = self.state.v[i];
            let q = self.state.q[i];

            // Guard against non-physical values that break powers / division
            let f_safe = f.max(1e-6);
            let v_safe = v.max(1e-6);

            // ODEs (Friston 2000 / TVB canonical form)
            let ds = z - kappa * s - gamma * (f_safe - 1.0);
            let df = s;
            let dv = tau_inv * (f_safe - v_safe.powf(1.0 / alpha));
            let dq = tau_inv
                * (f_safe * (1.0 - (1.0 - e0).powf(1.0 / f_safe)) / e0
                    - v_safe.powf(1.0 / alpha) * (q / v_safe));

            let s_new = s + dt_f * ds;
            let f_new = f_safe + dt_f * df;
            let v_new = v_safe + dt_f * dv;
            let q_new = q + dt_f * dq;

            // Clamp to keep physical meaning (TVB clamps to [0, ∞) for s,f,v,q)
            self.state.s[i] = s_new.max(0.0);
            self.state.f[i] = f_new.max(1e-6);
            self.state.v[i] = v_new.max(1e-6);
            self.state.q[i] = q_new.max(0.0);
        }
    }

    /// Compute the BOLD signal per node from the current state.
    pub fn signal(&self) -> Vec<f32> {
        let p = &self.params;
        let mut out = Vec::with_capacity(self.nnodes);
        for i in 0..self.nnodes {
            let v = self.state.v[i].max(1e-6);
            let q = self.state.q[i];
            let bold = p.v0
                * (p.k1 * (1.0 - q)
                    + p.k2 * (1.0 - q / v)
                    + p.k3 * (1.0 - v));
            out.push(bold);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bold_step_zero_input() {
        let mut model = BoldModel::new(2);
        let input = vec![0.0f32; 2];
        for _ in 0..100 {
            model.step(&input, 0.001);
        }
        let sig = model.signal();
        // With zero input BOLD should settle to a stable baseline and remain finite.
        assert!(sig[0].is_finite());
        assert!(sig[1].is_finite());
        // After long zero input the signal should be close to the resting-state
        // baseline: V0 * (k1*(1-1) + k2*(1-1/1) + k3*(1-1)) = 0
        assert!(sig[0].abs() < 1e-3, "baseline should be near zero: {}", sig[0]);
    }

    #[test]
    fn test_bold_step_ramps_up() {
        let mut model = BoldModel::new(2);
        let input = vec![1.0f32; 2];
        // Record signals every step for 5 s (dt=1 ms)
        let mut signals = Vec::new();
        for _ in 0..5000 {
            model.step(&input, 0.001);
            signals.push(model.signal()[0]);
        }
        // After 5 s of constant stimulation BOLD should exceed the resting baseline.
        assert!(
            signals.last().copied().unwrap() > 0.1,
            "BOLD should rise significantly under constant positive input (final={})",
            signals.last().unwrap()
        );
    }

    #[test]
    fn test_bold_parameters_defaults() {
        let p = BoldParameters::default();
        assert!((p.tau_s - 1.54).abs() < 1e-6);
        assert!((p.tau_f - 0.50).abs() < 1e-6);
        assert!((p.tau_0 - 1.02).abs() < 1e-6);
        assert!((p.alpha - 0.32).abs() < 1e-6);
        assert!((p.e0 - 0.34).abs() < 1e-6);
        assert!((p.v0 - 4.0).abs() < 1e-6);
        assert!((p.k1 - 7.0).abs() < 1e-6);
        assert!((p.k2 - 2.0).abs() < 1e-6);
        assert!((p.k3 - 2.0).abs() < 1e-6);
    }
}
