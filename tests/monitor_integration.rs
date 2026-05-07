use burn::backend::ndarray::NdArray;
use burn::tensor::{Tensor, TensorData};
use hyburn::engine::monitor::{Monitor, RawMonitor, TemporalAverageMonitor};

type B = NdArray<f32>;

/// Run a deterministic 100-step dummy simulation and verify that
/// `RawMonitor` and `TemporalAverageMonitor` accumulate data of
/// different lengths.
#[test]
fn test_raw_vs_temporal_average_trajectory_lengths() {
    let nvar = 2;
    let nnodes = 4;
    let nmodes = 1;
    let n_steps = 100;
    let period = 5;

    let mut raw = RawMonitor::new(nvar, nnodes, nmodes);
    let mut avg = TemporalAverageMonitor::new(nvar, nnodes, nmodes, period);

    // Deterministic dummy state evolution
    let mut state_data: Vec<f32> = vec![0.0; nvar * nnodes * nmodes];
    let dt = 0.1;

    for step in 0..n_steps {
        // Simple sinusoidal perturbation to keep state evolving
        for i in 0..state_data.len() {
            let phase = (step * 7 + i * 3) as f32;
            state_data[i] += 0.01 * phase.sin();
        }

        let state = Tensor::<B, 3>::from_floats(
            TensorData::new::<f32, Vec<usize>>(
                state_data.clone(),
                vec![nvar, nnodes, nmodes],
            ),
            &Default::default(),
        );

        let t = step as f64 * dt;
        raw.record(&state, step, t);
        avg.record(&state, step, t);
    }

    let raw_data = raw.flush();
    let avg_data = avg.flush();

    // RawMonitor records every step -> 100 snapshots
    assert_eq!(raw.nsteps, 100);
    assert_eq!(raw_data.len(), 100 * nvar * nnodes * nmodes);

    // TemporalAverageMonitor with period 5 -> 20 snapshots
    assert_eq!(avg.nobs, 20);
    assert_eq!(avg_data.len(), 20 * nvar * nnodes * nmodes);

    // Verify they are indeed different lengths
    assert_ne!(raw_data.len(), avg_data.len());
}
