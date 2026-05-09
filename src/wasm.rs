//! WASM bindings for hyburn.
//!
//! Exposes the core simulation engine to JavaScript/WebAssembly via
//! `wasm-bindgen`. This enables running neural mass simulations
//! directly in the browser with live, interactive trajectory visualization.
//!
//! SBI pipeline (v0.3+): uses `BatchHybridEngine` with batch-dim tensors
//! for efficient per-sweep simulation, replacing the old per-point
//! `HybridEngine` loop.

use burn::backend::ndarray::NdArray;
use burn::tensor::{Tensor, TensorData};
use burn::prelude::Backend;
use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

use crate::config::SimConfig;
use crate::engine::{HybridEngine, IntegratorKind};
use crate::engine::batch_engine::{BatchHybridEngine, SweepParam};
use crate::sbi::{extract_features, extract_features_with, parse_feature_set, MafConfig};

/// The NdArray f32 backend type used for WASM simulations.
type B = NdArray<f32>;

/// Initialize the console logger for WASM.
/// Call this once from JS before using any simulation functions.
#[wasm_bindgen]
pub fn init_logger() {
    console_log::init_with_level(log::Level::Info).ok();
}

/// Web-accessible simulation engine.
///
/// Wraps `HybridEngine<NdArray<f32>>` with a JS-friendly API.
/// Construct from a JSON config string, then call `step()` or `step_n()`
/// to advance the simulation, and `trajectory()` / `bold_signal()` to
/// retrieve data for visualization.
#[wasm_bindgen]
pub struct WebEngine {
    engine: HybridEngine<B>,
    n_steps_run: usize,
}

/// Metadata about a constructed engine, returned to JS after creation.
#[wasm_bindgen]
pub struct EngineInfo {
    /// Number of subnetworks.
    n_subnetworks: usize,
    /// Number of variables per subnetwork (first subnet).
    nvar: usize,
    /// Number of nodes per subnetwork (first subnet).
    nnodes: usize,
    /// Number of modes per subnetwork (first subnet).
    nmodes: usize,
    /// Integration time step.
    dt: f64,
    /// Total steps that will be run (sim_length / dt).
    total_steps: usize,
    /// Number of BOLD monitors.
    n_bold_monitors: usize,
}

#[wasm_bindgen]
impl EngineInfo {
    #[wasm_bindgen(getter)]
    pub fn n_subnetworks(&self) -> usize { self.n_subnetworks }

    #[wasm_bindgen(getter)]
    pub fn nvar(&self) -> usize { self.nvar }

    #[wasm_bindgen(getter)]
    pub fn nnodes(&self) -> usize { self.nnodes }

    #[wasm_bindgen(getter)]
    pub fn nmodes(&self) -> usize { self.nmodes }

    #[wasm_bindgen(getter)]
    pub fn dt(&self) -> f64 { self.dt }

    #[wasm_bindgen(getter)]
    pub fn total_steps(&self) -> usize { self.total_steps }

    #[wasm_bindgen(getter)]
    pub fn n_bold_monitors(&self) -> usize { self.n_bold_monitors }
}

#[wasm_bindgen]
impl WebEngine {
    /// Create a new engine from a JSON config string.
    ///
    /// The JSON must conform to the `SimConfig` schema. Example:
    /// ```json
    /// {
    ///   "sim_length": 1000.0,
    ///   "dt": 0.1,
    ///   "network": {
    ///     "subnetworks": [{
    ///       "model": "Generic2dOscillator",
    ///       "nnodes": 2,
    ///       "nmodes": 1,
    ///       "params": [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    ///       "initial_state": [0.0, 0.5, 0.0, 0.5]
    ///     }],
    ///     "projections": []
    ///   }
    /// }
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn from_json(json: &str) -> Result<WebEngine, JsValue> {
        let cfg = SimConfig::from_json_str(json)
            .map_err(|e| JsValue::from_str(&format!("Config error: {}", e)))?;
        cfg.validate()
            .map_err(|e| JsValue::from_str(&format!("Validation error: {}", e)))?;
        let device: <B as Backend>::Device = Default::default();
        let engine = HybridEngine::<B>::from_config(cfg.clone(), device)
            .map_err(|e| JsValue::from_str(&format!("Engine creation error: {}", e)))?;
        Ok(WebEngine { engine, n_steps_run: 0 })
    }

    /// Create a new engine from a TOML config string.
    #[wasm_bindgen]
    pub fn from_toml(toml: &str) -> Result<WebEngine, JsValue> {
        let cfg = SimConfig::from_toml_str(toml)
            .map_err(|e| JsValue::from_str(&format!("Config error: {}", e)))?;
        cfg.validate()
            .map_err(|e| JsValue::from_str(&format!("Validation error: {}", e)))?;
        let device: <B as Backend>::Device = Default::default();
        let engine = HybridEngine::<B>::from_config(cfg.clone(), device)
            .map_err(|e| JsValue::from_str(&format!("Engine creation error: {}", e)))?;
        Ok(WebEngine { engine, n_steps_run: 0 })
    }

    /// Get engine metadata (dimensions, dt, etc.).
    pub fn info(&self) -> EngineInfo {
        let sub = &self.engine.subnetworks[0];
        EngineInfo {
            n_subnetworks: self.engine.subnetworks.len(),
            nvar: sub.nvar,
            nnodes: sub.nnodes,
            nmodes: sub.nmodes,
            dt: self.engine.dt,
            total_steps: 0, // filled by caller from config if needed
            n_bold_monitors: self.engine.bold_monitors.len(),
        }
    }

    /// Advance the simulation by one step.
    pub fn step(&mut self) {
        self.engine.step();
        self.n_steps_run += 1;
    }

    /// Advance the simulation by `n` steps.
    pub fn step_n(&mut self, n: usize) {
        self.engine.run(n);
        self.n_steps_run += n;
    }

    /// Current step number.
    pub fn current_step(&self) -> usize {
        self.engine.step
    }

    /// Number of steps run so far.
    pub fn steps_run(&self) -> usize {
        self.n_steps_run
    }

    /// Get the raw trajectory data as a Float32Array (zero-copy).
    ///
    /// The trajectory is a flat array of f32 values with layout:
    /// `[step0_var0_node0_mode0, step0_var0_node0_mode1, ..., step0_var0_node1_mode0, ...]`
    ///
    /// For a single subnetwork with `nvar` variables, `nnodes` nodes,
    /// `nmodes` modes, and `n_steps` recorded steps, the shape is
    /// `[n_steps, nvar, nnodes, nmodes]`.
    ///
    /// For multiple subnetworks, the data is concatenated per step.
    pub fn trajectory(&self) -> Float32Array {
        let traj = &self.engine.trajectory;
        if traj.is_empty() {
            Float32Array::new_with_length(0)
        } else {
            Float32Array::from(traj.as_slice())
        }
    }

    /// Get the trajectory length (number of f32 values).
    pub fn trajectory_len(&self) -> usize {
        self.engine.trajectory.len()
    }

    /// Get the current state of the first subnetwork as a Float32Array.
    ///
    /// Shape: `[nvar, nnodes, nmodes]`
    pub fn current_state(&self) -> Float32Array {
        if self.engine.states.is_empty() {
            return Float32Array::new_with_length(0);
        }
        let (data, _shape) = crate::io::tensor_to_flat_f32::<B, 3>(
            self.engine.states[0].clone(),
        );
        Float32Array::from(data.as_slice())
    }

    /// Get the BOLD monitor signal as a Float32Array.
    ///
    /// Returns data for all BOLD monitors concatenated.
    /// Each monitor's data has shape `[n_bold_volumes, nnodes]`.
    pub fn bold_signal(&self) -> Float32Array {
        let mut all_bold = Vec::new();
        for monitor in &self.engine.bold_monitors {
            all_bold.extend_from_slice(&monitor.data);
        }
        Float32Array::from(all_bold.as_slice())
    }

    /// Number of BOLD volumes recorded so far.
    pub fn bold_volumes(&self) -> usize {
        self.engine.bold_monitors.first()
            .map(|m| {
                if m.nnodes > 0 { m.data.len() / m.nnodes } else { 0 }
            })
            .unwrap_or(0)
    }

    /// Get the current state of all subnetworks as a Float32Array.
    pub fn all_states(&self) -> Float32Array {
        let mut all_data = Vec::new();
        for state in &self.engine.states {
            let (data, _shape) = crate::io::tensor_to_flat_f32::<B, 3>(state.clone());
            all_data.extend(data);
        }
        Float32Array::from(all_data.as_slice())
    }

    /// Get the number of subnetworks.
    pub fn n_subnetworks(&self) -> usize {
        self.engine.subnetworks.len()
    }

    /// Get the nvar for a subnetwork.
    pub fn subnetwork_nvar(&self, idx: usize) -> usize {
        self.engine.subnetworks.get(idx).map(|s| s.nvar).unwrap_or(0)
    }

    /// Get the nnodes for a subnetwork.
    pub fn subnetwork_nnodes(&self, idx: usize) -> usize {
        self.engine.subnetworks.get(idx).map(|s| s.nnodes).unwrap_or(0)
    }

    /// Get the nmodes for a subnetwork.
    pub fn subnetwork_nmodes(&self, idx: usize) -> usize {
        self.engine.subnetworks.get(idx).map(|s| s.nmodes).unwrap_or(0)
    }

    /// Get the integration time step.
    pub fn dt(&self) -> f64 {
        self.engine.dt
    }

    /// Get the noise amplitude (nsig).
    pub fn nsig(&self) -> f32 {
        self.engine.nsig
    }

    /// Get the integrator kind as a string ("heun", "euler", "euler_stochastic", "heun_stochastic").
    pub fn integrator(&self) -> String {
        self.engine.integrator.to_string()
    }
}

/// Validate a JSON config string without creating an engine.
/// Returns an error message if the config is invalid, or empty string if valid.
#[wasm_bindgen]
pub fn validate_config_json(json: &str) -> String {
    match SimConfig::from_json_str(json) {
        Ok(cfg) => match cfg.validate() {
            Ok(()) => String::new(),
            Err(e) => format!("{}", e),
        },
        Err(e) => format!("{}", e),
    }
}

/// Validate a TOML config string without creating an engine.
#[wasm_bindgen]
pub fn validate_config_toml(toml: &str) -> String {
    match SimConfig::from_toml_str(toml) {
        Ok(cfg) => match cfg.validate() {
            Ok(()) => String::new(),
            Err(e) => format!("{}", e),
        },
        Err(e) => format!("{}", e),
    }
}

/// Get the model registry as a JSON string.
/// Returns an array of {name, nvar, ncvar, nparams} objects.
#[wasm_bindgen]
pub fn model_registry_json() -> String {
    use crate::config::MODEL_REGISTRY;
    let entries: Vec<serde_json::Value> = MODEL_REGISTRY.iter().map(|(name, nvar, ncvar, nparams)| {
        serde_json::json!({
            "name": name,
            "nvar": nvar,
            "ncvar": ncvar,
            "nparams": nparams,
        })
    }).collect();
    serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string())
}

/// Get the default parameters for a model as a JSON string.
#[wasm_bindgen]
pub fn model_default_params(model_name: &str) -> Result<String, JsValue> {
    let params: Vec<f32> = match model_name {
        "Generic2dOscillator" => crate::model::g2do::g2do_default_params(),
        "MontbrioPazoRoxin" => crate::model::mpr::mpr_default_params(),
        "ReducedWongWang" => crate::model::rww::rww_default_params(),
        "Kuramoto" => crate::model::kuramoto_model::kuramoto_default_params(),
        "JansenRit" => crate::model::jansen_rit::jansen_rit_default_params(),
        "WilsonCowan" => crate::model::wilson_cowan::wilson_cowan_default_params(),
        "Linear" => crate::model::linear::linear_default_params(),
        "SupHopf" => crate::model::sup_hopf::sup_hopf_default_params(),
        "Hopfield" => crate::model::hopfield::hopfield_default_params(),
        "CoombesByrne2D" => crate::model::coombes_byrne2d::coombes_byrne2d_default_params(),
        "CoombesByrne" => crate::model::coombes_byrne::coombes_byrne_default_params(),
        "GastSchmidtKnoscheSD" => crate::model::gast_schmidt_knosche_sd::gast_sd_default_params(),
        "GastSchmidtKnoscheSF" => crate::model::gast_schmidt_knosche_sf::gast_sf_default_params(),
        "LarterBreakspear" => crate::model::larter_breakspear::larter_breakspear_default_params(),
        "Epileptor2D" => crate::model::epileptor2d::epileptor2d_default_params(),
        "Epileptor" => crate::model::epileptor::epileptor_default_params(),
        "ReducedWongWangExcInh" => crate::model::rww_exc_inh::rww_exc_inh_default_params(),
        "DecoBalancedExcInh" => crate::model::deco_balanced_exc_inh::deco_balanced_exc_inh_default_params(),
        "EpileptorCodim3" => crate::model::epileptor_codim3::epileptor_codim3_default_params(),
        "EpileptorCodim3SlowMod" => crate::model::epileptor_codim3_slowmod::epileptor_codim3_slowmod_default_params(),
        "EpileptorRestingState" => crate::model::epileptor_rs::epileptor_rs_default_params(),
        "ZetterbergJansen" => crate::model::zetterberg_jansen::zetterberg_jansen_default_params(),
        "ReducedSetFitzHughNagumo" => crate::model::reduced_fhn::reduced_fhn_default_params(),
        "ReducedSetHindmarshRose" => crate::model::reduced_hr::reduced_hr_default_params(),
        "DumontGutkin" => crate::model::dumont_gutkin::dumont_gutkin_default_params(),
        "ZerlautAdaptationFirstOrder" => crate::model::zerlaut_first::zerlaut_first_default_params(),
        "ZerlautAdaptationSecondOrder" => crate::model::zerlaut_second::zerlaut_second_default_params(),
        "KIonEx" => crate::model::kionex::kionex_default_params(),
        _ => return Err(JsValue::from_str(&format!(
            "Default params not available for '{}'. Known models: Generic2dOscillator, MontbrioPazoRoxin, ReducedWongWang, Kuramoto, JansenRit, WilsonCowan, Linear, SupHopf, Hopfield, CoombesByrne2D, CoombesByrne, GastSchmidtKnoscheSD, GastSchmidtKnoscheSF, LarterBreakspear, Epileptor2D, Epileptor, ReducedWongWangExcInh, DecoBalancedExcInh, EpileptorCodim3, EpileptorCodim3SlowMod, EpileptorRestingState, ZetterbergJansen, ReducedSetFitzHughNagumo, ReducedSetHindmarshRose, DumontGutkin, ZerlautAdaptationFirstOrder, ZerlautAdaptationSecondOrder, KIonEx",
            model_name
        ))),
    };
    Ok(serde_json::to_string(&params).unwrap())
}

// ---------------------------------------------------------------------------
// Preset Examples
// ---------------------------------------------------------------------------

/// Get the list of available preset examples as a JSON string.
///
/// Returns an array of `{id, name, description}` objects.
#[wasm_bindgen]
pub fn list_presets() -> String {
    use crate::presets::PRESETS;
    let entries: Vec<serde_json::Value> = PRESETS.iter().map(|p| {
        serde_json::json!({
            "id": p.id,
            "name": p.name,
            "description": p.description,
        })
    }).collect();
    serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string())
}

/// Get a preset's JSON config by its ID.
///
/// Returns the full SimConfig JSON with inline initial_state data
/// (NPY files already resolved to float arrays at build time).
///
/// Returns an empty string if the ID is not found.
#[wasm_bindgen]
pub fn get_preset(id: &str) -> String {
    crate::presets::get_preset_json(id)
        .unwrap_or("")
        .to_string()
}

// ---------------------------------------------------------------------------
// SBI Pipeline (WASM)
// ---------------------------------------------------------------------------

/// Result of running the SBI pipeline in the browser.
///
/// JSON-serializable struct that the JS layer uses to render diagnostic plots.
#[derive(serde::Serialize)]
pub struct WebSbiResult {
    /// Parameter values used in the sweep.
    pub sweep_values: Vec<f32>,
    /// Training loss history: (epoch, loss) pairs.
    pub loss_history: Vec<(usize, f32)>,
    /// SBI diagnostics (z-scores, shrinkage, etc.).
    pub diagnostics: crate::sbi::SbiDiagnostics,
    /// Per-test-point posterior stats: (true_param, posterior_mean, posterior_std).
    pub posterior_stats: Vec<(f32, f32, f32)>,
    /// Feature dimension.
    pub feature_dim: usize,
    /// MAF configuration used.
    pub maf_config: crate::sbi::MafConfig,
}

/// Reshape a batch-dim trajectory (flat from [n_sweep, nnodes*nmodes, nvar]
/// tensors concatenated per step) into per-sweep-point layout expected by
/// `extract_features`: `[n_steps, nvar, nnodes, nmodes]` per point.
///
/// Batch layout (row-major 3D tensor per step): step → sweep_point → (node*mode, var)
///   flat[t * (ns * nmt * nv) + s * (nmt * nv) + (n*nm + m) * nv + v]
///
/// Per-point target: [n_steps, nvar, nnodes, nmodes]
///   per_point[((t * nv + v) * nn + n) * nm + m]
fn reshape_batch_to_per_point(
    batch_flat: &[f32],
    n_sweep: usize,
    n_steps: usize,
    nvar: usize,
    nnodes: usize,
    nmodes: usize,
) -> Vec<Vec<f32>> {
    let nm_total = nnodes * nmodes; // combined node-mode axis in batch tensor
    let nv = nvar;
    let per_point_elems = n_steps * nv * nnodes * nmodes;
    let mut result = Vec::with_capacity(n_sweep);
    for s in 0..n_sweep {
        let mut traj = vec![0.0f32; per_point_elems];
        for t in 0..n_steps {
            for v in 0..nv {
                for n in 0..nnodes {
                    for m in 0..nmodes {
                        let batch_idx = ((t * n_sweep + s) * nm_total
                            + (n * nmodes + m)) * nv
                            + v;
                        let per_idx = ((t * nv + v) * nnodes + n) * nmodes + m;
                        traj[per_idx] = batch_flat[batch_idx];
                    }
                }
            }
        }
        result.push(traj);
    }
    result
}

/// Run a full SBI pipeline in the browser and return results as JSON.
///
/// Uses `BatchHybridEngine` for the sweep (all points in one batch-dim
/// engine) instead of the old per-point `HybridEngine` loop. Removes
/// hardcoded G2DO/Euler/dt=0.1 assumptions — reads model, integrator,
/// dt, and initial state from the config.
///
/// # Arguments
/// * `config_json` - SimConfig JSON string
/// * `n_sweep` - number of sweep points
/// * `n_steps` - simulation steps per point
/// * `n_epochs` - MAF training epochs
/// * `batch_size` - MAF training batch size
/// * `n_post_samples` - posterior samples per test point
/// * `param_name` - sweep parameter name like "I_ext" or "subnetworks[0].params[1]"
///   (or numeric param_idx like "1" for backward compat)
/// * `range_min`, `range_max` - sweep value range
/// * `feature_set` - "classic" (3 stats) or "catch22" (22 features)
#[wasm_bindgen]
pub fn run_sbi_json_cfg(
    config_json: &str,
    n_sweep: usize,
    n_steps: usize,
    n_epochs: usize,
    batch_size: usize,
    n_post_samples: usize,
    param_name: &str,
    range_min: f32,
    range_max: f32,
    feature_set: &str,
) -> Result<String, JsValue> {
    let cfg = SimConfig::from_json_str(config_json)
        .map_err(|e| JsValue::from_str(&format!("Config error: {}", e)))?;
    cfg.validate()
        .map_err(|e| JsValue::from_str(&format!("Validation error: {}", e)))?;

    use burn::backend::autodiff::Autodiff;
    type AD = Autodiff<NdArray<f32>>;

    let device: <B as Backend>::Device = Default::default();
    let device_ad: <AD as Backend>::Device = Default::default();

    // Resolve sweep parameter to (sub_idx, param_idx)
    let (sub_idx, param_idx) = resolve_sweep_param_name(param_name, 0, 1);
    let sweep_param = SweepParam { sub_idx, param_idx };

    // Build sweep values
    let sweep_values: Vec<f32> = if n_sweep <= 1 {
        vec![range_min]
    } else {
        (0..n_sweep)
            .map(|i| range_min + i as f32 * (range_max - range_min) / (n_sweep - 1) as f32)
            .collect()
    };

    // --- 1. Run batch-dim sweep with trajectories ---
    let mut batch_engine = BatchHybridEngine::<B>::from_config(cfg.clone(), n_sweep, device.clone())
        .map_err(|e| JsValue::from_str(&format!("BatchHybridEngine error: {}", e)))?;

    let result = batch_engine.run_sweep_with_trajectory(
        &sweep_param,
        &sweep_values,
        n_steps,
    );

    // --- 2. Extract features from per-point trajectories ---
    let nnodes = cfg.network.subnetworks[0].nnodes;
    let nmodes = cfg.network.subnetworks[0].nmodes;
    let nvar = crate::config::lookup_model(&cfg.network.subnetworks[0].model)
        .map(|(nv, _, _)| nv)
        .unwrap_or(2);

    let batch_traj = result.trajectories.as_ref()
        .and_then(|t| t.first().cloned())
        .ok_or_else(|| JsValue::from_str("No trajectories in batch result"))?;

    let all_trajs = reshape_batch_to_per_point(
        &batch_traj, n_sweep, n_steps, nvar, nnodes, nmodes,
    );

    let fs = crate::sbi::parse_feature_set(feature_set).ok_or_else(|| {
        JsValue::from_str(&format!("Unknown feature_set: {}", feature_set))
    })?;

    let shape = [n_steps, nvar, nnodes, nmodes];
    let mut all_features: Vec<f32> = Vec::with_capacity(n_sweep * nvar * fs.features_per_var());
    let mut all_params: Vec<f32> = Vec::with_capacity(n_sweep);

    for s in 0..n_sweep {
        let feats = extract_features_with(&all_trajs[s], &shape, &fs);
        all_features.extend_from_slice(&feats);
        all_params.push(sweep_values[s]);
    }

    let feature_dim = if n_sweep > 0 { all_features.len() / n_sweep } else { 0 };

    // --- 3. Train MAF ---
    let maf_config = MafConfig {
        param_dim: 1,
        feature_dim,
        hidden_units: 16,
        n_flows: 2,
        learning_rate: 1e-2,
        feature_set: feature_set.to_string(),
    };

    let (maf, loss_history) = crate::sbi::train_maf_with_data_and_log(
        &maf_config,
        all_params.clone(),
        all_features.clone(),
        n_epochs,
        batch_size,
    );

    // --- 4. Posterior inference ---
    let prior_mean = (range_min + range_max) / 2.0;
    let prior_std = (range_max - range_min) / (2.0 * 3.0f32.sqrt());

    let mut posterior_stats: Vec<(f32, f32, f32)> = Vec::with_capacity(n_sweep);
    let mut all_posterior_samples: Vec<f32> = Vec::new();
    let mut all_true_params: Vec<f32> = Vec::new();

    for (i, &true_param) in all_params.iter().enumerate() {
        let f_start = i * feature_dim;
        let features_slice = &all_features[f_start..f_start + feature_dim];

        let context = Tensor::<AD, 2>::from_floats(
            TensorData::new::<f32, Vec<usize>>(features_slice.to_vec(), vec![1, feature_dim]),
            &device_ad,
        );

        let samples = maf.inverse_sample(context, n_post_samples);
        let data = samples.into_data();
        let slice = data.as_slice::<f32>().unwrap();

        let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
        let var: f32 = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / slice.len() as f32;
        let std = var.sqrt();

        posterior_stats.push((true_param, mean, std));
        all_posterior_samples.extend_from_slice(slice);
        all_true_params.push(true_param);
    }

    // --- 5. Diagnostics ---
    let diagnostics = crate::sbi::SbiDiagnostics::from_samples(
        &all_posterior_samples,
        &all_true_params,
        &[prior_mean],
        &[prior_std],
        n_post_samples,
        1,
    );

    let web_result = WebSbiResult {
        sweep_values,
        loss_history,
        diagnostics,
        posterior_stats,
        feature_dim,
        maf_config,
    };

    serde_json::to_string(&web_result)
        .map_err(|e| JsValue::from_str(&format!("JSON serialization error: {}", e)))
}

/// Resolve a sweep parameter name or index into (sub_idx, param_idx).
///
/// Supports:
/// - `"N"` where N is a number → (sub_idx=0, param_idx=N) [backward compat]
/// - `"subnetworks[K].params[P]"` → (sub_idx=K, param_idx=P) [config style]
fn resolve_sweep_param_name(
    name: &str,
    default_sub_idx: usize,
    default_param_idx: usize,
) -> (usize, usize) {
    // Try parse as plain integer → backward compat with old param_idx
    if let Ok(idx) = name.parse::<usize>() {
        return (default_sub_idx, idx);
    }
    // Try "subnetworks[X].params[Y]" format
    if name.contains(".params[") {
        // Extract sub_idx
        if let Some(start) = name.find('[') {
            if let Some(end) = name[start+1..].find(']') {
                let sub_end = start + 1 + end;
                if let Ok(sub_idx) = name[start+1..sub_end].parse::<usize>() {
                    // Extract param_idx
                    if let Some(pstart) = name.rfind('[') {
                        if let Some(pend) = name[pstart+1..].find(']') {
                            let p_end = pstart + 1 + pend;
                            if let Ok(p_idx) = name[pstart+1..p_end].parse::<usize>() {
                                return (sub_idx, p_idx);
                            }
                        }
                    }
                }
            }
        }
    }
    // Default fallback
    (default_sub_idx, default_param_idx)
}

// ---------------------------------------------------------------------------
// Backward-compatible wrapper (keeps JS API stable for existing users)
// ---------------------------------------------------------------------------

/// Run a small SBI pipeline (legacy wrapper — calls `run_sbi_json_cfg`
/// with default range [-0.5, 0.5] and Classic feature set).
#[wasm_bindgen]
pub fn run_sbi_json(
    config_json: &str,
    n_sweep: usize,
    n_steps: usize,
    n_epochs: usize,
    batch_size: usize,
    n_post_samples: usize,
    param_idx: usize,
) -> Result<String, JsValue> {
    run_sbi_json_cfg(
        config_json, n_sweep, n_steps, n_epochs, batch_size, n_post_samples,
        &param_idx.to_string(),
        -0.5, 0.5, "classic",
    )
}
