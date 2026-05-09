//! Configuration types (TOML-based).

use serde::{Deserialize, Serialize};
use crate::engine::integrator::IntegratorKind;
use crate::error::{Result, SimulationError};

/// Registry of known neural mass models and their metadata.
///
/// Used for config validation and parameter sizing.
pub const MODEL_REGISTRY: &[(&str, usize, usize, usize)] = &[
    ("Generic2dOscillator", 2, 1, 12),
    ("MontbrioPazoRoxin",     2, 2, 7),
    ("ReducedWongWang",      1, 1, 8),
    ("Kuramoto",             1, 1, 1),
    ("JansenRit",           6, 1, 13),
    ("WilsonCowan",          2, 1, 22),
    ("Linear",               1, 1, 1),
    ("SupHopf",              2, 2, 2),
    ("Hopfield",             2, 2, 3),
    ("CoombesByrne2D",       2, 2, 4),
    ("CoombesByrne",         4, 4, 5),
    ("GastSchmidtKnoscheSD",  4, 4, 9),
    ("GastSchmidtKnoscheSF",  4, 4, 9),
    ("LarterBreakspear",     3, 1, 32),
    ("Epileptor2D",          2, 1, 12),
    ("Epileptor",            6, 2, 17),
    ("ReducedWongWangExcInh", 2, 1, 19),
    ("DecoBalancedExcInh",    2, 1, 20),
    ("EpileptorCodim3",      3, 1, 13),
    ("EpileptorCodim3SlowMod", 5, 1, 27),
    ("EpileptorRestingState", 8, 3, 28),
    ("ZetterbergJansen",     12, 1, 18),
    ("ReducedSetFitzHughNagumo", 4, 2, 8),
    ("ReducedSetHindmarshRose", 6, 2, 12),
    ("DumontGutkin",         8, 4, 14),
    ("ZerlautAdaptationFirstOrder", 5, 1, 50),
    ("ZerlautAdaptationSecondOrder", 8, 1, 50),
    ("KIonEx",               5, 1, 14),
];

/// Look up a model in the registry by name.
pub fn lookup_model(name: &str) -> Option<(usize, usize, usize)> {
    MODEL_REGISTRY.iter()
        .find(|(n, _, _, _)| *n == name)
        .map(|(_, nvar, ncvar, nparams)| (*nvar, *ncvar, *nparams))
}

/// Top-level simulation configuration (matches a TOML file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    /// Total simulation time in ms.
    pub sim_length: f64,
    /// Integration step size in ms.
    pub dt: f64,
    /// Network topology definition.
    pub network: NetworkConfig,
    /// Integration scheme.
    #[serde(default)]
    pub integrator: IntegratorKind,
    /// Monitor configuration.
    #[serde(default)]
    pub monitors: Vec<MonitorConfig>,
    /// Optional stimulus configuration.
    #[serde(default)]
    pub stimuli: Vec<StimulusConfig>,
    /// Noise amplitude for stochastic integration.
    #[serde(default)]
    pub nsig: f32,
    /// Compute backend: "ndarray" (CPU), "wgpu" (GPU/Metal/Vulkan), or "cuda" (NVIDIA).
    /// Overridable by CLI flag; defaults to ndarray.
    #[serde(default = "default_backend")]
    pub backend: String,
}

fn default_backend() -> String {
    "ndarray".to_string()
}

impl SimConfig {
    /// Default nsig value.
    pub const DEFAULT_NSIG: f32 = 0.0;
}

/// Network topology: a collection of subnetworks and projections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub subnetworks: Vec<SubnetworkConfig>,
    #[serde(default)]
    pub projections: Vec<ProjectionConfig>,
}

/// A single subnetwork: a neural mass model + its nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetworkConfig {
    /// Model name: e.g., "Generic2dOscillator", "MPR", "RWW"
    pub model: String,
    /// Number of nodes.
    pub nnodes: usize,
    /// Number of modes (default 1).
    #[serde(default = "default_modes")]
    pub nmodes: usize,
    /// Initial state: either a NPY path or inline values.
    pub initial_state: InitialStateConfig,
    /// Model parameters.
    pub params: Vec<f32>,
}

fn default_modes() -> usize {
    1
}

/// How to initialise subnetwork state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum InitialStateConfig {
    /// Inline values: shape-compatible Vec.
    Inline(Vec<f32>),
    /// Path to a .npy file containing initial state tensor.
    NpyPath(String),
    /// In-memory tensor data with shape (for WASM / programmatic construction).
    /// Not intended for TOML deserialization — use `Inline` or `NpyPath` in config files.
    #[serde(skip)]
    Memory { data: Vec<f32>, shape: Vec<usize> },
}

/// A coupling projection between subnetworks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionConfig {
    /// Source subnetwork index.
    pub src: usize,
    /// Target subnetwork index.
    pub tgt: usize,
    /// Connectivity type: "all_to_all", "one_to_one", "csr"
    #[serde(default = "default_conn_type")]
    pub conn_type: String,
    /// Coupling weights (dense matrix or CSR components).
    pub weights: WeightsConfig,
    /// Per-edge delays in integration steps.
    #[serde(default)]
    pub delays: Vec<u32>,
    /// Coupling function name.
    #[serde(default = "default_coupling")]
    pub coupling_fn: String,
    /// Coupling function parameters.
    #[serde(default)]
    pub coupling_params: Vec<f32>,
    /// cvar mapping mode.
    #[serde(default = "default_cvar_map")]
    pub cvar_map: String,
}

fn default_conn_type() -> String {
    "all_to_all".into()
}
fn default_coupling() -> String {
    "Linear".into()
}
fn default_cvar_map() -> String {
    "0:0".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WeightsConfig {
    Dense(Vec<Vec<f32>>),
    Csr {
        data: Vec<f32>,
        indices: Vec<u32>,
        indptr: Vec<u32>,
    },
    Scalar(f32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Monitor type name.
    pub monitor_type: String,
    /// Sampling period in ms (generic, used by Raw / TemporalAverage / SubSample monitors).
    #[serde(default)]
    pub period: Option<f64>,
    /// Repetition time in **seconds** (BOLD-specific).  Default = 2.0 s.
    #[serde(default = "default_tr")]
    pub tr: Option<f64>,
    /// Number of neural steps between BW integrations (BOLD-specific).  Default = 10.
    #[serde(default = "default_bold_period")]
    pub bold_period: Option<usize>,
}

fn default_tr() -> Option<f64> {
    Some(2.0)
}
fn default_bold_period() -> Option<usize> {
    Some(10)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StimulusConfig {
    pub target: usize,
    pub temporal: String,
    #[serde(default)]
    pub params: Vec<f32>,
}

impl SimConfig {
    /// Load a SimConfig from a TOML file path.
    /// Not available in WASM builds.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let cfg: SimConfig = toml::from_str(&content)
            .map_err(|e| SimulationError::InvalidConfig(format!("TOML parse error: {}", e)))?;
        Ok(cfg)
    }

    /// Load a SimConfig from a TOML string.
    pub fn from_toml_str(s: &str) -> Result<Self> {
        let cfg: SimConfig = toml::from_str(s)
            .map_err(|e| SimulationError::InvalidConfig(format!("TOML parse error: {}", e)))?;
        Ok(cfg)
    }

    /// Load a SimConfig from a JSON string.
    pub fn from_json_str(s: &str) -> Result<Self> {
        let cfg: SimConfig = serde_json::from_str(s)
            .map_err(|e| SimulationError::InvalidConfig(format!("JSON parse error: {}", e)))?;
        Ok(cfg)
    }

    /// Validate the config (matches Python simulator expectations).
    pub fn validate(&self) -> Result<()> {
        if self.dt <= 0.0 {
            return Err(SimulationError::InvalidConfig("dt must be positive".into()));
        }
        if self.sim_length <= 0.0 {
            return Err(SimulationError::InvalidConfig("sim_length must be positive".into()));
        }
        if self.network.subnetworks.is_empty() {
            return Err(SimulationError::InvalidConfig("At least one subnetwork required".into()));
        }

        for (i, sub) in self.network.subnetworks.iter().enumerate() {
            let (expected_nvar, _expected_ncvar, expected_nparams) = match lookup_model(sub.model.as_str()) {
                Some(info) => info,
                None => return Err(SimulationError::InvalidConfig(
                    format!("Unknown model '{}' in subnetwork {}. Known models: {}",
                        sub.model, i, MODEL_REGISTRY.iter().map(|(n,_,_,_)| *n).collect::<Vec<_>>().join(", "))
                )),
            };
            if sub.nmodes < 1 {
                return Err(SimulationError::InvalidConfig(
                    format!("Subnetwork {}: nmodes must be >= 1", i)
                ));
            }
            if sub.params.len() != expected_nparams {
                return Err(SimulationError::InvalidConfig(format!(
                    "Subnetwork {}: model '{}' expects {} params, got {}",
                    i, sub.model, expected_nparams, sub.params.len()
                )));
            }
            match &sub.initial_state {
                InitialStateConfig::Inline(vals) => {
                    let expected = expected_nvar * sub.nnodes * sub.nmodes;
                    if vals.len() != expected {
                        return Err(SimulationError::InvalidConfig(format!(
                            "Subnetwork {}: initial_state has {} values, expected {} (nvar={} * nnodes={} * nmodes={})",
                            i, vals.len(), expected, expected_nvar, sub.nnodes, sub.nmodes
                        )));
                    }
                }
                InitialStateConfig::Memory { data, shape } => {
                    let expected = expected_nvar * sub.nnodes * sub.nmodes;
                    if data.len() != expected {
                        return Err(SimulationError::InvalidConfig(format!(
                            "Subnetwork {}: Memory initial_state has {} values, expected {}",
                            i, data.len(), expected
                        )));
                    }
                    if shape.len() != 3 {
                        return Err(SimulationError::InvalidConfig(format!(
                            "Subnetwork {}: Memory initial_state shape has {} dims, expected 3",
                            i, shape.len()
                        )));
                    }
                }
                InitialStateConfig::NpyPath(_) => {
                    // Cannot validate NPY file contents without reading the file
                }
            }
            // nvar/ncvar implicitly match when model is recognised; we just check sizes above.
        }

        let n_subs = self.network.subnetworks.len();
        for (i, proj) in self.network.projections.iter().enumerate() {
            if proj.src >= n_subs {
                return Err(SimulationError::InvalidConfig(format!(
                    "Projection {}: src {} >= number of subnetworks {}",
                    i, proj.src, n_subs
                )));
            }
            if proj.tgt >= n_subs {
                return Err(SimulationError::InvalidConfig(format!(
                    "Projection {}: tgt {} >= number of subnetworks {}",
                    i, proj.tgt, n_subs
                )));
            }
            let src_nnodes = self.network.subnetworks[proj.src].nnodes;
            let tgt_nnodes = self.network.subnetworks[proj.tgt].nnodes;
            match &proj.weights {
                WeightsConfig::Dense(mat) => {
                    if mat.len() != tgt_nnodes {
                        return Err(SimulationError::InvalidConfig(format!(
                            "Projection {}: weights has {} rows, expected {} (tgt nnodes)",
                            i, mat.len(), tgt_nnodes
                        )));
                    }
                    for (r, row) in mat.iter().enumerate() {
                        if row.len() != src_nnodes {
                            return Err(SimulationError::InvalidConfig(format!(
                                "Projection {}: weights row {} has {} cols, expected {} (src nnodes)",
                                i, r, row.len(), src_nnodes
                            )));
                        }
                    }
                }
                WeightsConfig::Scalar(_) => {
                    if src_nnodes != tgt_nnodes {
                        return Err(SimulationError::InvalidConfig(format!(
                            "Projection {}: scalar weights require src and tgt nnodes to match ({} != {})",
                            i, src_nnodes, tgt_nnodes
                        )));
                    }
                }
                WeightsConfig::Csr { data, indices, indptr } => {
                    if indptr.len() != tgt_nnodes + 1 {
                        return Err(SimulationError::InvalidConfig(format!(
                            "Projection {}: CSR indptr length {} != tgt_nnodes + 1 = {}",
                            i, indptr.len(), tgt_nnodes + 1
                        )));
                    }
                    if data.len() != indices.len() {
                        return Err(SimulationError::InvalidConfig(format!(
                            "Projection {}: CSR data.len() {} != indices.len() {}",
                            i, data.len(), indices.len()
                        )));
                    }
                }
            }
            let max_delay = proj.delays.iter().copied().max().unwrap_or(0) as usize;
            let horizon_needed = max_delay + 1;
            let n_steps = (self.sim_length / self.dt) as usize;
            if horizon_needed > n_steps + 1 {
                return Err(SimulationError::InvalidConfig(format!(
                    "Projection {}: max delay {} requires horizon >= {}, but sim only runs {} steps",
                    i, max_delay, horizon_needed, n_steps
                )));
            }
        }

        Ok(())
    }
}

/// Parameter sweep configuration (TOML-based).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepConfig {
    /// Parameter to sweep, e.g. `subnetworks[0].params[2]` or `dt`.
    pub parameter_name: String,
    /// Explicit list of values.
    pub values: Option<Vec<f32>>,
    /// Range specification: start, step, end.
    pub range: Option<SweepRange>,
}

/// Range specification for a parameter sweep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepRange {
    pub start: f32,
    pub step: f32,
    pub end: f32,
}

impl SweepConfig {
    /// Load a SweepConfig from a TOML file path.
    /// Not available in WASM builds.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let cfg: SweepConfig = toml::from_str(&content)
            .map_err(|e| SimulationError::InvalidConfig(format!("TOML parse error: {}", e)))?;
        Ok(cfg)
    }

    /// Generate the flat list of parameter values to sweep over.
    pub fn generate_values(&self) -> Vec<f32> {
        if let Some(ref vals) = self.values {
            vals.clone()
        } else if let Some(ref range) = self.range {
            if range.step <= 0.0 {
                return Vec::new();
            }
            let n = ((range.end - range.start) / range.step).ceil() as usize + 1;
            let mut vals = Vec::with_capacity(n);
            for i in 0..n {
                vals.push(range.start + (i as f32) * range.step);
            }
            vals
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_config() {
        let toml_str = r#"
sim_length = 1000.0
dt = 0.1

[network]
[[network.subnetworks]]
model = "Generic2dOscillator"
 nnodes = 2
params = [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0]
initial_state = [0.0, 0.5, 0.0, 0.5]
"#;
        let cfg: SimConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.sim_length, 1000.0);
        assert_eq!(cfg.dt, 0.1);
        assert_eq!(cfg.network.subnetworks.len(), 1);
        assert_eq!(cfg.network.subnetworks[0].model, "Generic2dOscillator");
        assert_eq!(cfg.integrator, IntegratorKind::Heun);
        assert_eq!(cfg.nsig, 0.0);
        cfg.validate().unwrap();
    }

    #[test]
    fn test_parse_config_with_integrator() {
        let toml_str = r#"
sim_length = 100.0
dt = 0.1
integrator = "euler_stochastic"
nsig = 0.5

[network]
[[network.subnetworks]]
model = "Generic2dOscillator"
nnodes = 2
params = [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0]
initial_state = [0.0, 0.5, 0.0, 0.5]
"#;
        let cfg: SimConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.integrator, IntegratorKind::EulerStochastic);
        assert_eq!(cfg.nsig, 0.5);
        cfg.validate().unwrap();
    }

    #[test]
    fn test_validate_invalid_dt() {
        let cfg = SimConfig {
            sim_length: 100.0,
            dt: -0.1,
            network: NetworkConfig { subnetworks: vec![], projections: vec![] },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: 0.0,
            backend: "ndarray".to_string(),
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_all_models() {
        let models: Vec<(&str, usize, usize)> = vec![
            ("Generic2dOscillator", 2, 12),
            ("MontbrioPazoRoxin", 2, 7),
            ("ReducedWongWang", 1, 8),
            ("Kuramoto", 1, 1),
            ("JansenRit", 6, 13),
            ("WilsonCowan", 2, 22),
        ];
        for (name, nvar, nparams) in models {
            let cfg = SimConfig {
                sim_length: 100.0,
                dt: 0.1,
                network: NetworkConfig {
                    subnetworks: vec![SubnetworkConfig {
                        model: name.to_string(),
                        nnodes: 2,
                        nmodes: 1,
                        initial_state: InitialStateConfig::Inline(vec![0.0f32; nvar * 2]),
                        params: vec![0.0f32; nparams],
                    }],
                    projections: vec![],
                },
                integrator: IntegratorKind::Heun,
                monitors: vec![],
                stimuli: vec![],
                nsig: 0.0,
            backend: "ndarray".to_string(),
            };
            cfg.validate().unwrap_or_else(|e| panic!("{} failed validation: {}", name, e));
        }
    }

    #[test]
    fn test_validate_unknown_model() {
        let cfg = SimConfig {
            sim_length: 100.0,
            dt: 0.1,
            network: NetworkConfig {
                subnetworks: vec![SubnetworkConfig {
                    model: "NonexistentModel".to_string(),
                    nnodes: 2,
                    nmodes: 1,
                    initial_state: InitialStateConfig::Inline(vec![0.0f32; 4]),
                    params: vec![0.0f32; 12],
                }],
                projections: vec![],
            },
            integrator: IntegratorKind::Heun,
            monitors: vec![],
            stimuli: vec![],
            nsig: 0.0,
            backend: "ndarray".to_string(),
        };
        let err = cfg.validate().unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("Unknown model"));
        assert!(msg.contains("Known models"));
    }
}
