//! Schema engine for hyburn config files.
//!
//! Provides a runtime schema tree derived from the SimConfig structs,
//! enabling introspection, validation, and structured editing.

use std::fmt;

/// Trait for types that can describe their config schema.
pub trait SchemaProvider {
    fn schema() -> ConfigNode;
}

/// A node in the config schema tree.
#[derive(Debug, Clone)]
pub struct ConfigNode {
    pub name: &'static str,
    pub description: &'static str,
    pub field_type: FieldType,
    pub required: bool,
    pub constraints: Vec<Constraint>,
    pub children: Vec<ConfigNode>,
}

/// The type of a config field.
#[derive(Debug, Clone)]
pub enum FieldType {
    Scalar(ScalarType),
    Array {
        item_type: ScalarType,
        variable_length: bool,
    },
    Enum(Vec<&'static str>),
    Struct,
    TableArray,
    UntaggedEnum(Vec<FieldType>),
}

/// Scalar value types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarType {
    String,
    Integer,
    Float,
    Boolean,
    Path,
}

/// Constraints on field values.
#[derive(Debug, Clone)]
pub enum Constraint {
    Positive,
    MinLen(usize),
    ExactLen(usize),
    Custom(&'static str),
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarType::String => write!(f, "string"),
            ScalarType::Integer => write!(f, "integer"),
            ScalarType::Float => write!(f, "float"),
            ScalarType::Boolean => write!(f, "boolean"),
            ScalarType::Path => write!(f, "path"),
        }
    }
}

impl fmt::Display for FieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FieldType::Scalar(st) => write!(f, "{}", st),
            FieldType::Array { item_type, .. } => write!(f, "[{}...]", item_type),
            FieldType::Enum(variants) => write!(f, "{}", variants.join(" | ")),
            FieldType::Struct => write!(f, "table"),
            FieldType::TableArray => write!(f, "[[table]]"),
            FieldType::UntaggedEnum(variants) => {
                let parts: Vec<String> = variants.iter().map(|v| format!("{}", v)).collect();
                write!(f, "{}", parts.join(" | "))
            }
        }
    }
}

impl ConfigNode {
    pub fn struct_node(
        name: &'static str,
        description: &'static str,
        children: Vec<ConfigNode>,
    ) -> Self {
        ConfigNode {
            name,
            description,
            field_type: FieldType::Struct,
            required: true,
            constraints: vec![],
            children,
        }
    }

    pub fn scalar(
        name: &'static str,
        scalar_type: ScalarType,
        description: &'static str,
        required: bool,
    ) -> Self {
        ConfigNode {
            name,
            description,
            field_type: FieldType::Scalar(scalar_type),
            required,
            constraints: vec![],
            children: vec![],
        }
    }

    pub fn with_constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn table_array(
        name: &'static str,
        description: &'static str,
        children: Vec<ConfigNode>,
    ) -> Self {
        ConfigNode {
            name,
            description,
            field_type: FieldType::TableArray,
            required: false,
            constraints: vec![],
            children,
        }
    }

    pub fn untagged_enum_node(
        name: &'static str,
        description: &'static str,
        variants: Vec<FieldType>,
        required: bool,
    ) -> Self {
        ConfigNode {
            name,
            description,
            field_type: FieldType::UntaggedEnum(variants),
            required,
            constraints: vec![],
            children: vec![],
        }
    }

    /// Find a direct child by name.
    pub fn child(&self, name: &str) -> Option<&ConfigNode> {
        self.children.iter().find(|c| c.name == name)
    }

    /// Return all child names (for completion).
    pub fn child_names(&self) -> Vec<&'static str> {
        self.children.iter().map(|c| c.name).collect()
    }
}

// ---------------------------------------------------------------------------
// SchemaProvider implementations
// ---------------------------------------------------------------------------

impl SchemaProvider for hyburn::config::SimConfig {
    fn schema() -> ConfigNode {
        ConfigNode::struct_node("SimConfig", "Top-level simulation configuration", vec![
            ConfigNode::scalar("sim_length", ScalarType::Float, "Total simulation time in ms", true)
                .with_constraints(vec![Constraint::Positive]),
            ConfigNode::scalar("dt", ScalarType::Float, "Integration step size in ms", true)
                .with_constraints(vec![Constraint::Positive]),
            <hyburn::config::NetworkConfig as SchemaProvider>::schema(),
            <hyburn::engine::integrator::IntegratorKind as SchemaProvider>::schema(),
            ConfigNode {
                name: "monitors",
                description: "Monitor configurations",
                field_type: FieldType::TableArray,
                required: false,
                constraints: vec![],
                children: vec![<hyburn::config::MonitorConfig as SchemaProvider>::schema()],
            },
            ConfigNode {
                name: "stimuli",
                description: "Stimulus configurations",
                field_type: FieldType::TableArray,
                required: false,
                constraints: vec![],
                children: vec![<hyburn::config::StimulusConfig as SchemaProvider>::schema()],
            },
            <hyburn::config::NsigConfig as SchemaProvider>::schema(),
            ConfigNode::scalar("speed", ScalarType::Float, "Signal propagation speed in mm/ms", false)
                .with_constraints(vec![Constraint::Positive]),
            ConfigNode::scalar("backend", ScalarType::String, "Compute backend: ndarray, wgpu, or cuda", false),
        ])
    }
}

impl SchemaProvider for hyburn::config::NetworkConfig {
    fn schema() -> ConfigNode {
        ConfigNode::struct_node("network", "Network topology: subnetworks and projections", vec![
            ConfigNode {
                name: "subnetworks",
                description: "Neural mass model subnetworks",
                field_type: FieldType::TableArray,
                required: true,
                constraints: vec![Constraint::MinLen(1)],
                children: vec![<hyburn::config::SubnetworkConfig as SchemaProvider>::schema()],
            },
            ConfigNode {
                name: "projections",
                description: "Coupling projections between subnetworks",
                field_type: FieldType::TableArray,
                required: false,
                constraints: vec![],
                children: vec![<hyburn::config::ProjectionConfig as SchemaProvider>::schema()],
            },
        ])
    }
}

impl SchemaProvider for hyburn::config::SubnetworkConfig {
    fn schema() -> ConfigNode {
        let model_names: Vec<&'static str> = hyburn::config::MODEL_REGISTRY
            .iter()
            .map(|(name, _, _, _)| *name)
            .collect();

        ConfigNode::struct_node("SubnetworkConfig", "A single subnetwork: model + nodes", vec![
            ConfigNode {
                name: "model",
                description: "Neural mass model name",
                field_type: FieldType::Enum(model_names),
                required: true,
                constraints: vec![],
                children: vec![],
            },
            ConfigNode::scalar("nnodes", ScalarType::Integer, "Number of nodes", true)
                .with_constraints(vec![Constraint::Positive]),
            ConfigNode::scalar("nmodes", ScalarType::Integer, "Number of modes (default: 1)", false)
                .with_constraints(vec![Constraint::Positive]),
            <hyburn::config::InitialStateConfig as SchemaProvider>::schema(),
            ConfigNode {
                name: "params",
                description: "Model parameters (length depends on model)",
                field_type: FieldType::Array {
                    item_type: ScalarType::Float,
                    variable_length: false,
                },
                required: true,
                constraints: vec![Constraint::Custom("Length must match model's expected param count")],
                children: vec![],
            },
        ])
    }
}

impl SchemaProvider for hyburn::config::ProjectionConfig {
    fn schema() -> ConfigNode {
        ConfigNode::struct_node("ProjectionConfig", "Coupling projection between subnetworks", vec![
            ConfigNode::scalar("src", ScalarType::Integer, "Source subnetwork index", true),
            ConfigNode::scalar("tgt", ScalarType::Integer, "Target subnetwork index", true),
            ConfigNode {
                name: "conn_type",
                description: "Connectivity type",
                field_type: FieldType::Enum(vec!["all_to_all", "one_to_one", "csr"]),
                required: false,
                constraints: vec![],
                children: vec![],
            },
            <hyburn::config::WeightsConfig as SchemaProvider>::schema(),
            ConfigNode {
                name: "delays",
                description: "Per-edge delays in integration steps",
                field_type: FieldType::Array { item_type: ScalarType::Integer, variable_length: true },
                required: false,
                constraints: vec![],
                children: vec![],
            },
            ConfigNode {
                name: "tract_lengths",
                description: "Per-edge tract lengths in mm (converts to delays if delays empty)",
                field_type: FieldType::Array { item_type: ScalarType::Float, variable_length: true },
                required: false,
                constraints: vec![],
                children: vec![],
            },
            ConfigNode {
                name: "coupling_fn",
                description: "Coupling function name",
                field_type: FieldType::Enum(vec!["Linear", "Sigmoidal", "Difference"]),
                required: false,
                constraints: vec![],
                children: vec![],
            },
            ConfigNode {
                name: "coupling_params",
                description: "Coupling function parameters",
                field_type: FieldType::Array { item_type: ScalarType::Float, variable_length: true },
                required: false,
                constraints: vec![],
                children: vec![],
            },
            ConfigNode::scalar("cvar_map", ScalarType::String, "Coupling variable mapping (e.g., '0:0')", false),
        ])
    }
}

impl SchemaProvider for hyburn::config::NsigConfig {
    fn schema() -> ConfigNode {
        ConfigNode {
            name: "nsig",
            description: "Noise amplitude for stochastic integration (scalar or per-variable)",
            field_type: FieldType::UntaggedEnum(vec![
                FieldType::Scalar(ScalarType::Float),
                FieldType::Array { item_type: ScalarType::Float, variable_length: true },
            ]),
            required: false,
            constraints: vec![],
            children: vec![],
        }
    }
}

impl SchemaProvider for hyburn::config::WeightsConfig {
    fn schema() -> ConfigNode {
        ConfigNode {
            name: "weights",
            description: "Coupling weights (dense matrix, CSR, or scalar)",
            field_type: FieldType::UntaggedEnum(vec![
                FieldType::Array { item_type: ScalarType::Float, variable_length: true },
                FieldType::Struct, // CSR { data, indices, indptr }
                FieldType::Scalar(ScalarType::Float),
            ]),
            required: true,
            constraints: vec![],
            children: vec![],
        }
    }
}

impl SchemaProvider for hyburn::config::MonitorConfig {
    fn schema() -> ConfigNode {
        ConfigNode::struct_node("MonitorConfig", "Monitor configuration", vec![
            ConfigNode {
                name: "monitor_type",
                description: "Monitor type name",
                field_type: FieldType::Enum(vec![
                    "Raw", "TemporalAverage", "SubSample", "GlobalAverage",
                    "AfferentCoupling", "Projection", "SensorProjection", "SpatialAverage", "Bold",
                ]),
                required: true,
                constraints: vec![],
                children: vec![],
            },
            ConfigNode::scalar("period", ScalarType::Float, "Sampling period in ms", false),
            ConfigNode::scalar("tr", ScalarType::Float, "Repetition time in seconds (BOLD)", false),
            ConfigNode::scalar("bold_period", ScalarType::Integer, "Neural steps between BW integrations (BOLD)", false),
            ConfigNode::scalar("gain_path", ScalarType::Path, "Path to gain matrix .npy file", false),
            ConfigNode {
                name: "voi",
                description: "Variable of interest indices (0-based)",
                field_type: FieldType::Array { item_type: ScalarType::Integer, variable_length: true },
                required: false,
                constraints: vec![],
                children: vec![],
            },
            ConfigNode {
                name: "spatial_mask",
                description: "Spatial averaging mask",
                field_type: FieldType::Array { item_type: ScalarType::Float, variable_length: true },
                required: false,
                constraints: vec![],
                children: vec![],
            },
        ])
    }
}

impl SchemaProvider for hyburn::config::StimulusConfig {
    fn schema() -> ConfigNode {
        ConfigNode::struct_node("StimulusConfig", "Stimulus configuration", vec![
            ConfigNode::scalar("target", ScalarType::Integer, "Target subnetwork index", true),
            ConfigNode::scalar("temporal", ScalarType::String, "Temporal pattern name", true),
            ConfigNode {
                name: "params",
                description: "Stimulus parameters",
                field_type: FieldType::Array { item_type: ScalarType::Float, variable_length: true },
                required: false,
                constraints: vec![],
                children: vec![],
            },
        ])
    }
}

/// Metadata for a registered neural mass model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelInfo {
    pub name: &'static str,
    pub nvar: usize,
    pub ncvar: usize,
    pub nparams: usize,
}

/// Look up a model in MODEL_REGISTRY and return structured info.
pub fn get_model_info(model_name: &str) -> Option<ModelInfo> {
    hyburn::config::MODEL_REGISTRY
        .iter()
        .find(|(n, _, _, _)| *n == model_name)
        .map(|(name, nvar, ncvar, nparams)| ModelInfo {
            name,
            nvar: *nvar,
            ncvar: *ncvar,
            nparams: *nparams,
        })
}

/// Look up a model in MODEL_REGISTRY and return (nvar, ncvar, nparams).
pub fn model_info(name: &str) -> Option<(usize, usize, usize)> {
    get_model_info(name).map(|m| (m.nvar, m.ncvar, m.nparams))
}

/// Return all known model names.
pub fn model_names() -> Vec<&'static str> {
    hyburn::config::MODEL_REGISTRY
        .iter()
        .map(|(name, _, _, _)| *name)
        .collect()
}

impl SchemaProvider for hyburn::config::InitialStateConfig {
    fn schema() -> ConfigNode {
        ConfigNode {
            name: "initial_state",
            description: "Initial state: inline values or path to .npy file",
            field_type: FieldType::UntaggedEnum(vec![
                FieldType::Array { item_type: ScalarType::Float, variable_length: true },
                FieldType::Scalar(ScalarType::Path),
            ]),
            required: true,
            constraints: vec![Constraint::Custom("Length must equal nvar * nnodes * nmodes")],
            children: vec![],
        }
    }
}

impl SchemaProvider for hyburn::engine::integrator::IntegratorKind {
    fn schema() -> ConfigNode {
        ConfigNode {
            name: "integrator",
            description: "Integration scheme",
            field_type: FieldType::Enum(vec![
                "heun", "euler", "euler_stochastic", "heun_stochastic", "rk4", "rk4_stochastic",
            ]),
            required: false,
            constraints: vec![],
            children: vec![],
        }
    }
}
