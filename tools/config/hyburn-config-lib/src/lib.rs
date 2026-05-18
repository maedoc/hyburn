//! hyburn-config-lib: Schema engine for hyburn config files.
//!
//! Provides runtime schema derivation, path-based config editing,
//! JSON Schema generation, and TOML round-trip manipulation.

pub mod editor;
pub mod json_schema;
pub mod path;
pub mod schema;

pub use schema::{ConfigNode, Constraint, FieldType, ScalarType, SchemaProvider};
pub use path::ConfigPath;
pub use editor::ConfigEditor;
pub use json_schema::generate_json_schema;

/// Return the root schema node for SimConfig.
pub fn root_schema() -> ConfigNode {
    <hyburn::config::SimConfig as SchemaProvider>::schema()
}
