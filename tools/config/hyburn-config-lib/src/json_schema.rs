//! JSON Schema generation from the config schema tree.
//!
//! Produces a JSON Schema document that agentic harnesses can use
//! for validation, completion, and code generation.

use crate::schema::{ConfigNode, FieldType, ScalarType};
use serde_json::{json, Value};

/// Generate the full JSON Schema for hyburn's SimConfig.
pub fn generate_json_schema() -> Value {
    let schema = <hyburn::config::SimConfig as crate::schema::SchemaProvider>::schema();
    let mut root = json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "HyburnSimConfig",
        "description": "Hyburn simulation configuration",
        "type": "object",
    });

    let properties = node_to_properties(&schema);
    root["properties"] = properties;

    let required: Vec<&str> = schema
        .children
        .iter()
        .filter(|c| c.required)
        .map(|c| c.name)
        .collect();
    if !required.is_empty() {
        root["required"] = json!(required);
    }

    root
}

/// Generate a partial JSON Schema for a subtree at the given path.
pub fn generate_json_schema_at_path(path: &str) -> Result<Value, String> {
    let schema = <hyburn::config::SimConfig as crate::schema::SchemaProvider>::schema();
    let parts: Vec<&str> = path.split('.').collect();
    let mut node = &schema;

    for part in &parts {
        // Strip array indices for schema lookup
        let field_name = part.split('[').next().unwrap_or(part);
        node = node
            .child(field_name)
            .ok_or_else(|| format!("unknown field: {}", field_name))?;
    }

    Ok(node_to_schema(node))
}

/// Convert a ConfigNode's children to a JSON Schema properties object.
fn node_to_properties(node: &ConfigNode) -> Value {
    let mut props = serde_json::Map::new();
    for child in &node.children {
        props.insert(child.name.to_string(), node_to_schema(child));
    }
    Value::Object(props)
}

/// Convert a single ConfigNode to a JSON Schema value.
fn node_to_schema(node: &ConfigNode) -> Value {
    let mut schema = serde_json::Map::new();

    if !node.description.is_empty() {
        schema.insert("description".into(), json!(node.description));
    }

    match &node.field_type {
        FieldType::Scalar(st) => {
            let type_str = match st {
                ScalarType::String | ScalarType::Path => "string",
                ScalarType::Integer => "integer",
                ScalarType::Float => "number",
                ScalarType::Boolean => "boolean",
            };
            schema.insert("type".into(), json!(type_str));
        }
        FieldType::Array { item_type, .. } => {
            schema.insert("type".into(), json!("array"));
            let item_schema = scalar_type_to_schema(item_type);
            schema.insert("items".into(), item_schema);
        }
        FieldType::Enum(variants) => {
            schema.insert("type".into(), json!("string"));
            schema.insert("enum".into(), json!(variants));
        }
        FieldType::Struct => {
            schema.insert("type".into(), json!("object"));
            if !node.children.is_empty() {
                schema.insert("properties".into(), node_to_properties(node));
                let required: Vec<&str> = node
                    .children
                    .iter()
                    .filter(|c| c.required)
                    .map(|c| c.name)
                    .collect();
                if !required.is_empty() {
                    schema.insert("required".into(), json!(required));
                }
            }
        }
        FieldType::TableArray => {
            schema.insert("type".into(), json!("array"));
            if let Some(child) = node.children.first() {
                let item_schema = node_to_schema(child);
                schema.insert("items".into(), item_schema);
            }
        }
        FieldType::UntaggedEnum(variants) => {
            let one_of: Vec<Value> = variants.iter().map(variant_to_schema).collect();
            schema.insert("oneOf".into(), Value::Array(one_of));
        }
    }

    // Add constraints as custom properties
    for constraint in &node.constraints {
        match constraint {
            crate::schema::Constraint::Positive => {
                schema.insert("exclusiveMinimum".into(), json!(0));
            }
            crate::schema::Constraint::MinLen(n) => {
                schema.insert("minItems".into(), json!(n));
            }
            crate::schema::Constraint::ExactLen(n) => {
                schema.insert("minItems".into(), json!(n));
                schema.insert("maxItems".into(), json!(n));
            }
            crate::schema::Constraint::Custom(desc) => {
                schema.insert("x-constraint".into(), json!(desc));
            }
        }
    }

    Value::Object(schema)
}

/// Convert a FieldType variant (from UntaggedEnum) to a JSON Schema.
fn variant_to_schema(variant: &FieldType) -> Value {
    match variant {
        FieldType::Scalar(st) => scalar_type_to_schema(st),
        FieldType::Array { item_type, .. } => {
            let mut s = serde_json::Map::new();
            s.insert("type".into(), json!("array"));
            s.insert("items".into(), scalar_type_to_schema(item_type));
            Value::Object(s)
        }
        FieldType::Struct => {
            json!({ "type": "object" })
        }
        _ => json!({}),
    }
}

/// Convert a scalar type to a JSON Schema.
fn scalar_type_to_schema(st: &ScalarType) -> Value {
    let type_str = match st {
        ScalarType::String | ScalarType::Path => "string",
        ScalarType::Integer => "integer",
        ScalarType::Float => "number",
        ScalarType::Boolean => "boolean",
    };
    json!({ "type": type_str })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::to_string;

    #[test]
    fn test_generate_json_schema_is_valid_json() {
        let schema = generate_json_schema();
        let json_str = to_string(&schema).unwrap();
        let parsed: Value = serde_json::from_str(&json_str).unwrap();
        assert!(parsed.is_object());
    }

    #[test]
    fn test_generate_json_schema_has_required_fields() {
        let root = generate_json_schema();
        assert_eq!(root["type"], "object");
        assert_eq!(root["title"], "HyburnSimConfig");
        assert_eq!(root["properties"]["sim_length"]["type"], "number");
        assert_eq!(root["properties"]["dt"]["type"], "number");
        assert_eq!(root["properties"]["network"]["type"], "object");
        let required = root["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "sim_length"));
    }

    #[test]
    fn test_generate_json_schema_model_registry() {
        let root = generate_json_schema();
        let models = root["properties"]["network"]["properties"]["subnetworks"]["items"]
            ["properties"]["model"]["enum"]
            .as_array()
            .unwrap();
        assert_eq!(models.len(), 28);
        let model_names: Vec<String> = models.iter().map(|v| v.as_str().unwrap().to_string()).collect();
        assert!(model_names.contains(&"Generic2dOscillator".to_string()));
        assert!(model_names.contains(&"WilsonCowan".to_string()));
        assert!(model_names.contains(&"Epileptor".to_string()));
    }

    #[test]
    fn test_generate_json_schema_untagged_enums() {
        let root = generate_json_schema();
        assert!(root["properties"]["nsig"].get("oneOf").is_some());
        assert!(
            root["properties"]["network"]["properties"]["subnetworks"]["items"]
                ["properties"]["initial_state"]
                .get("oneOf")
                .is_some()
        );
    }
}
