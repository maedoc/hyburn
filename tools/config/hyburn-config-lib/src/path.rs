//! Config path parsing.
//!
//! Parses dot-delimited paths like `network.subnetworks[0].params[2]`
//! into a sequence of path segments for navigating config structures.

use std::fmt;
use crate::schema::{ConfigNode, FieldType, Constraint};

/// A single segment in a config path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSegment {
    /// A named field (e.g., `network`, `model`, `params`).
    Field(String),
    /// An array index (e.g., `[0]`, `[2]`).
    Index(usize),
}

/// A parsed config path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigPath {
    pub segments: Vec<PathSegment>,
}

impl ConfigPath {
    /// Parse a dot-delimited path string.
    ///
    /// Examples:
    /// - `"sim_length"` → `[Field("sim_length")]`
    /// - `"network.subnetworks[0].model"` → `[Field("network"), Field("subnetworks"), Index(0), Field("model")]`
    /// - `"network.subnetworks[0].params[2]"` → `[Field("network"), Field("subnetworks"), Index(0), Field("params"), Index(2)]`
    pub fn parse(input: &str) -> Result<Self, PathParseError> {
        if input.is_empty() {
            return Err(PathParseError::Empty);
        }

        let mut segments = Vec::new();

        for part in input.split('.') {
            if part.is_empty() {
                return Err(PathParseError::EmptySegment);
            }

            // Split field name from optional trailing index: "subnetworks[0]" → ("subnetworks", Some(0))
            if let Some(bracket_start) = part.find('[') {
                let field_name = &part[..bracket_start];
                if field_name.is_empty() {
                    return Err(PathParseError::EmptyFieldName);
                }
                segments.push(PathSegment::Field(field_name.to_string()));

                // Parse all [N] suffixes: "params[0][2]" → Index(0), Index(2)
                let rest = &part[bracket_start..];
                let mut pos = 0;
                while pos < rest.len() {
                    if rest.as_bytes()[pos] != b'[' {
                        return Err(PathParseError::UnexpectedChar(rest.as_bytes()[pos] as char));
                    }
                    let close = rest[pos..]
                        .find(']')
                        .ok_or(PathParseError::UnclosedBracket)?;
                    let index_str = &rest[pos + 1..pos + close];
                    let index: usize = index_str
                        .parse()
                        .map_err(|_| PathParseError::InvalidIndex(index_str.to_string()))?;
                    segments.push(PathSegment::Index(index));
                    pos += close + 1;
                }
            } else {
                segments.push(PathSegment::Field(part.to_string()));
            }
        }

        Ok(ConfigPath { segments })
    }

    /// Validate this path against a schema tree.
    pub fn validate(&self, schema: &ConfigNode) -> Result<(), PathValidationError> {
        let mut node = schema;
        for (i, seg) in self.segments.iter().enumerate() {
            match seg {
                PathSegment::Field(name) => {
                    match &node.field_type {
                        FieldType::Struct | FieldType::TableArray => {
                            if let Some(child) = node.child(name) {
                                node = child;
                            } else {
                                return Err(PathValidationError::FieldNotFound {
                                    field: name.clone(),
                                    available: node.child_names().into_iter().map(|s| s.to_string()).collect(),
                                });
                            }
                        }
                        _ => {
                            return Err(PathValidationError::FieldNotFound {
                                field: name.clone(),
                                available: node.child_names().into_iter().map(|s| s.to_string()).collect(),
                            });
                        }
                    }
                }
                PathSegment::Index(idx) => {
                    match &node.field_type {
                        FieldType::TableArray => {
                            if let Some(child) = node.children.first() {
                                node = child;
                            }
                        }
                        FieldType::Array { .. } => {
                            let exact_len = node.constraints.iter().find_map(|c| {
                                if let Constraint::ExactLen(n) = c { Some(*n) } else { None }
                            });
                            if let Some(len) = exact_len
                                && *idx >= len {
                                    return Err(PathValidationError::IndexOutOfBounds {
                                        index: *idx,
                                        max: len.saturating_sub(1),
                                    });
                                }
                            if i + 1 < self.segments.len() {
                                return Err(PathValidationError::NotIndexable {
                                    field: node.name.to_string(),
                                    field_type: format!("{}", node.field_type),
                                });
                            }
                        }
                        _ => {
                            return Err(PathValidationError::NotIndexable {
                                field: node.name.to_string(),
                                field_type: format!("{}", node.field_type),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Return the segments as a slice.
    pub fn as_slice(&self) -> &[PathSegment] {
        &self.segments
    }

    /// Return true if the path is empty.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Return the number of segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Return the first segment, if any.
    pub fn head(&self) -> Option<&PathSegment> {
        self.segments.first()
    }

    /// Return the path without the first segment.
    pub fn tail(&self) -> ConfigPath {
        ConfigPath {
            segments: self.segments[1..].to_vec(),
        }
    }

    /// Return the parent path (all segments except the last).
    pub fn parent(&self) -> Option<ConfigPath> {
        if self.segments.len() <= 1 {
            None
        } else {
            Some(ConfigPath {
                segments: self.segments[..self.segments.len() - 1].to_vec(),
            })
        }
    }

    /// Return the last segment.
    pub fn last(&self) -> Option<&PathSegment> {
        self.segments.last()
    }
}

impl fmt::Display for ConfigPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, seg) in self.segments.iter().enumerate() {
            if i > 0 {
                match seg {
                    PathSegment::Index(_) => {} // no dot before index
                    PathSegment::Field(_) => write!(f, ".")?,
                }
            }
            match seg {
                PathSegment::Field(name) => write!(f, "{}", name)?,
                PathSegment::Index(idx) => write!(f, "[{}]", idx)?,
            }
        }
        Ok(())
    }
}

impl fmt::Display for PathSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathSegment::Field(name) => write!(f, "{}", name),
            PathSegment::Index(idx) => write!(f, "[{}]", idx),
        }
    }
}

/// Errors during path validation against schema.
#[derive(Debug, Clone)]
pub enum PathValidationError {
    FieldNotFound { field: String, available: Vec<String> },
    NotIndexable { field: String, field_type: String },
    IndexOutOfBounds { index: usize, max: usize },
}

impl fmt::Display for PathValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathValidationError::FieldNotFound { field, available } => {
                write!(f, "field '{}' not found in schema (available: {})", field, available.join(", "))
            }
            PathValidationError::NotIndexable { field, field_type } => {
                write!(f, "field '{}' is not indexable (type: {})", field, field_type)
            }
            PathValidationError::IndexOutOfBounds { index, max } => {
                write!(f, "index {} out of bounds (max: {})", index, max)
            }
        }
    }
}

impl std::error::Error for PathValidationError {}

/// Errors during path parsing.
#[derive(Debug, Clone)]
pub enum PathParseError {
    Empty,
    EmptySegment,
    EmptyFieldName,
    UnclosedBracket,
    InvalidIndex(String),
    UnexpectedChar(char),
}

impl fmt::Display for PathParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathParseError::Empty => write!(f, "empty path"),
            PathParseError::EmptySegment => write!(f, "empty segment in path (consecutive dots)"),
            PathParseError::EmptyFieldName => write!(f, "empty field name before bracket"),
            PathParseError::UnclosedBracket => write!(f, "unclosed bracket in path"),
            PathParseError::InvalidIndex(s) => write!(f, "invalid index: '{}'", s),
            PathParseError::UnexpectedChar(c) => write!(f, "unexpected character: '{}'", c),
        }
    }
}

impl std::error::Error for PathParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_against_schema() {
        let schema = crate::root_schema();
        let path = ConfigPath::parse("network.subnetworks[0].model").unwrap();
        assert!(path.validate(&schema).is_ok());

        let bad = ConfigPath::parse("network.subnetworks[0].nonexistent").unwrap();
        assert!(matches!(bad.validate(&schema), Err(PathValidationError::FieldNotFound { .. })));

        let idx_bad = ConfigPath::parse("network.subnetworks[0].params[0].foo").unwrap();
        assert!(matches!(idx_bad.validate(&schema), Err(PathValidationError::NotIndexable { .. })));
    }

    #[test]
    fn parse_simple_field() {
        let p = ConfigPath::parse("sim_length").unwrap();
        assert_eq!(p.segments, vec![PathSegment::Field("sim_length".into())]);
    }

    #[test]
    fn parse_nested_field() {
        let p = ConfigPath::parse("network.subnetworks").unwrap();
        assert_eq!(
            p.segments,
            vec![
                PathSegment::Field("network".into()),
                PathSegment::Field("subnetworks".into()),
            ]
        );
    }

    #[test]
    fn parse_indexed() {
        let p = ConfigPath::parse("network.subnetworks[0].model").unwrap();
        assert_eq!(
            p.segments,
            vec![
                PathSegment::Field("network".into()),
                PathSegment::Field("subnetworks".into()),
                PathSegment::Index(0),
                PathSegment::Field("model".into()),
            ]
        );
    }

    #[test]
    fn parse_nested_index() {
        let p = ConfigPath::parse("network.subnetworks[0].params[2]").unwrap();
        assert_eq!(
            p.segments,
            vec![
                PathSegment::Field("network".into()),
                PathSegment::Field("subnetworks".into()),
                PathSegment::Index(0),
                PathSegment::Field("params".into()),
                PathSegment::Index(2),
            ]
        );
    }

    #[test]
    fn parse_display_roundtrip() {
        let inputs = [
            "sim_length",
            "network.subnetworks[0].model",
            "network.subnetworks[0].params[2]",
            "network.projections[1].weights",
        ];
        for input in inputs {
            let p = ConfigPath::parse(input).unwrap();
            assert_eq!(format!("{}", p), input);
        }
    }

    #[test]
    fn parse_errors() {
        assert!(ConfigPath::parse("").is_err());
        assert!(ConfigPath::parse("a..b").is_err());
        assert!(ConfigPath::parse("[0]").is_err());
        assert!(ConfigPath::parse("a[").is_err());
        assert!(ConfigPath::parse("a[abc]").is_err());
    }

    #[test]
    fn path_parent() {
        let p = ConfigPath::parse("network.subnetworks[0].model").unwrap();
        let parent = p.parent().unwrap();
        assert_eq!(format!("{}", parent), "network.subnetworks[0]");
    }

    #[test]
    fn path_tail() {
        let p = ConfigPath::parse("network.subnetworks[0].model").unwrap();
        let tail = p.tail();
        assert_eq!(format!("{}", tail), "subnetworks[0].model");
    }
}
