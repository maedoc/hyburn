//! Config editor using toml_edit for round-trip TOML manipulation.
//!
//! Preserves comments, formatting, and key ordering when modifying config files.

use crate::path::{ConfigPath, PathSegment};
use crate::schema::{ConfigNode, SchemaProvider};
use std::fmt;
use toml_edit::{DocumentMut, Item, Table, Value};

/// Errors during config editing operations.
#[derive(Debug)]
pub enum EditError {
    Io(std::io::Error),
    Parse(toml_edit::TomlError),
    Path(crate::path::PathParseError),
    NotFound(String),
    IndexOutOfBounds { index: usize, len: usize },
    NotATable(String),
    NotAnArray(String),
    InvalidValue(String),
    Validation(Vec<ValidationError>),
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditError::Io(e) => write!(f, "IO error: {}", e),
            EditError::Parse(e) => write!(f, "TOML parse error: {}", e),
            EditError::Path(e) => write!(f, "Path error: {}", e),
            EditError::NotFound(s) => write!(f, "not found: {}", s),
            EditError::IndexOutOfBounds { index, len } => {
                write!(f, "index {} out of bounds (len {})", index, len)
            }
            EditError::NotATable(s) => write!(f, "not a table: {}", s),
            EditError::NotAnArray(s) => write!(f, "not an array: {}", s),
            EditError::InvalidValue(s) => write!(f, "invalid value: {}", s),
            EditError::Validation(errors) => {
                writeln!(f, "validation errors:")?;
                for e in errors {
                    writeln!(f, "  {}", e)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for EditError {}

impl From<std::io::Error> for EditError {
    fn from(e: std::io::Error) -> Self {
        EditError::Io(e)
    }
}

impl From<toml_edit::TomlError> for EditError {
    fn from(e: toml_edit::TomlError) -> Self {
        EditError::Parse(e)
    }
}

impl From<crate::path::PathParseError> for EditError {
    fn from(e: crate::path::PathParseError) -> Self {
        EditError::Path(e)
    }
}

/// A validation error with a field path.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

/// Config editor that wraps a toml_edit document for round-trip editing.
pub struct ConfigEditor {
    pub doc: DocumentMut,
    pub schema: ConfigNode,
}

impl ConfigEditor {
    /// Load a config file.
    pub fn from_file(path: &str) -> Result<Self, EditError> {
        let content = std::fs::read_to_string(path)?;
        let doc: DocumentMut = content.parse()?;
        Ok(ConfigEditor {
            doc,
            schema: <hyburn::config::SimConfig as SchemaProvider>::schema(),
        })
    }

    /// Parse a config from a string.
    pub fn from_str(s: &str) -> Result<Self, EditError> {
        let doc: DocumentMut = s.parse()?;
        Ok(ConfigEditor {
            doc,
            schema: <hyburn::config::SimConfig as SchemaProvider>::schema(),
        })
    }

    /// Write the document back to a file.
    pub fn save(&self, path: &str) -> Result<(), EditError> {
        std::fs::write(path, self.doc.to_string())?;
        Ok(())
    }

    /// Return the document as a string.
    pub fn to_string(&self) -> String {
        self.doc.to_string()
    }

    /// Get a value at the given path as a TOML string.
    pub fn get(&self, path: &str) -> Result<String, EditError> {
        let parsed = ConfigPath::parse(path)?;
        let item = navigate_to_item(&self.doc, &parsed)?;
        item_to_string(item, &parsed)
    }

    /// Set a value at the given path. The value is parsed as a TOML expression.
    pub fn set(&mut self, path: &str, value_str: &str) -> Result<(), EditError> {
        let parsed = ConfigPath::parse(path)?;
        if parsed.segments.is_empty() {
            return Err(EditError::InvalidValue("empty path".into()));
        }

        let value = parse_toml_value(value_str)?;
        let last = parsed.last().unwrap().clone();

        match last {
            PathSegment::Field(name) => {
                let parent_path = parsed.parent();
                let parent = if let Some(ref pp) = parent_path {
                    navigate_to_item_mut(&mut self.doc, pp)?
                } else {
                    self.doc.as_item_mut()
                };
                match parent {
                    Item::Table(table) => {
                        // Allow inserting new keys when the parent path contains
                        // an array index (i.e. we're inside a table array entry
                        // that may have been created empty). For regular tables,
                        // require the key to already exist.
                        let inside_array_entry = parent_path
                            .as_ref()
                            .map_or(false, |pp| {
                                pp.segments.iter().any(|s| matches!(s, PathSegment::Index(_)))
                            });
                        if !inside_array_entry && !table.contains_key(&name) {
                            return Err(EditError::NotFound(name.clone()));
                        }
                        if table.contains_key(&name) {
                            // Update existing key (preserves comments/formatting)
                            table[&name] = Item::Value(value);
                        } else {
                            // Insert new key (inside array entry only)
                            table.insert(&name, Item::Value(value));
                        }
                        Ok(())
                    }
                    _ => Err(EditError::NotATable(format!("at parent of '{}'", path))),
                }
            }
            PathSegment::Index(idx) => {
                let parent_path = parsed.parent().unwrap();
                let parent = navigate_to_item_mut(&mut self.doc, &parent_path)?;
                match parent {
                    Item::Value(Value::Array(arr)) => {
                        if idx >= arr.len() {
                            return Err(EditError::IndexOutOfBounds {
                                index: idx,
                                len: arr.len(),
                            });
                        }
                        arr.replace(idx, value);
                        Ok(())
                    }
                    Item::ArrayOfTables(_) => Err(EditError::InvalidValue(
                        "cannot set a table element directly; set a sub-field instead"
                            .into(),
                    )),
                    _ => Err(EditError::NotAnArray(format!("at index [{}]", idx))),
                }
            }
        }
    }

    /// Add an element to an array-of-tables at the given path.
    /// If `template` is Some, uses it as the initial content (TOML table string).
    /// If `model_name` is Some and the path is an array of subnetworks, populates
    /// the new table with model-aware defaults.
    pub fn add(
        &mut self,
        path: &str,
        template: Option<&str>,
        model_name: Option<&str>,
    ) -> Result<(), EditError> {
        let parsed = ConfigPath::parse(path)?;
        let item = navigate_to_item_mut(&mut self.doc, &parsed)?;

        match item {
            Item::ArrayOfTables(arr) => {
                let mut new_table = Table::new();
                populate_table_from_defaults(&mut new_table, template, model_name)?;
                arr.push(new_table);
                Ok(())
            }
            Item::None => {
                let parent_path = parsed.parent();
                let last = parsed.last().unwrap().clone();
                let parent = if let Some(ref pp) = parent_path {
                    navigate_to_item_mut(&mut self.doc, pp)?
                } else {
                    self.doc.as_item_mut()
                };

                match last {
                    PathSegment::Field(name) => match parent {
                        Item::Table(table) => {
                            let mut arr = toml_edit::ArrayOfTables::new();
                            let mut new_table = Table::new();
                            populate_table_from_defaults(&mut new_table, template, model_name)?;
                            arr.push(new_table);
                            table.insert(&name, Item::ArrayOfTables(arr));
                            Ok(())
                        }
                        _ => Err(EditError::NotATable(format!("at parent of '{}'", path))),
                    },
                    _ => Err(EditError::InvalidValue("cannot add to a non-field path".into())),
                }
            }
            _ => Err(EditError::NotAnArray(format!(
                "'{}' is not an array-of-tables",
                path
            ))),
        }
    }

    /// Remove an element from an array at the given path.
    /// The path must end with an index (e.g., `network.subnetworks[1]`).
    pub fn remove(&mut self, path: &str) -> Result<(), EditError> {
        let parsed = ConfigPath::parse(path)?;
        let last = parsed.last().cloned().ok_or_else(|| {
            EditError::InvalidValue("path must identify an element to remove".into())
        })?;

        match last {
            PathSegment::Index(idx) => {
                let parent_path = parsed.parent().unwrap();
                let parent = navigate_to_item_mut(&mut self.doc, &parent_path)?;

                match parent {
                    Item::ArrayOfTables(arr) => {
                        if idx >= arr.len() {
                            return Err(EditError::IndexOutOfBounds {
                                index: idx,
                                len: arr.len(),
                            });
                        }
                        arr.remove(idx);
                        Ok(())
                    }
                    Item::Value(Value::Array(arr)) => {
                        if idx >= arr.len() {
                            return Err(EditError::IndexOutOfBounds {
                                index: idx,
                                len: arr.len(),
                            });
                        }
                        arr.remove(idx);
                        Ok(())
                    }
                    _ => Err(EditError::NotAnArray(format!(
                        "'{}' is not an array",
                        parent_path
                    ))),
                }
            }
            PathSegment::Field(_) => Err(EditError::InvalidValue(
                "cannot remove a named field; specify an index to remove an array element"
                    .into(),
            )),
        }
    }

    /// List all keys/values at the given path (or root if None) as a tree.
    pub fn list(&self, path: Option<&str>) -> Result<String, EditError> {
        let item = if let Some(p) = path {
            let parsed = ConfigPath::parse(p)?;
            navigate_to_item(&self.doc, &parsed)?
        } else {
            self.doc.as_item()
        };

        let mut output = String::new();
        list_recursive(item, &mut String::new(), &mut output);
        Ok(output)
    }

    /// Validate the current document by deserializing to SimConfig and running validate().
    pub fn validate(&self) -> Result<(), Vec<ValidationError>> {
        let toml_str = self.doc.to_string();
        let cfg: hyburn::config::SimConfig = match toml::from_str(&toml_str) {
            Ok(cfg) => cfg,
            Err(e) => {
                return Err(vec![ValidationError {
                    path: String::new(),
                    message: format!("TOML parse error: {}", e),
                }]);
            }
        };
        if let Err(e) = cfg.validate() {
            return Err(vec![ValidationError {
                path: String::new(),
                message: format!("{}", e),
            }]);
        }
        Ok(())
    }

    /// Return the schema for the config.
    pub fn schema(&self) -> &ConfigNode {
        &self.schema
    }

    /// Get the schema node at a specific path.
    pub fn schema_at(&self, path: &str) -> Result<&ConfigNode, EditError> {
        let parsed = ConfigPath::parse(path)?;
        let mut node = &self.schema;
        for seg in &parsed.segments {
            match seg {
                PathSegment::Field(name) => {
                    node = node.child(name).ok_or_else(|| {
                        EditError::NotFound(format!("schema field '{}'", name))
                    })?;
                }
                PathSegment::Index(_) => {
                    if matches!(node.field_type, crate::schema::FieldType::TableArray)
                        && let Some(child) = node.children.first() {
                            node = child;
                        }
                }
            }
        }
        Ok(node)
    }
}

/// Populate a new table with template or model-aware defaults.
fn populate_table_from_defaults(
    table: &mut Table,
    template: Option<&str>,
    model_name: Option<&str>,
) -> Result<(), EditError> {
    if let Some(tmpl) = template {
        let tmpl_doc: DocumentMut = tmpl.parse().map_err(|e| {
            EditError::InvalidValue(format!("invalid template TOML: {}", e))
        })?;
        for (key, value) in tmpl_doc.as_table().iter() {
            table.insert(key, value.clone());
        }
    } else if let Some(model) = model_name
        && let Some(info) = crate::schema::get_model_info(model) {
        table.insert("model", Item::Value(Value::from(model.to_string())));
        table.insert("nnodes", Item::Value(Value::from(1i64)));
        table.insert("nmodes", Item::Value(Value::from(1i64)));

        let params: toml_edit::Array =
            (0..info.nparams).map(|_| Value::from(0.0f64)).collect();
        table.insert("params", Item::Value(Value::Array(params)));

        let state_len = info.nvar;
        let state: toml_edit::Array =
            (0..state_len).map(|_| Value::from(0.0f64)).collect();
        table.insert("initial_state", Item::Value(Value::Array(state)));
    }
    Ok(())
}

/// Navigate to an item in the document by path (immutable).
fn navigate_to_item<'a>(
    doc: &'a DocumentMut,
    path: &ConfigPath,
) -> Result<&'a Item, EditError> {
    let mut item = doc.as_item();
    for seg in &path.segments {
        match seg {
            PathSegment::Field(name) => {
                item = item
                    .get(name.as_str())
                    .ok_or_else(|| EditError::NotFound(name.clone()))?;
            }
            PathSegment::Index(idx) => {
                let len = match item {
                    Item::ArrayOfTables(arr) => arr.len(),
                    Item::Value(Value::Array(arr)) => arr.len(),
                    _ => {
                        return Err(EditError::NotAnArray(format!("at index [{}]", idx)));
                    }
                };
                item = item
                    .get(*idx)
                    .ok_or(EditError::IndexOutOfBounds { index: *idx, len })?;
            }
        }
    }
    Ok(item)
}

/// Navigate to an item in the document by path (mutable).
fn navigate_to_item_mut<'a>(
    doc: &'a mut DocumentMut,
    path: &ConfigPath,
) -> Result<&'a mut Item, EditError> {
    let mut item = doc.as_item_mut();
    for seg in &path.segments {
        match seg {
            PathSegment::Field(name) => {
                item = item
                    .get_mut(name.as_str())
                    .ok_or_else(|| EditError::NotFound(name.clone()))?;
            }
            PathSegment::Index(idx) => {
                let len = match item {
                    Item::ArrayOfTables(arr) => arr.len(),
                    Item::Value(Value::Array(arr)) => arr.len(),
                    _ => {
                        return Err(EditError::NotAnArray(format!("at index [{}]", idx)));
                    }
                };
                item = item
                    .get_mut(*idx)
                    .ok_or(EditError::IndexOutOfBounds { index: *idx, len })?;
            }
        }
    }
    Ok(item)
}

/// Convert a toml_edit Item to a display string.
fn item_to_string(item: &Item, path: &ConfigPath) -> Result<String, EditError> {
    match item {
        Item::None => Err(EditError::NotFound(format!("{}", path))),
        Item::Value(v) => Ok(format_value(v)),
        Item::Table(table) => {
            let mut parts = Vec::new();
            for (key, value) in table.iter() {
                match value {
                    Item::Value(v) => parts.push(format!("{} = {}", key, format_value(v))),
                    Item::Table(_) => parts.push(format!("{} = {{...}}", key)),
                    Item::ArrayOfTables(_) => parts.push(format!("{} = [[...]]", key)),
                    Item::None => {}
                }
            }
            Ok(parts.join("\n"))
        }
        Item::ArrayOfTables(arr) => Ok(format!("[array of {} tables]", arr.len())),
    }
}

/// Format a Value for display.
fn format_value(v: &Value) -> String {
    match v {
        Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(format_value).collect();
            format!("[{}]", items.join(", "))
        }
        Value::InlineTable(t) => format!("{}", t),
        _ => format!("{}", v).trim_start().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOML: &str = r#"
sim_length = 1000.0
dt = 0.1

[network]
[[network.subnetworks]]
model = "Generic2dOscillator"
nnodes = 2
params = [1.0, 0.0, -2.0, -10.0, 0.0, 0.02, 3.0, 1.0, 0.0, 1.0, 1.0, 1.0]
initial_state = [0.0, 0.5, 0.0, 0.5]
"#;

    #[test]
    fn editor_from_str_and_get() {
        let editor = ConfigEditor::from_str(TEST_TOML).unwrap();
        assert_eq!(editor.get("sim_length").unwrap(), "1000.0");
        assert_eq!(editor.get("network.subnetworks[0].model").unwrap(), "\"Generic2dOscillator\"");
    }

    #[test]
    fn editor_set_scalar() {
        let mut editor = ConfigEditor::from_str(TEST_TOML).unwrap();
        editor.set("sim_length", "500.0").unwrap();
        assert_eq!(editor.get("sim_length").unwrap(), "500.0");
    }

    #[test]
    fn editor_set_array_element() {
        let mut editor = ConfigEditor::from_str(TEST_TOML).unwrap();
        editor.set("network.subnetworks[0].params[0]", "99.0").unwrap();
        assert!(editor.get("network.subnetworks[0].params").unwrap().contains("99.0"));
    }

    #[test]
    fn editor_add_remove_subnetwork() {
        let mut editor = ConfigEditor::from_str(TEST_TOML).unwrap();
        let _before = editor.to_string();

        editor.add("network.subnetworks", None, Some("ReducedWongWang")).unwrap();
        assert!(editor.get("network.subnetworks[1].model").unwrap().contains("ReducedWongWang"));

        editor.remove("network.subnetworks[1]").unwrap();
        let _after = editor.to_string();
        // Should be equivalent to original (modulo whitespace/formatting changes)
        // Just check that the second subnetwork is gone
        assert!(editor.get("network.subnetworks[1]").is_err());
    }

    #[test]
    fn editor_validate_ok() {
        let editor = ConfigEditor::from_str(TEST_TOML).unwrap();
        assert!(editor.validate().is_ok());
    }

    #[test]
    fn editor_validate_bad_dt() {
        let mut editor = ConfigEditor::from_str(TEST_TOML).unwrap();
        editor.set("dt", "-0.1").unwrap();
        assert!(editor.validate().is_err());
    }

    #[test]
    fn editor_list_root() {
        let editor = ConfigEditor::from_str(TEST_TOML).unwrap();
        let out = editor.list(None).unwrap();
        assert!(out.contains("sim_length"));
        assert!(out.contains("network.subnetworks[0].model"));
    }

    #[test]
    fn test_set_nonexistent_path() {
        let mut editor = ConfigEditor::from_str(TEST_TOML).unwrap();
        assert!(editor.set("nonexistent", "5").is_err());
    }

    #[test]
    fn test_set_index_out_of_bounds() {
        let mut editor = ConfigEditor::from_str(TEST_TOML).unwrap();
        assert!(editor.set("network.subnetworks[0].params[999]", "5.0").is_err());
    }

    #[test]
    fn editor_roundtrip_preserve_comments() {
        let toml = "# preamble\nsim_length = 1000.0\n\n[network]\n# network comment\n\n";
        let mut editor = ConfigEditor::from_str(toml).unwrap();
        editor.set("sim_length", "500.0").unwrap();
        let out = editor.to_string();
        assert!(out.contains("# preamble"));
        assert!(out.contains("# network comment"));
        assert!(out.contains("500.0"));
    }

    #[test]
    fn test_schema_at() {
        let editor = ConfigEditor::from_str(TEST_TOML).unwrap();
        let node = editor.schema_at("sim_length").unwrap();
        assert_eq!(node.name, "sim_length");
        assert!(matches!(
            node.field_type,
            crate::schema::FieldType::Scalar(crate::schema::ScalarType::Float)
        ));

        let node = editor.schema_at("network.subnetworks[0].model").unwrap();
        assert_eq!(node.name, "model");
        assert!(matches!(node.field_type, crate::schema::FieldType::Enum(_)));

        assert!(editor.schema_at("nonexistent").is_err());
    }
}

/// Recursively list an item as a tree of dotted paths.
fn list_recursive(item: &Item, prefix: &mut String, output: &mut String) {
    match item {
        Item::Table(table) => {
            for (key, value) in table.iter() {
                let old_len = prefix.len();
                if !prefix.is_empty() {
                    prefix.push('.');
                }
                prefix.push_str(key);

                match value {
                    Item::Value(v) => {
                        output.push_str(&format!("{} = {}\n", prefix, format_value(v)));
                    }
                    Item::Table(_) => {
                        list_recursive(value, prefix, output);
                    }
                    Item::ArrayOfTables(arr) => {
                        for (i, tbl) in arr.iter().enumerate() {
                            let mut arr_prefix = format!("{}[{}]", prefix, i);
                            list_recursive(&Item::Table(tbl.clone()), &mut arr_prefix, output);
                        }
                    }
                    Item::None => {}
                }

                prefix.truncate(old_len);
            }
        }
        Item::ArrayOfTables(arr) => {
            for (i, tbl) in arr.iter().enumerate() {
                let mut arr_prefix = format!("{}[{}]", prefix, i);
                list_recursive(&Item::Table(tbl.clone()), &mut arr_prefix, output);
            }
        }
        Item::Value(v) => {
            output.push_str(&format!("{} = {}\n", prefix, format_value(v)));
        }
        Item::None => {}
    }
}

/// Parse a TOML value expression string.
fn parse_toml_value(s: &str) -> Result<Value, EditError> {
    let wrapper = format!("x = {}", s);
    let doc: DocumentMut = wrapper.parse().map_err(|e| {
        EditError::InvalidValue(format!("invalid TOML value '{}': {}", s, e))
    })?;
    doc.as_table()
        .get("x")
        .and_then(|item| {
            if let Item::Value(v) = item {
                Some(v.clone())
            } else {
                None
            }
        })
        .ok_or_else(|| EditError::InvalidValue(format!("'{}' is not a valid TOML value", s)))
}

