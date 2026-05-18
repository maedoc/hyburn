//! Application state and event handling for the TUI.

use std::collections::HashMap;

use crate::logger::Logger;
use crate::preview::PreviewPane;
use crate::tree_view::TreeView;
use crate::widgets::{self, ConfirmedAction, EditPopup, Popup, SearchPopup};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use hyburn_config_lib::{ConfigEditor, ConfigNode, FieldType, ScalarType, path::PathSegment};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, Paragraph},
    Frame,
};

/// Which pane has focus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Tree,
    Preview,
}

/// Application state.
pub struct App {
    pub editor: ConfigEditor,
    pub schema: ConfigNode,
    pub tree: TreeView,
    pub preview: PreviewPane,
    pub focus: Focus,
    pub file_path: String,
    pub modified: bool,
    pub status_message: String,
    pub popup: Option<Popup>,
    pub logger: Option<Logger>,
    pub validation_errors: Vec<hyburn_config_lib::editor::ValidationError>,
    pub undo_stack: Vec<String>,
}

impl App {
    pub fn new(file_path: &str, logger: Option<Logger>) -> Result<Self, Box<dyn std::error::Error>> {
        let editor = ConfigEditor::from_file(file_path)?;
        let schema = hyburn_config_lib::root_schema();
        let preview_text = editor.to_string();
        let tree = TreeView::new(&schema);
        let preview = PreviewPane::new(&preview_text);

        let mut app = App {
            editor,
            schema,
            tree,
            preview,
            focus: Focus::Tree,
            file_path: file_path.to_string(),
            modified: false,
            status_message: format!("Opened: {}", file_path),
            popup: None,
            logger,
            validation_errors: Vec::new(),
            undo_stack: Vec::new(),
        };

        // Populate initial values
        app.populate_tree_values();
        Ok(app)
    }

    /// Populate tree with current document values and array lengths.
    fn populate_tree_values(&mut self) {
        let mut values = HashMap::new();
        let mut array_lengths = HashMap::new();

        // Collect paths first to avoid borrow issues
        let paths: Vec<String> = self.tree.paths.clone();
        for path in &paths {
            if let Ok(val) = self.editor.get(path) {
                values.insert(path.clone(), val);
            }
        }

        // Also populate summary values for array entries (model, monitor_type, etc.)
        // even when the entries are collapsed
        for path in &paths {
            // If this path is an array entry (ends with [N]), also fetch its common summary fields
            if path.contains('[') && path.matches('[').count() == 1 {
                for field in &["model", "monitor_type", "temporal"] {
                    let sub_path = format!("{}.{}", path, field);
                    if !values.contains_key(&sub_path) {
                        if let Ok(val) = self.editor.get(&sub_path) {
                            values.insert(sub_path, val);
                        }
                    }
                }
            }
        }

        // Get array lengths for table arrays
        self.collect_array_lengths(&self.schema.clone(), String::new(), &mut array_lengths);

        self.tree.set_values(values);
        self.tree.set_array_lengths(array_lengths);
        // Rebuild labels with new values
        self.tree.refresh_labels();

        // Second pass: populate newly visible paths after label refresh
        let new_paths: Vec<String> = self.tree.paths.clone();
        let mut more_values = HashMap::new();
        for path in &new_paths {
            if let Ok(val) = self.editor.get(path) {
                more_values.insert(path.clone(), val);
            }
        }
        // Merge
        let mut all_values = self.tree.values.clone();
        all_values.extend(more_values);
        self.tree.set_values(all_values);
        self.tree.refresh_labels();
    }

    /// Recursively collect array-of-tables lengths from the editor document.
    fn collect_array_lengths(&self, node: &ConfigNode, parent_path: String, lengths: &mut HashMap<String, usize>) {
        for child in &node.children {
            let path = if parent_path.is_empty() {
                child.name.to_string()
            } else {
                format!("{}.{}", parent_path, child.name)
            };

            if matches!(child.field_type, FieldType::TableArray) {
                // Try to get the length from the document
                if let Ok(item) = self.editor.get(&path) {
                    // The get for an array of tables returns "[array of N tables]"
                    // Parse N from that string
                    if let Some(n) = item.trim()
                        .strip_prefix("[array of ")
                        .and_then(|s| s.strip_suffix(" tables]"))
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        lengths.insert(path.clone(), n);
                    }
                }
            }

            // Recurse into children (only for structs, not table arrays — those are handled separately)
            if matches!(child.field_type, FieldType::Struct | FieldType::UntaggedEnum(_)) {
                self.collect_array_lengths(child, path, lengths);
            }
        }
    }

    /// Handle a key event. Returns true if the app should quit.
    pub fn handle_key(&mut self, key: KeyEvent) -> bool {
        let code_str = format!("{:?}", key.code);
        if let Some(ref l) = self.logger {
            l.log_key(&code_str);
        }

        // Global keys
        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), _) if self.popup.is_none() => {
                if self.modified {
                    self.status_message = "Unsaved changes! Press Ctrl-Q to force quit, or Ctrl-S to save.".into();
                    return false;
                }
                if let Some(ref l) = self.logger {
                    l.log_quit(self.modified);
                }
                return true;
            }
            (KeyCode::Char('q'), KeyModifiers::CONTROL) => {
                if let Some(ref l) = self.logger {
                    l.log_quit(self.modified);
                }
                return true;
            }
            (KeyCode::Char('s'), KeyModifiers::CONTROL) => {
                self.save();
                return false;
            }
            (KeyCode::Char('z'), KeyModifiers::CONTROL) => {
                self.undo();
                return false;
            }
            (KeyCode::Char('?'), _) if self.popup.is_none() => {
                self.popup = Some(Popup::Help);
                return false;
            }
            (KeyCode::Char('/'), _) if self.popup.is_none() => {
                self.popup = Some(Popup::Search(SearchPopup::new()));
                return false;
            }
            _ => {}
        }

        // Handle popup first
        if let Some(ref mut popup) = self.popup {
            match popup {
                Popup::Edit(ep) => {
                    match key.code {
                        KeyCode::Enter => {
                            let value = ep.get_value();
                            let path = ep.path.clone();
                            self.popup = None;
                            self.apply_edit(&path, &value);
                        }
                        KeyCode::Esc => {
                            self.popup = None;
                        }
                        KeyCode::Char(c) => ep.input(c),
                        KeyCode::Backspace => ep.backspace(),
                        KeyCode::Tab => {
                            // Tab in edit popup: accept and move to next field
                            let value = ep.get_value();
                            let path = ep.path.clone();
                            self.popup = None;
                            self.apply_edit(&path, &value);
                            self.tree.move_down();
                            self.update_status_for_selection();
                        }
                        _ => {}
                    }
                }
                Popup::EnumSelect { items, selected, path, field_name, action } => {
                    match key.code {
                        KeyCode::Up | KeyCode::Char('k') => {
                            if *selected > 0 {
                                *selected -= 1;
                            }
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            if *selected < items.len() - 1 {
                                *selected += 1;
                            }
                        }
                        KeyCode::Enter => {
                            let value = items[*selected].clone();
                            let path = path.clone();
                            let action = *action;
                            let field_name = field_name.clone();
                            self.popup = None;
                            match action {
                                widgets::EnumAction::Edit => {
                                    // Quote the value as a TOML string
                                    let toml_value = format!("\"{}\"", value);
                                    self.apply_edit(&path, &toml_value);
                                }
                                widgets::EnumAction::Add => {
                                    self.push_undo();
                                    // Build a template with the selected enum value
                                    // For subnetworks, use model_name; for others, use template
                                    let is_subnetwork = value == "model" || path.contains("subnetwork");
                                    let result = if is_subnetwork {
                                        self.editor.add(&path, None, Some(&value))
                                    } else {
                                        let template = format!("{} = \"{}\"", field_name, value);
                                        self.editor.add(&path, Some(&template), None)
                                    };
                                    if let Err(e) = result {
                                        self.status_message = format!("Error adding element: {}", e);
                                    } else {
                                        self.modified = true;
                                        self.refresh_preview();
                                        self.status_message = format!("Added {} to {}", value, path);
                                        self.populate_tree_values();
                                        self.tree.refresh();
                                    }
                                }
                            }
                        }
                        KeyCode::Esc => {
                            self.popup = None;
                        }
                        _ => {}
                    }
                }
                Popup::VariantSelect { labels, selected, path, types, field_name, current_value } => {
                    match key.code {
                        KeyCode::Up | KeyCode::Char('k') => {
                            if *selected > 0 {
                                *selected -= 1;
                            }
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            if *selected < labels.len() - 1 {
                                *selected += 1;
                            }
                        }
                        KeyCode::Enter => {
                            let selected_idx = *selected;
                            let path = path.clone();
                            let field_name = field_name.clone();
                            let current = current_value.clone();
                            let selected_type = &types[selected_idx];
                            let initial = Self::variant_initial_value(&current, selected_type);
                            let hint = self.build_edit_hint(&path, &initial);
                            self.popup = Some(Popup::Edit(EditPopup::new(
                                &path,
                                &field_name,
                                &initial,
                                &hint,
                            )));
                        }
                        KeyCode::Esc => {
                            self.popup = None;
                        }
                        _ => {}
                    }
                }
                Popup::Confirm { message: _, confirmed_action } => {
                    match key.code {
                        KeyCode::Char('y') | KeyCode::Char('Y') => {
                            let action = confirmed_action.clone();
                            self.popup = None;
                            match action {
                                ConfirmedAction::Remove { path } => {
                                    self.do_remove(&path);
                                }
                            }
                        }
                        KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                            self.popup = None;
                            self.status_message = "Cancelled".into();
                        }
                        _ => {}
                    }
                }
                Popup::Help => {
                    match key.code {
                        KeyCode::Char('?') | KeyCode::Esc => {
                            self.popup = None;
                        }
                        _ => {}
                    }
                }
                Popup::Search(sp) => {
                    match key.code {
                        KeyCode::Esc => {
                            // Clear filter and close search
                            self.tree.clear_filter();
                            self.populate_tree_values();
                            self.popup = None;
                        }
                        KeyCode::Enter => {
                            // Jump to first match and close search
                            if !sp.query().is_empty() {
                                let query = sp.query().to_string();
                                self.tree.jump_to_match(&query);
                            }
                            self.tree.clear_filter();
                            self.populate_tree_values();
                            self.popup = None;
                        }
                        KeyCode::Backspace => {
                            sp.backspace();
                            self.tree.apply_filter(sp.query());
                            self.populate_tree_values();
                        }
                        KeyCode::Char(c) => {
                            sp.input_char(c);
                            self.tree.apply_filter(sp.query());
                            self.populate_tree_values();
                        }
                        _ => {}
                    }
                }
            }
            return false;
        }

        // Pane-specific keys
        match self.focus {
            Focus::Tree => self.handle_tree_key(key),
            Focus::Preview => self.handle_preview_key(key),
        }

        false
    }

    fn handle_tree_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                self.tree.move_up();
                self.update_status_for_selection();
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.tree.move_down();
                self.update_status_for_selection();
            }
            KeyCode::Left | KeyCode::Char('h') => {
                self.tree.collapse();
                self.populate_tree_values();
            }
            KeyCode::Right | KeyCode::Char('l') => {
                self.tree.expand();
                self.populate_tree_values();
            }
            KeyCode::Enter => self.activate_selected(),
            KeyCode::Tab => self.focus = Focus::Preview,
            KeyCode::Char('a') => self.add_element(),
            KeyCode::Char('d') => self.remove_selected(),
            _ => {}
        }
    }

    fn handle_preview_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Tab => self.focus = Focus::Tree,
            KeyCode::Up | KeyCode::Char('k') => self.preview.scroll_up(),
            KeyCode::Down | KeyCode::Char('j') => self.preview.scroll_down(),
            _ => {}
        }
    }

    /// Activate the selected node: expand if expandable, edit if leaf.
    fn activate_selected(&mut self) {
        if self.tree.selected_can_expand() {
            self.tree.expand();
            self.populate_tree_values();
            self.update_status_for_selection();
            return;
        }

        // It's a leaf node — edit it
        self.edit_selected();
    }

    /// Update status bar to show info about the currently selected node.
    fn update_status_for_selection(&mut self) {
        let path = match self.tree.selected_path() {
            Some(p) => p,
            None => return,
        };

        let schema_node = match self.editor.schema_at(&path) {
            Ok(n) => n.clone(),
            Err(_) => return,
        };

        let mut parts = Vec::new();

        // Description
        if !schema_node.description.is_empty() {
            parts.push(schema_node.description.to_string());
        }

        // Constraints
        for c in &schema_node.constraints {
            match c {
                hyburn_config_lib::schema::Constraint::Positive => parts.push("must be > 0".into()),
                hyburn_config_lib::schema::Constraint::MinLen(n) => parts.push(format!("min {} items", n)),
                hyburn_config_lib::schema::Constraint::ExactLen(n) => parts.push(format!("exactly {} items", n)),
                hyburn_config_lib::schema::Constraint::Custom(desc) => parts.push(desc.to_string()),
            }
        }

        // Enum variant count
        if let FieldType::Enum(variants) = &schema_node.field_type {
            parts.push(format!("{} options", variants.len()));
        }

        // Default value hint
        if !schema_node.required {
            parts.push("optional".into());
        }

        if let Some(error) = self.validation_errors.iter().find(|e| e.path == path) {
            parts.push(format!("ERROR: {}", error.message));
        }

        self.status_message = format!("{}: {}", path, parts.join(", "));
    }

    fn build_edit_hint(&self, path: &str, current_value: &str) -> String {
        let schema_node = match self.editor.schema_at(path) {
            Ok(n) => n,
            Err(_) => return String::new(),
        };

        let mut parts = Vec::new();

        if !schema_node.description.is_empty() {
            parts.push(schema_node.description.to_string());
        }

        for c in &schema_node.constraints {
            match c {
                hyburn_config_lib::schema::Constraint::Positive => parts.push("must be > 0".into()),
                hyburn_config_lib::schema::Constraint::MinLen(n) => parts.push(format!("min {} items", n)),
                hyburn_config_lib::schema::Constraint::ExactLen(n) => parts.push(format!("exactly {} items", n)),
                hyburn_config_lib::schema::Constraint::Custom(desc) => parts.push(desc.to_string()),
            }
        }

        if let FieldType::Enum(variants) = &schema_node.field_type {
            parts.push(format!("{} options", variants.len()));
        }

        if let FieldType::Scalar(ScalarType::Path) = &schema_node.field_type {
            let trimmed = current_value.trim_matches('"');
            let exists = std::path::Path::new(trimmed).exists();
            parts.push(if exists { "file exists" } else { "file not found" }.to_string());
        }

        parts.join(" | ")
    }

    fn variant_label(field_type: &FieldType) -> String {
        match field_type {
            FieldType::Scalar(ScalarType::Float) => "Scalar (number)".to_string(),
            FieldType::Scalar(ScalarType::Path) => "Path (file)".to_string(),
            FieldType::Scalar(ScalarType::String) => "Scalar (string)".to_string(),
            FieldType::Scalar(ScalarType::Integer) => "Scalar (integer)".to_string(),
            FieldType::Scalar(ScalarType::Boolean) => "Scalar (boolean)".to_string(),
            FieldType::Array { item_type, .. } => format!("Array [{}...]", item_type),
            FieldType::Struct => "Table { ... }".to_string(),
            FieldType::TableArray => "[[Table]]".to_string(),
            FieldType::Enum(v) => format!("Enum ({})", v.join(" | ")),
            FieldType::UntaggedEnum(_) => "UntaggedEnum".to_string(),
        }
    }

    fn variant_initial_value(current: &str, selected_type: &FieldType) -> String {
        let is_array = current.starts_with('[');
        match selected_type {
            FieldType::Scalar(ScalarType::Float) => {
                if is_array { "0.0".to_string() } else { current.to_string() }
            }
            FieldType::Scalar(ScalarType::Integer) => {
                if is_array { "0".to_string() } else { current.to_string() }
            }
            FieldType::Scalar(ScalarType::String) => {
                if is_array { "\"\"".to_string() } else { current.to_string() }
            }
            FieldType::Scalar(ScalarType::Boolean) => {
                if is_array { "false".to_string() } else { current.to_string() }
            }
            FieldType::Scalar(ScalarType::Path) => current.to_string(),
            FieldType::Array { .. } => {
                if is_array { current.to_string() } else { "[]".to_string() }
            }
            _ => current.to_string(),
        }
    }

    fn edit_selected(&mut self) {
        let path = match self.tree.selected_path() {
            Some(p) => p,
            None => return,
        };

        let schema_node = match self.editor.schema_at(&path) {
            Ok(n) => n.clone(),
            Err(_) => return,
        };

        // Boolean toggle on Enter
        if let FieldType::Scalar(ScalarType::Boolean) = &schema_node.field_type {
            let current = self.editor.get(&path).unwrap_or_else(|_| "false".to_string());
            let new_val = if current == "true" { "false" } else { "true" };
            self.apply_edit(&path, new_val);
            return;
        }

        match &schema_node.field_type {
            FieldType::Enum(variants) => {
                self.popup = Some(Popup::EnumSelect {
                    items: variants.iter().map(|s| s.to_string()).collect(),
                    selected: 0,
                    path: path.clone(),
                    field_name: schema_node.name.to_string(),
                    action: widgets::EnumAction::Edit,
                });
            }
            FieldType::UntaggedEnum(variants) => {
                let current = self.editor.get(&path).unwrap_or_default();
                let labels: Vec<String> = variants.iter().map(Self::variant_label).collect();
                self.popup = Some(Popup::VariantSelect {
                    labels,
                    types: variants.clone(),
                    selected: 0,
                    path: path.clone(),
                    field_name: schema_node.name.to_string(),
                    current_value: current,
                });
            }
            _ => {
                let current = self.editor.get(&path).unwrap_or_default();
                let hint = self.build_edit_hint(&path, &current);
                self.popup = Some(Popup::Edit(EditPopup::new(
                    &path,
                    &schema_node.name,
                    &current,
                    &hint,
                )));
            }
        }
    }

    fn add_element(&mut self) {
        let path = match self.tree.selected_path() {
            Some(p) => p,
            None => return,
        };

        // Check if the selected node is a table array
        if let Ok(schema_node) = self.editor.schema_at(&path) {
            if matches!(
                schema_node.field_type,
                FieldType::TableArray
            ) {
                let is_subnetwork = schema_node
                    .children
                    .first()
                    .and_then(|c| c.children.first())
                    .map_or(false, |c| c.name == "model");
                if is_subnetwork {
                    let models = hyburn_config_lib::schema::model_names()
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect();
                    self.popup = Some(Popup::EnumSelect {
                        items: models,
                        selected: 0,
                        path: path.clone(),
                        field_name: "model".to_string(),
                        action: widgets::EnumAction::Add,
                    });
                } else {
                    // Check if the struct has an enum as first field (e.g. monitor_type)
                    // If so, show a picker so the entry gets a meaningful type
                    if let Some(struct_node) = schema_node.children.first() {
                        if let Some(first_field) = struct_node.children.first() {
                            if let FieldType::Enum(variants) = &first_field.field_type {
                                let items = variants.iter().map(|s| s.to_string()).collect();
                                self.popup = Some(Popup::EnumSelect {
                                    items,
                                    selected: 0,
                                    path: path.clone(),
                                    field_name: first_field.name.to_string(),
                                    action: widgets::EnumAction::Add,
                                });
                                return;
                            }
                        }
                    }
                    self.push_undo();
                    if let Err(e) = self.editor.add(&path, None, None) {
                        self.status_message = format!("Error adding element: {}", e);
                    } else {
                        self.modified = true;
                        self.refresh_preview();
                        self.status_message = format!("Added element to {}", path);
                        self.populate_tree_values();
                        self.tree.refresh();
                    }
                }
            }
        }
    }

    fn remove_selected(&mut self) {
        let path = match self.tree.selected_path() {
            Some(p) => p,
            None => return,
        };

        // Only remove if the path ends with an index
        let has_index = match hyburn_config_lib::ConfigPath::parse(&path) {
            Ok(parsed) => matches!(parsed.last(), Some(PathSegment::Index(_))),
            Err(_) => false,
        };

        if has_index {
            self.popup = Some(Popup::Confirm {
                message: format!("Remove {}?", path),
                confirmed_action: ConfirmedAction::Remove { path },
            });
        } else {
            self.status_message = "Select an array element to remove (e.g., subnetworks[0])".into();
        }
    }

    fn do_remove(&mut self, path: &str) {
        self.push_undo();
        if let Err(e) = self.editor.remove(path) {
            self.status_message = format!("Error removing element: {}", e);
        } else {
            self.modified = true;
            self.refresh_preview();
            self.status_message = format!("Removed {}", path);
            self.populate_tree_values();
            self.tree.refresh();
        }
    }

    fn apply_edit(&mut self, path: &str, value: &str) {
        self.push_undo();
        if let Some(ref l) = self.logger {
            l.log_action(&format!("edit path={} value={}", path, value));
        }
        match self.editor.set(path, value) {
            Ok(()) => {
                self.modified = true;
                self.refresh_preview();
                self.status_message = format!("Set {} = {}", path, value);
                self.populate_tree_values();
                if let Some(ref l) = self.logger {
                    l.log_result(path, true);
                }
            }
            Err(e) => {
                self.status_message = format!("Error: {}", e);
                if let Some(ref l) = self.logger {
                    l.log_result(path, false);
                }
            }
        }
    }

    fn save(&mut self) {
        if let Some(ref l) = self.logger {
            l.log_action(&format!("save path={}", self.file_path));
        }
        match self.editor.save(&self.file_path) {
            Ok(()) => {
                self.modified = false;
                match self.editor.validate() {
                    Ok(()) => {
                        self.status_message = format!("Saved: {} (valid)", self.file_path);
                        self.validation_errors.clear();
                        self.tree.set_validation_errors(vec![]);
                    }
                    Err(errors) => {
                        self.status_message = format!("Saved: {} ({} validation errors)", self.file_path, errors.len());
                        let err_clone = errors.clone();
                        self.validation_errors = errors;
                        self.tree.set_validation_errors(err_clone);
                    }
                }
                if let Some(ref l) = self.logger {
                    l.log_result(&self.file_path, true);
                }
            }
            Err(e) => {
                self.status_message = format!("Error saving: {}", e);
                if let Some(ref l) = self.logger {
                    l.log_result(&self.file_path, false);
                }
            }
        }
    }

    /// Log quit event.
    pub fn log_quit(&self) {
        if let Some(ref l) = self.logger {
            l.log_quit(self.modified);
        }
    }

    fn refresh_preview(&mut self) {
        self.preview.set_content(&self.editor.to_string());
    }

    fn push_undo(&mut self) {
        self.undo_stack.push(self.editor.to_string());
        if self.undo_stack.len() > 20 {
            self.undo_stack.remove(0);
        }
    }

    fn undo(&mut self) {
        if let Some(state) = self.undo_stack.pop() {
            match ConfigEditor::from_str(&state) {
                Ok(editor) => {
                    self.editor = editor;
                    self.modified = true;
                    self.refresh_preview();
                    self.populate_tree_values();
                    self.tree.refresh();
                    self.status_message = "Undo: restored previous state".into();
                }
                Err(e) => {
                    self.status_message = format!("Undo failed: {}", e);
                }
            }
        } else {
            self.status_message = "Nothing to undo".into();
        }
    }

    /// Render the application.
    pub fn render(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(10),
                Constraint::Length(1),
            ])
            .split(area);

        let main_area = chunks[0];
        let status_area = chunks[1];

        // Split main area into tree + preview
        let panes = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(main_area);

        // Render tree pane
        let tree_style = if self.focus == Focus::Tree {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let tree_block = Block::default()
            .title("Config Tree")
            .borders(Borders::ALL)
            .border_style(tree_style);
        self.tree.render(f, panes[0], tree_block);

        // Render preview pane
        let preview_style = if self.focus == Focus::Preview {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        };
        let preview_block = Block::default()
            .title(if self.modified {
                "TOML Preview (modified)"
            } else {
                "TOML Preview"
            })
            .borders(Borders::ALL)
            .border_style(preview_style);
        self.preview.render(f, panes[1], preview_block);

        // Status bar
        let status = Paragraph::new(Line::from(vec![
            Span::styled(
                &self.status_message,
                Style::default().fg(Color::White),
            ),
        ]));
        f.render_widget(status, status_area);

        // Render popup if active
        if let Some(ref popup) = self.popup {
            let popup_area = match popup {
                Popup::Help => centered_rect(60, 50, area),
                _ => centered_rect(60, 20, area),
            };
            f.render_widget(Clear, popup_area);
            match popup {
                Popup::Edit(ep) => {
                    ep.render(f, popup_area);
                }
                Popup::EnumSelect {
                    items,
                    selected,
                    field_name,
                    ..
                } => {
                    widgets::render_enum_select(f, popup_area, field_name, items, *selected);
                }
                Popup::VariantSelect {
                    labels,
                    selected,
                    field_name,
                    ..
                } => {
                    widgets::render_enum_select(f, popup_area, field_name, labels, *selected);
                }
                Popup::Confirm { message, .. } => {
                    widgets::render_confirm(f, popup_area, message);
                }
                Popup::Help => {
                    widgets::render_help(f, popup_area);
                }
                Popup::Search(sp) => {
                    sp.render(f, popup_area);
                }
            }
        }
    }
}

/// Create a centered rectangle.
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
