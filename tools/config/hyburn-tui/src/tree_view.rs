//! Tree view widget for the config schema.
//!
//! Supports lazy expand/collapse: only children of expanded nodes are visible.
//! Shows inline current values from the document next to leaf fields.

use std::collections::{HashMap, HashSet};

use hyburn_config_lib::{ConfigNode, FieldType};
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, List, ListItem, ListState},
    Frame,
};

/// A visible item in the tree view.
#[derive(Debug, Clone)]
struct TreeItem {
    label: String,
    path: String,
    depth: usize,
    is_expandable: bool,
    is_expanded: bool,
}

/// Tree view state and rendering.
pub struct TreeView {
    schema: ConfigNode,
    items: Vec<TreeItem>,
    pub state: ListState,
    /// Set of dotted paths that are currently expanded.
    expanded: HashSet<String>,
    /// Paths parallel to items, for quick lookup.
    pub paths: Vec<String>,
    /// Current values from the document, keyed by dotted path.
    pub values: HashMap<String, String>,
    /// Number of elements in each table array, keyed by dotted path.
    array_lengths: HashMap<String, usize>,
    /// Paths with validation errors.
    validation_error_paths: HashSet<String>,
    /// Saved expanded state before search filter.
    _pre_filter_expanded: Option<HashSet<String>>,
    /// Whether a search filter is currently active.
    filter_active: bool,
}

impl TreeView {
    pub fn new(schema: &ConfigNode) -> Self {
        let mut tv = TreeView {
            schema: schema.clone(),
            items: Vec::new(),
            state: ListState::default(),
            expanded: HashSet::new(),
            paths: Vec::new(),
            values: HashMap::new(),
            array_lengths: HashMap::new(),
            validation_error_paths: HashSet::new(),
            _pre_filter_expanded: None,
            filter_active: false,
        };
        // Expand root by default so top-level fields are visible
        tv.expanded.insert("(root)".to_string());
        tv.rebuild_items();
        if !tv.items.is_empty() {
            tv.state.select(Some(0));
        }
        tv
    }

    /// Set the current document values for inline display.
    pub fn set_values(&mut self, values: HashMap<String, String>) {
        self.values = values;
    }

    /// Set the array lengths for table array display.
    pub fn set_array_lengths(&mut self, lengths: HashMap<String, usize>) {
        self.array_lengths = lengths;
    }

    pub fn set_validation_errors(&mut self, errors: Vec<hyburn_config_lib::editor::ValidationError>) {
        self.validation_error_paths = errors.iter().map(|e| e.path.clone()).collect();
    }

    /// Rebuild the tree from the schema (e.g., after add/remove).
    pub fn refresh(&mut self) {
        self.rebuild_items();
    }

    /// Rebuild visible items from schema + expanded state.
    fn rebuild_items(&mut self) {
        let old_selected = self.state.selected().and_then(|i| self.paths.get(i).cloned());

        self.items.clear();
        self.paths.clear();

        // Clone schema to avoid borrow conflict
        let schema = self.schema.clone();
        self.build_visible(&schema, String::new(), 0, true);

        // Restore selection if possible
        if let Some(sel_path) = old_selected {
            if let Some(idx) = self.paths.iter().position(|p| *p == sel_path) {
                self.state.select(Some(idx));
            } else if !self.items.is_empty() {
                self.state.select(Some(0));
            }
        } else if !self.items.is_empty() {
            self.state.select(Some(0));
        }
    }

    /// Refresh labels with current values (call after set_values).
    /// Skips if a search filter is active (would undo the filter).
    pub fn refresh_labels(&mut self) {
        if self.filter_active {
            return;
        }
        // Rebuild from scratch with current values
        self.rebuild_items();
    }

    /// Recursively build the visible items list.
    /// Only recurses into children if the parent path is expanded.
    fn build_visible(&mut self, node: &ConfigNode, parent_path: String, depth: usize, parent_expanded: bool) {
        for child in &node.children {
            let path = if parent_path.is_empty() {
                child.name.to_string()
            } else {
                format!("{}.{}", parent_path, child.name)
            };

            // Only show this child if parent is expanded (root is always expanded)
            if !parent_expanded {
                continue;
            }

            let is_expandable = is_node_expandable(child);
            let is_expanded = self.expanded.contains(&path);

            // Build label with current value if available
            let label = self.format_label(child, &path, is_expandable, is_expanded);

            self.items.push(TreeItem {
                label,
                path: path.clone(),
                depth,
                is_expandable,
                is_expanded,
            });
            self.paths.push(path.clone());

            // If this node is expanded, show its children
            if is_expanded {
                // For TableArrays, show indexed entries from the document
                if matches!(child.field_type, FieldType::TableArray) {
                    let count = self.array_lengths.get(&path).copied().unwrap_or(0);
                    for i in 0..count {
                        let idx_path = format!("{}[{}]", path, i);
                        let idx_expanded = self.expanded.contains(&idx_path);

                        // Get a short label for the array element
                        let idx_label = self.format_array_entry_label(&path, &idx_path, i, child, idx_expanded);

                        self.items.push(TreeItem {
                            label: idx_label,
                            path: idx_path.clone(),
                            depth: depth + 1,
                            is_expandable: true,
                            is_expanded: idx_expanded,
                        });
                        self.paths.push(idx_path.clone());

                        // If the indexed entry is expanded, show its children
                        if idx_expanded {
                            // Show the array entry's children directly (skip the wrapper struct)
                            // child is the TableArray node, child.children[0] is the item schema
                            // (e.g., SubnetworkConfig with fields: model, nnodes, params, etc.)
                            if let Some(item_schema) = child.children.first() {
                                for field in &item_schema.children {
                                    let field_path = format!("{}.{}", idx_path, field.name);
                                    let field_expandable = is_node_expandable(field);
                                    let field_expanded = self.expanded.contains(&field_path);
                                    let field_label = self.format_label(field, &field_path, field_expandable, field_expanded);
                                    self.items.push(TreeItem {
                                        label: field_label,
                                        path: field_path.clone(),
                                        depth: depth + 2,
                                        is_expandable: field_expandable,
                                        is_expanded: field_expanded,
                                    });
                                    self.paths.push(field_path.clone());
                                    if field_expanded {
                                        self.build_visible(field, field_path, depth + 3, true);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    self.build_visible(child, path, depth + 1, true);
                }
            }
        }
    }

    /// Format a label for a node.
    fn format_label(&self, node: &ConfigNode, path: &str, is_expandable: bool, is_expanded: bool) -> String {
        let prefix = if is_expandable {
            if is_expanded { "[-]" } else { "[+]" }
        } else {
            "   "
        };

        let value_hint = if !is_expandable {
            // Show current value for leaf nodes
            if let Some(val) = self.values.get(path) {
                let truncated = if val.len() > 30 {
                    format!("{}...", &val[..27])
                } else {
                    val.clone()
                };
                format!(" = {}", truncated)
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let type_hint = format_type_hint(node);
        let required = if node.required { " *" } else { "" };

        format!("{} {}{}{}", prefix, node.name, value_hint, if type_hint.is_empty() { required.to_string() } else { format!(" {}{}", type_hint, required) })
    }

    /// Format a label for an array entry (e.g., "[-] [0] Generic2dOscillator").
    fn format_array_entry_label(&self, _array_path: &str, idx_path: &str, index: usize, _schema: &ConfigNode, is_expanded: bool) -> String {
        let prefix = if is_expanded { "[-]" } else { "[+]" };

        // Try to get a summary value for the entry
        // Check common summary fields: model, monitor_type, temporal
        let summary = self.values.get(&format!("{}.model", idx_path))
            .or_else(|| self.values.get(&format!("{}.monitor_type", idx_path)))
            .or_else(|| self.values.get(&format!("{}.temporal", idx_path)))
            .map(|v| v.trim_matches('"').to_string())
            .unwrap_or_else(|| String::new());

        let summary_part = if summary.is_empty() {
            String::new()
        } else {
            format!(" {}", summary)
        };

        format!("{} [{}]{}", prefix, index, summary_part)
    }

    pub fn selected_path(&self) -> Option<String> {
        let idx = self.state.selected()?;
        self.paths.get(idx).cloned()
    }

    /// Return the schema node for the currently selected path.
    #[allow(dead_code)]
    pub fn selected_schema_node(&self) -> Option<&ConfigNode> {
        let path = self.selected_path()?;
        // Walk schema tree to find the node
        let mut node = &self.schema;
        for seg in path.split('.') {
            let field_name = seg.split('[').next().unwrap_or(seg);
            if let Some(child) = node.child(field_name) {
                node = child;
            } else {
                return None;
            }
        }
        Some(node)
    }

    /// Return true if the selected node is expandable and currently collapsed.
    pub fn selected_can_expand(&self) -> bool {
        if let Some(idx) = self.state.selected() {
            if let Some(item) = self.items.get(idx) {
                return item.is_expandable && !item.is_expanded;
            }
        }
        false
    }

    /// Return true if the selected node is currently expanded.
    #[allow(dead_code)]
    pub fn selected_can_collapse(&self) -> bool {
        if let Some(idx) = self.state.selected() {
            if let Some(item) = self.items.get(idx) {
                return item.is_expandable && item.is_expanded;
            }
        }
        false
    }

    pub fn move_up(&mut self) {
        let i = self.state.selected().unwrap_or(0);
        if i > 0 {
            self.state.select(Some(i - 1));
        }
    }

    pub fn move_down(&mut self) {
        let i = self.state.selected().unwrap_or(0);
        if i < self.items.len().saturating_sub(1) {
            self.state.select(Some(i + 1));
        }
    }

    /// Expand the selected node to show its children.
    pub fn expand(&mut self) {
        if let Some(idx) = self.state.selected() {
            if let Some(item) = self.items.get(idx) {
                if item.is_expandable && !item.is_expanded {
                    let path = item.path.clone();
                    self.expanded.insert(path);
                    self.rebuild_items();
                }
            }
        }
    }

    /// Collapse the selected node to hide its children.
    pub fn collapse(&mut self) {
        if let Some(idx) = self.state.selected() {
            if let Some(item) = self.items.get(idx) {
                if item.is_expandable && item.is_expanded {
                    let path = item.path.clone();
                    // Also collapse all descendants
                    self.expanded.retain(|p| !p.starts_with(&format!("{}.", path)) && p != &path);
                    self.rebuild_items();
                }
            }
        }
    }

    /// Expand all nodes in the schema tree (used for search/filter).
    fn expand_all(&mut self, node: &ConfigNode, parent_path: String) {
        for child in &node.children {
            let path = if parent_path.is_empty() {
                child.name.to_string()
            } else {
                format!("{}.{}", parent_path, child.name)
            };
            if is_node_expandable(child) {
                self.expanded.insert(path.clone());
            }
            // For TableArrays, expand array entries too
            if matches!(child.field_type, FieldType::TableArray) {
                let count = self.array_lengths.get(&path).copied().unwrap_or(0);
                for i in 0..count {
                    let idx_path = format!("{}[{}]", path, i);
                    self.expanded.insert(idx_path.clone());
                    if let Some(item_schema) = child.children.first() {
                        for field in &item_schema.children {
                            let field_path = format!("{}.{}", idx_path, field.name);
                            if is_node_expandable(field) {
                                self.expanded.insert(field_path);
                            }
                        }
                    }
                }
            }
            self.expand_all(child, path);
        }
    }

    /// Toggle expand/collapse on the selected node.
    #[allow(dead_code)]
    pub fn toggle(&mut self) {
        if self.selected_can_expand() {
            self.expand();
        } else if self.selected_can_collapse() {
            self.collapse();
        }
    }

    pub fn render(&mut self, f: &mut Frame, area: Rect, block: Block) {
        let items: Vec<ListItem> = self
            .items
            .iter()
            .map(|item| {
                let indent = "  ".repeat(item.depth);
                let has_error = self.validation_error_paths.contains(&item.path);
                let style = if has_error {
                    Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
                } else if item.is_expandable {
                    if item.is_expanded {
                        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                    }
                } else {
                    Style::default().fg(Color::White)
                };
                let prefix = if has_error { "✗ " } else { "" };
                ListItem::new(Line::from(Span::styled(
                    format!("{}{}{}", indent, prefix, item.label),
                    style,
                )))
            })
            .collect();

        let list = List::new(items).block(block).highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        );

        f.render_stateful_widget(list, area, &mut self.state);
    }

    // --- Search/Filter ---

    /// Filter the visible tree to only show items matching the query.
    pub fn apply_filter(&mut self, query: &str) {
        if query.is_empty() {
            self.clear_filter();
            return;
        }
        let query_lower = query.to_lowercase();

        // Temporarily expand everything to make all nodes searchable
        let saved_expanded = self.expanded.clone();
        self.expand_all(&self.schema.clone(), String::new());
        self.rebuild_items();

        // Collect paths of items that match
        let mut visible_paths: HashSet<String> = HashSet::new();
        for item in &self.items {
            if item.path.to_lowercase().contains(&query_lower)
                || item.label.to_lowercase().contains(&query_lower)
            {
                visible_paths.insert(item.path.clone());
                // Add all parent paths
                let mut p = item.path.as_str();
                while let Some(dot) = p.rfind('.') {
                    p = &p[..dot];
                    visible_paths.insert(p.to_string());
                }
                // Add parent for array index paths
                if let Some(bracket) = p.rfind('[') {
                    visible_paths.insert(p[..bracket].to_string());
                }
            }
        }

        // Filter: keep only items whose path is visible
        let mut new_items = Vec::new();
        let mut new_paths = Vec::new();
        for i in 0..self.items.len() {
            if visible_paths.contains(&self.paths[i]) {
                new_items.push(self.items[i].clone());
                new_paths.push(self.paths[i].clone());
            }
        }
        self.items = new_items;
        self.paths = new_paths;

        // Select first match
        if let Some(first_match) = self.paths.iter().position(|p| p.to_lowercase().contains(&query_lower)) {
            self.state.select(Some(first_match));
        } else if !self.items.is_empty() {
            self.state.select(Some(0));
        }

        // Store the saved expanded state for restoration on clear
        self._pre_filter_expanded = Some(saved_expanded);
        self.filter_active = true;
    }

    /// Clear any active filter and rebuild the full tree.
    pub fn clear_filter(&mut self) {
        self.filter_active = false;
        if let Some(saved) = self._pre_filter_expanded.take() {
            self.expanded = saved;
        }
        self.rebuild_items();
    }

    /// Jump to the first item matching the query.
    pub fn jump_to_match(&mut self, query: &str) {
        let query_lower = query.to_lowercase();
        if let Some(idx) = self.items.iter().position(|item| {
            item.path.to_lowercase().contains(&query_lower)
                || item.label.to_lowercase().contains(&query_lower)
        }) {
            self.state.select(Some(idx));
        }
    }
}

/// Check if a node is expandable (has visible children).
fn is_node_expandable(node: &ConfigNode) -> bool {
    matches!(
        node.field_type,
        FieldType::Struct | FieldType::TableArray | FieldType::UntaggedEnum(_)
    ) && !node.children.is_empty()
}

/// Format a type hint for a node.
fn format_type_hint(node: &ConfigNode) -> String {
    match &node.field_type {
        FieldType::Scalar(st) => format!("({})", st),
        FieldType::Array { item_type, variable_length } => {
            if *variable_length {
                format!("[{}...]", item_type)
            } else {
                format!("[{}]", item_type)
            }
        }
        FieldType::Enum(variants) => {
            if variants.len() <= 4 {
                format!("({})", variants.join("|"))
            } else {
                format!("({} opts)", variants.len())
            }
        }
        FieldType::Struct => "(table)".to_string(),
        FieldType::TableArray => "[[array]]".to_string(),
        FieldType::UntaggedEnum(_) => "(multi)".to_string(),
    }
}
