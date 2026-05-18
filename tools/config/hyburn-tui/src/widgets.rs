//! Popup widgets for editing values.

use hyburn_config_lib::FieldType;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph},
    Frame,
};

/// Action associated with an enum selection popup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnumAction {
    Edit,
    Add,
}

/// Action to execute after confirmation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfirmedAction {
    Remove { path: String },
}

/// A popup overlay.
pub enum Popup {
    Edit(EditPopup),
    EnumSelect {
        items: Vec<String>,
        selected: usize,
        path: String,
        field_name: String,
        action: EnumAction,
    },
    VariantSelect {
        labels: Vec<String>,
        types: Vec<FieldType>,
        selected: usize,
        path: String,
        field_name: String,
        current_value: String,
    },
    Confirm {
        message: String,
        confirmed_action: ConfirmedAction,
    },
    Help,
    Search(SearchPopup),
}

/// Inline text editor popup.
pub struct EditPopup {
    pub path: String,
    pub field_name: String,
    pub input: String,
    pub cursor: usize,
    pub hint: String,
}

impl EditPopup {
    pub fn new(path: &str, field_name: &str, initial: &str, hint: &str) -> Self {
        EditPopup {
            path: path.to_string(),
            field_name: field_name.to_string(),
            input: initial.to_string(),
            cursor: initial.len(),
            hint: hint.to_string(),
        }
    }

    pub fn input(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += 1;
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.input.remove(self.cursor);
        }
    }

    pub fn get_value(&self) -> String {
        self.input.clone()
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .title(format!("Edit: {}", self.field_name))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow));

        let inner = block.inner(area);
        f.render_widget(Clear, area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(0),
            ])
            .split(inner);

        // Path hint
        f.render_widget(
            Paragraph::new(Span::styled(
                format!("Path: {}", self.path),
                Style::default().fg(Color::DarkGray),
            )),
            chunks[0],
        );

        // Input field
        let input_text = format!("{}_", self.input);
        f.render_widget(
            Paragraph::new(Span::styled(
                input_text,
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            chunks[1],
        );

        // Constraint / hint line
        if !self.hint.is_empty() {
            f.render_widget(
                Paragraph::new(Span::styled(
                    self.hint.clone(),
                    Style::default().fg(Color::DarkGray),
                )),
                chunks[2],
            );
        }
    }
}

/// Render an enum selection popup.
pub fn render_enum_select(
    f: &mut Frame,
    area: Rect,
    field_name: &str,
    items: &[String],
    selected: usize,
) {
    let block = Block::default()
        .title(format!("Select: {}", field_name))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Yellow));

    let inner = block.inner(area);
    f.render_widget(Clear, area);
    f.render_widget(block, area);

    let list_items: Vec<ListItem> = items
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let style = if i == selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            ListItem::new(Line::from(Span::styled(format!("  {}", item), style)))
        })
        .collect();

    let list = List::new(list_items).highlight_style(
        Style::default()
            .bg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    );

    let mut state = ListState::default();
    state.select(Some(selected));
    f.render_stateful_widget(list, inner, &mut state);
}

/// Render a confirmation popup.
pub fn render_confirm(f: &mut Frame, area: Rect, message: &str) {
    let block = Block::default()
        .title("Confirm")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red));

    let inner = block.inner(area);
    f.render_widget(Clear, area);
    f.render_widget(block, area);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(inner);

    f.render_widget(
        Paragraph::new(Span::styled(
            message,
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        )),
        chunks[0],
    );

    f.render_widget(
        Paragraph::new(Span::styled(
            "y/N",
            Style::default().fg(Color::DarkGray),
        )),
        chunks[1],
    );
}

/// Render the help overlay popup.
pub fn render_help(f: &mut Frame, area: Rect) {
    let block = Block::default()
        .title("Help")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let inner = block.inner(area);
    f.render_widget(Clear, area);
    f.render_widget(block, area);

    let header_style = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
    let key_style = Style::default().fg(Color::Yellow);
    let desc_style = Style::default().fg(Color::White);

    let lines = vec![
        Line::from(""),
        Line::from(Span::styled("  Navigation", header_style)),
        Line::from(vec![
            Span::styled("    j/k, ↑/↓    ", key_style),
            Span::styled("Move up/down", desc_style),
        ]),
        Line::from(vec![
            Span::styled("    h/l, ←/→    ", key_style),
            Span::styled("Collapse/Expand", desc_style),
        ]),
        Line::from(vec![
            Span::styled("    Tab          ", key_style),
            Span::styled("Switch pane", desc_style),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Editing", header_style)),
        Line::from(vec![
            Span::styled("    Enter        ", key_style),
            Span::styled("Edit/Expand", desc_style),
        ]),
        Line::from(vec![
            Span::styled("    a            ", key_style),
            Span::styled("Add element", desc_style),
        ]),
        Line::from(vec![
            Span::styled("    d            ", key_style),
            Span::styled("Delete element", desc_style),
        ]),
        Line::from(vec![
            Span::styled("    Ctrl-S       ", key_style),
            Span::styled("Save", desc_style),
        ]),
        Line::from(vec![
            Span::styled("    Ctrl-Z       ", key_style),
            Span::styled("Undo", desc_style),
        ]),
        Line::from(""),
        Line::from(Span::styled("  Other", header_style)),
        Line::from(vec![
            Span::styled("    /            ", key_style),
            Span::styled("Search/filter", desc_style),
        ]),
        Line::from(vec![
            Span::styled("    ?            ", key_style),
            Span::styled("Toggle help", desc_style),
        ]),
        Line::from(vec![
            Span::styled("    q            ", key_style),
            Span::styled("Quit", desc_style),
        ]),
        Line::from(vec![
            Span::styled("    Ctrl-Q       ", key_style),
            Span::styled("Force quit", desc_style),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Press ? or Esc to close",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let paragraph = Paragraph::new(lines);
    f.render_widget(paragraph, inner);
}

/// Search/filter popup.
pub struct SearchPopup {
    pub input: String,
    pub cursor: usize,
}

impl SearchPopup {
    pub fn new() -> Self {
        SearchPopup {
            input: String::new(),
            cursor: 0,
        }
    }

    pub fn input_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += 1;
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.input.remove(self.cursor);
        }
    }

    pub fn query(&self) -> &str {
        &self.input
    }

    pub fn render(&self, f: &mut Frame, area: Rect) {
        let block = Block::default()
            .title("Search")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan));

        let inner = block.inner(area);
        f.render_widget(Clear, area);
        f.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(0),
            ])
            .split(inner);

        // Input field
        let input_text = format!("/{}", self.input);
        f.render_widget(
            Paragraph::new(Span::styled(
                input_text,
                Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
            )),
            chunks[0],
        );

        // Hint
        f.render_widget(
            Paragraph::new(Span::styled(
                "Esc to close, Enter to jump",
                Style::default().fg(Color::DarkGray),
            )),
            chunks[1],
        );
    }
}
