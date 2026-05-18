//! TOML preview pane widget.

use ratatui::{
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Paragraph, Wrap},
    Frame,
};

/// Preview pane state and rendering.
pub struct PreviewPane {
    lines: Vec<String>,
    scroll: usize,
}

impl PreviewPane {
    pub fn new(content: &str) -> Self {
        PreviewPane {
            lines: content.lines().map(|l| l.to_string()).collect(),
            scroll: 0,
        }
    }

    pub fn set_content(&mut self, content: &str) {
        self.lines = content.lines().map(|l| l.to_string()).collect();
        // Clamp scroll
        if self.scroll >= self.lines.len() {
            self.scroll = self.lines.len().saturating_sub(1);
        }
    }

    pub fn scroll_up(&mut self) {
        self.scroll = self.scroll.saturating_sub(1);
    }

    pub fn scroll_down(&mut self) {
        if self.scroll < self.lines.len().saturating_sub(1) {
            self.scroll += 1;
        }
    }

    pub fn render(&self, f: &mut Frame, area: Rect, block: Block) {
        let inner_height = area.height.saturating_sub(2) as usize; // borders
        let visible_lines: Vec<Line> = self
            .lines
            .iter()
            .skip(self.scroll)
            .take(inner_height)
            .map(|line| {
                // Basic syntax highlighting
                let trimmed = line.trim();
                if trimmed.starts_with('#') {
                    Line::from(Span::styled(line.clone(), Style::default().fg(Color::DarkGray)))
                } else if trimmed.starts_with('[') {
                    Line::from(Span::styled(line.clone(), Style::default().fg(Color::Yellow)))
                } else if trimmed.contains('=') {
                    // Highlight key = value
                    if let Some(eq_pos) = line.find('=') {
                        let key = &line[..eq_pos];
                        let rest = &line[eq_pos..];
                        Line::from(vec![
                            Span::styled(key.to_string(), Style::default().fg(Color::Green)),
                            Span::styled(rest.to_string(), Style::default().fg(Color::White)),
                        ])
                    } else {
                        Line::from(Span::styled(line.clone(), Style::default().fg(Color::White)))
                    }
                } else {
                    Line::from(Span::styled(line.clone(), Style::default().fg(Color::White)))
                }
            })
            .collect();

        let paragraph = Paragraph::new(visible_lines)
            .block(block)
            .wrap(Wrap { trim: false });

        f.render_widget(paragraph, area);
    }
}
