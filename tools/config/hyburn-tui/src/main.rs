//! hyburn-tui: Interactive TUI for editing hyburn config files.

mod app;
mod logger;
mod preview;
mod tree_view;
mod widgets;

use app::App;
use logger::Logger;
use crossterm::{
    event::{self, Event, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io;

fn parse_log_flag(args: &[String]) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == "--log" {
            if i + 1 < args.len() && !args[i + 1].starts_with("-") {
                return Some(args[i + 1].clone());
            }
        }
    }
    None
}

fn parse_file_path(args: &[String]) -> Option<String> {
    let mut skip_next = false;
    for arg in args.iter().skip(1) {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg == "--log" {
            skip_next = true;
            continue;
        }
        if !arg.starts_with("-") {
            return Some(arg.clone());
        }
    }
    None
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let file_path = parse_file_path(&args).unwrap_or_else(|| {
        eprintln!("Usage: hyburn-tui [--log <path>] <config.toml>");
        std::process::exit(1);
    });

    let log_path = parse_log_flag(&args).unwrap_or_else(|| "/tmp/hyburn-tui.log".to_string());

    let logger = match Logger::new(&log_path) {
        Ok(l) => {
            l.log_start(&file_path);
            Some(l)
        }
        Err(e) => {
            eprintln!("Warning: could not create log file '{}': {}", log_path, e);
            None
        }
    };

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app
    let mut app = App::new(&file_path, logger)?;

    // Main loop
    let result = run_app(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(e) = result {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }

    app.log_quit();

    Ok(())
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        terminal.draw(|f| {
            let size = f.area();
            app.render(f, size);
        })?;

        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if app.handle_key(key) {
                        return Ok(());
                    }
                }
            }
        }
    }
}
