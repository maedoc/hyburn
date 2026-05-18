//! Shell completion generation for the hyburn-config CLI.

use clap::Command;
use clap_complete::{generate, Shell};

/// Generate shell completions for the given clap Command.
pub fn generate_completions(shell: Shell, cmd: &mut Command) -> String {
    let mut buf = Vec::new();
    generate(shell, cmd, "hyburn-config", &mut buf);
    String::from_utf8(buf).unwrap_or_default()
}
