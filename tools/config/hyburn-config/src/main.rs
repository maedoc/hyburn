//! hyburn-config CLI: sysctl-style CRUD operations on hyburn config files.

use clap::{CommandFactory, Parser, Subcommand, ValueHint};
use clap_complete::{generate, Shell};
use hyburn_config_lib::{ConfigEditor, generate_json_schema};
use std::io;

#[derive(Parser)]
#[command(
    name = "hyburn-config",
    version,
    about = "CLI tool for hyburn config file operations"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Get a value from the config file.
    Get {
        /// Path to the TOML config file.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
        /// Config path (e.g., network.subnetworks[0].model).
        path: String,
    },

    /// Set a value in the config file.
    Set {
        /// Path to the TOML config file.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
        /// Config path (e.g., integrator, dt, network.subnetworks[0].params[2]).
        path: String,
        /// Value to set (TOML expression: string, number, array, etc.).
        value: String,
        /// Validate the config after setting the value.
        #[arg(long)]
        validate: bool,
    },

    /// Add an element to an array-of-tables.
    Add {
        /// Path to the TOML config file.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
        /// Path to the array (e.g., network.subnetworks, network.projections).
        path: String,
        /// Optional TOML template string for the new element.
        #[arg(long)]
        from: Option<String>,
        /// Optional model name for creating default subnetworks.
        #[arg(long)]
        model: Option<String>,
        /// Validate the config after adding the element.
        #[arg(long)]
        validate: bool,
    },

    /// Remove an element from an array by index.
    Remove {
        /// Path to the TOML config file.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
        /// Path with index to remove (e.g., network.subnetworks[1]).
        path: String,
        /// Validate the config after removing the element.
        #[arg(long)]
        validate: bool,
    },

    /// List the config tree (or a subtree at a path).
    List {
        /// Path to the TOML config file.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
        /// Optional config path to list a subtree.
        path: Option<String>,
    },

    /// Validate the config file.
    Validate {
        /// Path to the TOML config file.
        #[arg(value_hint = ValueHint::FilePath)]
        file: String,
    },

    /// Print the JSON Schema for hyburn config.
    Schema {
        /// Optional path to a subtree schema.
        #[arg(long)]
        path: Option<String>,
        /// Output format.
        #[arg(long, default_value = "json")]
        format: String,
    },

    /// Generate shell completions for hyburn-config.
    Completions {
        /// Shell to generate completions for.
        #[arg(value_parser = parse_shell)]
        shell: Shell,
    },
}

fn parse_shell(s: &str) -> Result<Shell, String> {
    match s.to_lowercase().as_str() {
        "bash" => Ok(Shell::Bash),
        "zsh" => Ok(Shell::Zsh),
        "fish" => Ok(Shell::Fish),
        "powershell" | "pwsh" => Ok(Shell::PowerShell),
        "elvish" => Ok(Shell::Elvish),
        _ => Err(format!("unsupported shell: {}", s)),
    }
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Command::Get { file, path } => cmd_get(&file, &path),
        Command::Set {
            file,
            path,
            value,
            validate,
        } => cmd_set(&file, &path, &value, validate),
        Command::Add { file, path, from, model, validate } => cmd_add(&file, &path, from.as_deref(), model.as_deref(), validate),
        Command::Remove { file, path, validate } => cmd_remove(&file, &path, validate),
        Command::List { file, path } => cmd_list(&file, path.as_deref()),
        Command::Validate { file } => cmd_validate(&file),
        Command::Schema { path, format } => cmd_schema(path.as_deref(), &format),
        Command::Completions { shell } => cmd_completions(shell),
    };

    if let Err(e) = result {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}

fn cmd_get(file: &str, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let editor = ConfigEditor::from_file(file)?;
    let value = editor.get(path)?;
    println!("{}", value);
    Ok(())
}

fn cmd_set(
    file: &str,
    path: &str,
    value: &str,
    validate: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut editor = ConfigEditor::from_file(file)?;
    editor.set(path, value)?;
    editor.save(file)?;
    if validate {
        let editor = ConfigEditor::from_file(file)?;
        match editor.validate() {
            Ok(()) => println!("Valid"),
            Err(errors) => {
                for e in &errors {
                    eprintln!("  {}", e);
                }
                return Err("validation failed".into());
            }
        }
    }
    Ok(())
}

fn cmd_add(
    file: &str,
    path: &str,
    template: Option<&str>,
    model: Option<&str>,
    validate: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut editor = ConfigEditor::from_file(file)?;
    editor.add(path, template, model)?;
    editor.save(file)?;
    if validate {
        let editor = ConfigEditor::from_file(file)?;
        match editor.validate() {
            Ok(()) => println!("Valid"),
            Err(errors) => {
                for e in &errors {
                    eprintln!("  {}", e);
                }
                return Err("validation failed".into());
            }
        }
    }
    Ok(())
}

fn cmd_remove(file: &str, path: &str, validate: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut editor = ConfigEditor::from_file(file)?;
    editor.remove(path)?;
    editor.save(file)?;
    if validate {
        let editor = ConfigEditor::from_file(file)?;
        match editor.validate() {
            Ok(()) => println!("Valid"),
            Err(errors) => {
                for e in &errors {
                    eprintln!("  {}", e);
                }
                return Err("validation failed".into());
            }
        }
    }
    Ok(())
}

fn cmd_list(file: &str, path: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let editor = ConfigEditor::from_file(file)?;
    let output = editor.list(path)?;
    print!("{}", output);
    Ok(())
}

fn cmd_validate(file: &str) -> Result<(), Box<dyn std::error::Error>> {
    let editor = ConfigEditor::from_file(file)?;
    match editor.validate() {
        Ok(()) => {
            println!("Valid");
            Ok(())
        }
        Err(errors) => {
            for e in &errors {
                eprintln!("  {}", e);
            }
            Err("validation failed".into())
        }
    }
}

fn cmd_schema(
    path: Option<&str>,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let schema = if let Some(p) = path {
        hyburn_config_lib::json_schema::generate_json_schema_at_path(p)
            .map_err(|e| format!("{}", e))?
    } else {
        generate_json_schema()
    };

    match format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&schema)?);
        }
        "toml" => {
            eprintln!("TOML schema output not yet supported; use --format json");
            std::process::exit(1);
        }
        _ => {
            eprintln!("unsupported format: {}", format);
            std::process::exit(1);
        }
    }
    Ok(())
}

fn cmd_completions(shell: Shell) -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Cli::command();
    generate(shell, &mut cmd, "hyburn-config", &mut io::stdout());
    Ok(())
}
