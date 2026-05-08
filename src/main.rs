#[cfg(feature = "cli")]
fn main() -> anyhow::Result<()> {
    use clap::Parser;
    env_logger::init();
    let cli = hyburn::cli::Cli::parse();
    cli.run()
}

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("hyburn was compiled without the 'cli' feature. Enable it to use the CLI.");
    std::process::exit(1);
}
