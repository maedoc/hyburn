use clap::Parser;
use hyburn::cli::Cli;

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();
    cli.run()
}
