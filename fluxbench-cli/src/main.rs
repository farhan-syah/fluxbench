//! FluxBench CLI - Benchmark Supervisor
//!
//! The supervisor process that orchestrates benchmark execution.

fn main() -> anyhow::Result<()> {
    fluxbench_cli::run()
}
