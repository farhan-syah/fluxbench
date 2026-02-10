//! FluxBench CLI Library
//!
//! This module provides the CLI infrastructure for benchmark binaries.
//! Use `fluxbench_cli::run()` in your main function to get the full
//! fluxbench CLI experience with your registered benchmarks.
//!
//! # Example
//!
//! ```ignore
//! use fluxbench::prelude::*;
//!
//! #[bench]
//! fn my_benchmark(b: &mut Bencher) {
//!     b.iter(|| expensive_operation());
//! }
//!
//! fn main() {
//!     fluxbench_cli::run();
//! }
//! ```

mod config;
mod executor;
mod planner;
mod supervisor;

pub use config::*;
pub use executor::{
    ExecutionConfig, Executor, IsolatedExecutor, build_report, compute_statistics,
    execute_verifications, format_human_output,
};
pub use supervisor::*;

use clap::{Parser, Subcommand};
use fluxbench_core::{BenchmarkDef, WorkerMain};
use fluxbench_logic::aggregate_verifications;
use fluxbench_report::{
    OutputFormat, generate_csv_report, generate_github_summary, generate_html_report,
    generate_json_report,
};
use rayon::ThreadPoolBuilder;
use regex::Regex;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

/// FluxBench CLI arguments
#[derive(Parser, Debug)]
#[command(name = "fluxbench")]
#[command(author, version, about = "FluxBench - benchmarking framework for Rust")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Filter benchmarks by regex pattern
    #[arg(default_value = ".*")]
    pub filter: String,

    /// Output format: json, github-summary, csv, html, human
    #[arg(long, default_value = "human")]
    pub format: String,

    /// Output file (stdout if not specified)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Load baseline for comparison
    #[arg(long)]
    pub baseline: Option<PathBuf>,

    /// Dry run - list benchmarks without executing
    #[arg(long)]
    pub dry_run: bool,

    /// Regression threshold percentage
    #[arg(long)]
    pub threshold: Option<f64>,

    /// Run benchmarks for this group only
    #[arg(long)]
    pub group: Option<String>,

    /// Filter by tag
    #[arg(long)]
    pub tag: Option<String>,

    /// Skip benchmarks with this tag
    #[arg(long)]
    pub skip_tag: Option<String>,

    /// Warmup time in seconds
    #[arg(long, default_value = "3")]
    pub warmup: u64,

    /// Measurement time in seconds
    #[arg(long, default_value = "5")]
    pub measurement: u64,

    /// Minimum number of iterations
    #[arg(long)]
    pub min_iterations: Option<u64>,

    /// Maximum number of iterations
    #[arg(long)]
    pub max_iterations: Option<u64>,

    /// Verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Run benchmarks in isolated worker processes (default: true)
    /// Use --isolated=false to disable and run in-process
    #[arg(long, default_value = "true", action = clap::ArgAction::Set)]
    pub isolated: bool,

    /// Use fresh worker process for each benchmark (One-Shot mode)
    /// Default is Persistent mode: reuse worker for safe Rust code
    #[arg(long)]
    pub one_shot: bool,

    /// Worker timeout in seconds
    #[arg(long, default_value = "60")]
    pub worker_timeout: u64,

    /// Number of threads for parallel statistics computation
    /// 0 = use all available cores (default), 1 = single-threaded
    #[arg(long, short = 'j', default_value = "0")]
    pub threads: usize,

    /// Internal: Run as worker process (used by supervisor)
    #[arg(long, hide = true)]
    pub flux_worker: bool,
}

/// CLI subcommands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// List all discovered benchmarks
    List,
    /// Run benchmarks (default)
    Run {
        /// Number of parallel workers
        #[arg(long, default_value = "1")]
        jobs: usize,
    },
    /// Compare against a git ref
    Compare {
        /// Git ref to compare against (e.g., origin/main)
        #[arg(name = "REF")]
        git_ref: String,
    },
}

/// Run the FluxBench CLI with the given arguments.
/// This is the main entry point for benchmark binaries.
///
/// # Returns
/// Returns `Ok(())` on success, or an error if something goes wrong.
pub fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    run_with_cli(cli)
}

/// Run the FluxBench CLI with pre-parsed arguments.
pub fn run_with_cli(cli: Cli) -> anyhow::Result<()> {
    // Handle worker mode first (before any other initialization)
    if cli.flux_worker {
        return run_worker_mode();
    }

    // Initialize logging
    if cli.verbose {
        tracing_subscriber::fmt()
            .with_env_filter("fluxbench=debug")
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_env_filter("fluxbench=info")
            .init();
    }

    // Discover flux.toml configuration (CLI flags override)
    let config = FluxConfig::discover().unwrap_or_default();

    // Parse output format
    let format: OutputFormat = cli.format.parse().unwrap_or(OutputFormat::Human);

    match cli.command {
        Some(Commands::List) => {
            list_benchmarks(&cli)?;
        }
        Some(Commands::Run { jobs }) => {
            run_benchmarks(&cli, &config, format, jobs)?;
        }
        Some(Commands::Compare { ref git_ref }) => {
            compare_benchmarks(&cli, &config, git_ref, format)?;
        }
        None => {
            // Default: run benchmarks
            if cli.dry_run {
                list_benchmarks(&cli)?;
            } else {
                run_benchmarks(&cli, &config, format, 1)?;
            }
        }
    }

    Ok(())
}

/// Run as a worker process (IPC mode)
fn run_worker_mode() -> anyhow::Result<()> {
    let mut worker = WorkerMain::new();
    worker
        .run()
        .map_err(|e| anyhow::anyhow!("Worker error: {}", e))
}

/// Filter benchmarks based on CLI options using the planner module.
///
/// Returns benchmarks sorted alphabetically by ID for deterministic execution.
fn filter_benchmarks(
    cli: &Cli,
    benchmarks: &[&'static BenchmarkDef],
) -> Vec<&'static BenchmarkDef> {
    let filter_re = Regex::new(&cli.filter).ok();

    let plan = planner::build_plan(
        benchmarks.iter().copied(),
        filter_re.as_ref(),
        cli.group.as_deref(),
        cli.tag.as_deref(),
        cli.skip_tag.as_deref(),
    );

    plan.benchmarks
}

fn list_benchmarks(cli: &Cli) -> anyhow::Result<()> {
    println!("FluxBench Plan:");

    let all_benchmarks: Vec<_> = inventory::iter::<BenchmarkDef>.into_iter().collect();
    let benchmarks = filter_benchmarks(cli, &all_benchmarks);

    let mut groups: std::collections::BTreeMap<&str, Vec<&BenchmarkDef>> =
        std::collections::BTreeMap::new();

    for bench in &benchmarks {
        groups.entry(bench.group).or_default().push(bench);
    }

    let mut total = 0;
    for (group, benches) in &groups {
        println!("├── group: {}", group);
        for bench in benches {
            let tags = if bench.tags.is_empty() {
                String::new()
            } else {
                format!(" [{}]", bench.tags.join(", "))
            };
            println!(
                "│   ├── {}{} ({}:{})",
                bench.id, tags, bench.file, bench.line
            );
            total += 1;
        }
    }

    println!("{} benchmarks found.", total);

    Ok(())
}

/// Build an ExecutionConfig by layering: flux.toml defaults → CLI overrides.
fn build_execution_config(cli: &Cli, config: &FluxConfig) -> ExecutionConfig {
    // Start from flux.toml values (parsed durations fall back to defaults on error)
    let warmup_ns = FluxConfig::parse_duration(&config.runner.warmup_time).unwrap_or(3_000_000_000);
    let measurement_ns =
        FluxConfig::parse_duration(&config.runner.measurement_time).unwrap_or(5_000_000_000);

    // CLI flags override config file values.
    // clap defaults are warmup=3, measurement=5, so we check if the user explicitly
    // passed different values by comparing against clap defaults. If the CLI value
    // differs from clap's default, the user explicitly set it and it wins.
    let warmup_time_ns = if cli.warmup != 3 {
        cli.warmup * 1_000_000_000
    } else {
        warmup_ns
    };
    let measurement_time_ns = if cli.measurement != 5 {
        cli.measurement * 1_000_000_000
    } else {
        measurement_ns
    };

    // min/max iterations: CLI wins if set, else config, else default
    let min_iterations = cli.min_iterations.or(config.runner.min_iterations);
    let max_iterations = cli.max_iterations.or(config.runner.max_iterations);

    ExecutionConfig {
        warmup_time_ns,
        measurement_time_ns,
        min_iterations,
        max_iterations,
        track_allocations: config.allocator.track,
        bootstrap_iterations: config.runner.bootstrap_iterations,
        confidence_level: config.runner.confidence_level,
    }
}

fn run_benchmarks(
    cli: &Cli,
    config: &FluxConfig,
    format: OutputFormat,
    jobs: usize,
) -> anyhow::Result<()> {
    let jobs = jobs.max(1);

    // Configure Rayon thread pool for statistics computation
    if cli.threads > 0 {
        ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .ok();
    }

    // Discover benchmarks
    let all_benchmarks: Vec<_> = inventory::iter::<BenchmarkDef>.into_iter().collect();
    let benchmarks = filter_benchmarks(cli, &all_benchmarks);

    if benchmarks.is_empty() {
        println!("No benchmarks found.");
        return Ok(());
    }

    let threads_str = if cli.threads == 0 {
        "all".to_string()
    } else {
        cli.threads.to_string()
    };
    let mode_str = if cli.isolated {
        if cli.one_shot {
            " (isolated, one-shot)"
        } else {
            " (isolated, persistent)"
        }
    } else {
        " (in-process)"
    };
    println!(
        "Running {} benchmarks{}, {} threads, {} worker(s)...\n",
        benchmarks.len(),
        mode_str,
        threads_str,
        jobs
    );

    let start_time = Instant::now();

    // Build execution config from flux.toml + CLI overrides
    let exec_config = build_execution_config(cli, config);

    // Execute benchmarks (isolated by default per TDD)
    let results = if cli.isolated {
        let timeout = std::time::Duration::from_secs(cli.worker_timeout);
        let reuse_workers = !cli.one_shot;
        let isolated_executor =
            IsolatedExecutor::new(exec_config.clone(), timeout, reuse_workers, jobs);
        isolated_executor.execute(&benchmarks)
    } else {
        if jobs > 1 {
            eprintln!(
                "Warning: --jobs currently applies only to isolated mode; running in-process serially."
            );
        }
        let mut executor = Executor::new(exec_config.clone());
        executor.execute(&benchmarks)
    };

    // Compute statistics
    let stats = compute_statistics(&results, &exec_config);

    // Build report
    let total_duration_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    let mut report = build_report(&results, &stats, &exec_config, total_duration_ms);

    // Run comparisons, synthetics, and verifications
    let (comparison_results, comparison_series, synthetic_results, verification_results) =
        execute_verifications(&results, &stats);
    let verification_summary = aggregate_verifications(&verification_results);
    report.comparisons = comparison_results;
    report.comparison_series = comparison_series;
    report.synthetics = synthetic_results;
    report.verifications = verification_results;

    // Update summary with verification info
    report.summary.critical_failures = verification_summary.critical_failures;
    report.summary.warnings = verification_summary.failed - verification_summary.critical_failures;

    // Generate output
    let output = match format {
        OutputFormat::Json => generate_json_report(&report)?,
        OutputFormat::GithubSummary => generate_github_summary(&report),
        OutputFormat::Html => generate_html_report(&report),
        OutputFormat::Csv => generate_csv_report(&report),
        OutputFormat::Human => format_human_output(&report),
    };

    // Write output
    if let Some(ref path) = cli.output {
        let mut file = std::fs::File::create(path)?;
        file.write_all(output.as_bytes())?;
        println!("Report written to: {}", path.display());
    } else {
        print!("{}", output);
    }

    // Exit with appropriate code
    let has_crashes = report
        .results
        .iter()
        .any(|r| matches!(r.status, fluxbench_report::BenchmarkStatus::Crashed));

    if verification_summary.should_fail_ci() || has_crashes {
        if has_crashes {
            eprintln!("\nBenchmark(s) crashed during execution");
        }
        if verification_summary.should_fail_ci() {
            eprintln!(
                "\n{} critical verification failure(s)",
                verification_summary.critical_failures + verification_summary.critical_errors
            );
        }
        std::process::exit(1);
    }

    Ok(())
}

fn compare_benchmarks(
    cli: &Cli,
    config: &FluxConfig,
    git_ref: &str,
    format: OutputFormat,
) -> anyhow::Result<()> {
    // Load baseline
    let baseline_path = cli.baseline.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "--baseline required for comparison, or use 'compare' command with a git ref"
        )
    })?;

    if !baseline_path.exists() {
        return Err(anyhow::anyhow!(
            "Baseline file not found: {}",
            baseline_path.display()
        ));
    }

    let baseline_json = std::fs::read_to_string(baseline_path)?;
    let baseline: fluxbench_report::Report = serde_json::from_str(&baseline_json)?;
    let resolved_git_ref = resolve_git_ref(git_ref)?;

    if let Some(baseline_commit) = baseline.meta.git_commit.as_deref() {
        let matches_ref = baseline_commit == resolved_git_ref
            || baseline_commit.starts_with(&resolved_git_ref)
            || resolved_git_ref.starts_with(baseline_commit);
        if !matches_ref {
            return Err(anyhow::anyhow!(
                "Baseline commit {} does not match git ref {} ({})",
                baseline_commit,
                git_ref,
                resolved_git_ref
            ));
        }
    } else {
        eprintln!(
            "Warning: baseline report has no commit metadata; git ref consistency cannot be verified."
        );
    }

    println!("Comparing against baseline: {}", baseline_path.display());
    println!("Git ref: {} ({})\n", git_ref, resolved_git_ref);

    // Run current benchmarks
    let all_benchmarks: Vec<_> = inventory::iter::<BenchmarkDef>.into_iter().collect();
    let benchmarks = filter_benchmarks(cli, &all_benchmarks);

    if benchmarks.is_empty() {
        println!("No benchmarks found.");
        return Ok(());
    }

    let start_time = Instant::now();

    let exec_config = build_execution_config(cli, config);

    let mut executor = Executor::new(exec_config.clone());
    let results = executor.execute(&benchmarks);
    let stats = compute_statistics(&results, &exec_config);

    let total_duration_ms = start_time.elapsed().as_secs_f64() * 1000.0;
    let mut report = build_report(&results, &stats, &exec_config, total_duration_ms);

    // Add comparison data
    let regression_threshold = cli.threshold.unwrap_or(config.ci.regression_threshold);
    let baseline_map: std::collections::HashMap<_, _> = baseline
        .results
        .iter()
        .filter_map(|r| r.metrics.as_ref().map(|m| (r.id.clone(), m.clone())))
        .collect();

    for result in &mut report.results {
        if let (Some(metrics), Some(baseline_metrics)) =
            (&result.metrics, baseline_map.get(&result.id))
        {
            let baseline_mean = baseline_metrics.mean_ns;
            let absolute_change = metrics.mean_ns - baseline_mean;
            let relative_change = if baseline_mean > 0.0 {
                (absolute_change / baseline_mean) * 100.0
            } else {
                0.0
            };

            // Determine significance via CI non-overlap and threshold crossing.
            let ci_non_overlap = metrics.ci_upper_ns < baseline_metrics.ci_lower_ns
                || metrics.ci_lower_ns > baseline_metrics.ci_upper_ns;
            let is_significant = relative_change.abs() > regression_threshold && ci_non_overlap;

            // Track regressions/improvements
            if relative_change > regression_threshold {
                report.summary.regressions += 1;
            } else if relative_change < -regression_threshold {
                report.summary.improvements += 1;
            }

            let mut effect_size = if metrics.std_dev_ns > f64::EPSILON {
                absolute_change / metrics.std_dev_ns
            } else {
                0.0
            };
            if !effect_size.is_finite() {
                effect_size = 0.0;
            }

            let probability_regression = if ci_non_overlap {
                if relative_change > 0.0 { 0.99 } else { 0.01 }
            } else if relative_change > 0.0 {
                0.60
            } else {
                0.40
            };

            result.comparison = Some(fluxbench_report::Comparison {
                baseline_mean_ns: baseline_mean,
                absolute_change_ns: absolute_change,
                relative_change,
                probability_regression,
                is_significant,
                effect_size,
            });
        }
    }

    // Run comparisons, synthetics, and verifications
    let (comparison_results, comparison_series, synthetic_results, verification_results) =
        execute_verifications(&results, &stats);
    let verification_summary = aggregate_verifications(&verification_results);
    report.comparisons = comparison_results;
    report.comparison_series = comparison_series;
    report.synthetics = synthetic_results;
    report.verifications = verification_results;
    report.summary.critical_failures = verification_summary.critical_failures;
    report.summary.warnings = verification_summary.failed - verification_summary.critical_failures;

    // Generate output
    let output = match format {
        OutputFormat::Json => generate_json_report(&report)?,
        OutputFormat::GithubSummary => generate_github_summary(&report),
        OutputFormat::Html => generate_html_report(&report),
        OutputFormat::Csv => generate_csv_report(&report),
        OutputFormat::Human => format_comparison_output(&report, &baseline),
    };

    if let Some(ref path) = cli.output {
        let mut file = std::fs::File::create(path)?;
        file.write_all(output.as_bytes())?;
        println!("Report written to: {}", path.display());
    } else {
        print!("{}", output);
    }

    // Exit with error if regressions exceed threshold or verifications fail
    let should_fail = report.summary.regressions > 0 || verification_summary.should_fail_ci();
    if should_fail {
        if report.summary.regressions > 0 {
            eprintln!(
                "\n{} regression(s) detected above {}% threshold",
                report.summary.regressions, regression_threshold
            );
        }
        if verification_summary.should_fail_ci() {
            eprintln!(
                "\n{} critical verification failure(s)",
                verification_summary.critical_failures + verification_summary.critical_errors
            );
        }
        std::process::exit(1);
    }

    Ok(())
}

fn resolve_git_ref(git_ref: &str) -> anyhow::Result<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--verify", git_ref])
        .output()
        .map_err(|e| anyhow::anyhow!("Failed to resolve git ref '{}': {}", git_ref, e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!(
            "Invalid git ref '{}': {}",
            git_ref,
            stderr.trim()
        ));
    }

    let resolved = String::from_utf8(output.stdout)?.trim().to_string();
    if resolved.is_empty() {
        return Err(anyhow::anyhow!(
            "Git ref '{}' resolved to an empty commit hash",
            git_ref
        ));
    }

    Ok(resolved)
}

/// Format comparison output for human display
fn format_comparison_output(
    report: &fluxbench_report::Report,
    baseline: &fluxbench_report::Report,
) -> String {
    let mut output = String::new();

    output.push('\n');
    output.push_str("FluxBench Comparison Results\n");
    output.push_str(&"=".repeat(60));
    output.push_str("\n\n");

    output.push_str(&format!(
        "Baseline: {} ({})\n",
        baseline.meta.git_commit.as_deref().unwrap_or("unknown"),
        baseline.meta.timestamp.format("%Y-%m-%d %H:%M:%S")
    ));
    output.push_str(&format!(
        "Current:  {} ({})\n\n",
        report.meta.git_commit.as_deref().unwrap_or("unknown"),
        report.meta.timestamp.format("%Y-%m-%d %H:%M:%S")
    ));

    for result in &report.results {
        let status_icon = match result.status {
            fluxbench_report::BenchmarkStatus::Passed => "✓",
            fluxbench_report::BenchmarkStatus::Failed => "✗",
            fluxbench_report::BenchmarkStatus::Crashed => "💥",
            fluxbench_report::BenchmarkStatus::Skipped => "⊘",
        };

        output.push_str(&format!("{} {}\n", status_icon, result.id));

        if let (Some(metrics), Some(comparison)) = (&result.metrics, &result.comparison) {
            let change_icon = if comparison.relative_change > 5.0 {
                "📈 REGRESSION"
            } else if comparison.relative_change < -5.0 {
                "📉 improvement"
            } else {
                "≈ no change"
            };

            output.push_str(&format!(
                "    baseline: {:.2} ns → current: {:.2} ns\n",
                comparison.baseline_mean_ns, metrics.mean_ns
            ));
            output.push_str(&format!(
                "    change: {:+.2}% ({:+.2} ns) {}\n",
                comparison.relative_change, comparison.absolute_change_ns, change_icon
            ));
        }

        output.push('\n');
    }

    // Summary
    output.push_str("Summary\n");
    output.push_str(&"-".repeat(60));
    output.push('\n');
    output.push_str(&format!(
        "  Regressions: {}  Improvements: {}  No Change: {}\n",
        report.summary.regressions,
        report.summary.improvements,
        report.summary.total_benchmarks - report.summary.regressions - report.summary.improvements
    ));

    output
}
