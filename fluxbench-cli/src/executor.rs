//! Benchmark Executor
//!
//! Runs benchmarks and collects results. Supports both in-process execution
//! (for development) and out-of-process execution via supervisor-worker IPC.

use crate::supervisor::{IpcBenchmarkResult, IpcBenchmarkStatus, Supervisor};
use fluxbench_core::{run_benchmark_loop, Bencher, BenchmarkDef};
use fluxbench_ipc::BenchmarkConfig;
use fluxbench_logic::{
    run_verifications, MetricContext, Verification, VerificationContext, VerificationResult,
    VerifyDef,
};
use fluxbench_report::{
    BenchmarkMetrics, BenchmarkStatus, FailureInfo, Report, ReportBenchmarkResult, ReportMeta,
    ReportSummary, SystemInfo,
};
use fluxbench_stats::{
    compute_bootstrap, compute_cycles_stats, compute_summary, BootstrapConfig, OutlierMethod,
    SummaryStatistics,
};
use fxhash::FxHashSet;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

/// Configuration for benchmark execution
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Warmup time in nanoseconds
    pub warmup_time_ns: u64,
    /// Measurement time in nanoseconds
    pub measurement_time_ns: u64,
    /// Minimum iterations
    pub min_iterations: Option<u64>,
    /// Maximum iterations
    pub max_iterations: Option<u64>,
    /// Track allocations
    pub track_allocations: bool,
    /// Number of bootstrap iterations for statistics
    pub bootstrap_iterations: usize,
    /// Confidence level for intervals
    pub confidence_level: f64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            warmup_time_ns: 3_000_000_000,     // 3 seconds
            measurement_time_ns: 5_000_000_000, // 5 seconds
            min_iterations: Some(100),
            max_iterations: None,
            track_allocations: true,
            bootstrap_iterations: 100_000, // Matches Criterion default
            confidence_level: 0.95,
        }
    }
}

/// Result from executing a single benchmark
#[derive(Debug)]
pub struct BenchExecutionResult {
    pub benchmark_id: String,
    pub benchmark_name: String,
    pub group: String,
    pub file: String,
    pub line: u32,
    pub status: BenchmarkStatus,
    pub samples: Vec<f64>,
    /// CPU cycles per sample (parallel with samples)
    pub cpu_cycles: Vec<u32>,
    pub alloc_bytes: u64,
    pub alloc_count: u64,
    pub duration_ns: u64,
    pub error_message: Option<String>,
}

/// Execute benchmarks and produce results
pub struct Executor {
    config: ExecutionConfig,
    results: Vec<BenchExecutionResult>,
}

impl Executor {
    pub fn new(config: ExecutionConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    /// Execute all provided benchmarks
    pub fn execute(&mut self, benchmarks: &[&BenchmarkDef]) -> Vec<BenchExecutionResult> {
        let pb = ProgressBar::new(benchmarks.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("#>-"),
        );

        for bench in benchmarks {
            pb.set_message(bench.id.to_string());
            let result = self.execute_single(bench);
            self.results.push(result);
            pb.inc(1);
        }

        pb.finish_with_message("Complete");
        std::mem::take(&mut self.results)
    }

    /// Execute a single benchmark
    fn execute_single(&self, bench: &BenchmarkDef) -> BenchExecutionResult {
        let start = Instant::now();

        // Run with panic catching
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let bencher = Bencher::new(self.config.track_allocations);

            run_benchmark_loop(
                bencher,
                |b| (bench.runner_fn)(b),
                self.config.warmup_time_ns,
                self.config.measurement_time_ns,
                self.config.max_iterations,
            )
        }));

        let duration_ns = start.elapsed().as_nanos() as u64;

        match result {
            Ok(bench_result) => {
                // Extract timing samples as f64 for statistics
                let samples: Vec<f64> = bench_result
                    .samples
                    .iter()
                    .map(|s| s.duration_nanos as f64)
                    .collect();

                // Extract CPU cycles (parallel array with samples)
                let cpu_cycles: Vec<u32> = bench_result
                    .samples
                    .iter()
                    .map(|s| s.cpu_cycles)
                    .collect();

                // Sum allocations
                let alloc_bytes: u64 = bench_result.samples.iter().map(|s| s.alloc_bytes).sum();
                let alloc_count: u64 =
                    bench_result.samples.iter().map(|s| s.alloc_count as u64).sum();

                BenchExecutionResult {
                    benchmark_id: bench.id.to_string(),
                    benchmark_name: bench.name.to_string(),
                    group: bench.group.to_string(),
                    file: bench.file.to_string(),
                    line: bench.line,
                    status: BenchmarkStatus::Passed,
                    samples,
                    cpu_cycles,
                    alloc_bytes,
                    alloc_count,
                    duration_ns,
                    error_message: None,
                }
            }
            Err(panic) => {
                let message = if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };

                BenchExecutionResult {
                    benchmark_id: bench.id.to_string(),
                    benchmark_name: bench.name.to_string(),
                    group: bench.group.to_string(),
                    file: bench.file.to_string(),
                    line: bench.line,
                    status: BenchmarkStatus::Crashed,
                    samples: Vec::new(),
                    cpu_cycles: Vec::new(),
                    alloc_bytes: 0,
                    alloc_count: 0,
                    duration_ns,
                    error_message: Some(message),
                }
            }
        }
    }
}

/// Executor that runs benchmarks in isolated worker processes via IPC
///
/// This provides crash isolation - if a benchmark panics or crashes,
/// it won't take down the supervisor process.
pub struct IsolatedExecutor {
    config: ExecutionConfig,
    timeout: Duration,
    reuse_workers: bool,
}

impl IsolatedExecutor {
    /// Create a new isolated executor
    pub fn new(config: ExecutionConfig, timeout: Duration, reuse_workers: bool) -> Self {
        Self {
            config,
            timeout,
            reuse_workers,
        }
    }

    /// Execute all provided benchmarks in isolated worker processes
    pub fn execute(&self, benchmarks: &[&BenchmarkDef]) -> Vec<BenchExecutionResult> {
        let pb = ProgressBar::new(benchmarks.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("#>-"),
        );
        pb.set_message("Starting isolated workers...");

        // Convert ExecutionConfig to BenchmarkConfig for IPC
        let ipc_config = BenchmarkConfig {
            warmup_time_ns: self.config.warmup_time_ns,
            measurement_time_ns: self.config.measurement_time_ns,
            min_iterations: self.config.min_iterations,
            max_iterations: self.config.max_iterations,
            track_allocations: self.config.track_allocations,
            fail_on_allocation: false,
            timeout_ns: self.timeout.as_nanos() as u64,
        };

        let supervisor = Supervisor::new(ipc_config, self.timeout, 1);

        // Run benchmarks via IPC
        let ipc_results = if self.reuse_workers {
            supervisor.run_with_reuse(benchmarks)
        } else {
            supervisor.run_all(benchmarks)
        };

        // Convert IPC results to BenchExecutionResult
        let mut results = Vec::with_capacity(benchmarks.len());

        match ipc_results {
            Ok(ipc_results) => {
                for (ipc_result, bench) in ipc_results.into_iter().zip(benchmarks.iter()) {
                    pb.set_message(bench.id.to_string());
                    results.push(self.convert_ipc_result(ipc_result, bench));
                    pb.inc(1);
                }
            }
            Err(e) => {
                // Supervisor-level failure - mark all as crashed
                for bench in benchmarks {
                    results.push(BenchExecutionResult {
                        benchmark_id: bench.id.to_string(),
                        benchmark_name: bench.name.to_string(),
                        group: bench.group.to_string(),
                        file: bench.file.to_string(),
                        line: bench.line,
                        status: BenchmarkStatus::Crashed,
                        samples: Vec::new(),
                        cpu_cycles: Vec::new(),
                        alloc_bytes: 0,
                        alloc_count: 0,
                        duration_ns: 0,
                        error_message: Some(format!("Supervisor error: {}", e)),
                    });
                    pb.inc(1);
                }
            }
        }

        pb.finish_with_message("Complete (isolated)");
        results
    }

    /// Convert an IPC result to a BenchExecutionResult
    fn convert_ipc_result(
        &self,
        ipc_result: IpcBenchmarkResult,
        bench: &BenchmarkDef,
    ) -> BenchExecutionResult {
        let (status, error_message) = match ipc_result.status {
            IpcBenchmarkStatus::Success => (BenchmarkStatus::Passed, None),
            IpcBenchmarkStatus::Failed { message } => (BenchmarkStatus::Failed, Some(message)),
            IpcBenchmarkStatus::Crashed { message } => (BenchmarkStatus::Crashed, Some(message)),
        };

        // Extract timing samples as f64 for statistics
        let samples: Vec<f64> = ipc_result
            .samples
            .iter()
            .map(|s| s.duration_nanos as f64)
            .collect();

        // Extract CPU cycles
        let cpu_cycles: Vec<u32> = ipc_result.samples.iter().map(|s| s.cpu_cycles).collect();

        // Sum allocations
        let alloc_bytes: u64 = ipc_result.samples.iter().map(|s| s.alloc_bytes).sum();
        let alloc_count: u64 = ipc_result
            .samples
            .iter()
            .map(|s| s.alloc_count as u64)
            .sum();

        BenchExecutionResult {
            benchmark_id: bench.id.to_string(),
            benchmark_name: bench.name.to_string(),
            group: bench.group.to_string(),
            file: bench.file.to_string(),
            line: bench.line,
            status,
            samples,
            cpu_cycles,
            alloc_bytes,
            alloc_count,
            duration_ns: ipc_result.total_duration_nanos,
            error_message,
        }
    }
}

/// Compute statistics for benchmark results
pub fn compute_statistics(
    results: &[BenchExecutionResult],
    _config: &ExecutionConfig,
) -> Vec<(String, Option<SummaryStatistics>)> {
    results
        .iter()
        .map(|r| {
            if r.samples.is_empty() {
                (r.benchmark_id.clone(), None)
            } else {
                let stats = compute_summary(&r.samples, OutlierMethod::Iqr { k: 3 }); // k=3 means 1.5*IQR
                (r.benchmark_id.clone(), Some(stats))
            }
        })
        .collect()
}

/// Build a complete Report from execution results
pub fn build_report(
    results: &[BenchExecutionResult],
    stats: &[(String, Option<SummaryStatistics>)],
    config: &ExecutionConfig,
    total_duration_ms: f64,
) -> Report {
    // Build stats lookup
    let stats_map: std::collections::HashMap<_, _> = stats.iter().cloned().collect();

    // Build benchmark results
    let mut benchmark_results = Vec::new();
    let mut summary = ReportSummary {
        total_benchmarks: results.len(),
        total_duration_ms,
        ..Default::default()
    };

    for result in results {
        let stats_opt = stats_map.get(&result.benchmark_id).cloned().flatten();

        let metrics = stats_opt.as_ref().map(|s| {
            // Compute bootstrap CI
            let bootstrap_config = BootstrapConfig {
                iterations: config.bootstrap_iterations,
                confidence_level: config.confidence_level,
                ..Default::default()
            };
            let bootstrap_result = compute_bootstrap(&result.samples, &bootstrap_config);

            let (ci_lower, ci_upper) = match bootstrap_result {
                Ok(br) => (br.confidence_interval.lower, br.confidence_interval.upper),
                Err(_) => (s.mean, s.mean), // Fallback to point estimate
            };

            let throughput = if s.mean > 0.0 {
                Some(1_000_000_000.0 / s.mean)
            } else {
                None
            };

            // Compute CPU cycles statistics
            let cycles_stats = compute_cycles_stats(&result.cpu_cycles, &result.samples);

            BenchmarkMetrics {
                samples: s.sample_count,
                mean_ns: s.mean,
                median_ns: s.median,
                std_dev_ns: s.std_dev,
                min_ns: s.min,
                max_ns: s.max,
                p50_ns: s.p50,
                p90_ns: s.p90,
                p95_ns: s.p95,
                p99_ns: s.p99,
                p999_ns: s.p999,
                ci_lower_ns: ci_lower,
                ci_upper_ns: ci_upper,
                ci_level: config.confidence_level,
                throughput_ops_sec: throughput,
                alloc_bytes: result.alloc_bytes,
                alloc_count: result.alloc_count,
                // CPU cycles from RDTSC (x86_64 only, 0 on other platforms)
                mean_cycles: cycles_stats.mean_cycles,
                median_cycles: cycles_stats.median_cycles,
                min_cycles: cycles_stats.min_cycles,
                max_cycles: cycles_stats.max_cycles,
                cycles_per_ns: cycles_stats.cycles_per_ns,
            }
        });

        let failure = result.error_message.as_ref().map(|msg| FailureInfo {
            kind: "panic".to_string(),
            message: msg.clone(),
            backtrace: None,
        });

        match result.status {
            BenchmarkStatus::Passed => summary.passed += 1,
            BenchmarkStatus::Failed => summary.failed += 1,
            BenchmarkStatus::Crashed => summary.crashed += 1,
            BenchmarkStatus::Skipped => summary.skipped += 1,
        }

        benchmark_results.push(ReportBenchmarkResult {
            id: result.benchmark_id.clone(),
            name: result.benchmark_name.clone(),
            group: result.group.clone(),
            status: result.status,
            file: result.file.clone(),
            line: result.line,
            metrics,
            comparison: None, // Filled when comparing to baseline
            failure,
        });
    }

    Report {
        meta: build_report_meta(),
        results: benchmark_results,
        verifications: Vec::new(), // Filled by run_verifications if any
        summary,
    }
}

/// Build report metadata
fn build_report_meta() -> ReportMeta {
    use chrono::Utc;

    // Get git info if available
    let git_commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string());

    let git_branch = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string());

    // Get system info
    let system = SystemInfo {
        os: std::env::consts::OS.to_string(),
        os_version: std::env::consts::ARCH.to_string(),
        cpu: get_cpu_model().unwrap_or_else(|| "Unknown".to_string()),
        cpu_cores: num_cpus(),
        memory_gb: get_memory_gb().unwrap_or(0.0),
    };

    ReportMeta {
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: Utc::now(),
        git_commit,
        git_branch,
        system,
    }
}

fn get_cpu_model() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

fn get_memory_gb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|l| l.starts_with("MemTotal"))
                    .and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|s| s.parse::<u64>().ok())
                    })
                    .map(|kb| kb as f64 / 1024.0 / 1024.0)
            })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Run verifications against computed metrics
pub fn execute_verifications(
    _results: &[BenchExecutionResult],
    stats: &[(String, Option<SummaryStatistics>)],
) -> Vec<VerificationResult> {
    // Build metric context with benchmark results
    let mut context = MetricContext::new();
    let mut unavailable = FxHashSet::default();

    for (bench_id, stats_opt) in stats {
        if let Some(stats) = stats_opt {
            // Add mean as the primary metric for each benchmark
            context.set(bench_id, stats.mean);
            // Add prefixed versions for explicit access
            context.set(format!("{}_mean", bench_id), stats.mean);
            context.set(format!("{}_median", bench_id), stats.median);
            context.set(format!("{}_min", bench_id), stats.min);
            context.set(format!("{}_max", bench_id), stats.max);
            context.set(format!("{}_p99", bench_id), stats.p99);
        } else {
            unavailable.insert(bench_id.clone());
        }
    }

    // Collect all registered verifications
    let verifications: Vec<Verification> = inventory::iter::<VerifyDef>
        .into_iter()
        .map(|v| Verification {
            id: v.id.to_string(),
            expression: v.expression.to_string(),
            severity: v.severity,
            margin: v.margin,
        })
        .collect();

    if verifications.is_empty() {
        return Vec::new();
    }

    // Run verifications
    let verification_context = VerificationContext::new(&context, unavailable);
    run_verifications(&verifications, &verification_context)
}

/// Format output for human-readable display
pub fn format_human_output(report: &Report) -> String {
    let mut output = String::new();

    output.push('\n');
    output.push_str("FluxBench Results\n");
    output.push_str(&"=".repeat(60));
    output.push_str("\n\n");

    // Group results
    let mut groups: std::collections::BTreeMap<&str, Vec<&ReportBenchmarkResult>> =
        std::collections::BTreeMap::new();
    for result in &report.results {
        groups.entry(&result.group).or_default().push(result);
    }

    for (group, results) in groups {
        output.push_str(&format!("Group: {}\n", group));
        output.push_str(&"-".repeat(60));
        output.push('\n');

        for result in results {
            let status_icon = match result.status {
                BenchmarkStatus::Passed => "✓",
                BenchmarkStatus::Failed => "✗",
                BenchmarkStatus::Crashed => "💥",
                BenchmarkStatus::Skipped => "⊘",
            };

            output.push_str(&format!("  {} {}\n", status_icon, result.id));

            if let Some(metrics) = &result.metrics {
                output.push_str(&format!(
                    "      mean: {:.2} ns  median: {:.2} ns  stddev: {:.2} ns\n",
                    metrics.mean_ns, metrics.median_ns, metrics.std_dev_ns
                ));
                output.push_str(&format!(
                    "      min: {:.2} ns  max: {:.2} ns  samples: {}\n",
                    metrics.min_ns, metrics.max_ns, metrics.samples
                ));
                output.push_str(&format!(
                    "      p50: {:.2} ns  p95: {:.2} ns  p99: {:.2} ns\n",
                    metrics.p50_ns, metrics.p95_ns, metrics.p99_ns
                ));
                output.push_str(&format!(
                    "      95% CI: [{:.2}, {:.2}] ns\n",
                    metrics.ci_lower_ns, metrics.ci_upper_ns
                ));
                if let Some(throughput) = metrics.throughput_ops_sec {
                    output.push_str(&format!("      throughput: {:.2} ops/sec\n", throughput));
                }
                if metrics.alloc_bytes > 0 {
                    output.push_str(&format!(
                        "      allocations: {} bytes ({} allocs)\n",
                        metrics.alloc_bytes, metrics.alloc_count
                    ));
                }
                // Show CPU cycles if available (x86_64 only)
                if metrics.mean_cycles > 0.0 {
                    output.push_str(&format!(
                        "      cycles: mean {:.0}  median {:.0}  ({:.2} GHz)\n",
                        metrics.mean_cycles, metrics.median_cycles, metrics.cycles_per_ns
                    ));
                }
            }

            if let Some(failure) = &result.failure {
                output.push_str(&format!("      error: {}\n", failure.message));
            }

            output.push('\n');
        }
    }

    // Verifications
    if !report.verifications.is_empty() {
        output.push_str("\nVerifications\n");
        output.push_str(&"-".repeat(60));
        output.push('\n');

        for v in &report.verifications {
            let icon = if v.passed() { "✓" } else { "✗" };
            output.push_str(&format!("  {} {} : {}\n", icon, v.id, v.message));
        }
    }

    // Summary
    output.push_str("\nSummary\n");
    output.push_str(&"-".repeat(60));
    output.push('\n');
    output.push_str(&format!(
        "  Total: {}  Passed: {}  Failed: {}  Crashed: {}  Skipped: {}\n",
        report.summary.total_benchmarks,
        report.summary.passed,
        report.summary.failed,
        report.summary.crashed,
        report.summary.skipped
    ));
    output.push_str(&format!(
        "  Duration: {:.2} ms\n",
        report.summary.total_duration_ms
    ));

    output
}
