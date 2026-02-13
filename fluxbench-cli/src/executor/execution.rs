//! Benchmark Execution
//!
//! Core execution logic for running benchmarks, including both in-process
//! and isolated (IPC-based) execution modes.
//!
//! ## Execution Modes
//!
//! - **In-process (`Executor`)**: Runs benchmarks in the same process. Fast but
//!   a panic in one benchmark will crash the entire run. Best for development.
//!
//! - **Isolated (`IsolatedExecutor`)**: Spawns worker processes via IPC. Provides
//!   crash isolation - a panic in one benchmark won't affect others. Recommended
//!   for production runs.
//!
//! ## Data Flow
//!
//! ```text
//! BenchmarkDef (from inventory)
//!        │
//!        ▼
//!   ExecutionConfig
//!        │
//!        ▼
//! ┌──────────────────┐
//! │  Executor/       │  Warmup → Measurement → Sample Collection
//! │  IsolatedExecutor│
//! └────────┬─────────┘
//!          │
//!          ▼
//!  BenchExecutionResult (samples, status, allocations, cycles)
//! ```

use crate::supervisor::{IpcBenchmarkResult, IpcBenchmarkStatus, Supervisor};
use fluxbench_core::{Bencher, BenchmarkDef, run_benchmark_loop};
use fluxbench_ipc::BenchmarkConfig;
use fluxbench_report::BenchmarkStatus;
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

impl ExecutionConfig {
    /// Merge per-benchmark configuration overrides with global defaults.
    ///
    /// Priority:
    /// 1. Per-benchmark `samples` (if set): overrides everything, runs fixed N iterations with no warmup
    /// 2. Per-benchmark `warmup_ns`/`measurement_ns`: override global values
    /// 3. Per-benchmark `min/max_iterations`: override global values
    /// 4. Falls back to global config for anything not overridden
    pub fn resolve_for_benchmark(&self, bench: &BenchmarkDef) -> ExecutionConfig {
        // Fixed sample count mode: per-bench samples override everything
        if let Some(n) = bench.samples {
            return ExecutionConfig {
                warmup_time_ns: 0,
                measurement_time_ns: 0,
                min_iterations: Some(n),
                max_iterations: Some(n),
                ..self.clone()
            };
        }

        ExecutionConfig {
            warmup_time_ns: bench.warmup_ns.unwrap_or(self.warmup_time_ns),
            measurement_time_ns: bench.measurement_ns.unwrap_or(self.measurement_time_ns),
            min_iterations: bench.min_iterations.or(self.min_iterations),
            max_iterations: bench.max_iterations.or(self.max_iterations),
            ..self.clone()
        }
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            warmup_time_ns: 3_000_000_000,      // 3 seconds
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
    pub cpu_cycles: Vec<u64>,
    pub alloc_bytes: u64,
    pub alloc_count: u64,
    pub duration_ns: u64,
    pub error_message: Option<String>,
    pub failure_kind: Option<String>,
    pub backtrace: Option<String>,
    pub severity: fluxbench_core::Severity,
    /// Per-benchmark regression threshold (0.0 = use global)
    pub threshold: f64,
}

/// Execute benchmarks and produce results (in-process mode)
pub struct Executor {
    config: ExecutionConfig,
    results: Vec<BenchExecutionResult>,
}

impl Executor {
    /// Create a new in-process executor with the given configuration
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
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
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
        let cfg = self.config.resolve_for_benchmark(bench);

        // Run with panic catching
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let bencher = Bencher::new(cfg.track_allocations);

            run_benchmark_loop(
                bencher,
                |b| (bench.runner_fn)(b),
                cfg.warmup_time_ns,
                cfg.measurement_time_ns,
                cfg.min_iterations,
                cfg.max_iterations,
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
                let cpu_cycles: Vec<u64> =
                    bench_result.samples.iter().map(|s| s.cpu_cycles).collect();

                // Sum allocations
                let alloc_bytes: u64 = bench_result.samples.iter().map(|s| s.alloc_bytes).sum();
                let alloc_count: u64 = bench_result
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
                    status: BenchmarkStatus::Passed,
                    samples,
                    cpu_cycles,
                    alloc_bytes,
                    alloc_count,
                    duration_ns,
                    error_message: None,
                    failure_kind: None,
                    backtrace: None,
                    severity: bench.severity,
                    threshold: bench.threshold,
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
                    failure_kind: Some("panic".to_string()),
                    backtrace: None,
                    severity: bench.severity,
                    threshold: bench.threshold,
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
    num_workers: usize,
}

impl IsolatedExecutor {
    /// Create a new isolated executor
    pub fn new(
        config: ExecutionConfig,
        timeout: Duration,
        reuse_workers: bool,
        num_workers: usize,
    ) -> Self {
        Self {
            config,
            timeout,
            reuse_workers,
            num_workers: num_workers.max(1),
        }
    }

    /// Execute all provided benchmarks in isolated worker processes
    pub fn execute(&self, benchmarks: &[&BenchmarkDef]) -> Vec<BenchExecutionResult> {
        let pb = ProgressBar::new(benchmarks.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
                )
                .unwrap_or_else(|_| ProgressStyle::default_bar())
                .progress_chars("#>-"),
        );
        pb.set_message("Starting isolated workers...");

        // Build per-benchmark IPC configs
        let ipc_configs: Vec<BenchmarkConfig> = benchmarks
            .iter()
            .map(|bench| {
                let cfg = self.config.resolve_for_benchmark(bench);
                BenchmarkConfig {
                    warmup_time_ns: cfg.warmup_time_ns,
                    measurement_time_ns: cfg.measurement_time_ns,
                    min_iterations: cfg.min_iterations,
                    max_iterations: cfg.max_iterations,
                    track_allocations: cfg.track_allocations,
                    fail_on_allocation: false,
                    timeout_ns: self.timeout.as_nanos() as u64,
                }
            })
            .collect();

        // Use the first config as default for the supervisor
        let default_config = ipc_configs.first().cloned().unwrap_or(BenchmarkConfig {
            warmup_time_ns: self.config.warmup_time_ns,
            measurement_time_ns: self.config.measurement_time_ns,
            min_iterations: self.config.min_iterations,
            max_iterations: self.config.max_iterations,
            track_allocations: self.config.track_allocations,
            fail_on_allocation: false,
            timeout_ns: self.timeout.as_nanos() as u64,
        });

        let supervisor = Supervisor::new(default_config, self.timeout, self.num_workers);

        // Run benchmarks via IPC with per-benchmark configs
        let ipc_results = if self.reuse_workers {
            supervisor.run_with_reuse_configs(benchmarks, &ipc_configs)
        } else {
            supervisor.run_all_configs(benchmarks, &ipc_configs)
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
                        failure_kind: Some("crashed".to_string()),
                        backtrace: None,
                        severity: bench.severity,
                        threshold: bench.threshold,
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
        let (status, error_message, failure_kind, backtrace) = match ipc_result.status {
            IpcBenchmarkStatus::Success => (BenchmarkStatus::Passed, None, None, None),
            IpcBenchmarkStatus::Failed {
                message,
                kind,
                backtrace,
            } => (
                BenchmarkStatus::Failed,
                Some(message),
                Some(kind),
                backtrace,
            ),
            IpcBenchmarkStatus::Crashed {
                message,
                kind,
                backtrace,
            } => (
                BenchmarkStatus::Crashed,
                Some(message),
                Some(kind),
                backtrace,
            ),
        };

        // Extract timing samples as f64 for statistics
        let samples: Vec<f64> = ipc_result
            .samples
            .iter()
            .map(|s| s.duration_nanos as f64)
            .collect();

        // Extract CPU cycles
        let cpu_cycles: Vec<u64> = ipc_result.samples.iter().map(|s| s.cpu_cycles).collect();

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
            failure_kind,
            backtrace,
            severity: bench.severity,
            threshold: bench.threshold,
        }
    }
}
