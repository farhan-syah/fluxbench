//! Report Building
//!
//! Constructs the complete benchmark report from execution results,
//! including parallel computation of bootstrap confidence intervals.
//!
//! ## Pipeline
//!
//! ```text
//! BenchExecutionResult + SummaryStatistics
//!              │
//!              ▼
//!   ┌─────────────────────┐
//!   │ Parallel Bootstrap  │  Rayon-parallelized CI computation
//!   │   CI Computation    │  (100k iterations per benchmark)
//!   └──────────┬──────────┘
//!              │
//!              ▼
//!   ┌─────────────────────┐
//!   │  BenchmarkMetrics   │  All timing stats + allocations + cycles
//!   └──────────┬──────────┘
//!              │
//!              ▼
//!   ┌─────────────────────┐
//!   │      Report         │  Ready for JSON/HTML/CSV/GitHub output
//!   └─────────────────────┘
//! ```
//!
//! The expensive bootstrap computation is parallelized via Rayon, making
//! report generation scale with available CPU cores.

use super::execution::{BenchExecutionResult, ExecutionConfig};
use super::metadata::build_report_meta;
use fluxbench_report::{
    BenchmarkMetrics, BenchmarkReportResult, BenchmarkStatus, FailureInfo, Report, ReportSummary,
};
use fluxbench_stats::{
    BootstrapConfig, SummaryStatistics, compute_bootstrap, compute_cycles_stats,
};
use rayon::prelude::*;

/// Build a complete Report from execution results
///
/// Uses parallel computation for bootstrap CI calculations (the expensive part).
///
/// # Arguments
/// * `results` - Benchmark execution results
/// * `stats` - Pre-computed summary statistics for each benchmark
/// * `config` - Execution configuration (for bootstrap settings)
/// * `total_duration_ms` - Total execution time in milliseconds
///
/// # Returns
/// Complete Report structure ready for output
pub fn build_report(
    results: &[BenchExecutionResult],
    stats: &[(String, Option<SummaryStatistics>)],
    config: &ExecutionConfig,
    total_duration_ms: f64,
) -> Report {
    // Build stats lookup
    let stats_map: std::collections::HashMap<_, _> = stats.iter().cloned().collect();

    // Compute metrics in parallel (bootstrap is expensive)
    let metrics_vec: Vec<_> = results
        .par_iter()
        .map(|result| {
            let stats_opt = stats_map.get(&result.benchmark_id).cloned().flatten();

            stats_opt.as_ref().map(|s| {
                // Compute bootstrap CI (expensive - parallelized)
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
                    skewness: s.skewness,
                    kurtosis: s.kurtosis,
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
            })
        })
        .collect();

    // Build final results sequentially (cheap - just aggregation)
    let mut benchmark_results = Vec::with_capacity(results.len());
    let mut summary = ReportSummary {
        total_benchmarks: results.len(),
        total_duration_ms,
        ..Default::default()
    };

    for (result, metrics) in results.iter().zip(metrics_vec) {
        let failure = result.error_message.as_ref().map(|msg| FailureInfo {
            kind: result
                .failure_kind
                .clone()
                .unwrap_or_else(|| "panic".to_string()),
            message: msg.clone(),
            backtrace: result.backtrace.clone(),
        });

        match result.status {
            BenchmarkStatus::Passed => summary.passed += 1,
            BenchmarkStatus::Failed => summary.failed += 1,
            BenchmarkStatus::Crashed => summary.crashed += 1,
            BenchmarkStatus::Skipped => summary.skipped += 1,
        }

        benchmark_results.push(BenchmarkReportResult {
            id: result.benchmark_id.clone(),
            name: result.benchmark_name.clone(),
            group: result.group.clone(),
            status: result.status,
            severity: result.severity,
            file: result.file.clone(),
            line: result.line,
            metrics,
            threshold: result.threshold,
            comparison: None, // Filled when comparing to baseline
            failure,
        });
    }

    Report {
        meta: build_report_meta(config),
        results: benchmark_results,
        comparisons: Vec::new(),       // Filled by execute_verifications
        comparison_series: Vec::new(), // Filled by execute_verifications
        synthetics: Vec::new(),        // Filled by execute_verifications
        verifications: Vec::new(),     // Filled by execute_verifications
        summary,
        baseline_meta: None,
    }
}
