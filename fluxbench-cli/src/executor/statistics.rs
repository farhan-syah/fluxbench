//! Statistics Computation
//!
//! Parallel computation of summary statistics for benchmark results.
//!
//! Uses Rayon to parallelize statistics computation across benchmarks.
//! Each benchmark's samples are processed independently to compute:
//! - Central tendency (mean, median)
//! - Dispersion (std dev, min, max)
//! - Percentiles (p50, p90, p95, p99, p999)
//! - Outlier detection via IQR method

use super::execution::{BenchExecutionResult, ExecutionConfig};
use fluxbench_stats::{OutlierMethod, SummaryStatistics, compute_summary};
use rayon::prelude::*;

/// Compute statistics for benchmark results (parallelized with Rayon)
///
/// Uses Rayon for parallel computation of summary statistics across all benchmarks.
/// Each benchmark's samples are independently processed, making this highly parallelizable.
///
/// # Arguments
/// * `results` - Benchmark execution results containing samples
/// * `_config` - Execution configuration. Currently unused but reserved for future
///   extensions such as configurable outlier detection methods, custom percentile
///   thresholds, or alternative statistical algorithms.
///
/// # Returns
/// Vector of (benchmark_id, optional statistics) pairs. Returns `None` for benchmarks
/// with no samples (e.g., crashed or skipped benchmarks).
pub fn compute_statistics(
    results: &[BenchExecutionResult],
    _config: &ExecutionConfig,
) -> Vec<(String, Option<SummaryStatistics>)> {
    results
        .par_iter() // Parallel iteration
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
