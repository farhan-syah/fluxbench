//! CSV Report Generation
//!
//! Generates CSV output for spreadsheet-compatible benchmark results.
//! Includes all timing metrics, allocation data, and CPU cycle information.

use crate::{BenchmarkStatus, Report};

/// Generate a CSV report from benchmark results
///
/// # Arguments
/// * `report` - Complete benchmark report
///
/// # Returns
/// CSV-formatted string with header row and data rows for each benchmark
///
/// # CSV Columns
/// - `id` - Benchmark identifier
/// - `name` - Human-readable benchmark name
/// - `group` - Benchmark group
/// - `status` - Execution status (passed/failed/crashed/skipped)
/// - `mean_ns` - Mean execution time in nanoseconds
/// - `median_ns` - Median execution time
/// - `std_dev_ns` - Standard deviation
/// - `min_ns` / `max_ns` - Min/max execution times
/// - `p50_ns` / `p95_ns` / `p99_ns` - Percentiles
/// - `samples` - Number of samples collected
/// - `alloc_bytes` / `alloc_count` - Allocation metrics
/// - `mean_cycles` / `median_cycles` - CPU cycle metrics (x86_64 only)
/// - `cycles_per_ns` - CPU frequency estimate
pub fn generate_csv_report(report: &Report) -> String {
    let mut csv = String::new();

    // Header (includes CPU cycles columns)
    csv.push_str("id,name,group,status,mean_ns,median_ns,std_dev_ns,min_ns,max_ns,p50_ns,p95_ns,p99_ns,samples,alloc_bytes,alloc_count,mean_cycles,median_cycles,cycles_per_ns\n");

    // Data rows
    for result in &report.results {
        let status = match result.status {
            BenchmarkStatus::Passed => "passed",
            BenchmarkStatus::Failed => "failed",
            BenchmarkStatus::Crashed => "crashed",
            BenchmarkStatus::Skipped => "skipped",
        };

        if let Some(metrics) = &result.metrics {
            csv.push_str(&format!(
                "{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{},{},{},{:.0},{:.0},{:.2}\n",
                result.id,
                result.name,
                result.group,
                status,
                metrics.mean_ns,
                metrics.median_ns,
                metrics.std_dev_ns,
                metrics.min_ns,
                metrics.max_ns,
                metrics.p50_ns,
                metrics.p95_ns,
                metrics.p99_ns,
                metrics.samples,
                metrics.alloc_bytes,
                metrics.alloc_count,
                metrics.mean_cycles,
                metrics.median_cycles,
                metrics.cycles_per_ns,
            ));
        } else {
            csv.push_str(&format!(
                "{},{},{},{},,,,,,,,,,,,,\n",
                result.id, result.name, result.group, status,
            ));
        }
    }

    csv
}
