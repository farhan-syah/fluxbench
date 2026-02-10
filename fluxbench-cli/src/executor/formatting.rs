//! Output Formatting
//!
//! Human-readable output formatting for benchmark reports.
//!
//! Generates terminal-friendly output with:
//! - Grouped benchmark results with status icons (✓/✗/💥/⊘)
//! - Timing metrics (mean, median, stddev, percentiles)
//! - Confidence intervals and throughput
//! - Allocation and CPU cycle statistics
//! - Comparison tables with speedup calculations
//! - Verification results summary

use fluxbench_report::{BenchmarkReportResult, BenchmarkStatus, Report};

/// Format a report for human-readable terminal display
///
/// # Arguments
/// * `report` - Complete benchmark report
///
/// # Returns
/// Formatted string suitable for terminal output
pub fn format_human_output(report: &Report) -> String {
    let mut output = String::new();

    output.push('\n');
    output.push_str("FluxBench Results\n");
    output.push_str(&"=".repeat(60));
    output.push_str("\n\n");

    // Group results
    let mut groups: std::collections::BTreeMap<&str, Vec<&BenchmarkReportResult>> =
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

    // Comparisons
    for cmp in &report.comparisons {
        output.push_str(&format!("\n{}\n", cmp.title));
        output.push_str(&"-".repeat(60));
        output.push('\n');

        // Find max benchmark name length for alignment
        let max_name_len = cmp
            .entries
            .iter()
            .map(|e| e.benchmark_id.len())
            .max()
            .unwrap_or(20);

        // Header
        output.push_str(&format!(
            "  {:<width$}  {:>12}  {:>10}\n",
            "Benchmark",
            cmp.metric,
            "Speedup",
            width = max_name_len
        ));
        output.push_str(&format!("  {}\n", "-".repeat(max_name_len + 26)));

        // Entries sorted by speedup (fastest first)
        let mut sorted_entries: Vec<_> = cmp.entries.iter().collect();
        sorted_entries.sort_by(|a, b| {
            b.speedup
                .partial_cmp(&a.speedup)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for entry in sorted_entries {
            let baseline_marker = if entry.is_baseline { " (baseline)" } else { "" };
            let speedup_str = if entry.is_baseline {
                "1.00x".to_string()
            } else {
                format!("{:.2}x", entry.speedup)
            };

            output.push_str(&format!(
                "  {:<width$}  {:>12.2}  {:>10}{}\n",
                entry.benchmark_id,
                entry.value,
                speedup_str,
                baseline_marker,
                width = max_name_len
            ));
        }
    }

    // Comparison Series (grouped multi-point comparisons for charts)
    for series in &report.comparison_series {
        output.push_str(&format!("\n{} ({})\n", series.title, series.metric));
        output.push_str(&"-".repeat(60));
        output.push('\n');

        // Find max series name length for alignment
        let max_name_len = series
            .series_names
            .iter()
            .map(|n| n.len())
            .max()
            .unwrap_or(12);

        // Determine column width based on x values and data
        let col_width = series
            .x_values
            .iter()
            .map(|x| x.len())
            .max()
            .unwrap_or(8)
            .max(10); // At least 10 chars for numbers

        // Header row with x values
        output.push_str(&format!("  {:<width$}", "", width = max_name_len));
        for x in &series.x_values {
            output.push_str(&format!(" | {:>w$}", x, w = col_width));
        }
        output.push('\n');

        // Separator
        output.push_str(&format!("  {}", "-".repeat(max_name_len)));
        for _ in &series.x_values {
            output.push_str(&format!("-+-{}", "-".repeat(col_width)));
        }
        output.push('\n');

        // Data rows
        for (series_idx, name) in series.series_names.iter().enumerate() {
            output.push_str(&format!("  {:<width$}", name, width = max_name_len));
            for x_idx in 0..series.x_values.len() {
                let value = series
                    .series_data
                    .get(series_idx)
                    .and_then(|row| row.get(x_idx))
                    .copied()
                    .unwrap_or(0.0);
                // Format nicely: use scientific notation for very large/small numbers
                let formatted = if value == 0.0 {
                    "-".to_string()
                } else if value.abs() >= 1_000_000.0 || (value.abs() < 0.001 && value != 0.0) {
                    format!("{:.2e}", value)
                } else if value.abs() >= 1000.0 {
                    format!("{:.0}", value)
                } else {
                    format!("{:.2}", value)
                };
                output.push_str(&format!(" | {:>w$}", formatted, w = col_width));
            }
            output.push('\n');
        }
    }

    // Computed Metrics (Synthetics)
    if !report.synthetics.is_empty() {
        output.push_str("\nComputed Metrics\n");
        output.push_str(&"-".repeat(60));
        output.push('\n');

        for s in &report.synthetics {
            let unit = s.unit.as_deref().unwrap_or("");
            output.push_str(&format!(
                "  {} = {:.2}{} ({})\n",
                s.id, s.value, unit, s.formula
            ));
        }
    }

    // Verifications (only show if there are non-skipped ones)
    let active_verifications: Vec<_> = report
        .verifications
        .iter()
        .filter(|v| {
            !matches!(
                v.status,
                fluxbench_logic::VerificationStatus::Skipped { .. }
            )
        })
        .collect();

    if !active_verifications.is_empty() {
        output.push_str("\nVerifications\n");
        output.push_str(&"-".repeat(60));
        output.push('\n');

        for v in active_verifications {
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
