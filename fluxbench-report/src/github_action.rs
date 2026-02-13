//! github-action-benchmark Compatible Output
//!
//! Generates JSON output compatible with the `customSmallerIsBetter` format used by
//! [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark).
//!
//! This allows FluxBench results to feed into GitHub Pages dashboards for
//! historical performance tracking across commits.
//!
//! ## Usage
//!
//! ```bash
//! cargo fluxbench --format github-action -o output.json
//! ```
//!
//! Then in your GitHub Actions workflow:
//!
//! ```yaml
//! - uses: benchmark-action/github-action-benchmark@v1
//!   with:
//!     tool: 'customSmallerIsBetter'
//!     output-file-path: output.json
//! ```

use crate::github::{scale_duration, scale_duration_with_reference, scale_throughput};
use crate::report::{BenchmarkStatus, Report};
use serde::Serialize;

/// A single entry in the github-action-benchmark JSON format.
#[derive(Debug, Serialize)]
struct GhaBenchmarkEntry {
    /// Benchmark identifier
    name: String,
    /// Measurement unit
    unit: String,
    /// Measured value
    value: f64,
    /// Variance / standard deviation (optional, displayed as `±range`)
    #[serde(skip_serializing_if = "Option::is_none")]
    range: Option<String>,
    /// Extra context for tooltip display
    #[serde(skip_serializing_if = "Option::is_none")]
    extra: Option<String>,
}

/// Generate github-action-benchmark compatible JSON.
///
/// Emits one entry per benchmark using `customSmallerIsBetter` (lower time = better).
/// Throughput data, if available, is included in the `extra` tooltip field rather than
/// as a separate entry, since github-action-benchmark applies a single direction
/// (`SmallerIsBetter` or `BiggerIsBetter`) to all entries in the file.
pub fn generate_github_action_benchmark(report: &Report) -> String {
    let mut entries: Vec<GhaBenchmarkEntry> = Vec::new();

    for result in &report.results {
        if result.status != BenchmarkStatus::Passed {
            continue;
        }

        let Some(metrics) = &result.metrics else {
            continue;
        };

        let (value, unit) = scale_duration(metrics.mean_ns);
        let (std_dev_scaled, _) =
            scale_duration_with_reference(metrics.std_dev_ns, metrics.mean_ns);

        let mut extra_lines = Vec::new();
        extra_lines.push(format!(
            "median: {:.3} {} | min: {:.3} {} | max: {:.3} {}",
            scale_duration_with_reference(metrics.median_ns, metrics.mean_ns).0,
            unit,
            scale_duration_with_reference(metrics.min_ns, metrics.mean_ns).0,
            unit,
            scale_duration_with_reference(metrics.max_ns, metrics.mean_ns).0,
            unit,
        ));
        extra_lines.push(format!(
            "p95: {:.3} {} | p99: {:.3} {}",
            scale_duration_with_reference(metrics.p95_ns, metrics.mean_ns).0,
            unit,
            scale_duration_with_reference(metrics.p99_ns, metrics.mean_ns).0,
            unit,
        ));
        extra_lines.push(format!("samples: {}", metrics.samples));
        if let Some(throughput) = metrics.throughput_ops_sec {
            let (tp_value, tp_unit) = scale_throughput(throughput);
            extra_lines.push(format!("throughput: {:.3} {}", tp_value, tp_unit));
        }

        entries.push(GhaBenchmarkEntry {
            name: result.id.clone(),
            unit: unit.to_string(),
            value: round3(value),
            range: Some(format!("± {:.3}", std_dev_scaled)),
            extra: Some(extra_lines.join("\n")),
        });
    }

    serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
}

/// Round to 3 decimal places for clean JSON output.
fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::*;

    fn test_metrics(mean_ns: f64) -> BenchmarkMetrics {
        BenchmarkMetrics {
            samples: 100,
            mean_ns,
            median_ns: mean_ns * 0.99,
            std_dev_ns: mean_ns * 0.02,
            min_ns: mean_ns * 0.9,
            max_ns: mean_ns * 1.1,
            p50_ns: mean_ns * 0.99,
            p90_ns: mean_ns * 1.05,
            p95_ns: mean_ns * 1.07,
            p99_ns: mean_ns * 1.09,
            p999_ns: mean_ns * 1.1,
            skewness: 0.1,
            kurtosis: 3.0,
            ci_lower_ns: mean_ns * 0.98,
            ci_upper_ns: mean_ns * 1.02,
            ci_level: 0.95,
            throughput_ops_sec: Some(1_000_000_000.0 / mean_ns),
            alloc_bytes: 0,
            alloc_count: 0,
            mean_cycles: 0.0,
            median_cycles: 0.0,
            min_cycles: 0,
            max_cycles: 0,
            cycles_per_ns: 0.0,
        }
    }

    fn test_report(results: Vec<BenchmarkReportResult>) -> Report {
        let total = results.len();
        Report {
            meta: ReportMeta {
                schema_version: 1,
                version: "0.1.0".to_string(),
                timestamp: chrono::Utc::now(),
                git_commit: Some("abc1234".to_string()),
                git_branch: Some("main".to_string()),
                system: SystemInfo {
                    os: "linux".to_string(),
                    os_version: "6.0".to_string(),
                    cpu: "test".to_string(),
                    cpu_cores: 8,
                    memory_gb: 16.0,
                },
                config: ReportConfig {
                    warmup_time_ns: 3_000_000_000,
                    measurement_time_ns: 5_000_000_000,
                    min_iterations: None,
                    max_iterations: None,
                    bootstrap_iterations: 10_000,
                    confidence_level: 0.95,
                    track_allocations: false,
                },
            },
            results,
            comparisons: vec![],
            comparison_series: vec![],
            synthetics: vec![],
            verifications: vec![],
            summary: ReportSummary {
                total_benchmarks: total,
                passed: total,
                ..Default::default()
            },
            baseline_meta: None,
        }
    }

    #[test]
    fn generates_valid_json_array() {
        let report = test_report(vec![BenchmarkReportResult {
            id: "matmul_512".to_string(),
            name: "matmul_512".to_string(),
            group: "matmul".to_string(),
            status: BenchmarkStatus::Passed,
            severity: fluxbench_core::Severity::Warning,
            file: "benches/matmul.rs".to_string(),
            line: 10,
            metrics: Some(test_metrics(1_500_000.0)), // 1.5ms
            threshold: 0.0,
            comparison: None,
            failure: None,
        }]);

        let json = generate_github_action_benchmark(&report);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0]["name"], "matmul_512");
        assert_eq!(parsed[0]["unit"], "ms");
        assert_eq!(parsed[0]["value"], 1.5);
        assert!(parsed[0]["range"].as_str().unwrap().starts_with("±"));
        let extra = parsed[0]["extra"].as_str().unwrap();
        assert!(extra.contains("throughput:"));
    }

    #[test]
    fn skips_failed_benchmarks() {
        let report = test_report(vec![BenchmarkReportResult {
            id: "crasher".to_string(),
            name: "crasher".to_string(),
            group: "test".to_string(),
            status: BenchmarkStatus::Crashed,
            severity: fluxbench_core::Severity::Critical,
            file: "test.rs".to_string(),
            line: 1,
            metrics: None,
            threshold: 0.0,
            comparison: None,
            failure: Some(FailureInfo {
                kind: "panic".to_string(),
                message: "boom".to_string(),
                backtrace: None,
            }),
        }]);

        let json = generate_github_action_benchmark(&report);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_empty());
    }

    #[test]
    fn scales_nanosecond_benchmarks() {
        let report = test_report(vec![BenchmarkReportResult {
            id: "fast_op".to_string(),
            name: "fast_op".to_string(),
            group: "test".to_string(),
            status: BenchmarkStatus::Passed,
            severity: fluxbench_core::Severity::Warning,
            file: "test.rs".to_string(),
            line: 1,
            metrics: Some(test_metrics(50.0)), // 50ns
            threshold: 0.0,
            comparison: None,
            failure: None,
        }]);

        let json = generate_github_action_benchmark(&report);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed[0]["unit"], "ns");
        assert_eq!(parsed[0]["value"], 50.0);
    }

    #[test]
    fn scales_microsecond_benchmarks() {
        let report = test_report(vec![BenchmarkReportResult {
            id: "medium_op".to_string(),
            name: "medium_op".to_string(),
            group: "test".to_string(),
            status: BenchmarkStatus::Passed,
            severity: fluxbench_core::Severity::Warning,
            file: "test.rs".to_string(),
            line: 1,
            metrics: Some(test_metrics(5_000.0)), // 5us
            threshold: 0.0,
            comparison: None,
            failure: None,
        }]);

        let json = generate_github_action_benchmark(&report);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed[0]["unit"], "us");
        assert_eq!(parsed[0]["value"], 5.0);
    }

    #[test]
    fn no_throughput_in_extra_when_absent() {
        let mut metrics = test_metrics(1_000_000.0);
        metrics.throughput_ops_sec = None;

        let report = test_report(vec![BenchmarkReportResult {
            id: "no_tp".to_string(),
            name: "no_tp".to_string(),
            group: "test".to_string(),
            status: BenchmarkStatus::Passed,
            severity: fluxbench_core::Severity::Warning,
            file: "test.rs".to_string(),
            line: 1,
            metrics: Some(metrics),
            threshold: 0.0,
            comparison: None,
            failure: None,
        }]);

        let json = generate_github_action_benchmark(&report);
        let parsed: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
        let extra = parsed[0]["extra"].as_str().unwrap();
        assert!(!extra.contains("throughput:"));
    }
}
