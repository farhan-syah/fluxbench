//! Report Data Structures

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use fluxbench_stats::SummaryStatistics;

/// Complete benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub meta: ReportMeta,
    pub results: Vec<BenchmarkReportResult>,
    pub comparisons: Vec<ComparisonResult>,
    pub comparison_series: Vec<ComparisonSeries>,
    pub synthetics: Vec<fluxbench_logic::SyntheticResult>,
    pub verifications: Vec<fluxbench_logic::VerificationResult>,
    pub summary: ReportSummary,
}

/// Result of a comparison group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Comparison identifier
    pub id: String,
    /// Human-readable title
    pub title: String,
    /// Baseline benchmark ID
    pub baseline: String,
    /// Metric used for comparison
    pub metric: String,
    /// Individual benchmark entries in the comparison
    pub entries: Vec<ComparisonEntry>,
}

/// Single entry in a comparison table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonEntry {
    /// Benchmark ID
    pub benchmark_id: String,
    /// Metric value (e.g., mean time in ns)
    pub value: f64,
    /// Speedup vs baseline (1.0 = same, >1.0 = faster, <1.0 = slower)
    pub speedup: f64,
    /// Whether this is the baseline
    pub is_baseline: bool,
}

/// Grouped comparison series for multi-point charts (e.g., batch size scaling)
/// Multiple #[compare(group = "...", x = "...")] combine into one series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSeries {
    /// Group identifier
    pub group: String,
    /// Chart title (from first comparison's title)
    pub title: String,
    /// X-axis values in order (e.g., ["1", "10", "100", "1000"])
    pub x_values: Vec<String>,
    /// Competitor/series names (benchmark IDs)
    pub series_names: Vec<String>,
    /// Data points: series_data[series_idx][x_idx] = value
    pub series_data: Vec<Vec<f64>>,
    /// Metric used
    pub metric: String,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMeta {
    pub version: String,
    pub timestamp: DateTime<Utc>,
    pub git_commit: Option<String>,
    pub git_branch: Option<String>,
    pub system: SystemInfo,
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub os_version: String,
    pub cpu: String,
    pub cpu_cores: u32,
    pub memory_gb: f64,
}

/// Individual benchmark result in the report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReportResult {
    pub id: String,
    pub name: String,
    pub group: String,
    pub status: BenchmarkStatus,
    pub file: String,
    pub line: u32,
    pub metrics: Option<BenchmarkMetrics>,
    pub comparison: Option<Comparison>,
    pub failure: Option<FailureInfo>,
}

/// Benchmark execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BenchmarkStatus {
    Passed,
    Failed,
    Crashed,
    Skipped,
}

/// Benchmark timing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub samples: usize,
    pub mean_ns: f64,
    pub median_ns: f64,
    pub std_dev_ns: f64,
    pub min_ns: f64,
    pub max_ns: f64,
    pub p50_ns: f64,
    pub p90_ns: f64,
    pub p95_ns: f64,
    pub p99_ns: f64,
    pub p999_ns: f64,
    pub ci_lower_ns: f64,
    pub ci_upper_ns: f64,
    pub ci_level: f64,
    pub throughput_ops_sec: Option<f64>,
    pub alloc_bytes: u64,
    pub alloc_count: u64,
    // CPU cycles metrics (x86_64 only, 0 on other platforms)
    pub mean_cycles: f64,
    pub median_cycles: f64,
    pub min_cycles: u32,
    pub max_cycles: u32,
    pub cycles_per_ns: f64,
}

impl From<&SummaryStatistics> for BenchmarkMetrics {
    fn from(stats: &SummaryStatistics) -> Self {
        Self {
            samples: stats.sample_count,
            mean_ns: stats.mean,
            median_ns: stats.median,
            std_dev_ns: stats.std_dev,
            min_ns: stats.min,
            max_ns: stats.max,
            p50_ns: stats.p50,
            p90_ns: stats.p90,
            p95_ns: stats.p95,
            p99_ns: stats.p99,
            p999_ns: stats.p999,
            ci_lower_ns: 0.0, // Filled by bootstrap
            ci_upper_ns: 0.0,
            ci_level: 0.95,
            throughput_ops_sec: None,
            alloc_bytes: 0,
            alloc_count: 0,
            // Cycles filled separately via CyclesStatistics
            mean_cycles: 0.0,
            median_cycles: 0.0,
            min_cycles: 0,
            max_cycles: 0,
            cycles_per_ns: 0.0,
        }
    }
}

/// Comparison against baseline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparison {
    pub baseline_mean_ns: f64,
    pub absolute_change_ns: f64,
    pub relative_change: f64,
    pub probability_regression: f64,
    pub is_significant: bool,
    pub effect_size: f64,
}

/// Failure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureInfo {
    pub kind: String,
    pub message: String,
    pub backtrace: Option<String>,
}

/// Report summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_benchmarks: usize,
    pub passed: usize,
    pub failed: usize,
    pub crashed: usize,
    pub skipped: usize,
    pub regressions: usize,
    pub improvements: usize,
    pub critical_failures: usize,
    pub warnings: usize,
    pub total_duration_ms: f64,
}
