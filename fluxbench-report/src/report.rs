//! Report Data Structures

use chrono::{DateTime, Utc};
use fluxbench_stats::SummaryStatistics;
use serde::{Deserialize, Serialize};

/// Complete benchmark report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    /// Report metadata and configuration
    pub meta: ReportMeta,
    /// Individual benchmark results
    pub results: Vec<BenchmarkReportResult>,
    /// Comparison groups between benchmarks
    pub comparisons: Vec<ComparisonResult>,
    /// Multi-point comparison series (e.g., batch size scaling)
    pub comparison_series: Vec<ComparisonSeries>,
    /// Synthetic measurement results
    pub synthetics: Vec<fluxbench_logic::SyntheticResult>,
    /// Verification results
    pub verifications: Vec<fluxbench_logic::VerificationResult>,
    /// Overall report summary statistics
    pub summary: ReportSummary,
    /// Baseline report metadata (populated when comparing against a baseline)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_meta: Option<ReportMeta>,
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
    /// Data points: `series_data[series_idx][x_idx] = value`
    pub series_data: Vec<Vec<f64>>,
    /// Metric used
    pub metric: String,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMeta {
    /// Schema version number
    pub schema_version: u32,
    /// Report version string
    pub version: String,
    /// Report generation timestamp
    pub timestamp: DateTime<Utc>,
    /// Git commit hash (if available)
    pub git_commit: Option<String>,
    /// Git branch name (if available)
    pub git_branch: Option<String>,
    /// System information where benchmarks ran
    pub system: SystemInfo,
    /// Benchmark execution configuration
    pub config: ReportConfig,
}

/// Execution configuration captured in report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    /// Warmup duration in nanoseconds
    pub warmup_time_ns: u64,
    /// Measurement duration in nanoseconds
    pub measurement_time_ns: u64,
    /// Minimum benchmark iterations (if specified)
    pub min_iterations: Option<u64>,
    /// Maximum benchmark iterations (if specified)
    pub max_iterations: Option<u64>,
    /// Bootstrap iterations for confidence intervals
    pub bootstrap_iterations: usize,
    /// Confidence level for intervals (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Whether memory allocations were tracked
    pub track_allocations: bool,
}

/// System information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    /// Operating system name
    pub os: String,
    /// Operating system version
    pub os_version: String,
    /// CPU model name
    pub cpu: String,
    /// Number of CPU cores
    pub cpu_cores: u32,
    /// Total system memory in gigabytes
    pub memory_gb: f64,
}

/// Individual benchmark result in the report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReportResult {
    /// Unique benchmark identifier
    pub id: String,
    /// Human-readable benchmark name
    pub name: String,
    /// Benchmark group/category
    pub group: String,
    /// Execution status (passed, failed, etc.)
    pub status: BenchmarkStatus,
    /// Severity level if failed
    pub severity: fluxbench_core::Severity,
    /// Source file path
    pub file: String,
    /// Source line number
    pub line: u32,
    /// Timing and statistical metrics (if successful)
    pub metrics: Option<BenchmarkMetrics>,
    /// Per-benchmark regression threshold percentage (0.0 = use global)
    #[serde(default)]
    pub threshold: f64,
    /// Comparison results against baseline (if applicable)
    pub comparison: Option<Comparison>,
    /// Failure details (if failed)
    pub failure: Option<FailureInfo>,
}

/// Benchmark execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BenchmarkStatus {
    /// Benchmark completed successfully
    Passed,
    /// Benchmark ran but assertion failed
    Failed,
    /// Benchmark crashed or panicked
    Crashed,
    /// Benchmark was skipped
    Skipped,
}

/// Benchmark timing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Number of samples collected
    pub samples: usize,
    /// Mean execution time in nanoseconds
    pub mean_ns: f64,
    /// Median execution time in nanoseconds
    pub median_ns: f64,
    /// Standard deviation in nanoseconds
    pub std_dev_ns: f64,
    /// Minimum execution time in nanoseconds
    pub min_ns: f64,
    /// Maximum execution time in nanoseconds
    pub max_ns: f64,
    /// 50th percentile (P50) in nanoseconds
    pub p50_ns: f64,
    /// 90th percentile (P90) in nanoseconds
    pub p90_ns: f64,
    /// 95th percentile (P95) in nanoseconds
    pub p95_ns: f64,
    /// 99th percentile (P99) in nanoseconds
    pub p99_ns: f64,
    /// 99.9th percentile (P99.9) in nanoseconds
    pub p999_ns: f64,
    /// Distribution skewness
    pub skewness: f64,
    /// Distribution kurtosis
    pub kurtosis: f64,
    /// Confidence interval lower bound in nanoseconds
    pub ci_lower_ns: f64,
    /// Confidence interval upper bound in nanoseconds
    pub ci_upper_ns: f64,
    /// Confidence level used for interval (e.g., 0.95)
    pub ci_level: f64,
    /// Throughput in operations per second (if applicable)
    pub throughput_ops_sec: Option<f64>,
    /// Total bytes allocated during measurement
    pub alloc_bytes: u64,
    /// Total allocation count during measurement
    pub alloc_count: u64,
    /// Mean CPU cycles (x86_64 only, 0 on other platforms)
    pub mean_cycles: f64,
    /// Median CPU cycles (x86_64 only, 0 on other platforms)
    pub median_cycles: f64,
    /// Minimum CPU cycles (x86_64 only, 0 on other platforms)
    pub min_cycles: u64,
    /// Maximum CPU cycles (x86_64 only, 0 on other platforms)
    pub max_cycles: u64,
    /// CPU cycles per nanosecond
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
            skewness: stats.skewness,
            kurtosis: stats.kurtosis,
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
    /// Baseline mean time in nanoseconds
    pub baseline_mean_ns: f64,
    /// Absolute change from baseline in nanoseconds
    pub absolute_change_ns: f64,
    /// Relative change from baseline (percentage)
    pub relative_change: f64,
    /// Probability of regression (0.0 to 1.0)
    pub probability_regression: f64,
    /// Whether the change is statistically significant
    pub is_significant: bool,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
}

/// Failure information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureInfo {
    /// Failure type/kind
    pub kind: String,
    /// Error message
    pub message: String,
    /// Stack backtrace (if available)
    pub backtrace: Option<String>,
}

/// Report summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Total number of benchmarks
    pub total_benchmarks: usize,
    /// Number of passed benchmarks
    pub passed: usize,
    /// Number of failed benchmarks
    pub failed: usize,
    /// Number of crashed benchmarks
    pub crashed: usize,
    /// Number of skipped benchmarks
    pub skipped: usize,
    /// Number of performance regressions detected
    pub regressions: usize,
    /// Number of performance improvements detected
    pub improvements: usize,
    /// Number of critical failures
    pub critical_failures: usize,
    /// Number of warnings
    pub warnings: usize,
    /// Total execution duration in milliseconds
    pub total_duration_ms: f64,
}
