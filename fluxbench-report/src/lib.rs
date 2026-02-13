#![warn(missing_docs)]
//! FluxBench Report - Reporting and Visualization
//!
//! Generates various output formats:
//! - JSON (machine-readable)
//! - GitHub Summary (Markdown for $GITHUB_STEP_SUMMARY)
//! - CSV (spreadsheet-compatible)
//! - HTML (interactive dashboard)

mod csv;
mod github;
mod github_action;
mod html;
mod json;
mod report;

pub use csv::generate_csv_report;
pub use github::{
    format_duration, generate_github_summary, scale_duration, scale_duration_with_reference,
    scale_throughput,
};
pub use github_action::generate_github_action_benchmark;
pub use html::generate_html_report;
pub use json::{ReportSchema, generate_json_report};
pub use report::{
    BenchmarkMetrics, BenchmarkReportResult, BenchmarkStatus, Comparison, ComparisonEntry,
    ComparisonResult, ComparisonSeries, FailureInfo, Report, ReportConfig, ReportMeta,
    ReportSummary, SystemInfo,
};

/// Output format selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// JSON with full schema
    Json,
    /// Markdown for GitHub Actions
    GithubSummary,
    /// CSV for spreadsheets
    Csv,
    /// Single-file HTML dashboard
    Html,
    /// Human-readable terminal output
    Human,
    /// github-action-benchmark compatible JSON (customSmallerIsBetter)
    GithubActionBenchmark,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(OutputFormat::Json),
            "github" | "github-summary" => Ok(OutputFormat::GithubSummary),
            "github-action" | "github-action-benchmark" => Ok(OutputFormat::GithubActionBenchmark),
            "csv" => Ok(OutputFormat::Csv),
            "html" => Ok(OutputFormat::Html),
            "human" | "text" => Ok(OutputFormat::Human),
            other => Err(format!("Unknown output format: {}", other)),
        }
    }
}
