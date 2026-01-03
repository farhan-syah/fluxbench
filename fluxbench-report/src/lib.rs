//! FluxBench Report - Reporting and Visualization
//!
//! Generates various output formats:
//! - JSON (machine-readable)
//! - GitHub Summary (Markdown for $GITHUB_STEP_SUMMARY)
//! - CSV (spreadsheet-compatible)
//! - HTML (interactive dashboard)

mod github;
mod html;
mod json;
mod report;

pub use github::generate_github_summary;
pub use html::generate_html_report;
pub use json::{generate_json_report, ReportSchema};
pub use report::{
    BenchmarkMetrics, BenchmarkResult as ReportBenchmarkResult, BenchmarkStatus, Comparison,
    FailureInfo, Report, ReportMeta, ReportSummary, SystemInfo,
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
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(OutputFormat::Json),
            "github" | "github-summary" => Ok(OutputFormat::GithubSummary),
            "csv" => Ok(OutputFormat::Csv),
            "html" => Ok(OutputFormat::Html),
            "human" | "text" => Ok(OutputFormat::Human),
            other => Err(format!("Unknown output format: {}", other)),
        }
    }
}
