//! Configuration loading from flux.toml
//!
//! FluxBench configuration can be specified in a `flux.toml` file in the project root.
//! The configuration is automatically discovered by walking up from the current directory.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// FluxBench configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FluxConfig {
    /// Runner configuration
    #[serde(default)]
    pub runner: RunnerConfig,
    /// Visualization configuration
    #[serde(default)]
    pub visuals: VisualsConfig,
    /// Allocator tracking configuration
    #[serde(default)]
    pub allocator: AllocatorConfig,
    /// Output configuration
    #[serde(default)]
    pub output: OutputConfig,
    /// CI/CD configuration
    #[serde(default)]
    pub ci: CiConfig,
}

/// Isolation mode for benchmark execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum IsolationMode {
    /// Run each benchmark in a separate worker process (default)
    #[default]
    Process,
    /// Run benchmarks in-process (no isolation, useful for debugging)
    InProcess,
    /// Run benchmarks in threads (no isolation)
    Thread,
}

impl IsolationMode {
    /// Whether this mode provides process isolation
    pub fn is_isolated(self) -> bool {
        matches!(self, IsolationMode::Process)
    }
}

/// Runner configuration for benchmark execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerConfig {
    /// Timeout for a single benchmark (e.g., "60s", "5m")
    #[serde(default = "default_timeout")]
    pub timeout: String,
    /// Isolation mode: "process", "in-process", or "thread"
    #[serde(default)]
    pub isolation: IsolationMode,
    /// Warmup duration before measurement (e.g., "3s")
    #[serde(default = "default_warmup")]
    pub warmup_time: String,
    /// Measurement duration (e.g., "5s")
    #[serde(default = "default_measurement")]
    pub measurement_time: String,
    /// Fixed sample count: skip warmup, run exactly N iterations (each = one sample)
    #[serde(default)]
    pub samples: Option<u64>,
    /// Minimum number of iterations
    #[serde(default)]
    pub min_iterations: Option<u64>,
    /// Maximum number of iterations
    #[serde(default)]
    pub max_iterations: Option<u64>,
    /// Number of bootstrap iterations for statistics
    #[serde(default = "default_bootstrap_iterations")]
    pub bootstrap_iterations: usize,
    /// Confidence level (e.g., 0.95 for 95%)
    #[serde(default = "default_confidence_level")]
    pub confidence_level: f64,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            timeout: default_timeout(),
            isolation: IsolationMode::default(),
            warmup_time: default_warmup(),
            measurement_time: default_measurement(),
            samples: None,
            min_iterations: None,
            max_iterations: None,
            bootstrap_iterations: default_bootstrap_iterations(),
            confidence_level: default_confidence_level(),
        }
    }
}

fn default_timeout() -> String {
    "60s".to_string()
}
fn default_warmup() -> String {
    "3s".to_string()
}
fn default_measurement() -> String {
    "5s".to_string()
}
fn default_bootstrap_iterations() -> usize {
    10_000
}
fn default_confidence_level() -> f64 {
    0.95
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualsConfig {
    /// Color theme: "light" or "dark"
    #[serde(default = "default_theme")]
    pub theme: String,
    /// Chart width in pixels
    #[serde(default = "default_width")]
    pub width: u32,
    /// Chart height in pixels
    #[serde(default = "default_height")]
    pub height: u32,
}

impl Default for VisualsConfig {
    fn default() -> Self {
        Self {
            theme: default_theme(),
            width: default_width(),
            height: default_height(),
        }
    }
}

fn default_theme() -> String {
    "light".to_string()
}
fn default_width() -> u32 {
    1280
}
fn default_height() -> u32 {
    720
}

/// Allocator tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocatorConfig {
    /// Enable allocation tracking
    #[serde(default = "default_track")]
    pub track: bool,
    /// Fail if any allocation occurs during measurement
    #[serde(default)]
    pub fail_on_allocation: bool,
    /// Maximum bytes allowed per iteration (None = unlimited)
    #[serde(default)]
    pub max_bytes_per_iter: Option<u64>,
}

impl Default for AllocatorConfig {
    fn default() -> Self {
        Self {
            track: default_track(),
            fail_on_allocation: false,
            max_bytes_per_iter: None,
        }
    }
}

fn default_track() -> bool {
    true
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Default output format: "human", "json", "github", "html", "csv"
    #[serde(default = "default_format")]
    pub format: String,
    /// Output directory for reports
    #[serde(default = "default_output_dir")]
    pub directory: String,
    /// Save JSON baseline after each run
    #[serde(default)]
    pub save_baseline: bool,
    /// Baseline file path
    #[serde(default)]
    pub baseline_path: Option<String>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: default_format(),
            directory: default_output_dir(),
            save_baseline: false,
            baseline_path: None,
        }
    }
}

fn default_format() -> String {
    "human".to_string()
}
fn default_output_dir() -> String {
    "target/fluxbench".to_string()
}

/// CI/CD configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CiConfig {
    /// Regression threshold percentage (fail if exceeded)
    #[serde(default = "default_threshold")]
    pub regression_threshold: f64,
    /// Enable GitHub Actions annotations
    #[serde(default)]
    pub github_annotations: bool,
    /// Fail on any critical verification failure
    #[serde(default = "default_fail_on_critical")]
    pub fail_on_critical: bool,
}

impl Default for CiConfig {
    fn default() -> Self {
        Self {
            regression_threshold: default_threshold(),
            github_annotations: false,
            fail_on_critical: default_fail_on_critical(),
        }
    }
}

fn default_threshold() -> f64 {
    5.0
}
fn default_fail_on_critical() -> bool {
    true
}

impl FluxConfig {
    /// Load configuration from a TOML file
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }

    /// Try to discover and load configuration by walking up from current directory
    pub fn discover() -> Option<Self> {
        let mut dir = std::env::current_dir().ok()?;
        loop {
            let config_path = dir.join("flux.toml");
            if config_path.exists() {
                return Self::load(&config_path).ok();
            }
            if !dir.pop() {
                break;
            }
        }
        None
    }

    /// Generate a default configuration as TOML string
    pub fn default_toml() -> String {
        r#"# FluxBench Configuration
# https://github.com/ml-rust/fluxbench

[runner]
# Warmup duration before measurement
warmup_time = "3s"
# Measurement duration
measurement_time = "5s"
# Timeout for a single benchmark
timeout = "60s"
# Isolation mode: "process" or "thread"
isolation = "process"  # "process", "in-process", or "thread"
# Fixed sample count: skip warmup, run exactly N iterations (uncomment to enable)
# samples = 5
# Minimum iterations (uncomment to enable)
# min_iterations = 100
# Maximum iterations (uncomment to enable)
# max_iterations = 1000000
# Bootstrap iterations for confidence intervals
bootstrap_iterations = 10000
# Confidence level (0.0 to 1.0)
confidence_level = 0.95

[allocator]
# Track memory allocations during benchmarks
track = true
# Fail if any allocation occurs during measurement
fail_on_allocation = false
# Maximum bytes per iteration (uncomment to enable)
# max_bytes_per_iter = 1024

[output]
# Default output format: human, json, github, html, csv
format = "human"
# Output directory for reports
directory = "target/fluxbench"
# Save JSON baseline after each run
save_baseline = false
# Baseline file for comparison (uncomment to enable)
# baseline_path = "baseline.json"

[visuals]
# Color theme: light or dark
theme = "light"
# Chart dimensions
width = 1280
height = 720

[ci]
# Regression threshold percentage (fail CI if exceeded)
regression_threshold = 5.0
# Enable GitHub Actions annotations
github_annotations = false
# Fail on critical verification failures
fail_on_critical = true
"#
        .to_string()
    }

    /// Parse duration string (e.g., "3s", "500ms", "2m") to nanoseconds
    pub fn parse_duration(s: &str) -> anyhow::Result<u64> {
        let s = s.trim();
        if s.is_empty() {
            return Err(anyhow::anyhow!("Empty duration string"));
        }

        // Find where the number ends and unit begins
        let (num_part, unit_part) = s
            .char_indices()
            .find(|(_, c)| c.is_alphabetic())
            .map(|(i, _)| s.split_at(i))
            .unwrap_or((s, "s"));

        let value: f64 = num_part
            .parse()
            .map_err(|_| anyhow::anyhow!("Invalid duration number: {}", num_part))?;

        let multiplier: u64 = match unit_part.to_lowercase().as_str() {
            "ns" => 1,
            "us" | "µs" => 1_000,
            "ms" => 1_000_000,
            "s" | "" => 1_000_000_000,
            "m" | "min" => 60_000_000_000,
            _ => return Err(anyhow::anyhow!("Unknown duration unit: {}", unit_part)),
        };

        Ok((value * multiplier as f64) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FluxConfig::default();
        assert_eq!(config.runner.warmup_time, "3s");
        assert_eq!(config.runner.measurement_time, "5s");
        assert!(config.allocator.track);
        assert!(!config.allocator.fail_on_allocation);
    }

    #[test]
    fn test_parse_duration() {
        assert_eq!(FluxConfig::parse_duration("3s").unwrap(), 3_000_000_000);
        assert_eq!(FluxConfig::parse_duration("500ms").unwrap(), 500_000_000);
        assert_eq!(FluxConfig::parse_duration("100us").unwrap(), 100_000);
        assert_eq!(FluxConfig::parse_duration("1000ns").unwrap(), 1000);
        assert_eq!(FluxConfig::parse_duration("2m").unwrap(), 120_000_000_000);
        assert_eq!(FluxConfig::parse_duration("1.5s").unwrap(), 1_500_000_000);
    }

    #[test]
    fn test_parse_toml() {
        let toml_str = r#"
            [runner]
            warmup_time = "1s"
            measurement_time = "2s"

            [allocator]
            track = false
        "#;

        let config: FluxConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.runner.warmup_time, "1s");
        assert_eq!(config.runner.measurement_time, "2s");
        assert!(!config.allocator.track);
        // Defaults should still apply
        assert_eq!(config.output.format, "human");
    }

    #[test]
    fn test_default_toml_parses() {
        let default_toml = FluxConfig::default_toml();
        let config: FluxConfig = toml::from_str(&default_toml).unwrap();
        assert_eq!(config.runner.warmup_time, "3s");
    }
}
