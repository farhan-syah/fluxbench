//! System Metadata Collection
//!
//! Collects system information for report metadata including CPU, memory,
//! OS details, and git information.
//!
//! ## Collected Data
//!
//! - **Git**: Current commit hash and branch name
//! - **OS**: Operating system name and architecture
//! - **CPU**: Model name and core count
//! - **Memory**: Total system RAM in GB
//! - **Timestamp**: UTC time of report generation
//!
//! Linux-specific data (CPU model, memory) gracefully degrades on other
//! platforms, returning "Unknown" or 0 values.

use super::execution::ExecutionConfig;
use chrono::Utc;
use fluxbench_report::{ReportConfig, ReportMeta, SystemInfo};

/// Build report metadata including system info and git details
pub fn build_report_meta(exec_config: &ExecutionConfig) -> ReportMeta {
    // Get git info if available
    let git_commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string());

    let git_branch = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string());

    // Get system info
    let system = SystemInfo {
        os: std::env::consts::OS.to_string(),
        os_version: std::env::consts::ARCH.to_string(),
        cpu: get_cpu_model().unwrap_or_else(|| "Unknown".to_string()),
        cpu_cores: num_cpus(),
        memory_gb: get_memory_gb().unwrap_or(0.0),
    };

    ReportMeta {
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: Utc::now(),
        git_commit,
        git_branch,
        system,
        config: ReportConfig {
            warmup_time_ns: exec_config.warmup_time_ns,
            measurement_time_ns: exec_config.measurement_time_ns,
            min_iterations: exec_config.min_iterations,
            max_iterations: exec_config.max_iterations,
            bootstrap_iterations: exec_config.bootstrap_iterations,
            confidence_level: exec_config.confidence_level,
            track_allocations: exec_config.track_allocations,
        },
    }
}

/// Get CPU model name from /proc/cpuinfo (Linux only)
fn get_cpu_model() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|l| l.starts_with("model name"))
                    .and_then(|l| l.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Get number of available CPU cores
fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
}

/// Get total system memory in GB (Linux only)
fn get_memory_gb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|l| l.starts_with("MemTotal"))
                    .and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|s| s.parse::<u64>().ok())
                    })
                    .map(|kb| kb as f64 / 1024.0 / 1024.0)
            })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}
