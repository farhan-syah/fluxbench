#![warn(missing_docs)]
//! FluxBench Core - Worker Runtime
//!
//! This crate provides the execution environment for benchmarks:
//! - `Bencher` struct for iteration-based benchmarking
//! - High-precision timing (RDTSC with Instant fallback)
//! - Global allocator interceptor for memory tracking
//! - CPU affinity pinning for stable TSC readings

mod allocator;
mod bencher;
mod measure;
mod worker;

pub use allocator::{TrackingAllocator, current_allocation, reset_allocation_counter};
pub use bencher::{Bencher, BenchmarkResult, IterationMode, run_benchmark_loop};
/// Whether this platform provides hardware cycle counters (x86_64 RDTSCP or AArch64 CNTVCT_EL0).
/// When `false`, cycle counts are reported as 0 and only wall-clock nanoseconds are available.
pub use measure::HAS_CYCLE_COUNTER;
pub use measure::Instant;
pub use measure::Timer;
pub use worker::WorkerMain;

/// Benchmark definition registered via `#[flux::bench]`
#[derive(Debug, Clone)]
pub struct BenchmarkDef {
    /// Unique identifier
    pub id: &'static str,
    /// Human-readable name
    pub name: &'static str,
    /// Group this benchmark belongs to
    pub group: &'static str,
    /// Severity level for CI reporting
    pub severity: Severity,
    /// Per-benchmark regression threshold percentage (0.0 = use global threshold)
    pub threshold: f64,
    /// Absolute time budget in nanoseconds
    pub budget_ns: Option<u64>,
    /// Tags for filtering
    pub tags: &'static [&'static str],
    /// Function pointer to the wrapper
    pub runner_fn: fn(&mut Bencher),
    /// Source file path
    pub file: &'static str,
    /// Source line number
    pub line: u32,
    /// Module path
    pub module_path: &'static str,
    /// Per-benchmark warmup override (nanoseconds)
    pub warmup_ns: Option<u64>,
    /// Per-benchmark measurement override (nanoseconds)
    pub measurement_ns: Option<u64>,
    /// Per-benchmark fixed sample count
    pub samples: Option<u64>,
    /// Per-benchmark minimum iterations
    pub min_iterations: Option<u64>,
    /// Per-benchmark maximum iterations
    pub max_iterations: Option<u64>,
    /// Benchmark IDs that must run before this one
    pub depends_on: &'static [&'static str],
}

/// Severity levels for CI integration
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    /// Critical benchmark - regression fails the build
    Critical,
    /// Warning level - logged but doesn't fail
    Warning,
    /// Informational only
    Info,
}

/// Group definition for organizing benchmarks
#[derive(Debug, Clone)]
pub struct GroupDef {
    /// Group identifier
    pub id: &'static str,
    /// Human-readable description
    pub description: &'static str,
    /// Tags for filtering
    pub tags: &'static [&'static str],
    /// Parent group (for nested groups)
    pub parent: Option<&'static str>,
}

/// Chart type for dashboard layout
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChartType {
    /// Violin plot
    Violin,
    /// Bar chart
    Bar,
    /// Scatter plot
    Scatter,
    /// Line chart
    Line,
    /// Histogram
    Histogram,
}

/// Chart definition for dashboard
#[derive(Debug, Clone)]
pub struct ChartDef {
    /// Chart title
    pub title: &'static str,
    /// Chart type
    pub chart_type: ChartType,
    /// Grid position (row, col)
    pub position: (u32, u32),
    /// Items to display (benchmark IDs)
    pub items: &'static [&'static str],
    /// Optional target line
    pub target_line: Option<f64>,
}

/// Report/dashboard definition
#[derive(Debug, Clone)]
pub struct ReportDef {
    /// Dashboard title
    pub title: &'static str,
    /// Grid layout (rows, cols)
    pub layout: (u32, u32),
    /// Charts in the dashboard
    pub charts: &'static [ChartDef],
}

/// Comparison group - groups multiple benchmarks for side-by-side comparison
#[derive(Debug, Clone)]
pub struct CompareDef {
    /// Comparison identifier
    pub id: &'static str,
    /// Human-readable title
    pub title: &'static str,
    /// Benchmark IDs to compare
    pub benchmarks: &'static [&'static str],
    /// Optional baseline benchmark (first one if not specified)
    pub baseline: Option<&'static str>,
    /// Metric to compare (default: mean)
    pub metric: &'static str,
    /// Group for chart generation (comparisons with same group form a chart)
    pub group: Option<&'static str>,
    /// X-axis value for this comparison point (e.g., "1", "10", "100")
    pub x: Option<&'static str>,
    /// Series labels (display names) - must match benchmarks array length
    /// If not specified, benchmark IDs are used as labels
    pub series: Option<&'static [&'static str]>,
}

// Collect all registered benchmarks
inventory::collect!(BenchmarkDef);
inventory::collect!(GroupDef);
inventory::collect!(ReportDef);
inventory::collect!(CompareDef);

/// Anchor to prevent LTO from stripping inventory entries
#[used]
#[doc(hidden)]
pub static REGISTRY_ANCHOR: fn() = || {
    for _ in inventory::iter::<BenchmarkDef> {}
    for _ in inventory::iter::<GroupDef> {}
    for _ in inventory::iter::<ReportDef> {}
    for _ in inventory::iter::<CompareDef> {}
};
