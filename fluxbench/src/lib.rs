//! # FluxBench
//!
//! Production-grade benchmarking for Rust.
//!
//! FluxBench is a next-generation benchmarking platform that provides:
//! - **Process Isolation**: Crash-resilient "Fail-Late" architecture
//! - **Zero-Copy IPC**: Efficient data transfer with rkyv
//! - **Statistical Rigor**: Bootstrap resampling with BCa intervals
//! - **CI Integration**: Severity levels, GitHub summaries, baseline comparison
//! - **Algebraic Verification**: Performance assertions in code
//!
//! ## Quick Start
//!
//! ```ignore
//! use fluxbench::prelude::*;
//!
//! #[flux::bench]
//! fn my_benchmark(b: &mut Bencher) {
//!     b.iter(|| {
//!         // Code to benchmark
//!         expensive_operation()
//!     });
//! }
//! ```
//!
//! ## Async Benchmarks
//!
//! ```ignore
//! #[flux::bench(runtime = "multi_thread", worker_threads = 4)]
//! async fn async_benchmark(b: &mut Bencher) {
//!     b.iter(|| async {
//!         tokio::time::sleep(Duration::from_millis(1)).await;
//!     });
//! }
//! ```
//!
//! ## Performance Assertions
//!
//! ```ignore
//! #[flux::verify(expr = "(raw - overhead) < 50000000", severity = "critical")]
//! struct NetTimeCheck;
//! ```

// Re-export core types
pub use fluxbench_core::{
    Bencher, BenchmarkDef, BenchmarkResult, ChartDef, ChartType, GroupDef, IterationMode,
    ReportDef, Severity, current_allocation, reset_allocation_counter,
};

// Re-export macros
pub use fluxbench_macros::{bench, group, report, synthetic, verify};

// Re-export logic types
pub use fluxbench_logic::{
    MetricContext, Verification, VerificationResult, VerificationStatus, VerifyDef,
    SyntheticDef,
};

// Re-export stats
pub use fluxbench_stats::{
    BootstrapConfig, BootstrapResult, SummaryStatistics,
    compute_bootstrap, compute_summary,
};

/// Internal re-exports for macro use
#[doc(hidden)]
pub mod internal {
    pub use inventory;
    pub use tokio;
}

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{
        bench, group, report, synthetic, verify,
        Bencher, GroupDef, ReportDef, Severity,
    };
}

/// Attribute namespace for flux macros
pub mod flux {
    pub use fluxbench_macros::{bench, group, report, synthetic, verify};
}
