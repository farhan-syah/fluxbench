//! Algebraic Verification Example
//!
//! Demonstrates performance assertions using verify macros.
//! Run with: cargo run --example verify

use fluxbench::prelude::*;
use std::hint::black_box;

/// Benchmark that we'll verify
#[flux::bench(id = "fast_operation")]
fn fast_operation(b: &mut Bencher) {
    b.iter(|| {
        let mut sum = 0u64;
        for i in 0..100 {
            sum = sum.wrapping_add(i);
        }
        black_box(sum)
    });
}

/// Benchmark with heavier computation
#[flux::bench(id = "heavy_operation")]
fn heavy_operation(b: &mut Bencher) {
    b.iter(|| {
        let mut sum = 0u64;
        for i in 0..10_000 {
            sum = sum.wrapping_add(i);
        }
        black_box(sum)
    });
}

/// Verify that fast_operation is under 1 microsecond (1000 ns)
/// Uses mean metric from the benchmark result
#[flux::verify(
    bench = "fast_operation",
    expr = "mean < 1000",
    severity = "warning"
)]
struct FastOperationCheck;

/// Verify that heavy operation doesn't regress too much
/// The expression can reference multiple metrics
#[flux::verify(
    bench = "heavy_operation",
    expr = "p99 < 100000",  // P99 under 100 microseconds
    severity = "critical"
)]
struct HeavyOperationLatency;

/// Verify memory efficiency - no allocations in fast path
#[flux::verify(
    bench = "fast_operation",
    expr = "alloc_bytes == 0",
    severity = "critical"
)]
struct NoAllocations;

/// Comparative verification - fast should be 10x faster than heavy
#[flux::verify(
    expr = "fast_operation.mean * 10 < heavy_operation.mean",
    severity = "info"
)]
struct PerformanceRatio;

fn main() {
    println!("Verification Examples");
    println!("=====================");
    println!();
    println!("Available metrics in expressions:");
    println!("  mean       - Mean duration in nanoseconds");
    println!("  median     - Median duration");
    println!("  min        - Minimum duration");
    println!("  max        - Maximum duration");
    println!("  stddev     - Standard deviation");
    println!("  p50, p90, p95, p99, p999 - Percentiles");
    println!("  alloc_bytes  - Total bytes allocated");
    println!("  alloc_count  - Number of allocations");
    println!("  raw        - Raw duration (before overhead subtraction)");
    println!("  overhead   - Measured overhead");
    println!();
    println!("Severity levels:");
    println!("  info     - Informational, no CI impact");
    println!("  warning  - Warning, may indicate regression");
    println!("  critical - Critical, fails CI if violated");
    println!();
    println!("Run: cargo bench to execute verifications");
}
