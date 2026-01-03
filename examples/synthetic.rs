//! Synthetic Metrics Example
//!
//! Demonstrates computed metrics derived from benchmark results.
//! Run with: cargo run --example synthetic

use fluxbench::prelude::*;
use std::hint::black_box;

/// Benchmark throughput test - processing items
#[flux::bench(id = "process_items")]
fn process_items(b: &mut Bencher) {
    let items: Vec<u64> = (0..1000).collect();

    b.iter(|| {
        let result: u64 = items.iter().map(|x| x.wrapping_mul(2)).sum();
        black_box(result)
    });
}

/// Benchmark memory bandwidth test
#[flux::bench(id = "memory_copy")]
fn memory_copy(b: &mut Bencher) {
    let src: Vec<u8> = vec![0u8; 1024 * 1024]; // 1 MB
    let mut dst: Vec<u8> = vec![0u8; 1024 * 1024];

    b.iter(|| {
        dst.copy_from_slice(&src);
        black_box(&dst)
    });
}

/// Synthetic: Compute throughput in items per second
#[flux::synthetic(
    id = "items_per_second",
    deps = "process_items",
    expr = "1_000_000_000 / process_items.mean * 1000"  // 1000 items per iteration
)]
struct ItemsPerSecond;

/// Synthetic: Compute memory bandwidth in GB/s
#[flux::synthetic(
    id = "memory_bandwidth_gbps",
    deps = "memory_copy",
    expr = "1_048_576 / memory_copy.mean"  // 1MB / time_ns = GB/s (simplified)
)]
struct MemoryBandwidth;

/// Synthetic: Cost per item in nanoseconds
#[flux::synthetic(
    id = "ns_per_item",
    deps = "process_items",
    expr = "process_items.mean / 1000"  // divide by item count
)]
struct NsPerItem;

/// Verify synthetic metric meets threshold
#[flux::verify(
    bench = "items_per_second",
    expr = "value > 1_000_000",  // At least 1M items/sec
    severity = "warning"
)]
struct ThroughputCheck;

fn main() {
    println!("Synthetic Metrics Examples");
    println!("==========================");
    println!();
    println!("Synthetic metrics are computed from benchmark results:");
    println!();
    println!("  #[flux::synthetic(");
    println!("      id = \"throughput\",");
    println!("      deps = \"my_bench\",");
    println!("      expr = \"1_000_000_000 / my_bench.mean * items_per_iter\"");
    println!("  )]");
    println!("  struct Throughput;");
    println!();
    println!("Use cases:");
    println!("  - Throughput (ops/sec, items/sec)");
    println!("  - Bandwidth (GB/s, MB/s)");
    println!("  - Cost metrics (ns/op, cycles/byte)");
    println!("  - Efficiency ratios");
    println!("  - Normalized comparisons");
    println!();
    println!("Run: cargo bench to compute synthetics");
}
