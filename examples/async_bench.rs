//! Async Benchmark Example
//!
//! Demonstrates async benchmarks with runtime configuration.
//! Run with: cargo run --example async_bench

use fluxbench::prelude::*;
use std::hint::black_box;
use std::time::Duration;

/// Single-threaded async benchmark (default)
#[flux::bench]
async fn async_sleep(b: &mut Bencher) {
    b.iter(|| async {
        tokio::time::sleep(Duration::from_micros(100)).await;
    });
}

/// Multi-threaded async benchmark with 4 worker threads
#[flux::bench(runtime = "multi_thread", worker_threads = 4)]
async fn concurrent_tasks(b: &mut Bencher) {
    b.iter(|| async {
        let handles: Vec<_> = (0..4)
            .map(|i| {
                tokio::spawn(async move {
                    tokio::time::sleep(Duration::from_micros(10)).await;
                    i * 2
                })
            })
            .collect();

        let results: Vec<_> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        black_box(results)
    });
}

/// Async I/O benchmark
#[flux::bench(runtime = "current_thread", enable_io = true)]
async fn async_file_read(b: &mut Bencher) {
    // Create a temp file for benchmarking
    let temp_path = "/tmp/fluxbench_test.txt";
    tokio::fs::write(temp_path, "Hello, FluxBench!").await.ok();

    b.iter(|| async {
        let content = tokio::fs::read_to_string(temp_path).await.unwrap();
        black_box(content)
    });

    // Cleanup
    tokio::fs::remove_file(temp_path).await.ok();
}

fn main() {
    println!("Run async benchmarks with: cargo bench");
    println!();
    println!("Async runtime options:");
    println!("  runtime = \"current_thread\" - Single-threaded (default)");
    println!("  runtime = \"multi_thread\"   - Multi-threaded with Tokio");
    println!("  worker_threads = N          - Number of worker threads");
    println!("  enable_time = true/false    - Enable Tokio time driver");
    println!("  enable_io = true/false      - Enable Tokio I/O driver");
}
