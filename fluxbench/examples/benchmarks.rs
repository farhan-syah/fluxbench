//! FluxBench Example Benchmarks
//!
//! This example demonstrates FluxBench features and serves as a template for
//! creating your own benchmark suite.
//!
//! Run with:
//!   cargo run --example benchmarks                      # Run all benchmarks (isolated)
//!   cargo run --example benchmarks -- --isolated=false  # Run in-process
//!   cargo run --example benchmarks -- --help            # Show all options
//!   cargo run --example benchmarks -- list              # List benchmarks
//!   cargo run --example benchmarks -- --group sorting   # Run only sorting group

use fluxbench::prelude::*;
use fluxbench::bench;
use std::hint::black_box;

// ============================================================================
// Basic Benchmarks
// ============================================================================

/// Simple arithmetic benchmark
#[bench]
fn bench_addition(b: &mut Bencher) {
    let x = 42u64;
    let y = 17u64;

    b.iter(|| black_box(black_box(x) + black_box(y)));
}

/// Vector sum benchmark
#[bench(group = "collections")]
fn bench_vector_sum(b: &mut Bencher) {
    let data: Vec<i64> = (0..1000).collect();

    b.iter(|| black_box(data.iter().sum::<i64>()));
}

/// Vector with larger dataset
#[bench(group = "collections", tags = "large")]
fn bench_vector_sum_large(b: &mut Bencher) {
    let data: Vec<i64> = (0..100_000).collect();

    b.iter(|| black_box(data.iter().sum::<i64>()));
}

// ============================================================================
// Benchmarks with Setup
// ============================================================================

/// Benchmark with per-iteration setup
#[bench(group = "setup")]
fn bench_with_setup(b: &mut Bencher) {
    b.iter_with_setup(
        || {
            // Setup: create fresh data each iteration
            vec![1u64, 2, 3, 4, 5]
        },
        |data| {
            // Measured: process the data
            black_box(data.iter().product::<u64>())
        },
    );
}

// ============================================================================
// HashMap Benchmarks
// ============================================================================

/// HashMap insertion benchmark
#[bench(group = "hashmap")]
fn bench_hashmap_insert(b: &mut Bencher) {
    use std::collections::HashMap;

    b.iter(|| {
        let mut map = HashMap::new();
        for i in 0..100 {
            map.insert(i, i * 2);
        }
        black_box(map)
    });
}

/// HashMap lookup benchmark
#[bench(group = "hashmap")]
fn bench_hashmap_lookup(b: &mut Bencher) {
    use std::collections::HashMap;

    let map: HashMap<i32, i32> = (0..1000).map(|i| (i, i * 2)).collect();

    b.iter(|| {
        let mut sum = 0;
        for i in 0..100 {
            if let Some(v) = map.get(&i) {
                sum += v;
            }
        }
        black_box(sum)
    });
}

// ============================================================================
// String Benchmarks
// ============================================================================

/// String concatenation benchmark
#[bench(group = "string")]
fn bench_string_concat(b: &mut Bencher) {
    b.iter(|| {
        let mut s = String::new();
        for i in 0..100 {
            s.push_str(&format!("{}", i));
        }
        black_box(s)
    });
}

/// String parsing benchmark
#[bench(group = "string")]
fn bench_string_parse(b: &mut Bencher) {
    let numbers: Vec<String> = (0..100).map(|i| i.to_string()).collect();

    b.iter(|| {
        let sum: i64 = numbers.iter().filter_map(|s| s.parse::<i64>().ok()).sum();
        black_box(sum)
    });
}

// ============================================================================
// Crash Isolation Test
// ============================================================================

/// Benchmark that panics - used to test crash isolation
#[bench(group = "crash_test")]
fn bench_will_panic(b: &mut Bencher) {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    b.iter(|| {
        let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if count >= 5 {
            panic!("Intentional panic for crash isolation test!");
        }
        black_box(count)
    });
}

/// Benchmark that runs after the crashing one - should still work in isolated mode
#[bench(group = "crash_test")]
fn bench_after_crash(b: &mut Bencher) {
    b.iter(|| {
        black_box(42 + 17)
    });
}

// ============================================================================
// Sorting Benchmarks
// ============================================================================

/// Sort small array
#[bench(group = "sorting")]
fn bench_sort_small(b: &mut Bencher) {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();

    b.iter(|| {
        let mut data: Vec<i32> = (0..100).map(|_| rng.gen()).collect();
        data.sort();
        black_box(data)
    });
}

/// Sort medium array
#[bench(group = "sorting", tags = "medium")]
fn bench_sort_medium(b: &mut Bencher) {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();

    b.iter(|| {
        let mut data: Vec<i32> = (0..10_000).map(|_| rng.gen()).collect();
        data.sort();
        black_box(data)
    });
}

// ============================================================================
// Memory Allocation Benchmarks
// ============================================================================

/// Benchmark that allocates memory
#[bench(group = "memory", tags = "alloc")]
fn bench_allocations(b: &mut Bencher) {
    b.iter(|| {
        let v1: Vec<u8> = vec![0; 1024];
        let v2: Vec<u8> = vec![1; 2048];
        let v3: Vec<u8> = vec![2; 512];
        black_box((v1, v2, v3))
    });
}

/// Benchmark with pre-allocated buffer (fewer allocations)
#[bench(group = "memory", tags = "prealloc")]
fn bench_preallocated(b: &mut Bencher) {
    let mut buffer = Vec::with_capacity(4096);

    b.iter(|| {
        buffer.clear();
        buffer.extend((0..1000).map(|i| i as u8));
        black_box(buffer.len())
    });
}

// ============================================================================
// Async Benchmarks
// ============================================================================

/// Basic async benchmark - sleep timer
#[bench(runtime = "multi_thread", worker_threads = 1, group = "async")]
async fn bench_async_sleep(b: &mut Bencher) {
    b.iter_async(|| async {
        tokio::time::sleep(std::time::Duration::from_micros(10)).await;
    });
}

/// Async benchmark with multi-threaded runtime
#[bench(runtime = "multi_thread", worker_threads = 2, group = "async")]
async fn bench_async_spawn(b: &mut Bencher) {
    b.iter_async(|| async {
        let handle = tokio::spawn(async {
            let mut sum = 0u64;
            for i in 0..100 {
                sum += i;
            }
            sum
        });
        handle.await.unwrap()
    });
}

/// Async channel benchmark
#[bench(runtime = "multi_thread", worker_threads = 2, group = "async")]
async fn bench_async_channel(b: &mut Bencher) {
    b.iter_async(|| async {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<u64>(1);
        tokio::spawn(async move {
            tx.send(42).await.ok();
        });
        rx.recv().await.unwrap()
    });
}

/// Async mutex contention benchmark
#[bench(runtime = "multi_thread", worker_threads = 4, group = "async")]
async fn bench_async_mutex(b: &mut Bencher) {
    use std::sync::Arc;
    let counter = Arc::new(tokio::sync::Mutex::new(0u64));

    b.iter_async(|| {
        let counter = counter.clone();
        async move {
            let mut guard = counter.lock().await;
            *guard += 1;
            *guard
        }
    });
}

/// Async file-like I/O simulation (using sleep as proxy)
#[bench(runtime = "multi_thread", worker_threads = 1, group = "async")]
async fn bench_async_io_simulation(b: &mut Bencher) {
    b.iter_async(|| async {
        // Simulate async I/O with small delay
        tokio::time::sleep(std::time::Duration::from_micros(1)).await;
        black_box(42)
    });
}

// ============================================================================
// Computation Benchmarks
// ============================================================================

/// Fibonacci (naive recursive) - expensive
#[bench(group = "compute", tags = "expensive")]
fn bench_fibonacci_naive(b: &mut Bencher) {
    fn fib(n: u32) -> u64 {
        if n <= 1 {
            n as u64
        } else {
            fib(n - 1) + fib(n - 2)
        }
    }

    b.iter(|| black_box(fib(20)));
}

/// Fibonacci (iterative) - fast
#[bench(group = "compute")]
fn bench_fibonacci_iter(b: &mut Bencher) {
    fn fib(n: u32) -> u64 {
        let mut a = 0u64;
        let mut b = 1u64;
        for _ in 0..n {
            let tmp = a;
            a = b;
            b = tmp + b;
        }
        a
    }

    b.iter(|| black_box(fib(20)));
}

// ============================================================================
// Main Entry Point - Uses FluxBench CLI
// ============================================================================

fn main() {
    // Use the standard FluxBench CLI infrastructure
    // All benchmarks defined above are automatically discovered via inventory
    if let Err(e) = fluxbench_cli::run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
