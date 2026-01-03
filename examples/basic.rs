//! Basic FluxBench Example
//!
//! Run with: cargo run --example basic

use fluxbench::prelude::*;
use std::hint::black_box;

/// Simple synchronous benchmark
#[flux::bench]
fn vector_sum(b: &mut Bencher) {
    let data: Vec<i64> = (0..1000).collect();

    b.iter(|| {
        black_box(data.iter().sum::<i64>())
    });
}

/// Benchmark with custom group and tags
#[flux::bench(group = "collections", tags = "vec,sum")]
fn vector_sum_large(b: &mut Bencher) {
    let data: Vec<i64> = (0..100_000).collect();

    b.iter(|| {
        black_box(data.iter().sum::<i64>())
    });
}

/// Benchmark with custom iteration count
#[flux::bench(iterations = 1000)]
fn hashmap_insert(b: &mut Bencher) {
    use std::collections::HashMap;

    b.iter(|| {
        let mut map = HashMap::new();
        for i in 0..100 {
            map.insert(i, i * 2);
        }
        black_box(map)
    });
}

fn main() {
    // FluxBench discovers benchmarks via inventory
    // Run: cargo bench or flux run
    println!("Run benchmarks with: cargo bench");
}
