//! Iteration Methods — All ways to drive a benchmark loop
//!
//! FluxBench offers three iteration strategies. Choose based on whether your
//! workload needs fresh input, amortized setup, or is purely computational.
//!
//! Run with: cargo run --example feature_iteration -p fluxbench --release

use fluxbench::bench;
use fluxbench::prelude::*;
use std::collections::HashMap;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// b.iter() — Simple closure, measured end-to-end
// ---------------------------------------------------------------------------
// Best for: pure computation where the input can be prepared once and reused.

/// Summing a pre-built vector — no per-iteration allocation.
#[bench(group = "iter")]
fn iter_vector_sum(b: &mut Bencher) {
    let data: Vec<i64> = (0..10_000).collect();
    b.iter(|| black_box(data.iter().sum::<i64>()));
}

/// Sorting inside iter() means we re-randomize outside the loop,
/// but the *same* shuffled vec is sorted every iteration — after the
/// first, it's already sorted. This is a common mistake; see
/// `setup_sort_fresh` below for the correct approach.
#[bench(group = "iter")]
fn iter_sort_already_sorted(b: &mut Bencher) {
    let mut data: Vec<u32> = (0..1_000).rev().collect();
    b.iter(|| {
        data.sort();
        black_box(data.len())
    });
}

// ---------------------------------------------------------------------------
// b.iter_with_setup() — Per-iteration setup (not measured)
// ---------------------------------------------------------------------------
// Best for: benchmarks that consume or mutate their input. The setup closure
// runs before each iteration but its time is excluded from the measurement.

/// Each iteration gets a freshly-shuffled vector, so we always measure
/// worst-case sort — not the already-sorted fast path.
#[bench(group = "setup")]
fn setup_sort_fresh(b: &mut Bencher) {
    use rand::prelude::*;
    b.iter_with_setup(
        || {
            let mut rng = rand::rng();
            let mut v: Vec<u32> = (0..1_000).collect();
            v.shuffle(&mut rng);
            v
        },
        |mut data| {
            data.sort();
            black_box(data)
        },
    );
}

/// HashMap benchmarks often need a fresh map per iteration because
/// insertion order affects bucket layout and performance.
#[bench(group = "setup")]
fn setup_hashmap_drain(b: &mut Bencher) {
    b.iter_with_setup(
        || {
            let mut map = HashMap::with_capacity(256);
            for i in 0u64..256 {
                map.insert(i, i.wrapping_mul(0x517cc1b727220a95));
            }
            map
        },
        |mut map| {
            // Measure drain — consumes the map
            let sum: u64 = map.drain().map(|(_, v)| v).sum();
            black_box(sum)
        },
    );
}

// ---------------------------------------------------------------------------
// b.iter_batched() — Amortized setup over N iterations
// ---------------------------------------------------------------------------
// Best for: when setup is expensive relative to the measured work. The setup
// runs once per *batch* of `batch_size` iterations, reducing overhead.

/// Allocating a large buffer is expensive; amortize it over 64 iterations.
#[bench(group = "batched")]
fn batched_buffer_reuse(b: &mut Bencher) {
    b.iter_batched(
        64,
        || vec![0u8; 64 * 1024], // 64 KiB buffer, allocated once per batch
        |buf| {
            // Simulate a hot-path read over the shared buffer
            let sum: u8 = buf.iter().fold(0u8, |a, &x| a.wrapping_add(x));
            black_box(sum)
        },
    );
}

/// Regex compilation is slow; compile once, match 128 times per batch.
#[bench(group = "batched")]
fn batched_regex_match(b: &mut Bencher) {
    let pattern = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$";
    let inputs = [
        "user@example.com",
        "bad@@email",
        "hello.world+tag@sub.domain.org",
        "nope",
    ];
    b.iter_batched(
        128,
        || regex::Regex::new(pattern).unwrap(),
        |re| {
            let hits: usize = inputs.iter().filter(|s| re.is_match(s)).count();
            black_box(hits)
        },
    );
}

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
