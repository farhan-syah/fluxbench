//! Parameterized Benchmarks — Scaling tests with `args`
//!
//! Use `args = [...]` to run the same benchmark function at multiple input
//! sizes. FluxBench generates one benchmark per arg value, named `id@value`.
//! Combine with groups and tags for filtering.
//!
//! Run with: cargo run --example feature_params -p fluxbench --release

use fluxbench::bench;
use fluxbench::prelude::*;
use std::collections::HashMap;
use std::hint::black_box;

// ---------------------------------------------------------------------------
// Scaling: O(n) linear scan
// ---------------------------------------------------------------------------

/// Linear search through an unsorted vector.
/// Generates: linear_scan@100, linear_scan@1000, linear_scan@10000, linear_scan@100000
#[bench(group = "scaling", args = [100, 1000, 10000, 100000])]
fn linear_scan(b: &mut Bencher, n: u32) {
    let data: Vec<u64> = (0..n as u64).collect();
    let target = n as u64 - 1; // worst case: last element
    b.iter(|| black_box(data.iter().position(|&x| x == target)));
}

// ---------------------------------------------------------------------------
// Scaling: O(n log n) sort
// ---------------------------------------------------------------------------

/// Sort a pre-shuffled vector of `n` elements.
#[bench(group = "scaling", args = [100, 1000, 10000, 100000])]
fn sort_scaling(b: &mut Bencher, n: u32) {
    use rand::prelude::*;
    let mut rng = rand::rng();
    b.iter_with_setup(
        || {
            let mut v: Vec<u32> = (0..n).collect();
            v.shuffle(&mut rng);
            v
        },
        |mut v| {
            v.sort_unstable();
            black_box(v)
        },
    );
}

// ---------------------------------------------------------------------------
// Scaling: HashMap operations
// ---------------------------------------------------------------------------

/// Insert `n` entries into a HashMap (no pre-allocation).
#[bench(group = "hashmap", args = [100, 1000, 10000])]
fn hashmap_insert(b: &mut Bencher, n: u32) {
    b.iter(|| {
        let mut map = HashMap::new();
        for i in 0..n {
            map.insert(i, i.wrapping_mul(2654435761));
        }
        black_box(map.len())
    });
}

/// Lookup `n` keys in a pre-built HashMap of size `n`.
#[bench(group = "hashmap", args = [100, 1000, 10000])]
fn hashmap_lookup(b: &mut Bencher, n: u32) {
    let map: HashMap<u32, u32> = (0..n).map(|i| (i, i * 3)).collect();
    b.iter(|| {
        let mut sum = 0u32;
        for i in 0..n {
            sum = sum.wrapping_add(*map.get(&i).unwrap());
        }
        black_box(sum)
    });
}

// ---------------------------------------------------------------------------
// Scaling: String building
// ---------------------------------------------------------------------------

/// Build a string by pushing `n` formatted integers.
#[bench(group = "strings", tags = "alloc", args = [100, 1000, 10000])]
fn string_build(b: &mut Bencher, n: u32) {
    b.iter(|| {
        let mut s = String::with_capacity(n as usize * 4);
        for i in 0..n {
            use std::fmt::Write;
            write!(s, "{i},").unwrap();
        }
        black_box(s.len())
    });
}

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
