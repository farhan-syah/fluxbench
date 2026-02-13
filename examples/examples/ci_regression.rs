//! CI Regression Pipeline Example
//!
//! Shows how to use FluxBench in a CI pipeline with baselines and thresholds.
//!
//! Workflow:
//!   # On main — save a baseline after merging:
//!   cargo run --example ci_regression -p fluxbench --release -- --save-baseline main
//!
//!   # On a PR branch — compare against the saved baseline:
//!   cargo run --example ci_regression -p fluxbench --release -- --baseline main
//!
//! Severity levels control CI exit codes:
//!   critical  — exit 1 (blocks merge)
//!   warning   — exit 0 but prints warnings
//!   info      — logged only
//!
//! Run with: cargo run --example ci_regression -p fluxbench --release

use fluxbench::prelude::*;
use fluxbench::{bench, compare, synthetic, verify};
use std::hint::black_box;

// ============================================================================
// Hot-path benchmarks — these guard critical performance
// ============================================================================

/// Simulated request handler: parse input, look up cache, format response.
#[bench(
    id = "request_handler",
    group = "hot_path",
    severity = "critical",
    threshold = 5.0,
    warmup = "2s",
    measurement = "3s",
    tags = "latency"
)]
fn request_handler(b: &mut Bencher) {
    let cache: std::collections::HashMap<u32, String> =
        (0..100).map(|i| (i, format!("value_{i}"))).collect();
    let input = "42";
    b.iter(|| {
        let key: u32 = input.parse().unwrap();
        let val = cache.get(&key).map(|s| s.as_str()).unwrap_or("miss");
        let response = format!("{{\"result\":\"{val}\"}}");
        black_box(response)
    });
}

/// Token scanning — lexer inner loop.
#[bench(
    id = "token_scan",
    group = "hot_path",
    severity = "critical",
    threshold = 3.0,
    warmup = "2s",
    measurement = "3s",
    tags = "throughput"
)]
fn token_scan(b: &mut Bencher) {
    let source = "fn main() { let x = 42 + y * (z - 1); println!(\"hello\"); }".repeat(100);
    b.iter(|| {
        let tokens: usize = source
            .split(|c: char| c.is_whitespace() || c.is_ascii_punctuation())
            .filter(|s| !s.is_empty())
            .count();
        black_box(tokens)
    });
}

/// Batch processing — bulk data transform.
#[bench(
    id = "batch_transform",
    group = "hot_path",
    severity = "warning",
    threshold = 10.0,
    warmup = "2s",
    measurement = "3s",
    tags = "throughput"
)]
fn batch_transform(b: &mut Bencher) {
    let data: Vec<f64> = (0..10_000).map(|i| i as f64 * 0.001).collect();
    b.iter(|| {
        let result: Vec<f64> = data.iter().map(|x| (x * 2.0 + 1.0).sqrt()).collect();
        black_box(result.len())
    });
}

// ============================================================================
// Cold-path benchmarks — informational, not blocking
// ============================================================================

/// Config file parsing (runs once at startup, not latency-critical).
#[bench(id = "config_parse", group = "cold_path", severity = "info")]
fn config_parse(b: &mut Bencher) {
    let config_text: String = (0..50)
        .map(|i| format!("setting_{i} = {}\n", i * 7))
        .collect();
    b.iter(|| {
        let entries: Vec<(&str, &str)> = config_text
            .lines()
            .filter_map(|line| line.split_once(" = "))
            .collect();
        black_box(entries.len())
    });
}

// ============================================================================
// Comparisons
// ============================================================================

#[compare(
    id = "hot_path_cmp",
    title = "Hot Path Latency",
    benchmarks = ["request_handler", "token_scan", "batch_transform"],
    baseline = "request_handler",
    metric = "mean"
)]
#[allow(dead_code)]
struct HotPathComparison;

// ============================================================================
// Synthetic metrics
// ============================================================================

/// Requests per second (single-threaded).
#[synthetic(id = "rps", formula = "1000000000 / request_handler", unit = "req/s")]
#[allow(dead_code)]
struct RequestsPerSec;

/// Token scanning throughput.
#[synthetic(
    id = "tokens_per_sec",
    formula = "1000000000 / token_scan",
    unit = "scans/s"
)]
#[allow(dead_code)]
struct TokensPerSec;

// ============================================================================
// Regression gates — these control CI pass/fail
// ============================================================================

/// Request handler must stay under 5 us (5000 ns).
#[verify(expr = "request_handler < 5000", severity = "critical")]
#[allow(dead_code)]
struct RequestLatencyBudget;

/// Token scanner must stay under 100 us.
#[verify(expr = "token_scan < 100000", severity = "critical")]
#[allow(dead_code)]
struct TokenScanBudget;

/// Batch transform must stay under 500 us.
#[verify(expr = "batch_transform < 500000", severity = "warning")]
#[allow(dead_code)]
struct BatchTransformBudget;

/// Request throughput must exceed 200k req/s.
#[verify(expr = "rps > 200000", severity = "warning")]
#[allow(dead_code)]
struct ThroughputFloor;

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
