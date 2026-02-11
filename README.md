# FluxBench

Benchmarking framework for Rust with crash isolation, statistical rigor, and CI integration.

## Features

- **Process-Isolated Benchmarks**: Panicking benchmarks don't terminate the suite. Fail-late architecture with supervisor-worker IPC.
- **Algebraic Verification**: Performance assertions directly in code: `#[verify(expr = "bench_a < bench_b")]`
- **Synthetic Metrics**: Compute derived metrics from benchmark results: `#[synthetic(formula = "bench_a / bench_b")]`
- **Multi-Way Comparisons**: Generate comparison tables and series charts with `#[compare]`
- **Bootstrap Confidence Intervals**: BCa (bias-corrected and accelerated) resampling, not just percentiles
- **Zero-Copy IPC**: Efficient data transfer between processes using rkyv serialization
- **RDTSC Timing**: High-precision cycle counting on x86_64 with `std::time::Instant` fallback
- **Flexible Execution**: Process-isolated by default; in-process mode available for debugging
- **Configuration**: `flux.toml` file with CLI override support
- **Multiple Output Formats**: JSON, HTML, CSV, GitHub Actions summaries
- **CI Integration**: Exit code 1 on critical failures; severity levels for different assertion types
- **Async Support**: Benchmarks with tokio runtimes via `#[bench(runtime = "multi_thread")]`

## Quick Start

### 1. Add Dependency

```toml
[dev-dependencies]
fluxbench = "<latest-version>"
```

### 2. Configure Bench Target

```toml
# Cargo.toml
[[bench]]
name = "my_benchmarks"
harness = false
```

### 3. Write Benchmarks

Create `benches/my_benchmarks.rs`:

```rust
use fluxbench::prelude::*;
use std::hint::black_box;

#[bench]
fn addition(b: &mut Bencher) {
    b.iter(|| black_box(42) + black_box(17));
}

#[bench(group = "compute")]
fn fibonacci(b: &mut Bencher) {
    fn fib(n: u32) -> u64 {
        if n <= 1 { n as u64 } else { fib(n - 1) + fib(n - 2) }
    }
    b.iter(|| black_box(fib(20)));
}

fn main() {
    fluxbench::run().unwrap();
}
```

Benchmarks can also live in `examples/` (`cargo run --example name --release`). Both `benches/` and `examples/` are only compiled on demand and never included in your production binary.

### 4. Run Benchmarks

```bash
cargo bench
```

Or with specific CLI options:

```bash
cargo bench -- --group compute --warmup 5 --measurement 10
```

## Defining Benchmarks

### Basic Benchmark

```rust
#[bench]
fn my_benchmark(b: &mut Bencher) {
    b.iter(|| {
        // Code to benchmark
        expensive_operation()
    });
}
```

### With Setup

```rust
#[bench]
fn with_setup(b: &mut Bencher) {
    b.iter_with_setup(
        || vec![1, 2, 3, 4, 5],  // Setup
        |data| data.iter().sum::<i32>()  // Measured code
    );
}
```

### Grouping Benchmarks

```rust
#[bench(group = "sorting")]
fn sort_small(b: &mut Bencher) {
    let data: Vec<i32> = (0..100).collect();
    b.iter(|| {
        let mut v = data.clone();
        v.sort();
        v
    });
}

#[bench(group = "sorting")]
fn sort_large(b: &mut Bencher) {
    let data: Vec<i32> = (0..100000).collect();
    b.iter(|| {
        let mut v = data.clone();
        v.sort();
        v
    });
}
```

### Tagging for Filtering

```rust
#[bench(group = "io", tags = "network")]
fn http_request(b: &mut Bencher) {
    // ...
}

#[bench(group = "io", tags = "file")]
fn disk_write(b: &mut Bencher) {
    // ...
}
```

Then run with: `cargo bench -- --tag network` or `cargo bench -- --skip-tag network`

### Async Benchmarks

```rust
#[bench(runtime = "multi_thread", worker_threads = 4, group = "async")]
async fn async_operation(b: &mut Bencher) {
    b.iter_async(|| async {
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    });
}
```

Runtimes: `"multi_thread"` or `"current_thread"`

## Performance Assertions

### Verification Macros

Assert that benchmarks meet performance criteria:

```rust
use fluxbench::verify;

#[verify(
    expr = "fibonacci < 50000",  // Less than 50us
    severity = "critical"
)]
struct FibUnder50us;

#[verify(
    expr = "fibonacci_iter < fibonacci_naive",
    severity = "warning"
)]
struct IterFasterThanNaive;

#[verify(
    expr = "fibonacci_naive_p99 < 1000000",  // p99 latency
    severity = "info"
)]
struct P99Check;
```

**Severity Levels:**

- `critical`: Fails the benchmark suite (exit code 1)
- `warning`: Reported but doesn't fail
- `info`: Informational only

**Available Metrics** (for benchmark name `bench_name`):

- `bench_name` - Mean time (nanoseconds)
- `bench_name_min` - Minimum time
- `bench_name_max` - Maximum time
- `bench_name_p50` - Median
- `bench_name_p95` - 95th percentile
- `bench_name_p99` - 99th percentile

### Synthetic Metrics

Compute derived metrics:

```rust
use fluxbench::synthetic;

#[synthetic(
    id = "speedup",
    formula = "fibonacci_naive / fibonacci_iter",
    unit = "x"
)]
struct FibSpeedup;

#[verify(expr = "speedup > 100", severity = "critical")]
struct SpeedupSignificant;
```

The formula supports:

- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparison: `<`, `>`, `<=`, `>=`, `==`, `!=`
- Logical: `&&`, `||`
- Parentheses for grouping

## Comparisons

### Simple Comparison

```rust
use fluxbench::compare;

#[compare(
    id = "string_ops",
    title = "String Operations",
    benchmarks = ["bench_string_concat", "bench_string_parse"],
    baseline = "bench_string_concat",
    metric = "mean"
)]
struct StringComparison;
```

Generates a table showing speedup vs baseline for each benchmark.

### Series Comparison

Create multi-point comparisons for scaling studies:

```rust
#[bench(group = "scaling")]
fn vec_sum_100(b: &mut Bencher) {
    let data: Vec<i64> = (0..100).collect();
    b.iter(|| data.iter().sum::<i64>());
}

#[bench(group = "scaling")]
fn vec_sum_1000(b: &mut Bencher) {
    let data: Vec<i64> = (0..1000).collect();
    b.iter(|| data.iter().sum::<i64>());
}

#[compare(
    id = "scale_100",
    title = "Vector Sum Scaling",
    benchmarks = ["bench_vec_sum_100"],
    group = "vec_scaling",
    x = "100"
)]
struct Scale100;

#[compare(
    id = "scale_1000",
    title = "Vector Sum Scaling",
    benchmarks = ["bench_vec_sum_1000"],
    group = "vec_scaling",
    x = "1000"
)]
struct Scale1000;
```

Multiple `#[compare]` with the same `group` combine into one chart.

## CLI Usage

Run benchmarks with options:

```bash
cargo bench -- [OPTIONS] [FILTER]
```

### Common Options

```bash
# List benchmarks without running
cargo bench -- list

# Run specific benchmark by regex
cargo bench -- bench_fib

# Run only a group
cargo bench -- --group sorting

# Filter by tag
cargo bench -- --tag expensive
cargo bench -- --skip-tag network

# Control execution
cargo bench -- --warmup 10 --measurement 20    # Seconds
cargo bench -- --min-iterations 100
cargo bench -- --isolated=false                # In-process mode
cargo bench -- --worker-timeout 120            # Worker timeout in seconds

# Output formats
cargo bench -- --format json --output results.json
cargo bench -- --format html --output results.html
cargo bench -- --format csv --output results.csv
cargo bench -- --format github-summary         # GitHub Actions summary

# Baseline comparison
cargo bench -- --baseline previous_results.json

# Dry run
cargo bench -- --dry-run
```

### Full Option Reference

- `--filter <PATTERN>` - Regex to match benchmark names
- `--group <GROUP>` - Run only benchmarks in this group
- `--tag <TAG>` - Include only benchmarks with this tag
- `--skip-tag <TAG>` - Exclude benchmarks with this tag
- `--warmup <SECONDS>` - Warmup duration before measurement (default: 3)
- `--measurement <SECONDS>` - Measurement duration (default: 5)
- `--min-iterations <N>` - Minimum iterations
- `--max-iterations <N>` - Maximum iterations
- `--isolated <BOOL>` - Run in separate processes (default: true)
- `--one-shot` - Fresh worker process per benchmark (default: reuse workers)
- `--worker-timeout <SECONDS>` - Worker process timeout (default: 60)
- `--threads <N>` / `-j <N>` - Threads for parallel statistics computation (default: 0 = all cores)
- `--format <FORMAT>` - Output format: json, html, csv, github-summary, human (default: human)
- `--output <FILE>` - Output file (default: stdout)
- `--baseline <FILE>` - Load baseline for comparison
- `--threshold <PCT>` - Regression threshold percentage
- `--verbose` / `-v` - Enable debug logging
- `--dry-run` - List benchmarks without executing

## Configuration

Create a `flux.toml` in your project root:

```toml
[runner]
warmup_time = "3s"
measurement_time = "5s"
timeout = "60s"
isolation = "process"
# min_iterations = 100
# max_iterations = 1000000
bootstrap_iterations = 10000
confidence_level = 0.95

[allocator]
track = true
fail_on_allocation = false
# max_bytes_per_iter = 1024

[output]
format = "human"
directory = "target/fluxbench"
save_baseline = false
# baseline_path = "baseline.json"

[visuals]
theme = "light"
width = 1280
height = 720

[ci]
regression_threshold = 5.0
github_annotations = false
fail_on_critical = true
```

CLI options override `flux.toml` settings.

## Output Formats

### Human (Default)

Console output with grouped results and statistics:

```
Group: compute
------------------------------------------------------------
  ✓ bench_fibonacci_iter
      mean: 127.42 ns  median: 127.00 ns  stddev: 0.77 ns
      min: 126.00 ns  max: 147.00 ns  samples: 60
      p50: 127.00 ns  p95: 129.00 ns  p99: 136.38 ns
      95% CI: [127.35, 129.12] ns
      throughput: 7847831.87 ops/sec
      cycles: mean 603  median 601  (4.72 GHz)
```

### JSON

Machine-readable format with full metadata:

```json
{
  "meta": {
    "version": "0.1.0",
    "timestamp": "2026-02-10T...",
    "git_commit": "abc123...",
    "system": { "os": "linux", "cpu": "...", "cpu_cores": 24 }
  },
  "results": [
    {
      "id": "bench_fibonacci_iter",
      "group": "compute",
      "status": "passed",
      "metrics": {
        "mean_ns": 127.42,
        "median_ns": 127.0,
        "std_dev_ns": 0.77,
        "p50_ns": 127.0,
        "p95_ns": 129.0,
        "p99_ns": 136.38
      }
    }
  ],
  "verifications": [...],
  "synthetics": [...]
}
```

### CSV

Spreadsheet-compatible format with all metrics:

```
id,name,group,status,mean_ns,median_ns,std_dev_ns,min_ns,max_ns,p50_ns,p95_ns,p99_ns,samples,alloc_bytes,alloc_count,mean_cycles,median_cycles,cycles_per_ns
bench_fibonacci_iter,bench_fibonacci_iter,compute,passed,127.42,...
```

### HTML

Self-contained interactive report with charts and tables.

### GitHub Summary

Renders verification results in GitHub Actions workflow:

```bash
cargo bench -- --format github-summary >> $GITHUB_STEP_SUMMARY
```

## Crash Isolation

Panicking benchmarks don't terminate the suite:

```rust
#[bench]
fn may_panic(b: &mut Bencher) {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    b.iter(|| {
        let count = COUNTER.fetch_add(1, SeqCst);
        if count >= 5 {
            panic!("Intentional panic!");  // Isolated; suite continues
        }
    });
}
```

With `--isolated=true` (default), the panic occurs in a worker process and is reported as a failure for that benchmark, not the suite.

## Advanced Usage

### Allocation Tracking

FluxBench can track heap allocations per benchmark iteration. To enable this, install the
`TrackingAllocator` as the global allocator in your benchmark binary:

```rust
use fluxbench::prelude::*;
use fluxbench::TrackingAllocator;

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

#[bench]
fn vec_allocation(b: &mut Bencher) {
    b.iter(|| vec![1, 2, 3, 4, 5]);
}

fn main() { fluxbench::run(); }
```

Results will include allocation metrics for each benchmark:
- **alloc_bytes** — total bytes allocated per iteration
- **alloc_count** — number of allocations per iteration

These appear in JSON, CSV, and human output automatically.

> **Note:** `#[global_allocator]` must be declared in the binary crate (your `benches/*.rs` file),
> not in a library. Rust allows only one global allocator per binary. Without it, `track = true`
> in `flux.toml` will report zero allocations.

You can also query allocation counters manually:

```rust
fluxbench::reset_allocation_counter();
// ... run some code ...
let (alloc_bytes, alloc_count) = fluxbench::current_allocation();
println!("Bytes: {}, Count: {}", alloc_bytes, alloc_count);
```

### In-Process Mode

For debugging, run benchmarks in the same process:

```bash
cargo bench -- --isolated=false
```

Panics will crash immediately, so use this only for development.

### Custom Bootstrap Configuration

Via `flux.toml`:

```toml
[runner]
bootstrap_iterations = 100000
confidence_level = 0.99
```

Higher iterations = more precise intervals, slower reporting.

## Project Structure

The fluxbench workspace consists of:

- **fluxbench** - Meta-crate, public API
- **fluxbench-cli** - Supervisor process and CLI
- **fluxbench-core** - Bencher, timer, worker, allocator
- **fluxbench-ipc** - Zero-copy IPC transport with rkyv
- **fluxbench-stats** - Bootstrap resampling and percentile computation
- **fluxbench-logic** - Verification, synthetic metrics, dependency graphs
- **fluxbench-macros** - Procedural macros for bench, verify, synthetic, compare
- **fluxbench-report** - JSON, HTML, CSV, GitHub output generation

## Examples

See `fluxbench/examples/benchmarks.rs` for a comprehensive example:

```bash
cargo run --example benchmarks -- list
cargo run --example benchmarks -- --group sorting
cargo run --example benchmarks -- --format json --output results.json
```

## License

Licensed under the Apache License, Version 2.0. See LICENSE for details.

## Contributing

Contributions welcome. Please ensure benchmarks remain crash-isolated and statistical integrity is maintained.
