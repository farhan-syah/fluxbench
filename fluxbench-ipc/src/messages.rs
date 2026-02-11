//! IPC Message Types
//!
//! All messages are serialized with rkyv for zero-copy deserialization.

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

/// A single measurement sample (32 bytes, cache-aligned).
///
/// Compact representation optimized for batching and cache efficiency.
/// All timing values are in nanoseconds.
///
/// # Layout history
/// - Protocol v1: 24 bytes (`cpu_cycles` was `u32`).
/// - Protocol v2+: 32 bytes (`cpu_cycles` widened to `u64` to avoid overflow beyond ~1 s at 4 GHz).
#[derive(Debug, Clone, Copy, Archive, RkyvSerialize, RkyvDeserialize)]
#[repr(C, align(8))]
pub struct Sample {
    /// Duration of this iteration in nanoseconds
    pub duration_nanos: u64,
    /// Bytes allocated during this iteration
    pub alloc_bytes: u64,
    /// Number of allocations during this iteration
    pub alloc_count: u32,
    /// CPU cycles (from RDTSC). Full u64 avoids overflow for benchmarks >1s at 4GHz.
    pub cpu_cycles: u64,
}

impl Sample {
    /// Create a new sample with the given measurements
    #[inline]
    pub fn new(duration_nanos: u64, alloc_bytes: u64, alloc_count: u32, cpu_cycles: u64) -> Self {
        Self {
            duration_nanos,
            alloc_bytes,
            alloc_count,
            cpu_cycles,
        }
    }

    /// Create a timing-only sample (no allocation tracking)
    #[inline]
    pub fn timing_only(duration_nanos: u64) -> Self {
        Self {
            duration_nanos,
            alloc_bytes: 0,
            alloc_count: 0,
            cpu_cycles: 0,
        }
    }
}

/// Reason for flushing a sample batch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, RkyvSerialize, RkyvDeserialize)]
pub enum FlushReason {
    /// Batch size limit reached (10K samples)
    BatchFull,
    /// Byte size limit reached (64KB)
    ByteLimitReached,
    /// Heartbeat timeout (100ms with no activity)
    HeartbeatTimeout,
    /// Benchmark completed normally
    BenchmarkComplete,
    /// Worker shutting down
    Shutdown,
}

/// A batch of samples sent from worker to supervisor
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct SampleBatch {
    /// Hash of the benchmark ID (for fast lookup without string comparison)
    pub bench_id_hash: u64,
    /// Sequence number for ordering and detecting dropped batches
    pub batch_sequence: u32,
    /// First iteration index in this batch
    pub start_iteration: u64,
    /// The actual samples
    pub samples: Vec<Sample>,
    /// Why this batch was flushed
    pub flush_reason: FlushReason,
}

impl SampleBatch {
    /// Estimated size in bytes for this batch
    pub fn estimated_size(&self) -> usize {
        // Header fields + samples
        std::mem::size_of::<u64>() * 2  // bench_id_hash, start_iteration
            + std::mem::size_of::<u32>()  // batch_sequence
            + std::mem::size_of::<FlushReason>()
            + self.samples.len() * std::mem::size_of::<Sample>()
    }
}

/// Worker capabilities advertised during handshake
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct WorkerCapabilities {
    /// Protocol version for compatibility
    pub protocol_version: u32,
    /// Whether RDTSC is available for high-precision timing
    pub has_rdtsc: bool,
    /// Whether allocation tracking is enabled
    pub has_alloc_tracking: bool,
    /// Whether CPU performance counters are available
    pub has_perf_counters: bool,
    /// Number of logical CPUs available
    pub cpu_count: u32,
    /// CPU model string (for reports)
    pub cpu_model: String,
}

impl Default for WorkerCapabilities {
    fn default() -> Self {
        Self {
            protocol_version: crate::PROTOCOL_VERSION,
            has_rdtsc: cfg!(target_arch = "x86_64"),
            has_alloc_tracking: true,
            has_perf_counters: false,
            cpu_count: num_cpus(),
            cpu_model: cpu_model_string(),
        }
    }
}

/// Messages sent from Worker to Supervisor
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub enum WorkerMessage {
    /// Initial handshake with worker capabilities
    Hello(WorkerCapabilities),

    /// A batch of measurement samples
    SampleBatch(SampleBatch),

    /// Warmup phase completed, ready for measurement
    WarmupComplete {
        /// Number of warmup iterations performed
        iterations: u64,
        /// Total warmup time in nanoseconds
        duration_nanos: u64,
    },

    /// Benchmark completed successfully
    Complete {
        /// Total number of iterations
        total_iterations: u64,
        /// Total measurement time in nanoseconds
        total_duration_nanos: u64,
    },

    /// Benchmark failed with an error
    Failure {
        /// Error category
        kind: FailureKind,
        /// Human-readable error message
        message: String,
        /// Optional backtrace
        backtrace: Option<String>,
    },

    /// Progress update for long-running benchmarks
    Progress {
        /// Iterations completed so far
        completed: u64,
        /// Estimated total iterations
        estimated_total: u64,
    },
}

/// Categories of benchmark failures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, RkyvSerialize, RkyvDeserialize)]
pub enum FailureKind {
    /// Rust panic (caught)
    Panic,
    /// Timeout exceeded
    Timeout,
    /// Assertion failure in verification
    Assertion,
    /// Allocation limit exceeded
    AllocationLimit,
    /// Signal received (SIGSEGV, SIGBUS, etc.)
    Signal,
    /// Unknown error
    Unknown,
}

/// Commands sent from Supervisor to Worker
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub enum SupervisorCommand {
    /// Run a specific benchmark
    Run {
        /// Benchmark identifier
        bench_id: String,
        /// Configuration for this run
        config: BenchmarkConfig,
    },

    /// Abort the current benchmark
    Abort,

    /// Request graceful shutdown
    Shutdown,

    /// Ping for health check
    Ping,
}

/// Configuration for a benchmark run
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct BenchmarkConfig {
    /// Warmup time in nanoseconds
    pub warmup_time_ns: u64,
    /// Measurement time in nanoseconds
    pub measurement_time_ns: u64,
    /// Minimum number of iterations (overrides time-based)
    pub min_iterations: Option<u64>,
    /// Maximum number of iterations (cap)
    pub max_iterations: Option<u64>,
    /// Whether to track allocations
    pub track_allocations: bool,
    /// Whether to fail on any allocation (for hot path testing)
    pub fail_on_allocation: bool,
    /// Timeout in nanoseconds (0 = no timeout)
    pub timeout_ns: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_time_ns: 3_000_000_000,      // 3 seconds
            measurement_time_ns: 5_000_000_000, // 5 seconds
            min_iterations: Some(100),
            max_iterations: None,
            track_allocations: true,
            fail_on_allocation: false,
            timeout_ns: 60_000_000_000, // 60 seconds
        }
    }
}

impl BenchmarkConfig {
    /// Validate configuration values, returning a description of the first error found.
    pub fn validate(&self) -> Result<(), String> {
        if self.warmup_time_ns == 0 {
            return Err("warmup_time_ns must be > 0".to_string());
        }
        if self.measurement_time_ns == 0 {
            return Err("measurement_time_ns must be > 0".to_string());
        }
        if self.timeout_ns > 0 && self.timeout_ns < self.measurement_time_ns {
            return Err(format!(
                "timeout_ns ({}) must be >= measurement_time_ns ({})",
                self.timeout_ns, self.measurement_time_ns
            ));
        }
        if let (Some(min), Some(max)) = (self.min_iterations, self.max_iterations) {
            if max < min {
                return Err(format!(
                    "max_iterations ({}) must be >= min_iterations ({})",
                    max, min
                ));
            }
        }
        Ok(())
    }
}

// Helper functions

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(1)
}

fn cpu_model_string() -> String {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|content| {
                content
                    .lines()
                    .find(|line| line.starts_with("model name"))
                    .and_then(|line| line.split(':').nth(1))
                    .map(|s| s.trim().to_string())
            })
            .unwrap_or_else(|| "Unknown CPU".to_string())
    }

    #[cfg(not(target_os = "linux"))]
    {
        "Unknown CPU".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_creation() {
        let sample = Sample::new(1000, 512, 5, 4000);
        assert_eq!(sample.duration_nanos, 1000);
        assert_eq!(sample.alloc_bytes, 512);
        assert_eq!(sample.alloc_count, 5);
        assert_eq!(sample.cpu_cycles, 4000);
    }

    #[test]
    fn test_sample_timing_only() {
        let sample = Sample::timing_only(5000);
        assert_eq!(sample.duration_nanos, 5000);
        assert_eq!(sample.alloc_bytes, 0);
        assert_eq!(sample.alloc_count, 0);
        assert_eq!(sample.cpu_cycles, 0);
    }

    #[test]
    fn test_benchmark_config_validate_default() {
        assert!(BenchmarkConfig::default().validate().is_ok());
    }

    #[test]
    fn test_benchmark_config_validate_zero_warmup() {
        let mut config = BenchmarkConfig::default();
        config.warmup_time_ns = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_benchmark_config_validate_timeout_lt_measurement() {
        let mut config = BenchmarkConfig::default();
        config.timeout_ns = 1_000_000; // 1ms
        config.measurement_time_ns = 5_000_000_000; // 5s
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_benchmark_config_validate_max_lt_min() {
        let mut config = BenchmarkConfig::default();
        config.min_iterations = Some(200);
        config.max_iterations = Some(50);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_worker_capabilities_default() {
        let caps = WorkerCapabilities::default();
        assert_eq!(caps.protocol_version, crate::PROTOCOL_VERSION);
        assert!(caps.cpu_count >= 1);
    }
}
