//! Sample Ring Buffer
//!
//! Accumulates samples and flushes in batches to minimize IPC syscall overhead.
//! Critical for preventing pipe saturation with nanosecond-scale benchmarks.

use crate::messages::{FlushReason, Sample, SampleBatch};
use crate::{HEARTBEAT_INTERVAL_NS, MAX_BATCH_BYTES, MAX_BATCH_SIZE};

/// Ring buffer for accumulating samples before IPC transmission.
///
/// Flushes when any of these conditions are met:
/// - 10,000 samples accumulated (MAX_BATCH_SIZE)
/// - 64KB of data accumulated (MAX_BATCH_BYTES)
/// - 100ms since last flush with pending data (HEARTBEAT_INTERVAL_NS)
/// - Benchmark completes
pub struct SampleRingBuffer {
    /// Accumulated samples
    samples: Vec<Sample>,
    /// Hash of the benchmark ID
    bench_id_hash: u64,
    /// Current batch sequence number
    batch_sequence: u32,
    /// Starting iteration index for current batch
    start_iteration: u64,
    /// Next iteration index
    next_iteration: u64,
    /// Timestamp of last flush (nanoseconds since arbitrary epoch)
    last_flush_ns: u64,
}

impl SampleRingBuffer {
    /// Create a new ring buffer for the given benchmark
    pub fn new(bench_id: &str) -> Self {
        Self {
            samples: Vec::with_capacity(MAX_BATCH_SIZE),
            bench_id_hash: hash_bench_id(bench_id),
            batch_sequence: 0,
            start_iteration: 0,
            next_iteration: 0,
            last_flush_ns: Self::now_ns(),
        }
    }

    /// Push a sample into the buffer, returning a batch if flush is triggered
    #[inline]
    pub fn push(&mut self, sample: Sample) -> Option<SampleBatch> {
        self.samples.push(sample);
        self.next_iteration += 1;

        // Check flush conditions
        if self.samples.len() >= MAX_BATCH_SIZE {
            return Some(self.flush(FlushReason::BatchFull));
        }

        if self.estimated_bytes() >= MAX_BATCH_BYTES {
            return Some(self.flush(FlushReason::ByteLimitReached));
        }

        None
    }

    /// Check if heartbeat timeout has elapsed, flush if needed
    pub fn check_timeout(&mut self) -> Option<SampleBatch> {
        if self.samples.is_empty() {
            return None;
        }

        let now = Self::now_ns();
        if now.saturating_sub(self.last_flush_ns) >= HEARTBEAT_INTERVAL_NS {
            return Some(self.flush(FlushReason::HeartbeatTimeout));
        }

        None
    }

    /// Flush remaining samples at benchmark completion
    pub fn flush_final(&mut self) -> Option<SampleBatch> {
        if self.samples.is_empty() {
            return None;
        }
        Some(self.flush(FlushReason::BenchmarkComplete))
    }

    /// Force flush with shutdown reason
    pub fn flush_shutdown(&mut self) -> Option<SampleBatch> {
        if self.samples.is_empty() {
            return None;
        }
        Some(self.flush(FlushReason::Shutdown))
    }

    /// Check if the buffer has pending samples
    #[inline]
    pub fn has_pending(&self) -> bool {
        !self.samples.is_empty()
    }

    /// Get the number of pending samples
    #[inline]
    pub fn pending_count(&self) -> usize {
        self.samples.len()
    }

    /// Get the current batch sequence number
    #[inline]
    pub fn current_sequence(&self) -> u32 {
        self.batch_sequence
    }

    /// Estimated size of current buffer in bytes
    fn estimated_bytes(&self) -> usize {
        self.samples.len() * std::mem::size_of::<Sample>()
    }

    /// Internal flush implementation
    fn flush(&mut self, reason: FlushReason) -> SampleBatch {
        let batch = SampleBatch {
            bench_id_hash: self.bench_id_hash,
            batch_sequence: self.batch_sequence,
            start_iteration: self.start_iteration,
            samples: std::mem::take(&mut self.samples),
            flush_reason: reason,
        };

        // Update state for next batch
        self.batch_sequence += 1;
        self.start_iteration = self.next_iteration;
        self.last_flush_ns = Self::now_ns();

        // Pre-allocate for next batch
        self.samples.reserve(MAX_BATCH_SIZE);

        batch
    }

    /// Get current time in nanoseconds (wall-clock).
    ///
    /// Uses `Instant` on all platforms for consistent units.
    /// This is only used for heartbeat timing, not benchmark measurement,
    /// so the ~20ns overhead of Instant is negligible.
    #[inline(always)]
    fn now_ns() -> u64 {
        use std::time::Instant;
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        let start = START.get_or_init(Instant::now);
        start.elapsed().as_nanos() as u64
    }
}

/// Hash a benchmark ID for fast comparison
fn hash_bench_id(id: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    id.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_buffer() {
        let buffer = SampleRingBuffer::new("test_benchmark");
        assert!(!buffer.has_pending());
        assert_eq!(buffer.pending_count(), 0);
        assert_eq!(buffer.current_sequence(), 0);
    }

    #[test]
    fn test_push_single() {
        let mut buffer = SampleRingBuffer::new("test");
        let sample = Sample::timing_only(1000);

        let result = buffer.push(sample);
        assert!(result.is_none()); // No flush yet
        assert!(buffer.has_pending());
        assert_eq!(buffer.pending_count(), 1);
    }

    #[test]
    fn test_flush_on_byte_limit() {
        let mut buffer = SampleRingBuffer::new("test");

        // Calculate how many samples fit below MAX_BATCH_BYTES.
        // Flush triggers when estimated_bytes() >= MAX_BATCH_BYTES.
        // With 32-byte Sample: 65536 / 32 = 2048 exactly, so the 2048th
        // sample hits the limit. Push one fewer without flush.
        let samples_before_flush = MAX_BATCH_BYTES / std::mem::size_of::<Sample>();

        // Push samples until just before byte limit
        for i in 0..(samples_before_flush - 1) {
            let sample = Sample::timing_only(i as u64);
            let result = buffer.push(sample);
            assert!(result.is_none(), "unexpected flush at sample {}", i);
        }

        // This push should trigger flush due to byte limit
        let sample = Sample::timing_only(9999);
        let batch = buffer.push(sample);
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert_eq!(batch.samples.len(), samples_before_flush);
        assert_eq!(batch.flush_reason, FlushReason::ByteLimitReached);
        assert_eq!(batch.batch_sequence, 0);

        // Buffer should be empty now
        assert!(!buffer.has_pending());
        assert_eq!(buffer.current_sequence(), 1);
    }

    #[test]
    fn test_flush_final() {
        let mut buffer = SampleRingBuffer::new("test");

        buffer.push(Sample::timing_only(100));
        buffer.push(Sample::timing_only(200));
        buffer.push(Sample::timing_only(300));

        let batch = buffer.flush_final();
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert_eq!(batch.samples.len(), 3);
        assert_eq!(batch.flush_reason, FlushReason::BenchmarkComplete);
    }

    #[test]
    fn test_flush_final_empty() {
        let mut buffer = SampleRingBuffer::new("test");
        assert!(buffer.flush_final().is_none());
    }

    #[test]
    fn test_bench_id_hash_consistency() {
        let hash1 = hash_bench_id("my_benchmark");
        let hash2 = hash_bench_id("my_benchmark");
        let hash3 = hash_bench_id("other_benchmark");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
