//! FluxBench IPC Protocol
//!
//! Zero-copy serialization protocol for supervisor-worker communication.
//! Uses rkyv for efficient binary serialization with no parsing overhead.

mod framing;
mod messages;
mod ring_buffer;

pub use framing::{read_frame, write_frame, FrameError, FrameReader, FrameWriter, MAX_FRAME_SIZE};
pub use messages::{
    BenchmarkConfig, FailureKind, FlushReason, Sample, SampleBatch, SupervisorCommand,
    WorkerCapabilities, WorkerMessage,
};
pub use ring_buffer::SampleRingBuffer;

/// Protocol version for compatibility checking
pub const PROTOCOL_VERSION: u32 = 1;

/// Maximum samples per batch (prevents unbounded memory growth)
pub const MAX_BATCH_SIZE: usize = 10_000;

/// Maximum batch size in bytes (64KB)
pub const MAX_BATCH_BYTES: usize = 64 * 1024;

/// Heartbeat interval in nanoseconds (100ms)
pub const HEARTBEAT_INTERVAL_NS: u64 = 100_000_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_size() {
        // Sample must be exactly 24 bytes for cache efficiency
        assert_eq!(std::mem::size_of::<Sample>(), 24);
    }

    #[test]
    fn test_sample_alignment() {
        // Sample must be 8-byte aligned
        assert_eq!(std::mem::align_of::<Sample>(), 8);
    }
}
