//! Worker Process Entry Point
//!
//! Handles the worker side of the supervisor-worker architecture.

use crate::measure::pin_to_cpu;
use crate::{Bencher, BenchmarkDef, run_benchmark_loop};
use fluxbench_ipc::{
    BenchmarkConfig, FailureKind, FrameReader, FrameWriter, SampleRingBuffer, SupervisorCommand,
    WorkerCapabilities, WorkerMessage,
};
use std::io::{stdin, stdout};

/// Worker main loop
pub struct WorkerMain {
    reader: FrameReader<std::io::Stdin>,
    writer: FrameWriter<std::io::Stdout>,
}

impl WorkerMain {
    /// Create a new worker connected to stdin/stdout
    pub fn new() -> Self {
        Self {
            reader: FrameReader::new(stdin()),
            writer: FrameWriter::new(stdout()),
        }
    }

    /// Run the worker main loop
    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Send capabilities
        self.writer
            .write(&WorkerMessage::Hello(WorkerCapabilities::default()))?;

        // Pin to CPU 0 for stable TSC
        let _ = pin_to_cpu(0);

        // Process commands
        loop {
            let command: SupervisorCommand = self.reader.read()?;

            match command {
                SupervisorCommand::Run { bench_id, config } => {
                    self.run_benchmark(&bench_id, &config)?;
                }
                SupervisorCommand::Abort => {
                    // Acknowledge and stop current work
                    break;
                }
                SupervisorCommand::Shutdown => {
                    break;
                }
                SupervisorCommand::Ping => {
                    // Just acknowledge by continuing
                }
            }
        }

        Ok(())
    }

    /// Run a single benchmark
    fn run_benchmark(
        &mut self,
        bench_id: &str,
        config: &BenchmarkConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Find the benchmark
        let bench = inventory::iter::<BenchmarkDef>
            .into_iter()
            .find(|b| b.id == bench_id);

        let bench = match bench {
            Some(b) => b,
            None => {
                self.writer.write(&WorkerMessage::Failure {
                    kind: FailureKind::Unknown,
                    message: format!("Benchmark not found: {}", bench_id),
                    backtrace: None,
                })?;
                return Ok(());
            }
        };

        // Create ring buffer for batched IPC
        let mut ring_buffer = SampleRingBuffer::new(bench_id);

        // Run with panic catching
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let bencher = Bencher::new(config.track_allocations);

            run_benchmark_loop(
                bencher,
                |b| (bench.runner_fn)(b),
                config.warmup_time_ns,
                config.measurement_time_ns,
                config.min_iterations,
                config.max_iterations,
            )
        }));

        match result {
            Ok(bench_result) => {
                // Send samples in batches
                for sample in bench_result.samples {
                    if let Some(batch) = ring_buffer.push(sample) {
                        self.writer.write(&WorkerMessage::SampleBatch(batch))?;
                    }
                }

                // Flush remaining samples
                if let Some(batch) = ring_buffer.flush_final() {
                    self.writer.write(&WorkerMessage::SampleBatch(batch))?;
                }

                // Send completion
                self.writer.write(&WorkerMessage::Complete {
                    total_iterations: bench_result.iterations,
                    total_duration_nanos: bench_result.total_time_ns,
                })?;
            }
            Err(panic) => {
                let message = if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "Unknown panic".to_string()
                };

                // Capture backtrace - requires RUST_BACKTRACE=1 for full traces
                let backtrace = std::backtrace::Backtrace::capture();
                let backtrace_str = match backtrace.status() {
                    std::backtrace::BacktraceStatus::Captured => Some(backtrace.to_string()),
                    _ => None,
                };

                self.writer.write(&WorkerMessage::Failure {
                    kind: FailureKind::Panic,
                    message,
                    backtrace: backtrace_str,
                })?;
            }
        }

        Ok(())
    }
}

impl Default for WorkerMain {
    fn default() -> Self {
        Self::new()
    }
}
