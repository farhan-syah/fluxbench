//! Worker Process Entry Point
//!
//! Handles the worker side of the supervisor-worker architecture.
//!
//! On Unix, uses fd 3/4 for IPC (set via `FLUX_IPC_FD` env var) and installs
//! a SIGTERM handler for graceful shutdown. On non-Unix, falls back to
//! stdin/stdout and skips signal handling.

use crate::measure::pin_to_cpu;
use crate::{Bencher, BenchmarkDef, run_benchmark_loop};
use fluxbench_ipc::{
    BenchmarkConfig, FailureKind, FrameReader, FrameWriter, SampleRingBuffer, SupervisorCommand,
    WorkerCapabilities, WorkerMessage,
};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(unix)]
use std::os::unix::io::FromRawFd;

/// Global flag set by SIGTERM handler to request graceful shutdown.
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Check if a graceful shutdown has been requested via SIGTERM.
pub fn shutdown_requested() -> bool {
    SHUTDOWN_REQUESTED.load(Ordering::Relaxed)
}

/// Install a SIGTERM handler that sets the `SHUTDOWN_REQUESTED` flag.
/// The handler is async-signal-safe (only sets an atomic).
#[cfg(unix)]
fn install_sigterm_handler() {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = sigterm_handler as usize;
        sa.sa_flags = libc::SA_RESTART;
        libc::sigemptyset(&mut sa.sa_mask);
        libc::sigaction(libc::SIGTERM, &sa, std::ptr::null_mut());
    }
}

#[cfg(unix)]
extern "C" fn sigterm_handler(_sig: libc::c_int) {
    SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);
}

/// No-op on non-Unix (no SIGTERM equivalent).
#[cfg(not(unix))]
fn install_sigterm_handler() {}

/// IPC transport: either inherited fd pair or stdin/stdout fallback.
enum IpcTransport {
    #[cfg(unix)]
    Fds {
        read_fd: i32,
        write_fd: i32,
    },
    Stdio,
}

fn detect_transport() -> IpcTransport {
    #[cfg(unix)]
    if let Ok(val) = std::env::var("FLUX_IPC_FD") {
        let parts: Vec<&str> = val.split(',').collect();
        if parts.len() == 2 {
            if let (Ok(r), Ok(w)) = (parts[0].parse::<i32>(), parts[1].parse::<i32>()) {
                return IpcTransport::Fds {
                    read_fd: r,
                    write_fd: w,
                };
            }
        }
        eprintln!(
            "fluxbench: warning: invalid FLUX_IPC_FD={val:?} (expected format: <read_fd>,<write_fd>), falling back to stdio"
        );
    }
    IpcTransport::Stdio
}

/// Worker main loop
pub struct WorkerMain {
    reader: FrameReader<Box<dyn std::io::Read>>,
    writer: FrameWriter<Box<dyn std::io::Write>>,
}

impl WorkerMain {
    /// Create a new worker, using fd 3/4 if FLUX_IPC_FD is set, otherwise stdin/stdout.
    pub fn new() -> Self {
        match detect_transport() {
            #[cfg(unix)]
            IpcTransport::Fds { read_fd, write_fd } => {
                let read_file = unsafe { std::fs::File::from_raw_fd(read_fd) };
                let write_file = unsafe { std::fs::File::from_raw_fd(write_fd) };
                Self {
                    reader: FrameReader::new(Box::new(read_file) as Box<dyn std::io::Read>),
                    writer: FrameWriter::new(Box::new(write_file) as Box<dyn std::io::Write>),
                }
            }
            IpcTransport::Stdio => Self {
                reader: FrameReader::new(Box::new(std::io::stdin()) as Box<dyn std::io::Read>),
                writer: FrameWriter::new(Box::new(std::io::stdout()) as Box<dyn std::io::Write>),
            },
        }
    }

    /// Run the worker main loop
    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        install_sigterm_handler();

        // Send capabilities
        self.writer
            .write(&WorkerMessage::Hello(WorkerCapabilities::default()))?;

        // Pin to CPU 0 for stable TSC
        let _ = pin_to_cpu(0);

        // Process commands
        loop {
            if shutdown_requested() {
                break;
            }

            let command: SupervisorCommand = self.reader.read()?;

            match command {
                SupervisorCommand::Run { bench_id, config } => {
                    self.run_benchmark(&bench_id, &config)?;
                    if shutdown_requested() {
                        break;
                    }
                }
                SupervisorCommand::Abort => {
                    break;
                }
                SupervisorCommand::Shutdown => {
                    break;
                }
                SupervisorCommand::Ping => {}
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

                // Flush any samples collected before the panic
                if let Some(batch) = ring_buffer.flush_final() {
                    let _ = self.writer.write(&WorkerMessage::SampleBatch(batch));
                }

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
