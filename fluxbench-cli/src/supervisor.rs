//! Supervisor Process
//!
//! Manages worker processes and aggregates results via IPC.
//!
//! On Unix, creates dynamically-allocated pipe pairs and passes their fd
//! numbers to the worker via the `FLUX_IPC_FD` environment variable,
//! leaving stdout/stderr free for user benchmark code. On non-Unix
//! platforms, falls back to stdin/stdout pipes (user `println!` may
//! corrupt the protocol stream in this mode).
//!
//! **Timeout behavior:** On Unix, sends SIGTERM for graceful shutdown then
//! drains pending samples (500ms window) before SIGKILL. On non-Unix,
//! kills immediately without draining — partial samples are lost.

use fluxbench_core::BenchmarkDef;
use fluxbench_ipc::{
    BenchmarkConfig, FrameError, FrameReader, FrameWriter, Sample, SupervisorCommand,
    WorkerCapabilities, WorkerMessage,
};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::env;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use thiserror::Error;

#[cfg(unix)]
use std::os::unix::io::{FromRawFd, RawFd};
#[cfg(unix)]
use std::os::unix::process::CommandExt;

#[derive(Debug, Error)]
pub enum SupervisorError {
    #[error("Failed to spawn worker: {0}")]
    SpawnFailed(#[from] std::io::Error),

    #[error("IPC error: {0}")]
    IpcError(String),

    #[error("Worker crashed: {0}")]
    WorkerCrashed(String),

    #[error("Timeout waiting for worker")]
    Timeout,

    #[error("Benchmark not found: {0}")]
    BenchmarkNotFound(String),

    #[error("Worker protocol error: expected {expected}, got {got}")]
    ProtocolError { expected: String, got: String },
}

impl From<FrameError> for SupervisorError {
    fn from(e: FrameError) -> Self {
        SupervisorError::IpcError(e.to_string())
    }
}

/// Result from a benchmark run via IPC
#[derive(Debug)]
pub struct IpcBenchmarkResult {
    pub bench_id: String,
    pub samples: Vec<Sample>,
    pub total_iterations: u64,
    pub total_duration_nanos: u64,
    pub status: IpcBenchmarkStatus,
}

#[derive(Debug, Clone)]
pub enum IpcBenchmarkStatus {
    Success,
    Failed { message: String },
    Crashed { message: String },
}

// ─── Platform-specific poll ──────────────────────────────────────────────────

/// Result of polling for data
#[derive(Debug)]
enum PollResult {
    DataAvailable,
    Timeout,
    PipeClosed,
    Error(std::io::Error),
}

/// Wait for data to be available on a raw fd with timeout (Unix: `poll(2)`).
#[cfg(unix)]
fn wait_for_data_fd(fd: i32, timeout_ms: i32) -> PollResult {
    let mut pollfd = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };

    let result = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };

    if result < 0 {
        PollResult::Error(std::io::Error::last_os_error())
    } else if result == 0 {
        PollResult::Timeout
    } else if pollfd.revents & libc::POLLIN != 0 {
        PollResult::DataAvailable
    } else if pollfd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
        PollResult::PipeClosed
    } else {
        PollResult::Timeout
    }
}

// ─── Platform-specific pipe/fd helpers (Unix only) ───────────────────────────

#[cfg(unix)]
fn create_pipe() -> Result<(RawFd, RawFd), std::io::Error> {
    let mut fds = [0 as RawFd; 2];
    let ret = unsafe { libc::pipe(fds.as_mut_ptr()) };
    if ret != 0 {
        return Err(std::io::Error::last_os_error());
    }
    for &fd in &fds {
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFD);
            libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC);
        }
    }
    Ok((fds[0], fds[1]))
}

#[cfg(unix)]
fn close_fd(fd: RawFd) {
    unsafe {
        libc::close(fd);
    }
}

/// Send SIGTERM to a process. Returns `Err` if the signal could not be delivered.
#[cfg(unix)]
fn send_sigterm(pid: u32) -> Result<(), std::io::Error> {
    let ret = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
    if ret == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

// ─── WorkerHandle ────────────────────────────────────────────────────────────

/// Worker process handle.
///
/// Wraps the IPC reader/writer as trait objects so both the Unix (File-backed)
/// and non-Unix (ChildStdin/ChildStdout-backed) paths share the same struct.
/// Trait objects are bounded by `Send` because the handle may be moved across
/// threads in the supervisor's `rayon` thread pool.
pub struct WorkerHandle {
    child: Child,
    reader: FrameReader<Box<dyn std::io::Read + Send>>,
    writer: FrameWriter<Box<dyn std::io::Write + Send>>,
    capabilities: Option<WorkerCapabilities>,
    timeout: Duration,
    /// Raw fd for `poll(2)` (Unix). On non-Unix this is unused (-1).
    poll_fd: i32,
}

impl WorkerHandle {
    /// Spawn a new worker process.
    pub fn spawn(timeout: Duration) -> Result<Self, SupervisorError> {
        let binary = env::current_exe().map_err(SupervisorError::SpawnFailed)?;
        Self::spawn_impl(&binary, timeout)
    }

    /// Spawn a worker for a specific binary (for testing).
    pub fn spawn_binary(binary: &str, timeout: Duration) -> Result<Self, SupervisorError> {
        Self::spawn_impl(binary.as_ref(), timeout)
    }

    // ── Unix spawn: dedicated pipe fds, stdout free for user code ────────

    #[cfg(unix)]
    fn spawn_impl(binary: &std::path::Path, timeout: Duration) -> Result<Self, SupervisorError> {
        // cmd_pipe: supervisor writes commands → worker reads from cmd_read
        let (cmd_read, cmd_write) = create_pipe()?;
        // msg_pipe: worker writes messages from msg_write → supervisor reads from msg_read
        let (msg_read, msg_write) = match create_pipe() {
            Ok(fds) => fds,
            Err(e) => {
                close_fd(cmd_read);
                close_fd(cmd_write);
                return Err(SupervisorError::SpawnFailed(e));
            }
        };

        // Pass the actual pipe fd numbers to the worker — no dup2 to hardcoded
        // fds 3/4 which may already be in use by library static initializers.
        let mut command = Command::new(binary);
        command
            .arg("--flux-worker")
            .env("FLUX_IPC_FD", format!("{},{}", cmd_read, msg_write))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());

        // In the child: clear CLOEXEC on the child-side fds so they survive exec,
        // and close the parent-side fds.
        unsafe {
            command.pre_exec(move || {
                // Keep child-side fds open across exec
                let flags = libc::fcntl(cmd_read, libc::F_GETFD);
                if flags == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::fcntl(cmd_read, libc::F_SETFD, flags & !libc::FD_CLOEXEC) == -1 {
                    return Err(std::io::Error::last_os_error());
                }

                let flags = libc::fcntl(msg_write, libc::F_GETFD);
                if flags == -1 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::fcntl(msg_write, libc::F_SETFD, flags & !libc::FD_CLOEXEC) == -1 {
                    return Err(std::io::Error::last_os_error());
                }

                // Close parent-side ends that leaked into the child
                libc::close(cmd_write);
                libc::close(msg_read);

                Ok(())
            });
        }

        let child = match command.spawn() {
            Ok(c) => c,
            Err(e) => {
                close_fd(cmd_read);
                close_fd(cmd_write);
                close_fd(msg_read);
                close_fd(msg_write);
                return Err(SupervisorError::SpawnFailed(e));
            }
        };

        // Close child-side ends in the parent
        close_fd(cmd_read);
        close_fd(msg_write);

        // SAFETY: msg_read and cmd_write are valid fds whose child-side
        // counterparts have been closed above. from_raw_fd takes ownership
        // and will close them on drop.
        let reader_file = unsafe { std::fs::File::from_raw_fd(msg_read) };
        let writer_file = unsafe { std::fs::File::from_raw_fd(cmd_write) };

        // Duplicate the fd for poll(2) so we have an independent handle that
        // doesn't alias the File-owned fd. Closed in WorkerHandle::drop().
        let poll_fd = unsafe { libc::dup(msg_read) };
        if poll_fd < 0 {
            return Err(SupervisorError::SpawnFailed(std::io::Error::last_os_error()));
        }

        let mut handle = Self {
            child,
            reader: FrameReader::new(Box::new(reader_file) as Box<dyn std::io::Read + Send>),
            writer: FrameWriter::new(Box::new(writer_file) as Box<dyn std::io::Write + Send>),
            capabilities: None,
            timeout,
            poll_fd,
        };

        handle.wait_for_hello()?;
        Ok(handle)
    }

    // ── Non-Unix spawn: stdin/stdout pipes (fallback) ────────────────────

    #[cfg(not(unix))]
    fn spawn_impl(binary: &std::path::Path, timeout: Duration) -> Result<Self, SupervisorError> {
        let mut child = Command::new(binary)
            .arg("--flux-worker")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take().expect("stdin should be available");
        let stdout = child.stdout.take().expect("stdout should be available");

        let mut handle = Self {
            child,
            reader: FrameReader::new(Box::new(stdout) as Box<dyn std::io::Read + Send>),
            writer: FrameWriter::new(Box::new(stdin) as Box<dyn std::io::Write + Send>),
            capabilities: None,
            timeout,
            poll_fd: -1,
        };

        handle.wait_for_hello()?;
        Ok(handle)
    }

    // ── Shared logic ─────────────────────────────────────────────────────

    /// Wait for Hello message from worker and validate protocol version
    fn wait_for_hello(&mut self) -> Result<(), SupervisorError> {
        let msg: WorkerMessage = self.reader.read()?;

        match msg {
            WorkerMessage::Hello(caps) => {
                if caps.protocol_version != fluxbench_ipc::PROTOCOL_VERSION {
                    return Err(SupervisorError::ProtocolError {
                        expected: format!("protocol version {}", fluxbench_ipc::PROTOCOL_VERSION),
                        got: format!("protocol version {}", caps.protocol_version),
                    });
                }
                self.capabilities = Some(caps);
                Ok(())
            }
            other => Err(SupervisorError::ProtocolError {
                expected: "Hello".to_string(),
                got: format!("{:?}", other),
            }),
        }
    }

    /// Get worker capabilities
    pub fn capabilities(&self) -> Option<&WorkerCapabilities> {
        self.capabilities.as_ref()
    }

    /// Run a benchmark on this worker
    pub fn run_benchmark(
        &mut self,
        bench_id: &str,
        config: &BenchmarkConfig,
    ) -> Result<IpcBenchmarkResult, SupervisorError> {
        self.writer.write(&SupervisorCommand::Run {
            bench_id: bench_id.to_string(),
            config: config.clone(),
        })?;

        let mut all_samples = Vec::new();
        let start = Instant::now();

        loop {
            let remaining = self.timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return self.handle_timeout(all_samples);
            }

            // Check if there's buffered data, or poll for new data.
            // Even with buffered data we verify the worker is alive — the buffer
            // might hold an incomplete frame that will never be completed.
            if self.reader.has_buffered_data() {
                if !self.is_alive() {
                    return Err(SupervisorError::WorkerCrashed(
                        "Worker process crashed with partial data buffered".to_string(),
                    ));
                }
            } else {
                self.wait_for_worker_data(remaining)?;
            }

            // Read next message (blocking — poll/sleep above confirmed data is available)
            let msg: WorkerMessage = match self.reader.read::<WorkerMessage>() {
                Ok(msg) => msg,
                Err(FrameError::EndOfStream) => {
                    return Err(SupervisorError::WorkerCrashed(
                        "Worker closed connection unexpectedly".to_string(),
                    ));
                }
                Err(e) => {
                    if !self.is_alive() {
                        return Err(SupervisorError::WorkerCrashed(
                            "Worker crashed during read".to_string(),
                        ));
                    }
                    return Err(SupervisorError::IpcError(e.to_string()));
                }
            };

            match msg {
                WorkerMessage::SampleBatch(batch) => {
                    all_samples.extend(batch.samples);
                }
                WorkerMessage::WarmupComplete { .. } | WorkerMessage::Progress { .. } => {
                    continue;
                }
                WorkerMessage::Complete {
                    total_iterations,
                    total_duration_nanos,
                } => {
                    return Ok(IpcBenchmarkResult {
                        bench_id: bench_id.to_string(),
                        samples: all_samples,
                        total_iterations,
                        total_duration_nanos,
                        status: IpcBenchmarkStatus::Success,
                    });
                }
                WorkerMessage::Failure {
                    kind,
                    message,
                    backtrace: _,
                } => {
                    return Ok(IpcBenchmarkResult {
                        bench_id: bench_id.to_string(),
                        samples: all_samples,
                        total_iterations: 0,
                        total_duration_nanos: 0,
                        status: match kind {
                            fluxbench_ipc::FailureKind::Panic => {
                                IpcBenchmarkStatus::Crashed { message }
                            }
                            _ => IpcBenchmarkStatus::Failed { message },
                        },
                    });
                }
                WorkerMessage::Hello(_) => {
                    return Err(SupervisorError::ProtocolError {
                        expected: "SampleBatch/Complete/Failure".to_string(),
                        got: "Hello".to_string(),
                    });
                }
            }
        }
    }

    /// Wait for data from the worker, checking liveness periodically.
    ///
    /// On Unix this uses `poll(2)` on the message fd.
    /// On non-Unix this uses a polling sleep loop (less efficient but portable).
    #[cfg(unix)]
    fn wait_for_worker_data(&mut self, remaining: Duration) -> Result<(), SupervisorError> {
        let poll_timeout = remaining.min(Duration::from_millis(100));
        match wait_for_data_fd(self.poll_fd, poll_timeout.as_millis() as i32) {
            PollResult::DataAvailable => {
                if !self.is_alive() {
                    return Err(SupervisorError::WorkerCrashed(
                        "Worker process crashed with data in pipe".to_string(),
                    ));
                }
                Ok(())
            }
            PollResult::Timeout => {
                if !self.is_alive() {
                    return Err(SupervisorError::WorkerCrashed(
                        "Worker process exited unexpectedly".to_string(),
                    ));
                }
                // Signal caller to re-loop (no data yet)
                Ok(())
            }
            PollResult::PipeClosed => Err(SupervisorError::WorkerCrashed(
                "Worker pipe closed unexpectedly".to_string(),
            )),
            PollResult::Error(e) => {
                Err(SupervisorError::WorkerCrashed(format!("Pipe error: {}", e)))
            }
        }
    }

    #[cfg(not(unix))]
    fn wait_for_worker_data(&mut self, _remaining: Duration) -> Result<(), SupervisorError> {
        // Without poll(2), sleep briefly then check liveness.
        // The subsequent blocking read will pick up data.
        std::thread::sleep(Duration::from_millis(10));
        if !self.is_alive() {
            return Err(SupervisorError::WorkerCrashed(
                "Worker process exited unexpectedly".to_string(),
            ));
        }
        Ok(())
    }

    /// Handle timeout: on Unix send SIGTERM → drain 500ms → SIGKILL.
    /// On non-Unix just kill immediately.
    #[cfg(unix)]
    fn handle_timeout(
        &mut self,
        mut samples: Vec<Sample>,
    ) -> Result<IpcBenchmarkResult, SupervisorError> {
        // Send SIGTERM for graceful shutdown (ignore error — worker may already be dead)
        let _ = send_sigterm(self.child.id());

        // Drain any messages the worker flushes in response to SIGTERM (500ms window)
        let drain_deadline = Instant::now() + Duration::from_millis(500);
        loop {
            let remaining = drain_deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }

            match wait_for_data_fd(self.poll_fd, remaining.as_millis() as i32) {
                PollResult::DataAvailable => match self.reader.read::<WorkerMessage>() {
                    Ok(WorkerMessage::SampleBatch(batch)) => {
                        samples.extend(batch.samples);
                    }
                    Ok(WorkerMessage::Complete { .. }) => break,
                    _ => break,
                },
                PollResult::PipeClosed => break,
                _ => break,
            }
        }

        if self.is_alive() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }

        Err(SupervisorError::Timeout)
    }

    #[cfg(not(unix))]
    fn handle_timeout(
        &mut self,
        _samples: Vec<Sample>,
    ) -> Result<IpcBenchmarkResult, SupervisorError> {
        // No SIGTERM on non-Unix — just kill immediately
        if self.is_alive() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        Err(SupervisorError::Timeout)
    }

    /// Ping the worker to check if it's alive
    pub fn ping(&mut self) -> Result<bool, SupervisorError> {
        self.writer.write(&SupervisorCommand::Ping)?;
        Ok(true)
    }

    /// Abort the current benchmark
    pub fn abort(&mut self) -> Result<(), SupervisorError> {
        self.writer.write(&SupervisorCommand::Abort)?;
        Ok(())
    }

    /// Shutdown the worker gracefully
    pub fn shutdown(mut self) -> Result<(), SupervisorError> {
        self.writer.write(&SupervisorCommand::Shutdown)?;
        let _ = self.child.wait();
        Ok(())
    }

    /// Check if worker process is still running
    pub fn is_alive(&mut self) -> bool {
        match self.child.try_wait() {
            Ok(Some(_)) => false,
            Ok(None) => true,
            Err(_) => false,
        }
    }

    /// Kill the worker process forcefully
    pub fn kill(&mut self) -> Result<(), SupervisorError> {
        self.child.kill().map_err(SupervisorError::SpawnFailed)?;
        let _ = self.child.wait();
        Ok(())
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        if self.is_alive() {
            #[cfg(unix)]
            {
                // Graceful: SIGTERM first, brief wait, then SIGKILL
                let _ = send_sigterm(self.child.id());
                std::thread::sleep(Duration::from_millis(50));
            }
            if self.is_alive() {
                let _ = self.child.kill();
            }
            let _ = self.child.wait();
        }

        // Close the duplicated poll fd (Unix only; -1 on non-Unix is a no-op)
        #[cfg(unix)]
        if self.poll_fd >= 0 {
            close_fd(self.poll_fd);
        }
    }
}

// ─── Supervisor ──────────────────────────────────────────────────────────────

/// Supervisor that manages worker pool and distributes benchmarks
pub struct Supervisor {
    config: BenchmarkConfig,
    timeout: Duration,
    num_workers: usize,
}

impl Supervisor {
    /// Create a new supervisor
    pub fn new(config: BenchmarkConfig, timeout: Duration, num_workers: usize) -> Self {
        Self {
            config,
            timeout,
            num_workers: num_workers.max(1),
        }
    }

    /// Run all benchmarks with process isolation
    pub fn run_all(
        &self,
        benchmarks: &[&BenchmarkDef],
    ) -> Result<Vec<IpcBenchmarkResult>, SupervisorError> {
        if benchmarks.is_empty() {
            return Ok(Vec::new());
        }

        if self.num_workers == 1 || benchmarks.len() == 1 {
            let mut results = Vec::with_capacity(benchmarks.len());
            for bench in benchmarks {
                results.push(self.run_isolated(bench)?);
            }
            return Ok(results);
        }

        let worker_count = self.num_workers.min(benchmarks.len());
        let pool = ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .map_err(|e| {
                SupervisorError::IpcError(format!("Failed to build worker pool: {}", e))
            })?;

        let outcomes: Vec<Result<IpcBenchmarkResult, SupervisorError>> = pool.install(|| {
            benchmarks
                .par_iter()
                .map(|bench| self.run_isolated(bench))
                .collect()
        });

        let mut results = Vec::with_capacity(outcomes.len());
        for outcome in outcomes {
            results.push(outcome?);
        }
        Ok(results)
    }

    /// Run a single benchmark in an isolated worker process
    fn run_isolated(&self, bench: &BenchmarkDef) -> Result<IpcBenchmarkResult, SupervisorError> {
        let mut worker = WorkerHandle::spawn(self.timeout)?;
        let result = worker.run_benchmark(bench.id, &self.config);
        let _ = worker.shutdown();
        result
    }

    fn crashed_result(bench: &BenchmarkDef, message: String) -> IpcBenchmarkResult {
        IpcBenchmarkResult {
            bench_id: bench.id.to_string(),
            samples: Vec::new(),
            total_iterations: 0,
            total_duration_nanos: 0,
            status: IpcBenchmarkStatus::Crashed { message },
        }
    }

    fn run_with_reuse_indexed(
        &self,
        benchmarks: &[(usize, &BenchmarkDef)],
    ) -> Vec<(usize, IpcBenchmarkResult)> {
        let mut results = Vec::with_capacity(benchmarks.len());
        if benchmarks.is_empty() {
            return results;
        }

        let mut worker = match WorkerHandle::spawn(self.timeout) {
            Ok(worker) => Some(worker),
            Err(e) => {
                let message = e.to_string();
                for &(index, bench) in benchmarks {
                    results.push((index, Self::crashed_result(bench, message.clone())));
                }
                return results;
            }
        };

        for &(index, bench) in benchmarks {
            if worker.is_none() {
                match WorkerHandle::spawn(self.timeout) {
                    Ok(new_worker) => worker = Some(new_worker),
                    Err(e) => {
                        results.push((index, Self::crashed_result(bench, e.to_string())));
                        continue;
                    }
                }
            }

            let run_result = match worker.as_mut() {
                Some(worker) => worker.run_benchmark(bench.id, &self.config),
                None => unreachable!("worker should exist after spawn check"),
            };

            match run_result {
                Ok(result) => results.push((index, result)),
                Err(e) => {
                    let worker_is_alive = worker.as_mut().map(|w| w.is_alive()).unwrap_or(false);
                    if !worker_is_alive {
                        if let Some(mut dead_worker) = worker.take() {
                            let _ = dead_worker.kill();
                        }
                    }
                    results.push((index, Self::crashed_result(bench, e.to_string())));
                }
            }
        }

        if let Some(worker) = worker {
            let _ = worker.shutdown();
        }

        results
    }

    /// Run benchmarks with worker reuse (less isolation but faster)
    pub fn run_with_reuse(
        &self,
        benchmarks: &[&BenchmarkDef],
    ) -> Result<Vec<IpcBenchmarkResult>, SupervisorError> {
        if benchmarks.is_empty() {
            return Ok(Vec::new());
        }

        let indexed_benchmarks: Vec<(usize, &BenchmarkDef)> = benchmarks
            .iter()
            .enumerate()
            .map(|(index, bench)| (index, *bench))
            .collect();

        let mut indexed_results = if self.num_workers == 1 || benchmarks.len() == 1 {
            self.run_with_reuse_indexed(&indexed_benchmarks)
        } else {
            let worker_count = self.num_workers.min(indexed_benchmarks.len());
            let mut shards: Vec<Vec<(usize, &BenchmarkDef)>> = vec![Vec::new(); worker_count];
            for (position, entry) in indexed_benchmarks.into_iter().enumerate() {
                shards[position % worker_count].push(entry);
            }

            let pool = ThreadPoolBuilder::new()
                .num_threads(worker_count)
                .build()
                .map_err(|e| {
                    SupervisorError::IpcError(format!("Failed to build worker pool: {}", e))
                })?;

            let shard_results: Vec<Vec<(usize, IpcBenchmarkResult)>> = pool.install(|| {
                shards
                    .into_par_iter()
                    .map(|shard| self.run_with_reuse_indexed(&shard))
                    .collect()
            });

            shard_results.into_iter().flatten().collect()
        };

        indexed_results.sort_by_key(|(index, _)| *index);
        if indexed_results.len() != benchmarks.len() {
            return Err(SupervisorError::IpcError(format!(
                "Internal error: expected {} results, got {}",
                benchmarks.len(),
                indexed_results.len()
            )));
        }

        Ok(indexed_results
            .into_iter()
            .map(|(_, result)| result)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires built binary
    fn test_supervisor_spawn() {
        let timeout = Duration::from_secs(30);
        let config = BenchmarkConfig::default();
        let supervisor = Supervisor::new(config, timeout, 1);
        assert_eq!(supervisor.num_workers, 1);
    }
}
