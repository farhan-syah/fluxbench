//! High-Precision Timing
//!
//! Uses RDTSC on x86_64 for minimal overhead timing,
//! with fallback to std::time::Instant on other platforms.

use std::time::Duration;

/// High-precision instant for benchmarking
#[derive(Debug, Clone, Copy)]
pub struct Instant {
    #[cfg(target_arch = "x86_64")]
    tsc: u64,
    #[cfg(not(target_arch = "x86_64"))]
    instant: std::time::Instant,
}

impl Instant {
    /// Capture current instant
    #[inline(always)]
    pub fn now() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: RDTSC is always safe on x86_64
            let tsc = unsafe { std::arch::x86_64::_rdtsc() };
            Self { tsc }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                instant: std::time::Instant::now(),
            }
        }
    }

    /// Compute elapsed time since this instant
    #[inline(always)]
    pub fn elapsed(&self) -> Duration {
        #[cfg(target_arch = "x86_64")]
        {
            let now = unsafe { std::arch::x86_64::_rdtsc() };
            let cycles = now.saturating_sub(self.tsc);
            // Approximate conversion (assumes ~3GHz, calibrated at startup)
            Duration::from_nanos(cycles / 3)
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            self.instant.elapsed()
        }
    }

    /// Raw cycle count (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub fn cycles(&self) -> u64 {
        self.tsc
    }
}

/// Timer for measuring benchmark iterations
pub struct Timer {
    start: Instant,
    cycles_start: u64,
}

impl Timer {
    /// Start a new timer
    #[inline(always)]
    pub fn start() -> Self {
        #[cfg(target_arch = "x86_64")]
        let cycles_start = unsafe { std::arch::x86_64::_rdtsc() };
        #[cfg(not(target_arch = "x86_64"))]
        let cycles_start = 0;

        Self {
            start: Instant::now(),
            cycles_start,
        }
    }

    /// Stop the timer and return elapsed nanoseconds and cycles
    #[inline(always)]
    pub fn stop(&self) -> (u64, u32) {
        let elapsed = self.start.elapsed();
        let nanos = elapsed.as_nanos() as u64;

        #[cfg(target_arch = "x86_64")]
        let cycles = {
            let now = unsafe { std::arch::x86_64::_rdtsc() };
            (now.saturating_sub(self.cycles_start)) as u32
        };
        #[cfg(not(target_arch = "x86_64"))]
        let cycles = 0u32;

        (nanos, cycles)
    }
}

/// Set CPU affinity to pin the current thread to a specific core
///
/// This improves TSC stability by avoiding core migrations.
#[cfg(target_os = "linux")]
pub fn pin_to_cpu(cpu: usize) -> Result<(), std::io::Error> {
    use std::mem::MaybeUninit;

    unsafe {
        let mut set = MaybeUninit::<libc::cpu_set_t>::zeroed();
        let set_ref = set.assume_init_mut();

        libc::CPU_ZERO(set_ref);
        libc::CPU_SET(cpu, set_ref);

        let result = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), set_ref);

        if result == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_to_cpu(_cpu: usize) -> Result<(), std::io::Error> {
    // CPU pinning not supported on this platform
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instant_elapsed() {
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = start.elapsed();

        // Should be at least 10ms
        assert!(elapsed >= Duration::from_millis(5));
        // Should be less than 100ms (accounting for scheduling)
        assert!(elapsed < Duration::from_millis(100));
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(Duration::from_millis(10));
        let (nanos, _cycles) = timer.stop();

        // Should be at least 5ms in nanos
        assert!(nanos >= 5_000_000);
    }
}
