//! Global Allocator Interceptor
//!
//! Tracks heap allocations during benchmark execution.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};

/// Thread-local allocation counter
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);

/// Tracking allocator that wraps the system allocator
pub struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Delegates to the system allocator with the provided layout.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Pointer/layout are provided by corresponding allocation operations.
        unsafe { System.dealloc(ptr, layout) };
        // Note: We don't decrement on dealloc to track total allocations
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: Delegates to the system allocator with the provided layout.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: Pointer/layout originate from allocator contracts; new_size is caller-provided.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            // Track the size difference
            if new_size > layout.size() {
                ALLOCATED_BYTES.fetch_add((new_size - layout.size()) as u64, Ordering::Relaxed);
            }
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        }
        new_ptr
    }
}

/// Get current allocation statistics
pub fn current_allocation() -> (u64, u64) {
    (
        ALLOCATED_BYTES.load(Ordering::Relaxed),
        ALLOCATION_COUNT.load(Ordering::Relaxed),
    )
}

/// Reset allocation counters (call before each iteration)
pub fn reset_allocation_counter() {
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reset_allocation_counter() {
        // Set some values
        ALLOCATED_BYTES.store(1000, std::sync::atomic::Ordering::Relaxed);
        ALLOCATION_COUNT.store(5, std::sync::atomic::Ordering::Relaxed);

        // Reset
        reset_allocation_counter();

        let (bytes, count) = current_allocation();
        assert_eq!(bytes, 0);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_current_allocation_reads_atomics() {
        ALLOCATED_BYTES.store(2048, std::sync::atomic::Ordering::Relaxed);
        ALLOCATION_COUNT.store(10, std::sync::atomic::Ordering::Relaxed);

        let (bytes, count) = current_allocation();
        assert_eq!(bytes, 2048);
        assert_eq!(count, 10);

        // Clean up
        reset_allocation_counter();
    }

    // Note: Full allocation tracking test requires TrackingAllocator
    // to be installed as #[global_allocator] in the binary crate.
    // This is done by end users, not the library.
}
