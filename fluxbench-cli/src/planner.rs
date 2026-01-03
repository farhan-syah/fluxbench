//! Benchmark Planner
//!
//! Builds execution plan with dependency ordering.
//!
//! NOTE: This module is planned but not yet integrated.
//! The planner will handle filtering, ordering, and dependency resolution.

use fluxbench_core::BenchmarkDef;

/// Execution plan for benchmarks
#[allow(dead_code)]
pub struct ExecutionPlan {
    /// Ordered list of benchmarks to run
    pub benchmarks: Vec<&'static BenchmarkDef>,
    /// Total estimated time
    pub estimated_duration_ns: u64,
}

/// Build execution plan from discovered benchmarks
#[allow(dead_code)]
pub fn build_plan(
    benchmarks: impl IntoIterator<Item = &'static BenchmarkDef>,
    filter: Option<&regex::Regex>,
    group: Option<&str>,
    tag: Option<&str>,
    skip_tag: Option<&str>,
) -> ExecutionPlan {
    let mut selected: Vec<_> = benchmarks
        .into_iter()
        .filter(|b| {
            // Apply filter
            if let Some(re) = filter {
                if !re.is_match(b.id) {
                    return false;
                }
            }

            // Apply group filter
            if let Some(g) = group {
                if b.group != g {
                    return false;
                }
            }

            // Apply tag filter
            if let Some(t) = tag {
                if !b.tags.contains(&t) {
                    return false;
                }
            }

            // Apply skip tag
            if let Some(st) = skip_tag {
                if b.tags.contains(&st) {
                    return false;
                }
            }

            true
        })
        .collect();

    // TODO: Apply dependency ordering using DependencyGraph
    // For now, just sort by id
    selected.sort_by_key(|b| b.id);

    ExecutionPlan {
        benchmarks: selected,
        estimated_duration_ns: 0, // TODO: estimate based on config
    }
}
