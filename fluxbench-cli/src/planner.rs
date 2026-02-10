//! Benchmark Planner
//!
//! Builds execution plan by filtering and ordering benchmarks.
//!
//! Filtering options:
//! - Regex pattern matching on benchmark ID
//! - Group filtering
//! - Tag inclusion/exclusion
//!
//! Ordering: Benchmarks are sorted alphabetically by ID for deterministic execution.

use fluxbench_core::BenchmarkDef;

/// Execution plan for benchmarks
pub struct ExecutionPlan {
    /// Ordered list of benchmarks to run
    pub benchmarks: Vec<&'static BenchmarkDef>,
}

/// Build execution plan from discovered benchmarks
///
/// Filters benchmarks based on CLI options and returns them in deterministic order.
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
            // Apply regex filter on benchmark ID
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

            // Apply tag inclusion filter
            if let Some(t) = tag {
                if !b.tags.contains(&t) {
                    return false;
                }
            }

            // Apply tag exclusion filter
            if let Some(st) = skip_tag {
                if b.tags.contains(&st) {
                    return false;
                }
            }

            true
        })
        .collect();

    // Sort alphabetically for deterministic execution order
    selected.sort_by_key(|b| b.id);

    ExecutionPlan {
        benchmarks: selected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxbench_core::Severity;

    fn make_bench(
        id: &'static str,
        group: &'static str,
        tags: &'static [&'static str],
    ) -> BenchmarkDef {
        BenchmarkDef {
            id,
            name: id,
            group,
            severity: Severity::Info,
            threshold: 0.0,
            budget_ns: None,
            tags,
            runner_fn: |_| {},
            file: "",
            line: 0,
            module_path: "",
        }
    }

    #[test]
    fn test_no_filter() {
        let benches = [
            make_bench("c_bench", "default", &[]),
            make_bench("a_bench", "default", &[]),
            make_bench("b_bench", "default", &[]),
        ];
        let refs: Vec<_> = benches
            .iter()
            .map(|b| unsafe { &*(b as *const _) })
            .collect();

        let plan = build_plan(refs, None, None, None, None);

        // Should be sorted alphabetically
        assert_eq!(plan.benchmarks.len(), 3);
        assert_eq!(plan.benchmarks[0].id, "a_bench");
        assert_eq!(plan.benchmarks[1].id, "b_bench");
        assert_eq!(plan.benchmarks[2].id, "c_bench");
    }

    #[test]
    fn test_group_filter() {
        let benches = [
            make_bench("bench1", "group_a", &[]),
            make_bench("bench2", "group_b", &[]),
            make_bench("bench3", "group_a", &[]),
        ];
        let refs: Vec<_> = benches
            .iter()
            .map(|b| unsafe { &*(b as *const _) })
            .collect();

        let plan = build_plan(refs, None, Some("group_a"), None, None);

        assert_eq!(plan.benchmarks.len(), 2);
        assert!(plan.benchmarks.iter().all(|b| b.group == "group_a"));
    }

    #[test]
    fn test_tag_filter() {
        let benches = [
            make_bench("bench1", "default", &["fast"]),
            make_bench("bench2", "default", &["slow"]),
            make_bench("bench3", "default", &["fast", "important"]),
        ];
        let refs: Vec<_> = benches
            .iter()
            .map(|b| unsafe { &*(b as *const _) })
            .collect();

        let plan = build_plan(refs, None, None, Some("fast"), None);

        assert_eq!(plan.benchmarks.len(), 2);
        assert!(plan.benchmarks.iter().all(|b| b.tags.contains(&"fast")));
    }

    #[test]
    fn test_skip_tag() {
        let benches = [
            make_bench("bench1", "default", &["fast"]),
            make_bench("bench2", "default", &["slow"]),
            make_bench("bench3", "default", &["fast", "skip_ci"]),
        ];
        let refs: Vec<_> = benches
            .iter()
            .map(|b| unsafe { &*(b as *const _) })
            .collect();

        let plan = build_plan(refs, None, None, None, Some("skip_ci"));

        assert_eq!(plan.benchmarks.len(), 2);
        assert!(plan.benchmarks.iter().all(|b| !b.tags.contains(&"skip_ci")));
    }
}
