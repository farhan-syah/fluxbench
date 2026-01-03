//! Synthetic Metrics
//!
//! Computed metrics derived from benchmark results.

use crate::context::{ContextError, MetricContext};
use serde::{Deserialize, Serialize};

/// Definition of a synthetic metric registered via `#[flux::synthetic]`
#[derive(Debug, Clone)]
pub struct SyntheticDef {
    /// Unique identifier
    pub id: &'static str,
    /// Formula to compute the metric
    pub formula: &'static str,
    /// Unit for display (e.g., "ns", "MB/s")
    pub unit: Option<&'static str>,
}

/// Result of computing a synthetic metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticResult {
    /// Metric identifier
    pub id: String,
    /// Computed value
    pub value: f64,
    /// Formula used
    pub formula: String,
    /// Display unit
    pub unit: Option<String>,
}

/// Compute all synthetic metrics
pub fn compute_synthetics(
    synthetics: &[SyntheticDef],
    context: &MetricContext,
) -> Vec<Result<SyntheticResult, ContextError>> {
    synthetics
        .iter()
        .map(|s| {
            context.evaluate(s.formula).map(|value| SyntheticResult {
                id: s.id.to_string(),
                value,
                formula: s.formula.to_string(),
                unit: s.unit.map(|u| u.to_string()),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_computation() {
        let mut ctx = MetricContext::new();
        ctx.set("raw", 100.0);
        ctx.set("overhead", 20.0);

        let synthetics = vec![SyntheticDef {
            id: "net_time",
            formula: "raw - overhead",
            unit: Some("ns"),
        }];

        let results = compute_synthetics(&synthetics, &ctx);
        assert_eq!(results.len(), 1);

        let result = results[0].as_ref().unwrap();
        assert_eq!(result.id, "net_time");
        assert!((result.value - 80.0).abs() < f64::EPSILON);
    }
}
