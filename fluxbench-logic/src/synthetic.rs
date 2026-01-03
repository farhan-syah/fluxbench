//! Synthetic Metrics
//!
//! Computed metrics derived from benchmark results.

use crate::context::{ContextError, MetricContext};

/// Definition of a synthetic metric
#[derive(Debug, Clone)]
pub struct SyntheticDef {
    /// Unique identifier
    pub id: String,
    /// Formula to compute the metric
    pub formula: String,
    /// Unit for display (e.g., "ns", "MB/s")
    pub unit: Option<String>,
}

/// Result of computing a synthetic metric
#[derive(Debug, Clone)]
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
#[allow(dead_code)]
pub fn compute_synthetics(
    synthetics: &[SyntheticDef],
    context: &MetricContext,
) -> Vec<Result<SyntheticResult, ContextError>> {
    synthetics
        .iter()
        .map(|s| {
            context.evaluate(&s.formula).map(|value| SyntheticResult {
                id: s.id.clone(),
                value,
                formula: s.formula.clone(),
                unit: s.unit.clone(),
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
            id: "net_time".to_string(),
            formula: "raw - overhead".to_string(),
            unit: Some("ns".to_string()),
        }];

        let results = compute_synthetics(&synthetics, &ctx);
        assert_eq!(results.len(), 1);

        let result = results[0].as_ref().unwrap();
        assert_eq!(result.id, "net_time");
        assert!((result.value - 80.0).abs() < f64::EPSILON);
    }
}
