//! Metric Context
//!
//! Provides variable bindings for expression evaluation.

use evalexpr::{
    ContextWithMutableVariables, EvalexprError, HashMapContext, Value, eval_with_context,
};
use fxhash::FxHashMap;
use thiserror::Error;

/// Errors from metric evaluation
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ContextError {
    #[error("Unknown metric: {0}")]
    UnknownMetric(String),

    #[error("Evaluation error: {0}")]
    EvalError(String),
}

/// Context holding benchmark metrics for expression evaluation
#[derive(Debug, Clone)]
pub struct MetricContext {
    metrics: FxHashMap<String, f64>,
}

impl MetricContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self {
            metrics: FxHashMap::default(),
        }
    }

    /// Add a metric value
    pub fn set(&mut self, name: impl Into<String>, value: f64) {
        self.metrics.insert(name.into(), value);
    }

    /// Get a metric value
    pub fn get(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    /// Evaluate an expression in this context
    pub fn evaluate(&self, expression: &str) -> Result<f64, ContextError> {
        let mut ctx = HashMapContext::new();

        // Add all metrics to evalexpr context
        // Sanitize `@` to `__` since evalexpr only supports identifier chars
        for (name, value) in &self.metrics {
            let safe_name = name.replace('@', "__");
            ctx.set_value(safe_name, Value::Float(*value))
                .map_err(|e: EvalexprError| ContextError::EvalError(e.to_string()))?;
        }

        // Sanitize expression: replace `@` in identifiers with `__`
        let expression = expression.replace('@', "__");

        // Evaluate
        let result = eval_with_context(&expression, &ctx)
            .map_err(|e| ContextError::EvalError(e.to_string()))?;

        // Convert to f64
        match result {
            Value::Float(f) => Ok(f),
            Value::Int(i) => Ok(i as f64),
            Value::Boolean(b) => Ok(if b { 1.0 } else { 0.0 }),
            other => Err(ContextError::EvalError(format!(
                "Expected numeric result, got {:?}",
                other
            ))),
        }
    }

    /// Check if a metric exists
    pub fn has(&self, name: &str) -> bool {
        self.metrics.contains_key(name)
    }

    /// List all metric names
    pub fn metric_names(&self) -> impl Iterator<Item = &String> {
        self.metrics.keys()
    }
}

impl Default for MetricContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_evaluation() {
        let mut ctx = MetricContext::new();
        ctx.set("x", 10.0);
        ctx.set("y", 5.0);

        let result = ctx.evaluate("x + y").unwrap();
        assert!((result - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_comparison() {
        let mut ctx = MetricContext::new();
        ctx.set("latency", 100.0);
        ctx.set("threshold", 200.0);

        let result = ctx.evaluate("latency < threshold").unwrap();
        assert!((result - 1.0).abs() < f64::EPSILON); // true = 1.0
    }

    #[test]
    fn test_complex_expression() {
        let mut ctx = MetricContext::new();
        ctx.set("raw", 150.0);
        ctx.set("overhead", 50.0);

        let result = ctx.evaluate("(raw - overhead) < 200").unwrap();
        assert!((result - 1.0).abs() < f64::EPSILON);
    }
}
