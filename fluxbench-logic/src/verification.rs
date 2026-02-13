//! Verification Execution
//!
//! Runs performance assertions with explicit status handling for missing dependencies.

use crate::MetricContext;
use fxhash::FxHashSet;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

// Re-export the single canonical Severity from fluxbench-core
pub use fluxbench_core::Severity;

/// A resolved metric value with its unit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedMetric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Unit (e.g., "ns", "req/s", "MB/s"). None means nanoseconds (default).
    pub unit: Option<String>,
}

/// Verification definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verification {
    /// Unique identifier for the verification
    pub id: String,
    /// Expression to evaluate (e.g., "(raw - overhead) < 50")
    pub expression: String,
    /// Severity level (Critical, Warning, Info)
    pub severity: Severity,
    /// Tolerance margin for numeric comparisons
    pub margin: f64,
}

/// Verification execution status with explicit states for all outcomes.
///
/// **Critical Design Decision**: Simple pass/fail boolean is insufficient.
/// We need explicit states to distinguish:
/// - Skipped: Dependency benchmark crashed or was filtered out
/// - Error: Expression evaluation failed (typo, type mismatch)
/// - Passed/Failed: Actual verification result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VerificationStatus {
    /// Verification passed (expression evaluated to non-zero/true)
    Passed,
    /// Verification failed (expression evaluated to zero/false)
    Failed,
    /// Verification could not run - dependency data missing
    Skipped {
        /// Which metric(s) were unavailable
        missing_metrics: String,
    },
    /// Verification encountered an error during evaluation
    Error {
        /// Error message
        message: String,
    },
}

impl VerificationStatus {
    /// Returns true if this is a Passed status.
    pub fn is_success(&self) -> bool {
        matches!(self, VerificationStatus::Passed)
    }

    /// Returns true if this is a Failed status.
    pub fn is_failure(&self) -> bool {
        matches!(self, VerificationStatus::Failed)
    }

    /// Returns true if this is either Failed or Error (requires action).
    pub fn is_actionable_failure(&self) -> bool {
        matches!(
            self,
            VerificationStatus::Failed | VerificationStatus::Error { .. }
        )
    }

    /// Returns true if this status should cause CI to fail given the severity level.
    ///
    /// Only critical failures and critical errors affect exit code; skipped verifications
    /// do not fail CI since the underlying dependency crash already did.
    pub fn affects_exit_code(&self, severity: Severity) -> bool {
        match (self, severity) {
            (VerificationStatus::Failed, Severity::Critical) => true,
            (VerificationStatus::Error { .. }, Severity::Critical) => true,
            // Skipped never fails CI - the dependency crash already did
            (VerificationStatus::Skipped { .. }, _) => false,
            _ => false,
        }
    }
}

/// Result of a verification check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Unique identifier for the verification
    pub id: String,
    /// Expression that was evaluated
    pub expression: String,
    /// Status of the verification (Passed, Failed, Skipped, or Error)
    pub status: VerificationStatus,
    /// Actual computed value from the expression, if available
    pub actual_value: Option<f64>,
    /// Resolved metric values used in the expression
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resolved_metrics: Vec<ResolvedMetric>,
    /// Severity level of the verification
    pub severity: Severity,
    /// Human-readable message describing the result
    pub message: String,
}

impl VerificationResult {
    /// Convenience method for backward compatibility
    pub fn passed(&self) -> bool {
        self.status.is_success()
    }
}

/// Context for verification with explicit missing metric tracking
pub struct VerificationContext<'a> {
    /// Reference to available metrics for expression evaluation
    metrics: &'a MetricContext,
    /// Metrics that are unavailable (crashed/filtered benchmarks)
    unavailable: FxHashSet<String>,
    /// Unit info for metrics (metric name → unit string). Absent means nanoseconds.
    metric_units: std::collections::HashMap<String, String>,
}

impl<'a> VerificationContext<'a> {
    /// Creates a new verification context with the given metrics and unavailable metric names.
    pub fn new(metrics: &'a MetricContext, unavailable: FxHashSet<String>) -> Self {
        Self {
            metrics,
            unavailable,
            metric_units: std::collections::HashMap::new(),
        }
    }

    /// Set the unit for a metric (e.g., "req/s", "MB/s")
    pub fn set_unit(&mut self, name: impl Into<String>, unit: impl Into<String>) {
        self.metric_units.insert(name.into(), unit.into());
    }

    /// Check if expression references any unavailable metrics
    pub fn check_dependencies(&self, expression: &str) -> Option<String> {
        let variables = extract_variables(expression);

        let missing: Vec<_> = variables
            .iter()
            .filter(|v| self.unavailable.contains(*v))
            .cloned()
            .collect();

        if missing.is_empty() {
            None
        } else {
            Some(missing.join(", "))
        }
    }

    /// Check if expression references any unknown variables (typos/renames).
    /// Returns `Some(list)` of unknown variable names that are neither in metrics,
    /// unavailable set, nor builtin functions.
    pub fn check_unknown_variables(&self, expression: &str) -> Option<Vec<String>> {
        let variables = extract_variables(expression);

        let unknown: Vec<String> = variables
            .into_iter()
            .filter(|v| {
                !self.metrics.has(v) && !self.unavailable.contains(v) && !is_builtin_function(v)
            })
            .collect();

        if unknown.is_empty() {
            None
        } else {
            Some(unknown)
        }
    }

    /// Get available metric names for error hints
    pub fn available_metric_names(&self) -> Vec<String> {
        self.metrics.metric_names().cloned().collect()
    }
}

/// Extract variable names from an evalexpr expression
fn extract_variables(expression: &str) -> Vec<String> {
    static IDENT_RE: OnceLock<Regex> = OnceLock::new();
    // Safety: this regex literal is guaranteed to compile
    let re = IDENT_RE
        .get_or_init(|| Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:@[a-zA-Z0-9_]+)*)\b").unwrap());

    re.captures_iter(expression)
        .map(|c| c[1].to_string())
        .filter(|s| !is_builtin_function(s))
        .collect()
}

fn is_builtin_function(name: &str) -> bool {
    matches!(
        name,
        "min" | "max" | "abs" | "floor" | "ceil" | "round" | "sqrt" | "if" | "len" | "str"
    )
}

/// Run all verifications
///
/// Evaluates each verification expression against available metrics. Handles missing dependencies,
/// unknown variables, and expression evaluation errors explicitly in the results.
///
/// # Arguments
/// * `verifications` - List of verification definitions to evaluate
/// * `context` - Context containing available metrics and tracking unavailable ones
///
/// # Returns
/// Vector of verification results with status and messages for each verification
pub fn run_verifications(
    verifications: &[Verification],
    context: &VerificationContext,
) -> Vec<VerificationResult> {
    verifications
        .iter()
        .map(|v| {
            // Step 1: Check if dependencies are available
            if let Some(missing) = context.check_dependencies(&v.expression) {
                return VerificationResult {
                    id: v.id.clone(),
                    expression: v.expression.clone(),
                    status: VerificationStatus::Skipped {
                        missing_metrics: missing.clone(),
                    },
                    actual_value: None,
                    resolved_metrics: Vec::new(),
                    severity: v.severity,
                    message: format!("Skipped: required metrics unavailable [{}]", missing),
                };
            }

            // Step 2: Check for unknown variables (typos/renames) before evaluation
            if let Some(unknown) = context.check_unknown_variables(&v.expression) {
                let mut available = context.available_metric_names();
                available.sort();
                return VerificationResult {
                    id: v.id.clone(),
                    expression: v.expression.clone(),
                    status: VerificationStatus::Error {
                        message: format!("unknown variable(s): {}", unknown.join(", ")),
                    },
                    actual_value: None,
                    resolved_metrics: Vec::new(),
                    severity: v.severity,
                    message: format!(
                        "Unknown variable '{}'. Available metrics: [{}]",
                        unknown.join("', '"),
                        available.join(", ")
                    ),
                };
            }

            // Resolve metric values used in this expression
            let vars = extract_variables(&v.expression);
            let resolved: Vec<ResolvedMetric> = vars
                .iter()
                .filter_map(|name| {
                    context.metrics.get(name).map(|val| ResolvedMetric {
                        name: name.clone(),
                        value: val,
                        unit: context.metric_units.get(name).cloned(),
                    })
                })
                .collect();

            // Step 3: Evaluate the expression
            match context.metrics.evaluate(&v.expression) {
                Ok(value) => {
                    let passed = value != 0.0;
                    VerificationResult {
                        id: v.id.clone(),
                        expression: v.expression.clone(),
                        status: if passed {
                            VerificationStatus::Passed
                        } else {
                            VerificationStatus::Failed
                        },
                        actual_value: Some(value),
                        resolved_metrics: resolved,
                        severity: v.severity,
                        message: if passed {
                            format!("{} = {:.2}", v.expression, value)
                        } else {
                            format!("{} = {:.2} (expected non-zero)", v.expression, value)
                        },
                    }
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    VerificationResult {
                        id: v.id.clone(),
                        expression: v.expression.clone(),
                        status: VerificationStatus::Error {
                            message: error_msg.clone(),
                        },
                        actual_value: None,
                        resolved_metrics: resolved,
                        severity: v.severity,
                        message: format!("Evaluation error: {}", error_msg),
                    }
                }
            }
        })
        .collect()
}

/// Summary of verification results
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct VerificationSummary {
    /// Number of verifications that passed
    pub passed: usize,
    /// Number of verifications that failed
    pub failed: usize,
    /// Number of verifications that were skipped (due to missing dependencies)
    pub skipped: usize,
    /// Number of verifications that had evaluation errors
    pub errors: usize,
    /// Number of critical verifications that failed
    pub critical_failures: usize,
    /// Number of critical verifications that had errors
    pub critical_errors: usize,
}

impl VerificationSummary {
    /// Should CI fail based on verification results?
    pub fn should_fail_ci(&self) -> bool {
        self.critical_failures > 0 || self.critical_errors > 0
    }

    /// Total verifications that ran (excludes skipped)
    pub fn total_executed(&self) -> usize {
        self.passed + self.failed + self.errors
    }
}

/// Aggregate verification results for CI reporting
///
/// Counts all verification statuses and identifies critical failures/errors that should
/// cause CI to fail.
///
/// # Arguments
/// * `results` - Verification results to summarize
///
/// # Returns
/// Summary with counts and critical status information
pub fn aggregate_verifications(results: &[VerificationResult]) -> VerificationSummary {
    let mut summary = VerificationSummary::default();

    for result in results {
        match &result.status {
            VerificationStatus::Passed => summary.passed += 1,
            VerificationStatus::Failed => {
                summary.failed += 1;
                if result.severity == Severity::Critical {
                    summary.critical_failures += 1;
                }
            }
            VerificationStatus::Skipped { .. } => summary.skipped += 1,
            VerificationStatus::Error { .. } => {
                summary.errors += 1;
                if result.severity == Severity::Critical {
                    summary.critical_errors += 1;
                }
            }
        }
    }

    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_variables() {
        let vars = extract_variables("(raw - overhead) < 50");
        assert!(vars.contains(&"raw".to_string()));
        assert!(vars.contains(&"overhead".to_string()));
        assert!(!vars.contains(&"min".to_string())); // builtin
    }

    #[test]
    fn test_verification_status_affects_exit() {
        assert!(VerificationStatus::Failed.affects_exit_code(Severity::Critical));
        assert!(!VerificationStatus::Failed.affects_exit_code(Severity::Warning));
        assert!(
            !VerificationStatus::Skipped {
                missing_metrics: "x".to_string()
            }
            .affects_exit_code(Severity::Critical)
        );
    }
}
