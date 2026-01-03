//! Verification Execution
//!
//! Runs performance assertions with explicit status handling for missing dependencies.

use crate::MetricContext;
use fxhash::FxHashSet;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Severity levels for CI integration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Critical - regression fails the build
    Critical,
    /// Warning - logged but doesn't fail
    Warning,
    /// Informational only
    Info,
}

/// Verification definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verification {
    pub id: String,
    pub expression: String,
    pub severity: Severity,
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
    pub fn is_success(&self) -> bool {
        matches!(self, VerificationStatus::Passed)
    }

    pub fn is_failure(&self) -> bool {
        matches!(self, VerificationStatus::Failed)
    }

    pub fn is_actionable_failure(&self) -> bool {
        matches!(
            self,
            VerificationStatus::Failed | VerificationStatus::Error { .. }
        )
    }

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
    pub id: String,
    pub expression: String,
    pub status: VerificationStatus,
    pub actual_value: Option<f64>,
    pub severity: Severity,
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
    metrics: &'a MetricContext,
    /// Metrics that are unavailable (crashed/filtered benchmarks)
    unavailable: FxHashSet<String>,
}

impl<'a> VerificationContext<'a> {
    pub fn new(metrics: &'a MetricContext, unavailable: FxHashSet<String>) -> Self {
        Self {
            metrics,
            unavailable,
        }
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
}

/// Extract variable names from an evalexpr expression
fn extract_variables(expression: &str) -> Vec<String> {
    static IDENT_RE: OnceLock<Regex> = OnceLock::new();
    let re = IDENT_RE.get_or_init(|| Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b").unwrap());

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
                    severity: v.severity,
                    message: format!("Skipped: required metrics unavailable [{}]", missing),
                };
            }

            // Step 2: Evaluate the expression
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
                        severity: v.severity,
                        message: if passed {
                            format!("{} = {:.2}", v.expression, value)
                        } else {
                            format!("{} = {:.2} (expected non-zero)", v.expression, value)
                        },
                    }
                }
                Err(e) => VerificationResult {
                    id: v.id.clone(),
                    expression: v.expression.clone(),
                    status: VerificationStatus::Error {
                        message: e.to_string(),
                    },
                    actual_value: None,
                    severity: v.severity,
                    message: format!("Evaluation error: {}", e),
                },
            }
        })
        .collect()
}

/// Summary of verification results
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct VerificationSummary {
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub errors: usize,
    pub critical_failures: usize,
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
        assert!(!VerificationStatus::Skipped {
            missing_metrics: "x".to_string()
        }
        .affects_exit_code(Severity::Critical));
    }
}
