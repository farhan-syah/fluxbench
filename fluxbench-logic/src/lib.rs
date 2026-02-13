#![warn(missing_docs)]
//! FluxBench Logic - Algebraic Verification Engine
//!
//! Evaluates performance assertions and synthetic metrics using mathematical expressions.
//! Provides verification rules (pass/fail/warn), synthetic metric computation, and dependency graphs.

mod context;
mod graph;
mod synthetic;
mod verification;

pub use context::MetricContext;
pub use graph::{DependencyGraph, GraphError};
pub use synthetic::{SyntheticDef, SyntheticResult, compute_synthetics};
pub use verification::{
    ResolvedMetric, Severity, Verification, VerificationContext, VerificationResult,
    VerificationStatus, VerificationSummary, aggregate_verifications, run_verifications,
};

/// Definition of a verification rule registered via `#[flux::verify]`
#[derive(Debug, Clone)]
pub struct VerifyDef {
    /// Unique identifier
    pub id: &'static str,
    /// Expression to evaluate
    pub expression: &'static str,
    /// Severity level
    pub severity: Severity,
    /// Tolerance margin for float comparison
    pub margin: f64,
}

// Collect all registered verifications and synthetics
inventory::collect!(VerifyDef);
inventory::collect!(SyntheticDef);
