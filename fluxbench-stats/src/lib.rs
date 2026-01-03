//! FluxBench Statistical Engine
//!
//! Provides robust statistical analysis for benchmark results including:
//! - Bootstrap resampling with BCa (Bias-Corrected and Accelerated) intervals
//! - Outlier detection via IQR method
//! - Percentile calculation preserving tail latency signals
//! - Distribution characterization
//! - A/B comparison with effect size and probability of regression

mod bootstrap;
mod comparison;
mod outliers;
mod percentiles;
mod summary;

pub use bootstrap::{
    compute_bootstrap, BootstrapConfig, BootstrapMethod, BootstrapResult, ConfidenceInterval,
};
pub use comparison::{
    compare_distributions, ComparisonConfig, ComparisonError, ComparisonResult,
    EffectInterpretation,
};
pub use outliers::{detect_outliers, OutlierAnalysis, OutlierMethod};
pub use percentiles::{compute_percentile, compute_percentiles, Percentiles};
pub use summary::{compute_cycles_stats, compute_summary, CyclesStatistics, SummaryStatistics};

/// Threshold below which BCa method is used instead of percentile
pub const BCA_THRESHOLD: usize = 100;

/// Default number of bootstrap iterations
pub const DEFAULT_BOOTSTRAP_ITERATIONS: usize = 100_000;

/// Default confidence level (95%)
pub const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(BCA_THRESHOLD, 100);
        assert_eq!(DEFAULT_BOOTSTRAP_ITERATIONS, 100_000);
        assert!((DEFAULT_CONFIDENCE_LEVEL - 0.95).abs() < f64::EPSILON);
    }
}
