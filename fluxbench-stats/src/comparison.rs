//! A/B Comparison Statistics
//!
//! Provides statistical comparison between baseline and candidate distributions
//! using bootstrap resampling to compute probability of regression.

use crate::outliers::OutlierMethod;
use crate::summary::{SummaryStatistics, compute_summary};
use rand::prelude::*;
use rayon::prelude::*;

/// Result of comparing two distributions
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Summary statistics for baseline
    pub baseline_stats: SummaryStatistics,
    /// Summary statistics for candidate
    pub candidate_stats: SummaryStatistics,
    /// Relative change: (candidate - baseline) / baseline
    pub relative_change: f64,
    /// Absolute change in nanoseconds
    pub absolute_change: f64,
    /// Probability that candidate is slower than baseline (0.0 to 1.0)
    pub probability_regression: f64,
    /// Confidence interval of the difference
    pub difference_ci_lower: f64,
    /// Confidence interval of the difference
    pub difference_ci_upper: f64,
    /// Whether the difference is statistically significant
    pub is_significant: bool,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
    /// Effect size interpretation
    pub effect_interpretation: EffectInterpretation,
}

/// Interpretation of effect size magnitude
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectInterpretation {
    /// |d| < 0.2 - negligible difference
    Negligible,
    /// 0.2 <= |d| < 0.5 - small difference
    Small,
    /// 0.5 <= |d| < 0.8 - medium difference
    Medium,
    /// |d| >= 0.8 - large difference
    Large,
}

impl std::fmt::Display for EffectInterpretation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EffectInterpretation::Negligible => write!(f, "negligible"),
            EffectInterpretation::Small => write!(f, "small"),
            EffectInterpretation::Medium => write!(f, "medium"),
            EffectInterpretation::Large => write!(f, "large"),
        }
    }
}

/// Configuration for comparison
#[derive(Debug, Clone)]
pub struct ComparisonConfig {
    /// Number of bootstrap iterations
    pub bootstrap_iterations: usize,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Threshold for significance (relative change %)
    pub significance_threshold: f64,
    /// Outlier detection method
    pub outlier_method: OutlierMethod,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            bootstrap_iterations: 10_000,
            confidence_level: 0.95,
            significance_threshold: 5.0, // 5% change threshold
            outlier_method: OutlierMethod::default(),
        }
    }
}

/// Compare two distributions using bootstrap resampling
///
/// Returns the probability that `candidate` is slower than `baseline`,
/// along with effect size and confidence intervals for the difference.
pub fn compare_distributions(
    baseline: &[f64],
    candidate: &[f64],
    config: &ComparisonConfig,
) -> Result<ComparisonResult, ComparisonError> {
    // Validate inputs
    if baseline.is_empty() {
        return Err(ComparisonError::EmptyBaseline);
    }
    if candidate.is_empty() {
        return Err(ComparisonError::EmptyCandidate);
    }
    if baseline.len() < 2 {
        return Err(ComparisonError::InsufficientBaseline);
    }
    if candidate.len() < 2 {
        return Err(ComparisonError::InsufficientCandidate);
    }

    // Compute summary statistics for both
    let baseline_stats = compute_summary(baseline, config.outlier_method);
    let candidate_stats = compute_summary(candidate, config.outlier_method);

    // Compute observed difference
    let observed_diff = candidate_stats.mean - baseline_stats.mean;
    let relative_change = if baseline_stats.mean > 0.0 {
        (observed_diff / baseline_stats.mean) * 100.0
    } else {
        0.0
    };

    // Bootstrap the difference of means
    let bootstrap_diffs: Vec<f64> = (0..config.bootstrap_iterations)
        .into_par_iter()
        .map_init(thread_rng, |rng, _| {
            // Resample baseline
            let baseline_mean: f64 = (0..baseline.len())
                .map(|_| baseline[rng.gen_range(0..baseline.len())])
                .sum::<f64>()
                / baseline.len() as f64;

            // Resample candidate
            let candidate_mean: f64 = (0..candidate.len())
                .map(|_| candidate[rng.gen_range(0..candidate.len())])
                .sum::<f64>()
                / candidate.len() as f64;

            candidate_mean - baseline_mean
        })
        .collect();

    // Probability of regression (candidate slower = positive difference)
    let regressions = bootstrap_diffs.iter().filter(|&&d| d > 0.0).count();
    let probability_regression = regressions as f64 / config.bootstrap_iterations as f64;

    // Confidence interval of the difference
    let mut sorted_diffs = bootstrap_diffs.clone();
    sorted_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - config.confidence_level;
    let lower_idx = (alpha / 2.0 * config.bootstrap_iterations as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * config.bootstrap_iterations as f64) as usize;
    let difference_ci_lower = sorted_diffs[lower_idx];
    let difference_ci_upper = sorted_diffs[upper_idx.min(sorted_diffs.len() - 1)];

    // Effect size (Cohen's d)
    // Pooled standard deviation
    let n1 = baseline.len() as f64;
    let n2 = candidate.len() as f64;
    let var1 = baseline_stats.std_dev.powi(2);
    let var2 = candidate_stats.std_dev.powi(2);
    let pooled_std = ((((n1 - 1.0) * var1) + ((n2 - 1.0) * var2)) / (n1 + n2 - 2.0)).sqrt();

    let effect_size = if pooled_std > 0.0 {
        observed_diff / pooled_std
    } else {
        0.0
    };

    let effect_interpretation = interpret_effect_size(effect_size);

    // Significance: CI doesn't include zero AND change exceeds threshold
    let ci_excludes_zero = (difference_ci_lower > 0.0) || (difference_ci_upper < 0.0);
    let exceeds_threshold = relative_change.abs() >= config.significance_threshold;
    let is_significant = ci_excludes_zero && exceeds_threshold;

    Ok(ComparisonResult {
        baseline_stats,
        candidate_stats,
        relative_change,
        absolute_change: observed_diff,
        probability_regression,
        difference_ci_lower,
        difference_ci_upper,
        is_significant,
        effect_size,
        effect_interpretation,
    })
}

/// Interpret effect size magnitude using Cohen's conventions
fn interpret_effect_size(d: f64) -> EffectInterpretation {
    let abs_d = d.abs();
    if abs_d < 0.2 {
        EffectInterpretation::Negligible
    } else if abs_d < 0.5 {
        EffectInterpretation::Small
    } else if abs_d < 0.8 {
        EffectInterpretation::Medium
    } else {
        EffectInterpretation::Large
    }
}

/// Errors from comparison operations
#[derive(Debug, Clone, thiserror::Error)]
#[non_exhaustive]
pub enum ComparisonError {
    #[error("Baseline samples are empty")]
    EmptyBaseline,
    #[error("Candidate samples are empty")]
    EmptyCandidate,
    #[error("Baseline needs at least 2 samples")]
    InsufficientBaseline,
    #[error("Candidate needs at least 2 samples")]
    InsufficientCandidate,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_identical() {
        let samples = vec![100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 101.0, 99.0];
        let config = ComparisonConfig {
            bootstrap_iterations: 1000,
            ..Default::default()
        };

        let result = compare_distributions(&samples, &samples, &config).unwrap();

        // Should have ~50% probability of regression (random noise)
        assert!(result.probability_regression > 0.3 && result.probability_regression < 0.7);
        assert!(result.relative_change.abs() < 1.0);
        assert!(!result.is_significant);
        assert_eq!(
            result.effect_interpretation,
            EffectInterpretation::Negligible
        );
    }

    #[test]
    fn test_compare_clear_regression() {
        let baseline = vec![100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 101.0, 99.0];
        let candidate = vec![200.0, 202.0, 198.0, 201.0, 199.0, 200.0, 201.0, 199.0];
        let config = ComparisonConfig {
            bootstrap_iterations: 1000,
            ..Default::default()
        };

        let result = compare_distributions(&baseline, &candidate, &config).unwrap();

        // Clear regression - probability should be very high
        assert!(result.probability_regression > 0.95);
        assert!(result.relative_change > 90.0); // ~100% regression
        assert!(result.is_significant);
        assert_eq!(result.effect_interpretation, EffectInterpretation::Large);
    }

    #[test]
    fn test_compare_clear_improvement() {
        let baseline = vec![200.0, 202.0, 198.0, 201.0, 199.0, 200.0, 201.0, 199.0];
        let candidate = vec![100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 101.0, 99.0];
        let config = ComparisonConfig {
            bootstrap_iterations: 1000,
            ..Default::default()
        };

        let result = compare_distributions(&baseline, &candidate, &config).unwrap();

        // Clear improvement - probability should be very low
        assert!(result.probability_regression < 0.05);
        assert!(result.relative_change < -40.0); // ~50% improvement
        assert!(result.is_significant);
        assert_eq!(result.effect_interpretation, EffectInterpretation::Large);
    }

    #[test]
    fn test_effect_size_interpretation() {
        assert_eq!(interpret_effect_size(0.1), EffectInterpretation::Negligible);
        assert_eq!(interpret_effect_size(0.3), EffectInterpretation::Small);
        assert_eq!(interpret_effect_size(0.6), EffectInterpretation::Medium);
        assert_eq!(interpret_effect_size(1.0), EffectInterpretation::Large);
        assert_eq!(interpret_effect_size(-0.5), EffectInterpretation::Medium);
    }

    #[test]
    fn test_empty_samples() {
        let config = ComparisonConfig::default();

        assert!(matches!(
            compare_distributions(&[], &[1.0, 2.0], &config),
            Err(ComparisonError::EmptyBaseline)
        ));
        assert!(matches!(
            compare_distributions(&[1.0, 2.0], &[], &config),
            Err(ComparisonError::EmptyCandidate)
        ));
    }

    #[test]
    fn test_insufficient_samples() {
        let config = ComparisonConfig::default();

        assert!(matches!(
            compare_distributions(&[1.0], &[1.0, 2.0], &config),
            Err(ComparisonError::InsufficientBaseline)
        ));
        assert!(matches!(
            compare_distributions(&[1.0, 2.0], &[1.0], &config),
            Err(ComparisonError::InsufficientCandidate)
        ));
    }
}
