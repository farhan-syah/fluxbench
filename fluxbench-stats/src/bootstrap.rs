//! Bootstrap Resampling
//!
//! Implements both percentile and BCa (Bias-Corrected and Accelerated) bootstrap
//! methods for computing confidence intervals.

use crate::{BCA_THRESHOLD, DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_CONFIDENCE_LEVEL};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use thiserror::Error;

/// Bootstrap configuration
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// Number of bootstrap iterations (default: 100,000)
    pub iterations: usize,
    /// Confidence level (default: 0.95 for 95% CI)
    pub confidence_level: f64,
    /// Whether to use parallel computation
    pub parallel: bool,
    /// Force BCa method even for large samples
    pub force_bca: bool,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            iterations: DEFAULT_BOOTSTRAP_ITERATIONS,
            confidence_level: DEFAULT_CONFIDENCE_LEVEL,
            parallel: true,
            force_bca: false,
        }
    }
}

/// Which bootstrap method was used
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapMethod {
    /// Standard percentile method (for N >= 100)
    Percentile,
    /// BCa method (for small samples or when forced)
    Bca,
}

/// Confidence interval bounds
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub level: f64,
}

/// Result of bootstrap analysis
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Point estimate (sample mean)
    pub point_estimate: f64,
    /// Confidence interval
    pub confidence_interval: ConfidenceInterval,
    /// Standard error of the mean
    pub standard_error: f64,
    /// Which method was used
    pub method: BootstrapMethod,
    /// Warning message if any
    pub warning: Option<String>,
}

/// Errors that can occur during bootstrap
#[derive(Debug, Error)]
pub enum BootstrapError {
    #[error("Not enough samples: got {got}, need at least {min}")]
    NotEnoughSamples { got: usize, min: usize },

    #[error("Invalid confidence level: {0} (must be between 0 and 1)")]
    InvalidConfidenceLevel(f64),

    #[error("All samples have the same value")]
    NoVariance,
}

/// Compute bootstrap confidence interval for the mean
///
/// Automatically selects BCa method for small samples (N < 100).
pub fn compute_bootstrap(
    samples: &[f64],
    config: &BootstrapConfig,
) -> Result<BootstrapResult, BootstrapError> {
    // Validate inputs
    if samples.len() < 3 {
        return Err(BootstrapError::NotEnoughSamples {
            got: samples.len(),
            min: 3,
        });
    }

    if config.confidence_level <= 0.0 || config.confidence_level >= 1.0 {
        return Err(BootstrapError::InvalidConfidenceLevel(config.confidence_level));
    }

    let n = samples.len();
    let point_estimate = mean(samples);

    // Check for zero variance
    let variance = samples.iter().map(|x| (x - point_estimate).powi(2)).sum::<f64>() / n as f64;
    if variance == 0.0 {
        return Ok(BootstrapResult {
            point_estimate,
            confidence_interval: ConfidenceInterval {
                lower: point_estimate,
                upper: point_estimate,
                level: config.confidence_level,
            },
            standard_error: 0.0,
            method: BootstrapMethod::Percentile,
            warning: Some("All samples have identical values".to_string()),
        });
    }

    // Select method based on sample size
    let use_bca = config.force_bca || n < BCA_THRESHOLD;

    // Generate bootstrap distribution
    let bootstrap_means = if config.parallel {
        generate_bootstrap_means_parallel(samples, config.iterations)
    } else {
        generate_bootstrap_means_serial(samples, config.iterations)
    };

    // Compute confidence interval
    let (ci, method) = if use_bca {
        let ci = bca_interval(samples, &bootstrap_means, config.confidence_level);
        (ci, BootstrapMethod::Bca)
    } else {
        let ci = percentile_interval(&bootstrap_means, config.confidence_level);
        (ci, BootstrapMethod::Percentile)
    };

    // Compute standard error from bootstrap distribution
    let bootstrap_mean = mean(&bootstrap_means);
    let se = (bootstrap_means
        .iter()
        .map(|x| (x - bootstrap_mean).powi(2))
        .sum::<f64>()
        / bootstrap_means.len() as f64)
        .sqrt();

    let warning = if n < 10 {
        Some("Very small sample size may lead to unreliable estimates".to_string())
    } else {
        None
    };

    Ok(BootstrapResult {
        point_estimate,
        confidence_interval: ConfidenceInterval {
            lower: ci.0,
            upper: ci.1,
            level: config.confidence_level,
        },
        standard_error: se,
        method,
        warning,
    })
}

/// Generate bootstrap means using parallel iteration (Rayon)
fn generate_bootstrap_means_parallel(samples: &[f64], iterations: usize) -> Vec<f64> {
    (0..iterations)
        .into_par_iter()
        .map_init(
            thread_rng,
            |rng, _| {
                let mut sum = 0.0;
                for _ in 0..samples.len() {
                    sum += *samples.choose(rng).unwrap();
                }
                sum / samples.len() as f64
            },
        )
        .collect()
}

/// Generate bootstrap means serially (for testing or small samples)
fn generate_bootstrap_means_serial(samples: &[f64], iterations: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    (0..iterations)
        .map(|_| {
            let mut sum = 0.0;
            for _ in 0..samples.len() {
                sum += *samples.choose(&mut rng).unwrap();
            }
            sum / samples.len() as f64
        })
        .collect()
}

/// Standard percentile interval
fn percentile_interval(bootstrap_means: &[f64], confidence: f64) -> (f64, f64) {
    let mut sorted = bootstrap_means.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let alpha = (1.0 - confidence) / 2.0;

    let lower_idx = ((alpha * n as f64).floor() as usize).min(n - 1);
    let upper_idx = (((1.0 - alpha) * n as f64).floor() as usize).min(n - 1);

    (sorted[lower_idx], sorted[upper_idx])
}

/// BCa (Bias-Corrected and Accelerated) interval
///
/// More accurate for small samples and skewed distributions.
fn bca_interval(samples: &[f64], bootstrap_means: &[f64], confidence: f64) -> (f64, f64) {
    let n = samples.len();
    let b = bootstrap_means.len();

    let theta_hat = mean(samples);

    // Sort bootstrap means
    let mut sorted = bootstrap_means.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Bias correction factor (z0)
    let count_below = bootstrap_means.iter().filter(|&&x| x < theta_hat).count();
    let prop = count_below as f64 / b as f64;
    let z0 = normal_quantile(prop.clamp(0.0001, 0.9999));

    // Acceleration factor (a) via jackknife
    let jackknife_means: Vec<f64> = (0..n)
        .map(|i| {
            let sum: f64 = samples
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, &v)| v)
                .sum();
            sum / (n - 1) as f64
        })
        .collect();

    let jack_mean = mean(&jackknife_means);
    let numerator: f64 = jackknife_means
        .iter()
        .map(|x| (jack_mean - x).powi(3))
        .sum();
    let denominator: f64 = jackknife_means
        .iter()
        .map(|x| (jack_mean - x).powi(2))
        .sum();

    let a = if denominator.abs() < 1e-10 {
        0.0
    } else {
        numerator / (6.0 * denominator.powf(1.5))
    };

    // Adjusted percentiles
    let alpha = (1.0 - confidence) / 2.0;
    let z_alpha = normal_quantile(alpha);
    let z_1_alpha = normal_quantile(1.0 - alpha);

    let alpha1 = normal_cdf(z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha)));
    let alpha2 = normal_cdf(z0 + (z0 + z_1_alpha) / (1.0 - a * (z0 + z_1_alpha)));

    let lower_idx = ((alpha1 * b as f64).floor() as usize).clamp(0, b - 1);
    let upper_idx = ((alpha2 * b as f64).floor() as usize).clamp(0, b - 1);

    (sorted[lower_idx], sorted[upper_idx])
}

/// Compute mean of samples
fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Standard normal quantile (inverse CDF)
fn normal_quantile(p: f64) -> f64 {
    // Rational approximation for the normal quantile function
    // Abramowitz and Stegun approximation (26.2.23)
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let p = p.clamp(1e-10, 1.0 - 1e-10);

    let sign = if p < 0.5 { -1.0 } else { 1.0 };
    let p = if p < 0.5 { p } else { 1.0 - p };

    let t = (-2.0 * p.ln()).sqrt();

    // Coefficients for rational approximation
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    sign * x
}

/// Standard normal CDF
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation (7.1.26)
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_bootstrap() {
        let samples: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let config = BootstrapConfig {
            iterations: 1000,
            ..Default::default()
        };

        let result = compute_bootstrap(&samples, &config).unwrap();

        // Mean should be around 49.5
        assert!((result.point_estimate - 49.5).abs() < 0.1);

        // CI should contain the mean
        assert!(result.confidence_interval.lower < result.point_estimate);
        assert!(result.confidence_interval.upper > result.point_estimate);
    }

    #[test]
    fn test_bca_for_small_samples() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = BootstrapConfig {
            iterations: 1000,
            ..Default::default()
        };

        let result = compute_bootstrap(&samples, &config).unwrap();

        // Should use BCa for small sample
        assert_eq!(result.method, BootstrapMethod::Bca);
    }

    #[test]
    fn test_percentile_for_large_samples() {
        let samples: Vec<f64> = (0..200).map(|x| x as f64).collect();
        let config = BootstrapConfig {
            iterations: 1000,
            force_bca: false,
            ..Default::default()
        };

        let result = compute_bootstrap(&samples, &config).unwrap();

        // Should use percentile for large sample
        assert_eq!(result.method, BootstrapMethod::Percentile);
    }

    #[test]
    fn test_force_bca() {
        let samples: Vec<f64> = (0..200).map(|x| x as f64).collect();
        let config = BootstrapConfig {
            iterations: 1000,
            force_bca: true,
            ..Default::default()
        };

        let result = compute_bootstrap(&samples, &config).unwrap();

        // Should use BCa when forced
        assert_eq!(result.method, BootstrapMethod::Bca);
    }

    #[test]
    fn test_not_enough_samples() {
        let samples = vec![1.0, 2.0];
        let config = BootstrapConfig::default();

        let result = compute_bootstrap(&samples, &config);
        assert!(matches!(result, Err(BootstrapError::NotEnoughSamples { .. })));
    }

    #[test]
    fn test_normal_quantile() {
        // Test known values
        assert!((normal_quantile(0.5) - 0.0).abs() < 0.01);
        assert!((normal_quantile(0.975) - 1.96).abs() < 0.01);
        assert!((normal_quantile(0.025) - (-1.96)).abs() < 0.01);
    }

    #[test]
    fn test_normal_cdf() {
        // Test known values
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.01);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.01);
    }
}
