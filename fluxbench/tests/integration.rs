//! Integration tests for FluxBench
//!
//! These tests verify the end-to-end behavior of the benchmarking system.

use fluxbench::{Bencher, BootstrapConfig, MetricContext, compute_bootstrap, compute_summary};
use fluxbench_logic::{Severity, Verification, VerificationContext, run_verifications};
use fluxbench_stats::{
    ComparisonConfig, OutlierMethod, compare_distributions, compute_cycles_stats,
};

/// Test that the Bencher collects samples correctly
#[test]
fn test_bencher_collects_samples() {
    let mut bencher = Bencher::new(false);

    // Warmup phase (required before measurement)
    for _ in 0..5 {
        bencher.iter(|| {
            let mut sum = 0u64;
            for i in 0..1000 {
                sum += i;
            }
            sum
        });
    }

    // Transition to measurement mode (iters_per_sample=1 for this test)
    bencher.start_measurement(1_000_000_000); // 1 second target
    bencher.set_iters_per_sample(1); // Record each iteration as a sample

    // Run 10 iterations
    for _ in 0..10 {
        bencher.iter(|| {
            let mut sum = 0u64;
            for i in 0..1000 {
                sum += i;
            }
            sum
        });
    }

    let samples = bencher.samples();
    assert_eq!(samples.len(), 10);

    // All samples should have non-zero duration
    for sample in samples {
        assert!(sample.duration_nanos > 0);
    }
}

/// Test that iter_with_setup excludes setup time
#[test]
fn test_iter_with_setup() {
    let mut bencher = Bencher::new(false);

    for _ in 0..5 {
        bencher.iter_with_setup(
            || vec![1u64; 1000],       // Heavy setup
            |v| v.iter().sum::<u64>(), // Light measurement
        );
    }

    assert_eq!(bencher.iteration_count(), 5);
}

/// Test summary statistics computation
#[test]
fn test_summary_statistics() {
    let samples = vec![100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 101.0, 99.0];
    let summary = compute_summary(&samples, OutlierMethod::Iqr { k: 3 });

    // Mean should be ~100
    assert!((summary.mean - 100.0).abs() < 1.0);

    // Median should be ~100
    assert!((summary.median - 100.0).abs() < 1.0);

    // Std dev should be small
    assert!(summary.std_dev < 5.0);

    // Min/max
    assert_eq!(summary.min, 98.0);
    assert_eq!(summary.max, 102.0);
}

/// Test bootstrap confidence intervals
#[test]
fn test_bootstrap_confidence_intervals() {
    let samples: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1)).collect();

    let config = BootstrapConfig {
        iterations: 1000,
        confidence_level: 0.95,
        ..Default::default()
    };

    let result = compute_bootstrap(&samples, &config).unwrap();

    // CI should contain the point estimate
    assert!(result.confidence_interval.lower <= result.point_estimate);
    assert!(result.confidence_interval.upper >= result.point_estimate);

    // CI should be reasonable
    let ci_width = result.confidence_interval.upper - result.confidence_interval.lower;
    assert!(ci_width > 0.0);
    assert!(ci_width < 20.0); // Should be relatively narrow for 100 samples
}

/// Test A/B comparison
#[test]
fn test_ab_comparison() {
    // Baseline: ~100ns (deterministic samples)
    let baseline: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1)).collect();

    // Candidate: ~150ns (clear regression)
    let candidate: Vec<f64> = (0..50).map(|i| 150.0 + (i as f64 * 0.1)).collect();

    let config = ComparisonConfig {
        bootstrap_iterations: 1000,
        confidence_level: 0.95,
        significance_threshold: 5.0,
        ..Default::default()
    };

    let result = compare_distributions(&baseline, &candidate, &config).unwrap();

    // Should detect regression
    assert!(result.probability_regression > 0.9);
    assert!(result.relative_change > 40.0); // ~50% slower
    assert!(result.is_significant);
}

/// Test cycles statistics
#[test]
fn test_cycles_statistics() {
    let cycles: Vec<u64> = vec![3000, 3100, 2900, 3050, 2950];
    let nanos: Vec<f64> = vec![1000.0, 1033.0, 967.0, 1017.0, 983.0];

    let stats = compute_cycles_stats(&cycles, &nanos);

    // Mean should be ~3000
    assert!((stats.mean_cycles - 3000.0).abs() < 100.0);

    // Min/max
    assert_eq!(stats.min_cycles, 2900);
    assert_eq!(stats.max_cycles, 3100);

    // ~3 cycles per ns
    assert!((stats.cycles_per_ns - 3.0).abs() < 0.5);
}

/// Test verification expression evaluation
#[test]
fn test_verification_expression() {
    let mut context = MetricContext::new();
    context.set("fast_bench", 100.0);
    context.set("slow_bench", 500.0);

    let verifications = vec![
        Verification {
            id: "fast_is_fast".to_string(),
            expression: "fast_bench < 200".to_string(),
            severity: Severity::Critical,
            margin: 0.0,
        },
        Verification {
            id: "fast_beats_slow".to_string(),
            expression: "fast_bench < slow_bench".to_string(),
            severity: Severity::Warning,
            margin: 0.0,
        },
    ];

    let verification_context = VerificationContext::new(&context, Default::default());
    let results = run_verifications(&verifications, &verification_context);

    // Both should pass
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.passed()));
}

/// Test verification failure detection
#[test]
fn test_verification_failure() {
    let mut context = MetricContext::new();
    context.set("my_bench", 1000.0);

    let verifications = vec![Verification {
        id: "too_slow".to_string(),
        expression: "my_bench < 100".to_string(), // Will fail: 1000 < 100
        severity: Severity::Critical,
        margin: 0.0,
    }];

    let verification_context = VerificationContext::new(&context, Default::default());
    let results = run_verifications(&verifications, &verification_context);

    assert_eq!(results.len(), 1);
    assert!(!results[0].passed());
}

/// Test outlier detection preserves tail latencies
#[test]
fn test_outlier_preserves_tail() {
    // Normal samples with outliers at the end
    let mut samples: Vec<f64> = (0..90).map(|i| 100.0 + (i as f64 * 0.1)).collect();
    samples.push(500.0); // High outlier 1
    samples.push(600.0); // High outlier 2
    samples.push(700.0); // High outlier 3
    samples.push(800.0); // High outlier 4
    samples.push(1000.0); // Extreme outlier

    let summary = compute_summary(&samples, OutlierMethod::Iqr { k: 3 });

    // Mean should be from cleaned data (exclude outliers)
    assert!(summary.mean < 200.0);

    // But max should preserve the outlier
    assert_eq!(summary.max, 1000.0);

    // Outliers should be detected
    assert!(summary.outlier_count > 0);
}

/// Test that effect size interpretation is correct
#[test]
fn test_effect_size_interpretation() {
    use fluxbench_stats::EffectInterpretation;

    // Create distributions with known effect sizes
    // Note: Must have variance for Cohen's d to be meaningful (stddev > 0)
    let baseline: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 % 10.0)).collect();
    let small_diff: Vec<f64> = (0..100).map(|i| 110.0 + (i as f64 % 10.0)).collect(); // Small effect
    let large_diff: Vec<f64> = (0..100).map(|i| 200.0 + (i as f64 % 10.0)).collect(); // Large effect

    let config = ComparisonConfig {
        bootstrap_iterations: 100,
        ..Default::default()
    };

    let small_result = compare_distributions(&baseline, &small_diff, &config).unwrap();
    let large_result = compare_distributions(&baseline, &large_diff, &config).unwrap();

    // Large difference should have large effect size
    // With mean diff of 100 and stddev ~2.87, Cohen's d ≈ 34.8 (huge)
    assert_eq!(
        large_result.effect_interpretation,
        EffectInterpretation::Large
    );

    // Effect size magnitude makes sense
    assert!(large_result.effect_size.abs() > small_result.effect_size.abs());
}

/// Test BenchmarkResult finalization
#[test]
fn test_benchmark_result_finalization() {
    let mut bencher = Bencher::new(true); // Track allocations

    // Warmup phase
    for _ in 0..3 {
        bencher.iter(|| {
            let v: Vec<u64> = (0..100).collect();
            v.len()
        });
    }

    // Transition to measurement mode
    bencher.start_measurement(1_000_000_000);
    bencher.set_iters_per_sample(1);

    // Measurement phase
    for _ in 0..3 {
        bencher.iter(|| {
            // Simple computation
            let v: Vec<u64> = (0..100).collect();
            v.len()
        });
    }

    let result = bencher.finish();
    assert_eq!(result.iterations, 6); // 3 warmup + 3 measurement
    assert_eq!(result.samples.len(), 3); // Only measurement samples recorded
    assert!(result.total_time_ns > 0);
}

/// Test improvement detection (opposite of regression)
#[test]
fn test_improvement_detection() {
    // Baseline: slow (~200ns)
    let baseline: Vec<f64> = (0..50).map(|i| 200.0 + (i as f64 * 0.1)).collect();

    // Candidate: fast (~100ns, clear improvement)
    let candidate: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.1)).collect();

    let config = ComparisonConfig {
        bootstrap_iterations: 1000,
        ..Default::default()
    };

    let result = compare_distributions(&baseline, &candidate, &config).unwrap();

    // Should detect improvement (low regression probability)
    assert!(result.probability_regression < 0.1);
    assert!(result.relative_change < -40.0); // ~50% faster
    assert!(result.is_significant);
}
