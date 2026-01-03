//! Percentile Computation
//!
//! Computes percentiles from raw samples (NOT cleaned data).
//! Tail latency percentiles (p99, p999) must include outliers as they ARE the signal.

/// Standard percentiles to compute
#[derive(Debug, Clone)]
pub struct Percentiles {
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub p999: f64,
}

/// Compute a single percentile from samples
///
/// Uses linear interpolation between nearest ranks.
pub fn compute_percentile(samples: &[f64], percentile: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    if samples.len() == 1 {
        return samples[0];
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let p = percentile / 100.0;

    // Linear interpolation between nearest ranks
    let rank = p * (n - 1) as f64;
    let lower_idx = rank.floor() as usize;
    let upper_idx = (lower_idx + 1).min(n - 1);
    let fraction = rank - lower_idx as f64;

    sorted[lower_idx] + fraction * (sorted[upper_idx] - sorted[lower_idx])
}

/// Compute all standard percentiles
pub fn compute_percentiles(samples: &[f64]) -> Percentiles {
    Percentiles {
        p50: compute_percentile(samples, 50.0),
        p75: compute_percentile(samples, 75.0),
        p90: compute_percentile(samples, 90.0),
        p95: compute_percentile(samples, 95.0),
        p99: compute_percentile(samples, 99.0),
        p999: compute_percentile(samples, 99.9),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p50 = compute_percentile(&samples, 50.0);
        assert!((p50 - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_quartiles() {
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let p25 = compute_percentile(&samples, 25.0);
        let p75 = compute_percentile(&samples, 75.0);

        assert!((p25 - 25.75).abs() < 1.0);
        assert!((p75 - 75.25).abs() < 1.0);
    }

    #[test]
    fn test_extreme_percentiles() {
        let samples: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
        let p99 = compute_percentile(&samples, 99.0);
        let p999 = compute_percentile(&samples, 99.9);

        assert!(p99 > 985.0 && p99 < 995.0);
        assert!(p999 > 998.0 && p999 <= 1000.0);
    }

    #[test]
    fn test_single_sample() {
        let samples = vec![42.0];
        let p50 = compute_percentile(&samples, 50.0);
        assert!((p50 - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_samples() {
        let samples: Vec<f64> = Vec::new();
        let p50 = compute_percentile(&samples, 50.0);
        assert!((p50 - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_compute_all_percentiles() {
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let percentiles = compute_percentiles(&samples);

        assert!(percentiles.p50 > 49.0 && percentiles.p50 < 51.0);
        assert!(percentiles.p90 > 89.0 && percentiles.p90 < 91.0);
        assert!(percentiles.p99 > 98.0 && percentiles.p99 < 100.0);
    }
}
