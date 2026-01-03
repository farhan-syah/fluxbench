//! Outlier Detection
//!
//! Uses IQR (Interquartile Range) method to identify outliers.
//!
//! **Critical Design Decision**: Outliers are detected but NOT removed from
//! percentile/min/max calculations. For tail latency metrics, outliers ARE the signal.
//! Only mean/median/stddev use cleaned data.

use crate::percentiles::compute_percentile;

/// Method for outlier detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutlierMethod {
    /// IQR method: outliers are outside [Q1 - k*IQR, Q3 + k*IQR]
    Iqr { k: u32 }, // k is multiplied by 0.5, so k=3 means 1.5*IQR
    /// Z-score method: outliers are beyond z standard deviations
    ZScore { threshold: u32 }, // threshold is multiplied by 0.5
    /// No outlier detection
    None,
}

impl Default for OutlierMethod {
    fn default() -> Self {
        // Standard IQR with k=1.5
        OutlierMethod::Iqr { k: 3 }
    }
}

/// Result of outlier analysis
#[derive(Debug, Clone)]
pub struct OutlierAnalysis {
    /// Original samples (ALL data preserved)
    pub all_samples: Vec<f64>,
    /// Samples with outliers removed (for mean/stddev computation)
    pub cleaned_samples: Vec<f64>,
    /// Indices of outlier samples
    pub outlier_indices: Vec<usize>,
    /// Number of low outliers (below lower bound)
    pub low_outlier_count: usize,
    /// Number of high outliers (above upper bound)
    pub high_outlier_count: usize,
    /// Lower bound used for detection
    pub lower_bound: f64,
    /// Upper bound used for detection
    pub upper_bound: f64,
    /// Detection method used
    pub method: OutlierMethod,
}

impl OutlierAnalysis {
    /// Percentage of samples that are outliers
    pub fn outlier_percentage(&self) -> f64 {
        if self.all_samples.is_empty() {
            return 0.0;
        }
        (self.outlier_indices.len() as f64 / self.all_samples.len() as f64) * 100.0
    }

    /// Check if outlier percentage exceeds threshold (indicates noisy environment)
    pub fn is_noisy(&self, threshold_pct: f64) -> bool {
        self.outlier_percentage() > threshold_pct
    }
}

/// Detect outliers in samples using specified method
pub fn detect_outliers(samples: &[f64], method: OutlierMethod) -> OutlierAnalysis {
    if samples.is_empty() {
        return OutlierAnalysis {
            all_samples: Vec::new(),
            cleaned_samples: Vec::new(),
            outlier_indices: Vec::new(),
            low_outlier_count: 0,
            high_outlier_count: 0,
            lower_bound: 0.0,
            upper_bound: 0.0,
            method,
        };
    }

    match method {
        OutlierMethod::None => OutlierAnalysis {
            all_samples: samples.to_vec(),
            cleaned_samples: samples.to_vec(),
            outlier_indices: Vec::new(),
            low_outlier_count: 0,
            high_outlier_count: 0,
            lower_bound: f64::NEG_INFINITY,
            upper_bound: f64::INFINITY,
            method,
        },
        OutlierMethod::Iqr { k } => detect_iqr_outliers(samples, k as f64 * 0.5),
        OutlierMethod::ZScore { threshold } => {
            detect_zscore_outliers(samples, threshold as f64 * 0.5)
        }
    }
}

/// IQR-based outlier detection
fn detect_iqr_outliers(samples: &[f64], k: f64) -> OutlierAnalysis {
    let q1 = compute_percentile(samples, 25.0);
    let q3 = compute_percentile(samples, 75.0);
    let iqr = q3 - q1;

    let lower_bound = q1 - k * iqr;
    let upper_bound = q3 + k * iqr;

    let mut outlier_indices = Vec::new();
    let mut low_count = 0;
    let mut high_count = 0;
    let mut cleaned = Vec::with_capacity(samples.len());

    for (i, &sample) in samples.iter().enumerate() {
        if sample < lower_bound {
            outlier_indices.push(i);
            low_count += 1;
        } else if sample > upper_bound {
            outlier_indices.push(i);
            high_count += 1;
        } else {
            cleaned.push(sample);
        }
    }

    OutlierAnalysis {
        all_samples: samples.to_vec(),
        cleaned_samples: cleaned,
        outlier_indices,
        low_outlier_count: low_count,
        high_outlier_count: high_count,
        lower_bound,
        upper_bound,
        method: OutlierMethod::Iqr {
            k: (k * 2.0) as u32,
        },
    }
}

/// Z-score based outlier detection
fn detect_zscore_outliers(samples: &[f64], threshold: f64) -> OutlierAnalysis {
    let n = samples.len() as f64;
    let mean: f64 = samples.iter().sum::<f64>() / n;
    let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        // No variance, no outliers
        return OutlierAnalysis {
            all_samples: samples.to_vec(),
            cleaned_samples: samples.to_vec(),
            outlier_indices: Vec::new(),
            low_outlier_count: 0,
            high_outlier_count: 0,
            lower_bound: mean,
            upper_bound: mean,
            method: OutlierMethod::ZScore {
                threshold: (threshold * 2.0) as u32,
            },
        };
    }

    let lower_bound = mean - threshold * std_dev;
    let upper_bound = mean + threshold * std_dev;

    let mut outlier_indices = Vec::new();
    let mut low_count = 0;
    let mut high_count = 0;
    let mut cleaned = Vec::with_capacity(samples.len());

    for (i, &sample) in samples.iter().enumerate() {
        let z_score = (sample - mean) / std_dev;
        if z_score < -threshold {
            outlier_indices.push(i);
            low_count += 1;
        } else if z_score > threshold {
            outlier_indices.push(i);
            high_count += 1;
        } else {
            cleaned.push(sample);
        }
    }

    OutlierAnalysis {
        all_samples: samples.to_vec(),
        cleaned_samples: cleaned,
        outlier_indices,
        low_outlier_count: low_count,
        high_outlier_count: high_count,
        lower_bound,
        upper_bound,
        method: OutlierMethod::ZScore {
            threshold: (threshold * 2.0) as u32,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_outliers() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = detect_outliers(&samples, OutlierMethod::default());

        assert!(result.outlier_indices.is_empty());
        assert_eq!(result.cleaned_samples.len(), 5);
    }

    #[test]
    fn test_with_outliers() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100 is outlier
        let result = detect_outliers(&samples, OutlierMethod::default());

        assert!(!result.outlier_indices.is_empty());
        assert_eq!(result.high_outlier_count, 1);
        assert_eq!(result.all_samples.len(), 6); // Original preserved
        assert_eq!(result.cleaned_samples.len(), 5); // Outlier removed
    }

    #[test]
    fn test_outlier_percentage() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let result = detect_outliers(&samples, OutlierMethod::default());

        // 1 out of 6 = ~16.7%
        assert!(result.outlier_percentage() > 15.0);
        assert!(result.outlier_percentage() < 20.0);
    }

    #[test]
    fn test_no_detection() {
        let samples = vec![1.0, 2.0, 100.0];
        let result = detect_outliers(&samples, OutlierMethod::None);

        assert!(result.outlier_indices.is_empty());
        assert_eq!(result.cleaned_samples.len(), 3);
    }

    #[test]
    fn test_empty_samples() {
        let samples: Vec<f64> = Vec::new();
        let result = detect_outliers(&samples, OutlierMethod::default());

        assert!(result.outlier_indices.is_empty());
        assert!(result.all_samples.is_empty());
    }
}
