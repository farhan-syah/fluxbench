//! JSON Output

use crate::report::Report;
use serde::{Deserialize, Serialize};

/// Schema information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSchema {
    pub schema: String,
    pub version: String,
}

/// Generate JSON report
pub fn generate_json_report(report: &Report) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(report)
}
