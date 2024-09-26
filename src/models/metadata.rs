// metadata.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Stores metadata about a processed file.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Metadata {
    /// Name of the file.
    pub file_name: String,
    /// Last modified timestamp.
    pub last_modified: DateTime<Utc>,
    /// Content of the processed file.
    pub processed_file: String,
    /// Original content of the file.
    pub original_file: String,
}
