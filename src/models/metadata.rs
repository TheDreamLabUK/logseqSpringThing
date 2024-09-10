use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Metadata {
    pub file_name: String,
    pub last_modified: DateTime<Utc>,
    pub processed_file: String,
    pub original_file: String,
}
