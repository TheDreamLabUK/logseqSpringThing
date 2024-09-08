use serde::{Deserialize, Serialize};
use chrono::prelude::*;

#[derive(Default, Serialize, Deserialize)]
pub struct Metadata {
    pub file_name: String,
    pub last_modified: DateTime<Utc>,
    pub processed_file: String,
    pub original_file: String,
}
