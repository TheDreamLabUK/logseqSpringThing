use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Stores metadata about a processed file.
/// All fields use camelCase serialization for client compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Metadata {
    #[serde(default)]
    pub file_name: String,
    #[serde(default)]
    pub file_size: usize,
    #[serde(default)]
    pub node_size: f64,
    #[serde(default)]
    pub hyperlink_count: usize,
    #[serde(default)]
    pub sha1: String,
    #[serde(default = "Utc::now")]
    pub last_modified: DateTime<Utc>,
    #[serde(default)]
    pub perplexity_link: String,
    #[serde(default)]
    pub last_perplexity_process: Option<DateTime<Utc>>,
    #[serde(default)]
    pub topic_counts: HashMap<String, usize>,
}

/// Type alias for metadata storage with camelCase keys
pub type MetadataStore = HashMap<String, Metadata>;

// Implement helper methods directly on HashMap<String, Metadata>
pub trait MetadataOps {
    fn validate_files(&self, markdown_dir: &str) -> bool;
}

impl MetadataOps for MetadataStore {
    fn validate_files(&self, markdown_dir: &str) -> bool {
        if self.is_empty() {
            return false;
        }

        // Check if the markdown files referenced in metadata actually exist
        for filename in self.keys() {
            let file_path = format!("{}/{}", markdown_dir, filename);
            if !std::path::Path::new(&file_path).exists() {
                return false;
            }
        }
        
        true
    }
}
