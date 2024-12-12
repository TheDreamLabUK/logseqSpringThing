use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Edge structure representing connections between nodes
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Edge {
    pub source: String,
    pub target: String,
    pub weight: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl Edge {
    pub fn new(source: String, target: String, weight: f32) -> Self {
        Self {
            source,
            target,
            weight,
            edge_type: None,
            metadata: None,
        }
    }
}
