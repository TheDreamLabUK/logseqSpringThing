use crate::utils::socket_flow_messages::Node;
use super::edge::Edge;
use super::metadata::MetadataStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc, Duration};

/// Represents the graph data structure containing nodes, edges, and metadata.
/// All fields use camelCase serialization for client compatibility.
#[derive(Default, Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct GraphData {
    /// List of nodes in the graph.
    pub nodes: Vec<Node>,
    /// List of edges connecting the nodes.
    pub edges: Vec<Edge>,
    /// Metadata associated with the graph, using camelCase keys.
    pub metadata: MetadataStore,
    /// Timestamp when the graph was last validated against metadata
    pub last_validated: DateTime<Utc>,
    /// Mapping from numeric ID to metadata ID (filename) for lookup
    #[serde(skip)]
    pub id_to_metadata: HashMap<String, String>,
    /// Flag to indicate if the graph has been hot-started from cache
    #[serde(skip)]
    pub hot_started: bool,
}

/// Status of background graph updates for WebSocket clients to check
#[derive(Clone, Debug)]
pub struct GraphUpdateStatus {
    /// Timestamp of the last update check
    pub last_check: DateTime<Utc>,
    /// Timestamp of the most recent update
    pub last_update: DateTime<Utc>,
    /// Whether an update is available that clients should fetch
    pub update_available: bool,
    /// Count of nodes changed in the last update
    pub nodes_changed: i32,
}

impl Default for GraphUpdateStatus {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            last_check: now,
            last_update: now - Duration::hours(1), // Initial offset to ensure first check is considered new
            update_available: false,
            nodes_changed: 0,
        }
    }
}

/// Implementation of GraphData for creating and manipulating graph data
/// All fields use camelCase serialization for client compatibility
impl GraphData {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: MetadataStore::new(),
            last_validated: Utc::now(),
            id_to_metadata: HashMap::new(),
            hot_started: false,
        }
    }
}
