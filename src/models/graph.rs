use crate::utils::socket_flow_messages::Node;
use super::edge::Edge;
use super::metadata::MetadataStore;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    /// Mapping from numeric ID to metadata ID (filename) for lookup
    #[serde(skip)]
    pub id_to_metadata: HashMap<String, String>,
}

impl GraphData {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: MetadataStore::new(),
            id_to_metadata: HashMap::new(),
        }
    }
}
