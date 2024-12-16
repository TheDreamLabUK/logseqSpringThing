use serde::{Deserialize, Serialize};
use crate::models::edge::Edge;
use crate::utils::socket_flow_messages::Node;

#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    pub page: Option<u32>,
    pub page_size: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct PaginatedGraphData {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub total_pages: u32,
    pub current_page: u32,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub metadata: serde_json::Value,
}
