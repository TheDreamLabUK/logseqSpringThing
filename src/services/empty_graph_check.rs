use crate::models::graph::GraphData;
use std::io::{Error, ErrorKind};
use log::warn;

/// This function checks if a graph is empty or contains too few nodes
/// It is used before GPU operations to prevent errors
pub fn check_empty_graph(graph: &GraphData, min_nodes: usize) -> Result<(), Error> {
    // Check for completely empty graph
    if graph.nodes.is_empty() {
        return Err(Error::new(ErrorKind::InvalidData, 
            "Graph contains no nodes, cannot perform GPU computation on empty graph"));
    }
    
    // Check if graph is below recommended threshold
    if graph.nodes.len() < min_nodes {
        warn!("[Empty Graph Check] Graph contains only {} nodes, which is below the recommended minimum of {}. 
              This may cause instability in GPU computation.", graph.nodes.len(), min_nodes);
    }
    
    Ok(())
}