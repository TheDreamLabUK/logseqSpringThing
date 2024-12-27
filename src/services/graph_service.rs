use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::{HashMap, HashSet};
use actix_web::web;
use log::{info, warn, error};
use rand::Rng;
use serde_json;

use crate::models::graph::GraphData;
use crate::utils::socket_flow_messages::Node;
use crate::models::edge::Edge;
use crate::models::metadata::MetadataStore;
use crate::app_state::AppState;
use crate::utils::gpu_compute::GPUCompute;
use crate::models::simulation_params::SimulationParams;
use crate::models::pagination::PaginatedGraphData;
use crate::services::file_service::FileService;

#[derive(Clone)]
pub struct GraphService {
    pub graph_data: Arc<RwLock<GraphData>>,
}

impl GraphService {
    pub fn new() -> Self {
        Self {
            graph_data: Arc::new(RwLock::new(GraphData::default())),
        }
    }

    pub async fn new_with_metadata(metadata_store: &MetadataStore) -> Self {
        let graph_data = Self::build_graph_from_metadata(metadata_store)
            .await
            .unwrap_or_else(|e| {
                error!("Failed to build graph from metadata: {}", e);
                GraphData::default()
            });
        
        Self {
            graph_data: Arc::new(RwLock::new(graph_data)),
        }
    }

    pub async fn build_graph_from_metadata(metadata_store: &MetadataStore) -> Result<GraphData, Box<dyn std::error::Error>> {
        let mut graph = GraphData::new();
        let mut edge_map = HashMap::new();

        info!("Building graph from {} metadata entries", metadata_store.len());

        // First pass: Create nodes from files in metadata
        let mut valid_nodes = HashSet::new();
        for file_name in metadata_store.keys() {
            let node_id = file_name.trim_end_matches(".md").to_string();
            valid_nodes.insert(node_id);
        }

        // Create nodes for all valid node IDs
        for node_id in &valid_nodes {
            let mut node = Node::new(node_id.clone());
            
            // Get metadata for this node
            if let Some(metadata) = metadata_store.get(&format!("{}.md", node_id)) {
                node.size = Some(metadata.node_size as f32);
                node.file_size = metadata.file_size as u64;
                node.label = node_id.clone(); // Set label to node ID (filename without .md)
                
                // Add metadata fields to node's metadata map
                node.metadata.insert("fileSize".to_string(), metadata.file_size.to_string());
                node.metadata.insert("hyperlinkCount".to_string(), metadata.hyperlink_count.to_string());
                node.metadata.insert("lastModified".to_string(), metadata.last_modified.to_string());
            }
            
            // Add node to graph
            graph.nodes.push(node);
        }

        // Store metadata in graph
        graph.metadata = metadata_store.clone();

        // Second pass: Create edges from topic counts
        for (source_file, metadata) in metadata_store.iter() {
            let source_id = source_file.trim_end_matches(".md").to_string();
            
            for (target_file, count) in &metadata.topic_counts {
                let target_id = target_file.trim_end_matches(".md").to_string();
                
                // Only create edge if both nodes exist and they're different
                if source_id != target_id && valid_nodes.contains(&target_id) {
                    let edge_key = if source_id < target_id {
                        (source_id.clone(), target_id.clone())
                    } else {
                        (target_id.clone(), source_id.clone())
                    };

                    edge_map.entry(edge_key)
                        .and_modify(|weight| *weight += *count as f32)
                        .or_insert(*count as f32);
                }
            }
        }

        // Convert edge map to edges
        graph.edges = edge_map.into_iter()
            .map(|((source, target), weight)| {
                Edge::new(source, target, weight)
            })
            .collect();

        // Initialize random positions using the same method everywhere
        Self::initialize_random_positions(&mut graph);

        info!("Built graph with {} nodes and {} edges", graph.nodes.len(), graph.edges.len());
        Ok(graph)
    }

    pub async fn build_graph(state: &web::Data<AppState>) -> Result<GraphData, Box<dyn std::error::Error + Send + Sync>> {
        let current_graph = state.graph_service.graph_data.read().await;
        let mut graph = GraphData::new();

        // Copy metadata from current graph
        graph.metadata = current_graph.metadata.clone();

        let mut edge_map = HashMap::new();

        // Create nodes from metadata entries
        let mut valid_nodes = HashSet::new();
        for file_name in graph.metadata.keys() {
            let node_id = file_name.trim_end_matches(".md").to_string();
            valid_nodes.insert(node_id);
        }

        // Create nodes for all valid node IDs
        for node_id in &valid_nodes {
            let mut node = Node::new(node_id.clone());
            
            // Get metadata for this node
            if let Some(metadata) = graph.metadata.get(&format!("{}.md", node_id)) {
                node.size = Some(metadata.node_size as f32);
                node.file_size = metadata.file_size as u64;
                node.label = node_id.clone(); // Set label to node ID (filename without .md)
                
                // Add metadata fields to node's metadata map
                node.metadata.insert("fileSize".to_string(), metadata.file_size.to_string());
                node.metadata.insert("hyperlinkCount".to_string(), metadata.hyperlink_count.to_string());
                node.metadata.insert("lastModified".to_string(), metadata.last_modified.to_string());
            }
            
            graph.nodes.push(node);
        }

        // Create edges from metadata topic counts
        for (source_file, metadata) in graph.metadata.iter() {
            let source_id = source_file.trim_end_matches(".md").to_string();
            
            // Process outbound links from this file to other topics
            for (target_file, count) in &metadata.topic_counts {
                let target_id = target_file.trim_end_matches(".md").to_string();
                
                // Only create edge if both nodes exist and they're different
                if source_id != target_id && valid_nodes.contains(&target_id) {
                    let edge_key = if source_id < target_id {
                        (source_id.clone(), target_id.clone())
                    } else {
                        (target_id.clone(), source_id.clone())
                    };

                    // Sum the weights for bi-directional references
                    edge_map.entry(edge_key)
                        .and_modify(|w| *w += *count as f32)
                        .or_insert(*count as f32);
                }
            }
        }

        // Convert edge map to edges
        graph.edges = edge_map.into_iter()
            .map(|((source, target), weight)| {
                Edge::new(source, target, weight)
            })
            .collect();

        // Initialize random positions for all nodes
        Self::initialize_random_positions(&mut graph);

        info!("Built graph with {} nodes and {} edges", graph.nodes.len(), graph.edges.len());
        Ok(graph)
    }

    fn initialize_random_positions(graph: &mut GraphData) {
        let mut rng = rand::thread_rng();
        let initial_radius = 100.0; // Match default spring length
        
        for node in &mut graph.nodes {
            // Use spherical coordinates for uniform distribution
            let theta = rng.gen_range(0.0..std::f32::consts::PI * 2.0);
            let phi = rng.gen_range(0.0..std::f32::consts::PI);
            let r = initial_radius * rng.gen::<f32>().cbrt(); // Cube root for uniform volume distribution
            
            // Convert to Cartesian coordinates
            node.set_x(r * theta.cos() * phi.sin());
            node.set_y(r * theta.sin() * phi.sin());
            node.set_z(r * phi.cos());
            node.set_vx(0.0);
            node.set_vy(0.0);
            node.set_vz(0.0);
        }
    }

    pub async fn calculate_layout(
        gpu_compute: &Option<Arc<RwLock<GPUCompute>>>,
        graph: &mut GraphData,
        params: &SimulationParams,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Initialize positions if needed, regardless of GPU availability
        if graph.nodes.iter().all(|n| n.x() == 0.0 && n.y() == 0.0 && n.z() == 0.0) {
            Self::initialize_random_positions(graph);
        }

        // Only proceed with force-directed layout if GPU is available
        if let Some(gpu) = gpu_compute {
            info!("Using GPU for layout calculation");
            let mut gpu_compute = gpu.write().await;
            
            gpu_compute.update_graph_data(graph)?;
            gpu_compute.update_simulation_params(params)?;
            
            // Run iterations with more frequent updates
            for _ in 0..params.iterations {
                gpu_compute.step()?;
                
                // Update positions every iteration for smoother motion
                let updated_nodes = gpu_compute.get_node_data()?;
                for (i, node) in graph.nodes.iter_mut().enumerate() {
                    node.update_from_gpu_node(&updated_nodes[i]);
                }
            }
        } else {
            warn!("GPU not available. Using static random positions.");
        }
        Ok(())
    }

    pub async fn get_paginated_graph_data(
        &self,
        page: u32,
        page_size: u32,
    ) -> Result<PaginatedGraphData, Box<dyn std::error::Error + Send + Sync>> {
        let graph = self.graph_data.read().await;
        
        // Convert page and page_size to usize for vector operations
        let page = page as usize;
        let page_size = page_size as usize;
        let total_nodes = graph.nodes.len();
        
        let start = page * page_size;
        let end = std::cmp::min((page + 1) * page_size, total_nodes);

        let page_nodes: Vec<Node> = graph.nodes
            .iter()
            .skip(start)
            .take(end - start)
            .cloned()
            .collect();

        // Get edges that connect to these nodes
        let node_ids: HashSet<String> = page_nodes.iter()
            .map(|n| n.id.clone())
            .collect();

        let edges: Vec<Edge> = graph.edges
            .iter()
            .filter(|e| node_ids.contains(&e.source) || node_ids.contains(&e.target))
            .cloned()
            .collect();

        Ok(PaginatedGraphData {
            nodes: page_nodes,
            edges: edges.clone(),
            metadata: serde_json::to_value(graph.metadata.clone()).unwrap_or_default(),
            total_nodes,
            total_edges: graph.edges.len(),
            total_pages: ((total_nodes as f32 / page_size as f32).ceil()) as u32,
            current_page: page as u32,
        })
    }

    pub async fn get_node_positions(&self) -> Vec<Node> {
        let graph = self.graph_data.read().await;
        graph.nodes.clone()
    }

    pub async fn update_graph(&self) -> Result<GraphData, String> {
        info!("Updating graph data");

        // Load or create metadata
        let metadata_store = match FileService::load_or_create_metadata() {
            Ok(store) => store,
            Err(e) => {
                error!("Failed to load metadata: {}", e);
                return Err(format!("Failed to load metadata: {}", e));
            }
        };

        info!("Loaded metadata with {} entries", metadata_store.len());

        // Build graph from metadata
        let graph = match Self::build_graph_from_metadata(&metadata_store).await {
            Ok(g) => g,
            Err(e) => {
                error!("Failed to build graph: {}", e);
                return Err(format!("Failed to build graph: {}", e));
            }
        };

        Ok(graph)
    }
}
