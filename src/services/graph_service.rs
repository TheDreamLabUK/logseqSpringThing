// src/services/graph_service.rs

use crate::AppState;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::models::metadata::Metadata;
use log::{info, warn, debug};
use std::collections::HashMap;
use tokio::fs;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::utils::gpu_compute::GPUCompute;
use serde_json;
use regex::Regex;

/// Service responsible for building and managing the graph data structure.
pub struct GraphService;

impl GraphService {
    /// Builds the graph data structure from processed Markdown files.
    /// 
    /// This function performs the following steps:
    /// 1. Reads the metadata JSON file.
    /// 2. Creates nodes for each file in the metadata.
    /// 3. Analyzes content for connections between nodes.
    /// 4. Extracts hyperlinks from the content.
    /// 5. Constructs edges based on both interconnectedness and hyperlinks.
    /// 6. Uses GPUCompute to calculate force-directed layout if available, otherwise falls back to CPU.
    /// 7. Returns the complete `GraphData` structure.
    ///
    /// # Arguments
    ///
    /// * `state` - Shared application state containing settings and file cache.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `GraphData` on success or an error on failure.
    pub async fn build_graph(state: &AppState) -> Result<GraphData, Box<dyn std::error::Error + Send + Sync>> {
        info!("Building graph data from metadata");

        let metadata_path = "/app/data/markdown/metadata.json";
        let metadata_content = fs::read_to_string(metadata_path).await?;
        let metadata: HashMap<String, Metadata> = serde_json::from_str(&metadata_content)?;

        let mut graph = GraphData::default();
        let mut node_map: HashMap<String, Node> = HashMap::new();
        let mut edge_map: HashMap<(String, String), (f32, u32)> = HashMap::new();

        // Create nodes and analyze content for connections
        for (file_name, file_metadata) in &metadata {
            let node_id = file_name.trim_end_matches(".md").to_string();
            
            // Create or get the node
            node_map.entry(node_id.clone()).or_insert(Node {
                id: node_id.clone(),
                label: node_id.clone(),
                metadata: HashMap::new(),
                x: 0.0, y: 0.0, z: 0.0,
                vx: 0.0, vy: 0.0, vz: 0.0,
            });

            // Analyze content for connections
            for (other_file, _) in &metadata {
                if file_name != other_file {
                    let other_node_id = other_file.trim_end_matches(".md");
                    let count = file_metadata.processed_file.matches(other_node_id).count() as f32;
                    
                    if count > 0.0 {
                        let edge_key = if node_id < other_node_id.to_string() {
                            (node_id.clone(), other_node_id.to_string())
                        } else {
                            (other_node_id.to_string(), node_id.clone())
                        };
                        
                        edge_map.entry(edge_key)
                            .and_modify(|(weight, _)| *weight += count)
                            .or_insert((count, 0));
                    }
                }
            }

            // Extract hyperlinks
            let links = Self::extract_links(&file_metadata.processed_file);
            for link in links {
                if link != node_id {
                    let edge_key = if node_id < link {
                        (node_id.clone(), link)
                    } else {
                        (link, node_id.clone())
                    };

                    edge_map.entry(edge_key)
                        .and_modify(|(_, hyperlinks)| *hyperlinks += 1)
                        .or_insert((0.0, 1));
                }
            }
        }

        // Create edges from the edge map
        for ((source, target), (weight, hyperlinks)) in edge_map {
            graph.edges.push(Edge {
                source,
                target_node: target,
                weight,
                hyperlinks: hyperlinks as f32,
            });
        }

        graph.nodes = node_map.into_values().collect();

        info!("Graph data built with {} nodes and {} edges", graph.nodes.len(), graph.edges.len());
        debug!("Sample node data: {:?}", graph.nodes.first());
        debug!("Sample edge data: {:?}", graph.edges.first());

        // Calculate layout using GPU if available, otherwise fall back to CPU
        Self::calculate_layout(&state.gpu_compute, &mut graph).await?;
        
        debug!("Final sample node data after layout calculation: {:?}", graph.nodes.first());
        
        Ok(graph)
    }

    /// Extracts links from the Markdown content.
    /// 
    /// This function looks for `[[Link]]` patterns and extracts the link targets.
    ///
    /// # Arguments
    ///
    /// * `content` - The Markdown content as a string.
    ///
    /// # Returns
    ///
    /// A vector of link targets as strings.
    fn extract_links(content: &str) -> Vec<String> {
        let mut links = Vec::new();
        let re = Regex::new(r"\[\[(.*?)\]\]").unwrap();
        for cap in re.captures_iter(content) {
            if let Some(link) = cap.get(1) {
                links.push(link.as_str().to_string());
            }
        }
        links
    }

    /// Calculates the force-directed layout using GPUCompute if available, otherwise falls back to CPU.
    ///
    /// # Arguments
    ///
    /// * `gpu_compute` - Optional reference to the GPUCompute instance.
    /// * `graph` - Mutable reference to the GraphData to be updated.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure.
    async fn calculate_layout(gpu_compute: &Option<Arc<RwLock<GPUCompute>>>, graph: &mut GraphData) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match gpu_compute {
            Some(gpu) => {
                info!("Using GPU for layout calculation");
                // GPU-based calculation
                let mut gpu_compute = gpu.write().await; // Acquire write lock
                gpu_compute.set_graph_data(graph)?;
                gpu_compute.compute_forces()?;
                let updated_nodes = gpu_compute.get_updated_positions().await?;

                // Update graph nodes with new positions
                for (i, node) in graph.nodes.iter_mut().enumerate() {
                    node.x = updated_nodes[i].x;
                    node.y = updated_nodes[i].y;
                    node.z = updated_nodes[i].z;
                    node.vx = updated_nodes[i].vx;
                    node.vy = updated_nodes[i].vy;
                    node.vz = updated_nodes[i].vz;
                }
                debug!("GPU layout calculation complete. Sample updated node: {:?}", graph.nodes.first());
            },
            None => {
                warn!("GPU not available. Falling back to CPU-based layout calculation.");
                Self::calculate_layout_cpu(graph);
                debug!("CPU layout calculation complete. Sample updated node: {:?}", graph.nodes.first());
            }
        }

        Ok(())
    }

    fn calculate_layout_cpu(graph: &mut GraphData) {
        const ITERATIONS: usize = 100;
        const REPULSION: f32 = 1.0;
        const ATTRACTION: f32 = 0.01;

        for _ in 0..ITERATIONS {
            // Calculate repulsive forces
            for i in 0..graph.nodes.len() {
                for j in (i + 1)..graph.nodes.len() {
                    let dx = graph.nodes[j].x - graph.nodes[i].x;
                    let dy = graph.nodes[j].y - graph.nodes[i].y;
                    let dz = graph.nodes[j].z - graph.nodes[i].z;
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                    let force = REPULSION / (distance * distance);
                    let fx = force * dx / distance;
                    let fy = force * dy / distance;
                    let fz = force * dz / distance;

                    graph.nodes[i].vx -= fx;
                    graph.nodes[i].vy -= fy;
                    graph.nodes[i].vz -= fz;
                    graph.nodes[j].vx += fx;
                    graph.nodes[j].vy += fy;
                    graph.nodes[j].vz += fz;
                }
            }

            // Calculate attractive forces
            for edge in &graph.edges {
                let source = graph.nodes.iter().position(|n| n.id == edge.source).unwrap();
                let target = graph.nodes.iter().position(|n| n.id == edge.target_node).unwrap();
                let dx = graph.nodes[target].x - graph.nodes[source].x;
                let dy = graph.nodes[target].y - graph.nodes[source].y;
                let dz = graph.nodes[target].z - graph.nodes[source].z;
                let distance = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                let force = ATTRACTION * distance * edge.weight;
                let fx = force * dx / distance;
                let fy = force * dy / distance;
                let fz = force * dz / distance;

                graph.nodes[source].vx += fx;
                graph.nodes[source].vy += fy;
                graph.nodes[source].vz += fz;
                graph.nodes[target].vx -= fx;
                graph.nodes[target].vy -= fy;
                graph.nodes[target].vz -= fz;
            }

            // Update positions
            for node in &mut graph.nodes {
                node.x += node.vx;
                node.y += node.vy;
                node.z += node.vz;
                node.vx *= 0.9; // Damping
                node.vy *= 0.9;
                node.vz *= 0.9;
            }
        }
    }
}
