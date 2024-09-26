// src/services/graph_service.rs

use crate::AppState;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;
use log::info;
use std::collections::HashMap;
use tokio::fs;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::utils::gpu_compute::GPUCompute;

/// Service responsible for building and managing the graph data structure.
pub struct GraphService;

impl GraphService {
    /// Builds the graph data structure from processed Markdown files.
    /// 
    /// This function performs the following steps:
    /// 1. Reads all processed Markdown files from the designated directory.
    /// 2. Parses each file to extract nodes and their relationships.
    /// 3. Constructs nodes and edges based on bidirectional references.
    /// 4. Uses GPUCompute to calculate force-directed layout.
    /// 5. Returns the complete `GraphData` structure.
    ///
    /// # Arguments
    ///
    /// * `state` - Shared application state containing settings and file cache.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `GraphData` on success or an error on failure.
    pub async fn build_graph(state: &AppState) -> Result<GraphData, Box<dyn std::error::Error + Send + Sync>> {
        info!("Building graph data from processed files");

        // Define the directory where processed Markdown files are stored.
        let markdown_dir = "/app/data/markdown";

        // Read the directory entries.
        let mut entries = fs::read_dir(markdown_dir).await?;

        let mut graph = GraphData::default();
        let mut node_map: HashMap<String, Node> = HashMap::new();
        let mut edge_set: HashMap<(String, String), bool> = HashMap::new();

        // Iterate over each file in the directory.
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("md") {
                let file_name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
                let content = fs::read_to_string(&path).await?;

                // Parse the content to extract links and other metadata.
                let links = Self::extract_links(&content);

                // Create or update the node in the node_map.
                node_map.entry(file_name.clone()).or_insert(Node {
                    id: file_name.clone(),
                    label: file_name.clone(),
                    metadata: HashMap::new(),
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    vx: 0.0,
                    vy: 0.0,
                    vz: 0.0,
                });

                // Iterate over each link to create edges.
                for link in links {
                    if link != file_name {
                        // Ensure both source and target nodes exist.
                        node_map.entry(link.clone()).or_insert(Node {
                            id: link.clone(),
                            label: link.clone(),
                            metadata: HashMap::new(),
                            x: 0.0,
                            y: 0.0,
                            z: 0.0,
                            vx: 0.0,
                            vy: 0.0,
                            vz: 0.0,
                        });

                        // To avoid duplicate edges, use a sorted tuple as the key.
                        let (source, target) = if file_name < link.clone() {
                            (file_name.clone(), link.clone())
                        } else {
                            (link.clone(), file_name.clone())
                        };

                        edge_set.entry((source.clone(), target.clone())).or_insert(true);

                        // Add edge to graph.edges.
                        graph.edges.push(Edge {
                            source: source.clone(),
                            target: target.clone(),
                            weight: 1.0, // You can adjust weight based on criteria.
                        });
                    }
                }
            }
        }

        // Populate the nodes in the graph.
        graph.nodes = node_map.into_iter().map(|(_, node)| node).collect();

        info!("Graph data built with {} nodes and {} edges", graph.nodes.len(), graph.edges.len());

        // Use GPUCompute to calculate force-directed layout
        Self::calculate_layout(&state.gpu_compute, &mut graph).await?;
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
        let re = regex::Regex::new(r"\[\[(.*?)\]\]").unwrap();
        for cap in re.captures_iter(content) {
            if let Some(link) = cap.get(1) {
                links.push(link.as_str().to_string());
            }
        }
        links
    }

    /// Calculates the force-directed layout using GPUCompute.
    ///
    /// # Arguments
    ///
    /// * `gpu_compute` - Reference to the GPUCompute instance.
    /// * `graph` - Mutable reference to the GraphData to be updated.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure.
    async fn calculate_layout(gpu_compute: &Arc<RwLock<GPUCompute>>, graph: &mut GraphData) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Set the graph data in GPUCompute
        let mut gpu_compute = gpu_compute.write().await; // Acquire write lock
        gpu_compute.set_graph_data(graph)?;
        
        // Perform force calculations
        gpu_compute.compute_forces()?;

        // Update node positions
        gpu_compute.update_positions()?;

        // Retrieve updated positions from GPUCompute
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

        Ok(())
    }
}
