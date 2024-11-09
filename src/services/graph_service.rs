use crate::AppState;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::models::metadata::Metadata;
use crate::utils::gpu_compute::GPUCompute;
use crate::models::simulation_params::SimulationParams;
use log::{info, warn, debug};
use std::collections::{HashMap, HashSet};
use tokio::fs;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json;

/// Service responsible for building and managing the graph data structure.
pub struct GraphService;

impl GraphService {
    /// Builds the graph data structure from processed Markdown files.
    pub async fn build_graph(app_state: &AppState) -> Result<GraphData, Box<dyn std::error::Error + Send + Sync>> {
        info!("Building graph data from metadata");
        let metadata_path = "/app/data/markdown/metadata.json";
        let metadata_content = fs::read_to_string(metadata_path).await?;
        let metadata: HashMap<String, Metadata> = serde_json::from_str(&metadata_content)?;
    
        let mut graph = GraphData::default();
        let mut edge_map: HashMap<(String, String), f32> = HashMap::new();
    
        // Create nodes with default positions first
        for (file_name, file_metadata) in &metadata {
            let node_id = file_name.trim_end_matches(".md").to_string();
            let mut node_metadata = HashMap::new();
            
            // Add file metadata to node
            node_metadata.insert("file_size".to_string(), file_metadata.file_size.to_string());
            node_metadata.insert("node_size".to_string(), file_metadata.node_size.to_string());
            node_metadata.insert("last_modified".to_string(), file_metadata.last_modified.to_string());
            node_metadata.insert("hyperlink_count".to_string(), file_metadata.hyperlink_count.to_string());
            
            graph.nodes.push(Node {
                id: node_id.clone(),
                label: node_id.clone(),
                metadata: node_metadata,
                x: 0.0,
                y: 0.0,
                z: 0.0,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
            });
            graph.metadata.insert(node_id.clone(), file_metadata.clone());
        }
    
        // Build edges from topic counts
        for (file_name, file_metadata) in &metadata {
            let source_id = file_name.trim_end_matches(".md").to_string();
            
            for (target_id, reference_count) in &file_metadata.topic_counts {
                if source_id != *target_id {
                    let edge_key = if source_id < *target_id {
                        (source_id.clone(), target_id.clone())
                    } else {
                        (target_id.clone(), source_id.clone())
                    };

                    edge_map.entry(edge_key)
                        .and_modify(|weight| *weight += *reference_count as f32)
                        .or_insert(*reference_count as f32);
                }
            }
        }
    
        // Convert edge_map to edges
        graph.edges = edge_map.into_iter().map(|((source, target), weight)| {
            Edge::new(source, target, weight)
        }).collect();
        
        info!("Graph data built with {} nodes and {} edges", graph.nodes.len(), graph.edges.len());

        // Calculate initial layout
        let settings = app_state.settings.read().await;
        let params = SimulationParams::new(
            settings.visualization.force_directed_iterations as u32,
            settings.visualization.force_directed_spring,
            settings.visualization.force_directed_repulsion,
            settings.visualization.force_directed_attraction,
            settings.visualization.force_directed_damping,
            true // Initial layout
        );

        Self::calculate_layout(&app_state.gpu_compute, &mut graph, &params).await?;
        
        Ok(graph)
    }

    fn initialize_random_positions(graph: &mut GraphData) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let initial_radius = 30.0;
        
        for node in &mut graph.nodes {
            let theta = rng.gen_range(0.0..std::f32::consts::PI * 2.0);
            let phi = rng.gen_range(0.0..std::f32::consts::PI);
            let r = rng.gen_range(0.0..initial_radius);
            
            node.x = r * theta.cos() * phi.sin();
            node.y = r * theta.sin() * phi.sin();
            node.z = r * phi.cos();
            node.vx = 0.0;
            node.vy = 0.0;
            node.vz = 0.0;
        }
    }

    pub async fn calculate_layout(
        gpu_compute: &Option<Arc<RwLock<GPUCompute>>>,
        graph: &mut GraphData,
        params: &SimulationParams,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match gpu_compute {
            Some(gpu) => {
                info!("Using GPU for layout calculation");
                let mut gpu_compute = gpu.write().await;
                
                // Only initialize positions for new graphs
                if graph.nodes.iter().all(|n| n.x == 0.0 && n.y == 0.0 && n.z == 0.0) {
                    Self::initialize_random_positions(graph);
                }
                
                gpu_compute.update_graph_data(graph)?;
                gpu_compute.update_simulation_params(params)?;
                
                // Run iterations with more frequent updates
                for _ in 0..params.iterations {
                    gpu_compute.step()?;
                    
                    // Update positions every iteration for smoother motion
                    let updated_nodes = gpu_compute.get_node_positions().await?;
                    for (i, node) in graph.nodes.iter_mut().enumerate() {
                        node.update_from_gpu_node(&updated_nodes[i]);
                        
                        // Apply bounds
                        let max_coord = 100.0;
                        node.x = node.x.clamp(-max_coord, max_coord);
                        node.y = node.y.clamp(-max_coord, max_coord);
                        node.z = node.z.clamp(-max_coord, max_coord);
                    }
                }
            },
            None => {
                warn!("GPU not available. Falling back to CPU-based layout calculation.");
                Self::calculate_layout_cpu(graph, params.iterations, params.spring_strength, params.damping);
            }
        }
        Ok(())
    }

    fn calculate_layout_cpu(graph: &mut GraphData, iterations: u32, spring_strength: f32, damping: f32) {
        let repulsion_strength = spring_strength * 10000.0;
        
        for _ in 0..iterations {
            // Calculate repulsive forces
            for i in 0..graph.nodes.len() {
                for j in (i + 1)..graph.nodes.len() {
                    let dx = graph.nodes[j].x - graph.nodes[i].x;
                    let dy = graph.nodes[j].y - graph.nodes[i].y;
                    let dz = graph.nodes[j].z - graph.nodes[i].z;
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                    let force = repulsion_strength / (distance * distance);
                    let fx = force * dx / distance;
                    let fy = force * dy / distance;
                    let fz = force * dz / distance;

                    let max_force = 50.0;
                    let force_magnitude = (fx * fx + fy * fy + fz * fz).sqrt();
                    let scale = if force_magnitude > max_force {
                        max_force / force_magnitude
                    } else {
                        1.0
                    };

                    graph.nodes[i].vx -= fx * scale;
                    graph.nodes[i].vy -= fy * scale;
                    graph.nodes[i].vz -= fz * scale;
                    graph.nodes[j].vx += fx * scale;
                    graph.nodes[j].vy += fy * scale;
                    graph.nodes[j].vz += fz * scale;
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
                
                let force = spring_strength * distance * edge.weight;
                let fx = force * dx / distance;
                let fy = force * dy / distance;
                let fz = force * dz / distance;

                let max_force = 50.0;
                let force_magnitude = (fx * fx + fy * fy + fz * fz).sqrt();
                let scale = if force_magnitude > max_force {
                    max_force / force_magnitude
                } else {
                    1.0
                };

                graph.nodes[source].vx += fx * scale;
                graph.nodes[source].vy += fy * scale;
                graph.nodes[source].vz += fz * scale;
                graph.nodes[target].vx -= fx * scale;
                graph.nodes[target].vy -= fy * scale;
                graph.nodes[target].vz -= fz * scale;
            }

            // Update positions
            for node in &mut graph.nodes {
                let max_velocity = 10.0;
                let velocity_magnitude = (node.vx * node.vx + node.vy * node.vy + node.vz * node.vz).sqrt();
                if velocity_magnitude > max_velocity {
                    let scale = max_velocity / velocity_magnitude;
                    node.vx *= scale;
                    node.vy *= scale;
                    node.vz *= scale;
                }

                node.x += node.vx;
                node.y += node.vy;
                node.z += node.vz;

                let max_coord = 100.0;
                node.x = node.x.clamp(-max_coord, max_coord);
                node.y = node.y.clamp(-max_coord, max_coord);
                node.z = node.z.clamp(-max_coord, max_coord);

                node.vx *= damping;
                node.vy *= damping;
                node.vz *= damping;
            }
        }
    }

    pub fn find_shortest_path(graph: &GraphData, start: &str, end: &str) -> Result<Vec<String>, String> {
        let mut distances: HashMap<String, f32> = HashMap::new();
        let mut previous: HashMap<String, Option<String>> = HashMap::new();
        let mut unvisited: HashSet<String> = HashSet::new();
    
        for node in &graph.nodes {
            distances.insert(node.id.clone(), f32::INFINITY);
            previous.insert(node.id.clone(), None);
            unvisited.insert(node.id.clone());
        }
        distances.insert(start.to_string(), 0.0);
    
        while !unvisited.is_empty() {
            let current = unvisited.iter()
                .min_by(|a, b| distances[*a].partial_cmp(&distances[*b]).unwrap())
                .cloned()
                .unwrap();
    
            if current == end {
                break;
            }
    
            unvisited.remove(&current);
    
            for edge in &graph.edges {
                if edge.source == current || edge.target_node == current {
                    let neighbor = if edge.source == current { &edge.target_node } else { &edge.source };
                    if unvisited.contains(neighbor) {
                        let alt = distances[&current] + edge.weight;
                        if alt < distances[neighbor] {
                            distances.insert(neighbor.to_string(), alt);
                            previous.insert(neighbor.to_string(), Some(current.to_string()));
                        }
                    }
                }
            }
        }
    
        let mut path = Vec::new();
        let mut current = end.to_string();
        while let Some(prev) = previous[&current].clone() {
            path.push(current.clone());
            current = prev;
            if current == start {
                path.push(start.to_string());
                path.reverse();
                return Ok(path);
            }
        }
    
        Err("No path found".to_string())
    }
}
