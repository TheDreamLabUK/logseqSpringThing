use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::{HashMap, HashSet};
use actix_web::web;
use log::{info, warn, debug, error};
use rand::Rng;
use serde_json;

use crate::models::graph::GraphData;
use crate::utils::socket_flow_messages::Node;
use crate::models::edge::Edge;
use crate::models::metadata::MetadataStore;
use crate::app_state::AppState;
use crate::utils::gpu_compute::GPUCompute;
use crate::utils::gpu_initializer::initialize_gpu;
use crate::Settings;
use crate::models::simulation_params::{SimulationParams, SimulationPhase, SimulationMode};
use crate::models::pagination::PaginatedGraphData;

#[derive(Clone)]
pub struct GraphService {
    pub graph_data: Arc<RwLock<GraphData>>,
    gpu_compute: Option<Arc<RwLock<GPUCompute>>>,
}

impl GraphService {
    pub async fn new() -> Self {
        let graph_data = Arc::new(RwLock::new(GraphData::default()));
        let graph_service = Self {
            graph_data: graph_data.clone(),
            gpu_compute: None,
        };

        // Start simulation loop
        let graph_data = graph_service.graph_data.clone();
        let gpu_compute = graph_service.gpu_compute.clone();
        tokio::spawn(async move {
            let params = SimulationParams {
                iterations: 1,  // One iteration per frame
                spring_strength: 5.0,            // Strong spring force for tight clustering
                repulsion: 0.05,                 // Minimal repulsion
                damping: 0.98,                   // Very high damping for stability
                max_repulsion_distance: 0.1,     // Small repulsion range
                viewport_bounds: 1.0,            // Small bounds for tight clustering
                mass_scale: 1.0,                 // Default mass scaling
                boundary_damping: 0.95,          // Strong boundary damping
                enable_bounds: true,             // Enable bounds by default
                time_step: 0.01,                 // Smaller timestep for stability
                phase: SimulationPhase::Dynamic,
                mode: SimulationMode::Remote,    // Force GPU-accelerated computation
            };

            loop {
                // Update positions
                let mut graph = graph_data.write().await;
                if let Err(e) = Self::calculate_layout(&gpu_compute, &mut graph, &params).await {
                    warn!("[Graph] Error updating positions: {}", e);
                }
                drop(graph); // Release lock

                // Sleep for ~16ms (60fps)
                tokio::time::sleep(tokio::time::Duration::from_millis(16)).await;
            }
        });

        graph_service
    }

    pub async fn initialize_gpu(
        &mut self,
        settings: Arc<RwLock<Settings>>,
        graph: &GraphData
    ) -> std::io::Result<()> {
        debug!("Initializing GPU compute for graph with {} nodes", graph.nodes.len());
        
        match initialize_gpu(settings, graph).await {
            Ok(Some(gpu)) => {
                self.gpu_compute = Some(gpu);
                info!("GPU compute initialized and ready for graph computations");
                Ok(())
            },
            Ok(None) => {
                warn!("GPU initialization skipped - using CPU fallback");
                Ok(())
            },
            Err(e) => {
                Err(e)
            },
        }
    }

    pub async fn build_graph_from_metadata(metadata: &MetadataStore) -> Result<GraphData, Box<dyn std::error::Error + Send + Sync>> {
        let mut graph = GraphData::new();
        let mut edge_map = HashMap::new();

        // First pass: Create nodes from files in metadata
        let mut valid_nodes = HashSet::new();
        for file_name in metadata.keys() {
            let node_id = file_name.trim_end_matches(".md").to_string();
            valid_nodes.insert(node_id);
        }

        // Create nodes for all valid node IDs
        for node_id in &valid_nodes {
            let mut node = Node::new(node_id.clone());
            
            // Get metadata for this node
            if let Some(metadata) = metadata.get(&format!("{}.md", node_id)) {
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

        // Store metadata in graph
        graph.metadata = metadata.clone();

        // Second pass: Create edges from topic counts
        for (source_file, metadata) in metadata.iter() {
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

        // Initialize random positions
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
        let node_count = graph.nodes.len() as f32;
        let initial_radius = 0.5; // Half of viewport bounds
        let golden_ratio = (1.0 + 5.0_f32.sqrt()) / 2.0;
        
        // Use Fibonacci sphere distribution for more uniform initial positions
        for (i, node) in graph.nodes.iter_mut().enumerate() {
            let i = i as f32;
            
            // Calculate Fibonacci sphere coordinates
            let theta = 2.0 * std::f32::consts::PI * i / golden_ratio;
            let phi = (1.0 - 2.0 * (i + 0.5) / node_count).acos();
            
            // Add slight randomness to prevent exact overlaps
            let r = initial_radius * (0.9 + rng.gen_range(0.0..0.2));
            
            node.set_x(r * phi.sin() * theta.cos());
            node.set_y(r * phi.sin() * theta.sin());
            node.set_z(r * phi.cos());
            
            // Initialize with zero velocity
            node.set_vx(0.0);
            node.set_vy(0.0);
            node.set_vz(0.0);
        }
    }

    pub async fn calculate_layout(
        gpu_compute: &Option<Arc<RwLock<GPUCompute>>>,
        graph: &mut GraphData,
        params: &SimulationParams,
    ) -> std::io::Result<()> {
        match gpu_compute {
            Some(gpu) => {
                info!("Using GPU for layout calculation");
                let mut gpu_compute = gpu.write().await;
                
                // Only initialize positions for new graphs
                if graph.nodes.iter().all(|n| n.x() == 0.0 && n.y() == 0.0 && n.z() == 0.0) {
                    Self::initialize_random_positions(graph);
                    debug!("Initialized random positions for new graph");
                }
                
                gpu_compute.update_graph_data(graph)?;
                gpu_compute.update_simulation_params(params)?;
                
                // Run iterations with debug logging
                for iter in 0..params.iterations {
                    gpu_compute.step()?;
                    debug!("Completed GPU iteration {}/{}", iter + 1, params.iterations);
                    let updated_nodes = gpu_compute.get_node_data()?;
                    if log::log_enabled!(log::Level::Debug) {
                        if let Some(first) = updated_nodes.get(0) {
                            debug!("GPU layout iteration {}: updated node[0]: {:?}", iter, first);
                        }
                    }
                    for (i, node) in graph.nodes.iter_mut().enumerate() {
                        node.data = updated_nodes[i].clone();
                    }
                }
                debug!("GPU layout calculation completed successfully");
                Ok(())
            },
            None => {
                let err = "GPU computation is required. CPU fallback is disabled.";
                error!("{}", err);
                Err(std::io::Error::new(std::io::ErrorKind::Unsupported, err))
            }
        }
    }


    fn calculate_layout_cpu(graph: &mut GraphData, iterations: u32, spring_strength: f32, damping: f32) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let repulsion_strength = 0.05; // Match CUDA implementation
        let max_repulsion_distance = 0.1; // Match CUDA implementation
        let bounds = 1.0; // Match viewport_bounds
        let min_distance = 0.1; // Minimum distance to prevent division by zero
        
        for iter in 0..iterations {
            if log::log_enabled!(log::Level::Debug) {
                if let Some(first) = graph.nodes.get(0) {
                    debug!("CPU Iteration {}: initial node[0] position: x={}, y={}, z={}", 
                           iter, first.x(), first.y(), first.z());
                }
            }
            // Calculate forces between nodes
            let mut forces = vec![(0.0, 0.0, 0.0); graph.nodes.len()];
            
            // Calculate repulsion forces
            for i in 0..graph.nodes.len() {
                for j in i+1..graph.nodes.len() {
                    let dx = graph.nodes[j].x() - graph.nodes[i].x();
                    let dy = graph.nodes[j].y() - graph.nodes[i].y();
                    let dz = graph.nodes[j].z() - graph.nodes[i].z();
                    
                    let distance_squared = dx * dx + dy * dy + dz * dz;
                    let distance = distance_squared.sqrt().max(min_distance);
                    
                    if distance < max_repulsion_distance {
                        // Quadratic falloff matching CUDA implementation
                        let falloff = 1.0 - (distance / max_repulsion_distance);
                        let falloff = falloff * falloff;
                        
                        // Mass-weighted repulsion
                        let force = repulsion_strength * falloff / distance_squared;

                    
                    // Normalize direction vector
                    let fx = (dx / distance) * force;
                    let fy = (dy / distance) * force;
                    let fz = (dz / distance) * force;
                    
                    forces[i].0 -= fx;
                    forces[i].1 -= fy;
                    forces[i].2 -= fz;
                    
                    forces[j].0 += fx;
                    forces[j].1 += fy;
                    forces[j].2 += fz;
                    }
                }
            }

            // Calculate spring forces along edges
            let rest_length = 100.0; // Base rest length
            let max_spring_force = spring_strength * 10.0; // Scale with spring strength
            
            for edge in &graph.edges {
                // Find indices of source and target nodes
                let source_idx = graph.nodes.iter().position(|n| n.id == edge.source);
                let target_idx = graph.nodes.iter().position(|n| n.id == edge.target);
                
                if let (Some(si), Some(ti)) = (source_idx, target_idx) {
                    let source = &graph.nodes[si];
                    let target = &graph.nodes[ti];
                    
                    let dx = target.x() - source.x();
                    let dy = target.y() - source.y();
                    let dz = target.z() - source.z();
                    
                    let distance_squared = dx * dx + dy * dy + dz * dz;
                    let distance = distance_squared.sqrt().max(0.1); // Prevent division by zero
                    
                    // Calculate spring force with ideal length and weight
                    let mass_adjusted_length = rest_length * (1.0 + 0.5 * (edge.weight / 10.0));
                    let force = (spring_strength * (distance - mass_adjusted_length) * edge.weight)
                        .clamp(-max_spring_force, max_spring_force);
                    
                    // Normalize direction vector and apply force
                    let fx = (dx / distance) * force;
                    let fy = (dy / distance) * force;
                    let fz = (dz / distance) * force;
                    
                    forces[si].0 += fx;
                    forces[si].1 += fy;
                    forces[si].2 += fz;
                    
                    forces[ti].0 -= fx;
                    forces[ti].1 -= fy;
                    forces[ti].2 -= fz;
                }
            }
            
            // Apply forces and update positions with stability constraints
            let dt = 0.1; // Time step for integration
            
            // Store first node's position for logging after updates
            let first_node_initial = if log::log_enabled!(log::Level::Debug) {
                graph.nodes.get(0).map(|n| (n.x(), n.y(), n.z()))
            } else {
                None
            };
            
            for (i, node) in graph.nodes.iter_mut().enumerate() {
                // Update velocity with damping and clamping
                let mut vx = node.vx() + forces[i].0 * dt;
                let mut vy = node.vy() + forces[i].1 * dt;
                let mut vz = node.vz() + forces[i].2 * dt;
                
                // Apply damping
                vx *= damping;
                vy *= damping;
                vz *= damping;
                
                // Mass-aware velocity limiting
                let mass = node.size.unwrap_or(1.0);
                let max_velocity = 2.0 / (0.5 + mass);
                // Clamp velocity magnitude
                let v_mag = (vx * vx + vy * vy + vz * vz).sqrt();
                if v_mag > max_velocity {
                    let scale = max_velocity / v_mag;
                    vx *= scale;
                    vy *= scale;
                    vz *= scale;
                }
                
                // Update position
                let mut x = node.x() + vx * dt;
                let mut y = node.y() + vy * dt;
                let mut z = node.z() + vz * dt;
                
                // Apply boundary constraints with additional damping
                if bounds > 0.0 {
                    let near_boundary = x.abs() > bounds * 0.9 || 
                                      y.abs() > bounds * 0.9 || 
                                      z.abs() > bounds * 0.9;
                    if near_boundary {
                        vx *= 0.9;
                        vy *= 0.9;
                        vz *= 0.9;
                    }
                    x = x.clamp(-bounds, bounds);
                    y = y.clamp(-bounds, bounds);
                    z = z.clamp(-bounds, bounds);
                }
                
                // Store updated values
                node.set_vx(vx);
                node.set_vy(vy);
                node.set_vz(vz);
                node.set_x(x);
                node.set_y(y);
                node.set_z(z);
            }
            
            // Log position update after the loop
            if log::log_enabled!(log::Level::Debug) {
                if let Some((old_x, old_y, old_z)) = first_node_initial {
                    debug!("CPU Iteration {}: Node[0] positions - Initial: ({}, {}, {}), Final: ({}, {}, {})",
                        iter, old_x, old_y, old_z,
                        graph.nodes[0].x(), graph.nodes[0].y(), graph.nodes[0].z());
                }
            }
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
}
