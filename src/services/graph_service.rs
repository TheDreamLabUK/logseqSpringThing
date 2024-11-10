use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::io::Error;
use std::sync::Arc;
use tokio::sync::RwLock;
use actix_web::web;
use log::{info, warn};
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::models::simulation_params::SimulationParams;
use crate::utils::gpu_compute::GPUCompute;

pub struct GraphService {
    pub graph_data: Arc<RwLock<GraphData>>,
}

impl GraphService {
    pub fn new() -> Self {
        GraphService {
            graph_data: Arc::new(RwLock::new(GraphData {
                nodes: Vec::new(),
                edges: Vec::new(),
                metadata: HashMap::new(),
            })),
        }
    }

    pub async fn load_graph(&self, path: &Path) -> Result<(), Error> {
        info!("Loading graph from {}", path.display());
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut edge_map = HashMap::new();

        // Read directory and create nodes
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let file_size = fs::metadata(&path)
                        .map(|m| m.len())
                        .unwrap_or(0);

                    let file_name = path.file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    nodes.push(Node {
                        id: file_name.clone(),
                        label: file_name,
                        metadata: HashMap::new(),
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                        vx: 0.0,
                        vy: 0.0,
                        vz: 0.0,
                        file_size,
                    });
                }
            }
        }

        // Convert edge_map to edges
        edges = edge_map.into_iter().map(|((source, target), weight)| {
            Edge::new(source, target, weight)
        }).collect();
        
        info!("Graph data built with {} nodes and {} edges", nodes.len(), edges.len());

        // Update graph data
        let mut graph_data = self.graph_data.write().await;
        *graph_data = GraphData {
            nodes,
            edges,
            metadata: HashMap::new(),
        };

        info!("Graph loaded with {} nodes and {} edges", 
            graph_data.nodes.len(), graph_data.edges.len());
        Ok(())
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
                Ok(())
            },
            None => {
                warn!("GPU not available. Falling back to CPU-based layout calculation.");
                Self::calculate_layout_cpu(graph, params.iterations, params.spring_strength, params.damping);
                Ok(())
            }
        }
    }

    fn calculate_layout_cpu(graph: &mut GraphData, iterations: u32, spring_strength: f32, damping: f32) {
        let repulsion_strength = spring_strength * 10000.0;
        
        for _ in 0..iterations {
            // Calculate forces between nodes
            for i in 0..graph.nodes.len() {
                for j in i+1..graph.nodes.len() {
                    let dx = graph.nodes[j].x - graph.nodes[i].x;
                    let dy = graph.nodes[j].y - graph.nodes[i].y;
                    let dz = graph.nodes[j].z - graph.nodes[i].z;
                    
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    if distance > 0.0 {
                        let force = repulsion_strength / (distance * distance);
                        
                        let fx = dx * force / distance;
                        let fy = dy * force / distance;
                        let fz = dz * force / distance;
                        
                        graph.nodes[i].vx -= fx;
                        graph.nodes[i].vy -= fy;
                        graph.nodes[i].vz -= fz;
                        
                        graph.nodes[j].vx += fx;
                        graph.nodes[j].vy += fy;
                        graph.nodes[j].vz += fz;
                    }
                }
            }
            
            // Apply velocities and damping
            for node in &mut graph.nodes {
                node.x += node.vx;
                node.y += node.vy;
                node.z += node.vz;
                
                node.vx *= damping;
                node.vy *= damping;
                node.vz *= damping;
            }
        }
    }

    pub fn get_service(data: web::Data<crate::AppState>) -> Arc<GraphService> {
        data.graph_service.clone()
    }
}
