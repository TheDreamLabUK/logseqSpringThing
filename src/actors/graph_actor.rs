//! Graph Service Actor to replace Arc<RwLock<GraphService>>

use actix::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tokio::time::{interval, Duration};
use log::{debug, info, warn, error};
use actix::fut::WrapFuture;

use crate::actors::messages::*;
use crate::actors::client_manager_actor::ClientManagerActor;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::models::metadata::MetadataStore;
use crate::services::graph_service::GraphData;
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::utils::binary_protocol;
use crate::actors::gpu_compute_actor::GPUComputeActor;

pub struct GraphServiceActor {
    graph_data: GraphData,
    node_map: HashMap<u32, Node>,
    gpu_compute_addr: Option<Addr<GPUComputeActor>>,
    client_manager: Addr<ClientManagerActor>,
    simulation_running: AtomicBool,
    shutdown_complete: Arc<AtomicBool>,
    next_node_id: AtomicU32,
}

impl GraphServiceActor {
    pub fn new(
        client_manager: Addr<ClientManagerActor>,
        gpu_compute_addr: Option<Addr<GPUComputeActor>>,
    ) -> Self {
        Self {
            graph_data: GraphData {
                nodes: Vec::new(),
                edges: Vec::new(),
            },
            node_map: HashMap::new(),
            gpu_compute_addr,
            client_manager,
            simulation_running: AtomicBool::new(false),
            shutdown_complete: Arc::new(AtomicBool::new(false)),
            next_node_id: AtomicU32::new(1),
        }
    }

    pub fn get_graph_data(&self) -> &GraphData {
        &self.graph_data
    }

    pub fn get_node_map(&self) -> &HashMap<u32, Node> {
        &self.node_map
    }

    pub fn add_node(&mut self, node: Node) {
        // Update node_map
        self.node_map.insert(node.id, node.clone());
        
        // Add to graph data if not already present
        if !self.graph_data.nodes.iter().any(|n| n.id == node.id) {
            self.graph_data.nodes.push(node);
        } else {
            // Update existing node
            if let Some(existing) = self.graph_data.nodes.iter_mut().find(|n| n.id == node.id) {
                *existing = node;
            }
        }
        
        debug!("Added/updated node: {}", node.id);
    }

    pub fn remove_node(&mut self, node_id: u32) {
        // Remove from node_map
        self.node_map.remove(&node_id);
        
        // Remove from graph data
        self.graph_data.nodes.retain(|n| n.id != node_id);
        
        // Remove related edges
        self.graph_data.edges.retain(|e| e.source != node_id && e.target != node_id);
        
        debug!("Removed node: {}", node_id);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        // Add to graph data if not already present
        if !self.graph_data.edges.iter().any(|e| e.id == edge.id) {
            self.graph_data.edges.push(edge);
        } else {
            // Update existing edge
            if let Some(existing) = self.graph_data.edges.iter_mut().find(|e| e.id == edge.id) {
                *existing = edge;
            }
        }
        
        debug!("Added/updated edge: {}", edge.id);
    }

    pub fn remove_edge(&mut self, edge_id: &str) {
        self.graph_data.edges.retain(|e| e.id != edge_id);
        debug!("Removed edge: {}", edge_id);
    }

    pub fn build_from_metadata(&mut self, metadata: MetadataStore) -> Result<(), String> {
        // Clear existing data
        self.graph_data.nodes.clear();
        self.graph_data.edges.clear();
        self.node_map.clear();

        // Build nodes from metadata
        for (filename, file_metadata) in metadata.files {
            let node_id = self.next_node_id.fetch_add(1, Ordering::SeqCst);
            
            let node = Node {
                id: node_id,
                metadata_id: filename.clone(),
                label: file_metadata.title.unwrap_or(filename),
                data: BinaryNodeData {
                    position: crate::types::vec3::Vec3Data { x: 0.0, y: 0.0, z: 0.0 },
                    velocity: crate::types::vec3::Vec3Data { x: 0.0, y: 0.0, z: 0.0 },
                    mass: 1.0,
                    flags: 0,
                    padding: [0, 0],
                },
                properties: file_metadata.properties.unwrap_or_default(),
                references: file_metadata.references.unwrap_or_default(),
            };

            self.add_node(node);
        }

        // TODO: Build edges from references
        // This would involve parsing the references in each node and creating edges
        
        info!("Built graph from metadata: {} nodes, {} edges",
              self.graph_data.nodes.len(), self.graph_data.edges.len());
        
        Ok(())
    }

    pub fn update_node_positions(&mut self, positions: Vec<(u32, BinaryNodeData)>) {
        let mut updated_count = 0;
        
        for (node_id, position_data) in positions {
            // Update in node_map
            if let Some(node) = self.node_map.get_mut(&node_id) {
                node.data.position = position_data.position;
                node.data.velocity = position_data.velocity;
                updated_count += 1;
            }
            
            // Update in graph_data.nodes
            if let Some(node) = self.graph_data.nodes.iter_mut().find(|n| n.id == node_id) {
                node.data.position = position_data.position;
                node.data.velocity = position_data.velocity;
            }
        }
        
        debug!("Updated positions for {} nodes", updated_count);
    }

    fn start_simulation_loop(&mut self, ctx: &mut Context<Self>) {
        if self.simulation_running.load(Ordering::SeqCst) {
            warn!("Simulation already running");
            return;
        }

        self.simulation_running.store(true, Ordering::SeqCst);
        info!("Starting physics simulation loop");

        // Start the simulation interval
        ctx.run_interval(Duration::from_millis(16), |actor, _ctx| {
            if !actor.simulation_running.load(Ordering::SeqCst) {
                return;
            }

            actor.run_simulation_step();
        });
    }

    fn run_simulation_step(&mut self) {
        // Run physics calculation (GPU or CPU fallback)
        match self.calculate_layout() {
            Ok(updated_positions) => {
                if !updated_positions.is_empty() {
                    // Update positions
                    self.update_node_positions(updated_positions.clone());
                    
                    // Broadcast to clients
                    if let Ok(binary_data) = self.encode_node_positions(&updated_positions) {
                        self.client_manager.do_send(BroadcastNodePositions { 
                            positions: binary_data 
                        });
                    }
                }
            }
            Err(e) => {
                error!("Physics simulation step failed: {}", e);
            }
        }
    }

    fn calculate_layout(&self) -> Result<Vec<(u32, BinaryNodeData)>, String> {
        // For now, always use CPU fallback since GPU actor communication is async
        // TODO: Refactor simulation loop to handle async GPU computation properly
        self.calculate_layout_cpu()
    }

    fn initiate_gpu_computation(&self, ctx: &mut Context<Self>) {
        // Send GPU computation request if GPU compute actor is available
        if let Some(ref gpu_compute_addr) = self.gpu_compute_addr {
            // Update graph data in GPU
            let graph_data = crate::models::graph::GraphData {
                nodes: self.graph_data.nodes.clone(),
                edges: self.graph_data.edges.clone(),
            };
            
            gpu_compute_addr.do_send(UpdateGPUGraphData { graph: graph_data });
            
            // Request computation
            let addr = gpu_compute_addr.clone();
            let future = async move {
                match addr.send(ComputeForces).await {
                    Ok(Ok(())) => {
                        // Now get the results
                        match addr.send(GetNodeData).await {
                            Ok(Ok(node_data)) => {
                                // Convert to position updates
                                let mut positions = Vec::new();
                                for (index, data) in node_data.iter().enumerate() {
                                    // We need to map index back to node ID
                                    // For now, assume index matches node order in graph
                                    if let Some(node) = self.graph_data.nodes.get(index) {
                                        positions.push((node.id, data.clone()));
                                    }
                                }
                                positions
                            },
                            _ => Vec::new()
                        }
                    },
                    _ => Vec::new()
                }
            };
            
            // Convert future to ActorFuture and spawn it
            ctx.wait(future.into_actor(self).map(|positions, actor, _ctx| {
                if !positions.is_empty() {
                    actor.update_node_positions(positions.clone());
                    
                    // Broadcast to clients
                    if let Ok(binary_data) = actor.encode_node_positions(&positions) {
                        actor.client_manager.do_send(BroadcastNodePositions {
                            positions: binary_data
                        });
                    }
                }
            }));
        }
    }

    fn calculate_layout_cpu(&self) -> Result<Vec<(u32, BinaryNodeData)>, String> {
        // Simple CPU physics simulation
        let mut updated_positions = Vec::new();
        
        for node in &self.graph_data.nodes {
            // Simple physics: apply some random movement for demo
            let mut new_data = node.data.clone();
            new_data.position.x += (rand::random::<f32>() - 0.5) * 0.1;
            new_data.position.y += (rand::random::<f32>() - 0.5) * 0.1;
            new_data.position.z += (rand::random::<f32>() - 0.5) * 0.1;
            
            updated_positions.push((node.id, new_data));
        }
        
        Ok(updated_positions)
    }

    fn encode_node_positions(&self, positions: &[(u32, BinaryNodeData)]) -> Result<Vec<u8>, String> {
        // Now binary_protocol expects (u32, BinaryNodeData) directly
        binary_protocol::encode_node_data(positions)
            .map_err(|e| format!("Failed to encode node data: {}", e))
    }
}

impl Actor for GraphServiceActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("GraphServiceActor started");
        self.start_simulation_loop(ctx);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        self.simulation_running.store(false, Ordering::SeqCst);
        self.shutdown_complete.store(true, Ordering::SeqCst);
        info!("GraphServiceActor stopped");
    }
}

// Message handlers
impl Handler<GetGraphData> for GraphServiceActor {
    type Result = Result<GraphData, String>;

    fn handle(&mut self, _msg: GetGraphData, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.graph_data.clone())
    }
}

impl Handler<UpdateNodePositions> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateNodePositions, _ctx: &mut Self::Context) -> Self::Result {
        self.update_node_positions(msg.positions);
        Ok(())
    }
}

impl Handler<AddNode> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: AddNode, _ctx: &mut Self::Context) -> Self::Result {
        self.add_node(msg.node);
        Ok(())
    }
}

impl Handler<RemoveNode> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: RemoveNode, _ctx: &mut Self::Context) -> Self::Result {
        self.remove_node(msg.node_id);
        Ok(())
    }
}

impl Handler<AddEdge> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: AddEdge, _ctx: &mut Self::Context) -> Self::Result {
        self.add_edge(msg.edge);
        Ok(())
    }
}

impl Handler<RemoveEdge> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: RemoveEdge, _ctx: &mut Self::Context) -> Self::Result {
        self.remove_edge(&msg.edge_id);
        Ok(())
    }
}

impl Handler<GetNodeMap> for GraphServiceActor {
    type Result = Result<HashMap<u32, Node>, String>;

    fn handle(&mut self, _msg: GetNodeMap, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.node_map.clone())
    }
}

impl Handler<BuildGraphFromMetadata> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BuildGraphFromMetadata, _ctx: &mut Self::Context) -> Self::Result {
        self.build_from_metadata(msg.metadata)
    }
}

impl Handler<StartSimulation> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: StartSimulation, ctx: &mut Self::Context) -> Self::Result {
        self.start_simulation_loop(ctx);
        Ok(())
    }
}

impl Handler<StopSimulation> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: StopSimulation, _ctx: &mut Self::Context) -> Self::Result {
        self.simulation_running.store(false, Ordering::SeqCst);
        Ok(())
    }
}

impl Handler<UpdateNodePosition> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateNodePosition, _ctx: &mut Self::Context) -> Self::Result {
        // Update node in the node map
        if let Some(node) = self.node_map.get_mut(&msg.node_id) {
            // Preserve existing mass and flags
            let original_mass = node.data.mass;
            let original_flags = node.data.flags;
            
            node.data.position = msg.position;
            node.data.velocity = msg.velocity;
            
            // Restore mass and flags
            node.data.mass = original_mass;
            node.data.flags = original_flags;
        } else {
            debug!("Received update for unknown node ID: {}", msg.node_id);
            return Err(format!("Unknown node ID: {}", msg.node_id));
        }
        
        // Update corresponding node in graph
        for node in &mut self.graph_data.nodes {
            if node.id == msg.node_id {
                // Preserve mass and flags
                let original_mass = node.data.mass;
                let original_flags = node.data.flags;
                
                node.data.position = msg.position;
                node.data.velocity = msg.velocity;
                
                // Restore mass and flags
                node.data.mass = original_mass;
                node.data.flags = original_flags;
                break;
            }
        }
        
        Ok(())
    }
}

impl Handler<SimulationStep> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: SimulationStep, _ctx: &mut Self::Context) -> Self::Result {
        // Just run one simulation step
        self.run_simulation_step();
        Ok(())
    }
}

impl Handler<UpdateGraphData> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateGraphData, _ctx: &mut Self::Context) -> Self::Result {
        info!("Updating graph data with {} nodes, {} edges",
              msg.graph_data.nodes.len(), msg.graph_data.edges.len());
        
        // Update graph data
        self.graph_data = msg.graph_data;
        
        // Rebuild node map
        self.node_map.clear();
        for node in &self.graph_data.nodes {
            self.node_map.insert(node.id, node.clone());
        }
        
        info!("Graph data updated successfully");
        Ok(())
    }
}