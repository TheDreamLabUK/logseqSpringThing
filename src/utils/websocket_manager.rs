use actix::prelude::*;
use bytemuck;
use log::{debug, error};
use std::sync::Arc;

use crate::models::node::GPUNode;

// Constants for binary data
const FLOAT32_SIZE: usize = std::mem::size_of::<f32>();
const HEADER_SIZE: usize = FLOAT32_SIZE; // isInitialLayout flag
const NODE_SIZE: usize = 6 * FLOAT32_SIZE; // x, y, z, vx, vy, vz

pub struct WebSocketManager {
    // Pre-allocate buffer for binary data to avoid repeated allocations
    binary_buffer: Vec<u8>,
}

impl WebSocketManager {
    pub fn new() -> Self {
        Self {
            binary_buffer: Vec::with_capacity(1024 * 1024), // 1MB initial capacity
        }
    }

    pub async fn broadcast_binary(&self, nodes: &[GPUNode], is_initial: bool) {
        // Calculate required buffer size
        let total_size = HEADER_SIZE + nodes.len() * NODE_SIZE;
        
        // Resize buffer if needed
        if self.binary_buffer.capacity() < total_size {
            self.binary_buffer.reserve(total_size - self.binary_buffer.capacity());
        }
        
        // Clear buffer but keep capacity
        self.binary_buffer.clear();
        
        // Write isInitialLayout flag
        let initial_flag: f32 = if is_initial { 1.0 } else { 0.0 };
        self.binary_buffer.extend_from_slice(bytemuck::bytes_of(&initial_flag));

        // Write node positions and velocities directly
        for node in nodes {
            // Pack position and velocity data tightly
            let node_data: [f32; 6] = [
                node.x, node.y, node.z,
                node.vx, node.vy, node.vz
            ];
            self.binary_buffer.extend_from_slice(bytemuck::cast_slice(&node_data));
        }

        debug!(
            "Broadcasting binary update: {} nodes, {} bytes",
            nodes.len(),
            self.binary_buffer.len()
        );

        // Send binary data to all connected clients
        if let Err(e) = self.broadcast_raw(&self.binary_buffer).await {
            error!("Failed to broadcast binary update: {}", e);
        }
    }

    async fn broadcast_raw(&self, data: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
        // Implementation of actual broadcasting to clients
        // This would depend on your specific WebSocket implementation
        Ok(())
    }
}

impl Actor for WebSocketManager {
    type Context = Context<Self>;
}

// Message definitions
#[derive(Message)]
#[rtype(result = "()")]
pub struct BroadcastBinary {
    pub nodes: Arc<Vec<GPUNode>>,
    pub is_initial: bool,
}

impl Handler<BroadcastBinary> for WebSocketManager {
    type Result = ResponseFuture<()>;

    fn handle(&mut self, msg: BroadcastBinary, _: &mut Context<Self>) -> Self::Result {
        Box::pin(async move {
            self.broadcast_binary(&msg.nodes, msg.is_initial).await;
        })
    }
}
