use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};

/// Represents a position and velocity update for a node (24 bytes)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable, Serialize, Deserialize)]
pub struct NodePositionVelocity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
}

/// Represents a batch of position updates for multiple nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionUpdate {
    /// Maps node indices to their new positions and velocities
    pub updates: HashMap<usize, NodePositionVelocity>,
    /// Optional timestamp for synchronization
    pub timestamp: Option<u64>,
    /// Whether this is an initial layout update
    pub is_initial_layout: bool,
}

impl PositionUpdate {
    pub fn new() -> Self {
        Self {
            updates: HashMap::new(),
            timestamp: Some(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64),
            is_initial_layout: false,
        }
    }

    /// Adds a position and velocity update for a node
    pub fn add_update(&mut self, index: usize, x: f32, y: f32, z: f32, vx: f32, vy: f32, vz: f32) {
        self.updates.insert(index, NodePositionVelocity { x, y, z, vx, vy, vz });
    }

    /// Creates a binary representation for network transfer
    pub fn to_binary(&self) -> Vec<u8> {
        let mut binary = Vec::with_capacity(4 + self.updates.len() * 24);
        
        // Write header (4 bytes)
        let header_value = if self.is_initial_layout { 1.0f32 } else { 0.0f32 };
        binary.extend_from_slice(&header_value.to_le_bytes());

        // Write position updates (24 bytes each)
        let mut updates: Vec<_> = self.updates.iter().collect();
        updates.sort_by_key(|&(k, _)| k); // Sort by index for consistent order
        
        for (_, update) in updates {
            binary.extend_from_slice(&update.x.to_le_bytes());
            binary.extend_from_slice(&update.y.to_le_bytes());
            binary.extend_from_slice(&update.z.to_le_bytes());
            binary.extend_from_slice(&update.vx.to_le_bytes());
            binary.extend_from_slice(&update.vy.to_le_bytes());
            binary.extend_from_slice(&update.vz.to_le_bytes());
        }

        binary
    }

    /// Creates a position update from binary data
    pub fn from_binary(data: &[u8], num_nodes: usize) -> Result<Self, String> {
        if data.len() != 4 + num_nodes * 24 {
            return Err(format!("Invalid data length: expected {}, got {}", 
                4 + num_nodes * 24, data.len()));
        }

        // Read header
        let mut header_bytes = [0u8; 4];
        header_bytes.copy_from_slice(&data[0..4]);
        let header_value = f32::from_le_bytes(header_bytes);
        let is_initial_layout = header_value >= 1.0;

        let mut updates = HashMap::with_capacity(num_nodes);

        // Read position updates
        for i in 0..num_nodes {
            let offset = 4 + i * 24;
            let mut pos = [0u8; 24];
            pos.copy_from_slice(&data[offset..offset + 24]);

            let x = f32::from_le_bytes(pos[0..4].try_into().unwrap());
            let y = f32::from_le_bytes(pos[4..8].try_into().unwrap());
            let z = f32::from_le_bytes(pos[8..12].try_into().unwrap());
            let vx = f32::from_le_bytes(pos[12..16].try_into().unwrap());
            let vy = f32::from_le_bytes(pos[16..20].try_into().unwrap());
            let vz = f32::from_le_bytes(pos[20..24].try_into().unwrap());

            updates.insert(i, NodePositionVelocity { x, y, z, vx, vy, vz });
        }

        Ok(Self {
            updates,
            timestamp: Some(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64),
            is_initial_layout,
        })
    }
}

/// Message types for WebSocket communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphMessage {
    /// Complete graph initialization
    InitialGraph {
        nodes: Vec<String>,  // Node IDs only
        edges: Vec<(String, String, f32)>,  // (source, target, weight)
        metadata: serde_json::Value,
    },
    /// Position updates only
    PositionUpdate(PositionUpdate),
    /// Parameter updates
    ParameterUpdate {
        spring_strength: Option<f32>,
        damping: Option<f32>,
        iterations: Option<u32>,
    },
}
