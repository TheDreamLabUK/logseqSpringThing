use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, Pod, Zeroable, Serialize, Deserialize)]
#[repr(C)]
pub struct NodePositionVelocity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
}

/// Represents a batch of position updates for multiple nodes
#[derive(Debug, Clone)]
pub struct PositionUpdate {
    /// Maps node indices to their new positions and velocities
    pub updates: HashMap<usize, NodePositionVelocity>,
    /// Whether this is an initial layout update
    pub is_initial_layout: bool,
}

impl PositionUpdate {
    pub fn new(is_initial_layout: bool) -> Self {
        Self {
            updates: HashMap::new(),
            is_initial_layout,
        }
    }

    pub fn add_update(&mut self, index: usize, x: f32, y: f32, z: f32, vx: f32, vy: f32, vz: f32) {
        self.updates.insert(index, NodePositionVelocity { x, y, z, vx, vy, vz });
    }

    pub fn from_binary(data: &[u8], num_nodes: usize) -> Result<Self, String> {
        if data.len() != 4 + num_nodes * 24 {
            return Err(format!("Invalid data length: expected {}, got {}", 
                4 + num_nodes * 24, data.len()));
        }

        // Read is_initial_layout flag
        let mut flag_bytes = [0u8; 4];
        flag_bytes.copy_from_slice(&data[0..4]);
        let is_initial_layout = f32::from_le_bytes(flag_bytes) >= 1.0;

        let mut updates = HashMap::with_capacity(num_nodes);

        // Read position updates
        for i in 0..num_nodes {
            let offset = 4 + i * 24;
            
            let mut x_bytes = [0u8; 4];
            let mut y_bytes = [0u8; 4];
            let mut z_bytes = [0u8; 4];
            let mut vx_bytes = [0u8; 4];
            let mut vy_bytes = [0u8; 4];
            let mut vz_bytes = [0u8; 4];

            x_bytes.copy_from_slice(&data[offset..offset + 4]);
            y_bytes.copy_from_slice(&data[offset + 4..offset + 8]);
            z_bytes.copy_from_slice(&data[offset + 8..offset + 12]);
            vx_bytes.copy_from_slice(&data[offset + 12..offset + 16]);
            vy_bytes.copy_from_slice(&data[offset + 16..offset + 20]);
            vz_bytes.copy_from_slice(&data[offset + 20..offset + 24]);

            let x = f32::from_le_bytes(x_bytes);
            let y = f32::from_le_bytes(y_bytes);
            let z = f32::from_le_bytes(z_bytes);
            let vx = f32::from_le_bytes(vx_bytes);
            let vy = f32::from_le_bytes(vy_bytes);
            let vz = f32::from_le_bytes(vz_bytes);

            updates.insert(i, NodePositionVelocity { x, y, z, vx, vy, vz });
        }

        Ok(Self {
            updates,
            is_initial_layout,
        })
    }

    pub fn to_binary(&self) -> Vec<u8> {
        let mut binary = Vec::with_capacity(4 + self.updates.len() * 24);

        // Write is_initial_layout flag
        binary.extend_from_slice(&(if self.is_initial_layout { 1.0f32 } else { 0.0f32 }).to_le_bytes());

        // Sort updates by index to ensure consistent order
        let mut updates: Vec<_> = self.updates.iter().collect();
        updates.sort_by_key(|&(k, _)| k);

        // Write position updates
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
}
