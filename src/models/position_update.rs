use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Represents a minimal position update for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePosition {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Represents a batch of position updates for multiple nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionUpdate {
    /// Maps node indices to their new positions
    pub positions: HashMap<usize, NodePosition>,
    /// Optional timestamp for synchronization
    pub timestamp: Option<u64>,
}

impl PositionUpdate {
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
            timestamp: Some(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64),
        }
    }

    /// Adds a position update for a node
    pub fn add_position(&mut self, index: usize, x: f32, y: f32, z: f32) {
        self.positions.insert(index, NodePosition { x, y, z });
    }

    /// Creates a position update from changes only
    pub fn from_changes(old_positions: &[(f32, f32, f32)], new_positions: &[(f32, f32, f32)]) -> Self {
        let mut update = Self::new();
        
        for (i, (old, new)) in old_positions.iter().zip(new_positions.iter()).enumerate() {
            if (old.0 - new.0).abs() > 0.001 || 
               (old.1 - new.1).abs() > 0.001 || 
               (old.2 - new.2).abs() > 0.001 {
                update.add_position(i, new.0, new.1, new.2);
            }
        }
        
        update
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
