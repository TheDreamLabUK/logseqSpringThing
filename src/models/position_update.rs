use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};
use serde::{Serialize, Deserialize};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodePositionVelocity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
}

#[derive(Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PositionUpdate {
    /// Maps node indices to their new positions and velocities
    pub positions: HashMap<usize, NodePositionVelocity>,
    /// Whether this is the final update in a sequence
    pub is_final: bool,
}

impl PositionUpdate {
    pub fn new(positions: HashMap<usize, NodePositionVelocity>, is_final: bool) -> Self {
        Self {
            positions,
            is_final,
        }
    }
}
